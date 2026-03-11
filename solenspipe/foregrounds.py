import numpy as np
from pathlib import Path
import yaml
from solenspipe import utility as simgen
from pixell import curvedsky as cs
import healpy as hp

"""
Utilities for saving/loading foreground cross-spectra and
assembling covariance cubes.

Conventions
-----------
- Each 1-D spectrum is indexed by multipole ell starting at 0 with step 1.
- Entries with ell < 2 are forcibly set to zero (on save and on assembly).
- Covariance arrays have shape (ncomp, ncomp, nell), i.e. (i, j, ell).
"""


# small helpers
def _normalize_pair(p):
    a, b = str(p[0]), str(p[1])
    return (a, b) if a <= b else (b, a)

def _pair_to_key(qid1, qid2, sep="|"):
    qlo, qhi = _normalize_pair((qid1, qid2))
    if sep in qlo or sep in qhi:
        raise ValueError(f"qids must not contain the separator {sep!r}: got {qlo!r}, {qhi!r}")
    return f"{qlo}{sep}{qhi}"


def _key_to_pair(key, sep="|"):
    parts = key.split(sep)
    if len(parts) != 2:
        raise ValueError(f"Invalid stored pair key {key!r}")
    return _normalize_pair(parts)


def _check_all_same_length(arrs):
    n = None
    for a in arrs:
        a = np.asarray(a)
        if a.ndim != 1:
            raise ValueError("All cross-spectra must be 1-D arrays")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError("All cross-spectra must have the same length")
    if n is None:
        raise ValueError("No cross-spectra provided")
    return int(n)


def _zero_low_ells(arr):
    arr = np.array(arr, copy=True)
    if arr.shape[0] >= 1:
        arr[0] = 0.0
    if arr.shape[0] >= 2:
        arr[1] = 0.0
    return arr


# public API
def save_cross_spectra(path, cross_spectra, sep="|", extra_meta=None):
    """
    Save cross-spectra to a compressed .npz file (always overwrites).

    Assumptions:
    - Spectra are 1-D arrays on integer ell starting at 0.
    - ell<2 entries are set to zero before writing.

    Parameters
    ----------
    path : str or Path
        Output file ('.npz' recommended).
    cross_spectra : mapping
        Dict mapping (qid1, qid2) -> 1-D array. All arrays must have equal length.
    sep : str, optional
        Pair-key separator for internal storage.
    extra_meta : mapping, optional
        Small metadata (strings/numbers) to include.

    Returns
    -------
    Path
        The path written.

    Examples
    --------
    >>> import numpy as np
    >>> d = {("A","A"): np.arange(5)+1, ("A","B"): 2*(np.arange(5)+1), ("B","B"): 3*(np.arange(5)+1)}
    >>> _ = save_cross_spectra("fg_simple.npz", d)
    """
    path = Path(path)

    canon = {}
    for (q1, q2), arr in cross_spectra.items():
        k = _pair_to_key(q1, q2, sep=sep)
        canon[k] = _zero_low_ells(np.asarray(arr))

    _check_all_same_length(canon.values())

    qids = set()
    for k in canon.keys():
        a, b = _key_to_pair(k, sep=sep)
        qids.add(a); qids.add(b)
    qids = np.array(sorted(qids), dtype=object)

    save_kwargs = {}
    for k, arr in canon.items():
        save_kwargs[f"spec:{k}"] = arr

    save_kwargs["meta:qids"] = qids
    save_kwargs["meta:sep"] = np.array([sep], dtype=object)

    if extra_meta:
        for mk, mv in extra_meta.items():
            save_kwargs[f"meta:extra:{mk}"] = np.array([mv], dtype=object)

    np.savez_compressed(path, **save_kwargs)
    return path

def fg_covariance_cube(path, qids, require_all=True, fill_missing=np.nan, symmetrize=True, qid_aliases=None):
    """
    Build a covariance array of shape (ncomp, ncomp, nell) for the given qids,
    reading **only the spectra needed** directly from the NPZ on disk.

    The element C[i, j, ell] equals the cross-spectrum between qids[i]
    and qids[j] at multipole 'ell'. Missing pairs are either filled with
    'fill_missing' or raise an error depending on 'require_all'.

    Aliases
    -------
    You can pass a mapping ``qid_aliases`` where keys are requested qids and
    values are the underlying qids to use on disk. This is useful when some
    requested components are exact aliases/copies of others. For example:
    
    >>> # If NPZ contains only 'a' but you want 'x' to behave like 'a':
    >>> # covariance_cube(path, ['a','x','b'], qid_aliases={'x':'a'})
    >>> # All spectra involving 'x' are taken from those with 'a'.

    Notes
    -----
    - ell<2 entries in the output are set to zero.
    - Input spectra are assumed to live on ell = 0, 1, 2, ... with unit spacing.
    - Only spectra required by the (qid_i, qid_j) pairs (after aliasing) are
      read from the file.

    Parameters
    ----------
    path : str or Path
        Path to an NPZ written by :func:`save_cross_spectra`.
    qids : sequence of str
        Order of components along the first two axes.
    require_all : bool, default True
        If True, raise on missing pairs; else fill with 'fill_missing'.
    fill_missing : float, default np.nan
        Used only when require_all=False.
    symmetrize : bool, default True
        If True, symmetrize exactly over (i, j) by averaging with its transpose.
    qid_aliases : mapping, optional
        Dict mapping requested qids to substitute qids to use from disk.

    Returns
    -------
    numpy.ndarray
        Covariance array with shape (ncomp, ncomp, nell).

    Examples
    --------
    >>> import numpy as np
    >>> d = {("a","a"): np.ones(5), ("a","b"): 2*np.ones(5), ("b","b"): 3*np.ones(5)}
    >>> _ = save_cross_spectra("fg_cov_alias.npz", d)
    >>> # Request ['a','x','b'] with x->a alias; file has no 'x' spectra
    >>> C = covariance_cube("fg_cov_alias.npz", ["a","x","b"], qid_aliases={"x":"a"})
    >>> C.shape
    (3, 3, 5)
    >>> np.all(C[..., :2] == 0)
    True
    """

    if len(qids)==0: raise ValueError("No qids provided")
    qids = list(map(str, qids))
    ncomp = len(qids)
    alias = dict(qid_aliases) if qid_aliases is not None else {}
    resolved = [alias.get(q, q) for q in qids]

    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        try:
            sep = str(z["meta:sep"][0])
        except KeyError:
            sep = "|"

        # Build set of unique needed (resolved) pairs and their NPZ keys
        needed_pairs = set()
        for qi in resolved:
            for qj in resolved:
                # use normalized resolved pair
                a, b = _normalize_pair((qi, qj))
                needed_pairs.add((a, b))

        key_for_pair = {pair: f"spec:{_pair_to_key(pair[0], pair[1], sep=sep)}" for pair in needed_pairs}

        # Determine nell by loading the first present needed spectrum
        nell = None
        loaded = {}
        for pair, key in key_for_pair.items():
            if key in z.files:
                arr = _zero_low_ells(np.asarray(z[key]))
                nell = arr.shape[0]
                loaded[pair] = arr
                break
        
        if nell is None:
            if require_all:
                raise KeyError("None of the requested pair spectra (after aliasing) were found in the file.")
            any_spec_keys = [k for k in z.files if k.startswith("spec:")]
            if not any_spec_keys:
                raise ValueError("The NPZ file contains no spectra.")
            arr = _zero_low_ells(np.asarray(z[any_spec_keys[0]]))
            nell = arr.shape[0]

        # Allocate covariance and fill
        C = np.empty((ncomp, ncomp, nell), dtype=float)
        C.fill(fill_missing)

        # Fill using resolved pairs
        for i, qi in enumerate(resolved):
            for j, qj in enumerate(resolved):
                pair = _normalize_pair((qi, qj))
                key = key_for_pair[pair]
                if pair in loaded:
                    arr = loaded[pair]
                elif key in z.files:
                    arr = _zero_low_ells(np.asarray(z[key]))
                    if arr.shape[0] != nell:
                        raise ValueError(f"Spectrum length mismatch for pair {pair}: got {arr.shape[0]}, expected {nell}")
                    loaded[pair] = arr
                else:
                    if require_all and i <= j:
                        raise KeyError(f"Missing spectrum for pair {pair} (resolved from aliases)")
                    continue
                C[i, j, :] = arr

        if symmetrize:
            C = 0.5 * (C + np.swapaxes(C, 0, 1))

        if nell > 0:
            C[:, :, 0] = 0.0
        if nell > 1:
            C[:, :, 1] = 0.0

        return C

class ForegroundHandler:
    '''
    A class to handle the generation and management of foregrounds for simulations
    
    ### Initialization parameters:
    - datamodel: DataModel object, used to read sofind products
    - args: argparse.Namespace(), must contain the following attributes:
        * args.fg_type: str, type of foregrounds, 'sims' or 'theory'
        * args.fgs_path: str, path to foreground sims
        * args.is_noiseless: bool, if True, no foregrounds are generated
        * args.lmax_signal: int, maximum ell of signal sims
    
    ### Methods:
    - generate_cov_fgs(fgs_path, lmax): 
        Generates the foreground covariance matrix (power spectrum) for the given lmax 
        at two frequencies (90 and 150 GHz) and their cross-spectrum.
    - get_map_fgs(qid, alms_f):
        Returns the foreground map corresponding to the given qid and alms_f. qid informs whether to use f150 or f090 from alms_f.
    - get_fg_alms(fgcov, qid, cmb_set, sim_indices):
        Generates alm from cl for the given qid, cmb_set, and sim_indices (the latter 2 are used in the seed).
    '''

    def __init__(self, datamodel, args, debug=False):
        
        '''
        datamodel: DataModel, sofind datamodel
        args.fg_type: str, type of foregrounds, 'sims' or 'theory'
        args.fgs_path: str, path to foreground sims
        args.is_noiseless: bool, if True, no foregrounds are generated
        args.lmax_signal: int, maximum ell of signal sims
        args.maps_subproduct: str, subproduct name for maps
        args.qids: str, qids delimited by spaces, e.g., "pa5a pa5b pa6a pa6b"
        debug: bool, print foreground debug messages
        '''

        self.datamodel = datamodel
        self.args = args
        # 'sims' loads from foreground 2pt file (measured in sims), 'theory' estimates analytically
        # 'sims_actplanck' loads from expanded foreground 2pt covariance estimated from ACT+Planck fits
        assert self.args.fg_type in ['sims', 'theory', 'sims_actplanck'] 
        self.debug = debug
        self.fgcov_func = self._define_fgcov_func()
        # defined from compute_fg_alms(), but set to None by default 
        self.alms_f = None

    def split_qids(self):
        # may be more generalized?
        if isinstance(self.args.qids, str):
            return self.args.qids.split(" ")
        else:
            return self.args.qids

    def _define_fgcov_func(self):
        ''' load foreground covariance matrix (power spectra)'''
        # lmax conditioned by max ell of signal sims (van Engelen)
        if self.args.fg_type == 'sims':
            return lambda: self.generate_cov_fgs(self.args.fgs_path,
                                                 self.args.lmax_signal)
        elif self.args.fg_type == 'theory':
            # currently unsupported
            raise NotImplementedError
        elif self.args.fg_type == 'sims_actplanck':
            assert self.args.fgs_path.endswith(".npz"), \
                   "Unsupported format for fgs file (should be <filename>.npz)"
            qids_split = self.split_qids()
            if self.debug: print("qids_split: ", qids_split)
            qid_aliases = { qid: qid[:-3] for qid in qids_split
                                          if '_dw' in qid or '_dd' in qid }
            if self.debug: print("qid_aliases: ", qid_aliases)
            return lambda: fg_covariance_cube(self.args.fgs_path, qids_split,
                                              qid_aliases=qid_aliases)
        return None

    def generate_cov_fgs(self, fgs_path, lmax):
        
        '''
        returns the foreground covariance matrix (power spectrum) for the given lmax 
        at two frequencies (90 and 150 GHz) and their cross-spectrum
        
        fgs_path: str, path to foreground sims
        lmax: int, maximum ell of fg power spectrum
        '''
        
        wfacs_file= fgs_path + "wfacs.yml"
        with open(wfacs_file, "rb") as f:
            wfacs = yaml.load(f, yaml.Loader)

        w_2_foreground = wfacs["w2"]

        # Load foreground alms
        foreground_alms_93 = hp.read_alm(fgs_path + 'fg_nonoise_alms_0093.fits')
        foreground_alms_150 = hp.read_alm(fgs_path + 'fg_nonoise_alms_0145.fits')

        # Generate foreground Cls and smooth them
        cls_90 = simgen.smooth_rolling_cls(cs.alm2cl(foreground_alms_93)/ w_2_foreground, N=10) 
        cls_150 = simgen.smooth_rolling_cls(cs.alm2cl(foreground_alms_150)/ w_2_foreground, N=10) 
        cls_150x90 = simgen.smooth_rolling_cls(cs.alm2cl(foreground_alms_93, foreground_alms_150) / w_2_foreground, N=10)

        # Initialize the covariance matrix of the map (a.k.a. power spectrum)
        cov_matrix = np.zeros((2, 2, lmax + 1))

        # Assign values to the covariance matrix
        # 0 == f150, 1 == f090
        cov_matrix[0, 0] = cls_150[:lmax + 1]
        cov_matrix[0, 1] = cls_150x90[:lmax + 1]
        cov_matrix[1, 0] = cov_matrix[0, 1]
        cov_matrix[1, 1] = cls_90[:lmax + 1]

        return cov_matrix

    def get_map_fgs(self, qid, alms_f):
        '''
        qid of the map (frequency it corresponds to) is mapped to component of alms_f
        if type is "sims_actplanck", generalize beyond [f150, f090] and use the order
        provided in the qids parameter
        '''
        if self.args.fg_type == "sims":
            qid_freq_dict = {'f150': 0, 'f090': 1}
            qid_dict = self.datamodel.get_qid_kwargs_by_subproduct(product='maps', qid=qid,
                                                                   subproduct=self.args.maps_subproduct)
            index_fg = qid_freq_dict[qid_dict['freq']]
            return alms_f[index_fg]
        else:
            qids_split = self.split_qids()
            return alms_f[qids_split.index(qid)]

    def compute_fg_alms(self, fgcov, cmb_set=None, sim_indices=None):
        '''
        generate alm from cl and store in object
        
        fgcov: np.ndarray, foreground covariance matrix (2pt)
        qid: str, qid of the map
        cmb_set: int, set of cmb sim (for seed, optional)
        sim_indices: int, index of the sim (for seed, optional)
        '''
        # no seed info provided or theory fg
        if None in [cmb_set, sim_indices] or self.args.fg_type == 'theory':
            self.alms_f = cs.rand_alm(fgcov, lmax=self.args.lmax_signal)
        else:
            assert 'sims' in self.args.fg_type, \
                "need fgcov from sims to generate fg alms with a seed"
            self.alms_f = cs.rand_alm(fgcov, seed=(0, cmb_set, sim_indices),
                                      lmax=self.args.lmax_signal)

        return None

    def get_fg_alms(self, fgcov, qid, cmb_set=None,
                    sim_indices=None, rerun=False):
        '''
        generate alm from cl (if not generated) and return
        
        fgcov: np.ndarray, foreground covariance matrix (2pt)
        qid: str, qid of the map
        cmb_set: int, set of cmb sim (for seed, optional)
        sim_indices: int, index of the sim (for seed, optional)
        '''
        if self.alms_f is None or rerun:
            self.compute_fg_alms(fgcov, cmb_set, sim_indices)

        if self.debug: print("fg alms shape: ", self.alms_f.shape)
        return self.get_map_fgs(qid, self.alms_f)