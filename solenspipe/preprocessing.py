from orphics import maps
from pixell import enmap, utils as u, curvedsky as cs, reproject
import numpy as np
from mnms import noise_models as nm
from sofind import DataModel
from pixell import bunch
from solenspipe import utility as simgen
import os
import healpy as hp
import re

specs_weights = {'EpureB': ['T','E','pureB'],
        'EB': ['T','E','B']}
nspecs = len(specs_weights['EB'])

def is_planck(qid):
    return (parse_qid_experiment(qid)=='planck')

def get_text(qid,args):
    if qid[:2]=='p0':
        exp = 'Planck'
    else:
        exp = 'ACT'
    freq = args.freq
    return f'{exp} {qid} {freq}'

def parse_qid_experiment(qid):
    if qid in ['p01','p02','p03','p04','p05','p06','p07','p08','p09']:
        return 'planck'
    else:
        return 'act'

def get_kspace_mask(args):
    
    if (args.khfilter is None) and (args.kvfilter is None):
        return None
    else:
        return np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)

def get_inpaint_mask(args, datamodel):
    
    '''
    Generates a mask where to inpaint the map.
    
    ### Parameters:
    - datamodel: DataModel object, used to read sofind products
    - args: argparse.Namespace(), must contain the following attributes: 
        args.inpaint: bool, if True, you want to inpaint
        args.inpaint_subproduct: str, subproduct name for inpainting catalog
        args.cat_date: str, date of inpaint catalog, e.g. '20241002'
        args.regular_hole: float, radius of hole [arcmin] for regular sources
        args.large_hole: float, radius of hole [arcmin] for large sources
        args.shape: tuple, shape of mask
        args.wcs: wcs object, wcs of mask
    '''
    
    if args.inpaint:
        print('inpainting')
        assert args.cat_date is not None, "cat_date must be provided for inpaint"

        # read catalog coordinates
        rdecs, rras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_regular_{args.cat_date}.csv', subproduct = args.inpaint_subproduct))
        ldecs, lras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_large_{args.cat_date}.csv', subproduct = args.inpaint_subproduct))

        # Make masks for gapfill
        mask1 = maps.mask_srcs(args.shape,args.wcs,np.asarray((ldecs,lras)),args.large_hole)
        mask2 = maps.mask_srcs(args.shape,args.wcs,np.asarray((rdecs,rras)),args.regular_hole)
        jmask = mask1 & mask2
        if jmask.dtype!=np.bool_: raise ValueError
        jmask = ~jmask
       
        return jmask
    
    else:
        print('not inpainting')
        return None

def get_metadata(qid, splitnum=0, coadd=False, args=None):
    """
    Retrieves metadata for a specific qid (split/coadd).
    
    ### Paramters:
    - qid: str, unique identifier for the data (must be in sofind)
    - splitnum: int, split of data to be used, will be ignored if coadd is True
    - coadd: bool, if True, coadd the data (ignore splitnum)
    - args: argparse.Namespace(), must contain the following attributes:
        * args needed for get_inpaint_mask()
        * args.maps_subproduct: str, subproduct name for maps (e.g. 'maps_srcfree', 'maps)
        * args.cal_subproduct: str, subproduct name for calibration
        * args.poleff_subproduct: str, subproduct name for polarization efficiency
        * args needed for BeamHelper()
        * args needed for NoiseMetadata()
        * args.khfilter: int, horizontal fourier strip width [used for ground-pickup removal]
        * args.kvfilter: int, vertical fourier strip width [used for ground-pickup removal]     
    """
    
    meta = bunch.Bunch({})
    #assert 0 <= splitnum < 4, "only supporting splits [0,1,2,3]"
    if parse_qid_experiment(qid)=='planck':
        meta.Name = 'planck_npipe'
        meta.dm = DataModel.from_config(meta.Name)
        meta.splits = np.array([1,2])
        meta.nsplits = 2
        meta.nsims = 600
        # assigning ACT splits 0 + 1 to Planck split 1
        # and ACT splits 2 + 3 to Planck split 2
        # ! isplit = None if coadd else (splitnum // 2 + 1)
        isplit = None if coadd else (splitnum // 2 + 1)
        meta.calibration = meta.dm.read_calibration(qid, subproduct=args.cal_subproduct, which='cals')
        meta.pol_eff = meta.dm.read_calibration(qid, subproduct=args.poleff_subproduct, which='poleffs')
        meta.Beam = PlanckBeamHelper(meta.dm, args, qid, isplit, coadd=coadd)
        meta.beam_fells = meta.Beam.get_effective_beam()[1]
        meta.transfer_fells = meta.Beam.get_effective_beam()[2]
        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = None
        meta.nspecs = nspecs
        meta.cal_cluster = meta.dm.read_calibration(qid, subproduct=args.nemo_calibration, which='cals')
        meta.specs = specs_weights['EpureB'] if args.pureEB else specs_weights['EB']
        meta.maptype = 'reprojected'
        meta.noisemodel = PlanckNoiseMetadata(qid, verbose=True,
                                              config_name=meta.Name,
                                              subproduct_name="noise_sims")

        
    elif parse_qid_experiment(qid)=='act':
        
        meta.Name = 'act_dr6v4'
        meta.dm = DataModel.from_config(meta.Name)
        qid_dict = meta.dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
        
        meta.nsims = 800
        meta.nsplits = qid_dict['num_splits']
        meta.splits = np.arange(meta.nsplits)
        meta.daynight = qid_dict['daynight']
   
        cal_kwargs = getattr(args, "cal_subproduct_kwargs", {})
        meta.calibration = meta.dm.read_calibration(qid, subproduct=args.cal_subproduct, which='cals', **cal_kwargs)
        meta.pol_eff = meta.dm.read_calibration(qid.split('_')[0], subproduct=args.poleff_subproduct, which='poleffs', **cal_kwargs)

        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = get_kspace_mask(args)
        meta.maptype = 'native'
        meta.noisemodel = ACTNoiseMetadata(qid, verbose=True)
        meta.nspecs = nspecs
        meta.specs = specs_weights['EpureB'] if args.pureEB else specs_weights['EB']
        isplit = None if coadd else splitnum
        meta.Beam = ACTBeamHelper(meta.dm, args, qid, isplit, coadd=coadd)
        meta.beam_fells = meta.Beam.get_effective_beam()[1]
        meta.transfer_fells = meta.Beam.get_effective_beam()[2]
        
        meta.cal_cluster = meta.dm.read_calibration(qid.split('_')[0], subproduct=args.nemo_calibration, which='cals')
        
        meta.leakage_matrix = None
        meta.leakage_corr = args.leakage_corr
        if meta.leakage_corr:
            meta.leakage_matrix = meta.Beam.get_invleakage_beaminv_matrix()

    return meta, isplit

def process_beam(sofind_beam, norm=True, interp=True):
    '''
    normalized beam if required and then interpolate
    '''
    ell_bells, bells = sofind_beam[0], sofind_beam[1]
    assert ell_bells[0] == 0

    if norm:
        bells /= bells[0]
        
    if interp:
        return maps.interp(ell_bells, bells, fill_value='extrapolate')
    else:
        return ell_bells, bells
      
class ACTBeamHelper:
    
    """
    Helper class to read and process the beam and transfer function for ACT data.
    
    ### Parameters:
    - datamodel: DataModel object, used to read sofind products
    - args: argparse.Namespace(), must contain the following attributes:
        * args.mlmax: int, maximum multipole for the beam. Will use default (4000) if unset.
        * args.beam_subproduct: str, subproduct name for beam (e.g. 'beams_v4_20230902'). Will use default ('dummy_beams') if unset.
        * args.tf_subproduct: str, subproduct name for transfer function (e.g. 'tf_v4_20230902'). Will use default ('dummy_tf') if unset.
    - qid: str, unique identifier for the data (must be in sofind)
    - isplit: int, split of data to be used, will be ignored if coadd is True
    - coadd: bool, if True, coadd the data (ignore splitnum)    
    
    ### Methods:
    
    - get_beam(): returns the beam 
    - get_tf(): returns the transfer function 
    - get_effective_beam(): returns the effective beam (effective beam, beam, transfer function)
    """

    def __init__(self, datamodel, args, qid, isplit=0, coadd=False):
        
        default_values = {'tf_subproduct': 'dummy',
                          'beam_subproduct': 'dummy',
                          'mlmax': 4000}
        for key, value in default_values.items():
            if not hasattr(args, key):
                print(f"Setting default value for '{key}': {value}")
            setattr(args, key, getattr(args, key, value))

        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.qid = qid
        self.isplit = isplit
        self.coadd = coadd
        self.beam_subproduct = args.beam_subproduct
        self.tf_subproduct = args.tf_subproduct
        self.beam_subproduct_kwargs = getattr(args, "beam_subproduct_kwargs", {})

    def get_beam(self, interp=True):
        beam_map = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid,
                                                split_num=self.isplit, coadd=self.coadd,
                                                **self.beam_subproduct_kwargs)
        return process_beam(beam_map, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct), interp=interp)
            
    def get_tf(self):
        ells_tf, tf = self.datamodel.read_tf(subproduct=self.tf_subproduct, qid=self.qid)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')

    def get_effective_beam(self):
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        
        tf = self.get_tf()(np.arange(self.mlmax+1))
        beam = self.get_beam()(np.arange(self.mlmax+1))
        
        fkbeam[0] = beam * tf
        fkbeam[1] = beam
        fkbeam[2] = beam

        return fkbeam, beam, tf
    
    def get_invleakage_beaminv_matrix(self):
        
        '''This matrix gets applied to T,E,B to do both beam deconvolution and leakage correction
        
        / B_s            0    0   \ -1  / alm_T \
        | gamma_E * B_c  B_s  0   |     | alm_E |
        \ gamma_B * B_c  0    B_s /     \ alm_B /
        
        where B_s is the per-split beam
        B_c is the coadd beam
        gamma is defined as leakage correction / beam(T)
        (see https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_harmonic_beams_profiles_info.html)
        '''
        
        te = self.datamodel.read_beam(qid=self.qid, subproduct='beams_leakage', coadd=True, leakage_comp='e')
        ell = te[0]
        
        assert int(ell[0]) == 0, 'ell start 0 important, we are doing multiplications afterwards'

        te = te[1]
        tb = self.datamodel.read_beam(qid=self.qid, subproduct='beams_leakage', coadd=True, leakage_comp='b')
        assert np.all(tb[0] == ell)
        tb = tb[1]
                
        # assert self.coadd is False
        
        sbeam = self.get_beam(interp=False)
        assert np.all(sbeam[0] == ell)
        sbeam = sbeam[1]
        
        inv_tf = 1 / self.get_tf()(ell)
        
        self.coadd = True
        cbeam = self.get_beam(interp=False)
        self.coadd = False
        assert np.all(cbeam[0] == ell)
        cbeam = cbeam[1]
        
        array0 = np.zeros(len(ell))
        
        inv_sbeam = 1/sbeam
        
        inv_matrix = np.array([[ inv_tf * inv_sbeam,                  array0,    array0],
                                [-te * cbeam * inv_sbeam**2 * inv_tf, inv_sbeam, array0],
                                [-tb * cbeam * inv_sbeam**2 * inv_tf, array0,    inv_sbeam]])
        
        return inv_matrix

class PlanckBeamHelper:

    def __init__(self, datamodel, args, qid, isplit=0, coadd=False):
        default_values = {'tf_subproduct': 'dummy',
                          'beam_subproduct': 'dummy',
                          'mlmax': 4000}
        for key, value in default_values.items():
            if not hasattr(args, key):
                print(f"Setting default value for '{key}': {value}")
            setattr(args, key, getattr(args, key, value))

        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.qid = qid
        self.isplit = isplit
        self.coadd = coadd
        self.beam_subproduct = args.beam_subproduct
        self.tf_subproduct = args.tf_subproduct
        self.beam_subproduct_kwargs = getattr(args, "beam_subproduct_kwargs", {})

    def get_beam(self, interp=True, pixwin=True):
        """
        Retrieve the Planck beam for a given QID and split number.

        Parameters:
        - pixwin (bool): Apply pixel window function if True.

        Returns:
        - np.ndarray: The beam function array.
        """
        # overwrite with custom
        # if isplit is None: isplit = self.isplit
        # if qid is None: qid = self.qid

        # Determine split letter (coadd case doesn't matter)
        sl = 'A' if self.isplit == 1 else 'B'

        # Load and interpolate the beam
        ell_b, bl = self.datamodel.read_beam(self.qid,
                        subproduct=self.beam_subproduct,
                        split_num=sl,
                        coadd=self.coadd,
                        **self.beam_subproduct_kwargs)
        
        # process_beam essentially but with pixwin
        beam_f = maps.interp(ell_b, bl, fill_value='extrapolate')

        # Generate the beam function values
        beam_fells = beam_f(np.arange(self.mlmax+1))
        if pixwin:
            beam_fells *= hp.pixwin(2048)[:self.mlmax+1]
        
        # normalize if required
        if self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct):
            beam_fells /= beam_fells[0]

        if interp:
            return maps.interp(np.arange(self.mlmax+1), beam_fells,
                               fill_value='extrapolate')
        else:
            return np.arange(self.mlmax+1), beam_fells
    
    def get_tf(self):
        ells_tf, tf = np.arange(2, 3000, dtype=float), np.ones(2998)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')
    
    def get_effective_beam(self):
        
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        
        tf = self.get_tf()(np.arange(self.mlmax+1))
        beam = self.get_beam()(np.arange(self.mlmax+1))
        
        fkbeam[0] = beam * tf
        fkbeam[1] = beam
        fkbeam[2] = beam

        return fkbeam, beam, tf
    
    # def planck_processed_beam(self, qid, splitnum, pixwin=True):
    #     return self.get_beam(qid=qid, isplit=splitnum, pixwin=pixwin)

class PlanckNoiseMetadata:
    
    def __init__(self, qid, verbose=False,
                 config_name="planck_npipe",
                 subproduct_name="noise_sims"):
        self.qid = qid

        qid_dict_config_noise_name = {'p01': '030',
                                    'p02': '044',
                                    'p03': '070',
                                    'p04': '100',
                                    'p05': '143',
                                    'p06': '217',
                                    'p07': '353',
                                    'p08': '547',
                                    'p09': '857'}
        
        self.planck_config_name = config_name
        self.planck_noise_sims_subproduct = subproduct_name

        if verbose:
            print(f"Initializing NoiseMetadata with qid: {self.qid}")
        self.qid_freq = qid_dict_config_noise_name[qid]

    # moved Frank's residual noise alm function here...
    def noise_map_path(self, isplit, index):
        datamodel = DataModel.from_config(self.planck_config_name)
        maptag = str(index+200).zfill(4)
        assert isplit in [1,2], "Planck splits are either 1 or 2"
        split_num = "A" if isplit == 1 else "B"
        return datamodel.get_map_fn(qid=self.qid, coadd=False,
                                    split_num=split_num,
                                    subproduct="noise_sims",
                                    maptag=maptag)
    
    def read_in_sim(self, isplit, index, lmax=4000):
        # instead of modifying pixell (save_alm option), 
        # have added healpix2map snippet in code
        try:
            residual_map = hp.read_map(self.noise_map_path(isplit, index),
                                       field=(0,1,2))
        except IndexError:
            residual_map = hp.read_map(self.noise_map_path(isplit, index),
                                       field=(0))
            print("No pol found, setting E/B to 0.")
            residual_map = np.array([residual_map,
                                     residual_map*0.,
                                     residual_map*0.])
        return healpix2map(residual_map, lmax=lmax,
                                     rot='gal,equ',save_alm=True)*10**6


def healpix2map(iheal, lmax, rot=None, spin=[0,2], method="harm", niter=0, save_alm=False):

    assert method in ["harm", "harmonic"]
    alm = cs.map2alm_healpix(iheal, lmax=lmax, spin=spin, niter=niter)
    if rot is not None:
        cs.rotate_alm(alm, *reproject.rot2euler(rot), inplace=True)
    if save_alm:
        print("saving alm")
        return alm
   
class ACTNoiseMetadata:
    
    '''
    A class to handle the noise metadata for ACT
    
    ### Initialization paramters:
    - qid: str, unique identifier for the data (must be in sofind)
    - verbose: bool, if True, print additional information during initialization

    ### Methods:
    - read_in_sim(split_num, sim_num, lmax=5400, alm=True): 
        Reads in a simulated noise realization (alm) for the given split and simulation number.
    - get_index_sim_qid(qid): 
        mnms sims from the same array (e.g. 'pa5a_pa5b') are stored in the same file.
        depending on the qid, it will return the corresponding component (e.g. 'pa5a' or 'pa5b').
    '''
    
    def __init__(self, qid, verbose=False):
        self.qid = qid
            
        # noise model qids
        qid_dict_noise = {'pa4a': ['pa4a', 'pa4b'],
                        'pa4b': ['pa4a', 'pa4b'],
                        'pa5a': ['pa5a', 'pa5b'],
                        'pa5b': ['pa5a', 'pa5b'],
                        'pa6a': ['pa6a', 'pa6b'],
                        'pa6b': ['pa6a', 'pa6b'],
                        'pa5a_dd': ['pa5a_dd', 'pa5b_dd'],
                        'pa5b_dd': ['pa5a_dd', 'pa5b_dd'],
                        'pa6a_dd': ['pa6a_dd', 'pa6b_dd'],
                        'pa6b_dd': ['pa6a_dd', 'pa6b_dd'],
                        'pa5a_dw': ['pa5a_dw', 'pa5b_dw'],
                        'pa5b_dw': ['pa5a_dw', 'pa5b_dw']}

        qid_dict_noise_model_name = {'pa4b': 'tile_cmbmask',
                                'pa5a': 'tile_cmbmask',
                            'pa5b': 'tile_cmbmask',
                            'pa6a': 'tile_cmbmask_ivfwhm2',
                            'pa6b': 'tile_cmbmask_ivfwhm2',
                            'pa5a_dd': 'tile_cmbmask_daydeep_250513',
                            'pa5b_dd': 'tile_cmbmask_daydeep_250513',
                            'pa6a_dd': 'tile_cmbmask_daydeep_250513',
                            'pa6b_dd': 'tile_cmbmask_daydeep_250513',
                            'pa5a_dw': 'tile_cmbmask_daywide_250513',
                            'pa5b_dw': 'tile_cmbmask_daywide_250513'}

        qid_dict_config_noise_name = {'pa4b': 'act_dr6v4',
                                    'pa5a': 'act_dr6v4',
                                    'pa5b': 'act_dr6v4',
                                    'pa6a': 'act_dr6v4',
                                    'pa6b': 'act_dr6v4',
                                    'pa5a_dd': 'act_dr6v4_day',
                                    'pa5b_dd': 'act_dr6v4_day',
                                    'pa6a_dd': 'act_dr6v4_day',
                                    'pa6b_dd': 'act_dr6v4_day',
                                    'pa5a_dw': 'act_dr6v4_day',
                                    'pa5b_dw': 'act_dr6v4_day'}
        if verbose:
            print(f"Initializing NoiseMetadata with qid: {self.qid}")

        self.tnm = nm.BaseNoiseModel.from_config(qid_dict_config_noise_name[self.qid],
                                        qid_dict_noise_model_name[self.qid],
                                        *qid_dict_noise[self.qid])
    
    def get_index_sim_qid(self,qid):

        # Define a mapping from order to index
        order_to_index = {'a': 0, 'b': 1}
        # Extract the order from the qid
        order = qid.split('pa')[1][1]
        # Get the index corresponding to the order
        index = order_to_index[order]

        return index

    def read_in_sim(self,split_num, sim_num, lmax=5400,
                    alm=True, generate=False, write=False):
        
        # grab a sim from disk, fail if does not exist on-disk (by default)
        my_sim = self.tnm.get_sim(split_num=split_num, sim_num=sim_num,
                                  lmax=lmax, alm=alm, generate=generate,
                                  write=write)
        index = self.get_index_sim_qid(self.qid)
        my_sim = my_sim[index].squeeze()
        return my_sim

"""
Some subtleties when applying this to different experiments:

A CAR map at X resolution has a pixel window p(X).
A block-downgrade operation to Y resolution will induce an additional
pixel window of p(Y)/p(X).

Therefore, the correct pixel window to deconvolve from the block-downgraded
map is p(Y).

Planck maps have a healpix window function h(nside).  The CAR reprojected
maps do not correct for this and do not contain the usual p(X) pixel window
either since they are reprojected using SHTs. Therefore, if you downgrade
from X to Y, the pixel window that needs to be deconvolved is:
h(nside)p(Y)/p(X).

"""

def depix_map(imap,maptype='native',dfact=None,kspace_mask=None):
    """
    Remove rectangular pixel window effects from the map.
    If the map was produced in native rect pixelization,
    then what factor it was downgraded by (dfact) is not relevant
    and can be ignored, with the deconvolved window corresponding
    to that of the current map.

    If the map was reprojected (e.g. from healpix), then the
    ratio of pixel windows is used.  This function will not
    account for the original healpix pixel window.

    This will perform necessary Fourier operations on input maps.
    This includes:
    1. Correcting for a map-maker pixel window
    2. Correctiong for a downgrade pixel window
    3. k-space masking

    
    (1) and (3) are only done if the map originates from ACT/SO
    (2) is done if the map was downgraded

    """
    if kspace_mask is not None:
        if kspace_mask.dtype != np.bool_: raise ValueError("kspace mask must be boolean.")

    wy1, wx1 = enmap.calc_window(imap.shape)
    if maptype=='native':
        wy2 = wx2 = 1
    elif maptype=='reprojected':
        if (dfact is None) or (dfact==1):
            if kspace_mask is None: return imap
            wy2 = wx2 = wy1 = wx1 = 1
        else:
            wy2, wx2 = enmap.calc_window(imap.shape, scale=dfact)
    else:
        raise ValueError
    fmap = enmap.fft(imap)
    
    fmap *= (wy2/wy1)[:,None]
    fmap *= (wx2/wx1)
    if kspace_mask is not None:
        if imap.ndim==3:
            fmap[:,~kspace_mask] = 0
        else:
            fmap[~kspace_mask] = 0

    imap = enmap.ifft(fmap).real
    return imap

def inpaint_temperature_only(tqu_map, mask):
    omap = tqu_map.copy()
    omap[0] = maps.gapfill_edge_conv_flat(tqu_map[0], mask)
    return omap

def preprocess_core(imap, mask,
                    calibration, pol_eff, ivar=None,
                    maptype='native',
                    dfact = None,
                    inpaint_mask=None,
                    kspace_mask=None, 
                    foreground_cluster=None, cal_cluster=1.):
    """
    This function will load a rectangular pixel map and pre-process it.
    This involves inpainting, masking in real and Fourier space
    and removing a pixel window function. It also removes a calibration
    and polarization efficiency.
    For simulations ivar processing is redundant, we should probably set ivar as an optional argument
    Returns beam convolved (transfer uncorrected) T, Q, U maps.
    """
    # for Planck, assert that we extract the RA DEC of the ACT footprint only
    
    if dfact!=1 and (dfact is not None):
        imap = enmap.downgrade(imap,dfact)
        if ivar is not None:
            ivar = enmap.downgrade(ivar,dfact,op=np.sum)
    
    oshape = (3,) + mask.shape if imap.ndim==3 else mask.shape
    if imap[0].shape != mask.shape:
        imap = enmap.extract(imap, oshape, mask.wcs)
        if ivar is not None:
            ivar = enmap.extract(ivar, oshape, mask.wcs)
    
    # Subtract cluster model first, accounting for calibration
    if foreground_cluster is not None:
        if imap.ndim==3:
            imap[0] = imap[0] - (foreground_cluster / cal_cluster)
        else:
            imap = imap - (foreground_cluster / cal_cluster) 

    # Then inpaint
    if inpaint_mask is not None:
        # assert ivar is not None, "need ivar for inpainting" -- not true, random noise ivar
        # imap = inpaint_temperature_only(imap, inpaint_mask) 
        imap = maps.gapfill_edge_conv_flat(imap, inpaint_mask) # , ivar=ivar)
            
    # Check that non-finite regions are in masked region; then set non-finite to zero
    if not(np.all((np.isfinite(imap[...,mask>1e-3])))): raise ValueError
    imap[~np.isfinite(imap)] = 0
    
    # if deconvolve_beam_bool:
    #     print('deconv beam')
    #     imap = deconvolve_beam(imap, mask, mlmax, beam=beam, leakage=leakage)

    imap = imap * mask
    imap = depix_map(imap,maptype=maptype,dfact=dfact,kspace_mask=kspace_mask)

    imap = imap * calibration
    if imap.ndim==3:
        imap[1:] = imap[1:] / pol_eff
    
    if ivar is not None:
        ivar = ivar / calibration**2.
        if imap.ndim==3:
            ivar[1:] = ivar[1:] * pol_eff**2.
    
    return imap, ivar # ivar will be none if nothing happened to it

def get_signal_sim_core(shape,wcs,signal_alms,
                        beam_fells,transfer_fells,
                        calibration,pol_eff,
                        apod_y_arcmin = 0.,apod_x_arcmin = 0.,
                        maptype='native'):
    """
    This needs to only be called once if the beam, transfer, cal, poleff, etc. are the same for each split.
    """
    
    isignal_alms = signal_alms.copy()
    isignal_alms[0] = cs.almxfl(isignal_alms[0],beam_fells * transfer_fells)
    isignal_alms[1] = cs.almxfl(isignal_alms[1],beam_fells)
    isignal_alms[2] = cs.almxfl(isignal_alms[2],beam_fells)
    omap = cs.alm2map(isignal_alms,enmap.empty((3,)+shape,wcs,dtype=np.float32))
    if maptype=='native':
        if (apod_y_arcmin>1e-3) or (apod_x_arcmin>1e-3):
            res = maps.resolution(shape,wcs) / u.arcmin
            omap = enmap.apod(omap, (apod_y_arcmin/res,apod_x_arcmin/res))
        omap = enmap.apply_window(omap,pow=1)
    elif maptype=='reprojected':
        pass
    else:
        raise ValueError

    # notice how these are inverse of what's in preprocess
    # not applied to nmap because it is already based on data (which includes them)
    omap = omap / calibration  
    omap[1:] = omap[1:] * pol_eff
    
    return omap
    
def get_noise_sim_core(shape,wcs,
                       noise_alms=None,
                       noise_mask=None,
                       rms_uk_arcmin=None,
                       noise_lmax = 5400,
                       lcosine=80):
    """
    This needs to be called each time for each split.
    """
    
    if noise_alms is not None:
        if noise_mask is not None:
            lstitch = noise_lmax - 200
            mlmax = noise_lmax + 600
            nmap = maps.stitched_noise(shape,wcs,noise_alms,noise_mask,rms_uk_arcmin=rms_uk_arcmin,
                                       lstitch=lstitch,lcosine=lcosine,mlmax=mlmax)
        else:
            nmap = cs.alm2map(noise_alms,enmap.empty((3,)+shape,wcs))
            nmap[~np.isfinite(nmap)] = 0.
    else:
        nmap = 0.

    return nmap
    
def get_sim_core(shape,wcs,signal_alms,
                 beam_fells, transfer_fells,
                 calibration,pol_eff,
                 maptype='native',
                 noise_alms=None,
                 apod_y_arcmin = 0.,
                 apod_x_arcmin = 0.,
                 noise_mask=None,
                 rms_uk_arcmin=None,
                 noise_lmax = 5400,
                 lcosine=80): # these are all optional arguments for noise stitching

    """
    shape,wcs should ideally correspond to native ACT/SO pixelization of 0.5 arcmin
    beam_fells and transfer_fells should be same size and go from ell=0 to beyond relevant ells

    You probably shouldn't be calling this function since it is wasteful when doing splits.

    """
    omap = get_signal_sim_core(shape,wcs,signal_alms,
                               beam_fells, transfer_fells,
                               calibration, pol_eff,
                               apod_y_arcmin, apod_x_arcmin,
                               maptype)
    nmap = get_noise_sim_core(shape,wcs,
                       noise_alms,
                       noise_mask,
                       rms_uk_arcmin,
                       noise_lmax,
                       lcosine)
    omap = omap + nmap
    return omap


def calculate_noise_power(nmap, mask, mlmax, nsplits, pureEB):    
    
    ''' 
    equation (9) Qu+24 paper
    '''
    
    cl_ab = []
    for k in range(nsplits):

        if pureEB:
            _,Balm=simgen.pureEB(nmap[k][1],nmap[k][2],mask,returnMask=0,lmax=mlmax,isHealpix=False)
            alm_T, alm_E, _ =cs.map2alm(nmap[k],lmax=mlmax)
            alm_a=np.array([alm_T,alm_E,Balm], dtype=np.complex128)
            cl_ab.append(cs.alm2cl(alm_a))
        else:
            alm_a = np.array(cs.map2alm(nmap[k],lmax=mlmax), dtype=np.complex128)
            cl_ab.append(cs.alm2cl(alm_a)) 

    w2=simgen.w_n(mask,2)
    cl_sum = np.sum(cl_ab, axis=0)
    power = 1/nsplits/(nsplits-1) * cl_sum
    power[~np.isfinite(power)] = 0
    
    return power / w2

def get_name_weights(qid, spec):

    return f'noise_{qid}_{spec}'

def get_name_cluster_fgmap(qid, isplit=1, coadd=False):
    
    '''
    the cluster finding algorithm has a corresponding beam.
    we take that beam out and apply in the beam of the experiment. 
    this could be the beam for the exact qid and split or all splits with the same (coadd) beam, hence the two options
    '''
    
    if coadd:
        return f'{qid}_nemo' # !!!
    else:
        return f'{qid}_split{isplit}_nemo'

def get_name_run(args, split=None, coadd=False):

    name_run = f'mnemo{args.cluster_subtraction}_{args.mask_tag}'
    
    if split is not None:
        name_run += f'_split{split}'
    
    if coadd:
        name_run += '_coadd'
        
    return name_run, f'{"_".join(args.qids)}_{name_run}'

def get_name_sim(sim_tag, task, args):
    
    name_run = f'{sim_tag}_{(args.sims_start+task):05}'
    return name_run, f'{"_".join(args.qids)}_{name_run}'

def get_mask_tag(mask_fn, mask_subproduct):

    """
    Extracts and returns the tag from the mask filename.
    """

    assert 'lensing_masks' in mask_subproduct, 'mask tag only implemented for lensing masks'
    # find daynight tag "daydeep", "daywide" o "night"
    daynight = re.search(r'(daydeep|daywide|night|pipe4|mss0002)', mask_fn).group(0)
    # find skyfraction. it is 2 digit number. the string between the last "_" and ".fits"
    skyfrac  = re.search(r'_(\d{2})(?:_[^_]+)?\.fits$', mask_fn).group(1) # re.search(r'_([^_]+)\.fits$', mask_fn).group(1)

    return f'{daynight}_{skyfrac}'

def apply_ellmin_taper(noise, ellmin, delta_ell=15, blowup=1e10):
    """
    Apply an ℓmin taper to a noise power spectrum.
    
    Parameters
    ----------
    noise : np.ndarray
        Input noise array of shape (Nell,), indexed by ell.
    ellmin : int
        ℓmin cutoff (array-specific).
    delta_ell : int, optional
        Width of smooth transition (default=15). Set to 0 for a hard cut.
    blowup : float, optional
        Factor to inflate the noise below cutoff (default=1e10).
    
    Returns
    -------
    noise_mod : np.ndarray
        Modified noise array with inflated values below ellmin.
    """
    ell = np.arange(len(noise))
    noise_mod = noise.copy()

    if delta_ell == 0:
        # Hard cut: multiply by huge number below ellmin
        mask = ell < ellmin
        noise_mod[mask] *= blowup
    else:
        # Smooth taper from blowup at (ellmin - delta_ell) → normal at (ellmin + delta_ell)
        x = (ell - (ellmin - delta_ell)) / (2 * delta_ell)
        # window goes from 0 to 1 smoothly
        window = np.clip(0.5 * (1 - np.cos(np.pi * np.clip(x, 0, 1))), 0, 1)
        # effective multiplier: blowup below cutoff, ~1 above
        mult = blowup * (1 - window) + 1.0 * window
        noise_mod *= mult

    return noise_mod

def read_weights(args):
    
    '''
    reads in the fcoadd weights
    args.use_ps_cut:
    '''

    nqids = len(args.qids)
    noise_specs = np.zeros((nspecs, nqids, args.mlmax+1), dtype=np.float64)

    if args.pureEB:
        specs = specs_weights['EpureB']
    else:
        specs = specs_weights['EB']

    def get_ellmin(qid):
        if not args.use_ps_cut:
            return args.lmin
        ps_cut_values = {"pa4b": 1000, "pa5a": 1000, "pa5b": 800, "pa6a": 1000, "pa6b": 600}
        return ps_cut_values.get(qid, args.lmin)

    for i, qid in enumerate(args.qids):
        for ispec, spec in enumerate(specs):
            noise=np.loadtxt(get_fout_name(get_name_weights(qid, spec), args, stage='weights'))[:args.mlmax+1]
            noise_cut = apply_ellmin_taper(noise, get_ellmin(qid), delta_ell=25, blowup=1e10)
            noise_specs[ispec, i] = noise_cut
    
    return noise_specs

def get_fout_name(fname, args, stage, tag=None):

    '''
    fname: name of file to be saved
    args: argparse object containing args.output_dir
    stage: str
        ['weights', 'cluster_fgmap', 'kspace_coadd', 'nilc_coadd']
    tag: optional, used in kspace_coadd and nilc_coadd to distinguish between sim (tag='sim') and data (tag=None), because we store them in different folders
    '''
    fname = fname.split('.fits')[0]
    
    try:
        fcoadd_folder = f'{args.mask_tag}_fcoadd'
    except AttributeError:
        fcoadd_folder = "default_fcoadd"

    if hasattr(args, "no_fcoadd_folder"):
        no_fcoadd_folder = args.no_fcoadd_folder
    else:
        no_fcoadd_folder = False

    if stage == 'weights':
        fname += '_weights.txt'
        if no_fcoadd_folder:
            folder = f'../stage_compute_weights/'
        else:
            folder = f'../{fcoadd_folder}/stage_compute_weights/'
    
    elif stage == 'cluster_fgmap':
        fname += '_cluster_fgmap.fits'
        if no_fcoadd_folder:
            folder = f'../stage_cluster_fgmap/'
        else:
            folder = f'../../{fcoadd_folder}/stage_cluster_fgmap/'

    elif stage == 'kspace_coadd':
        fname  = 'kspace_coadd_' + fname + '.fits'
        if tag == 'sim':
            folder = '../stage_kspace_coadd_sims/'
        else:
            folder = '../stage_kspace_coadd/'

    elif stage == 'nilc_coadd':
        fname  = 'nilc_coadd_' + fname + '.fits'
        if tag == 'sim':
            folder = '../stage_nilc_coadd_sims/'
        else:
            folder = '../stage_nilc_coadd'

    output_dir = os.path.join(args.output_dir, folder)
    # create output folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, fname)
