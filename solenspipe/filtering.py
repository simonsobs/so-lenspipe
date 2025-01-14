from timeit import default_timer as timer
import warnings

import numpy as np

from falafel import qe, utils as futils
import pytempura
import healpy as hp
from orphics import maps, stats
from pixell import enmap, reproject, lensing as plensing, curvedsky as cs
from optweight import map_utils, mat_utils, solvers, preconditioners, sht

### optional function
# performs isotropic Wiener filtering with / without TE mode mixing
def isotropic_wfilter(alm,ucls,tcls,lmin,lmax,ignore_te=True):
    ucltt, tcltt = ucls['TT'], tcls['TT']
    uclte, tclte = ucls['TE'], tcls['TE']
    uclee, tclee = ucls['EE'], tcls['EE']
    uclbb, tclbb = ucls['BB'], tcls['BB']

    if ignore_te:
        filt_T, filt_E, filt_B = tcltt*0, tclee*0, tclbb*0
        with np.errstate(divide='ignore', invalid='ignore'):
            filt_T[2:] = ucltt[2:]/tcltt[2:]
            filt_E[2:] = uclee[2:]/tclee[2:]
            filt_B[2:] = uclbb[2:]/tclbb[2:]
        talm = qe.filter_alms(alm[0],filt_T,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[1],filt_E,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_B,lmin=lmin,lmax=lmax)

    else:
        filt_TT, filt_TE, filt_ET, filt_EE = tcltt*0, tclte*0, tclte*0, tclee*0
        filt_BB = tclbb*0

        with np.errstate(divide='ignore', invalid='ignore'):
            te_det = tcltt[2:]*tclee[2:] - tclte[2:]**2.
            filt_TT[2:] = (ucltt[2:]*tclee[2:] - uclte[2:]*tcltt[2:]) / te_det
            filt_TE[2:] = (uclte[2:]*tcltt[2:] - ucltt[2:]*tclte[2:]) / te_det
            # these two are no longer symmetric 
            filt_ET[2:] = (uclte[2:]*tclee[2:] - uclee[2:]*tclte[2:]) / te_det
            filt_EE[2:] = (uclee[2:]*tcltt[2:] - uclte[2:]*tclte[2:]) / te_det
            filt_B[2:] = uclbb[2:]/tclbb[2:]
        talm = qe.filter_alms(alm[0],filt_TT,lmin=lmin,lmax=lmax) + \
               qe.filter_alms(alm[1],filt_TE,lmin=lmin,lmax=lmax)
        ealm = qe.filter_alms(alm[0],filt_ET,lmin=lmin,lmax=lmax) + \
               qe.filter_alms(alm[1],filt_EE,lmin=lmin,lmax=lmax)
        balm = qe.filter_alms(alm[2],filt_BB,lmin=lmin,lmax=lmax)
        
    return [talm,ealm,balm]

### helper functions
# criterion for filtering
def convergence(errors_obj, est, limit=1e-3):
    return errors_obj[est][-1] <= limit

# calculate worst agreement between cl_new vs cl_old
# between lmin to lmax with linearly spaced nbins
def worst_agreement(cl_new, cl_old, lmin=600, lmax=3000,
                    mlmax=4000, nbins=50):
    ells = np.arange(2, mlmax+1)
    bin_edges = np.linspace(2, mlmax+1, nbins)
    binner = stats.bin1D(bin_edges)
    def binned(x): return binner.bin(ells, x)

    binned_ells, cl_new_binned = binned(cl_new[ells])
    _          , cl_old_binned = binned(cl_old[ells])

    filtered_ells = np.logical_and(binned_ells >= lmin,
                                   binned_ells <= lmax)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cl_ratio = np.nan_to_num(cl_new_binned / cl_old_binned - 1.)[filtered_ells]
    return np.max(np.abs(cl_ratio))

# return normalized, (likely biased) full-sky reconstruction
def reconstruct(falm, px, ests, ucls, lmax, Als=None):
    ialm = [falm, falm*0., falm*0.] if len(falm.shape) == 1 else falm

    phis = qe.qe_all(px, ucls, lmax,
                     fTalm=ialm[0], fEalm=ialm[1], fBalm=ialm[2],
                     estimators=ests,
                     xfTalm=ialm[0], xfEalm=ialm[1], xfBalm=ialm[2])
    kappas = {}
    for est in ests:
        kappa = plensing.phi_to_kappa(
                    hp.almxfl(phis[est][0].astype(np.complex128),
                              Als[est.upper()][0])
                )
        # zero out nans and ell < 2
        kappa[~np.isfinite(kappa)] = 0.
        kappas[est] = hp.alm2cl(kappa, kappa)
    return kappas

class CGPixFilter(object):
    def __init__(self, theory_cls, b_ell, icov_pix, mask_bool,
                 include_te=True, q_low=0, q_high=1, swap_bm=False,
                 scale_a=False, lmax=None, mlmax=None, lmax_prec_cg=None):
        """
        Prepare to filter maps using a pixel-space instrument noise model
        and a harmonic space signal model. 

        Parameters
        ----------
        theory_cls : dict
            A dictionary mapping the keys TT and optionally TE, EE and
            BB to 1d numpy arrays containing CMB C_ell power spectra 
            (no 2pi, ell^2 or beam factors), starting at ell=0 and up to at
            least lmax. Should have units (e.g. uK^2) consistent with alm 
            and icov inputs.
        b_ell : (nells,) or (ncomp,nells) array
            A numpy array containing the map-space beam transfer function
            (starting at ell=0) to assume in the noise model. Separate
            beams can be specified for T,E,B if the array is 2d.
        icov_pix : (ncomp,ncomp,Ny,Nx), (ncomp,Ny,Nx) ndmap
            An enmap containing the inverse (co-)variance per pixel in units
            (e.g. 1/uK^2) consistent with the alms and theory_cls. IQ, IU, QU
            elements can also be specified if icov_pix is 4-dimensional.
        mask_bool : (ncomp,Ny,Nx) or (Ny,Nx) ndmap
            Boolean mask (True for observed pixels). Geometry must match that of
            'icov_pix'.
        include_te : bool, optional
            Whether or not to jointly filter T,E,B maps by accounting for the
            signal TE correlation. If True, the returned alms will be optimally
            filtered, but the "T" and "E" maps will not be pure-T and pure-E.
        q_low : float or (ncomp) array, optional
            Pixels in icov map with values below this quantile are thresholded.
            May be set per polarization.
        q_high : float or (ncomp) array, optional
            Pixels in icov map with values above this quantile are thresholded.
            May be set  per polarization.                
        swap_bm : bool, optional
            Swap the order of the beam and mask operations. Helps convergence
            with large beams and high SNR data.
        scale_a : bool, optional
            If set, scale the A matrix to localization of N^-1 term. This may
            help convergence with small beams and high SNR data.
        lmax : int, optional
            If given, solve the system up to this lmax. Will be determined from
            icov_pix geometry is not provided.
        lmax_prec_cg : int, optional
            Only apply the masked CG precondtioner to multipoles up to lmax.
            Can be set to multipole where S/N < 1 to speed up the precondition.
        """

        if np.any(np.logical_not(np.isfinite(b_ell))): raise Exception

        if np.any(np.logical_not(np.isfinite(icov_pix))): raise Exception

        shape_in = icov_pix.shape[-2:]
        
        icov_pix = mat_utils.atleast_nd(icov_pix, 3)
        mask_bool = mat_utils.atleast_nd(mask_bool.astype(bool, copy=False), 3)        
                            
        ncomp = icov_pix.shape[0]
        if mask_bool.shape[0] == 1:
            mask_bool = (np.ones(ncomp)[:,np.newaxis,np.newaxis] * mask_bool).astype(bool)

        for mtype in ['CC', 'fejer1',]:
            try:
                minfo = map_utils.match_enmap_minfo(
                    icov_pix.shape, icov_pix.wcs, mtype=mtype)
            except ValueError:
                continue
            else:
                break

        if lmax is None:
            lmax = map_utils.minfo2lmax(minfo)
        if mlmax is None: mlmax = lmax + 1000
        icov_pix = map_utils.view_1d(icov_pix, minfo)
        mask_bool = map_utils.view_1d(mask_bool, minfo)        

        if q_low != 0 or q_high != 1:
            icov_pix = map_utils.threshold_icov(icov_pix, q_low=q_low, q_high=q_high)
                                                
        tlmax = theory_cls['TT'].size - 1
        if not(tlmax >= lmax): raise Exception
        cov_ell = np.zeros((ncomp, ncomp, lmax + 1))
        cov_ell[0,0] = theory_cls['TT'][:lmax+1]
        if ncomp > 1:
            if include_te:
                cov_ell[0,1] = theory_cls['TE'][:lmax+1]
                cov_ell[1,0] = theory_cls['TE'][:lmax+1]
            cov_ell[1,1] = theory_cls['EE'][:lmax+1]
            cov_ell[2,2] = theory_cls['BB'][:lmax+1]
                                                
        # Invert to get inverse signal cov.
        icov_ell = np.zeros_like(cov_ell)
        for lidx in range(icov_ell.shape[-1]):
            icov_ell[:,:,lidx] = np.linalg.pinv(cov_ell[:,:,lidx])
                
        if b_ell.ndim == 1:
            b_ell = b_ell[np.newaxis] * np.asarray((1, 1, 1)[:ncomp])[:,np.newaxis]
        elif b_ell.ndim == 2:
            if b_ell.shape[0] != ncomp: raise Exception
        else:
            raise ValueError

        b_ell = np.ascontiguousarray(b_ell[:,:lmax+1])
        
        if scale_a:
            sfilt = mat_utils.matpow(b_ell, -0.5)
        else:
            sfilt = None

        ainfo = cs.alm_info(lmax)

        if ncomp == 1:
            spin = 0
        elif ncomp == 3:
            spin = [0, 2]

        prec_pinv = preconditioners.PseudoInvPreconditioner(
            ainfo, icov_ell, icov_pix, minfo, spin, b_ell=b_ell, sfilt=sfilt)
        
        prec_harm = preconditioners.HarmonicPreconditioner(
            ainfo, icov_ell, icov_pix=icov_pix, minfo=minfo, 
            b_ell=b_ell, sfilt=sfilt
        )

        # if at least something is masked (zero)
        prec_masked_mg = None

        if np.nonzero(mask_bool[0])[0].size < mask_bool[0].size:
            prec_masked_cg = preconditioners.MaskedPreconditionerCG(
                ainfo, icov_ell, spin, mask_bool[0].astype(bool), minfo,
                lmax=lmax_prec_cg if lmax_prec_cg else lmax, nsteps=15,
                lmax_r_ell=None, sfilt=sfilt)
        else:
            prec_masked_cg = None

        self.shape_in = shape_in
        self.icov_ell = icov_ell
        self.icov_pix = icov_pix
        self.mask_bool = mask_bool
        self.minfo = minfo
        self.b_ell = b_ell
        self.sfilt = sfilt
        self.ncomp = ncomp
        self.swap_bm = swap_bm
        self.lmax = lmax
        self.mlmax = mlmax
        self.ainfo = ainfo
        self.spin = spin
        self.prec_pinv = prec_pinv
        self.prec_masked_cg = prec_masked_cg
        self.prec_masked_mg = prec_masked_mg 
        self.prec_harm = prec_harm 
        self.theory_cls = theory_cls 

    # err_tol is defined as |Ax-b| / |b| < err_tol
    # harmonic preconditioner 
    def filter(self, imap, niter=None, niter_masked_cg=5, 
               benchmark=False, verbose=True, err_tol=1e-5,
               compute_qe=None, compute_qe_after=None,
               eval_every_niters=1, tcls=None):

        assert imap.shape == (self.ncomp,) + self.shape_in 

        imap = map_utils.view_1d(imap.astype(self.icov_pix.dtype,
                                            copy=False),
                                 self.minfo)

        solver = solvers.CGWienerMap.from_arrays(imap, self.minfo, self.ainfo, self.icov_ell, 
                                                 self.icov_pix, b_ell=self.b_ell,
                                                 draw_constr=False, mask_pix=self.mask_bool,
                                                 swap_bm=self.swap_bm, spin=self.spin,
                                                 sfilt=self.sfilt)
        
        solver.add_preconditioner(self.prec_harm)

        is_mask_present = np.nonzero(self.mask_bool[0])[0].size < self.mask_bool[0].size
        if is_mask_present:
            solver.add_preconditioner(self.prec_masked_cg)
            pass
        else:
            print("No masked pixels detected; not using masked preconditioners")

        solver.init_solver()
        
        times = []
        errors = []
        errors.append(np.nan)

        if benchmark:
            warnings.warn("optweight: Benchmarking is turned on. "\
                            "This significantly slows down the filtering.")
            chisqs = []
            residuals = []
            qforms = []
            ps_c_ells = []
            itnums = []
            chisqs.append(solver.get_chisq())
            residuals.append(solver.get_residual())
            itnums.append(0)
            if verbose:
                print('|b| :', np.sqrt(solver.dot(solver.b0, solver.b0)))

        if niter is None:
            niter = 15
            warnings.warn(f"optweight: Using the default number of iterations :"\
                            f"{niter_masked_cg=} + {niter=}.")
            
        if compute_qe is not None:
            kappas = {}
            kappa_errors = {}
            ests = ['TT', 'mvpol', 'mv']
            for est in ests:
                kappas[est] = {}
                kappa_errors[est] = []
            px = qe.pixelization(self.shape_in, imap.wcs)
            with np.errstate(divide='ignore', invalid='ignore'):
                Als = pytempura.get_norms([est.upper() for est in ests],
                                          self.theory_cls, self.theory_cls,
                                          (tcls if tcls is not None else self.theory_cls),
                                          2,self.lmax,k_ellmax=self.mlmax)

        if verbose:
            print(f"Running optimal filtering using {sht.get_nthreads()} threads per process.")
        for idx in range(niter_masked_cg + niter):
            if idx == niter_masked_cg:
                solver.reset_preconditioner()
                solver.add_preconditioner(self.prec_harm)

                solver.b_vec = solver.b0
                solver.init_solver(x0=solver.x)

            t_start = timer()
            solver.step()
            t_eval = timer() - t_start
            
            if (idx % eval_every_niters == 0):
                b0 = solver.b0
                error = solver.get_residual() / np.sqrt(solver.dot(b0, b0))
                errors.append(error)

                if verbose:
                    print(f"optweight step {idx + 1} / {niter_masked_cg + niter}, "\
                          f"|Ax-b|/|b|: {errors[-1]:.2e}, time {t_eval:.3f} s")
                if errors[-1] < err_tol**(2 if not is_mask_present else 1):
                    warnings.warn(f"Stopping early because {error=} is below {err_tol=}")
                    break

                if compute_qe is not None and (idx % compute_qe == 0):
                    if compute_qe_after is None or idx >= compute_qe_after:
                        falm = solver.get_icov()
                        kappa_dict = reconstruct(falm, px, ests,
                                                self.theory_cls, self.mlmax, Als)
                        
                        if verbose:
                            print(f"optweight step {idx + 1} / {niter_masked_cg + niter}, "\
                                 f"kappa maxdiff: ")
                        for est in ests:
                            kappas[est][idx] = kappa_dict[est]
                            try:
                                kappa_errors[est].append(
                                    worst_agreement(kappa_dict[est],
                                                    kappas[est][idx-compute_qe],
                                                    lmax=self.lmax, mlmax=self.mlmax)
                                )
                            except KeyError:
                                kappa_errors[est].append(1.)
                            
                            if verbose: print(f"| {est}: {kappa_errors[est][-1]: .2%} ", end="")

                        del kappa_dict
                        if verbose: print(f" |")

                        if convergence(kappa_errors, 'mv'):
                            if verbose: print("optweight converged due to MV QE limit.")
                            warnings.warn(f"Stopping early because MV QE max err ({kappa_errors[est][-1]}) is below 1e-3.")
                            break
        
            times.append(t_eval)
            if benchmark:
                if (idx+1)%benchmark==0:
                    chisq = solver.get_chisq()
                    residual = solver.get_residual()
                    qform = solver.get_qform()
                    chisqs.append(chisq)
                    residuals.append(residual)
                    qforms.append(qform)                    
                    ps_c_ells.append(self.ainfo.alm2cl(
                        solver.get_wiener()[:,None,:], solver.get_wiener()[None,:,:]))
                    itnums.append(idx)
                    print(f"optweight benchmark: \t chisq : {chisq:.2f} \t "
                            f"residual : {residual:.2f} \t qform : {qform:.2f}")
        
        if verbose:
            print(f"Total time elapsed: {np.sum(np.array(times)):.3f} seconds")

        output = {}
        output['walm'] = solver.get_wiener()
        output['ialm'] = solver.get_icov()
        output['solver'] = solver
        output['errors'] = errors
        output['time'] = np.sum(np.array(times))
        output['niters'] = idx+1
        if benchmark:
            output['chisqs'] = chisqs
            output['residuals'] = residuals
            output['qforms'] = qforms
            output['ps'] = ps_c_ells
            output['itnums'] = itnums
            
        return output