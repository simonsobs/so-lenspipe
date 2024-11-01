from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
from orphics import maps,io,cosmology,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs, utils, enplot,bunch
import pytempura
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from falafel import qe
import os
import glob
import traceback,warnings
from . import bias, solenspipe, filtering as optfilt
from falafel import utils as futils

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']
output_sim_path = config['output_sim_path']

NITER = 200
NITER_MASKED_CG = 25
ERR_TOL = 1e-5
COMPUTE_QE = None
EVAL_EVERY_NITERS = 10

class LensingSandboxOF(solenspipe.LensingSandbox):
    def __init__(self, lmax_of=None, mlmax_of=None,
                 ivar=None, lmax_prec_cg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lmax_of = self.lmax if lmax_of is None else lmax_of
        self.mlmax_of = self.mlmax if mlmax_of is None else mlmax_of

        assert self.mlmax_of >= self.mlmax
        assert self.lmax_of >= self.lmax
        
        self.ivar = ivar if ivar is not None else \
                    maps.ivar(self.shape, self.wcs, self.noise)

        self.lmax_prec_cg = self.lmax - 1000 if lmax_prec_cg is None \
                            else lmax_prec_cg
        
        # for optimal filtering
        
        # for pol, sqrt(2) times noise level is 1/2 times ivar 
        self.icov_pix = enmap.enmap([self.ivar, self.ivar*0.5, self.ivar*0.5])

        self.mask_bool = enmap.enmap(
                            np.concatenate([self.mask[np.newaxis, :]]*3, axis=0)
                         ).astype(bool)
        
        self.output_sim_path = output_sim_path
    
    def get_observed_map(self,index,iset=0):
        shape,wcs = self.shape,self.wcs
        calm = futils.change_alm_lmax(futils.get_cmb_alm(index,iset),
                                      self.mlmax_of)
        calm = cs.almxfl(calm,lambda x: maps.gauss_beam(x,self.fwhm))
        # ignoring pixel window function here
        omap = cs.alm2map(calm,enmap.empty((3,)+shape,wcs,
                                           dtype=np.float32),spin=[0,2])
        if self.add_noise:
            nmap = maps.white_noise((3,)+shape,wcs,self.noise)
            nmap[1:] *= np.sqrt(2.)
        else:
            nmap = 0.
        return enmap.enmap(self._apply_mask_binary(omap + nmap, self.mask),
                           omap.wcs)

    def lmin_filter(self, ialm, lmin=None):
        # high pass filter to lmin
        if lmin is None: lmin = self.lmin
        hpass = np.ones(self.mlmax_of)
        hpass[:lmin] = 0.

        return cs.almxfl(ialm, hpass)
    
    def lmax_filter(self, ialm, lmax=None):
        # low pass filter to lmax
        if lmax is None: lmax = self.lmax
        lpass = np.ones(self.mlmax_of)
        lpass[lmax:] = 0.

        return cs.almxfl(ialm, lpass)
    
    def prepare(self, omap, save_output=None,
                niter=NITER,
                niter_masked_cg=NITER_MASKED_CG,
                err_tol=ERR_TOL,
                compute_qe=COMPUTE_QE,
                eval_every_niters=EVAL_EVERY_NITERS):
        # run optimal filtering
        ucls, tcls = futils.get_theory_dicts_white_noise(self.fwhm, self.noise,
                                                         grad=False, lmax=self.mlmax_of)
        b_ell = maps.gauss_beam(np.arange(self.mlmax_of), self.fwhm)

        filt = optfilt.CGPixFilter(ucls, b_ell, icov_pix=self.icov_pix,
                                   mask_bool=self.mask_bool,
                                   include_te=(not self.no_te_corr),
                                   lmax=self.lmax_of, swap_bm=True,
                                   lmax_prec_cg=self.lmax_prec_cg, mlmax=self.mlmax_of)

        # zero out input map at masked locations
        omap[~self.mask_bool] = 0.

        alm_dict = filt.filter(omap, niter=niter,
                               niter_masked_cg=niter_masked_cg, 
                               benchmark=False, verbose=True,
                               err_tol=err_tol, compute_qe=compute_qe,
                               eval_every_niters=eval_every_niters,
                               tcls=tcls)

        ialm = alm_dict['ialm']
        if save_output is not None:
            walm = alm_dict['walm']
            hp.write_alm(save_output.replace(".fits", f"_ialm_lmax{self.lmax_of}.fits"), ialm)
            hp.write_alm(save_output.replace(".fits", f"_walm_lmax{self.lmax_of}.fits"), walm)
        # top hat filter
        return self.lmax_filter(self.lmin_filter(ialm))
    
    def kmap(self, stuple, nstep=512):
        icov,ip,i = stuple
        if i>nstep: raise ValueError
        if ip==0 or ip==1:
            iset = 0
            index = nstep*ip + i
        elif ip==2 or ip==3:
            iset = ip - 2
            index = 2*nstep + i
        dmap = self.get_observed_map(index,iset)

        # specific scheme for v0.4 sims
        filename = self.output_sim_path + \
                   f"fullskyLensedCMB_alm_set{str(iset).zfill(2)}_{str(i).zfill(5)}.fits"
        filename_ialm = filename.replace(".fits", f"_ialm_lmax{self.lmax_of}.fits")

        if os.path.exists(filename_ialm):
            if self.verbose:
                print(f"Found {filename_ialm}, skipping filtering.")
            X = hp.read_alm(filename_ialm, hdu=(1,2,3))
            X = self.lmax_filter(self.lmin_filter(X))
        else:
            X = self.prepare(dmap, save_output=filename)
        return X

    def get_rdn0(self,prepared_data_alms,est,comm,nsims=None):
        Xdata = prepared_data_alms
        if nsims is None: nsims = self.n0_sims

        return bias.simple_rdn0(0,est,est,lambda alpha,X,Y: self.qfuncs[alpha](X,Y),
                                self.kmap,comm,cs.alm2cl,nsims,Xdata)

    def get_mcn1(self,est,comm,nsims=None):
        if nsims is None: nsims = self.n1_sims
        return bias.mcn1(0,self.kmap,cs.alm2cl,nsims,
                         self.qfuncs[est],comm=comm,verbose=True).mean(axis=0)
    
    def get_mcmf(self,est,comm,nsims=None):
        if nsims is None: nsims = self.mf_sims
        return bias.mcmf_pair(0,self.qfuncs[est],
                              self.kmap,
                              comm,nsims)

# for testing purposes only
class LensingSandboxOFHyperparams(LensingSandboxOF):
    def __init__(self,
                 niter=NITER,
                 niter_masked_cg=NITER_MASKED_CG,
                 err_tol=ERR_TOL,
                 compute_qe=COMPUTE_QE,
                 eval_every_niters=EVAL_EVERY_NITERS,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.niter = niter
        self.niter_masked_cg = niter_masked_cg
        self.err_tol = err_tol
        self.compute_qe = compute_qe
        self.eval_every_niters = eval_every_niters

    def prepare(self, omap, save_output=None, lmax_of=5400, mlmax_of=6000):
        return super().prepare(omap,
                               save_output=save_output,
                               niter=self.niter,
                               niter_masked_cg=self.niter_masked_cg,
                               err_tol=self.err_tol,
                               compute_qe=self.compute_qe,
                               eval_every_niters=self.eval_every_niters,
                               lmax_of=lmax_of,
                               mlmax_of=mlmax_of)


