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
from falafel.utils import get_cmb_alm, get_kappa_alm, \
    get_theory_dicts, get_theory_dicts_white_noise, \
    change_alm_lmax
from falafel import utils as futils

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

NITER = 200
NITER_MASKED_CG = 10
ERR_TOL = 5e-5
COMPUTE_QE = None
EVAL_EVERY_NITERS = 10

class LensingSandboxOF(solenspipe.LensingSandbox):
    def __init__(self, ivar=None, lmax_prec_cg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        
        # change as desired
        self.output_sim_path = "/data5/sims/v0.4_filter/"
        
    def kmap(self,stuple, nstep=500):
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
        filename_ialm = filename.replace(".fits", "_ialm.fits")

        if os.path.exists(filename_ialm):
            if self.verbose:
                print(f"Found {filename_ialm}, skipping filtering.")
            X = hp.read_alm(filename_ialm, hdu=(1,2,3))
        else:
            X = self.prepare(dmap, save_output=filename)
        return X

    def prepare(self, omap, save_output=None):
        # run optimal filtering
        ucls, tcls = get_theory_dicts_white_noise(self.fwhm, self.noise,
                                                  grad=False, lmax=self.mlmax)
        b_ell = maps.gauss_beam(np.arange(self.mlmax), self.fwhm)

        filt = optfilt.CGPixFilter(ucls, b_ell, icov_pix=self.icov_pix,
                                   mask_bool=self.mask_bool,
                                   include_te=(not self.no_te_corr),
                                   lmax=self.lmax, swap_bm=True,
                                   lmax_prec_cg=self.lmax_prec_cg, mlmax=self.mlmax)

        # zero out input map at masked locations
        omap[~self.mask_bool] = 0.

        alm_dict = filt.filter(omap, niter=NITER,
                               niter_masked_cg=NITER_MASKED_CG, 
                               benchmark=False, verbose=True,
                               err_tol=ERR_TOL, compute_qe=COMPUTE_QE,
                               eval_every_niters=EVAL_EVERY_NITERS,
                               tcls=tcls)

        # high pass filter to lmin
        hpass = np.ones(self.mlmax)
        hpass[:self.lmin] = 0.

        ialm = cs.almxfl(alm_dict['ialm'], hpass)
        if save_output is not None:
            walm = cs.almxfl(alm_dict['walm'], hpass)
            hp.write_alm(save_output.replace(".fits", "_ialm.fits"), ialm)
            hp.write_alm(save_output.replace(".fits", "_walm.fits"), walm)

        return ialm




