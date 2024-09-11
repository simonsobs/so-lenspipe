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
    change_alm_lmax, get_theory_dicts_white_noise
from falafel import utils as futils

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

NITER = 30
NITER_MASKED_CG = 0
ERR_TOL = 1e-5
COMPUTE_QE = None
EVAL_EVERY_NITERS = 2

class LensingSandboxOF(solenspipe.LensingSandbox):
    def __init__(self, mask=None, ivar=None, lmax_prec_cg=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask if mask is not None else \
                    enmap.ones(self.shape, self.wcs)
        self.ivar = ivar if ivar is not None else \
                    maps.ivar(self.shape, self.wcs, self.noise)

        self.lmax_prec_cg = self.lmax - 1000 if lmax_prec_cg is None \
                            else lmax_prec_cg

    def prepare(self, omap):
        # run optimal filtering
        ucls, tcls = futils.get_theory_dicts_white_noise(self.fwhm, self.noise,
                                                         grad=False, lmax=self.mlmax)
        b_ell = maps.gauss_beam(self.fwhm, np.arange(self.mlmax))

        #icov_pix = np.expand_dims(self.ivar.copy(), axis=0)
        icov_pix  = enmap.enmap(
                        [self.ivar,
                         maps.ivar(self.shape, self.wcs, self.noise * np.sqrt(2)),
                         maps.ivar(self.shape, self.wcs, self.noise * np.sqrt(2))]
                    )

        mask_bool = enmap.enmap(
                        np.concatenate([self.mask[np.newaxis, :]]*3, axis=0)
                    ).astype(bool)

        filt = optfilt.CGPixFilter(ucls, b_ell, icov_pix=icov_pix,
                                   mask_bool=mask_bool, lmax=self.lmax, swap_bm=True,
                                   lmax_prec_cg=self.lmax_prec_cg, mlmax=self.mlmax)

        # zero out input map at masked locations
        omap[~mask_bool] = 0.

        alm_dict = filt.filter(omap, niter=NITER,
                               niter_masked_cg=NITER_MASKED_CG, 
                               benchmark=False, verbose=True,
                               err_tol=ERR_TOL, compute_qe=COMPUTE_QE,
                               eval_every_niters=EVAL_EVERY_NITERS,
                               tcls=tcls)
        
        return alm_dict['ialm']


