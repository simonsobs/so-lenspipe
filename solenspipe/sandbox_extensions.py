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
from . import bias, solenspipe
from falafel.utils import get_cmb_alm, get_kappa_alm, \
    get_theory_dicts, get_theory_dicts_white_noise, \
    change_alm_lmax
from falafel import utils as futils

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

class LensingSandboxILC(solenspipe.LensingSandbox):
    def __init__(self, *args, start_index=0, nilc_sims_per_set=250,
                 nilc_sims_path="/data5/depot/needlets/proto/",
                 downgrade_res = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.nilc_sims_per_set = nilc_sims_per_set
        self.nilc_sims_path = nilc_sims_path
        self.start_index = start_index
        self.downgrade_res = downgrade_res
        
    def _apply_mask(self,imap,mask,eps=1e-8):
        if len(imap.shape) == 3:
            return enmap.enmap(np.array([
                self._apply_mask(imap[0],mask,eps),
                self._apply_mask(imap[1],mask,eps),
                self._apply_mask(imap[2],mask,eps)
            ]), imap.wcs)
        
        # should now be 2d
        omap = imap * mask
        # handle edge cases
        omap[mask < eps] = 0.
        omap[mask >= (1-eps)] = imap[mask >= (1-eps)]
        return omap

    def _downgrade_res(self, imap):
        if self.downgrade_res is not None:
            omap = enmap.downgrade(imap, int(self.downgrade_res / (21600 / imap.shape[1])),
                                   op=np.mean)
        else:
            omap = imap
        return omap

    def get_observed_map(self,index,iset=0):
        sim_idx = index % self.nilc_sims_per_set + iset * self.nilc_sims_per_set
        imap = enmap.read_map(self.nilc_sims_path + \
                              f"sim_{sim_idx}_lensmode_coadd_covsmooth_64.fits")
        # (optionally) downgrade before applying mask
        return self._apply_mask(self._downgrade_res(imap), self.mask)

    def kmap(self,stuple):
        icov,ip,i = stuple
        # if i > self.nilc_sims_per_set: raise ValueError
        index, iset = i % self.nilc_sims_per_set, ip
        dmap = self.get_observed_map(index,iset)
        X = self.prepare(dmap)
        return X