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
    def __init__(self, *args, start_index=0, nilc_sims_per_set=400,
                 nilc_sims_path="/data5/depot/needlets/needproto/",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.nilc_sims_per_set = nilc_sims_per_set
        self.nilc_sims_path = nilc_sims_path
        self.start_index = start_index
        
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

    def get_observed_map(self,index,iset=0):
        imap = enmap.read_map(self.nilc_sims_path + \
                              f"sim_{index}_iset_{iset}_simple2_coadd_covsmooth_64.fits")
        # assume we want the mask to match the resolution of the map
        return self._apply_mask(imap, self.mask)

    def kmap(self,stuple):
        icov,ip,i = stuple
        # if i > self.nilc_sims_per_set: raise ValueError
        if ip < 2:
            iset = 0
            index = self.nilc_sims_per_set * (ip + 1) + i
        else:
            iset = ip - 2
            index = i
        dmap = self.get_observed_map(index,iset)
        X = self.prepare(dmap)
        return X
    
    def kmap_mf(self,stuple):
        # build two sets specifically for mcmf
        icov,ip,i = stuple
        # if i > self.nilc_sims_per_set: raise ValueError
        iset = 0
        index = self.nilc_sims_per_set * ip + i
        dmap = self.get_observed_map(index,iset)
        X = self.prepare(dmap)
        return X
    
    def prepare(self,omap):
        # TT only for now
        alm = cs.map2alm(omap,lmax=self.mlmax)
        with np.errstate(divide='ignore', invalid='ignore'):
            alm = cs.almxfl(alm,lambda x: 1./maps.gauss_beam(self.fwhm,x))
        ftalm,fealm,fbalm = futils.isotropic_filter([alm, alm*0., alm*0.],
                                                    self.tcls,self.lmin,self.lmax)
        return [ftalm,fealm,fbalm]
    
    # use above function for kmap for appropriate sims
    def get_mcmf_ilc(self,est,nsims,comm):
        return bias.mcmf_pair(0,self.qfuncs[est],self.kmap_mf,comm,
                              nsims,start=self.start_index)