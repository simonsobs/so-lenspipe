from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,lensing as plensing
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,Channel,SOStandalonePrecomputedCMB
from mapsims import SO_Noise_Calculator_Public_20180822 as sonoise
from falafel import qe


config = io.config_from_yaml("input/config.yml")
opath = config['data_path']


def get_cmb_alm(i,iset,path=config['signal_path']):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullskyLensedUnabberatedCMB_alm_set%s_%s.fits" % (sstr,istr)
    return hp.read_alm(fname,hdu=(1,2,3))

def get_kappa_alm(i,path=config['signal_path']):
    istr = str(i).zfill(5)
    fname = path + "fullskyPhi_alm_%s.fits" % istr
    return plensing.phi_to_kappa(hp.read_alm(fname))


class SOLensInterface(object):
    def __init__(self,mask):
        self.mask = mask
        self.nside = hp.npix2nside(mask.size)
        self.nsim = noise.SONoiseSimulator(nside=self.nside,apply_beam_correction=True)    
        thloc = "data/" + config['theory_root']
        theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
        self.cltt = lambda x: theory.lCl('TT',x) 
        self.clee = lambda x: theory.lCl('EE',x) 
        self.clbb = lambda x: theory.lCl('BB',x) 
        self.cache = {}
        self.theory = theory
        

    def prepare_map(self,channel,seed,lmin,lmax):
        """
        Generates a beam-deconvolved simulation.
        Filters it and caches it.
        """

        icov,cmb_set,i = seed
        assert icov==0, "Covariance from sims not yet supported."
        nsims = 2000
        ndiv = 4
        nstep = nsims//ndiv
        if cmb_set==0 or cmb_set==1:
            s_i = i + cmb_set*nstep
            s_set = 0
            noise_seed = (icov,cmb_set,i)+(2,)
        elif cmb_set==2 or cmb_set==3:
            s_i = i + nstep*2
            s_set = cmb_set - 2
            noise_seed = (icov,cmb_set,i)+(2,)

        cmb_alm = get_cmb_alm(s_i,s_set).astype(np.complex128)
        cmb_map = hp.alm2map(cmb_alm,nside=self.nside)
        noise_map = self.nsim.simulate(channel,seed=noise_seed+(channel.band,))
        noise_map[noise_map<-1e24] = 0
        imap = (cmb_map + noise_map)*self.mask
        oalms = hp.map2alm(imap)


        nells_T = maps.interp(self.nsim.ell,self.nsim.noise_ell_T[channel])
        nells_P = maps.interp(self.nsim.ell,self.nsim.noise_ell_P[channel])
        filt_t = lambda x: 1./(self.cltt(x) + nells_T(x))
        filt_e = lambda x: 1./(self.clee(x) + nells_P(x))
        filt_b = lambda x: 1./(self.clbb(x) + nells_P(x))

        almt = qe.filter_alms(oalms[0],filt_t,lmin=lmin,lmax=lmax)
        alme = qe.filter_alms(oalms[1],filt_e,lmin=lmin,lmax=lmax)
        almb = qe.filter_alms(oalms[2],filt_b,lmin=lmin,lmax=lmax)

        self.cache = {}
        self.cache[seed] = (almt,alme,almb,oalms[0],oalms[1],oalms[2])


    def get_kmap(self,channel,X,seed,lmin,lmax,filtered=True):
        xs = {'T':0,'E':1,'B':2}
        assert X in xs.keys()
        if not(seed in self.cache.keys()): self.prepare_map(channel,seed,lmin,lmax)
        return self.cache[seed][xs[X]] if filtered else self.cache[seed][xs[X]+3]

    def get_mv_kappa(self,polcomb,talm,ealm,balm):
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(0.5*8192/self.nside/60.))
        mlmax = 2*self.nside
        res = qe.qe_all(shape,wcs,lambda x,y: self.theory.lCl(x,y),mlmax,talm,ealm,balm,estimators=[polcomb])
        return res[polcomb]
        

def initialize_mask(nside,smooth_deg):
    omaskfname = "lensing_mask_nside_%d_apodized_%.1f.fits" % (nside,smooth_deg)
    try:
        return hp.read_map(opath + omaskfname)
    except:
        mask = hp.ud_grade(hp.read_map(opath + config['mask_name']),nside)
        mask[mask<0] = 0
        mask = hp.smoothing(mask,np.deg2rad(smooth_deg))
        mask[mask<0] = 0
        hp.write_map(opath + omaskfname,mask,overwrite=True)
        return mask


def initialize_norm(solint,ch,lmin,lmax):
    onormfname = opath+"norm_lmin_%d_lmax_%d.txt" % (lmin,lmax)
    try:
        return np.loadtxt(onormfname,unpack=True)
    except:
        thloc = "data/" + config['theory_root']
        theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
        ells = np.arange(lmax+100)
        uctt = theory.lCl('TT',ells)
        ucee = theory.lCl('EE',ells)
        ucte = theory.lCl('TE',ells)
        ucbb = theory.lCl('BB',ells)
        ls,nells = solint.nsim.ell,solint.nsim.noise_ell_T[ch]
        ls,nells_P = solint.nsim.ell,solint.nsim.noise_ell_P[ch]
        tctt = uctt + maps.interp(ls,nells)(ells)
        tcee = ucee + maps.interp(ls,nells_P)(ells)
        tcte = ucte 
        tcbb = ucbb + maps.interp(ls,nells_P)(ells)
        ls,Als,al_mv_pol,al_mv,Al_te_hdv = qe.symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        io.save_cols(onormfname,(ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv))
        return ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv


