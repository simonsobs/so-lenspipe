import basic
from cmblensplus import curvedsky as cs
import healpy as hp
from orphics import maps,io,cosmology 
import numpy as np
# define parameters


Tcmb = 2.726e6    # CMB temperature
lmax = 2048       # maximum multipole of output normalization
rlmin = 500
rlmax = 2048      # reconstruction multipole range
beam=1.5
wnoise=6.
ell=np.arange(lmax+1)
def gauss_beam(ell,fwhm):
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))
bfact = maps.gauss_beam(beam,ell)**2.
nells = (wnoise*np.pi/180/60)**2./(Tcmb**2*np.ones(len(ell))*bfact)
nellsp=2*nells
# load unlensed and lensed Cls

lcl  = basic.aps.read_cambcls('../data/cosmo2017_10K_acc3_lensedCls.dat',2,lmax,4,bb=True)/Tcmb**2
QDO = [True,True,True,True,True,False]

bins=np.arange(0,lmax+1)
noise=np.array([nells,nellsp,nellsp,nellsp])
ocl = lcl+noise


#normalization
Ag, Ac, Wg, Wc = cs.norm_lens.qall(QDO,lmax,rlmin,rlmax,lcl,ocl)

fac=ell*(ell+1)/2
#compute convergence Als
normTT=Ag[0]*fac**2
normEE=Ag[2]*fac**2
normTE=Ag[1]*fac**2
normTB=Ag[3]*fac**2
normEB=Ag[4]*fac**2
normMV=Ag[5]*fac**2