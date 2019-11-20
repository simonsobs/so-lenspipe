
from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,SOChannel
from mapsims import SO_Noise_Calculator_Public_20180822 as sonoise
from falafel import qe
from solenspipe import initialize_mask, initialize_norm, SOLensInterface,get_kappa_alm
import solenspipe as s
import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Demo lensing pipeline.')
parser.add_argument("polcomb", type=str,help='Polarization combination. Possibilities include mv (all), mvpol (all pol), TT, EE, TE, EB or TB.')
parser.add_argument("--nside",     type=int,  default=2048,help="nside")
parser.add_argument("--smooth-deg",     type=float,  default=4.,help="Gaussian smoothing sigma for mask in degrees.")
parser.add_argument("--lmin",     type=int,  default=300,help="lmin")
parser.add_argument("--lmax",     type=int,  default=3000,help="lmax")
parser.add_argument("--freq",     type=int,  default=145,help="channel freq")
args = parser.parse_args()

nside = args.nside
smooth_deg = args.smooth_deg
ch = SOChannel('LA',args.freq)
lmin = args.lmin
lmax = args.lmax
polcomb = args.polcomb
config = io.config_from_yaml("../input/config.yml")
mask = initialize_mask(nside,smooth_deg) #solenspipe code that creates the mask
solint = SOLensInterface(mask)
thloc = "../data/" + config['theory_root']
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)


# norm dict
Als = {}
with bench.show("norm"):
    ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv = initialize_norm(solint,ch,lmin,lmax)
Als['mv'] = al_mv
Als['mvpol'] = al_mv_pol
al_mv = Als[polcomb]
l=ls.astype(int)
a=[l,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB']]
#y=np.transpose(a)
#np.savetxt('testall.txt',y)
#why use filter?
#The observed sky maps are cut by a galactic mask and have noise.
# Wiener filter
nls = al_mv * ls**2./4.  #theory noise per mode
NOISE_LEVEL=nls
"""
tclkk = theory.gCl('kk',ls)
wfilt = tclkk/(tclkk+nls)/ls**2.
wfilt[ls<50] = 0
wfilt[ls>500] = 0
wfilt[~np.isfinite(wfilt)] = 0

# Filtered alms
talm  = solint.get_kmap(ch,"T",(0,0,0),lmin,lmax,filtered=True)
ealm  = solint.get_kmap(ch,"E",(0,0,0),lmin,lmax,filtered=True)
balm  = solint.get_kmap(ch,"B",(0,0,0),lmin,lmax,filtered=True)

# Reconstruction
with bench.show("recon"):
    rkalm = hp.almxfl(solint.get_mv_kappa(polcomb,talm,ealm,balm)[0],al_mv)
hp.write_map(config['data_path']+"mbs_sim_v0.1.0_mv_lensing_map.fits",hp.alm2map(rkalm,nside),overwrite=True)
hp.write_map(config['data_path']+"mbs_sim_v0.1.0_mv_lensing_mask.fits",mask,overwrite=True)

# Filtered reconstruction
fkalm = hp.almxfl(rkalm,wfilt)
frmap = hp.alm2map(fkalm,nside=256)

# Input kappa
ikalm = maps.change_alm_lmax(hp.map2alm(hp.alm2map(get_kappa_alm(0).astype(np.complex128),nside=solint.nside)*solint.mask),2*solint.nside)
"""

phi='../data/cosmo2017_10K_acc3_lenspotentialCls.dat'
lensed='../data/cosmo2017_10K_acc3_lensedCls.dat'
FWHM=1.5
LMIN=2
LMAXOUT=2990
LMAX=2990
LMAX_TT=2990
TMP_OUTPUT='./output'
LCORR_TT=0

bins, phiphi, n0_mat, indices = s.compute_n0_py(phi,lensed,FWHM,NOISE_LEVEL,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
bins, n1_mat, indices = s.compute_n1_py(phi,lensed,FWHM,NOISE_LEVEL,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
tphi = lambda l: (l + 0.5)**4 / (2. * np.pi)
N1_array=n1_mat
indices = ['TT','EE','EB','TE','TB']
dict_int={'TT':0,'EE':1,'EB':2,'TE':3,'TB':4}
# Verify input x cross
"""
w3 = np.mean(solint.mask**3) # Mask factors
w2 = np.mean(solint.mask**2)
xcls = hp.alm2cl(rkalm,ikalm)/w3
icls = hp.alm2cl(ikalm,ikalm)/w2
ells = np.arange(len(icls))
clkk = theory.gCl('kk',ells)
"""
pl = io.Plotter(xyscale='loglog',xlabel='$L$',ylabel='$C_L$')
"""
pl.add(ells,clkk,ls="-",lw=3,label='theory input')
pl.add(ells,xcls,alpha=0.4,label='input x recon')
pl.add(ells,icls,alpha=0.4,label='input x input')
pl.add(ls,nls,ls="--",label='theory noise per mode')
"""
#Plot N1 noise
i=dict_int[polcomb]
pl.add(bins,N1_array[i][i][:] * tphi(bins),ls='dashdot',lw=2,label=indices[i]+indices[i],alpha=0.8)
#Plot all the N1
"""
for i in range(len(N1_array)-1):
	pl.add(bins,N1_array[i][i][:] * tphi(bins),ls='--',lw=2,label=indices[i]+indices[i],alpha=0.5)
"""
pl._ax.set_xlim(20,4000)
pl.done(config['data_path']+"xcls_%s.png" % polcomb)

"""
# Filtered input
fikalm = hp.almxfl(ikalm,wfilt)
fimap = hp.alm2map(fikalm,nside=256)

# Resampled mask
dmask = hp.ud_grade(mask,nside_out=256)
dmask[dmask<0] = 0


# Mollview plots
io.mollview(frmap*dmask,config['data_path']+"wrmap.png",xsize=1600,lim=8e-6)
io.mollview(fimap*dmask,config['data_path']+"wimap.png",xsize=1600,lim=8e-6)

# CAR plots
shape,wcs = enmap.band_geometry(np.deg2rad((-70,30)),res=np.deg2rad(0.5*8192/512/60.))
omask = reproject.enmap_from_healpix(dmask, shape, wcs,rot=None)
omap = cs.alm2map(fkalm, enmap.empty(shape,wcs))
io.hplot(omap*omask,config['data_path']+"cwrmap",grid=True,ticks=20,color='gray')
omap = cs.alm2map(fikalm, enmap.empty(shape,wcs))
io.hplot(omap*omask,config['data_path']+"cwimap",grid=True,ticks=20,color='gray')
"""
