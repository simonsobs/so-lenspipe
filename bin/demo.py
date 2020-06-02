"""

This script demos lensing reconstruction from mbs sims.

Many caveats:
1. no foregrounds
2. only single frequency SO maps are used -- noise is drawn on the fly from simonsobs/mapsims
3. the lensed CMB sky is currently loaded directly from disk rather than through mapsims
4. sub-optimal filtering only in harmonic space
5. no mean-field subtraction, so plotting maps filtered to L>50
6. theory normalization from flat-sky symlens code

Dependencies:
    Main: mapsims, falafel, symlens, pixell, healpy, orphics
    Benchmarking: enlib


"""


from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,SOChannel
from falafel import qe
from solenspipe import initialize_mask, initialize_norm, SOLensInterface,get_kappa_alm
import solenspipe as s
import argparse

# Parse command line
parser = argparse.ArgumentParser(description='Demo lensing pipeline.')
parser.add_argument("polcomb", type=str,help='Polarization combination. Possibilities include mv (all), mvpol (all pol), TT, EE, TE, EB or TB.')
parser.add_argument("--nside",     type=int,  default=2048,help="nside")
parser.add_argument("--smooth-deg",     type=float,  default=4.,help="Gaussian smoothing sigma for mask in degrees.")
parser.add_argument("--lmin",     type=int,  default=100,help="lmin")
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

mask = initialize_mask(nside,smooth_deg)
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





# Wiener filter
nls = al_mv * ls**2./4.
tclkk = theory.gCl('kk',ls)
wfilt = tclkk/(tclkk+nls)/ls**2.
wfilt[ls<50] = 0
wfilt[ls>500] = 0
wfilt[~np.isfinite(wfilt)] = 0







# Filtered alms
seed = (0,0,0)
t_alm,e_alm,b_alm = solint.get_kmap(ch,seed,lmin,lmax,filtered=True)

# Reconstruction
with bench.show("recon"):
    rkalm = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ls,Als[polcomb]))
hp.write_map(config['data_path']+"mbs_sim_v0.1.0_mv_lensing_map.fits",hp.alm2map(rkalm,nside),overwrite=True)
hp.write_map(config['data_path']+"mbs_sim_v0.1.0_mv_lensing_mask.fits",mask,overwrite=True)

# Filtered reconstruction
fkalm = hp.almxfl(rkalm,wfilt)
frmap = hp.alm2map(fkalm,nside=256)
rmap=hp.alm2map(rkalm,nside=256)
# Input kappa
# TODO: Does this really need to be masked?
ikalm = maps.change_alm_lmax(hp.map2alm(hp.alm2map(get_kappa_alm(0).astype(np.complex128),nside=solint.nside)*solint.mask),2*solint.nside)


# Verify input x cross
w4=np.mean(solint.mask**4)
w3 = np.mean(solint.mask**3) # Mask factors
w2 = np.mean(solint.mask**2)
xcls = hp.alm2cl(rkalm,ikalm)/w3
icls = hp.alm2cl(ikalm,ikalm)/w2
rcls=hp.alm2cl(rkalm,rkalm)/w4 #reconstruction auto
ells = np.arange(len(icls))
clkk = theory.gCl('kk',ells)
pl = io.Plotter(xyscale='loglog',xlabel='$L$',ylabel='$C_L$')
pl.add(ells,clkk,ls="-",lw=3,label='theory input')
pl.add(ells,xcls,alpha=0.4,label='input x recon')
pl.add(ells,icls,alpha=0.4,label='input x input')
clkk=clkk[:2989:20]
pl.add(ells,rcls,alpha=0.4,label='rec x rec')
pl.add(ls,nls,ls="--",label='theory noise per mode')
pl._ax.set_xlim(20,4000)
pl.done(config['data_path']+"xcls_%s.png" % polcomb)


# Filtered input
fikalm = hp.almxfl(ikalm,wfilt)
fimap = hp.alm2map(fikalm,nside=256)



# Resampled mask
dmask = hp.ud_grade(mask,nside_out=256)
dmask[dmask<0] = 0


# Mollview plots
io.mollview(frmap*dmask,config['data_path']+"wrmap1.png",xsize=1600,lim=8e-6)
io.mollview(fimap*dmask,config['data_path']+"wimap1.png",xsize=1600,lim=8e-6)

# CAR plots
shape,wcs = enmap.band_geometry(np.deg2rad((-70,30)),res=np.deg2rad(0.5*8192/512/60.))
omask = reproject.enmap_from_healpix(dmask, shape, wcs,rot=None)
omap = cs.alm2map(fkalm, enmap.empty(shape,wcs))
io.hplot(omap*omask,config['data_path']+"cwrmap1",grid=True,ticks=20,color='gray')
omap = cs.alm2map(fikalm, enmap.empty(shape,wcs))
io.hplot(omap*omask,config['data_path']+"cwimap1",grid=True,ticks=20,color='gray')



