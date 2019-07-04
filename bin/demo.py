"""

this script will run simulations through a toy pipeline for lensing bandpower
recovery.

The broad plan is:
1. no foregrounds
2. only 150 GHz SO maps
3. sub-optimal filtering only in harmonic space
4. RDN0
5. MCN1
6. MCMF
7. use mapsims to obtain beamed lensed sky and noise realizations
8. treat sim 0 as data
9. use sim >=1 for bias subtraction
10. only show results for MV
11. theory errors from (Clkk + N0)

"""

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
from solenspipe import initialize_mask, initialize_norm, SOLensInterface

nside = 2048
smooth_deg = 4.
ch = Channel('LA',145)
lmin = 300
lmax = 3000
polcomb = sys.argv[1]



with bench.show("mask"):
    mask = initialize_mask(nside,smooth_deg)

fsky = mask.sum()/mask.size

with bench.show("init"):
    solint = SOLensInterface(mask)
# with bench.show("kmap T"):
#     alms  = solint.get_kmap(ch,"T",(0,0,0),lmin,lmax,filtered=False)
# with bench.show("kmap E"):
#     alms_E  = solint.get_kmap(ch,"E",(0,0,0),lmin,lmax,filtered=False)
# with bench.show("kmap B"):
#     alms_B  = solint.get_kmap(ch,"B",(0,0,0),lmin,lmax,filtered=False)

# acls = hp.alm2cl(alms)
# aclee = hp.alm2cl(alms_E)
# aclbb = hp.alm2cl(alms_B)
# cls = acls
# ells = np.arange(len(cls))

theory = cosmology.default_theory()
# ls,nells = solint.nsim.ell,solint.nsim.noise_ell_T[ch]
# ls,nells_P = solint.nsim.ell,solint.nsim.noise_ell_P[ch]
# cltt = theory.lCl('TT',ells) + maps.interp(ls,nells)(ells)
# clee = theory.lCl('EE',ells) + maps.interp(ls,nells_P)(ells)
# clbb = theory.lCl('BB',ells) + maps.interp(ls,nells_P)(ells)



# pl  = io.Plotter(xyscale='linlog',xlabel='l',ylabel='Dl',scalefn = lambda x: x**2./2./np.pi)
# pl.add(ells,cls/fsky)
# pl.add(ells,cltt,ls="--")
# pl.add(ells,aclee/fsky)
# pl.add(ells,clee,ls="--")
# pl.add(ells,aclbb/fsky)
# pl.add(ells,clbb,ls="--")
# pl.done("cls.png")


Als = {}
with bench.show("norm"):
    ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv = initialize_norm(ch,lmin,lmax)
Als['mv'] = al_mv
Als['mvpol'] = al_mv_pol

al_mv = Als[polcomb] # !!!!

nls = al_mv * ls**2./4.
tclkk = theory.gCl('kk',ls)
wfilt = tclkk/(tclkk+nls)/ls**2.
wfilt[ls<50] = 0
wfilt[ls>500] = 0
wfilt[~np.isfinite(wfilt)] = 0

talm  = solint.get_kmap(ch,"T",(0,0,0),lmin,lmax,filtered=True)
ealm  = solint.get_kmap(ch,"E",(0,0,0),lmin,lmax,filtered=True)
balm  = solint.get_kmap(ch,"B",(0,0,0),lmin,lmax,filtered=True)
with bench.show("recon"):
    rkalm = hp.almxfl(solint.get_mv_kappa(polcomb,talm,ealm,balm)[0],al_mv)
fkalm = hp.almxfl(rkalm,wfilt)
frmap = hp.alm2map(fkalm,nside=256)

ikalm = maps.change_alm_lmax(hp.map2alm(hp.alm2map(get_kappa_alm(0).astype(np.complex128),nside=solint.nside)*solint.mask),2*solint.nside)
fikalm = hp.almxfl(ikalm,wfilt)
fimap = hp.alm2map(fikalm,nside=256)
dmask = hp.ud_grade(mask,nside_out=256)
dmask[dmask<0] = 0

io.mollview(frmap*dmask,"wrmap.png",xsize=1600)
io.mollview(fimap*dmask,"wimap.png",xsize=1600)

w3 = np.mean(solint.mask**3)
w2 = np.mean(solint.mask**2)
xcls = hp.alm2cl(rkalm,ikalm)/w3
icls = hp.alm2cl(ikalm,ikalm)/w2
ells = np.arange(len(icls))
clkk = theory.gCl('kk',ells)
pl = io.Plotter(xlabel='l',ylabel='C')
pl.add(ells,xcls)
pl.add(ells,clkk,ls="--")
pl.add(ells,icls)
pl.hline(y=0)
pl._ax.set_ylim(-1e-8,1e-7)
pl.done("xcls_%s.png" % polcomb)

sys.exit()




with bench.show("init noise"):
    noise_sim = noise.SONoiseSimulator(nside=2048)#,apply_beam_correction=False)

with bench.show("noise sim"):
    noise_map = noise_sim.simulate(ch)[0]

noise_map[noise_map<-1e24] = 0
io.mollview(mask*noise_map,"noisemap.png")

