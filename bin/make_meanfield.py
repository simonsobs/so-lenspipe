from __future__ import print_function
from orphics import maps,io,cosmology,mpi,stats
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,Channel
from mapsims import SO_Noise_Calculator_Public_20180822 as sonoise
from falafel import qe
from solenspipe import initialize_mask, initialize_norm, SOLensInterface,get_kappa_alm

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Demo lensing pipeline.')
parser.add_argument("polcomb", type=str,help='Polarization combination. Possibilities include mv (all), mvpol (all pol), TT, EE, TE, EB or TB.')
parser.add_argument("--nsims",     type=int,  default=100,help="nsims")
parser.add_argument("--nside",     type=int,  default=32,help="nside")
parser.add_argument("--smooth-deg",     type=float,  default=4.,help="Gaussian smoothing sigma for mask in degrees.")
parser.add_argument("--lmin",     type=int,  default=300,help="lmin")
parser.add_argument("--lmax",     type=int,  default=3000,help="lmax")
parser.add_argument("--freq",     type=int,  default=145,help="channel freq")
args = parser.parse_args()

nside = args.nside
smooth_deg = args.smooth_deg
ch = Channel('LA',args.freq)
lmin = args.lmin
lmax = args.lmax
polcomb = args.polcomb
config = io.config_from_yaml("input/config.yml")

comm,rank,my_tasks = mpi.distribute(args.nsims,verbose=True)


mask = initialize_mask(nside,smooth_deg)
solint = SOLensInterface(mask)
thloc = "data/" + config['theory_root']
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

# norm dict
Als = {}
with bench.show("norm"):
    ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv = initialize_norm(solint,ch,lmin,lmax)
Als['mv'] = al_mv
Als['mvpol'] = al_mv_pol
al_mv = Als[polcomb]

s = stats.Stats(comm)
for task in my_tasks:
    talm  = solint.get_kmap(ch,"T",(0,0,task+1),lmin,lmax,filtered=True)
    ealm  = solint.get_kmap(ch,"E",(0,0,task+1),lmin,lmax,filtered=True)
    balm  = solint.get_kmap(ch,"B",(0,0,task+1),lmin,lmax,filtered=True)
    rkalm = hp.almxfl(solint.get_mv_kappa(polcomb,talm,ealm,balm)[0],al_mv)
    s.add_to_stack("mf_alm",rkalm)
s.get_stacks()

if rank==0:
    mf_alm = s.stacks['mf_alm']
    hp.write_alm(config['data_path']+"meanfield_alm_%s.fits" % polcomb,mf_alm,overwrite=True)
    shape,wcs = enmap.band_geometry(np.deg2rad((-70,30)),res=np.deg2rad(0.5*8192/256/60.))
    omap = cs.alm2map(mf_alm, enmap.empty(shape,wcs))
    io.hplot(omap,config['data_path']+"meanfield_%s" % polcomb)
    


