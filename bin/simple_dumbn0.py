from __future__ import print_function
from orphics import stats,io,mpi,maps,cosmology
from pixell import enmap
import numpy as np
import os,sys
import solenspipe
from falafel import qe
import mapsims
from enlib import bench
import healpy as hp
from solenspipe import bias

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("label", type=str,help='Label.')
parser.add_argument("polcomb", type=str,help='polcomb.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="Number of sims.")
parser.add_argument("--sindex",     type=int,  default=0,help="Declination band.")
parser.add_argument("--lmin",     type=int,  default=100,help="Minimum multipole.")
parser.add_argument("--lmax",     type=int,  default=3000,help="Minimum multipole.")
parser.add_argument("--isotropic", action='store_true',help='Isotropic sims.')
parser.add_argument("--no-atmosphere", action='store_true',help='Disable atmospheric noise.')
parser.add_argument("--use-cached-norm", action='store_true',help='Use  cached norm.')
parser.add_argument("--wnoise",     type=float,  default=None,help="Override white noise.")
parser.add_argument("--beam",     type=float,  default=None,help="Override beam.")
parser.add_argument("--disable-noise", action='store_true',help='Disable noise.')
parser.add_argument("--zero-sim", action='store_true',help='Just make a sim of zeros. Useful for benchmarking.')
parser.add_argument("--healpix", action='store_true',help='Use healpix.')
parser.add_argument("--no-mask", action='store_true',help='No mask. Use with the isotropic flag.')
parser.add_argument("--debug", action='store_true',help='Debug plots.')
parser.add_argument("--flat-sky-norm", action='store_true',help='Use flat-sky norm.')
args = parser.parse_args()
solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)
      
car = "healpix_" if args.healpix else "car_"

w4 = solint.wfactor(4)
get_kmap = lambda seed: solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
power = lambda x,y: hp.alm2cl(x,y)
qfunc = solint.qfunc

nmax = len(ils)
ls,nells,nells_P = solint.get_noise_power(channel,beam_deconv=True)
ells=np.arange(0,solint.mlmax)
config = io.config_from_yaml("../input/config.yml")
thloc = "../data/" + config['theory_root']
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)


ells,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])
class T:
    def __init__(self):
        self.lCl = lambda p,x: maps.interp(ells,gt)(x)
theory_cross = T()

get_sim_power = lambda seed: solint.get_sim_power(channel,seed,lmin,lmax)
ls,rd=solenspipe.diagonal_RDN0(get_sim_power,nells,nells_P,nells_P,theory,theory_cross,lmin,lmax,nsims)
