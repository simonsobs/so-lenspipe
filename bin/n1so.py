from __future__ import print_function
from orphics import stats,io,mpi,maps,cosmology
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
import solenspipe
from falafel import qe
import mapsims
from enlib import bench
import healpy as hp
from solenspipe import biastheory as nbias

"""
We will do a simple lensing reconstruction test.
"""

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
parser.add_argument("--healpix", action='store_true',help='Use healpix.')
parser.add_argument("--no-mask", action='store_true',help='No mask. Use with the isotropic flag.')
parser.add_argument("--debug", action='store_true',help='Debug plots.')
parser.add_argument("--flat-sky-norm", action='store_true',help='Use flat-sky norm.')
args = parser.parse_args()
car = "healpix_" if args.healpix else "car_"
mask="nomask_" if args.no_mask else "mask_"
noise="sonoise" if args.wnoise==None else "wnoise"

solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)
print(isostr)   
w2 = solint.wfactor(2)
w3 = solint.wfactor(3)
w4 = solint.wfactor(4)
s = stats.Stats(comm)

config = io.config_from_yaml("../input/config.yml")

#load noise nells

#length of ell determines maximum l used for N1 calculation
ls,nells,nells_P = solint.get_noise_power(channel,beam_deconv=True)
NOISE_LEVEL=nells[:3000]
polnoise=nells_P[:3000]

lmin=100
LMAXOUT=2992
LMAX_TT=2992
TMP_OUTPUT=config['data_path']
LCORR_TT=0
Lstep=20
Lmin_out=2

lens=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lenspotentialCls.dat",unpack=True)
cls=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lensedCls.dat",unpack=True)

#arrays with l starting at l=2"
#clphiphi array starting at l=2
clpp=lens[5,:][:8249]


#cls is an array containing [cltt,clee,clbb,clte] used for the filters
cltt=cls[1]       
clee=cls[2]
clbb=cls[3]
clte=cls[4]

#Input normalisation as an array of arrays of lensing potential phi n0s.
norms=np.loadtxt(config['data_path']+"norm_test_lmin_100_lmax_3000.txt")
bins=norms[2:,0]
ntt=norms[2:,1]
nee=norms[2:,2]
neb=norms[2:,3]
nte=norms[2:,4]
ntb=norms[2:,5]
nbb=np.ones(len(ntb))
norms=np.array([[ntt/bins**2],[nee/bins**2],[neb/bins**2],[nte/bins**2],[ntb/bins**2],[nbb]])

"""

"""
#N1 bias calculation
n1bins=np.arange(Lmin_out,LMAXOUT,Lstep)
n1=nbias.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
np.savetxt("/global/cscratch1/sd/jia_qu/cmblensfastdata/n1so.txt",n1)


