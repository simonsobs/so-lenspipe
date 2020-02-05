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
nells=solint.nsim.noise_ell_T[ch.telescope][int(ch.band)][0:3000]
nells_P =solint.nsim.noise_ell_P[ch.telescope][int(ch.band)][0:3000]
NOISE_LEVEL=nells
polnoise=nells_P


FWHM=1.4
LMIN=2
LMAXOUT=2992
LMAX=2992
LMAX_TT=2992
TMP_OUTPUT=config['data_path']
LCORR_TT=0
n0=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/N0_analytical.txt",unpack=True) #contains bins n0tt,n0ee,n0eb,n0te,n0tb 
clkk=np.loadtxt('/global/homes/j/jia_qu/so-lenspipe/data/ckk.txt')
lens=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/cosmo2017_10K_acc3_lenspotentialCls.dat",unpack=True)
cls=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/cosmo2017_10K_acc3_lensedCls.dat",unpack=True)

#arrays with l starting at l=2"
clpp=lens[5,:][:8249]
#cls is an array containing [cltt,clee,clbb,clte] used for the filters
cltt=cls[1]       
clee=cls[2]
clbb=cls[3]
clte=cls[4]

"""
Currently fortran code takes in the normalization as a text file, found here at "/global/homes/j/jia_qu/so-lenspipe/data/N0_analytical.txt", still need to
fix this to take in the N0 as an array instead.
"""
s.compute_n1_py(clpp,cls,cltt,clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)

"""returns arrays n1tt,n1ee,n1bb,n1te"""

#np.savetxt('../data/norms.txt',c)
#cls and clpp must have same dimensions.



