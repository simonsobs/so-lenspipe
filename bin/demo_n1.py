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

clkk=np.loadtxt(config['data_path']+"ckk.txt")
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

norms=np.loadtxt(config['data_path']+"norm_lmin_300_lmax_3000.txt")
bins=norms[2:,0]
ntt=norms[2:,1]
nee=norms[2:,2]
neb=norms[2:,3]
nte=norms[2:,4]
ntb=norms[2:,5]
nbb=np.ones(len(ntb))
norms=np.array([[ntt/bins**2],[nee/bins**2],[neb/bins**2],[nte/bins**2],[ntb/bins**2],[nbb]])

"""
Input normalisation as an array of arrays of deflection n0s.
"""
#N1 bias calculation
n1tt,n1ee,n1eb,n1te,n1tb=s.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)

n0tt,n0ee,n0eb,n0te,n0tb=s.compute_n0_py(clpp,cls,cltt,clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)

#derivative wrt clphi
#n1=s.n1_derivatives('TB','TB',clpp,norms,cls,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)


#np.savetxt('../data/norms.txt',c)
#cls and clpp must have same dimensions.

#s.compute_n0_py(clpp,cls,cltt,clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
indices = ['TT', 'EE', 'EB', 'TE', 'TB', 'BB']
n0 = np.loadtxt(os.path.join(TMP_OUTPUT, 'N0_analytical.dat')).T
bins = n0[0]
phiphi = n0[1]
n0_mat = np.reshape(n0[2:], (len(indices), len(indices), len(bins)))

MV_n0, weights = s.minimum_variance_n0(n0_mat, indices, checkit=False)


#s.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
n1 = np.loadtxt(os.path.join(TMP_OUTPUT, 'N1_All_analytical.dat')).T
indices = ['TT', 'EE', 'EB', 'TE', 'TB', 'BB']
bins = n1[0]
n1_mat = np.reshape(n1[1:], (len(indices), len(indices), len(bins)))
MV_n1 = s.minimum_variance_n1(bins, n1_mat, weights, indices, bin_function=None)

