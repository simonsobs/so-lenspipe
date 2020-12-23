from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,SOChannel
from falafel import qe
from solenspipe import initialize_mask, SOLensInterface,get_kappa_alm
import solenspipe as s
import argparse
from solenspipe import biastheory as nbias

config = io.config_from_yaml("../input/config.yml")

#load noise nells
beam=7.0
wnoise=30
#length of ell determines maximum l used for N1 calculation
ell=np.arange(0,3000)
bfact = maps.gauss_beam(beam,ell)**2.
nells = (wnoise*np.pi/180/60)**2./(np.ones(len(ell))*bfact)
nells_P=2*nells
NOISE_LEVEL=nells[:3000]
polnoise=nells_P[:3000]


lmin=100
LMAXOUT=2992
LMAX_TT=2992
TMP_OUTPUT=config['data_path']
LCORR_TT=0
Lstep=30
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

"""
#N1 bias calculation
n1bins=np.arange(Lmin_out,LMAXOUT,Lstep)
n1tt,n1ee,n1eb,n1te,n1tb=nbias.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)



#N0 and N1 derivatives wrt clee
#bins=np.array([1200,2000])
#n0eeder=nbias.n0derivative_clee(clee,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
#n1bbder=nbias.n1derivative_clee(clee,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)

#N1 mixed bias calculation
#n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb=nbias.compute_n1mix(clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)

#N0 mixed  lensing field bias calculation
#nbias.compute_n0mix_py(clpp,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out,Lstep)

#N0 lensing field Fortran calculation
#n0tt,n0ee,n0eb,n0te,n0tb=nbias.compute_n0_py(clpp,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lmin_out, Lstep)



