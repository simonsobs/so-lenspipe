from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
from solenspipe import biastheory as nbias

from orphics import maps,io,cosmology,stats,mpi # msyriac/orphics ; pip install -e . --user
from pixell import enmap,lensing as plensing, enplot
from pixell import curvedsky as cs,utils as putils
import solenspipe
from solenspipe import SOLensInterface,cmblensplus_norm,convert_seeds,get_cmb_alm,get_kappa_alm,wfactor
from solenspipe import utility as simgen
from solenspipe.utility import get_mask,w_n,kspace_mask
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from soapack import interfaces as sints
from falafel import qe
from falafel import utils
import os
import glob
import traceback
import pytempura
import argparse

parser = argparse.ArgumentParser(description='New Reconstruction Code')
parser.add_argument("est1", type=str,help='Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("-der_est", type=str,help='Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.')

parser.add_argument("-N", "--nsims",     type=int,  default=1,help="Number of sims.")
parser.add_argument( "--healpix", action='store_true',help='Use healpix instead of CAR.')
parser.add_argument( "--lmax",     type=int,  default=3000,help="Maximum multipole for lensing.")
parser.add_argument( "--lmin",     type=int,  default=600,help="Minimum multipole for lensing.")
parser.add_argument('-qid','-qID',dest='qID',nargs='+',type=str,default=['s19_03','s19_04'],help="first map is the 90Ghz map, second map is the 150 Ghz map")
parser.add_argument("--output-dir", type=str,  default=None,help='Output directory.')
parser.add_argument("--mask", type=str,  default="/global/cscratch1/sd/jia_qu/maps/downgrade/nulltest_aggmask.fits",help='mask directory.')
parser.add_argument("--noise", type=str,  default="/global/cscratch1/sd/jia_qu/maps/2dsims/",help='noise directory.')
parser.add_argument( "--bh", action='store_true',help='Use pointsource hardening')
parser.add_argument( "--mh", action='store_true',help='Use mask hardening')

args = parser.parse_args()
path=args.output_dir

lmin = args.lmin; lmax = args.lmax


mask=get_mask("/home/r/rbond/jiaqu/scratch/DR6/downgrade/downgrade/act_mask_20220316_GAL060_rms_70.00_d2sk.fits")
# Geometry specified by the mask, DR6 analysis use CAR maps
print(f"mask shape,wcs: {mask.shape}, {mask.wcs}")
shape,wcs = mask.shape,mask.wcs
nside = None
print(f"shape,wcs: {shape}, {wcs}")
px = qe.pixelization(shape=shape,wcs=wcs,nside=nside)
res_arcmin = np.rad2deg(enmap.pixshape(shape, wcs)[0])*60.
mlmax=4000
input_path="/home/r/rbond/jiaqu/scratch/N1test/"

ucls,tcls = utils.get_theory_dicts(lmax=mlmax,grad=True)
filters=np.loadtxt(f'/gpfs/fs1/home/r/rbond/jiaqu/DR6lensing/DR6lensing/4split_null/output/v3calfinal60newph/stage_filter/filters.txt')


est_norm_list = ['TT','TE','EE','EB','TB','MV']
bh = args.bh

tcls['TT'] = filters[0][:mlmax+1]
tcls['TE'] = filters[-1][:mlmax+1]
tcls['EE'] = filters[1][:mlmax+1]
tcls['BB'] = filters[2][:mlmax+1]



Als = pytempura.get_norms(est_norm_list,ucls,tcls,lmin,lmax,k_ellmax=mlmax)
print(Als)
config = io.config_from_yaml("/home/r/rbond/jiaqu/falafel/input" + "/config.yml")
opath = config['data_path']
thloc = opath+ config['theory_root']
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
ells,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])




ntt=Als['TT'][0][2:]
nee=Als['EE'][0][2:]
neb=Als['EB'][0][2:]
nte=Als['TE'][0][2:]
ntb=Als['TB'][0][2:]
nbb=np.ones(len(ntb))
norms=np.array([[ntt],[nee],[neb],[nte],[ntb],[nbb]])

theory = cosmology.loadTheorySpectraFromCAMB(config['data_path']+"cosmo2017_10K_acc3",get_dimensionless=False)
cltt = theory.lCl('TT',np.arange(8000)) 
clte= theory.lCl('TE',np.arange(8000))
clee = theory.lCl('EE',np.arange(8000)) 
clbb = theory.lCl('BB',np.arange(8000)) 

nells=np.loadtxt('/home/r/rbond/jiaqu/scratch/N1test/noise_n1.txt')
noise={'TT':nells[0],'EE':nells[1],'BB':nells[2]}

LCORR_TT=0
Lmax_TT=3000
Lmin_out=2
Lmaxout=3000
Lstep=20
lens=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lenspotentialCls.dat",unpack=True)
cls=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lensedCls.dat",unpack=True)
ls = np.arange(Als['TT'][0].size)
clpp=lens[5,:][:8249]
tmp_output="/home/r/rbond/jiaqu/scratch/DR6/coaddMV/stage_norm/"
n1bins=np.arange(Lmin_out,Lmaxout,Lstep)


#N1=nbias.compute_n1mv(clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,Lmax_TT,LCORR_TT,tmp_output,Lstep,Lmin_out)


def analytic_n1(min,noise,lmax,Als,Lmin_out=2,Lmaxout=3000,Lstep=20):
    
    from solenspipe import biastheory as nbias
    opath = config['data_path']

    n1fname=opath+"analytic_n1.txt"
    #ls,nells,nells_P=splitnull.get_ACT_noisepower(qids[0],1,0,mlmax,mask,args.noise) #normalization coming from the 150Ghz map
    nells=noise['TT'][:lmax]
    nellsp=noise['EE'][:lmax]
    Lmax_TT=Lmaxout
    tmp_output="/home/r/rbond/jiaqu/scratch/DR6/coaddMV/stage_norm/"
    LCORR_TT=0
    lens=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lenspotentialCls.dat",unpack=True)
    cls=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lensedCls.dat",unpack=True)
    ls = np.arange(Als['TT'][0].size)
    #arrays with l starting at l=2"
    #clphiphi array starting at l=2
    clpp=lens[5,:][:8249]
    #cls is an array containing [cltt,clee,clbb,clte] used for the filters
    cltt=cls[1]       
    clee=cls[2]
    clbb=cls[3]
    clte=cls[4]
    bins=np.arange(Als['TT'][0].size)[2:]
    ntt=Als['TT'][0][2:]
    nee=Als['EE'][0][2:]
    neb=Als['EB'][0][2:]
    nte=Als['TE'][0][2:]
    ntb=Als['TB'][0][2:]
    nbb=np.ones(len(ntb))
    norms=np.array([[ntt],[nee],[neb],[nte],[ntb],[nbb]])
    #n1tt,n1ee,n1eb,n1te,n1tb=nbias.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,Lmax_TT,LCORR_TT,tmp_output,Lstep,Lmin_out)
    n1mv=nbias.compute_n1mv(clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,Lmax_TT,LCORR_TT,tmp_output,Lstep,Lmin_out)
    n1bins=np.arange(Lmin_out,Lmaxout,Lstep)

    return n1bins,n1mv 


#N1=analytic_n1(args.lmin,noise,args.lmax,Als,Lmin_out=2,Lmaxout=3000,Lstep=20)
#np.savetxt(f'{input_path}/n1mv_lmin{args.lmin}_lmax{args.lmax}.txt',N1)

#DERIVATIVES WRT cls
Lmin_out=2
Lmaxout=3000
Lstep=20
nells=noise['TT'][:lmax]
nellsp=noise['EE'][:lmax]
Lmax_TT=Lmaxout
tmp_output="/home/r/rbond/jiaqu/scratch/DR6/coaddMV/stage_norm/"
LCORR_TT=0
lens=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lenspotentialCls.dat",unpack=True)
cls=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lensedCls.dat",unpack=True)
ls = np.arange(Als['TT'][0].size)
#arrays with l starting at l=2"
#clphiphi array starting at l=2
clpp=lens[5,:][:8249]
#cls is an array containing [cltt,clee,clbb,clte] used for the filters
cltt=cls[1]       
clee=cls[2]
clbb=cls[3]
clte=cls[4]
bins=np.arange(Als['TT'][0].size)[2:]
ntt=Als['TT'][0][2:]
nee=Als['EE'][0][2:]
neb=Als['EB'][0][2:]
nte=Als['TE'][0][2:]
ntb=Als['TB'][0][2:]
nbb=np.ones(len(ntb))
norms=np.array([[ntt],[nee],[neb],[nte],[ntb],[nbb]])

bins=np.arange(20,3000,30) #ells where derivatives are taken
n1bins=np.arange(Lmin_out,Lmaxout,Lstep) #Lbins

n1=nbias.n1mvderivative_clcmb(args.der_est,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nellsp,lmin,Lmaxout,Lmax_TT,LCORR_TT,tmp_output,Lstep,Lmin_out)
n1=nbias.extend_matrix(3000,np.nan_to_num(n1))

np.savetxt(f'{input_path}/N1der_{args.der_est}_lmin{args.lmin}_lmax{args.lmax}.txt',n1)  