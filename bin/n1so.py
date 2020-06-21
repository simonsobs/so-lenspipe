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
parser.add_argument("--zero-sim", action='store_true',help='Just make a sim of zeros. Useful for benchmarking.')
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

bins=np.array([0.000e+00, 2.000e+00, 1.200e+01, 2.200e+01, 3.200e+01, 4.200e+01,
       5.200e+01, 6.200e+01, 7.200e+01, 8.200e+01, 9.200e+01, 1.020e+02,
       1.320e+02, 1.620e+02, 1.920e+02, 2.220e+02, 2.520e+02, 2.820e+02,
       3.120e+02, 3.420e+02, 3.720e+02, 4.020e+02, 4.320e+02, 4.620e+02,
       4.920e+02, 5.220e+02, 5.520e+02, 6.520e+02, 7.520e+02, 8.520e+02,
       9.520e+02, 1.052e+03, 1.152e+03, 1.252e+03, 1.352e+03, 1.452e+03,
       1.752e+03, 2.052e+03, 2.352e+03, 2.652e+03, 2.952e+03])
       
n1bins=np.arange(Lmin_out,LMAXOUT,Lstep)
n1=nbias.compute_n1mv(clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
n1mv_dclkk=nbias.n1mv_dclkk(clpp,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mv_dcltt=nbias.n1mv_dcltt(cltt,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)transpose()
n1mv_dclee=nbias.n1mv_dclee(clee,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mv_dclbb=nbias.n1mv_dclbb(clbb,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mv_dclte=nbias.n1mv_dclte(clte,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mv_dcltt=nbias.n0mvderivative_cltt(cltt,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mv_dclee=nbias.n0mvderivative_clee(clee,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mv_dclbb=nbias.n0mvderivative_clbb(clbb,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mv_dclte=nbias.n0mvderivative_clte(clte,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()

def extend_matrix(sizeL,_matrix):
    #unpacked=true
    #sizeL 3000 size of total unbinned clkk used
    #return sizeLxsizeL interpolated matrix
    matrix=_matrix
    n1bins=matrix[0][1:]
    derbins=matrix.transpose()[0][1:]
    bins=np.arange(sizeL)
    a=[]
    for i in range(1,len(matrix)):
        narray=maps.interp(n1bins,matrix[i][1:])(bins)
        a.append(narray)
    y=np.array(a).transpose()
    b=[]
    #the derivatives L' are calculated about the derbins values, this is the binned version, so set the other l' to 0
    for i in range(len(y)):
        ext=np.zeros(sizeL)
        ext[derbins.astype(int)]=y[i]
        b.append(ext)
    b=np.array(b)
    a=b.transpose()    
    return a
    


n0mvdcltte=extend_matrix(3000,n0mvdcltt)
n0mvdcltee=extend_matrix(3000,n0mvdclte)
n0mvdcleee=extend_matrix(3000,n0mvdclee)
n0mvdclbbe=extend_matrix(3000,n0mvdclbb)
n1mvdclkke=extend_matrix(3000,n1mvdclkk)

n1mvdcltte=extend_matrix(3000,n1mvdcltt)
n1mvdcleee=extend_matrix(3000,n1mvdclee)
n1mvdclbbe=extend_matrix(3000,n1mvdclbb)
n1mvdcltee=extend_matrix(3000,n1mvdclte)

np.savetxt(f"{solenspipe.opath}/n1mvdclkk1.txt",n1mvdclkke)
np.savetxt(f"{solenspipe.opath}/n0mvdcltt1.txt",n0mvdcltte)
np.savetxt(f"{solenspipe.opath}/n0mvdclte1.txt",n0mvdcltee)
np.savetxt(f"{solenspipe.opath}/n0mvdclee1.txt",n0mvdcleee)
np.savetxt(f"{solenspipe.opath}/n0mvdclbb1.txt",n0mvdclbbe)
np.savetxt(f"{solenspipe.opath}/n1mvdcltte1.txt",n1mvdcltte)
np.savetxt(f"{solenspipe.opath}/n1mvdcleee1.txt",n1mvdcleee)
np.savetxt(f"{solenspipe.opath}/n1mvdclbbe1.txt",n1mvdclbbe)
np.savetxt(f"{solenspipe.opath}/n1mvdcltee1.txt",n1mvdcltee)


