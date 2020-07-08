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
parser.add_argument("-N", "--nsims",     type=int,  default=100,help="Number of sims.")
parser.add_argument("--sindex",     type=int,  default=0,help="Declination band.")
parser.add_argument("--lmin",     type=int,  default=500,help="Minimum multipole.")
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

solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)


config = io.config_from_yaml("../input/config.yml")

#load noise nells

#length of ell determines maximum l used for N1 calculation
#load the noises
nells=np.loadtxt("/global/homes/j/jia_qu/ACT_analysis/data/D56TTnoise0_3000.txt")
nells_P=np.loadtxt("/global/homes/j/jia_qu/ACT_analysis/data/D56EEnoise0_3000.txt")
nells=nells[:3000]
nells_P=nells_P[:3000]

lmin=500 #minimum reconstruction multipole
LMAXOUT=3000 #maximum output L
Lmin_out=2 #minimum output multipole
LMAX_TT=3000
TMP_OUTPUT=config['data_path']
LCORR_TT=0
Lstep=20 #step size

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


#bins to calculate the L' derivatives
bins=np.array([0.000e+00, 2.000e+00, 1.200e+01, 2.200e+01, 3.200e+01, 4.200e+01,
       5.200e+01, 6.200e+01, 7.200e+01, 8.200e+01, 9.200e+01, 1.020e+02,
       1.320e+02, 1.620e+02, 1.920e+02, 2.220e+02, 2.520e+02, 2.820e+02,
       3.120e+02, 3.420e+02, 3.720e+02, 4.020e+02, 4.320e+02, 4.620e+02,
       4.920e+02, 5.220e+02, 5.520e+02, 6.520e+02, 7.520e+02, 8.520e+02,
       9.520e+02, 1.052e+03, 1.152e+03, 1.252e+03, 1.352e+03, 1.452e+03,
       1.752e+03, 2.052e+03, 2.352e+03, 2.652e+03, 2.952e+03])


#Input normalisation as an array of arrays of lensing potential phi n0s.


thloc = "../data/" + config['theory_root']
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
ells = np.arange(lmax+100)
uctt = theory.lCl('TT',ells)
ucee = theory.lCl('EE',ells)
ucte = theory.lCl('TE',ells)
ucbb = theory.lCl('BB',ells)
tctt = uctt + maps.interp(np.arange(len(nells)),nells)(ells)
tcee = ucee + maps.interp(np.arange(len(nells)),nells_P)(ells)
tcte = ucte 
tcbb = ucbb + maps.interp(np.arange(len(nells)),nells_P)(ells)
ls,Als,al_mv_pol,al_mv,Al_te_hdv = qe.symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)

nbb=np.ones(len(Als['TT']))
norms=np.array([[Als['TT']/ls**2],[Als['EE']/ls**2],[Als['EB']/ls**2],[Als['TE']/ls**2],[Als['TB']/ls**2],[nbb]])

n1bins=np.arange(Lmin_out,LMAXOUT,Lstep)
n1=nbias.compute_n1mv(clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
np.savetxt("/global/homes/j/jia_qu/ACT_analysis/data/testD56n1.txt",n1)

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

n1mv_dclkk=nbias.n1mv_dclkk(clpp,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mvdclkke=extend_matrix(3000,np.nan_to_num(n1mv_dclkk))
np.savetxt(f"{solenspipe.opath}/actn1mvdclkk1.txt",n1mvdclkke)

n1mv_dcltt=nbias.n1mv_dcltt(cltt,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mvdcltte=extend_matrix(3000,np.nan_to_num(n1mv_dcltt))
np.savetxt(f"{solenspipe.opath}/actn1mvdcltte1.txt",n1mvdcltte)

n1mv_dclee=nbias.n1mv_dclee(clee,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mvdcleee=extend_matrix(3000,np.nan_to_num(n1mv_dclee))
np.savetxt(f"{solenspipe.opath}/actn1mvdcleee1.txt",n1mvdcleee)


n1mv_dclbb=nbias.n1mv_dclbb(clbb,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mvdclbbe=extend_matrix(3000,np.nan_to_num(n1mv_dclbb))
np.savetxt(f"{solenspipe.opath}/actn1mvdclbbe1.txt",n1mv_dclbb)

n1mv_dclte=nbias.n1mv_dclte(clte,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n1mvdcltee=extend_matrix(3000,np.nan_to_num(n1mv_dclte))
np.savetxt(f"{solenspipe.opath}/actn1mvdcltee1.txt",n1mvdcltee)

n0mv_dcltt=nbias.n0mvderivative_cltt(cltt,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mvdcltte=extend_matrix(3000,np.nan_to_num(n0mvdcltt))
np.savetxt(f"{solenspipe.opath}/actn0mvdcltt1.txt",n0mvdcltte)

n0mv_dclee=nbias.n0mvderivative_clee(clee,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mvdcleee=extend_matrix(3000,np.nan_to_num(n0mvdclee))
np.savetxt(f"{solenspipe.opath}/actn0mvdclee1.txt",n0mvdcleee)

n0mv_dclbb=nbias.n0mvderivative_clbb(clbb,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mvdclbbe=extend_matrix(3000,np.nan_to_num(n0mvdclbb))
np.savetxt(f"{solenspipe.opath}/actn0mvdclbb1.txt",n0mvdclbbe)

n0mv_dclte=nbias.n0mvderivative_clte(clte,bins,n1bins,clpp,norms,cls,cltt,clee,clbb,clte,nells,nells_P,lmin,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out).transpose()
n0mvdcltee=extend_matrix(3000,np.nan_to_num(n0mvdclte))
np.savetxt(f"{solenspipe.opath}/actn0mvdclte1.txt",n0mvdcltee)
