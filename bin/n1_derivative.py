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
from solenspipe._lensing_biases import lensingbiases as lensingbiases_f

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
n1bins=np.loadtxt('../data/n1bins.txt') #correspond to the ells used to compute N1
clkk=np.loadtxt('../data/ckk.txt')
lens=np.loadtxt("../data/cosmo2017_10K_acc3_lenspotentialCls.dat",unpack=True)
cls=np.loadtxt("../data/cosmo2017_10K_acc3_lensedCls.dat",unpack=True)
clpp=lens[5,:][:8249]
norms=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/norm_lmin_300_lmax_3000.txt")
bins=norms[2:,0]
ntt=norms[2:,1]
nee=norms[2:,2]
neb=norms[2:,3]
nte=norms[2:,4]
ntb=norms[2:,5]
nbb=np.ones(len(ntb))
norms=np.array([ntt/bins**2,nee/bins**2,neb/bins**2,nte/bins**2,ntb/bins**2,nbb])
"""
lprime=np.array([2.000e+00, 1.200e+01, 2.200e+01, 3.200e+01, 4.200e+01, 5.200e+01,
       6.200e+01, 7.200e+01, 8.200e+01, 9.200e+01, 1.020e+02, 1.320e+02,
       1.620e+02, 1.920e+02, 2.220e+02, 2.520e+02, 2.820e+02, 3.120e+02,
       3.420e+02, 3.720e+02, 4.020e+02, 4.320e+02, 4.620e+02, 4.920e+02,
       5.220e+02, 5.520e+02, 6.520e+02, 7.520e+02, 8.520e+02, 9.520e+02,
       1.052e+03, 1.152e+03, 1.252e+03, 1.352e+03, 1.452e+03, 1.752e+03,
       2.052e+03, 2.352e+03, 2.652e+03, 2.952e+03])
lprime=np.arange(2.,2992.,20)
"""
lprime=np.arange(1393,1413)
#because cls array start from 2 shift everything by -1
bind=lprime-2
#unperturbed cl arrays
cltt=cls[1]
clee=cls[2]
clbb=cls[3]
clte=cls[4]

def perturbe_clist(cl_array,bins,amount):
    """generate a list of cls where the cls at the position bins are perturbed by amount keeping other cls unperturbed"""
    cltt_list=[]
    for i in range(len(bins)):
        cl=cl_array.copy()
        cl[int(bins[i])]=amount*cl_array[int(bins[i])]
        cltt_list.append(cl)
    return cltt_list

def diff_cl(cl_array,bins,perturbation):
    """deltacls used in the denominator of finite difference derivatives"""
    ls=np.arange(2,len(cl_array)+2)
    cls=cl_array*2*np.pi/(ls*(ls+1))
    dcltt=[]
    for i in range(len(bins)):
        dcltt.append(2*perturbation*cls[int(bins[i])])
    return dcltt
    
#clphiphi derivative. Still need to change output into arrays
#s.n1_derivatives('TT','TT',clpp,cls,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)



def derivative_clee(cl_array,bins):

    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
        print(i)
        a=s.compute_n1_py(clpp,norms,cls,cltt,array1001[i],clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        b=s.compute_n1_py(clpp,norms,cls,cltt,array999[i],clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    
    
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:150]-N0999[k][i][:150])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(lprime,0,0),axis=0)
        np.savetxt('../data/n1clee{}.txt'.format(keys[k]),der)       
    
def derivative_clbb(cl_array,bins):
    
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
        print(i)
        a=s.compute_n1_py(clpp,norms,cls,cltt,clee,array1001[i],clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        b=s.compute_n1_py(clpp,norms,cls,cltt,clee,array999[i],clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    
    
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:150]-N0999[k][i][:150])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(lprime,0,0),axis=0)
        np.savetxt('../data/n1clbb{}.txt'.format(keys[k]),der)    

def derivative_clte(cl_array,bins):
    
    array1001=perturbe_clist(cl_array,bins,1.001)
    array999=perturbe_clist(cl_array,bins,0.999)
    
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
        print(i)
        a=s.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,array1001[i],FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        b=s.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,array999[i],FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])

    delta=diff_cl(cl_array,bins)
    
    
    
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(len(keys)):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            der=((N1001[k][i][:150]-N0999[k][i][:150])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(lprime,0,0),axis=0)
        np.savetxt('../data/n1clte{}.txt'.format(keys[k]),der) 

def derivative_cltt(cl_array,bins,perturbation):
    
    array1001=perturbe_clist(cl_array,bins,1+perturbation)
    array999=perturbe_clist(cl_array,bins,1-perturbation)
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    delta=diff_cl(cl_array,bins,perturbation)
    #print(delta)
    
    for i in range(len(array1001)):
    #for i in range(30,31):
        a=s.compute_n1_py(clpp,norms,cls,array1001[i],clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        b=s.compute_n1_py(clpp,norms,cls,array999[i],clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
        for j in range(len(N1001)):
            N1001[j].append(a[j])
            N0999[j].append(b[j])


    
    
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(1):
        diff=[n1bins]
        for i in range(len(N1001[1])):
            #print(((N1001[k][i][:150]-N0999[k][i][:150])*(n1bins*(n1bins+1))**2*0.25))
            der=((N1001[k][i][:150]-N0999[k][i][:150])*(n1bins*(n1bins+1))**2*0.25)/delta[i]
            diff.append(der)
        der=np.insert(np.transpose(diff),0,np.insert(lprime,0,0),axis=0)
        np.savetxt('../data/n1clt{}.txt'.format(keys[k]),der)

listi=[0.001]
for i in range(len(listi)):
    derivative_cltt(cltt,bind,listi[i]) 

#derivative_clee(clee,bind)
#derivative_clbb(clbb,bind)
#derivative_clte(clte,bind)

"""
cltest=np.loadtxt('/global/homes/j/jia_qu/so-lenspipe/data/cltest1.txt')
cltest9=np.loadtxt('/global/homes/j/jia_qu/so-lenspipe/data/cltest9.txt')
"""

"""
cltest=cltt.copy()
clow=cltt.copy()
for i in range(len(bind)):
    cltest[int(bind[i])]=1.001*cltt[int(bind[i])]
    clow[int(bind[i])]=0.999*cltt[int(bind[i])]
""" 
"""
cltest=cltt.copy()
clow=cltt.copy()
for i in range(2992):
    cltest[i]=1.001*cltt[i]
    clow[i]=0.999*cltt[i]
"""
"""
clband=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/clpoint.txt")

boostf=s.compute_n1_py(clpp,cls,1.001*clband,clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
lower=s.compute_n1_py(clpp,cls,0.999*clband,clee,clbb,clte,FWHM,NOISE_LEVEL,polnoise,LMIN,LMAXOUT,LMAX_TT,LCORR_TT,TMP_OUTPUT)
np.savetxt('../data/testp1.txt',boostf[0])
np.savetxt('../data/testp0.txt',lower[0])
"""
