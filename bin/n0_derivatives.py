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
#Parse command line
parser = argparse.ArgumentParser(description='Demo lensing pipeline.')

#parser.add_argument("polcomb", type=str,help='Polarization combination. Possibilities include mv (all), mvpol (all pol), TT, EE, TE, EB or TB.')
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
ells = np.arange(lmax+100)
ls,nells = solint.nsim.ell,solint.nsim.noise_ell_T[ch.telescope][int(ch.band)]
ls,nells_P = solint.nsim.ell,solint.nsim.noise_ell_P[ch.telescope][int(ch.band)]

uctt = theory.lCl('TT',ells)
ucee = theory.lCl('EE',ells)
ucte = theory.lCl('TE',ells)
ucbb = theory.lCl('BB',ells)

#lprime value to evaluate the derivatives
lprime=np.array([4.000e+00, 1.200e+01, 2.200e+01, 3.200e+01, 4.200e+01, 5.200e+01,
       6.200e+01, 7.200e+01, 8.200e+01, 9.200e+01, 1.020e+02, 1.320e+02,
       1.620e+02, 1.920e+02, 2.220e+02, 2.520e+02, 2.820e+02, 3.120e+02,
       3.420e+02, 3.720e+02, 4.020e+02, 4.320e+02, 4.620e+02, 4.920e+02,
       5.220e+02, 5.520e+02, 6.520e+02, 7.520e+02, 8.520e+02, 9.520e+02,
       1.052e+03, 1.152e+03, 1.252e+03, 1.352e+03, 1.452e+03, 1.752e+03,
       2.052e+03, 2.352e+03, 2.652e+03, 2.952e+03])


def perturbe_clist(cl_array,bins,amount):
    """generate a list of cls where the cls at the position bins are perturbed by amount keeping other cls unperturbed"""
    cltt_list=[]
    for i in range(len(bins)):
        cl=cl_array.copy()
        cl[int(bins[i])]=amount*cl_array[int(bins[i])]
        cltt_list.append(cl)
    return cltt_list
    
def diff_cl(cl_array,bins):
    """deltacls used in the denominator of finite difference derivatives"""
    dcltt=[]
    for i in range(len(bins)):
        dcltt.append(2*0.001*cl_array[int(bins[i])])
    return dcltt
    
def norm_der_cltt(solint,ch,lmin,lmax,uctt_,ucee_,ucte_,ucbb_):
    #these are the unperturbed ones
    uctt=uctt_
    ucee=ucee_
    ucte=ucte_
    ucbb=ucbb_
    #the total goes with the Fs hence are unperturbed
    tctt = uctt + maps.interp(ls,nells)(ells)
    tcee = ucee + maps.interp(ls,nells_P)(ells)
    tcte = ucte 
    tcbb = ucbb + maps.interp(ls,nells_P)(ells)
    #onormfname = opath+"norm_lmin_%d_lmax_%d.txt" % (lmin,lmax)
    
    array1001=perturbe_clist(uctt,lprime,1.001)
    array999=perturbe_clist(uctt,lprime,0.999)
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):

        print(i)
        als,aAls,aal_mv_pol,aal_mv,aAl_te_hdv=qe.symlens_norm(array1001[i],tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        bls,bAls,bal_mv_pol,bal_mv,bAl_te_hdv=qe.symlens_norm(array999[i],tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        
        N1001[0].append(aAls['TT'])
        N0999[0].append(bAls['TT'])
        N1001[1].append(aAls['EE'])
        N0999[1].append(bAls['EE'])
        N1001[2].append(aAls['EB'])
        N0999[2].append(bAls['EB'])
        N1001[3].append(aAls['TE'])
        N0999[3].append(bAls['TE'])      
        N1001[4].append(aAls['TB'])
        N0999[4].append(bAls['TB'])
        
    delta=diff_cl(uctt,lprime)
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(len(keys)):
        diff=[als]
        for i in range(len(N1001[1])):
        
            der=((N1001[k][i]-N0999[k][i]))/delta[i]
            diff.append(der)    
        #np.savetxt('/global/homes/j/jia_qu/so-lenspipe/data/n0{}_cltt.txt'.format(keys[k]),der)
        np.savetxt('../data/n0{}_cltt.txt'.format(keys[k]),diff)
  
def norm_der_clee(solint,ch,lmin,lmax,uctt_,ucee_,ucte_,ucbb_):
    #these are the unperturbed ones
    uctt=uctt_
    ucee=ucee_
    ucte=ucte_
    ucbb=ucbb_
    #the total goes with the Fs hence are unperturbed
    tctt = uctt + maps.interp(ls,nells)(ells)
    tcee = ucee + maps.interp(ls,nells_P)(ells)
    tcte = ucte 
    tcbb = ucbb + maps.interp(ls,nells_P)(ells)
    #onormfname = opath+"norm_lmin_%d_lmax_%d.txt" % (lmin,lmax)
    
    array1001=perturbe_clist(ucee,lprime,1.001)
    array999=perturbe_clist(ucee,lprime,0.999)
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
    
        print(i)
        als,aAls,aal_mv_pol,aal_mv,aAl_te_hdv=qe.symlens_norm(uctt,tctt,array1001[i],tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        bls,bAls,bal_mv_pol,bal_mv,bAl_te_hdv=qe.symlens_norm(uctt,tctt,array999[i],tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        
        N1001[0].append(aAls['TT'])
        N0999[0].append(bAls['TT'])
        N1001[1].append(aAls['EE'])
        N0999[1].append(bAls['EE'])
        N1001[2].append(aAls['EB'])
        N0999[2].append(bAls['EB'])
        N1001[3].append(aAls['TE'])
        N0999[3].append(bAls['TE'])      
        N1001[4].append(aAls['TB'])
        N0999[4].append(bAls['TB'])
        
    delta=diff_cl(ucee,lprime)
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(len(keys)):
        diff=[als]
        for i in range(len(N1001[1])):
            der=((N1001[k][i]-N0999[k][i]))/delta[i]
            diff.append(der)    
        np.savetxt('../data/n0{}_clee.txt'.format(keys[k]),diff)

def norm_der_clte(solint,ch,lmin,lmax,uctt_,ucee_,ucte_,ucbb_):
    #these are the unperturbed ones
    uctt=uctt_
    ucee=ucee_
    ucte=ucte_
    ucbb=ucbb_

    #the total goes with the Fs hence are unperturbed
    tctt = uctt + maps.interp(ls,nells)(ells)
    tcee = ucee + maps.interp(ls,nells_P)(ells)
    tcte = ucte 
    tcbb = ucbb + maps.interp(ls,nells_P)(ells)
    #onormfname = opath+"norm_lmin_%d_lmax_%d.txt" % (lmin,lmax)
    
    array1001=perturbe_clist(ucte,lprime,1.001)
    array999=perturbe_clist(ucte,lprime,0.999)
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
    
        print(i)
        als,aAls,aal_mv_pol,aal_mv,aAl_te_hdv=qe.symlens_norm(uctt,tctt,ucee,tcee,array1001[i],tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        bls,bAls,bal_mv_pol,bal_mv,bAl_te_hdv=qe.symlens_norm(uctt,tctt,ucee,tcee,array999[i],tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        
        N1001[0].append(aAls['TT'])
        N0999[0].append(bAls['TT'])
        N1001[1].append(aAls['EE'])
        N0999[1].append(bAls['EE'])
        N1001[2].append(aAls['EB'])
        N0999[2].append(bAls['EB'])
        N1001[3].append(aAls['TE'])
        N0999[3].append(bAls['TE'])      
        N1001[4].append(aAls['TB'])
        N0999[4].append(bAls['TB'])
        
    delta=diff_cl(ucte,lprime)
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(len(keys)):
        diff=[als]
        for i in range(len(N1001[1])):
            der=((N1001[k][i]-N0999[k][i]))/delta[i]
            diff.append(der)    
        np.savetxt('../data/n0{}_clte.txt'.format(keys[k]),diff)
        
def norm_der_clbb(solint,ch,lmin,lmax,uctt_,ucee_,ucte_,ucbb_):
    #these are the unperturbed ones
    uctt=uctt_
    ucee=ucee_
    ucte=ucte_
    ucbb=ucbb_

    #the total goes with the Fs hence are unperturbed
    tctt = uctt + maps.interp(ls,nells)(ells)
    tcee = ucee + maps.interp(ls,nells_P)(ells)
    tcte = ucte 
    tcbb = ucbb + maps.interp(ls,nells_P)(ells)
    #onormfname = opath+"norm_lmin_%d_lmax_%d.txt" % (lmin,lmax)
    
    array1001=perturbe_clist(ucbb,lprime,1.001)
    array999=perturbe_clist(ucbb,lprime,0.999)
    N1001=[[],[],[],[],[]] #list of lists containing tt,ee,eb,te,tb
    N0999=[[],[],[],[],[]]
    
    for i in range(len(array1001)):
    
        print(i)
        als,aAls,aal_mv_pol,aal_mv,aAl_te_hdv=qe.symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,array1001[i],tcbb,lmin=lmin,lmax=lmax,plot=False)
        bls,bAls,bal_mv_pol,bal_mv,bAl_te_hdv=qe.symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,array999[i],tcbb,lmin=lmin,lmax=lmax,plot=False)
        
        N1001[0].append(aAls['TT'])
        N0999[0].append(bAls['TT'])
        N1001[1].append(aAls['EE'])
        N0999[1].append(bAls['EE'])
        N1001[2].append(aAls['EB'])
        N0999[2].append(bAls['EB'])
        N1001[3].append(aAls['TE'])
        N0999[3].append(bAls['TE'])      
        N1001[4].append(aAls['TB'])
        N0999[4].append(bAls['TB'])
        
    delta=diff_cl(ucbb,lprime)
    keys=['TT','EE','EB','TE','TB']
    
    for k in range(len(keys)):
        diff=[als]
        for i in range(len(N1001[1])):
            der=((N1001[k][i]-N0999[k][i]))/delta[i]
            diff.append(der)    
        np.savetxt('../data/n0{}_clbb.txt'.format(keys[k]),diff)

norm_der_cltt(solint,ch,lmin,lmax,uctt,ucee,ucte,ucbb)
norm_der_clee(solint,ch,lmin,lmax,uctt,ucee,ucte,ucbb)
norm_der_clte(solint,ch,lmin,lmax,uctt,ucee,ucte,ucbb)
norm_der_clbb(solint,ch,lmin,lmax,uctt,ucee,ucte,ucbb)

