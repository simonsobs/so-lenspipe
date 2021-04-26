from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs
import numpy as np
import os,sys
from falafel import qe,utils
import solenspipe
from solenspipe import bias,biastheory
import pytempura
import healpy as hp
from enlib import bench
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

"""
Here we calculate RDN0, MCN0 and MCN1 for any estimator combination, e.g.
MVMV
Mvpol Mvpol
TTTT
TTTE
TTEE
TTEB
etc.
and compare against theory N0 on the full noiseless sky
for both gradient and curl
with and without bias hardening of TT.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Verify and benchmark RDN0 on the full noiseless sky.')
parser.add_argument("version", type=str,help='Version name.')
parser.add_argument("est1", type=str,help='Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("est2", type=str,help='Estimator 2, same as above.')
parser.add_argument("--nsims-n0",     type=int,  default=1,help="Number of sims.")
parser.add_argument("--nsims-n1",     type=int,  default=1,help="Number of sims.")
parser.add_argument( "--healpix", action='store_true',help='Use healpix instead of CAR.')
parser.add_argument( "--new-scheme", action='store_true',help='New simulation scheme.')
parser.add_argument( "--lmax",     type=int,  default=3000,help="Maximum multipole for lensing.")
parser.add_argument( "--lmin",     type=int,  default=100,help="Minimum multipole for lensing.")
parser.add_argument( "--biases",     type=str,  default="rdn0,mcn1",help="Maximum multipole for lensing.")
args = parser.parse_args()

biases = args.biases.split(',')
opath = f"{solenspipe.opath}/{args.version}_"

# Multipole limits and resolution
lmin = args.lmin; lmax = args.lmax
mlmax = int(4000 * (args.lmax / 3000)) # for alms
grad = True # Use gradient-field spectra in norm

nsims_n0 = args.nsims_n0 # number of sims to test RDN0
nsims_n1 = args.nsims_n1 # number of sims to test MCN1
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()


# Geometry
px = solenspipe.get_sim_pixelization(args.lmax,args.healpix,verbose=(rank==0))

# Get CMB Cls for response and total Cls corresponding to a noiseless configuration
ucls,tcls = utils.get_theory_dicts(grad=grad)

# Get norms
bh,ells,Als,R_src_tt,Nl_g,Nl_c,Nl_g_bh = solenspipe.get_tempura_norms(args.est1,args.est2,ucls,tcls,args.lmin,args.lmax,mlmax)

e1 = args.est1.upper()
e2 = args.est2.upper()
diag = (e1==e2)
if rank==0:
    np.savetxt(f'{opath}Nlg_{e1}_{e2}.txt',Nl_g)
    np.savetxt(f'{opath}Nlc_{e1}_{e2}.txt',Nl_c)
    if bh: np.savetxt(f'{opath}Nlg_bh_{e1}_{e2}.txt',Nl_g_bh)


# Plot expected noise performance
theory = cosmology.default_theory()

if rank==0:
    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(ells[1:],Nl_g[1:],ls='--',label='grad')
    pl.add(ells[2:],Nl_c[2:],ls='-.',label='curl')
    if bh:
        pl.add(ells[1:],Nl_g_bh[1:],ls=':',label='grad BH')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{opath}bh_noise_{e1}_{e2}.png')

bin_edges = np.geomspace(2,mlmax,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)[1]
cents = binner.cents

# These are the qfunc lambda functions we will use with RDN0 and MCN1
q_nobh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2=None,Al2=None,R12=None)
q_nobh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2=None,Al2=None,R12=None)
if bh:
    q_bh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e1 in ['TT','MV'] else q_nobh_1
    q_bh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e2 in ['TT','MV'] else q_nobh_2


# Build get_kmap functions

def get_kmap(seed=None,s_i=None,s_set=None,noise_seed=None):
    if not(seed is None):
        s_i,s_set,noise_seed = solenspipe.convert_seeds(seed)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

# Get data
Xdat = get_kmap(s_i=1999,s_set=0) #TODO: fix hardcoding
dlmax = hp.Alm.getlmax(Xdat[0].size)
if rank==0: print(f"Data lmax: {dlmax}")

if args.new_scheme:
    def get_kmap_rdn0(seed):
        icov,iset,i = seed
        nsims = 100
        cmb_set = 0
        phi_set = 0
        if iset==0:
            s_i = i
        elif iset==1:
            s_i = nsims//2 + i
        else:
            raise ValueError
        dalm = utils.get_cmb_alm_v0p5(cmb_set,phi_set,s_i)
        dalm = utils.change_alm_lmax(dalm.astype(np.complex128),dlmax)
        return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

    def get_kmap_mcn1(seed):
        icov,iset,i = seed
        nsims = 100
        if (iset==0) or (iset==2):
            cmb_set = 1
            phi_set = 1
            s_i = i
        elif iset==1:
            cmb_set = 2
            phi_set = 2
            s_i = i
        elif iset==3:
            cmb_set = 2
            phi_set = 1
            s_i = i
        else:
            raise ValueError
        dalm = utils.get_cmb_alm_v0p5(cmb_set,phi_set,s_i)
        dalm = utils.change_alm_lmax(dalm.astype(np.complex128),dlmax)
        return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

    kmap_func = {'rdn0': get_kmap_rdn0, 'mcn1': get_kmap_mcn1}

else:

    # This is the filtered map loading function that RDN0 and MCN1 will use
    kmap_func = {'rdn0': get_kmap, 'mcn1': get_kmap}



# Let's make a single map and cross-correlate with input
if rank==0:
    # Get kappa alm
    ikalm = utils.change_alm_lmax(utils.get_kappa_alm(1999).astype(np.complex128),mlmax) # TODO: fix hardcoding
    

    # New convention in falafel means maps are potential; we convert to convergence
    r_nobh_1 = plensing.phi_to_kappa(q_nobh_1(Xdat,Xdat))
    r_nobh_2 = plensing.phi_to_kappa(q_nobh_2(Xdat,Xdat))

    uicls = cs.alm2cl(ikalm,ikalm)
    uxcls_nobh_1 = cs.alm2cl(r_nobh_1[0],ikalm)
    uxcls_nobh_2 = cs.alm2cl(r_nobh_2[0],ikalm)
    uacls_nobh = cs.alm2cl(r_nobh_1,r_nobh_2)
    np.save(f'{opath}uicls_{e1}_{e2}.npy',uicls)
    np.save(f'{opath}uxcls_nobh_1_{e1}_{e2}.npy',uxcls_nobh_1)
    np.save(f'{opath}uxcls_nobh_2_{e1}_{e2}.npy',uxcls_nobh_2)
    np.save(f'{opath}uacls_nobh_{e1}_{e2}.npy',uacls_nobh)

    if bh:
        r_bh_1 = plensing.phi_to_kappa(q_bh_1(Xdat,Xdat))
        r_bh_2 = plensing.phi_to_kappa(q_bh_2(Xdat,Xdat))
        uxcls_bh_1 = cs.alm2cl(r_bh_1[0],ikalm)
        uxcls_bh_2 = cs.alm2cl(r_bh_2[0],ikalm)
        uacls_bh = cs.alm2cl(r_bh_1,r_bh_2)
        np.save(f'{opath}uxcls_bh_1_{e1}_{e2}.npy',uxcls_bh_1)
        np.save(f'{opath}uxcls_bh_2_{e1}_{e2}.npy',uxcls_bh_2)
        np.save(f'{opath}uacls_bh_{e1}_{e2}.npy',uacls_bh)


powfunc = lambda x,y: cs.alm2cl(x,y)

bias_nobh = {}
bias_bh = {}

# Loop over bias hardened and non-bias hardened
for odict,isbh,bhlab in zip([bias_nobh,bias_bh],[False,True],['nobh','bh']):
    if isbh:
        if not(bh): continue
        q1 = q_bh_1
        q2 = None if diag else q_bh_2
    else:
        q1 = q_nobh_1
        q2 = None if diag else q_nobh_2

    if 'rdn0' in biases:
        # RDN0 or MCN1 for lensing potential using the callables defined earlier
        # converted to lensing convergence
        rdn0,mcn0 = bias.mcrdn0(0, kmap_func['rdn0'], powfunc, nsims_n0, 
                                q1, 
                                qfunc2=q2,
                                Xdat=Xdat,
                                use_mpi=True) 

        rdn0 = rdn0 * (ells*(ells+1.)/2.)**2.
        mcn0 = mcn0 * (ells*(ells+1.)/2.)**2.
        if rank==0: 
            np.save(f'{opath}rdn0_{bhlab}_{e1}_{e2}.npy',rdn0)
            np.save(f'{opath}mcn0_{bhlab}_{e1}_{e2}.npy',mcn0)
        rdn0 = rdn0.mean(axis=0) # just use average for this script
        mcn0 = mcn0.mean(axis=0) # just use average for this script
        odict['rdn0'] = np.asarray([bin(x) for x in rdn0])
        odict['mcn0'] = np.asarray([bin(x) for x in mcn0])

    if 'mcn1' in biases:
        mcn1 = bias.mcn1(0, kmap_func['mcn1'], powfunc, nsims_n1, 
                               q1, 
                               qfunc2=q2, 
                               comm=comm) 

        mcn1 = mcn1 * (ells*(ells+1.)/2.)**2.
        if rank==0: 
            np.save(f'{opath}mcn1_{bhlab}_{e1}_{e2}.npy',mcn1)
        mcn1 = mcn1.mean(axis=0) # just use average for this script
        odict['mcn1'] = np.asarray([bin(x) for x in mcn1])


# PLOTS
if rank==0:

    labs = solenspipe.get_labels()
    icls = bin(uicls)
    xcls_nobh_1 = bin(uxcls_nobh_1)
    xcls_nobh_2 = bin(uxcls_nobh_2)
    acls_nobh = np.asarray([bin(x) for x in uacls_nobh])
    if bh:
        xcls_bh_1 = bin(uxcls_bh_1)
        xcls_bh_2 = bin(uxcls_bh_2)
        acls_bh = np.asarray([bin(x) for x in uacls_bh])

    if 'rdn0' in biases:
        rdn0_nobh = bias_nobh['rdn0']
        if bh: rdn0_bh = bias_bh['rdn0']
        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
        pl.add(ells[1:],Nl_g[1:],ls='--',label=labs.nlg)
        pl.add(cents,icls+bin(Nl_g),ls='--',label=labs.tclng)
        pl.add(cents,rdn0_nobh[0] ,ls=':',label=labs.rdn0g,marker='d')
        pl.add(cents,icls+(rdn0_nobh[0] ),ls=':',label=labs.tcrdn0g,marker='d')
        pl.add(cents,xcls_nobh_1,marker='o',label=labs.xcl)
        pl.add(cents,xcls_nobh_2,marker='o',label=labs.xcl)
        pl.add(cents,acls_nobh[0],marker='o',label=labs.acl)
        pl._ax.set_ylim(1e-10,1e-6)
        pl.legend('outside')
        pl.done(f'{opath}rdn0_nobh_{e1}_{e2}.png')

        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(ells[1:],Nl_c[1:],ls='--',label=labs.nlc)
        pl.add(cents,rdn0_nobh[1] ,ls=':',label=labs.rdn0c,marker='d')
        if bh: pl.add(cents,rdn0_bh[1] ,ls=':',label='BH ' + labs.rdn0c,marker='d')
        pl._ax.set_ylim(1e-10,1e-6)
        pl.legend('outside')
        pl.done(f'{opath}rdn0_curl_{e1}_{e2}.png')


        if bh:
            pl = io.Plotter('CL',xyscale='loglog')
            pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
            pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
            # Convert RDN0 to convergence
            pl.add(cents,rdn0_bh[0] ,ls=':',label='BH ' + labs.rdn0g,marker='d')
            pl.add(cents,icls+(rdn0_bh[0] ),ls=':',label='BH ' + labs.tcrdn0g,marker='d')
            pl.add(ells[1:],Nl_g_bh[1:],ls='--',label='BH ' + labs.nlg)
            pl.add(cents,icls+bin(Nl_g_bh),ls='--',label='BH ' + labs.tclng)
            pl.add(cents,xcls_bh_1,marker='o',label=labs.xcl)
            pl.add(cents,xcls_bh_2,marker='o',label=labs.xcl)
            pl.add(cents,acls_bh[0],marker='o',label=labs.acl)
            pl._ax.set_ylim(1e-10,1e-6)
            pl.legend('outside')
            pl.done(f'{opath}rdn0_bh_{e1}_{e2}.png')

    if 'mcn1' in biases:
        mcn1_nobh = bias_nobh['mcn1']
        if bh: mcn1_bh = bias_bh['mcn1']
        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(ells[1:],Nl_g[1:],ls='--',label=labs.nlg)
        pl.add(cents,icls+bin(Nl_g),ls='--',label=labs.tclng)
        pl.add(cents,mcn1_nobh[0] ,ls=':',label=labs.mcn1g,marker='d')
        pl.add(cents,xcls_nobh_1,marker='o',label=labs.xcl)
        pl.add(cents,xcls_nobh_2,marker='o',label=labs.xcl)
        pl.add(cents,acls_nobh[0],marker='o',label=labs.acl)
        pl._ax.set_ylim(1e-10,1e-6)
        pl.legend('outside')
        pl.done(f'{opath}mcn1_nobh_{e1}_{e2}.png')

        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(ells[1:],Nl_c[1:],ls='--',label=labs.nlc)
        pl.add(cents,mcn1_nobh[1] ,ls=':',label=labs.mcn1c,marker='d')
        if bh: pl.add(cents,mcn1_bh[1] ,ls=':',label='BH ' + labs.mcn1c,marker='d')
        pl._ax.set_ylim(1e-10,1e-6)
        pl.legend('outside')
        pl.done(f'{opath}mcn1_curl_{e1}_{e2}.png')


        if bh:
            pl = io.Plotter('CL',xyscale='loglog')
            pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
            pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
            pl.add(cents,mcn1_bh[0] ,ls=':',label='BH ' + labs.mcn1g,marker='d')
            pl.add(ells[1:],Nl_g_bh[1:],ls='--',label='BH ' + labs.nlg)
            pl.add(cents,icls+bin(Nl_g_bh),ls='--',label='BH ' + labs.tclng)
            pl.add(cents,xcls_bh_1,marker='o',label=labs.xcl)
            pl.add(cents,xcls_bh_2,marker='o',label=labs.xcl)
            pl.add(cents,acls_bh[0],marker='o',label=labs.acl)
            pl._ax.set_ylim(1e-10,1e-6)
            pl.legend('outside')
            pl.done(f'{opath}mcn1_bh_{e1}_{e2}.png')


    if ('mcn1' in biases) and ('rdn0' in biases):
        mcn1_nobh = bias_nobh['mcn1']
        rdn0_nobh = bias_nobh['rdn0']
        est_nobh = acls_nobh - rdn0_nobh - mcn1_nobh
        if bh: 
            rdn0_bh = bias_bh['rdn0']
            mcn1_bh = bias_bh['mcn1']
            est_bh = acls_bh - rdn0_bh - mcn1_bh
        

        pl = io.Plotter('rCL',xyscale='loglin')
        pl.add(cents,(est_nobh[0]-icls)/icls,label='No BH Grad',marker='o')
        if bh: pl.add(cents,(est_bh[0]-icls)/icls,label='BH Grad',marker='o',ls='--')
        pl.add(cents,(xcls_nobh_1-icls)/icls,label='No BH Grad Cross',marker='d')
        if bh: pl.add(cents,(xcls_bh_1-icls)/icls,label='BH Grad Cross',marker='d',ls='--')
        pl.add(cents,(xcls_nobh_2-icls)/icls,label='No BH Grad Cross',marker='d')
        if bh: pl.add(cents,(xcls_bh_2-icls)/icls,label='BH Grad Cross',marker='d',ls='--')
        pl._ax.set_ylim(-0.2,0.2)
        pl.hline(y=0)
        pl.legend('outside')
        pl.done(f'{opath}verify_{e1}_{e2}.png')

        pl = io.Plotter('rCL',xyscale='loglin')
        pl.add(cents,est_nobh[1]/icls,label='No BH Curl',marker='d')
        if bh: pl.add(cents,est_bh[1]/icls,label='BH Curl',marker='d',ls='--')
        pl._ax.set_ylim(-0.2,0.2)
        pl.hline(y=0)
        pl.legend('outside')
        pl.done(f'{opath}verify_curl_{e1}_{e2}.png')

