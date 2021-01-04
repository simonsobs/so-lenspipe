from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs,bunch,utils as putils
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
Here we calculate verify lensing reconstruction for any estimator combination, e.g.
MVMV
Mvpol Mvpol
TTTT
TTTE
TTEE
TTEB
etc.
on the full noiseless sky
for both gradient and curl
with and without bias hardening of TT.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Verify and benchmark RDN0 on the full noiseless sky.')
parser.add_argument("est1", type=str,help='Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("est2", type=str,help='Estimator 2, same as above.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="Number of sims.")
parser.add_argument( "--healpix", action='store_true',help='Use healpix instead of CAR.')
parser.add_argument( "--lmax",     type=int,  default=3000,help="Maximum multipole for lensing.")
parser.add_argument( "--lmin",     type=int,  default=100,help="Minimum multipole for lensing.")
args = parser.parse_args()

with bench.show("INIT"):
    # Multipole limits and resolution
    lmin = args.lmin; lmax = args.lmax
    mlmax = int(4000 * (args.lmax / 3000)) # for alms
    grad = True # Use gradient-field spectra in norm

    nsims = args.nsims # number of sims to test RDN0 and MCN1
    comm,rank,my_tasks = mpi.distribute(nsims)

    # Geometry
    if args.healpix:
        nside = utils.closest_nside(args.lmax)
        shape = None ; wcs = None
        print(f"NSIDE: {nside}")
    else:
        px_arcmin = 2.0  / (args.lmax / 3000)
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(px_arcmin/60.),proj='car')
        nside = None
        print(f"shape,wcs: {shape}, {wcs}")
    px = qe.pixelization(shape=shape,wcs=wcs,nside=nside)

    # Get CMB Cls for response and total Cls corresponding to a noiseless configuration
    ucls,tcls = utils.get_theory_dicts(grad=grad)

with bench.show("NORM"):
    # Get norms for lensing potential, sources and cross
    est_norm_list = [args.est1]
    if args.est2!=args.est1:
        est_norm_list.append(args.est2)
    bh = False
    for e in est_norm_list:
        if e.upper()=='TT' or e.upper()=='MV':
            bh = True
    if bh:
        est_norm_list.append('src')
        R_src_tt = pytempura.get_cross('SRC','TT',ucls,tcls,lmin,lmax,k_ellmax=mlmax)
    Als = pytempura.get_norms(est_norm_list,ucls,tcls,lmin,lmax,k_ellmax=mlmax)
    ls = np.arange(Als[args.est1][0].size)

    # Convert to noise per mode on lensing convergence
    diag = args.est1==args.est2 
    e1 = args.est1.upper()
    e2 = args.est2.upper()
    if diag:
        Nl_g = Als[e1][0] * (ls*(ls+1.)/2.)**2.
        Nl_c = Als[e1][1] * (ls*(ls+1.)/2.)**2.
        if bh:
            Nl_g_bh = solenspipe.bias_hardened_n0(Als[e1][0],Als['src'],R_src_tt) * (ls*(ls+1.)/2.)**2.
    else:
        assert ('MV' not in [e1,e2]) and ('MVPOL' not in [e1,e2])
        R_e1_e2 = pytempura.get_cross(e1,e2,ucls,tcls,lmin,lmax,k_ellmax=mlmax)
        Nl_phi_g = Als[e1][0]*Als[e2][0]*R_e1_e2[0]
        Nl_phi_c = Als[e1][1]*Als[e2][1]*R_e1_e2[1]
        Nl_g = Nl_phi_g * (ls*(ls+1.)/2.)**2.
        Nl_c = Nl_phi_c * (ls*(ls+1.)/2.)**2.
        if bh:
            Nl_g_bh = solenspipe.bias_hardened_n0(Nl_phi_g,Als['src'],R_src_tt) * (ls*(ls+1.)/2.)**2.

    if rank==0:
        np.savetxt(f'{solenspipe.opath}Nlg_{e1}_{e2}.txt',Nl_g)
        np.savetxt(f'{solenspipe.opath}Nlc_{e1}_{e2}.txt',Nl_c)
        if bh: np.savetxt(f'{solenspipe.opath}Nlg_bh_{e1}_{e2}.txt',Nl_g_bh)


theory = cosmology.default_theory()
ells = ls

bin_edges = np.geomspace(2,mlmax,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)[1]
cents = binner.cents

with bench.show("QFUNC"):
    # These are the qfunc lambda functions we will use with RDN0 and MCN1
    q_nobh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2=None,Al2=None,R12=None)
    q_nobh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2=None,Al2=None,R12=None)
    if bh:
        q_bh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e1 in ['TT','MV'] else q_nobh_1
        q_bh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e2 in ['TT','MV'] else q_nobh_2


# This is the filtered map loading function that RDN0 and MCN1 will use
def get_kmap(seed):
    s_i,s_set,noise_seed = solenspipe.convert_seeds(seed)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

uicls = []
uxcls_nobh_1 = []
uxcls_nobh_2 = []
uacls_nobh = []
if bh:
    uxcls_bh_1 = []
    uxcls_bh_2 = []
    uacls_bh = []

# Let's make a single map and cross-correlate with input
for task in my_tasks:
    with bench.show("iKALM"):
        # Get kappa alm
        ikalm = utils.change_alm_lmax(utils.get_kappa_alm(task).astype(np.complex128),mlmax)
    
    with bench.show("Xdat"):
        # Get data
        Xdat = get_kmap((0,0,task))

    with bench.show("QE"):
        # New convention in falafel means maps are potential; we convert to convergence
        r_nobh_1 = plensing.phi_to_kappa(q_nobh_1(Xdat,Xdat))
        r_nobh_2 = plensing.phi_to_kappa(q_nobh_2(Xdat,Xdat)) if not(diag) else r_nobh_1.copy()

    with bench.show("alm2cl and append"):
        uicl = cs.alm2cl(ikalm,ikalm)
        uxcl_nobh_1 = cs.alm2cl(r_nobh_1[0],ikalm)
        uxcl_nobh_2 = cs.alm2cl(r_nobh_2[0],ikalm)
        uacl_nobh = cs.alm2cl(r_nobh_1[0],r_nobh_2[0])

        uicls.append(uicl)
        uxcls_nobh_1.append(uxcl_nobh_1)
        uxcls_nobh_2.append(uxcl_nobh_2)
        uacls_nobh.append(uacl_nobh)

        if bh:
            r_bh_1 = plensing.phi_to_kappa(q_bh_1(Xdat,Xdat))
            r_bh_2 = plensing.phi_to_kappa(q_bh_2(Xdat,Xdat)) if not(diag) else r_bh_1.copy()
            uxcl_bh_1 = cs.alm2cl(r_bh_1[0],ikalm)
            uxcl_bh_2 = cs.alm2cl(r_bh_2[0],ikalm)
            uacl_bh = cs.alm2cl(r_bh_1[0],r_bh_2[0])
            uxcls_bh_1.append(uxcl_bh_1)
            uxcls_bh_2.append(uxcl_bh_2)
            uacls_bh.append(uacl_bh)



with bench.show("MPI Gather"):
    uicls = putils.allgatherv(uicls,comm)
    uxcls_nobh_1 = putils.allgatherv(uxcls_nobh_1,comm)
    uxcls_nobh_2 = putils.allgatherv(uxcls_nobh_2,comm)
    uacls_nobh = putils.allgatherv(uacls_nobh,comm)
    if bh:
        uxcls_bh_1 = putils.allgatherv(uxcls_bh_1,comm)
        uxcls_bh_2 = putils.allgatherv(uxcls_bh_2,comm)
        uacls_bh = putils.allgatherv(uacls_bh,comm)

if rank==0:


    with bench.show("Labels"):
        labs = solenspipe.get_labels()

    with bench.show("Stats"):
        suicls = stats.get_stats(uicls)
        suxcls_nobh_1 = stats.get_stats(uxcls_nobh_1)
        suxcls_nobh_2 = stats.get_stats(uxcls_nobh_2)
        suacls_nobh = stats.get_stats(uacls_nobh)
        if bh:
            suxcls_bh_1 = stats.get_stats(uxcls_bh_1)
            suxcls_bh_2 = stats.get_stats(uxcls_bh_2)
            suacls_bh = stats.get_stats(uacls_bh)

    with bench.show("Save"):
        np.save(f'{solenspipe.opath}mean_uicls_{e1}_{e2}.npy',uicls)
        np.save(f'{solenspipe.opath}mean_uxcls_nobh_1_{e1}_{e2}.npy',uxcls_nobh_1)
        np.save(f'{solenspipe.opath}mean_uxcls_nobh_2_{e1}_{e2}.npy',uxcls_nobh_2)
        np.save(f'{solenspipe.opath}mean_uacls_nobh_{e1}_{e2}.npy',uacls_nobh)
        if bh:
            np.save(f'{solenspipe.opath}mean_uxcls_bh_1_{e1}_{e2}.npy',uxcls_bh_1)
            np.save(f'{solenspipe.opath}mean_uxcls_bh_2_{e1}_{e2}.npy',uxcls_bh_2)
            np.save(f'{solenspipe.opath}mean_uacls_bh_{e1}_{e2}.npy',uacls_bh)


    with bench.show("Bin"):
        uicls = suicls['mean']
        uxcls_nobh_1 = suxcls_nobh_1['mean']
        uxcls_nobh_2 = suxcls_nobh_2['mean']
        uacls_nobh = suacls_nobh['mean']
        if bh:
            uxcls_bh_1 = suxcls_bh_1['mean']
            uxcls_bh_2 = suxcls_bh_2['mean']
            uacls_bh = suacls_bh['mean']


        icls = bin(uicls)
        xcls_nobh_1 = bin(uxcls_nobh_1)
        xcls_nobh_2 = bin(uxcls_nobh_2)
        acls_nobh = bin(uacls_nobh)
        if bh:
            xcls_bh_1 = bin(uxcls_bh_1)
            xcls_bh_2 = bin(uxcls_bh_2)
            acls_bh = bin(uacls_bh)


    with bench.show("Plot"):
        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
        pl.add(ells[1:],Nl_g[1:],ls='--',label=labs.nlg)
        pl.add(cents,icls+bin(Nl_g),ls='--',label=labs.tclng)
        pl.add(cents,xcls_nobh_1,marker='o',label=labs.xcl)
        pl.add(cents,xcls_nobh_2,marker='o',label=labs.xcl)
        pl.add(cents,acls_nobh,marker='o',label=labs.acl)
        pl._ax.set_ylim(1e-10,1e-6)
        pl.legend('outside')
        pl.done(f'{solenspipe.opath}cross_verify_nobh_{e1}_{e2}.png')

        if bh:
            pl = io.Plotter('CL',xyscale='loglog')
            pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
            pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
            pl.add(ells[1:],Nl_g_bh[1:],ls='--',label='BH ' + labs.nlg)
            pl.add(cents,icls+bin(Nl_g_bh),ls='--',label='BH ' + labs.tclng)
            pl.add(cents,xcls_bh_1,marker='o',label=labs.xcl)
            pl.add(cents,xcls_bh_2,marker='o',label=labs.xcl)
            pl.add(cents,acls_bh,marker='o',label=labs.acl)
            pl._ax.set_ylim(1e-10,1e-6)
            pl.legend('outside')
            pl.done(f'{solenspipe.opath}cross_verify_bh_{e1}_{e2}.png')


