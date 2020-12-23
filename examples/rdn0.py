from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs,bunch
import numpy as np
import os,sys
from falafel import qe,utils
import solenspipe
from solenspipe import bias
import pytempura
import healpy as hp
from enlib import bench
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

"""
Here we calculate RDN0 for any estimator combination, e.g.
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
parser.add_argument("est1", type=str,help='Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.')
parser.add_argument("est2", type=str,help='Estimator 2, same as above.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="Number of sims.")
parser.add_argument( "--lmax",     type=int,  default=3000,help="Maximum multipole for lensing.")
parser.add_argument( "--lmin",     type=int,  default=100,help="Minimum multipole for lensing.")
args = parser.parse_args()


# Multipole limits and resolution
lmin = args.lmin; lmax = args.lmax
mlmax = int(4000 * (args.lmax / 3000)) # for alms
px_arcmin = 2.0  / (args.lmax / 3000)
grad = True # Use gradient-field spectra in norm

nsims = args.nsims # number of sims to test RDN0
comm,rank,my_tasks = mpi.distribute(nsims)

# Geometry
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(px_arcmin/60.),proj='car')
px = qe.pixelization(shape,wcs)

# Get CMB Cls for response and total Cls corresponding to a noiseless configuration
ucls,tcls = utils.get_theory_dicts(grad=grad)

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

# Plot expected noise performance
theory = cosmology.default_theory()
ells = ls

if rank==0:
    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(ells[1:],Nl_g[1:],ls='--',label='grad')
    pl.add(ells[2:],Nl_c[2:],ls='-.',label='curl')
    if bh:
        pl.add(ells[1:],Nl_g_bh[1:],ls=':',label='grad BH')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{solenspipe.opath}bh_noise_{e1}_{e2}.png')

bin_edges = np.geomspace(2,mlmax,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)[1]
cents = binner.cents

# These are the qfunc lambda functions we will use with RDN0
q_nobh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2=None,Al2=None,R12=None)
q_nobh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2=None,Al2=None,R12=None)
if bh:
    q_bh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e1 in ['TT','MV'] else q_nobh_1
    q_bh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2='SRC',Al2=Als['src'],R12=R_src_tt) if e2 in ['TT','MV'] else q_nobh_2

# This is the filtered map loading function that RDN0 will use
def get_kmap(seed):
    s_i,s_set,noise_seed = solenspipe.convert_seeds(seed)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)


# Let's make a single map and cross-correlate with input
if rank==0:
    # Get kappa alm
    ikalm = utils.change_alm_lmax(utils.get_kappa_alm(0).astype(np.complex128),mlmax)
    
    # Get data
    Xdat = get_kmap((0,0,0))

    # New convention in falafel means maps are potential; we convert to convergence
    r_nobh_1 = plensing.phi_to_kappa(q_nobh_1(Xdat,Xdat)[0])

    uicls = hp.alm2cl(ikalm,ikalm)
    icls = bin(uicls)
    xcls_nobh = bin(hp.alm2cl(r_nobh_1,ikalm))
    acls_nobh = bin(hp.alm2cl(r_nobh_1,r_nobh_1))

    if bh:
        r_bh_1 = plensing.phi_to_kappa(q_bh_1(Xdat,Xdat)[0])
        xcls_bh = bin(hp.alm2cl(r_bh_1,ikalm))
        acls_bh = bin(hp.alm2cl(r_bh_1,r_bh_1))


powfunc = lambda x,y: hp.alm2cl(x,y)

# RDN0 for lensing potential using the callables defined earlier
# converted to lensing convergence
# For the usual quadratic estimator
rdn0_nobh = np.asarray([bin(x) for x in bias.rdn0(0, get_kmap, powfunc, nsims, 
                                                  q_nobh_1, 
                                                  qfunc2=None if diag else q_nobh_2, 
                                                  comm=comm) * (ells*(ells+1.)/2.)**2.])

if bh:
    # For the bias hardened one; note we use the same RDN0 function,
    # but a different qfunc function
    rdn0_bh = np.asarray([bin(x) for x in bias.rdn0(0, get_kmap, powfunc, nsims, 
                                                    q_bh_1, 
                                                    qfunc2=None if diag else q_bh_2, 
                                                    comm=comm) * (ells*(ells+1.)/2.)**2.])

if rank==0:

    labs = bunch.Bunch()
    labs.clii = r'$C_L^{\phi \phi}$'
    labs.nlg = r'$N_L^{0,\phi\phi}$'
    labs.nlc = r'$N_L^{0,\omega\omega}$'
    labs.tclng = r'$C_L^{\phi \phi} + N_L^{0,\phi\phi}$'
    labs.rdn0g = r'$RDN_L^{0,\phi\phi}$'
    labs.rdn0c = r'$RDN_L^{0,\omega\omega}$'
    labs.tcrdn0g = r'$C_L^{\phi \phi} + RDN_L^{0,\phi\phi}$'
    labs.xcl = r'$C_L^{\hat{\phi} \phi}$'
    labs.acl = r'$C_L^{\hat{\phi} \hat{\phi}}$'


    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
    pl.add(ells[1:],Nl_g[1:],ls='--',label=labs.nlg)
    pl.add(cents,icls+bin(Nl_g),ls='--',label=labs.tclng)
    pl.add(cents,rdn0_nobh[0] ,ls=':',label=labs.rdn0g,marker='d')
    pl.add(cents,icls+(rdn0_nobh[0] ),ls=':',label=labs.tcrdn0g,marker='d')
    pl.add(cents,xcls_nobh,marker='o',label=labs.xcl)
    pl.add(cents,acls_nobh,marker='o',label=labs.acl)
    pl._ax.set_ylim(1e-9,1e-6)
    pl.legend('outside')
    pl.done(f'{solenspipe.opath}rdn0_nobh_{e1}_{e2}.png')

    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(ells[1:],Nl_c[1:],ls='--',label=labs.nlc)
    pl.add(cents,rdn0_nobh[1] ,ls=':',label=labs.rdn0c,marker='d')
    if bh: pl.add(cents,rdn0_bh[1] ,ls=':',label='BH ' + labs.rdn0c,marker='d')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.legend('outside')
    pl.done(f'{solenspipe.opath}rdn0_curl_{e1}_{e2}.png')


    if bh:
        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
        # Convert RDN0 to convergence
        pl.add(cents,rdn0_bh[0] ,ls=':',label='BH ' + labs.rdn0g,marker='d')
        pl.add(cents,icls+(rdn0_bh[0] ),ls=':',label='BH ' + labs.tcrdn0g,marker='d')
        pl.add(ells[1:],Nl_g_bh[1:],ls='--',label='BH ' + labs.nlg)
        pl.add(cents,icls+bin(Nl_g_bh),ls='--',label='BH ' + labs.tclng)
        pl.add(cents,xcls_bh,marker='o',label=labs.xcl)
        pl.add(cents,acls_bh,marker='o',label=labs.acl)
        pl._ax.set_ylim(1e-9,1e-6)
        pl.legend('outside')
        pl.done(f'{solenspipe.opath}rdn0_bh_{e1}_{e2}.png')
