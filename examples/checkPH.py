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
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="Number of sims.")
parser.add_argument( "--healpix", action='store_true',help='Use healpix instead of CAR.')
parser.add_argument( "--lmax",     type=int,  default=3000,help="Maximum multipole for lensing.")
parser.add_argument( "--lmin",     type=int,  default=100,help="Minimum multipole for lensing.")
args = parser.parse_args()

spath='/home/r/rbond/jiaqu/scratch/so_lens/shear/'

profile=np.loadtxt(f"{spath}tsz_profile.txt")

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
    est_norm_list = ['TT','src']

    #normalization for TT
    Als = pytempura.get_norms(est_norm_list,ucls,tcls,lmin,lmax,k_ellmax=mlmax)

    #tcls_norm["TT"][0:3000] /= (profile**2)[0:3000]
    tcls["TT"][0:3000] /= profile[0:3000]
    

with bench.show("NORM"):
    # Get response for TSZ profile
    R_src_tt = pytempura.get_cross('SRC','TT',ucls,tcls,lmin,lmax,k_ellmax=mlmax)/=profile[0:3000]
    #normalization filter for tsz has an extra profile function
    tcls["TT"][0:3000] /= profile[0:3000]
    Als_src = pytempura.get_norms(est_norm_list,ucls,tcls,lmin,lmax,k_ellmax=mlmax)
    Als_src['src'][0:3000]*=(profile**2)[0:3000]
    ls = np.arange(Als['TT'][0].size)

    # Convert to noise per mode on lensing convergence
    e1 = 'TT'
    e2=e1
    Nl_g = Als[e1][0] * (ls*(ls+1.)/2.)**2.
    Nl_c = Als[e1][1] * (ls*(ls+1.)/2.)**2.
    Nl_g_bh = solenspipe.bias_hardened_n0(Als[e1][0],Als_src['src'],R_src_tt) * (ls*(ls+1.)/2.)**2.
  
    if rank==0:
        np.savetxt(f'{solenspipe.opath}Nlg_{e1}_{e2}.txt',Nl_g)
        np.savetxt(f'{solenspipe.opath}Nlc_{e1}_{e2}.txt',Nl_c)
        np.savetxt(f'{solenspipe.opath}Nlg_bh_{e1}_{e2}.txt',Nl_g_bh)


theory = cosmology.default_theory()
ells = ls

bin_edges = np.geomspace(2,mlmax,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)[1]
cents = binner.cents




with bench.show("QFUNC"):
    # These are the qfunc lambda functions we will use with RDN0 and MCN1

    q_bh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2='PH',Al2=Als_src['src'],R12=R_src_tt,profile=profile) 
    q_bh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2='PH',Al2=Als_src['src'],R12=R_src_tt,profile=profile) 


#input maps are still normal inverse variance filtered maps
tcls["TT"][0:3000] *= (profile**2)[0:3000]
def get_kmap(seed):
    s_i,s_set,noise_seed = solenspipe.convert_seeds(seed)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

uicls = []

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


    with bench.show("alm2cl and append"):
        #tsz profile reconstruction
        qfunc_tsz = lambda X,Y: qe.qe_source(px,mlmax,profile,fTalm=Y[0],xfTalm=X[0])
        tsz_reconalm=qfunc_tsz(Xdat,Xdat)
        tsz_reconalm=cs.almxfl(tsz_reconalm,Als_src['src'])
        tsz_cl = cs.alm2cl(tsz_reconalm,tsz_reconalm)
        np.savetxt(f'{solenspipe.opath}tsz_cl',tsz_cl)
        uicl = cs.alm2cl(ikalm,ikalm)
        uicls.append(uicl)
        r_bh_1 = plensing.phi_to_kappa(q_bh_1(Xdat,Xdat))
        uxcl_bh_1 = cs.alm2cl(r_bh_1[0],ikalm)
        uacl_bh = cs.alm2cl(r_bh_1[0],r_bh_1[0])
        uxcls_bh_1.append(uxcl_bh_1)
        uacls_bh.append(uacl_bh)



with bench.show("MPI Gather"):
    uicls = putils.allgatherv(uicls,comm)
    uxcls_bh_1 = putils.allgatherv(uxcls_bh_1,comm)
    uacls_bh = putils.allgatherv(uacls_bh,comm)

if rank==0:


    with bench.show("Labels"):
        labs = solenspipe.get_labels()

    with bench.show("Stats"):
        suicls = stats.get_stats(uicls)

        suxcls_bh_1 = stats.get_stats(uxcls_bh_1)
        suacls_bh = stats.get_stats(uacls_bh)

    with bench.show("Save"):
        np.save(f'{solenspipe.opath}mean_uicls_{e1}_{e2}.npy',uicls)
        np.save(f'{solenspipe.opath}mean_uxcls_bh_1_{e1}_{e2}.npy',uxcls_bh_1)
        np.save(f'{solenspipe.opath}mean_uacls_bh_{e1}_{e2}.npy',uacls_bh)


    with bench.show("Bin"):
        uicls = suicls['mean']
        uxcls_bh_1 = suxcls_bh_1['mean']
        uacls_bh = suacls_bh['mean']

        np.savetxt(f'{solenspipe.opath}mean_uicls_{e1}_{e2}dif.txt',uicls)
        np.savetxt(f'{solenspipe.opath}mean_uxcls_bh_1_{e1}_{e2}dif.txt',uxcls_bh_1)
        np.savetxt(f'{solenspipe.opath}mean_uacls_bh_{e1}_{e2}dif.txt',uacls_bh)
        np.savetxt(f'{solenspipe.opath}mean_noise{e1}_{e2}dif.txt',Nl_g_bh)


        icls = bin(uicls)

        
        xcls_bh_1 = bin(uxcls_bh_1)
        acls_bh = bin(uacls_bh)

    print(xcls_bh_1)
    with bench.show("Plot"):

        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
        pl.add(ells[1:],Nl_g_bh[1:],ls='--',label='BH ' + labs.nlg)
        pl.add(cents,icls+bin(Nl_g_bh),ls='--',label='BH ' + labs.tclng)
        pl.add(cents,xcls_bh_1,marker='o',label=labs.xcl)
        pl.add(cents,acls_bh,marker='o',label=labs.acl)
        pl._ax.set_ylim(1e-10,1e-2)
        pl.legend('outside')
        pl.done(f'{solenspipe.opath}cross_verify_phnew_{e1}_{e2}.png')


