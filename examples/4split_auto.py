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
#Xdatp_0=None,Xdatp_1=None,Xdatp_2=None,Xdatp_3=None,
def four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,q_func1,Xdatp_0=None,Xdatp_1=None,Xdatp_2=None,Xdatp_3=None,qfunc2=None):
    """Return kappa_alms combinations required for the 4cross estimator.

    Args:
        Xdat_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0
        Xdat_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1
        Xdat_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2
        Xdat_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3
        q_func1 (function): function for quadratic estimator
        Xdatp_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0 used for RDN0 for different sim data combination
        Xdatp_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1 used for RDN0 for different sim data combination
        Xdatp_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2 used for RDN0 for different sim data combination
        Xdatp_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3 used for RDN0 for different sim data combination
        qfunc2 ([type], optional): [description]. Defaults to None.

    Returns:
        array: Combination of reconstructed kappa alms
    """
    q_bh_1=q_func1
    if Xdatp_0 is None:
        
        phi_xy00 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_0))
        phi_xy11 = plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_1))
        phi_xy22 = plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_2))
        phi_xy33 = plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_3))
        phi_xy01 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_1))
        phi_xy02 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_2))
        phi_xy03 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_3))
        phi_xy10=phi_xy01
        phi_xy12= plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_2))
        phi_xy13= plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_3))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_3))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4
    
    else:

        phi_xy00 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_0))
        phi_xy11 = plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_1))
        phi_xy22 = plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_2))
        phi_xy33 = plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_3))
        phi_xy01 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_1))
        phi_xy02 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_2))
        phi_xy03 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_3))
        phi_xy10=phi_xy01

        phi_xy12= plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_2))
        phi_xy13= plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_3))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_3))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4

    phi_xy=np.array([phi_xy_X,phi_xy01,phi_xy02,phi_xy03,phi_xy12,phi_xy13,phi_xy23,phi_xy_x0,phi_xy_x1,phi_xy_x2,phi_xy_x3])
    

    return phi_xy

def split_phi_to_cl(xy,uv,m=4,cross=False,ikalm=None):
    """Generate 4cross Clkk

    Args:
        xy (array): array of kappa_alms obtained from four_split_phi
        uv (array): array of kappa_alms obtained from four_split_phi
        m (int, optional): [description]. Defaults to 4.
        cross (bool, optional): Compute the cross spectrum between auto and input. Defaults to False.
        ikalm (array, optional): Used if cross is True, input theory kappa alms. Defaults to None.

    Returns:
        [array]: reconstructed clkk
    """
    phi_x=xy[0];phi01=xy[1];phi02=xy[2];phi03=xy[3];phi12=xy[4];phi13=xy[5];phi23=xy[6];phi_x0=xy[7];phi_x1=xy[8];phi_x2=xy[9];phi_x3=xy[10]
    phi_xp=uv[0];phi01p=uv[1];phi02p=uv[2];phi03p=uv[3];phi12p=uv[4];phi13p=uv[5];phi23p=uv[6];phi_x0p=uv[7];phi_x1p=uv[8];phi_x2p=uv[9];phi_x3p=uv[10]
    if cross is False:
        tg1=m**4*cs.alm2cl(phi_x,phi_xp)
        tg2=-4*m**2*(cs.alm2cl(phi_x0,phi_x0p)+cs.alm2cl(phi_x1,phi_x1p)+cs.alm2cl(phi_x2,phi_x2p)+cs.alm2cl(phi_x3,phi_x3p))
        tg3=m*(cs.alm2cl(phi01,phi01p)+cs.alm2cl(phi02,phi02p)+cs.alm2cl(phi03,phi03p)+cs.alm2cl(phi12,phi12p)+cs.alm2cl(phi13,phi13p)+cs.alm2cl(phi23,phi23p))
    else:
        tg1=m**4*cs.alm2cl(phi_x,ikalm)
        tg2=-4*m**2*(cs.alm2cl(phi_x0,ikalm)+cs.alm2cl(phi_x1,ikalm)+cs.alm2cl(phi_x2,ikalm)+cs.alm2cl(phi_x3,ikalm))
        tg3=m*(cs.alm2cl(phi01,ikalm)+cs.alm2cl(phi02,ikalm)+cs.alm2cl(phi03,ikalm)+cs.alm2cl(phi12,ikalm)+cs.alm2cl(phi13,ikalm)+cs.alm2cl(phi23,ikalm))


    auto =(1/(m*(m-1)*(m-2)*(m-3)))*(tg1+tg2+tg3)
    return auto

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

    def get_noise_power(wnoise):
        ls = np.arange(mlmax+1)
        nells = ls*0 +(wnoise*np.pi/180./60.)**2.
        nells_P = nells * 2.
        return ls,nells,nells_P

    def rand_map(power,seed):
        noise= maps.white_noise((3,)+shape,wcs,power,seed=seed)
        print(noise.shape)
        return noise

    def get_noise_map(wnoise,noise_seed):
        ls,nells,nells_P = get_noise_power(wnoise)
        noise_map = rand_map(wnoise,noise_seed)
        return noise_map


    # Get CMB Cls for response and total Cls corresponding to a noiseless configuration
    ls,nells,nells=get_noise_power(10)
    noise={'TT':nells,'EE':nells,'BB':nells}
    ucls,tcls = utils.get_theory_dicts(nells=noise,lmax=mlmax,grad=True)

with bench.show("NORM"):
    # Get norms for lensing potential, sources and cross
    est_norm_list = [args.est1]
    if args.est2!=args.est1:
        est_norm_list.append(args.est2)
    bh = False
    for e in est_norm_list:
        if e.upper()=='TT' or e.upper()=='MV':
            bh = False
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
    noise_map = get_noise_map(10,noise_seed)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    signal=cs.alm2map(dalm,enmap.empty((3,)+shape,wcs))
    total=signal+noise_map
    dalm = cs.map2alm(total,lmax=mlmax)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

def get_kmap1(seed):
    s_i,s_set,_ = solenspipe.convert_seeds(seed)
    noise_seed=np.array(seed)
    noise_seed[2]=noise_seed[2]+200
    noise_seed=tuple(noise_seed)
    _,_,ns = solenspipe.convert_seeds(noise_seed)
    noise_map = get_noise_map(10,ns)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    signal=cs.alm2map(dalm,enmap.empty((3,)+shape,wcs))
    total=signal+noise_map
    dalm = cs.map2alm(total,lmax=mlmax)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

def get_kmap2(seed):
    s_i,s_set,_ = solenspipe.convert_seeds(seed)
    noise_seed=np.array(seed)
    noise_seed[2]=noise_seed[2]+400
    noise_seed=tuple(noise_seed)
    _,_,ns = solenspipe.convert_seeds(noise_seed)
    noise_map = get_noise_map(10,ns)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    signal=cs.alm2map(dalm,enmap.empty((3,)+shape,wcs))
    total=signal+noise_map
    dalm = cs.map2alm(total,lmax=mlmax)
    return utils.isotropic_filter(dalm,tcls,lmin,lmax,ignore_te=True)

def get_kmap3(seed):
    s_i,s_set,_ = solenspipe.convert_seeds(seed)
    noise_seed=np.array(seed)
    noise_seed[2]=noise_seed[2]+600
    noise_seed=tuple(noise_seed)
    _,_,ns = solenspipe.convert_seeds(noise_seed)
    noise_map = get_noise_map(10,ns)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)
    signal=cs.alm2map(dalm,enmap.empty((3,)+shape,wcs))
    total=signal+noise_map
    dalm = cs.map2alm(total,lmax=mlmax)
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
        s_i,s_set,noise_seed = solenspipe.convert_seeds((0,0,0))
        Xdat_0 = get_kmap((0,0,task))
        Xdat_1 = get_kmap1((0,0,task))
        Xdat_2 = get_kmap2((0,0,task))
        Xdat_3 = get_kmap3((0,0,task))

        powfunc = lambda x,y,m,cross,ikalm: split_phi_to_cl(x,y,m,cross,ikalm)
        phifunc=lambda Xdat_0,Xdat_1,Xdat_2,Xdat_3,qf: four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,qf)
    with bench.show("alm2cl and append"):
        uicl = cs.alm2cl(ikalm,ikalm)
        #xy=phifunc(Xdat_0,Xdat_1,Xdat_2,Xdat_3,q_nobh_1,Xdat_0,Xdat_1,Xdat_2,Xdat_3)
        xy=phifunc(Xdat_0,Xdat_1,Xdat_2,Xdat_3,q_nobh_1)
        xy_a=[]
        for i in range(len(xy)):
            xy_a.append(xy[i][0])
        xy_a=np.array(xy_a)
        uacl_nobh=powfunc(xy_a,xy_a,m=4,cross=False,ikalm=None)
        uxcl_nobh=powfunc(xy_a,xy_a,m=4,cross=True,ikalm=ikalm)
        uxcls_nobh_1.append(uxcl_nobh)
        uacls_nobh.append(uacl_nobh)
        uicls.append(uicl)

        if bh:
            xy,uv=four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,q_bh_1,qfunc2=None)
            xy_a=[]
            for i in range(len(xy)):
                xy_a.append(xy[i][0])
            xy_a=np.array(xy_a)
            uacl_bh=split_phi_to_cl(xy_a,xy_a,m=4,cross=False,ikalm=None)
            uxcl_bh=split_phi_to_cl(xy_a,xy_a,m=4,cross=True,ikalm=ikalm)
            uxcls_bh_1.append(uxcl_bh)
            uacls_bh.append(uacl_bh)



with bench.show("MPI Gather"):
    uicls = putils.allgatherv(uicls,comm)
    uxcls_nobh_1 = putils.allgatherv(uxcls_nobh_1,comm)
    uxcls_nobh_2 = putils.allgatherv(uxcls_nobh_2,comm)
    uacls_nobh = putils.allgatherv(uacls_nobh,comm)
    if bh:
        uxcls_bh_1 = putils.allgatherv(uxcls_bh_1,comm)
        uacls_bh = putils.allgatherv(uacls_bh,comm)

if rank==0:


    with bench.show("Labels"):
        labs = solenspipe.get_labels()

    with bench.show("Stats"):
        suicls = stats.get_stats(uicls)
        suxcls_nobh_1 = stats.get_stats(uxcls_nobh_1)
        suacls_nobh = stats.get_stats(uacls_nobh)
        if bh:
            suxcls_bh_1 = stats.get_stats(uxcls_bh_1)
            suacls_bh = stats.get_stats(uacls_bh)

    with bench.show("Save"):
        np.save(f'{solenspipe.opath}mean_uicls_{e1}_{e2}.npy',uicls)
        np.save(f'{solenspipe.opath}mean_uacls_nobh_{e1}_{e2}.npy',uacls_nobh)
        np.save(f'{solenspipe.opath}mean_uxcls_nobh_{e1}_{e2}.npy',uxcls_nobh_1)

        if bh:
            np.save(f'{solenspipe.opath}mean_uxcls_bh_1_{e1}_{e2}.npy',uxcls_bh_1)
            np.save(f'{solenspipe.opath}mean_uacls_bh_{e1}_{e2}.npy',uacls_bh)


    with bench.show("Bin"):
        uicls = suicls['mean']
        uxcls_nobh_1 = suxcls_nobh_1['mean']
        uacls_nobh = suacls_nobh['mean']
        if bh:
            uxcls_bh_1 = suxcls_bh_1['mean']
            uacls_bh = suacls_bh['mean']


        icls = bin(uicls)
        xcls_nobh_1 = bin(uxcls_nobh_1)
        acls_nobh = bin(uacls_nobh)
        if bh:
            xcls_bh_1 = bin(uxcls_bh_1)
            acls_bh = bin(uacls_bh)


    with bench.show("Plot"):
        pl = io.Plotter('CL',xyscale='loglog')
        pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
        pl.add(cents,icls,color='k',marker='x',ls='none',label=labs.clii)
        pl.add(ells[1:],Nl_g[1:],ls='--',label=labs.nlg)
        pl.add(cents,icls+bin(Nl_g),ls='--',label=labs.tclng)
        pl.add(cents,xcls_nobh_1,marker='o',label=labs.xcl)
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
            pl.add(cents,acls_bh,marker='o',label=labs.acl)
            pl._ax.set_ylim(1e-10,1e-6)
            pl.legend('outside')
            pl.done(f'{solenspipe.opath}cross_verify_bh_{e1}_{e2}.png')