from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs
import numpy as np
import os,sys
from falafel import qe,utils
import solenspipe
from enlib import bench
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
parser.add_argument("--nsims-n0",     type=int,  default=40,help="Number of sims.")
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

def four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0=None,Xdatp_1=None,Xdatp_2=None,Xdatp_3=None,q_func1=None,qfunc2=None):
    """
    return the phi_alms used for the 4 split estimator
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

        #phi_xy10=plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_0))
        phi_xy12= plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_2))
        phi_xy13= plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_3))
        #phi_xy20=plensing.phi_to_kappa(q_bh_1(Xdatp_2,Xdatp_0))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        #phi_xy21=plensing.phi_to_kappa(q_bh_1(Xdatp_2,Xdatp_1))
        phi_xy23=plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_3))
        #phi_xy30=plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_0))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        #phi_xy31=plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_1))
        #phi_xy32=plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_2))
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

# Geometry
px_arcmin = 2.0  / (args.lmax / 3000)
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(px_arcmin/60.),proj='car')
nside = None
print(f"shape,wcs: {shape}, {wcs}")
px = qe.pixelization(shape=shape,wcs=wcs,nside=nside)

# Get CMB Cls for response and total Cls corresponding to a noiseless configuration



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

with bench.show("INIT"):
    ls,nells,nells=get_noise_power(10)
    noise={'TT':nells,'EE':nells,'BB':nells}
    ucls,tcls = utils.get_theory_dicts(nells=noise,lmax=mlmax,grad=True)
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

    bin_edges = np.geomspace(2,mlmax,15)
    binner = stats.bin1D(bin_edges)
    bin = lambda x: binner.bin(ells,x)[1]
    cents = binner.cents

    # These are the qfunc lambda functions we will use with RDN0 and MCN1
    q_nobh_1 = solenspipe.get_qfunc(px,ucls,mlmax,e1,Al1=Als[e1],est2=None,Al2=None,R12=None)
    q_nobh_2 = solenspipe.get_qfunc(px,ucls,mlmax,e2,Al1=Als[e2],est2=None,Al2=None,R12=None)



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








with bench.show("prep"):
    q1 = q_nobh_1
    q2 = q_nobh_1 

    powfunc = lambda x,y: split_phi_to_cl(x,y)

    phifunc=lambda Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0,Xdatp_1,Xdatp_2,Xdatp_3,qf:four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0,Xdatp_1,Xdatp_2,Xdatp_3,qf)

    bias_nobh = {}
    bias_bh = {}




# Get data

with bench.show("data"):
    Xdat0 = get_kmap((0,0,0)) #TODO: fix hardcoding
    Xdat1= get_kmap1((0,0,0)) #TODO: fix hardcoding
    Xdat2 = get_kmap2((0,0,0)) #TODO: fix hardcoding
    Xdat3= get_kmap3((0,0,0)) #TODO: fix hardcoding

# RDN0 or MCN1 for lensing potential using the callables defined earlier
# converted to lensing convergence


with bench.show("rd"):
    rdn0,mcn0= bias.mcrdn0_s4(0, get_kmap, powfunc,phifunc, nsims_n0, 
                            q1, get_kmap1=get_kmap1,get_kmap2=get_kmap2,get_kmap3=get_kmap3,
                            qfunc2=None,
                            Xdat=Xdat0,Xdat1=Xdat1,Xdat2=Xdat2,Xdat3=Xdat3,
                            use_mpi=True, skip_rd=True) 



    #rdn0 = rdn0 * (ells*(ells+1.)/2.)**2.
    mcn0 = mcn0 * (ells*(ells+1.)/2.)**2.
    bhlab='nobh'
    if rank==0: 
        #np.save(f'{opath}rdn0_{bhlab}_{e1}_{e2}no.npy',rdn0)
        np.save(f'{opath}mcn0_{bhlab}_{e1}_{e2}no.npy',mcn0)


