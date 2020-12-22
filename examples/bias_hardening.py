from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs
import numpy as np
import os,sys
from falafel import qe,utils
import solenspipe
from solenspipe import bias
import pytempura
import healpy as hp

"""
In this example, we show how to:
1. obtain a bias-hardened CMB lensing map
2. obtain its RDN0

We will work with TT only, no beam, no noise.
"""

# Multipole limits and resolution
lmin = 100; lmax = 3000
mlmax = 4000 # for alms
px_arcmin = 2.0

# lmin = 100; lmax = 300
# mlmax = 400 # for alms
# px_arcmin = 20.0

# beam_fwhm = 1.5
# noise_t = 6.0

beam_fwhm = 1.5
noise_t = 0

grad = True

nsims = 2 # number of sims to test RDN0
comm,rank,my_tasks = mpi.distribute(nsims)

# Geometry
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(px_arcmin/60.),proj='car')
px = qe.pixelization(shape,wcs)

# Get CMB Cls for response and total Cls corresponding to a noiseless configuration
ucls,tcls = utils.get_theory_dicts_white_noise(beam_fwhm,noise_t,grad=grad)

# Get norms for lensing potential, sources and cross
Als = pytempura.get_norms(['TT','src','src_x_tt'],ucls,tcls,lmin,lmax,k_ellmax=mlmax)
ls = np.arange(Als['TT'][0].size)

# Convert to noise per mode on lensing convergence
Nl_tt = Als['TT'][0] * (ls*(ls+1.)/2.)**2.
Nl_src = Als['src']  * (ls*(ls+1.)/2.)**2.
Nl_bh = solenspipe.bias_hardened_n0(Als['TT'][0],Als['src'],Als['src_x_tt']) * (ls*(ls+1.)/2.)**2.

# Plot expected noise performance
theory = cosmology.default_theory()
ells = np.arange(Als['TT'][0].size)

if rank==0:
    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(ells[1:],Nl_tt[1:])
    pl.add(ells[1:],Nl_bh[1:],ls='--')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{solenspipe.opath}bh_noise.png')
    
bin_edges = np.geomspace(2,mlmax,15)
binner = stats.bin1D(bin_edges)
bin = lambda x: binner.bin(ells,x)[1]
cents = binner.cents

# These are the qfunc lambda functions we will use with RDN0
q_tt = solenspipe.get_qfunc(px,ucls,mlmax,'TT',Al1=Als['TT'][0],est2=None,Al2=None,R12=None,curl=False)
q_bh = solenspipe.get_qfunc(px,ucls,mlmax,'TT',Al1=Als['TT'][0],est2='SRC',Al2=Als['src'],R12=Als['src_x_tt'],curl=False)

if noise_t>0:
    lbeam = maps.gauss_beam(beam_fwhm,ells)
    ilbeam = lbeam*0
    ilbeam[2:] = 1./lbeam[2:]

# This is the filtered map loading function that RDN0 will use
def get_kmap(seed):
    s_i,s_set,noise_seed = solenspipe.convert_seeds(seed)
    dalm = solenspipe.get_cmb_alm(s_i,s_set)[0]
    if noise_t>0:
        nmap = maps.white_noise(shape,wcs,noise_muK_arcmin=noise_t,seed=noise_seed)
        dalm = cs.almxfl(px.map2alm(px.alm2map(cs.almxfl(dalm,lbeam),spin=0,ncomp=1,mlmax=mlmax)[0] + nmap,lmax=mlmax),ilbeam)
    # Filter isotropically
    tcltt = tcls['TT']
    filt_T = tcltt*0
    filt_T[2:] = 1./tcltt[2:]
    talm = qe.filter_alms(dalm,filt_T,lmin=lmin,lmax=lmax)
    return [talm,0,0]


# Let's make a single map and cross-correlate with input
if rank==0:
    # Get kappa alm
    ikalm = utils.change_alm_lmax(utils.get_kappa_alm(0).astype(np.complex128),mlmax)
    
    # Get data
    Xdat = get_kmap((0,0,0))

    # New convention in falafel means maps are potential; we convert to convergence
    r_tt = plensing.phi_to_kappa(q_tt(Xdat,Xdat))
    r_bh = plensing.phi_to_kappa(q_bh(Xdat,Xdat))

    uicls = hp.alm2cl(ikalm,ikalm)
    icls = bin(uicls)
    xcls_tt = bin(hp.alm2cl(r_tt,ikalm))
    xcls_bh = bin(hp.alm2cl(r_bh,ikalm))

    acls_tt = bin(hp.alm2cl(r_tt,r_tt))
    acls_bh = bin(hp.alm2cl(r_bh,r_bh))


# RDN0 for lensing potential using the callables defined earlier
# converted to lensing convergence
# For the usual quadratic estimator
rdn0_tt = bin(bias.rdn0(0, q_tt,q_tt, get_kmap, comm,
                lambda x,y: hp.alm2cl(x,y), nsims) * (ells*(ells+1.)/2.)**2.)
# For the bias hardened one; note we use the same RDN0 function,
# but a different qfunc function
rdn0_bh = bin(bias.rdn0(0, q_bh,q_bh, get_kmap, comm,
                        lambda x,y: hp.alm2cl(x,y), nsims) * (ells*(ells+1.)/2.)**2.)

if rank==0:
    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(cents,icls,color='k',marker='x',ls='none')
    pl.add(ells[1:],Nl_tt[1:],ls='--')
    pl.add(cents,icls+bin(Nl_tt),ls='--')
    # Convert RDN0 to convergence
    pl.add(cents,rdn0_tt ,ls=':')
    pl.add(cents,icls+(rdn0_tt ),ls=':')
    pl.add(cents,xcls_tt,marker='o')
    pl.add(cents,acls_tt,marker='o')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{solenspipe.opath}nobh.png')

    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ells[2:],theory.gCl('kk',ells)[2:],color='k')
    pl.add(cents,icls,color='k',marker='x',ls='none')
    # Convert RDN0 to convergence
    pl.add(cents,rdn0_bh ,ls=':')
    pl.add(cents,icls+(rdn0_bh ),ls=':')
    pl.add(ells[1:],Nl_bh[1:],ls='--')
    pl.add(cents,icls+bin(Nl_bh),ls='--')
    pl.add(cents,xcls_bh,marker='o')
    pl.add(cents,acls_bh,marker='o')
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{solenspipe.opath}bh.png')
