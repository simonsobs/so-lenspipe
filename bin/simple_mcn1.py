from __future__ import print_function
from orphics import stats,io,mpi,maps,cosmology
from pixell import enmap
import numpy as np
import os,sys
import solenspipe
from falafel import qe
import mapsims
from enlib import bench
import healpy as hp
from solenspipe import bias

"""
We will do a simple lensing reconstruction test.
No mask.
"""

# Lensing reconstruction ell range
lmin = 100
lmax = 3000

polcomb = 'EE'

# Number of sims
nsims = 100

# CAR resolution is decided based on lmax
res = np.deg2rad(2.0 *(3000/lmax) /60.)

# Make the full sky geometry
shape,wcs = enmap.fullsky_geometry(res=res)

# We use a mask of ones for this test
mask = enmap.ones(shape,wcs)

# Initialize the lens simulation interface
solint = solenspipe.SOLensInterface(mask=mask,data_mode=None,scanning_strategy="isotropic",fsky=0.4)
      

# Choose the frequency channel
channel = mapsims.SOChannel("LA", 145)

# norm dict
Als = {}
ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv = solenspipe.initialize_norm(solint,channel,lmin,lmax)
Als['mv'] = al_mv
Als['mvpol'] = al_mv_pol
al_mv = Als[polcomb]

comm,rank,my_tasks = mpi.distribute(nsims)

get_kmap = lambda seed: solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
power = lambda x,y: hp.alm2cl(x,y)
qfunc = solint.qfunc

assert ls[0]==0
assert len(ls) == Als[polcomb].size
nmax = len(ls)

mcn1 = bias.mcn1(0,polcomb,polcomb,qfunc,get_kmap,comm,power,nsims,verbose=True)

mcn1[:nmax] = mcn1[:nmax] * Als[polcomb]**2.  #why not multiply by 0.25
mcn1[nmax:] = 0
io.save_cols(f'{solenspipe.opath}/n1_ee.txt',(ls,mcn1[:nmax]))


if rank==0:
    theory = cosmology.default_theory()
    
    ls = np.arange(rdn0.size)
    pl = io.Plotter('CL')
    pl.add(ls,mcn1)
    pl.add(ls,theory.gCl('kk',ls))
    #pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{solenspipe.opath}/recon_mcn1.png')
                                                                                   

                                                                                

