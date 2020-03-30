from __future__ import print_function
from orphics import stats,io,mpi,maps
from pixell import enmap
import numpy as np
import os,sys
import solenspipe
from falafel import qe
import mapsims
from enlib import bench
import healpy as hp

"""
We will do a simple lensing reconstruction test.
No mask.
"""

# Lensing reconstruction ell range
lmin = 2
lmax = 5000

polcomb = 'TT'

# Number of sims
nsims = 1

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

s = stats.Stats(comm)



for task in my_tasks:

    # Choose a seed. This has to be varied when simulating.
    seed = (0,0,task)

    # Get the simulated, prepared T, E, B maps
    t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
    print(t_alm.shape,e_alm.shape,b_alm.shape)
    # Get the reconstructed map for the TT estimator
    recon_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ls,Als[polcomb]))
    
    kalms = solint.get_kappa_alm(task)
    xcl = hp.alm2cl(recon_alms,kalms)
    icl = hp.alm2cl(kalms,kalms)
    s.add_to_stats('xcl',xcl)
    s.add_to_stats('icl',icl)


s.get_stats()

if rank==0:
    xcl = s.stats['xcl']['mean']
    icl = s.stats['icl']['mean']
    ls = np.arange(xcl.size)
    pl = io.Plotter('CL')
    pl.add(ls,xcl)
    pl.add(ls,icl)
    pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{solenspipe.opath}/recon.png')
                                                                                   

                                                                                

# Cross correlate with input

# Calculate autospectrum
# Subtract biases
