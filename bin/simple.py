from __future__ import print_function
from pixell import enmap
import numpy as np
import os,sys
import solenspipe
import mapsims


"""
We will do a simple lensing reconstruction test.
No mask.
"""

# Lensing reconstruction ell range
lmin = 100
lmax = 3000

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
      
# Choose a seed. This has to be varied when simulating.
seed = (0,0,0)

# Choose the frequency channel
channel = mapsims.SOChannel("LA", 145)

# Get the simulated, prepared T, E, B maps
t,e,b = [solint.get_kmap(channel,X,seed,lmin,lmax,filtered=True) for X in ['T','E','B']]

# Get the reconstructed map for the TT estimator
recon = solint.get_mv_kappa("TT",t,e,b)    
