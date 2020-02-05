from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap
import numpy as np
import os,sys
import solenspipe
import mapsims
from enlib import bench


"""
We will do a simple lensing reconstruction test.
No mask.
Full sky CAR.


"""


lmin = 100
lmax = 3000

res = np.deg2rad(2.0 *(3000/lmax) /60.)

shape,wcs = enmap.fullsky_geometry(res=res)
mask = enmap.ones(shape,wcs)
with bench.show("init"):
    solint = solenspipe.SOLensInterface(mask=mask,data_mode=None,scanning_strategy="isotropic",fsky=0.4)
      
seed = (0,0,0)
channel = mapsims.SOChannel("LA", 145)
with bench.show("sim"):
    t,e,b = [solint.get_kmap(channel,X,seed,lmin,lmax,filtered=True) for X in ['T','E','B']]
with bench.show("recon"):
    solint.get_mv_kappa("TT",t,e,b)    
