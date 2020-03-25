
from __future__ import print_function
from orphics import stats,io,mpi,maps
from pixell import enmap,lensing as plensing,curvedsky as cs,reproject
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
lmin = 100
lmax = 3000

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


#empty map showing the shape and geometry of the system
omap = enmap.zeros((2,)+shape[-2:],wcs)

#Calculation for a given L
L=100
seed = (0,0,0)
ells=np.arange(0,solint.mlmax)

#calculate real space map of T_bar
t,e,b = solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
print(t.shape,e.shape,b.shape)

rmapT=qe.alm2map_spin(np.stack((t,t)),0,0,omap)
print(rmapT)
#rmapT=solint.alm2map(np.stack((t,t))) how do you use alm2map?

#t_alm,e_alm,b_alm=solint.get_kmap(channel,seed,lmin,lmax)

# filtered field multiplied by 1/l need to find a way to multiply by dC_l/dl. This is spin 0

t_alm=solint.prepare_shear_map(channel,seed,lmin,lmax)
#find spin +2 and -2 transform of t_alm
ells=np.arange(0,lmax)
#multiply by sqrt(ls)
alms=qe.almxfl(np.stack((t_alm,t_alm)) ,np.sqrt((ells-1.)*ells*(ells+1.)*(ells+2.)))
print(np.shape(alms))


# return real space spin pm 2 components
rmap=qe.alm2map_spin(alms,0,2,omap)

prodmap=rmap*rmapT
realsp2=prodmap[0] #spin +2 real space
realsm2=prodmap[0] #spin -2 real space



res1 = cs.map2alm(enmap.enmap(-qe.irot2d(np.stack((realsp2,realsp2.conj())),spin=0).real,omap.wcs),spin=2,lmax=solint.mlmax)
res2= cs.map2alm(enmap.enmap(-qe.irot2d(np.stack((realsm2,realsm2.conj())),spin=0).real,omap.wcs),spin=2,lmax=solint.mlmax)

#spin 2 ylm 
ttalmsp2=res1[0]
ttalmsm2=res2[1]

#multiply by sqrt(L(L+1) as a way of multiplying by exponential
qe.almxfl(np.stack((ttalmsm2,ttalmsm2)) ,np.sqrt((ells-1.)*ells*(ells+1.)*(ells+2.)))+qe.almxfl(np.stack((ttalmsp2,ttalmsp2)) ,np.sqrt((ells-1.)*ells*(ells+1.)*(ells+2.)))

"""
#find the exponential function at a given L in real space
modrmap = enmap.modrmap(omap.shape,omap.wcs)
exp=np.cos(modrmap* L)-1j*np.sin(modrmap*L)
realp=solint.map2alm(np.cos(modrmap* L))
imp=1j*solint.map2alm(np.sin(modrmap* L))
ealm=realp-imp
factor=0.25*np.sqrt((L-1)*L*(L+1)*(L+2))
espm2=factor*ealm
#spin pm2 of the exponential
np.stack((espm2,espm2))
"""




#print(rmap*realT)



#print(type(t_almtest))
#a=qe.gradient_spin(shape,wcs,np.stack((t_almtest,t_almtest)),solint.mlmax,spin=0)
#a=qe.qe_temperature_only(shape,wcs,t_almtest,t_almtest,solint.mlmax)




#qe.alm2map_spin(alms,0,2,omap)


#find spin two transform of exponential
#treat the exponential as a filter, find the spin pm 2 components of it and multiply it by the other terms



#example of alm2map
"""
cmb_alm = solenspipe.get_cmb_alm(0,0).astype(np.complex128)
print(np.shape(cmb_alm))
a=solint.alm2map(cmb_alm)
print(a)
"""

