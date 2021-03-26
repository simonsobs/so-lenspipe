#Run ksz estimator on lensing sims

#Import stuff
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

lmin=100
lmax=2000
mlmax=3000

#Read in sim alms
alm = solenspipe.get_cmb_alm(0,0)

#Filter with with W_S,i from eqn. 1 of 1803.07036
def cl_kszksz(ell):
    #From fig 1
    #l(l+1)/2pi Cl \approx 1.5
    return 2*np.pi * 1.5 / ell / (ell+1)

ucls,tcls = utils.get_theory_dicts(grad=False)
ells = np.arange(len(tcls["TT"]))
tcls["TT"] /= (cl_kszksz(ells))**0.5

filtered_alm = utils.isotropic_filter(alm,tcls,lmin,lmax,ignore_te=True)

#Get the point-source normalization from pytempura,
#using filter from above
Al = pytempura.get_norms(
    ['src'], ucls, tcls, lmin, lmax, k_ellmax=mlmax)

#Get the "qfunc" 
#run on the filtered maps.
#This is just the falafel
#point-sources estimator
#optionally provide normalization as norm
def qfunc_ksz(px, mlmax, norm=None):
    qfunc =  lambda X,Y: qe.qe_pointsources(
        px, mlmax, fTalm=Y[0], xfTalm=X[0])
    if norm is None:
        return qfunc
    else:
        def retfunc(X,Y):
            recon = qfunc(X,Y)
            return np.asarray(cs.almxfl(recon, norm))
        return retfunc

px_arcmin = 2.0  / (lmax / 3000)
shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(px_arcmin/60.),proj='car')
px = qe.pixelization(shape=shape,wcs=wcs)

qfunc = qfunc_ksz(px, mlmax, norm=Al['src'])
ksz_est = qfunc(filtered_alm, filtered_alm)

cl = cs.alm2cl(ksz_est,ksz_est)

plt.plot(np.arange(len(cl)), cl)
plt.savefig("cl.png")
