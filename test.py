from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap
import numpy as np
import os,sys
import sims, bias
import symlens

class Prepare(object):
    def __init__(self,nsims):
        shape,wcs = maps.rect_geometry(width_deg = 20.,px_res_arcmin=2.0)
        beam_arcmin = 1.5
        noise_uk_arcmin = 1.5
        theory = cosmology.default_theory()
        self.flsims = sims.FlatLensingSims(shape,wcs,theory,beam_arcmin,noise_uk_arcmin)
        self.theory = theory
        self.nsims = nsims
        self.fc = maps.FourierCalc(shape,wcs)
        
        
    def get_prepared_kmap(self,X,seed):
        icov,iset,i = seed
        if i==0:
            iset = 0
            icov = 0
        if iset==0 or iset==1:
            seed_cmb = (icov,iset,i)+(0,)
            seed_kappa = (icov,iset,i)+(1,)
            seed_noise = (icov,iset,i)+(2,)
        elif iset==2 or iset==3:
            seed_cmb = (icov,iset,i)+(0,)
            seed_kappa = (icov,2,i)+(1,)
            seed_noise = (icov,iset,i)+(2,)
        print(icov,iset,i)
        observed = self.flsims.get_sim(seed_cmb,seed_kappa,seed_noise,lens_order=5,return_intermediate=False)
        _,kmap,_ = self.fc.power2d(observed)
        return kmap/self.flsims.kbeam
        
nsims = 20
njobs = nsims
comm,rank,my_tasks = mpi.distribute(njobs)
sobj = Prepare(nsims=nsims)
shape,wcs= sobj.flsims.shape,sobj.flsims.wcs
modlmap = enmap.modlmap(shape,wcs)
tmask = maps.mask_kspace(shape,wcs,lmin=100,lmax=3000)
feed_dict = {}
feed_dict['tC_T_T'] = sobj.theory.lCl('TT',modlmap) + sobj.flsims.ps_noise[0,0]/sobj.flsims.kbeam**2.
feed_dict['uC_T_T'] = sobj.theory.lCl('TT',modlmap)
qe = symlens.QE(shape,wcs,feed_dict,estimator="hu_ok",XY="TT",
                xmask=tmask,ymask=tmask)
def qfunc(dummy,x,y):
    feed_dict['X'] = x
    feed_dict['Y'] = y
    return qe.reconstruct(feed_dict,xname='X_l1',yname='Y_l2')

icov = 0
alpha = "TT"
beta = "TT"
power =  lambda x,y : sobj.fc.f2power(x,y)
#n1 = bias.mcn1(icov,alpha,beta,qfunc,sobj,comm,power)
n1 = bias.rdn0(icov,alpha,beta,qfunc,sobj,comm,power)

bin_edges = np.arange(100,3000,40)
binner = stats.bin2D(modlmap,bin_edges)
cents,n11d = binner.bin(n1)

ells = np.arange(2,3000,1)
clkk = sobj.theory.gCl('kk',ells)
pl = io.Plotter(xyscale='linlin',scalefn = lambda x: x)
pl.add(ells,clkk,lw=3,color='k')
pl.add(cents,n11d)
pl.hline(y=0)
pl.done("n1.png")
