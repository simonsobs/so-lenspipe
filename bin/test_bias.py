from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap
import numpy as np
import os,sys
import solenspipe
from enlib import bench
from falafel import qe
import healpy as hp
from solenspipe import bias

lmin = 300
lmax = 3000
mlmax = lmax + 500
polcomb = 'TT'
nside = 2048

# lmin = 300
# lmax = 1500
# mlmax = lmax + 500
# polcomb = 'TT'
# nside = 1024

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']
thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

def get_mv_kappa(polcomb,xalm,yalm):
    shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(0.5*8192/nside/60.))
    res = qe.qe_all(shape,wcs,lambda x,y: theory.lCl(x,y),mlmax,yalm[0],yalm[1],yalm[2],
                    estimators=[polcomb],xfTalm=xalm[0],xfEalm=xalm[1],xfBalm=xalm[2])
    return res[polcomb]


def qfunc(XY,ixalm,iyalm):
    xalm = ixalm.copy()
    yalm = iyalm.copy()
    filt = [lambda x: 1./(theory.lCl('TT',x)),
            lambda x: 1./(theory.lCl('EE',x)),
            lambda x: 1./(theory.lCl('BB',x))]
    xs = {'T':0,'E':1,'B':2}
    X,Y = XY
    for i in range(3): xalm[i] = qe.filter_alms(ixalm[i],filt[i],lmin=lmin,lmax=lmax)
    for i in range(3): yalm[i] = qe.filter_alms(iyalm[i],filt[i],lmin=lmin,lmax=lmax)
    return hp.almxfl(get_mv_kappa(XY,xalm,yalm)[0],Als[polcomb])

# norm dict
Als = {}
with bench.show("norm"):
    ls,Als['TT'],Als['EE'],Als['EB'], \
        Als['TE'],Als['TB'],al_mv_pol, \
        al_mv,Al_te_hdv = solenspipe.initialize_generic_norm(lmin,lmax, \
                                                             tag='noiseless')
Als['mv'] = al_mv
Als['mvpol'] = al_mv_pol
al_mv = Als[polcomb]

with bench.show("load alm"):
    alm = solenspipe.get_cmb_alm(i=0,iset=0)
    alm = maps.change_alm_lmax(alm, mlmax)

ikalm = maps.change_alm_lmax(solenspipe.get_kappa_alm(i=0).astype(np.complex128),mlmax)

with bench.show("recon"):
    rkalm = qfunc(polcomb,alm,alm)

nls = al_mv * ls**2./4.


xcls = hp.alm2cl(rkalm,ikalm)
rcls = hp.alm2cl(rkalm,rkalm)
icls = hp.alm2cl(ikalm,ikalm)
ells = np.arange(len(icls))
clkk = theory.gCl('kk',ells)


pl = io.Plotter(xyscale='loglog',xlabel='$L$',ylabel='$C_L$')
pl.add(ells,clkk,ls="-",lw=3,label='theory input')
pl.add(ells,xcls,alpha=0.6,label='input x recon')
pl.add(ells,icls,alpha=0.6,label='input x input')
pl.add(ls,nls,ls="--",label='theory noise per mode')
pl.add(ls,nls+theory.gCl('kk',ls),ls="-.",label='C+A')
pl.add(ells,rcls,alpha=0.6,label='recon x recon')
pl.add(ls,rcls[:len(ls)]-nls,alpha=0.6,label='recon x recon - N0')
pl._ax.set_xlim(0.9,4000)
pl._ax.set_ylim(1e-9,1e-6)
pl.done(config['data_path']+"xcls_%s.png" % polcomb)


bin_edges = np.geomspace(2,4000,30)
binner = stats.bin1D(bin_edges)
diff = (rcls[:len(ls)]-nls-icls[:len(ls)])/icls[:len(ls)]
diff2 = (xcls[:len(ls)]-icls[:len(ls)])/icls[:len(ls)]
cents,bdiff = binner.bin(ls,diff)
cents,bdiff2 = binner.bin(ls,diff2)

pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$\\Delta C_L $')
pl.add(cents,bdiff,marker='o',label='auto-AL')
pl.add(cents,bdiff2,marker='o',label='cross')
pl.hline()
pl._ax.set_xlim(0.9,4000)
pl._ax.set_ylim(-0.25,0.25)
pl.done(config['data_path']+"diffcls_%s.png" % polcomb)


class MapCacher(object):
    def __init__(self,mlmax,maxcache=5):
        self.mlmax = mlmax
        self.maxcache = maxcache
        self.cache = {}
        self.keys = []
    
    def load_alms(self,i,iset):
        key = (i,iset)
        if len(self.keys)>self.maxcache:
            key0 = self.keys[0]
            del self.cache[key0]
            self.keys.pop(0)
        alms = maps.change_alm_lmax(solenspipe.get_cmb_alm(i,iset),self.mlmax)
        self.cache[key] = alms
        return alms

    def get_kmap(self,X,seed):
        s_i,s_set,_ = solenspipe.convert_seeds(seed)
        key = (s_i,s_set)
        if key in self.cache:
            alms = self.cache[key]
        else:
            alms = self.load_alms(s_i,s_set)
        return alms


nsims = 40
comm,_,_ = mpi.distribute(nsims,verbose=True)
power = lambda x,y: hp.alm2cl(x,y)
mcache = MapCacher(mlmax)
get_kmap = mcache.get_kmap
cl_n1 = bias.mcn1(0,polcomb,polcomb,qfunc,get_kmap,comm,power,nsims,verbose=True)
ls = np.arange(len(cl_n1))
cents,cn1 = binner.bin(ls,cl_n1)

pl = io.Plotter(xyscale='loglin',xlabel='$L$',ylabel='$\\Delta C_L $',scalefn = lambda x: x)
pl.add(cents,bdiff,marker='o')
pl.add(cents,cn1,marker='o')
pl.add(cents,bdiff-cn1,marker='o',label='mcn1 subbed')
pl.hline()
pl._ax.set_xlim(0.9,4000)
#pl._ax.set_ylim(-1,1)
pl.done(config['data_path']+"diffcls_n1_%s.png" % polcomb)
