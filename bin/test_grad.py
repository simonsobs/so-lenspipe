from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi
from pixell import enmap,utils
import numpy as np
import os,sys
import solenspipe
from falafel import qe
from enlib import bench
import healpy as hp

nsims = 400
config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

comm,rank,my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

mlmax = 3500
lmin = 20
lmax = 3000
shape,wcs  = enmap.fullsky_geometry(res=2.5 * utils.arcmin)
px = qe.pixelization(shape=shape,wcs=wcs)
theory = cosmology.default_theory()
thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
#theory_cross=cosmology.load_theory_from_glens(thloc+"_camb_1.0.12",total=False,lpad=9000,TCMB=2.7255e6)
ells,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])
class T:
    def __init__(self):
        self.lCl = lambda p,x: maps.interp(ells,gt)(x)
theory_cross = T()

ls = np.arange(mlmax)
cltt = theory.lCl('TT',ls)
clgt = theory_cross.lCl('TT',ls)
pl = io.Plotter('rCl')
pl.add(ls,cltt/clgt)
pl.hline(y=1)
pl._ax.set_ylim(0.99,1.15)
pl.done(f'{opath}gclrat.png')

with bench.show("norm"):
    ls,Als = solenspipe.quicklens_norm("TT",ls*0,ls*0,ls*0,theory,theory,lmin,lmax,lmin,lmax)
    ls,gAls = solenspipe.quicklens_norm("TT",ls*0,ls*0,ls*0,theory,theory_cross,lmin,lmax,lmin,lmax)
assert ls[0]==0
assert ls[1]==1

bin_edges = np.geomspace(2,3000,18).astype(int)
binner = stats.bin1D(bin_edges)

for task in my_tasks:

    print(task)
    talm = solenspipe.get_cmb_alm(task,0,hdu=1)
    filt_t = lambda x: 1./theory.lCl('TT',x)
    xalm = qe.filter_alms(talm.copy(),filt_t,lmin=lmin,lmax=lmax)
    X = []
    X.append(xalm)
    X.append(xalm*0)
    X.append(xalm*0)
    Y = list(X)

    with bench.show("recon"):
        urecon = qe.qe_all(px,lambda x,y: theory.lCl(x,y),lambda x,y: theory.lCl(x,y),
                           mlmax,Y[0],Y[1],Y[2],estimators=['TT'],
                           xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])['TT'][0]
        gurecon = qe.qe_all(px,lambda x,y: theory.lCl(x,y),lambda x,y: theory_cross.lCl(x,y),
                           mlmax,Y[0],Y[1],Y[2],estimators=['TT'],
                           xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])['TT'][0]

    kalm = maps.change_alm_lmax(solenspipe.get_kappa_alm(task),mlmax)
    krecon = hp.almxfl(urecon,Als)
    gkrecon = hp.almxfl(gurecon,gAls)

    ubin = hp.alm2cl(kalm,krecon)
    gubin = hp.alm2cl(kalm,gkrecon)
    ls = np.arange(ubin.size)
    cents,clri = binner.bin(ls,ubin)
    cents,gclri = binner.bin(ls,gubin)
    cents,clii = binner.bin(ls,hp.alm2cl(kalm,kalm))

    s.add_to_stats("rat",(clri-clii)/clii)
    s.add_to_stats("grat",(gclri-clii)/clii)

s.get_stats()

if rank==0:
    m = s.stats['rat']['mean']
    e = s.stats['rat']['errmean']

    gm = s.stats['grat']['mean']
    ge = s.stats['grat']['errmean']

    pl = io.Plotter('rCL',xyscale='loglin')
    pl.add_err(cents,m,yerr=e,ls="-",label='$C_{TT}$')
    pl.add_err(cents,gm,yerr=ge,ls="--",label='$C_{T\\nabla T}$')
    pl.hline(y=0)
    pl.hline(y=-0.005)
    pl._ax.set_ylim(-0.05,0.05)
    pl.done(f'{opath}gradrat.png')
