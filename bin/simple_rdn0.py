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

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Simple rdn0 calculation')
parser.add_argument("label", type=str,help='Version label.')
parser.add_argument("polcomb", type=str,help='Polarizaiton combination: one of mv,TT,TE,EB,TB,EE.')
parser.add_argument("-N", "--nsims",     type=int,  default=60,help="Number of sims.")
parser.add_argument("--sindex",     type=int,  default=0,help="Start index for sims.")
parser.add_argument("--lmin",     type=int,  default=100,help="Minimum multipole.")
parser.add_argument("--lmax",     type=int,  default=3000,help="Minimum multipole.")
parser.add_argument("--isotropic", action='store_true',help='Isotropic sims.')
parser.add_argument("--no-atmosphere", action='store_true',help='Disable atmospheric noise.')
parser.add_argument("--use-cached-norm", action='store_true',help='Use  cached norm.')
parser.add_argument("--wnoise",     type=float,  default=None,help="Override white noise.")
parser.add_argument("--beam",     type=float,  default=None,help="Override beam.")
parser.add_argument("--disable-noise", action='store_true',help='Disable noise.')
parser.add_argument("--zero-sim", action='store_true',help='Just make a sim of zeros. Useful for benchmarking.')
parser.add_argument("--write-meanfield", action='store_true',help='Calculate and save mean-field map.')
parser.add_argument("--read-meanfield", action='store_true',help='Read and subtract mean-field map.')
parser.add_argument("--healpix", action='store_true',help='Use healpix instead of CAR.')
parser.add_argument("--no-mask", action='store_true',help='No mask. Use with the isotropic flag.')
parser.add_argument("--debug", action='store_true',help='Debug plots.')
parser.add_argument("--flat-sky-norm", action='store_true',help='Use flat-sky norm.')
parser.add_argument("--foreground", action='store_true',help='Use or not use foregrounds')
parser.add_argument("--ps_bias_hardening", action='store_true',help='TT point source hardening estimator.')
parser.add_argument("--mask_bias_hardening", action='store_true',help='TT mask hardening estimator.')
parser.add_argument("--curl", action='store_true',help='curl reconstruction')

args = parser.parse_args()
solint,Als,Als_curl,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)
car = "healpix_" if args.healpix else "car_"
spath="/home/r/rbond/jiaqu/scratch/so_lens/shear/"

w4 = solint.wfactor(4)
print(w4)
get_kmap = lambda seed: solint.get_kmap(channel,seed,lmin,lmax,filtered=True,foreground=args.foreground)
power = lambda x,y: hp.alm2cl(x,y)
if args.curl:
    qfunc = solint.qfunc_curl
elif args.ps_bias_hardening:
    ls,nells,nells_P = solint.get_noise_power(channel,beam_deconv=True)
    ells=np.arange(0,solint.mlmax)
    config = io.config_from_yaml("../input/config.yml")
    thloc = "../data/" + config['theory_root']
    theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
    
    
    ellsi,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])
    class T:
        def __init__(self):
            self.lCl = lambda p,x: maps.interp(ellsi,gt)(x)
 
    #ls,blens,bhps,Alpp,A_ps,bhclkknorm=solenspipe.bias_hard_ps_norms(nells,nells_P,nells_P,theory,theory_cross,lmin,lmax)
    A_ps=np.loadtxt(f'{solenspipe.opath}/A_ps_{args.label}.txt')
    bhps=np.loadtxt(f'{solenspipe.opath}/bhps_{args.label}.txt')
    Alpp=np.loadtxt(f'{solenspipe.opath}/Alpp_{args.label}.txt')
    blens=np.loadtxt(f'{solenspipe.opath}/invdet_{args.label}.txt')
    qfunc=solint.qfunc_bh
else:
    print("qfunc")
    qfunc = solint.qfunc
nmax = len(Als['L'])

if args.ps_bias_hardening:
    rdn0= bias.rdn0(icov=0,alpha=polcomb,beta=polcomb,qfunc=qfunc,get_kmap=get_kmap,comm=comm,power=power,nsims=nsims,type='bh',ils=ils, blens=blens, bhps=bhps, Alpp=Alpp, A_ps=A_ps)
    rdn0[nmax:] = 0
    
else:
    print("normal rdn0")
    rdn0 = bias.rdn0(icov=0,alpha=polcomb,beta=polcomb,qfunc=qfunc,get_kmap=get_kmap,comm=comm,power=power,nsims=nsims)
    print(rdn0)
    rdn0[:nmax] = rdn0[:nmax] * Als[polcomb]**2.
if not(args.no_mask):
    rdn0[:nmax]=rdn0[:nmax]/w4
rdn0[nmax:] = 0
if args.curl:
    io.save_cols(f'{solenspipe.opath}/rdn0_curl_{polcomb}_{isostr}_{car}_{nsims}_{args.label}.txt',(ils,rdn0[:nmax]))
elif args.ps_bias_hardening:
    io.save_cols(f'{solenspipe.opath}/rdn0_bh_{polcomb}_{isostr}_{car}_{nsims}_{args.label}.txt',(ils,rdn0[:nmax]))
else:
    if args.foreground:
        io.save_cols(f'{spath}/rdn0_foregrounds_{polcomb}_{isostr}_{car}_{nsims}_{args.label}_lmin{args.lmin}_lmax{args.lmax}.txt',(Als['L'],rdn0[:nmax]))
    else:
        print("saving no foregrounds")
        io.save_cols(f'{spath}/rdn0_{polcomb}_{isostr}_{car}_{nsims}_{args.label}_lmin{args.lmin}_lmax{args.lmax}.txt',(Als['L'],rdn0[:nmax]))

if rank==0:
    theory = cosmology.default_theory()
    
    ls = np.arange(rdn0.size)
    pl = io.Plotter('CL')
    pl.add(Als['L'],rdn0[:nmax])
    pl.add(Als['L'],Nl)
    pl.add(Als['L'],theory.gCl('kk',Als['L']))
    #pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'{spath}recon_rdn0.png')

                                                                                

