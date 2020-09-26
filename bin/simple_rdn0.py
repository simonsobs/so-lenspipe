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
parser.add_argument("-N", "--nsims",     type=int,  default=100,help="Number of sims.")
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
parser.add_argument("--ps_bias_hardening", action='store_true',help='TT point source hardening estimator.')
parser.add_argument("--mask_bias_hardening", action='store_true',help='TT mask hardening estimator.')
parser.add_argument("--curl", action='store_true',help='curl reconstruction')
args = parser.parse_args()
solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)
      
car = "healpix_" if args.healpix else "car_"

w4 = solint.wfactor(4)
get_kmap = lambda seed: solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
power = lambda x,y: hp.alm2cl(x,y)
if args.curl:
    qfunc = solint.qfunc_curl
else:
    qfunc = solint.qfunc
nmax = len(ils)

rdn0 = bias.rdn0(icov=0,alpha=polcomb,beta=polcomb,qfunc=qfunc,get_kmap=get_kmap,comm=comm,power=power,nsims=nsims)
rdn0[:nmax] = rdn0[:nmax] * Als[polcomb]**2.
if not(args.no_mask):
    rdn0[:nmax]=rdn0[:nmax]/w4
rdn0[nmax:] = 0
power = lambda x,y: hp.alm2cl(x,y)
if args.curl:
	io.save_cols(f'/global/homes/j/jia_qu/so-lenspipe/data/rdn0_curl_{polcomb}_{isostr}_{car}_{nsims}_{args.label}.txt',(ils,rdn0[:nmax]))

else:
	io.save_cols(f'/global/homes/j/jia_qu/so-lenspipe/data/rdn0_curl_{polcomb}_{isostr}_{car}_{nsims}_{args.label}.txt',(ils,rdn0[:nmax]))
if rank==0:
    theory = cosmology.default_theory()
    
    ls = np.arange(rdn0.size)
    pl = io.Plotter('CL')
    pl.add(ils,rdn0[:nmax])
    pl.add(ils,Nl)
    pl.add(ils,theory.gCl('kk',ils))
    #pl._ax.set_ylim(1e-9,1e-6)
    pl.done(f'/global/homes/j/jia_qu/so-lenspipe/data/recon_rdn0.png')


                                                                                

