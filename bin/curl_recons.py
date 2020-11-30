
from __future__ import print_function
from orphics import stats,io,mpi,maps,cosmology
from pixell import enmap, curvedsky as cs, utils, enplot
import numpy as np
import os,sys
import 	solenspipe
from falafel import qe
import mapsims
from enlib import bench
import healpy as hp


"""
Similar to simple.py except for curl reconstruction and meanfield calculation.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Simple lensing reconstruction test.')
parser.add_argument("label", type=str,help='Version label.')
parser.add_argument("polcomb", type=str,help='Polarizaiton combination: one of mv,TT,TE,EB,TB,EE.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="Number of sims.")
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

print(len(ils))
print(len(Als))
w1 = solint.wfactor(1)      
w2 = solint.wfactor(2)
w3 = solint.wfactor(3)
w4 = solint.wfactor(4)
print(w1)
car = "healpix_" if args.healpix else "car_"
noise="wnoise" if args.wnoise!=None else "sonoise"
mask="nomask" if args.no_mask else "mask"


if args.write_meanfield: assert not(args.read_meanfield)


s = stats.Stats(comm)

if args.read_meanfield:
    mf_alm = hp.read_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm.fits')
else:
    mf_alm = 0



s = stats.Stats(comm)

for task in my_tasks:

    # Choose a seed. This has to be varied when simulating.
    seed = (0,0,task+sindex)


    with bench.show("sim"):
        print(seed)
        t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
        # Get the reconstructed map for the TT estimator
        
        
        curl=solint.get_mv_curl(polcomb,t_alm,e_alm,b_alm)
        curl_alms=qe.filter_alms(curl,maps.interp(ils,Als[polcomb]))

    
    if args.read_meanfield:
        print("substracting meanfield from auto")
        curl_alms = curl_alms - mf_alm


    kalms = solint.get_kappa_alm(1)
    ccl=hp.alm2cl(curl_alms,curl_alms)
    s.add_to_stats('ccl',ccl/w4)
    if args.write_meanfield:
        s.add_to_stack('rmf',curl_alms.real)
        s.add_to_stack('imf',curl_alms.imag)



with io.nostdout():
    s.get_stats()
    s.get_stacks()



if rank==0:
    with io.nostdout():
        ccl = s.stats['ccl']['mean']

        if args.ps_bias_hardening:
            np.savetxt(f'{solenspipe.opath}/curl_{args.label}_{args.lmin}_{args.lmax}psh.txt',ccl)
        elif args.read_meanfield:
            np.savetxt(f'{solenspipe.opath}/curlmfsub_{args.label}_{args.lmin}_{args.lmax}_{nsims}.txt',ccl)
        else:
            np.savetxt(f'{solenspipe.opath}/curl_{args.label}_{args.lmin}_{args.lmax}_{nsims}.txt',ccl)


    if args.write_meanfield:
        mf_alm = s.stacks['rmf'] + 1j*s.stacks['imf']
        hp.write_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm.fits',mf_alm)
        
    

    ls = np.arange(ccl.size)


        
    
    Ncl=maps.interp(ils,Nl)(ls)
    np.savetxt(f'{solenspipe.opath}/N0curl_{args.label}_{args.polcomb}_lmin{args.lmin}_lmax{args.lmax}.txt',Ncl)


    pl = io.Plotter('CL',xyscale='loglog')
    if args.write_meanfield or args.read_meanfield:
        mf_cl = hp.alm2cl(mf_alm,mf_alm) / w4
        np.savetxt(f'{solenspipe.opath}/mfcurl_{args.label}_{args.polcomb}_lmin{args.lmin}_lmax{args.lmax}.txt',mf_cl)
        pl.add(ls,mf_cl,alpha=0.5,label='mcmf cl')
    pl.add(ls,ccl-Ncl,ls='--',label='curl')
    pl._ax.set_ylim(1e-10,1e-2)
    pl._ax.set_xlim(1,3100)
    pl.done(f'{solenspipe.opath}/{args.label}_{args.polcomb}_{isostr}curl.png')




