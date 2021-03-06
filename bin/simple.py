from __future__ import print_function
from orphics import stats,io,mpi,maps,cosmology
from pixell import enmap,curvedsky as cs
import numpy as np
import os,sys
import solenspipe
from falafel import qe
import mapsims
from enlib import bench
import healpy as hp

"""
We will do a simple lensing reconstruction test.
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
parser.add_argument("--flat-sky-rdn0", action='store_true',help='Do flat-sky rdn0.')
args = parser.parse_args()

solint,Als,Alcurl,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)
      
w2 = solint.wfactor(2)
w3 = solint.wfactor(3)
w4 = solint.wfactor(4)

if args.write_meanfield: assert not(args.read_meanfield)


s = stats.Stats(comm)

if args.read_meanfield:
    mf_alm = hp.read_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm.fits')
else:
    mf_alm = 0


for task in my_tasks:

    # Choose a seed. This has to be varied when simulating.
    seed = (0,0,task+sindex)

    # If debugging, get unfiltered maps and plot Cls
    if task==0 and debug_cmb:
        t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=False)
        tcl = hp.alm2cl(t_alm)
        ls = np.arange(tcl.size)
        pl = io.Plotter('Cell')
        pl.add(ls,tcl/w2)
        ls2,nells,nells_P = solint.get_noise_power(channel,beam_deconv=True)
        theory = cosmology.default_theory()
        pl.add(ls,theory.lCl('TT',ls) + maps.interp(ls2,nells)(ls),color='k')
        pl._ax.set_xlim(1,6000)
        pl._ax.set_ylim(1e-6,1e3)
        pl.done(f'{solenspipe.opath}/tcl.png')
        imap = enmap.downgrade(solint.alm2map(np.asarray([t_alm,e_alm,b_alm]),ncomp=3) * maps.binary_mask(mask),2)
        for i in range(3): io.hplot(imap[i],f'{solenspipe.opath}/imap_{i}',mask=0)


    with bench.show("sim"):
        # Get simulated, prepared filtered T, E, B maps, i.e. (1/(C+N) * teb_alm)
        t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
        # Get the reconstructed kappa map alms and filter it with the normalization
        recon_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(Als['L'],Als[polcomb]))
    
    # Subtract a meanfield if necessary
    recon_alms = recon_alms - mf_alm

    if task==0 and debug_cmb:
        rmap = solint.alm2map(recon_alms,ncomp=1)[0] * maps.binary_mask(mask)
        io.hplot(rmap,f'{solenspipe.opath}/rmap',mask=0,color='gray')
        falms = recon_alms.copy()
        ls = np.arange(solint.mlmax+1)
        fls = ls * 1
        fls[ls<2] = 0
        fls[ls>100] = 0
        falms = hp.almxfl(falms,fls)
        rmap = solint.alm2map(falms,ncomp=1)[0] * maps.binary_mask(mask)
        io.hplot(rmap,f'{solenspipe.opath}/frmap',mask=0,color='gray')

    # Get the input kappa map alms
    kalms = solint.get_kappa_alm(task+sindex)

    acl = hp.alm2cl(recon_alms,recon_alms) # reconstruction raw autopower
    xcl = hp.alm2cl(recon_alms,kalms)  # cross-power of input and reconstruction
    icl = hp.alm2cl(kalms,kalms)  # autopower of input

    # Apply mask corrections and add to stats collecter
    s.add_to_stats('acl',acl/w4)
    s.add_to_stats('xcl',xcl/w3)
    s.add_to_stats('icl',icl/w2)

    # Stack meanfield alms
    if args.write_meanfield:
        s.add_to_stack('rmf',recon_alms.real)
        s.add_to_stack('imf',recon_alms.imag)

with io.nostdout():
    s.get_stats()
    s.get_stacks()

if rank==0:

    # Collect statistics and plot
    with io.nostdout():
        acl = s.stats['acl']['mean']
        xcl = s.stats['xcl']['mean']
        icl = s.stats['icl']['mean']

    if args.write_meanfield:
        mf_alm = s.stacks['rmf'] + 1j*s.stacks['imf']
        hp.write_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm.fits',mf_alm,overwrite=True)
        
    
    ls = np.arange(xcl.size)
    Nl = maps.interp(Als['L'],Nl)(ls)
    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ls,acl,alpha=0.5,label='rr')
    if args.write_meanfield or args.read_meanfield:
        mf_cl = hp.alm2cl(mf_alm,mf_alm) / w4
        pl.add(ls,mf_cl,alpha=0.5,label='mcmf cl')
        pl.add(ls,acl-mf_cl,label='rr - mf')
    pl.add(ls,xcl,label='ri')
    pl.add(ls,icl,color='k')
    pl.add(ls,icl+Nl,ls='--',label='ii + Nl')
    pl._ax.set_ylim(1e-10,1e-2)
    pl._ax.set_xlim(1,3100)
    pl.done(f'{solenspipe.opath}/{args.label}_{args.polcomb}_{isostr}recon.png')
                                                                                   

                                                                                

