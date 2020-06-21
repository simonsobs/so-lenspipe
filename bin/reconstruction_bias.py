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
import traceback
from solenspipe import bias



def get_mf_alms(solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr,label):

    almfname=f'{solenspipe.opath}mf_{label}_{polcomb}_{isostr}_{nsims}_alm.fits'
    try:
        return hp.read_alm(almfname)
    except:
        print(traceback.format_exc()) 
        s = stats.Stats(comm)
        
        for task in my_tasks:
        
            # Choose a seed. This has to be varied when simulating.
            seed = (0,0,task+sindex)
        
            # Get the simulated, prepared T, E, B maps
        
         
            t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
            # Get the reconstructed map for the TT estimator
            recon_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,Als[polcomb]))
            s.add_to_stack('rmf',recon_alms.real)
            s.add_to_stack('imf',recon_alms.imag)
        
        
        with io.nostdout():
            s.get_stacks()
        
        if rank==0:
            mf_alm = s.stacks['rmf'] + 1j*s.stacks['imf']
            hp.write_alm(f'{solenspipe.opath}mf_{label}_{polcomb}_{isostr}_{nsims}_alm.fits',mf_alm,overwrite=True)

def recon_rdn0(solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr,label):
    rdn0fname=f'{solenspipe.opath}/rdn0_{label}_{polcomb}_{isostr}_{nsims}.txt'
    try:
        return np.loadtxt(rdn0fname)
    except:
        print(traceback.format_exc())     
        w4 = solint.wfactor(4)
        get_kmap = lambda seed: solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
        power = lambda x,y: hp.alm2cl(x,y)
        qfunc = solint.qfunc
        
        nmax = len(ils)
        
        rdn0 = bias.mean_rdn0(icov=0,alpha=polcomb,beta=polcomb,qfunc=qfunc,get_kmap=get_kmap,comm=comm,power=power,nsims=nsims)
        rdn0[:nmax] = rdn0[:nmax] * Als[polcomb]**2.
        rdn0[:nmax]=rdn0[:nmax]/w4
        rdn0[nmax:] = 0
        io.save_cols(f'{solenspipe.opath}/rdn0_{label}_{polcomb}_{isostr}_{nsims}.txt',(ils,rdn0[:nmax]))
    return ils,rdn0[:nmax]

def recon_mcn1(solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr,label):
    mcn1fname=f'{solenspipe.opath}/mcn1_{label}_{polcomb}_{isostr}_{nsims}.txt'
    try:
        return np.loadtxt(rdn0fname)
    except:
        print(traceback.format_exc())     
        w4 = solint.wfactor(4)
        get_kmap = lambda seed: solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
        power = lambda x,y: hp.alm2cl(x,y)
        qfunc = solint.qfunc
        
        nmax = len(ils)
        
        mcn1 = bias.mcn1(0,polcomb,polcomb,qfunc,get_kmap,comm,power,nsims,verbose=True)
        mcn1[:nmax] = mcn1[:nmax] * Als[polcomb]**2. 
        mcn1[:nmax] = mcn1[:nmax]/w4
            
        mcn1[nmax:] = 0
        io.save_cols(f'{solenspipe.opath}/mcn1_{label}_{polcomb}_{isostr}_{nsims}.txt',(ils,mcn1[:nmax]))
    return ils,mcn1[:nmax]
# Calculate autospectrum
# Subtract biases