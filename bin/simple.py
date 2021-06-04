from __future__ import print_function
from orphics import stats,io,mpi,maps,cosmology
from pixell import enmap,curvedsky as cs
from pixell import enplot
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
parser.add_argument("--foreground", action='store_true',help='Use or not use foregrounds')
parser.add_argument("--flat-sky-norm", action='store_true',help='Use flat-sky norm.')
parser.add_argument("--ps_bias_hardening", action='store_true',help='TT point source hardening estimator.')
parser.add_argument("--mask_bias_hardening", action='store_true',help='TT mask hardening estimator.')
parser.add_argument("--curl", action='store_true',help='curl reconstruction')
parser.add_argument("--trispectrum", action='store_true',help='foreground trispectrum reconstruction')


args = parser.parse_args()

solint,Als,Als_curl,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)
ils=Als['L']

w1 = solint.wfactor(1)      
w2 = solint.wfactor(2)
w3 = solint.wfactor(3)
w4 = solint.wfactor(4)
print(w2)
car = "healpix_" if args.healpix else "car_"
noise="wnoise" if args.wnoise!=None else "sonoise"
mask="nomask" if args.no_mask else "mask"

if args.write_meanfield: assert not(args.read_meanfield)


s = stats.Stats(comm)
if args.read_meanfield:
    mf_alm = hp.read_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm_100.fits')
else:
    mf_alm = 0



s = stats.Stats(comm)

for task in my_tasks:

    # Choose a seed. This has to be varied when simulating.
    seed = (0,0,task)

    with bench.show("sim"):
        # Get simulated, prepared filtered T, E, B maps, i.e. (1/(C+N) * teb_alm)
        t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=True,foreground=args.foreground,trispectrum=args.trispectrum)
        Tcmb = 2.726e6
        # Get the reconstructed map for the TT estimator
        
        if args.ps_bias_hardening:
            ls,nells,nells_P = solint.get_noise_power(channel,beam_deconv=True)
            ells=np.arange(0,solint.mlmax)
            config = io.config_from_yaml("../input/config.yml")
            thloc = "../data/" + config['theory_root']
            theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
            
            
            ellsi,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])
            class T:
                def __init__(self):
                    self.lCl = lambda p,x: maps.interp(ellsi,gt)(x)
            theory_cross = T()

            ls,blens,bhps,Alpp,A_ps,bhclkknorm=solenspipe.bias_hard_ps_norms(nells,nells_P,nells_P,theory,theory_cross,lmin,lmax)
            np.savetxt(f'{solenspipe.opath}/normell_{args.label}.txt',ls)
            np.savetxt(f'{solenspipe.opath}/N0bhclkknorm_{args.label}.txt',bhclkknorm)
            np.savetxt(f'{solenspipe.opath}/invdet_{args.label}.txt',blens)
            np.savetxt(f'{solenspipe.opath}/bhps_{args.label}.txt',bhps)
            np.savetxt(f'{solenspipe.opath}/A_ps_{args.label}.txt',A_ps)
            np.savetxt(f'{solenspipe.opath}/Alpp_{args.label}.txt',Alpp)
            """
            s_alms=qe.filter_alms(solint.get_pointsources(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,A_ps*bhps*Tcmb**2))
            phi_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,2*Alpp*blens))
            balms=phi_alms-s_alms
            recon_alms=hp.almxfl(balms,ils*(ils+1)*0.5)
            """
            alpha="TT"
            qa = lambda x,y,ils,blens,bhps,Alpp,A_ps: solint.qfunc_bh(alpha,x,y,ils,blens,bhps,Alpp,A_ps)
            recon_alms=qa([t_alm,e_alm,b_alm],[t_alm,e_alm,b_alm],ils,blens,bhps,Alpp,A_ps)
            
        elif args.mask_bias_hardening:
            ls,nells,nells_P = solint.get_noise_power(channel,beam_deconv=True)
            ells=np.arange(0,solint.mlmax)
            config = io.config_from_yaml("../input/config.yml")
            thloc = "../data/" + config['theory_root']
            theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
            
            
            ellsi,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])
            class T:
                def __init__(self):
                    self.lCl = lambda p,x: maps.interp(ellsi,gt)(x)
            theory_cross = T()

            ls,blens,bmask,Alpp,Amask,bhclkknorm=solenspipe.bias_hard_mask_norms(nells,nells_P,nells_P,theory,theory_cross,lmin,lmax)
            mask_alms=qe.filter_alms(solint.get_mv_mask(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,Amask*bmask))
            phi_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,2*Alpp*blens))
            recon_alms=phi_alms-mask_alms
            recon_alms=hp.almxfl(recon_alms,ils*(ils+1)*0.5)
        
        else:
            recon_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,Als[polcomb]))
            """
            falms = recon_alms.copy()
            ls = np.arange(solint.mlmax+1)
            fls = ls * 1
            fls[ls<2] = 0
            fls[ls>200] = 0
            falms = hp.almxfl(falms,fls)
            rmap = solint.alm2map(falms,ncomp=1)[0] * solint.mask
            plots = enplot.get_plots(rmap, downgrade = 1,color="gray")
            print("make plot")
            spath="/home/r/rbond/jiaqu/scratch/so_lens/shear/"
            #enplot.write(f'{spath}/reconstruct',plots)
            """
    
    if args.read_meanfield:
        recon_alms = recon_alms - mf_alm


    if task==0 and debug_cmb:
        maskb=solenspipe.get_mask(healpix=args.healpix,lmax=lmax,no_mask=args.no_mask,car_deg=2,hp_deg=4)
        rmap = solint.alm2map(recon_alms,ncomp=1)[0] * maskb
        io.hplot(rmap,f'{solenspipe.opath}/rmap',mask=0,color='gray')
        falms = recon_alms.copy()
        ls = np.arange(solint.mlmax+1)
        fls = ls * 1
        fls[ls<2] = 0
        fls[ls>100] = 0
        falms = hp.almxfl(falms,fls)
        rmap = solint.alm2map(falms,ncomp=1)[0] * maskb
        io.hplot(rmap,f'{solenspipe.opath}/frmap',mask=0,color='gray')

    # Get the input kappa map alms from websky if seed=(0,0,0)
    if seed==(0,0,0):
        kappa_path='/home/r/rbond/jiaqu/scratch/websky/websky/kap.fits' 
        kappa_map=hp.fitsfunc.read_map(kappa_path)
        kalms = hp.map2alm(kappa_map, lmax=solint.mlmax)
    else:
        kalms = solint.get_kappa_alm(task)

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
        spath="/home/r/rbond/jiaqu/scratch/so_lens/shear/"
        if args.foreground:
            np.savetxt(f'{spath}/acl_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}foreground_masked.txt',acl)
            np.savetxt(f'{spath}/xcl_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}foregroundt_masked.txt',xcl)
            np.savetxt(f'{spath}/norm_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}foregroundt.txt',xcl/icl)
        
        elif args.trispectrum:
            np.savetxt(f'{spath}/acl_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}trispectrum_masked.txt',acl)
            np.savetxt(f'{spath}/xcl_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}trispectrum_masked.txt',xcl)
            np.savetxt(f'{spath}/norm_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}trispectrum_masked.txt',xcl/icl)
        else:
            np.savetxt(f'{spath}/acl_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}.txt',acl)
            np.savetxt(f'{spath}/xcl_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}.txt',xcl)
            np.savetxt(f'{spath}/norm_{args.label}_lmin{args.lmin}_lmax{args.lmax}_nsims{args.nsims}.txt',xcl/icl)

        np.savetxt(f'{spath}/icl.txt',icl)

    if args.write_meanfield:
        mf_alm = s.stacks['rmf'] + 1j*s.stacks['imf']
        hp.write_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm_{nsims}.fits',mf_alm,overwrite=True)
        
    

    ls = np.arange(xcl.size)
    if args.ps_bias_hardening:
        Nlbh=maps.interp(ils,bhclkknorm)(ls)
        np.savetxt(f'{solenspipe.opath}/bhnorm.txt',Nlbh)
    
    elif args.mask_bias_hardening:
        Nl=maps.interp(ils,bhclkknorm)(ls)
        np.savetxt(f'{solenspipe.opath}/masknorm.txt',Nl)
        
        
    else:
        Nl = maps.interp(ils,Nl)(ls)
        np.savetxt(f'{spath}/N0.txt',Nl)


    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ls,acl,alpha=0.5,label='raw auto')
    if args.write_meanfield or args.read_meanfield:
        mf_cl = hp.alm2cl(mf_alm,mf_alm) / w4
        pl.add(ls,mf_cl,alpha=0.5,label='mcmf cl')
        pl.add(ls,acl-mf_cl,label='rr - mf')
    pl.add(ls,xcl,label='raw auto x input')
    pl.add(ls,icl,color='k')
    pl.add(ls,icl+Nl,ls='--',label='input + N0 bias')
    pl._ax.set_ylim(1e-10,1e-2)
    pl._ax.set_xlim(1,3100)
    pl.done(f'{spath}/{args.label}_{args.polcomb}_{isostr}recon.png')
    


                                                                                   

                                                                                

