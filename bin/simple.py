from __future__ import print_function
from orphics import stats,io,mpi,maps
from pixell import enmap
import numpy as np
import os,sys
import solenspipe
from falafel import qe
import mapsims
from enlib import bench
import healpy as hp

"""
We will do a simple lensing reconstruction test.
No mask.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("label", type=str,help='Label.')
parser.add_argument("polcomb", type=str,help='polcomb.')
parser.add_argument("-N", "--nsims",     type=int,  default=1,help="Number of sims.")
parser.add_argument("--band",     type=int,  default=None,help="Declination band.")
parser.add_argument("--isotropic", action='store_true',help='Isotropic sims.')
# parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()


# Lensing reconstruction ell range
lmin = 500
lmax = 3000
use_cached_norm = True
quicklens = True
#wnoise = None

wnoise = 36.0
beam = 7.0

# wnoise = 6.0
# beam = 1.4

polcomb = args.polcomb

# Number of sims
nsims = args.nsims
comm,rank,my_tasks = mpi.distribute(nsims)

isostr = "isotropic_" if args.isotropic else ""

# CAR resolution is decided based on lmax
res = np.deg2rad(2.0 *(3000/lmax) /60.)

# Make the full sky geometry
if args.band is not None:
    shape,wcs = enmap.band_geometry(np.deg2rad(args.band),res=res)
else:
    shape,wcs = enmap.fullsky_geometry(res=res)

# We use a mask of ones for this test
if args.isotropic:
    mask = enmap.ones(shape,wcs)


"""
The following needs to be cleaned up to allow for testing different masks
"""
#mask,_ = maps.get_taper_deg(shape,wcs,4.,30.,only_y=True)
#mask,_ = maps.get_taper_deg(shape,wcs,4.,30.,only_y=False)
#sys.exit()

mask1,_ = maps.get_taper_deg(shape,wcs,4.,45.,only_y=False)
#mask = np.roll(mask1,int(10*60/0.5),axis=0) + np.roll(mask1,-int(10*60/0.5),axis=0)
#mask = np.roll(mask1,int(10*60/0.5),axis=0) + np.roll(np.roll(mask1,-int(10*60/0.5),axis=0),int(90*60/0.5),axis=1)
#mask = np.roll(mask1.copy(),-int(10*60/0.5),axis=0) + np.roll(mask1.copy(),(int(10*60/0.5),int(45*60/0.5)),axis=(0,1))
mask = np.roll(mask1.copy(),(int(10*60/0.5),int(45*60/0.5)),axis=(0,1))

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']
#deg = 6
#afname = f'{opath}/car_mask_lmax_{lmax}_apodized_{deg:.1f}_deg.fits'
deg = 2
#afname = f'{opath}/car_mask_lmax_{lmax}_smoothed_{deg:.1f}_deg_south.fits'
#afname = f'{opath}/car_mask_lmax_{lmax}.fits'

afname = f'{opath}/car_mask_planck_lmax_{lmax}_gal.fits'
mask = enmap.read_map(afname)[0]

mask = enmap.enmap(mask,wcs)
if rank==0: io.plot_img(mask,f'{solenspipe.opath}/{args.label}_{args.polcomb}_{isostr}mask.png')
#sys.exit()

# Initialize the lens simulation interface
solint = solenspipe.SOLensInterface(mask=mask,data_mode=None,scanning_strategy="isotropic" if args.isotropic else "classical",fsky=0.4 if args.isotropic else None,white_noise=wnoise,beam_fwhm=beam)
      
w2 = solint.wfactor(2)
w3 = solint.wfactor(3)
w4 = solint.wfactor(4)
print(w2,w3,w4)

# Choose the frequency channel
channel = mapsims.SOChannel("LA", 145)

# norm dict
Als = {}
ils,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv = solint.initialize_norm(channel,lmin,lmax,recalculate=not(use_cached_norm),quicklens=quicklens,label=args.label)
Als['mv'] = al_mv
Als['mvpol'] = al_mv_pol
al_mv = Als[polcomb]
Nl = al_mv * ils*(ils+1.) / 4.


s = stats.Stats(comm)

#mf_alm = hp.read_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm.fits')


for task in my_tasks:

    # Choose a seed. This has to be varied when simulating.
    seed = (0,0,task)

    # Get the simulated, prepared T, E, B maps
    t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
    # Get the reconstructed map for the TT estimator
    recon_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,Als[polcomb]))
    #recon_alms = recon_alms - mf_alm
    
    kalms = solint.get_kappa_alm(task)
    acl = hp.alm2cl(recon_alms,recon_alms)
    xcl = hp.alm2cl(recon_alms,kalms)
    icl = hp.alm2cl(kalms,kalms)
    s.add_to_stats('acl',acl/w4)
    s.add_to_stats('xcl',xcl/w3)
    s.add_to_stats('icl',icl/w2)
    s.add_to_stack('mf',recon_alms)


s.get_stats()
s.get_stacks()

if rank==0:
    acl = s.stats['acl']['mean']
    xcl = s.stats['xcl']['mean']
    icl = s.stats['icl']['mean']
    mf_alm = s.stacks['mf']
    
    #hp.write_alm(f'{solenspipe.opath}/mf_{args.label}_{args.polcomb}_{isostr}_alm.fits',mf_alm,overwrite=True)
    mf_cl = hp.alm2cl(mf_alm,mf_alm) / w4
    ls = np.arange(xcl.size)
    Nl = maps.interp(ils,Nl)(ls)
    pl = io.Plotter('CL',xyscale='loglog')
    pl.add(ls,acl,alpha=0.5,label='rr')
    pl.add(ls,mf_cl,alpha=0.5,label='mcmf cl')
    pl.add(ls,acl-mf_cl,label='rr - mf')
    pl.add(ls,xcl,label='ri')
    pl.add(ls,icl,color='k')
    pl.add(ls,icl+Nl,ls='--',label='ii + Nl')
    #pl._ax.set_ylim(1e-9,1e-6)
    pl._ax.set_xlim(1,3100)
    pl.done(f'{solenspipe.opath}/{args.label}_{args.polcomb}_{isostr}recon.png')
                                                                                   

                                                                                

# Cross correlate with input

# Calculate autospectrum
# Subtract biases
