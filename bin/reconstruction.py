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
import matplotlib.pyplot as plt
import reconstruction_bias

"""
We will do a simple lensing reconstruction test.
"""

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("label", type=str,help='Label.')
parser.add_argument("polcomb", type=str,help='polcomb.')
parser.add_argument("-N", "--nsims",     type=int,  default=10,help="Number of sims.")
parser.add_argument("--sindex",     type=int,  default=0,help="Declination band.")
parser.add_argument("--lmin",     type=int,  default=100,help="Minimum multipole.")
parser.add_argument("--lmax",     type=int,  default=3000,help="Minimum multipole.")
parser.add_argument("--isotropic", action='store_true',help='Isotropic sims.')
parser.add_argument("--no-atmosphere", action='store_true',help='Disable atmospheric noise.')
parser.add_argument("--use-cached-norm", action='store_true',help='Use  cached norm.')
parser.add_argument("--wnoise",     type=float,  default=None,help="Override white noise.")
parser.add_argument("--beam",     type=float,  default=None,help="Override beam.")
parser.add_argument("--disable-noise", action='store_true',help='Disable noise.')
parser.add_argument("--zero-sim", action='store_true',help='Just make a sim of zeros. Useful for benchmarking.')
parser.add_argument("--healpix", action='store_true',help='Use healpix.')
parser.add_argument("--no-mask", action='store_true',help='No mask. Use with the isotropic flag.')
parser.add_argument("--debug", action='store_true',help='Debug plots.')
parser.add_argument("--flat-sky-norm", action='store_true',help='Use flat-sky norm.')

args = parser.parse_args()

solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr = solenspipe.initialize_args(args)
      
w2 = solint.wfactor(2)
w3 = solint.wfactor(3)
w4 = solint.wfactor(4)
car = "healpix_" if args.healpix else "car_"
noise="wnoise" if args.wnoise!=None else "sonoise"
mask="nomask" if args.no_mask else "mask"



seed=(0,0,0)
#prepare the talms
t_alm,e_alm,b_alm = solint.get_kmap(channel,seed,lmin,lmax,filtered=True)
#reconstructed biased alms
recon_alms = qe.filter_alms(solint.get_mv_kappa(polcomb,t_alm,e_alm,b_alm),maps.interp(ils,Als[polcomb]))
kalms = solint.get_kappa_alm(0)

#remove mean field
label=args.label
reconstruction_bias.get_mf_alms(solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr,label)
mfalm=hp.read_alm(f'{solenspipe.opath}mf_{label}_{polcomb}_{isostr}_{nsims}_alm.fits')
print("loaded mfalm")

acl = hp.alm2cl(recon_alms-mfalm,recon_alms-mfalm)/w4
xcl = hp.alm2cl(recon_alms,kalms)/w3
icl = hp.alm2cl(kalms,kalms)/w2
ls = np.arange(xcl.size)
Nl = maps.interp(ils,Nl)(ls)
#theory N1
N1 = {}
n1bins,N1['TT'],N1['EE'],N1['EB'],N1['TE'],N1['TB'],N1['mv']=solint.analytic_n1(channel,lmin,lmax,Lmin_out=2,Lmaxout=3000,Lstep=20)
n1_mv = N1[polcomb]
N1l = n1_mv * (n1bins*(n1bins+1.))**2 / 4.
N1l=maps.interp(n1bins,N1l)(ls)

#call for rdn0 and mcn1
print("Calculating RDNO")
rdn0=reconstruction_bias.recon_rdn0(solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr,label)

print("Calculating MCN1")
mcn1=reconstruction_bias.recon_mcn1(solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr,label)
rdn0=maps.interp(rdn0[0],rdn0[1])(ls)
mcn1=maps.interp(mcn1[0],mcn1[1])(ls)
auto=acl-rdn0-mcn1+rdn0/nsims

#Put these into bandpowers
weights=(icl/(icl+Nl))**2
bin_edges=np.geomspace(2,3000,20).astype(int)
binner = solenspipe.weighted_bin1D(bin_edges)
cents,autob = binner.bin(ls,auto,weights)
cents,mcn1b = binner.bin(ls,mcn1,weights)
cents,rdn0b = binner.bin(ls,rdn0,weights)

#theory covariance matrix
cov=np.ones(len(autob))
f_sky=0.4
for i in range(len(autob)):
    cov[i]=(1/(cents[i]*np.diff(bin_edges)[i]*f_sky))*(autob[i]+rdn0b[i]+mcn1b[i])**2
                                                           
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
plt.errorbar(cents,autob,color='b',yerr=np.sqrt(cov),fmt='.',label="Autopower spectrum before simulation correction")
plt.plot(cents,mcn1b,'-.',color='g',label="$N1$ bias",alpha=0.7)
plt.semilogx(cents,rdn0b,'-.',color='k',label="$N0$ bias",alpha=0.7)


config = io.config_from_yaml("../input/config.yml")
thloc = "../data/" + config['theory_root']
theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

clkk=theory.gCl('kk',ls)
plt.semilogx(ls,clkk[:4001],color='k',linewidth=0.5)
plt.axhline(linewidth=0.9, color='k')
plt.ylabel("$C^{\kappa\kappa}_L$")
plt.ylim(-0.1e-7,2.5e-7)
plt.xlim(2,3000)
plt.xlabel('$L$')
plt.legend()
plt.savefig(f'{solenspipe.opath}auto.png")


####################################################################################################

    
