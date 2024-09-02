from pixell import enmap, curvedsky as cs, utils as u, lensing as plensing
import numpy as np
import os,sys
import healpy as hp
from orphics import maps, io, cosmology, stats
from falafel import utils as futils
import pytempura
import pyfisher
from enlib import bench
import solenspipe
from mpi4py import MPI
comm = MPI.COMM_WORLD

debug = False
save_map_plots = False

# Specify instrument
fwhm_arcmin = 1.5
noise_uk = 10.0
dec_min = None
dec_max = None
res = 2.0 if not(debug) else 8.0
add_noise = False

# Specify analysis
lmin = 600
lmax = 3000 if not(debug) else 1000
mlmax = 4000 if not(debug) else 1000
Lmax = 2000 if not(debug) else 1000
est = 'MV' if not(debug) else 'TT'
ests = [est]
nsims = 128 if not(debug) else 8

mg = solenspipe.LensingSandbox(fwhm_arcmin,noise_uk,dec_min,dec_max,res,lmin,lmax,mlmax,ests,add_noise=add_noise)
data_map = mg.get_observed_map(0)
Xdata = mg.prepare(data_map)
galm,calm = mg.qfuncs[est](Xdata,Xdata)
if save_map_plots: io.hplot(data_map,'data_map',downgrade=4)
rdn0 = mg.get_rdn0(Xdata,est,nsims,comm)[0]
mcn1 = mg.get_mcn1(est,nsims,comm)[0]
mcmf_alm = mg.get_mcmf(est,nsims,comm)[0]

if comm.Get_rank()==0:
    # Subtract mean-field alms and convert from phi to kappa
    galm = plensing.phi_to_kappa(galm - mcmf_alm)
    # Get the input kappa
    kalm = maps.change_alm_lmax(futils.get_kappa_alm(0),mlmax)
    if save_map_plots: io.hplot(cs.alm2map(galm,enmap.empty(mg.shape,mg.wcs,dtype=np.float32)),'kappa_map',downgrade=4)
    clkk_xx = cs.alm2cl(galm,galm) # Raw auto-spectrum (mean-field subtracted)
    clkk_ix = cs.alm2cl(kalm,galm) # Input x Recon
    clkk_ii = cs.alm2cl(kalm,kalm) # Input x Input

    # Binner
    ls = np.arange(clkk_ii.size)
    bin_edges = np.append([2,6,12,20,30,40,60],  np.arange(80,Lmax,80))
    binner = stats.bin1D(bin_edges)

    # Get theory error bars
    specs = ['kk']
    cls_dict = {'kk':lambda x : cosmology.default_theory().gCl('kk',x)}
    nls = mg.Nls[est]
    lns = np.arange(nls.size)
    nls_dict = {'kk': maps.interp(lns, nls)}
    cov = pyfisher.gaussian_band_covariance(bin_edges,specs,cls_dict,nls_dict)
    errs = np.sqrt(cov[:,0,0])

    # Bin
    cents,bclkk_ii = binner.bin(ls,clkk_ii)
    cents,bclkk_ix = binner.bin(ls,clkk_ix)
    cents,bclkk_xx = binner.bin(ls,clkk_xx)

    # Collect biases and convert from phi to kappa
    rdn0 = rdn0 * (ls*(ls+1))**2./4.
    mcn1 = mcn1 * (ls*(ls+1))**2./4.
    mcmf = cs.alm2cl(mcmf_alm) * (ls*(ls+1))**2./4. # Just for diagnostics; already subtracted
    cents,brdn0 = binner.bin(ls,rdn0)
    cents,bmcn1 = binner.bin(ls,mcn1)
    cents,bmcmf = binner.bin(ls,mcmf)

    # Plot relative difference from input
    for xscale in ['log','lin']:
        pl = io.Plotter('rCL',xyscale=f'{xscale}lin')
        pl.add(cents,bclkk_ix/bclkk_ii,label=r'$C_L^{\kappa\hat{\kappa}} / C_L^{\kappa\kappa}$ Mul. bias') # this is the multiplicative bias
        pl.add_err(cents,(bclkk_xx-brdn0-bmcn1-bmcmf)/bclkk_ii,yerr=errs/bclkk_ii,
                   label=r'$(C_L^{\hat{\kappa}\hat{\kappa}}-N_L^{0,\rm RD} - N_L^{1,\rm MC} - C_L^{\rm MCMF} ) / C_L^{\kappa\kappa}$ Add. bias') # this is the additive bias
        pl.hline(y=1)
        pl._ax.set_ylim(0.8,1.5)
        pl._ax.set_xlim(2,Lmax)
        pl.legend('outside')
        pl.done(f'rclkk_ix_{xscale}.png')

    # Plot all spectra and components
    for xscale in ['log','lin']:
        pl = io.Plotter('CL',xyscale=f'{xscale}log')
        pl.add(cents,bclkk_ix,label=r'$C_L^{\kappa\hat{\kappa}}$')
        pl.add(cents,bclkk_ii,label=r'$C_L^{\kappa\kappa}$')
        pl.add(cents,bclkk_xx,label=r'$C_L^{\hat{\kappa}\hat{\kappa}}$')
        pl.add(cents,brdn0,label=r'$N_L^{0,\rm RD}$')
        pl.add(cents,bmcn1,label=r'$N_L^{1,\rm MC}$')
        pl.add(cents,bmcmf,label=r'$C_L^{\rm MCMF}$')
        pl.add(lns,nls,label=r'$N_L$ opt. theory',ls='--')
        pl.add_err(cents,bclkk_xx-brdn0-bmcn1-bmcmf,yerr=errs,label=r'Debiased $C_L^{\hat{\kappa}\hat{\kappa}}-N_L^{0,\rm RD} - N_L^{1,\rm MC} - C_L^{\rm MCMF}$' )
        pl._ax.set_ylim(1e-9,3e-7)
        pl._ax.set_xlim(2,Lmax)
        pl.legend('outside')
        pl.done(f'clkk_ix_{xscale}.png')

