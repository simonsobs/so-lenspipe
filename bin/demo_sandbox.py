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
from solenspipe import sandbox_extensions as sb_ext
from mpi4py import MPI
comm = MPI.COMM_WORLD

# ARGPARSE

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Run the lensing sandbox.')
parser.add_argument("outname", type=str,help='Name of outputs. Could include a path.')
parser.add_argument("--estimator", type=str, default='MV',help="Estimator.")
parser.add_argument("--nsims", type=int, default=8,help="No. of RDN0 sims. Defaults to 8.")
parser.add_argument("--nsims-n1", type=int, default=None,
                    help="No. of MCN1 sims. Same as nsims if not specified.")
parser.add_argument("--n1-file", type=str, default=None,
                    help="Read N1 bias from provided output debug .txt file.")
parser.add_argument("--nsims-mf", type=int, default=None,
                    help="No. of MCMF sims. Same as nsims if not specified.")
parser.add_argument("--decmin", type=float, default=None,help="Min. declination in deg.")
parser.add_argument("--decmax", type=float, default=None,help="Max. declination in deg.")
parser.add_argument("-d", "--debug", action='store_true',
                    help='Overrides arguments and does a debug run where nsims is 8 and lmaxes are low.')
parser.add_argument("--mask", type=str, help="Path to mask .fits file, should be a pixell enmap.")
parser.add_argument("--apodize", type=float, help='Apodize the mask with input width in degrees. Only applies if a mask is passed in.')
parser.add_argument("--no-save", action='store_true',help='Dont save outputs other than plots.')
parser.add_argument("--add-noise", action='store_true',help='Whether to add noise to data and sim maps.')
parser.add_argument("--map-plots", action='store_true',help='Whether to plot data maps.')
required_args = parser.add_argument_group('Required arguments')
args = parser.parse_args()

debug = args.debug
outname = args.outname
save_map_plots = args.map_plots

# Specify instrument
fwhm_arcmin = 1.6
noise_uk = 10.0
dec_min = args.decmin
dec_max = args.decmax
res = 2.0 if not(debug) else 8.0
add_noise = args.add_noise

# Specify analysis
lmin = 600
lmax = 3000 if not(debug) else 1000
mlmax = 4000 if not(debug) else 1000
Lmax = 2000 if not(debug) else 1000
est = args.estimator if not(debug) else 'TT'
ests = [est]
nsims_rdn0 = args.nsims if not(debug) else 8
nsims_n1 = args.nsims_n1 if not(args.nsims_n1 is None) else nsims_rdn0
nsims_mf = args.nsims_mf if not(args.nsims_mf is None) else nsims_rdn0

# Placeholder until better way of inferring downgrade resolution
if args.mask is not None:
    mask = enmap.read_map(args.mask)
    mask = enmap.downgrade(mask, int(res / (21600 / mask.shape[1])),
                           op=np.mean)
    if args.apodize is not None:
        mask = maps.cosine_apodize(mask, args.apodize)
else:
    mask = args.mask

# data sim index
DATA_SIM_INDEX = 50
START_INDEX = DATA_SIM_INDEX

mg = sb_ext.LensingSandboxILC(fwhm_arcmin,noise_uk,dec_min,dec_max,res,
                              lmin,lmax,mlmax,ests,start_index=START_INDEX,
                              downgrade_res=res,add_noise=add_noise,
                              mask=mask,verbose=True)
data_map = mg.get_observed_map(DATA_SIM_INDEX)
Xdata = mg.prepare(data_map)
galm,calm = mg.qfuncs[est](Xdata,Xdata)
if save_map_plots: io.hplot(data_map,f'{outname}_data_map',downgrade=4)
rdn0 = mg.get_rdn0(Xdata,est,nsims_rdn0,comm)[0]

if args.n1_file is not None:
    try:
        mcn1 = np.loadtxt(args.n1_file, usecols=[5])
        print(f"Loaded MCN1 from {args.n1_file}, skipping N1 calculation.")
    except FileNotFoundError:
        mcn1 = mg.get_mcn1(est,nsims_n1,comm)[0]
else:
    if nsims_n1 > 0:
        mcn1 = mg.get_mcn1(est,nsims_n1,comm)[0]
    else:
        mcn1 = rdn0 * 0.

if nsims_mf==0:
    print("Skipping meanfield...")
    mcmf_alm_1 = 0.
    mcmf_alm_2 = 0.
else:
    print("Meanfield...")
    mcmf_alm_1, mcmf_alm_2 = mg.get_mcmf(est,nsims_mf,comm)
    # Get only gradient components of mcmf
    mcmf_alm_1 = mcmf_alm_1[0]
    mcmf_alm_2 = mcmf_alm_2[0]

if comm.Get_rank()==0:
    # Subtract mean-field alms and convert from phi to kappa
    galm_1 = plensing.phi_to_kappa(galm - mcmf_alm_1)
    galm_2 = plensing.phi_to_kappa(galm - mcmf_alm_2)
    # Get the input kappa
    kalm = maps.change_alm_lmax(futils.get_kappa_alm(DATA_SIM_INDEX),mlmax)
    if save_map_plots: io.hplot(cs.alm2map(galm,
                                           enmap.empty(mg.shape,
                                                       mg.wcs,
                                                       dtype=np.float32)),
                                f'{outname}_kappa_map',downgrade=4)
    clkk_xx = cs.alm2cl(galm_1,galm_2)/mg.w4 # MF-subtracted raw auto-spectrum
    # Cross-correlate input with the averaged MF-subtracted alms
    clkk_ix = cs.alm2cl(kalm,0.5*(galm_1+galm_2))/mg.w2 # Input x Recon
    clkk_ii = cs.alm2cl(kalm,kalm) # Input x Input
    ls = np.arange(clkk_ii.size)

    # Binner
    bin_edges = np.append([2,6,12,20,30,40,60],  np.arange(80,Lmax,80))
    binner = stats.bin1D(bin_edges)

    # Get theory error bars
    specs = ['kk']
    cls_dict = {'kk':lambda x : cosmology.default_theory().gCl('kk',x)}
    nls = mg.Nls[est]
    lns = np.arange(nls.size)
    nls_dict = {'kk': maps.interp(lns, nls)}
    cov = pyfisher.gaussian_band_covariance(bin_edges,specs,cls_dict,nls_dict)
    errs = np.sqrt(cov[:,0,0] / mg.w4) # TODO: double-check fsky factor

    # Bin
    cents,bclkk_ii = binner.bin(ls,clkk_ii)
    cents,bclkk_ix = binner.bin(ls,clkk_ix)
    cents,bclkk_xx = binner.bin(ls,clkk_xx)

    # Collect biases and convert from phi to kappa
    rdn0 = rdn0 * (ls*(ls+1))**2./4. / mg.w4
    if args.n1_file is None:
        mcn1 = mcn1 * (ls*(ls+1))**2./4. / mg.w4

    if nsims_mf>0:
        # Just for diagnostics; already subtracted
        mcmf = cs.alm2cl(mcmf_alm_1, mcmf_alm_2) * (ls*(ls+1))**2./4. / mg.w4 
        cents,bmcmf = binner.bin(ls,mcmf)
    else:
        mcmf = clkk_xx * 0.
        bmcmf = 0.
    cents,brdn0 = binner.bin(ls,rdn0)
    cents,bmcn1 = binner.bin(ls,mcn1)
    bclkk_final = bclkk_xx-brdn0-bmcn1 # Final debiased power spectrum

    if not(args.no_save): io.save_cols(f"{outname}_output_clkk.txt",
                                       (ls,clkk_ii,clkk_xx,clkk_ix,rdn0,
                                        mcn1,mcmf,clkk_xx-rdn0-mcn1))

    # Plot relative difference from input
    for xscale in ['log','lin']:
        pl = io.Plotter('rCL',xyscale=f'{xscale}lin')
        # this is the multiplicative bias
        pl.add(cents,bclkk_ix/bclkk_ii,
               label=r'$C_L^{\kappa\hat{\kappa}} / C_L^{\kappa\kappa}$ Mul. bias')
        # this is the additive bias
        pl.add_err(cents,bclkk_final/bclkk_ii,yerr=errs/bclkk_ii,
                   label=r'$(C_L^{\hat{\kappa}\hat{\kappa}}-N_L^{0,\rm RD} - N_L^{1,\rm MC} ) / C_L^{\kappa\kappa}$ Add. bias')
        pl.hline(y=0)
        pl._ax.set_ylim(-0.3,0.3)
        pl._ax.set_xlim(2,Lmax)
        pl.legend('outside')
        pl.done(f'{outname}_rclkk_ix_{xscale}.png')

    # Plot all spectra and components
    for xscale in ['log','lin']:
        pl = io.Plotter('CL',xyscale=f'{xscale}log')
        pl.add(cents,bclkk_ix,label=r'$C_L^{\kappa\hat{\kappa}}$')
        pl.add(cents,bclkk_ii,label=r'$C_L^{\kappa\kappa}$')
        pl.add(cents,bclkk_xx,label=r'$C_L^{\hat{\kappa}\hat{\kappa}}$')
        pl.add(cents,brdn0,label=r'$N_L^{0,\rm RD}$ ' + f'({nsims_rdn0} sims)')
        pl.add(cents,bmcn1,label=r'$N_L^{1,\rm MC}$ ' + (f'({nsims_n1} sims)' if args.n1_file is None
                                                                             else '(from file)'))
        if nsims_mf>0: pl.add(cents,bmcmf,label=r'$C_L^{\rm MCMF}$ ' + f'({nsims_mf} sims)')
        pl.add(lns,nls,label=r'$N_L$ opt. theory',ls='--')
        pl.add_err(cents,bclkk_final,yerr=errs,
                   label=r'Debiased $C_L^{\hat{\kappa}\hat{\kappa}}-N_L^{0,\rm RD} - N_L^{1,\rm MC} $')
        pl._ax.set_ylim(1e-11,3e-7)
        pl._ax.set_xlim(2,Lmax)
        pl.legend('outside')
        pl.done(f'{outname}_clkk_ix_{xscale}.png')

