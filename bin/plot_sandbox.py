import numpy as np
from orphics import stats, io, cosmology, maps
from falafel import utils as futils
from pixell import enmap
import pytempura
import pyfisher
import argparse

# Parse command line
parser = argparse.ArgumentParser(description='Run the lensing sandbox.')
parser.add_argument("--mask", type=str, default=None,
                    help="Path to mask .fits file, should be a pixell enmap.")
parser.add_argument("in_name", type=str, help='Input "..._output_clkk.txt" file. Could include a path.')
parser.add_argument("out_name", type=str, help='Name of outputs. Could include a path.')
required_args = parser.add_argument_group('Required arguments')
args = parser.parse_args()

in_name = args.in_name
out_name = args.out_name

if args.mask is not None:
    mask = enmap.read_map(args.mask)
    w4 = maps.wfactor(4, mask)
else:
    mask = None
    w4 = 1.0

ls,clkk_ii,clkk_xx,clkk_ix,rdn0,mcn1,mcmf,debiased = np.loadtxt(in_name, unpack=True)
lmin = 600
lmax = 3000
Lmax = 2000

bin_edges = np.append([2,6,12,20,30,40,60],  np.arange(80,Lmax,80))
binner = stats.bin1D(bin_edges)

ucls, tcls = futils.get_theory_dicts_white_noise(1.6, 10., grad=True, lmax=lmax)
Als = pytempura.get_norms(['MV'], ucls, ucls, tcls, lmin, lmax)
ls = np.arange(Als['MV'][0].size)

specs = ['kk']
cls_dict = {'kk':lambda x : cosmology.default_theory().gCl('kk',x)}
nls = Als['MV'][0] * (ls*(ls+1.)/2.)**2.
lns = np.arange(nls.size)
nls_dict = {'kk': maps.interp(lns, nls)}
cov = pyfisher.gaussian_band_covariance(bin_edges,specs,cls_dict,nls_dict)
errs = np.sqrt(cov[:,0,0] / w4) # TODO: double-check fsky factor

ells = np.arange(clkk_ii.size)
cents,bclkk_ii = binner.bin(ells,clkk_ii)
cents,bclkk_ix = binner.bin(ells,clkk_ix)
cents,bclkk_xx = binner.bin(ells,clkk_xx)
cents,brdn0 = binner.bin(ells,rdn0)
cents,bmcn1 = binner.bin(ells,mcn1)
cents,bmcmf = binner.bin(ells,mcmf)

bclkk_final = bclkk_xx-brdn0-bmcn1

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
    pl.done(f'{out_name}_rclkk_ix_{xscale}.png')

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
    pl.add_err(cents,bclkk_final,yerr=errs,
                label=r'Debiased $C_L^{\hat{\kappa}\hat{\kappa}}-N_L^{0,\rm RD} - N_L^{1,\rm MC} $')
    pl._ax.set_ylim(1e-11,3e-7)
    pl._ax.set_xlim(2,Lmax)
    pl.legend('outside')
    pl.done(f'{out_name}_clkk_ix_{xscale}.png')
