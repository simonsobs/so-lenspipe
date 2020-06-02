from __future__ import print_function
from orphics import io
from pixell import enmap,reproject
import numpy as np
import os,sys
import healpy as hp

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

lmax = 3000

# CAR resolution is decided based on lmax
res = np.deg2rad(2.0 *(3000/lmax) /60.)

pmask = hp.read_map("/scratch/r/rbond/msyriac/data/planck/data/pr3/COM_Lensing_4096_R3.00/mask.fits.gz")
shape,wcs = enmap.fullsky_geometry(res=res)

imap = reproject.ivar_hp_to_cyl(pmask, shape, wcs, rot=False,do_mask=False,extensive=False)
fname = f'{opath}/car_mask_planck_lmax_{lmax}_gal.fits'
enmap.write_map(fname,imap)

imap = reproject.ivar_hp_to_cyl(pmask, shape, wcs, rot=True,do_mask=False,extensive=False)
fname = f'{opath}/car_mask_planck_lmax_{lmax}_equ.fits'
enmap.write_map(fname,imap)
