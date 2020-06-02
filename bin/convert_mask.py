from __future__ import print_function
from orphics import io
from pixell import enmap,reproject
import numpy as np
import os,sys
import healpy as hp

from orphics.maps import galactic_mask, north_galactic_mask, south_galactic_mask



config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']
mask = hp.read_map(opath + config['mask_name'])
io.mollview(mask,'hp_carmask.png')
print(mask.max(),mask.min())
nside = hp.nside2npix(mask.size)

lmax = 3000

# CAR resolution is decided based on lmax
res = np.deg2rad(2.0 *(3000/lmax) /60.)

# Make the full sky geometry
shape,wcs = enmap.band_geometry(np.deg2rad((-70,30)),res=res)
#shape,wcs = enmap.fullsky_geometry(res=res)

# nside = 256
# nmask = north_galactic_mask(shape,wcs,nside)
# io.plot_img(nmask,'low_north_gal.png')

# nside = 256
# smask = south_galactic_mask(shape,wcs,nside)
# io.plot_img(smask,'low_south_gal.png')

#sys.exit()



imap = reproject.ivar_hp_to_cyl(mask, shape, wcs, rot=False,do_mask=False,extensive=False)

fname = f'{opath}/car_mask_lmax_{lmax}.fits'
enmap.write_map(fname,imap)

deg = 2.0

smoothed = enmap.smooth_gauss(imap,np.deg2rad(deg))

fname = f'{opath}/car_mask_lmax_{lmax}_smoothed_{deg:.1f}_deg.fits'
enmap.write_map(fname,smoothed)

# fname = f'{opath}/car_mask_lmax_{lmax}_smoothed_{deg:.1f}_deg_south.fits'
# enmap.write_map(fname,smoothed*nmask)

# fname = f'{opath}/car_mask_lmax_{lmax}_smoothed_{deg:.1f}_deg_north.fits'
# enmap.write_map(fname,smoothed*smask)

# io.plot_img(smoothed,'sm_low_carmask.png')
# io.plot_img(smoothed*nmask,'sm_low_south_carmask.png')
# io.plot_img(smoothed*smask,'sm_low_north_carmask.png')



deg = 2.0

r = np.deg2rad(deg)
apodized = 0.5*(1-np.cos(imap.distance_transform(rmax=r)*(np.pi/r)))

io.plot_img(imap,'low_carmask.png')
#io.hplot(imap,'carmask')

io.plot_img(apodized,'low_ap_carmask')
#io.hplot(apodized,'ap_carmask')

afname = f'{opath}/car_mask_lmax_{lmax}_apodized_{deg:.1f}_deg.fits'
enmap.write_map(afname,apodized)



