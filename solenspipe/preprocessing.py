from orphics import maps
from pixell import enmap, utils as u, curvedsky as cs
import numpy as np

"""
Some subtleties when applying this to different experiments:

A CAR map at X resolution has a pixel window p(X).
A block-downgrade operation to Y resolution will induce an additional
pixel window of p(Y)/p(X).

Therefore, the correct pixel window to deconvolve from the block-downgraded
map is p(Y).

Planck maps have a healpix window function h(nside).  The CAR reprojected
maps do not correct for this and do not contain the usual p(X) pixel window
either since they are reprojected using SHTs. Therefore, if you downgrade
from X to Y, the pixel window that needs to be deconvolved is:
h(nside)p(Y)/p(X).

"""

def preprocess_core(imap, mask,
                    calibration, pol_eff,
                    inpaint_ivar=None,
                    inpaint_mask=None,
                    lxcut=None,lycut=None,kspace_mask=None,
                    is_binary_mask=False):
    """
    This function will load a rectangular pixel map and pre-process it.
    This involves inpainting, masking in real and Fourier space
    and removing a pixel window function. It also removes a calibration
    and polarization efficiency.
    It assumes the map has already been downgraded.

    Returns beam convolved (transfer uncorrected) T, Q, U maps.
    """

    if inpaint_mask:
        imap = maps.gapfill_edge_conv_flat(imap, inpaint_mask, ivar=inpaint_ivar)

    if is_binary_mask:
        imap[~mask] = 0
    else:
        imap = imap * mask
        
    if lxcut is not None:
        imap = utility.kspace_mask(imap, vk_mask=[-lxcut,lxcut], hk_mask=[-lycut,lycut], normalize="phys", deconvolve=True)
    else:
        fmap = enmap.fft(imap)
        fmap = enmap.apply_window(fmap,pow=-1,nofft=True)
        fmap[:,~kspace_mask] = 0
        imap = enmap.ifft(fmap).real
        
    imap = imap * calibration
    imap[1:] = imap[1:] / pol_eff
    
    ivar = ivar / calibration**2.
    ivar[1:] = ivar[1:] * pol_eff**2.
    return imap, ivar


def get_sim_core(shape,wcs,signal_alms,noise_alms,
            beam_fells, transfer_fells,
            calibration,pol_eff,
            dfact,
            apod_y_arcmin = 0.,
            apod_x_arcmin = 10.,
            noise_mask=None,
            noise_lmax = 5400,
            lcosine=80,alpha=-4,flmin = 700): # these are all optional arguments for noise stitching

    """
    shape,wcs should ideally correspond to native ACT/SO pixelization of 0.5 arcmin
    beam_fells and transfer_fells should be same size and go from ell=0 to beyond relevant ells

    """
    
    signal_alms[0] = cs.almxfl(signal_alms[0],beam_fells * transfer_fells)
    signal_alms[1] = cs.almxfl(signal_alms[1],beam_fells)
    signal_alms[2] = cs.almxfl(signal_alms[2],beam_fells)
    omap = cs.alm2map(signal_alms,enmap.empty((3,)+shape,wcs,dtype=np.float32))
    
    res = maps.resolution(shape,wcs) / u.arcmin
    omap = enmap.apod(omap, (apod_y_arcmin/res,apod_x_arcmin/res))
    omap = enmap.apply_window(omap,pow=1)
    if noise_mask:
        lstitch = noise_lmax - 200
        mlmax = noise_lmax + 600
        nmap = maps.stitched_noise(shape,wcs,noise_alms,noise_mask,
                                   lstitch=lstitch,lcosine=lcosime,mlmax=mlmax,alpha=alpha,flmin = flmin)
    else:
        nmap = cs.alm2map(noise_alms,enmap.empty((3,)+shape,wcs,dtype=np.float32))
    omap = omap + nmap
    # notice how these are inverse of what's in preprocess
    omap = omap / calibration  
    omap[1:] = omap[1:] * pol_eff
    return enmap.downgrade(omap,dfact)


# Note: downgrading not added yet

# orphics.maps.kspace_coadd_alms
# pure E,B:   Q, U maps and a mask -> E, B alms
# kspace_coadd:  T, E, B alms and noise spectra -> T, E, B alms

"""
kspace_coadd_sims:

for qid in qids:
  sim = get_sim(qid)
  psim = preprocess_core(sim)
  kmaps = map2alm(psim)  OR  pure_eb(psim)

kcoadd = kspace_coadd_alms(kmaps) # beam deconvolved kspace coadd alms

"""

"""
def calculate_noise_power(qid):
    coadd # load downgraded coadd map
    #ivar_coadd
    splits # load downgraded splits
    ivar_splits

    nmaps = 0.
    for k in range(nsplits):
        diff = coadd - splits[k]
        ivar = utility.ivar_eff(k,ivar_splits)
        nmap, nivar = preprocess_core(diff, mask,
                                      calibration, pol_eff)
        nmaps = nmaps + nmap

    nmaps = nmaps / nsplits
    
"""

"""
    # load geometry
    shape,wcs # non-downgraded : get from sofind
    # load alex alms
    alm_file # get from sofind
    alms = hp.read_alm(alm_file,hdu=(1,2,3))
    signal_alms = alms + fg_power_alms_qid

    beam_fells
    transfer_fells # get from sofind?
    nalms = enmap.read_map(noise_map_alm_file,seed_noise) # get from sofind?
    
    return get_sim_core(shape,wcs,signal_alms,noise_alms,
                        beam_fells, transfer_fells,
                        calibration,pol_eff,
                        dfact,
                        apod_y_arcmin = 0.,
                        apod_x_arcmin = 10.,
                        noise_mask=None,
                        noise_lmax = 5400,
                        **kwargs)
    
"""
