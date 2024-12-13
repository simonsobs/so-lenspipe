from orphics import maps

# Note: downgrading not added yet

def preprocess_core(imap, ivar, mask,
               inpaint_mask, kspace_mask,
               calibration, pol_eff,
               is_binary_mask=False):
    """
    This function will load a rectangular pixel map and pre-process it.
    This involves inpainting, masking in real and Fourier space
    and removing a pixel window function. It also removes a calibration
    and polarization efficiency.
    It assumes the map has already been downgraded.
    """

    imap = maps.gapfill_edge_conv_flat(imap, inpaint_mask, ivar=ivar)
    if is_binary_mask:
        imap[!mask] = 0
    else:
        imap = imap * mask
    fmap = enmap.fft(imap)
    fmap = enmap.apply_window(imap,pow=-1,nofft=True)
    fmap[!kspace_mask] = 0
    imap = enmap.ifft(fmap).real
    imap = imap * calibration
    imap[1:] = imap[1:] / pol_eff
    return imap

# This needs to be expanded to access sofind
def get_sim(qid,dfact,
            lstitch=5200,lcosine=80,mlmax=6000,alpha=-4,flmin = 700): # these are all optional arguments for noise stitching
    # load geometry
    shape,wcs # non-downgraded : get from sofind
    # load alex alms
    alm_file # get from sofind
    alms = hp.read_alm(alm_file,hdu=(1,2,3))
    alms_fg = get_foreground_power() # get from sofind? !!!
    alms = alms + alms_fg
    eff_beam_transfer_fells = beam_fells * transfer_fells # get from sofind?
    alms[0] = cs.almxfl(alms[0],eff_beam_transfer_fells)
    alms[1:] = cs.almxfl(alms[1:],beam_fells)
    omap = enmap.alm2map(alms,enmap.empty((3,)+shape,wcs,dtype=np.float32))
    omap = enmap.apod(omap) # 10 arcmin
    omap = enmap.apply_window(omap,pow=1)
    nalms = enmap.read_map(noise_map_alm_file) # get from sofind?
    # Add white noise beyond lmax of 5200
    nmap = maps.stitched_noise(shape,wcs,nalms,noise_mask,lstitch=5200,lcosine=80,mlmax=6000,alpha=-4,flmin = 700)
    omap = omap + nmap
    # notice how these are inverse of what's in preprocess
    omap = imap / calibration  
    omap[1:] = imap[1:] * pol_eff
    
    return enmap.downgrade(omap,dfact)
