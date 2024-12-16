from orphics import maps
from pixell import enmap, utils as u, curvedsky as cs
import numpy as np


def get_metadata(qid,splitnum):
    """
    SOFind-aware function to get map metadata
    """
    meta = bunch.Bunch({})
    if is_planck(qid):
        meta.beam_fells = get_planck_beam(qid,pixwin=True)
        meta.transfer_fells = 1.0
        meta.calibration = 1.0
        meta.pol_eff = 1.0
        meta.inpaint_mask = None
        meta.kspace_mask = None
        meta.maptype = 'reprojected'
        meta.nsplits = 2
        if splitnum==0 or splitnum==1:
            isplit = 0
        elif splitnum==2 or splitnum==3:
            isplit = 1
    else:
        meta.beam_fells = get_act_beam(qid)
        meta.transfer_fells = get_act_transfer(qid)
        meta.calibration = get_act_cal(qid)
        meta.pol_eff = get_act_poleff(qid)
        meta.inpaint_mask = get_inpaint_mask(inpaint_cat_version)
        meta.kspace_mask = get_kspace_mask(lxcut,lycut)
        meta.maptype = 'native'
        meta.nsplits = 4
        isplit = splitnum
    return meta, isplit


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

def depix_map(imap,maptype='native',dfact=None,kspace_mask=None):
    """
    Remove rectangular pixel window effects from the map.
    If the map was produced in native rect pixelization,
    then what factor it was downgraded by (dfact) is not relevant
    and can be ignored, with the deconvolved window corresponding
    to that of the current map.

    If the map was reprojected (e.g. from healpix), then the
    ratio of pixel windows is used.  This function will not
    account for the original healpix pixel window.

    This will perform necessary Fourier operations on input maps.
    This includes:
    1. Correcting for a map-maker pixel window
    2. Correctiong for a downgrade pixel window
    3. k-space masking

    
    (1) and (3) are only done if the map originates from ACT/SO
    (2) is done if the map was downgraded

    """
    wy1, wx1 = enmap.calc_window(imap.shape)
    if maptype=='native':
        wy2 = wx2 = 1
    elif maptype=='reprojected':
        if (dfact is None) or (dfact==1):
            if kspace_mask is None: return imap
            wy2 = wx2 = wy1 = wx1 = 1
        else:
            wy2, wx2 = enmap.calc_window(imap.shape, scale=dfact)
    else:
        raise ValueError
    fmap = enmap.fft(imap)
    fmap *= (wy2/wy1)[:,None]
    fmap *= (wx2/wx1)
    if kspace_mask is not None:
        fmap[:,~kspace_mask] = 0
    imap = enmap.ifft(fmap).real
    return imap


def obtain_kspacemask(shape,wcs, vk_mask=None, hk_mask=None):
    
    lymap, lxmap = enmap.lmap(shape, wcs)
    ly, lx = lymap[:,0], lxmap[0,:]
    
    if vk_mask is not None:
        id_vk = np.where((lx > vk_mask[0]) & (lx < vk_mask[1]))
    if hk_mask is not None:
        id_hk = np.where((ly > hk_mask[0]) & (ly < hk_mask[1]))
    
    # i don't know how to continue this
    
    # ft[...,: , id_vk] = 0.
    # ft[...,id_hk,:]   = 0.
    
    kspace_mask = np.empty((2, ly.shape, lx.shape))
    kspace_mask[0] = id_vk[]
    kspace_mask[1] = id_hk
    # !!!!
    return kspace_mask    



def preprocess_core(imap, ivar, mask,
                    calibration, pol_eff,
                    maptype='native',
                    dfact = None,
                    inpaint_mask=None,
                    vk_mask=None, hk_mask=None):
    """
    This function will load a rectangular pixel map and pre-process it.
    This involves inpainting, masking in real and Fourier space
    and removing a pixel window function. It also removes a calibration
    and polarization efficiency.

    Returns beam convolved (transfer uncorrected) T, Q, U maps.
    """
    if dfact!=1 and (dfact is not None):
        imap = enmap.downgrade(imap,dfact)
        ivar = enmap.downgrade(ivar,dfact,op=np.sum)
        
    if inpaint_mask:
        imap = maps.gapfill_edge_conv_flat(imap, inpaint_mask, ivar=ivar)
        
    if vk_mask is not None or hk_mask is not None:
        kspace_mask = obtain_kspacemask(imap.shape, imap.wcs, vk_mask=vk_mask, hk_mask=hk_mask)
    else:
        kspace_mask = None

    imap = imap * mask
    imap = depix_map(imap,maptype=maptype,dfact=dfact,kspace_mask=kspace_mask)
        
    imap = imap * calibration
    imap[1:] = imap[1:] / pol_eff
    
    ivar = ivar / calibration**2.
    ivar[1:] = ivar[1:] * pol_eff**2.
    return imap, ivar


def get_sim_core(shape,wcs,signal_alms,
                 beam_fells, transfer_fells,
                 calibration,pol_eff,
                 maptype='native',
                 noise_alms=None,
                 apod_y_arcmin = 0.,
                 apod_x_arcmin = 0.,
                 noise_mask=None,
                 rms_uk_arcmin=None,
                 noise_lmax = 5400,
                 lcosine=80): # these are all optional arguments for noise stitching

    """
    shape,wcs should ideally correspond to native ACT/SO pixelization of 0.5 arcmin
    beam_fells and transfer_fells should be same size and go from ell=0 to beyond relevant ells

    """
    
    signal_alms[0] = cs.almxfl(signal_alms[0],beam_fells * transfer_fells)
    signal_alms[1] = cs.almxfl(signal_alms[1],beam_fells)
    signal_alms[2] = cs.almxfl(signal_alms[2],beam_fells)
    omap = cs.alm2map(signal_alms,enmap.empty((3,)+shape,wcs,dtype=np.float32))
    if maptype=='native':
        if (apod_y_arcmin>1e-3) or (apod_x_arcmin>1e-3):
            res = maps.resolution(shape,wcs) / u.arcmin
            omap = enmap.apod(omap, (apod_y_arcmin/res,apod_x_arcmin/res))
        omap = enmap.apply_window(omap,pow=1)
    elif maptype=='reprojected':
        pass
    else:
        raise ValueError        

    if noise_alms is not None:
        if noise_mask:
            lstitch = noise_lmax - 200
            mlmax = noise_lmax + 600
            nmap = maps.stitched_noise(shape,wcs,noise_alms,noise_mask,rms_uk_arcmin=rms_uk_arcmin,
                                       lstitch=lstitch,lcosine=lcosine,mlmax=mlmax)
        else:
            nmap = cs.alm2map(noise_alms,enmap.empty((3,)+shape,wcs,dtype=np.float32))
    else:
        nmap = 0.
        
    omap = omap + nmap
    # notice how these are inverse of what's in preprocess
    omap = omap / calibration  
    omap[1:] = omap[1:] * pol_eff
    return omap


def calculate_noise_power(qid, args, mask,
                          calibration, pol_eff,
                          maptype='native',
                          dfact = None,
                          inpaint_mask=None,
                          vk_mask=None, hk_mask=None):
    
    '''
    args:
    args.config_name: str, sofind datamodel, e.g. 'act_dr6v4'
    args.maps_subproduct: str, subproduct name for maps, e.g. 'default'
    
    '''
    
    datamodel = DataModel.from_config(args.config_name)
    qid_kwargs = datamodel.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
    
    nsplits = qid_kwargs['num_splits']
    
    coadd_map = datamodel.read_map(qid=qid, coadd=True, subproduct=args.maps_subproduct, maptag='map_srcfree')

    # load map and ivar splits
    ivar_splits = [] 
    map_splits = []
    for k in range(nsplits):
        ivar_splits.append(datamodel.read_map(qid=qid, split_num=k, subproduct=args.maps_subproduct, maptag='ivar'))
        map_splits.append(datamodel.read_map(qid=qid, split_num=k, subproduct=args.maps_subproduct, maptag='map_srcfree'))
    
    nmaps = 0.
    for k in range(nsplits):
        diff = coadd - map_splits[k]
        ivar = utility.ivar_eff(k,ivar_splits)
        nmap, nivar = preprocess_core(diff, ivar, mask,
                                      calibration, pol_eff,
                                      maptype=maptype,
                                      dfact=dfact,
                                      inpaint_mask=inpaint_mask,
                                      vk_mask=vk_mask, hk_mask=hk_mask)
        nmaps = nmaps + nmap

    nmaps = nmaps / nsplits
    
    Ealm,Balm=pureEB(noise_a[1],noise_a[2],mask,returnMask=0,lmax=lmax,isHealpix=False)
    alm_T=cs.map2alm(noise_a[0],lmax=lmax)
    alm_a=np.array([alm_T,Ealm,Balm])
    alm_a=alm_a.astype(np.complex128)
    ????????
    cl_ab = cs.alm2cl(alm_a)
    w2=w_n(mask,2)
    cl_sum = np.sum(cl_ab, axis=0) # is this necessary anymore, just one now
    power = 1/n_splits/(n_splits-1) * cl_sum
    power[~np.isfinite(power)] = 0
    power/=w2
    
    return nmaps


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
