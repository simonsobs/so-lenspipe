from orphics import maps
from pixell import enmap, utils as u, curvedsky as cs
import numpy as np
from mnms import noise_models as nm
from sofind import DataModel
from pixell import bunch
from solenspipe import utility as simgen
import os

specs_weights = {'QU': ['I','Q','U'],
        'EB': ['I','E','B']}
nspecs = len(specs_weights['QU'])

def is_planck(qid):
    if qid in ['p01','p02','p03','p04','p05', 'p06', 'p07']:
        return True
    else:
        return False
    
def get_inpaint_mask(args):
    
    '''
    args.config_name: str, sofind datamodel, e.g. 'act_dr6v4'
    args.cat_date: str, date of inpaint catalog, e.g. '20241002'
    args.regular_hole: float, radius of hole [arcmin] for regular sources
    args.large_hole: float, radius of hole [arcmin] for large sources
    args.shape: tuple, shape of mask
    args.wcs: wcs object, wcs of mask
    '''
    
    datamodel = DataModel.from_config(args.config_name)
    
    # read catalog coordinates
    rdecs, rras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_regular_{args.cat_date}.csv', subproduct = 'inpaint_catalogs'))
    ldecs, lras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_large_{args.cat_date}.csv', subproduct = 'inpaint_catalogs'))

    # Make masks for gapfill
    mask1 = maps.mask_srcs(args.shape,args.wcs,np.asarray((ldecs,lras)),args.large_hole)
    mask2 = maps.mask_srcs(args.shape,args.wcs,np.asarray((rdecs,rras)),args.regular_hole)
    jmask = mask1 & mask2
    jmask = ~jmask
    
    return jmask

def get_metadata(qid, splitnum=0, coadd=False, args=None):
    """
    SOFind-aware function to get map metadata
    
    args.config_name: str, sofind datamodel, e.g. 'act_dr6v4'
    args.cat_date: str, date of inpaint catalog, e.g. '20241002'
    args.regular_hole: float, radius of hole [arcmin] for regular sources
    args.large_hole: float, radius of hole [arcmin] for large sources
    args.shape: tuple, shape of mask
    args.wcs: wcs object, wcs of mask
    args.beam_subproduct: str, subproduct name for beam
    args.tf_subproduct: str, subproduct name for transfer function
    args.cal_subproduct: str, subproduct name for calibration
    args.poleff_subproduct: str, subproduct name for polarization efficiency
    args.khfilter: int, horizontal fourier strip width
    args.kvfilter: int, vertical fourier strip width
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
        meta.noisemodel = PlanckNoiseMetadata(qid)
        if splitnum==0 or splitnum==1:
            isplit = 0
        elif splitnum==2 or splitnum==3:
            isplit = 1

    else:
        dm = DataModel.from_config(args.config_name)
        
        meta.beam_fells = dm.read_beam(subproduct=args.beam_subproduct, qid=qid, split=splitnum, coadd=coadd)
        meta.transfer_fells = dm.read_tf(qid, subproduct = args.tf_subproduct)
        meta.calibration = dm.read_calibration(qid, subproduct=args.cal_subproduct)
        meta.pol_eff = dm.read_calibration(qid, subproduct=args.poleff_subproduct)
        meta.inpaint_mask = get_inpaint_mask(args)
        meta.kspace_mask = np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)
        meta.maptype = 'native'
        meta.nsplits = dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)['num_splits']
        meta.noisemodel = ACTNoiseMetadata(qid)
        meta.nspecs = nspecs
        meta.specs = specs_weights['EB'] if args.pureEB else specs_weights['QU']
        isplit = None if coadd else splitnum

    return meta, isplit

# The following 2 functions require:
# args.config_name: str, config name of the datamodel, e.g. "act_dr6v4"
# args.maps_subproduct: str, config name of the maps subproduct, e.g. "default"
def get_data_ivar(qid, splitnum=0, coadd=False, args=None):
    datamodel = DataModel.from_config(args.config_name)
    return datamodel.read_map(qid=qid, coadd=coadd,
                              split_num=splitnum,
                              subproduct=args.maps_subproduct,
                              maptag='ivar')

def get_data_map(qid, splitnum=0, coadd=False, args=None):
    datamodel = DataModel.from_config(args.config_name)
    return datamodel.read_map(qid=qid, coadd=coadd,
                              split_num=splitnum,
                              subproduct=args.maps_subproduct,
                              maptag='map_srcfree')

class EffectiveBeam:

    def __init__(self, datamodel, args, qid, isplit=0, coadd=False):
        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.beam_subproduct = args.beam_subproduct
        self.tf_subproduct = args.tf_subproduct
        self.qid = qid
        self.isplit = isplit
        self.coadd = coadd

    def process_beam(self, sofind_beam, norm=True):
        '''
        normalized beam if required and then interpolate
        '''
        ell_bells, bells = sofind_beam[0], sofind_beam[1]
        assert ell_bells[0] == 0

        if norm:
            bells /= bells[0]

        beam = maps.interp(ell_bells, bells, fill_value='extrapolate')
        return beam

    def get_beam(self):
        beam_map = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd)
        beam_map = self.process_beam(beam_map, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
        return beam_map

    def get_tf(self):
        ells_tf, tf = self.datamodel.read_tf(subproduct=self.tf_subproduct, qid=self.qid)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')

    def get_effective_beam(self):
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        beam = self.get_beam()(np.arange(self.mlmax+1))
        tf = self.get_tf()(np.arange(self.mlmax+1))
        
        fkbeam[0] = beam * tf
        fkbeam[1] = beam
        fkbeam[2] = beam
        return fkbeam, beam, tf


class PlanckNoiseMetadata:
    
    def __init__(self, qid):
        self.qid = qid
        print('under development')

class ACTNoiseMetadata:
    
    def __init__(self, qid):
        self.qid = qid
            
        # noise model qids
        qid_dict_noise = {'pa4a': ['pa4a', 'pa4b'],
                        'pa4b': ['pa4a', 'pa4b'],
                        'pa5a': ['pa5a', 'pa5b'],
                        'pa5b': ['pa5a', 'pa5b'],
                        'pa6a': ['pa6a', 'pa6b'],
                        'pa6b': ['pa6a', 'pa6b'],
                        'pa5a_dd': ['pa5a_dd', 'pa5b_dd'],
                        'pa5b_dd': ['pa5a_dd', 'pa5b_dd'],
                        'pa6a_dd': ['pa6a_dd', 'pa6b_dd'],
                        'pa6b_dd': ['pa6a_dd', 'pa6b_dd'],
                        'pa5a_dw': ['pa5a_dw', 'pa5b_dw'],
                        'pa5b_dw': ['pa5a_dw', 'pa5b_dw']}

        qid_dict_noise_model_name = {'pa4b': 'tile_cmbmask',
                                'pa5a': 'tile_cmbmask',
                            'pa5b': 'tile_cmbmask',
                            'pa6a': 'tile_cmbmask_ivfwhm2',
                            'pa6b': 'tile_cmbmask_ivfwhm2',
                            'pa5a_dd': 'tile_cmbmask_daydeep',
                            'pa5b_dd': 'tile_cmbmask_daydeep',
                            'pa6a_dd': 'tile_cmbmask_daydeep',
                            'pa6b_dd': 'tile_cmbmask_daydeep',
                            'pa5a_dw': 'tile_cmbmask_daywide',
                            'pa5b_dw': 'tile_cmbmask_daywide'}

        qid_dict_config_noise_name = {'pa4b': 'act_dr6v4',
                                    'pa5a': 'act_dr6v4',
                                    'pa5b': 'act_dr6v4',
                                    'pa6a': 'act_dr6v4',
                                    'pa6b': 'act_dr6v4',
                                    'pa5a_dd': 'act_dr6v4_day',
                                    'pa5b_dd': 'act_dr6v4_day',
                                    'pa6a_dd': 'act_dr6v4_day',
                                    'pa6b_dd': 'act_dr6v4_day',
                                    'pa5a_dw': 'act_dr6v4_day',
                                    'pa5b_dw': 'act_dr6v4_day'}
        
        print(f"Initializing NoiseMetadata with qid: {self.qid}")        
        self.tnm = nm.BaseNoiseModel.from_config(qid_dict_config_noise_name[self.qid],
                                        qid_dict_noise_model_name[self.qid],
                                        *qid_dict_noise[self.qid])

    # def get_noise_fn(sim_num, split_num, lmax=5400, alm=True):
    #     return tnm.get_sim_fn(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm)
    
    def get_index_sim_qid(self,qid):

        # Define a mapping from order to index
        order_to_index = {'a': 0, 'b': 1}
        # Extract the order from the qid
        order = qid.split('pa')[1][1]
        # Get the index corresponding to the order
        index = order_to_index[order]

        return index

    def read_in_sim(self,split_num, sim_num, lmax=5400, alm=True):
        
        # grab a sim from disk, fail if does not exist on-disk
        my_sim = self.tnm.get_sim(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, generate=False)
        index = self.get_index_sim_qid(self.qid)
        my_sim = my_sim[index].squeeze()
        
        return my_sim        



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


def preprocess_core(imap, ivar, mask,
                    calibration, pol_eff,
                    maptype='native',
                    dfact = None,
                    inpaint_mask=None,
                    kspace_mask=None, 
                    foreground_cluster=None):
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
        
    if inpaint_mask is not None:
        imap = maps.gapfill_edge_conv_flat(imap, inpaint_mask, ivar=ivar)

    # Check that non-finite regions are in masked region; then set non-finite to zero
    if not(np.all((np.isfinite(imap[...,mask>1e-3])))): raise ValueError
    imap[~np.isfinite(imap)] = 0

    if foreground_cluster is not None:
        imap[0] = imap[0] - foreground        
        
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

def cmb_sims_nomask(sim_index, cmb_set, args):

    """
    Generate CMB simulations without mask.

    Parameters:
    sim_index: int
        CMB sim number to read
    cmb_set: int
        CMB set of sim to read
    args: argument parser
        .window_pow: window power to deconvolve with
        .apo_width_pix: apodization width in pixels
        .fullpix_shape: shape of the full resolution map
        .fullpix_wcs: WCS of the full resolution map
        .dfact: int, downgrade factor

    Returns:
        CMB noiseless simulation on full sky (downgraded, with window function)
    """

    # Read in full sky simulations without beams
    signal = simgen.get_beamed_signal(sim_index, cmb_set,
                                      beam=None, shape=args.fullpix_shape, wcs=args.fullpix_wcs)

    # Apodize before applying window function for FFTs in DEC direction only (RA geometry periodic), width in number of pixels
    signal = enmap.apod(signal,[args.apo_width_pix, 0])
    # Apply window function and downgrade
    wsignal = enmap.apply_window(signal, pow=args.window_pow)
    wsignal = enmap.downgrade(wsignal, args.dfact)
    
    return wsignal


def calculate_noise_power(nmap, mask, mlmax, nsplits, pureEB):    
    
    cl_ab = []
    for k in range(nsplits):

        if pureEB:
            Ealm,Balm=simgen.pureEB(nmap[k][1],nmap[k][2],mask,returnMask=0,lmax=mlmax,isHealpix=False)
            alm_T=cs.map2alm(nmap[k][0],lmax=mlmax)
            alm_a=np.array([alm_T,Ealm,Balm], dtype=np.complex128)
            cl_ab.append(cs.alm2cl(alm_a))
        else:
            alm_a = np.array(cs.map2alm(nmap[k],lmax=mlmax), dtype=np.complex128)
            cl_ab.append(cs.alm2cl(alm_a)) 

    w2=simgen.w_n(mask,2)
    cl_sum = np.sum(cl_ab, axis=0)
    power = 1/nsplits/(nsplits-1) * cl_sum
    power[~np.isfinite(power)] = 0
    
    return power / w2

def get_name_weights(qid, spec):

    return f'noise_{qid}_{spec}'

def read_weights(args):

    nqids = len(args.qids)
    noise_specs = np.zeros((nspecs, nqids, args.mlmax+1), dtype=np.float64)

    if args.pureEB:
        specs = specs_weights['EB']
    else:
        specs = specs_weights['QU']

    for i, qid in enumerate(args.qids):
        for ispec, spec in enumerate(specs):
            noise_specs[ispec, i] = np.loadtxt(get_fout_name(get_name_weights(qid, spec), args, stage='weights'))[:args.mlmax+1]
    
    return noise_specs

def get_fout_name(fname, args, stage, tag=None):

    '''
    fname: name of file to be saved
    args: argparse object including args.output_dir
    stage: str
        ['inpaint', 'cluster_subtraction']
    '''
    fname = fname.split('.fits')[0]

    # if stage == 'inpaint':
    #     fname += '_d2_gapfill.fits' # dowgranded (dfact 2) and inpainted
    #     folder = 'stage_inpaint/'

    # elif stage == 'cluster_subtraction':
    #     fname += '_mnemo.fits'
    #     folder = 'stage_cluster_subtraction/'

    # elif stage == 'preprocessing':
    #     fname += '_preproccesed_alms.npy'
    #     folder = 'stage_preprocessing/'

    if stage == 'weights':
        fname += '_weights.txt'
        folder = 'stage_compute_weights/'

    # elif stage == 'kspace_coadd':
    #     fname  = 'kspace_coadd_' + fname + '.fits'
    #     if tag == 'sim':
    #         folder = 'stage_kspace_coadd_sims/'
    #     else:
    #         folder = 'stage_kspace_coadd/'

    # elif stage == 'noiseless_sims':
    #     fname += '.fits'
    #     folder = 'stage_noiseless_sims/'

    output_dir = os.path.join(args.output_dir, f'../{folder}')
    # create output folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, fname)

# orphics.maps.kspace_coadd_alms
# pure E,B:   Q, U maps and a mask -> E, B alms
# kspace_coadd:  T, E, B alms and noise spectra -> T, E, B alms

"""
kspace_coadd_sims:

for qid in qids:
  sim = get_sim(qid)
  psim = preprocess_core(sim)
  kmaps = map2alm(psim)  OR  pure_eb(psim)

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
