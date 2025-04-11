from orphics import maps
from pixell import enmap, utils as u, curvedsky as cs, reproject
import numpy as np
from mnms import noise_models as nm
from sofind import DataModel
from pixell import bunch
from solenspipe import utility as simgen
import os
import healpy as hp
from scipy import interpolate
import re

specs_weights = {'QU': ['I','Q','U'],
        'EB': ['I','E','B']}
nspecs = len(specs_weights['QU'])

def is_planck(qid):
    return (parse_qid_experiment(qid)=='planck')

def is_pipe4_BN(qid):
    return (parse_qid_experiment(qid)=='pipe4_BN')
def is_SOsims(qid):
    return (parse_qid_experiment(qid)=='so_sims')

def parse_qid_experiment(qid):
    if qid in ['p01','p02','p03','p04','p05','p06','p07']:
        return 'planck'
    elif qid[:3]=='sobs_':
        return 'sobs'
    elif qid in ["ot_i1_f150", "ot_i1_f090", "ot_i3_f150", "ot_i3_f090", "ot_i4_f150", "ot_i4_f090", "ot_i6_f150", "ot_i6_f090"]:
        return 'pipe4_BN'
    elif qid in ['lfa', 'lfb','mfa','mfb', 'uhfa', 'uhfb']:
        return 'so_sims'
    else:
        return 'act'

# leaving this for archival purposes for now
# function called in v5 preprocessing is in PlanckNoiseMetadata
def process_residuals_alms(isplit, freq, task,root_path="/gpfs/fs0/project/r/rbond/jiaqu/"):
    """
    Rotate the residuals from healpix to enmap and return the residual alms. Note that extraction of the ACT footprint is not performed here.
    This is done for a given simulation type, frequency, and task number.

    Parameters
    ----------
    isplit : int
        The Planck split that the residual corresponds to, 0 for 'npipe6v20A', 1 for 'npipe6v20B'
        (This format is automatically compatible with the Planck metadata format from get_metadata)

    freq : int
        The frequency (in GHz) of the data to be processed, used to select the 
        corresponding residual files. Can be provided using:
        metadata.noisemodel.qid_dict_config_noise_name[qid]

    task : int
        The task identifier for the simulation. This is used to generate a unique index 
        for identifying the residual files. +200 is added to the task number to match the simulation label of Planck simulations

    output_path : str
        The directory path where the processed enmap residual files will be saved.
    """

    n_index = str(task+200).zfill(4)
    sim_type = 'npipe6v20' + ('A' if isplit == 1 else 'B')
    #TODO (not urgent) Put the path in sofind
    residual = hp.read_map(f'{root_path}/{sim_type}_sim/{n_index}/residual/residual_{sim_type}_{freq}_{n_index}.fits', field=(0,1,2))
    #multiply by Planck map factor to convert to uKarcmin
    residual_alm = reproject.healpix2map(residual, lmax=3000, rot='gal,equ',save_alm=True)*10**6
    return residual_alm



def get_inpaint_mask(args, datamodel):
    
    '''
    args.inpaint: bool, if True, you want to inpaint
    args.Name: str, sofind datamodel, e.g. 'act_dr6v4', for catalog
    args.cat_date: str, date of inpaint catalog, e.g. '20241002'
    args.regular_hole: float, radius of hole [arcmin] for regular sources
    args.large_hole: float, radius of hole [arcmin] for large sources
    args.shape: tuple, shape of mask
    args.wcs: wcs object, wcs of mask
    '''
    
    if args.inpaint:
        print('inpainting')
        assert args.cat_date is not None, "cat_date must be provided for inpaint"

        # read catalog coordinates
        rdecs, rras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_regular_{args.cat_date}.csv', subproduct = 'inpaint_catalogs'))
        ldecs, lras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_large_{args.cat_date}.csv', subproduct = 'inpaint_catalogs'))

        # Make masks for gapfill
        mask1 = maps.mask_srcs(args.shape,args.wcs,np.asarray((ldecs,lras)),args.large_hole)
        mask2 = maps.mask_srcs(args.shape,args.wcs,np.asarray((rdecs,rras)),args.regular_hole)
        jmask = mask1 & mask2
        jmask = ~jmask
        
        return jmask
    
    else:
        print('not inpainting')
        return None

def get_metadata(qid, splitnum=0, coadd=False, args=None):  #add args.config_name 
    """
    SOFind-aware function to get map metadata
    
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
    # print("splitnum", splitnum)
    # print(type(splitnum))
    # assert 0 <= int(splitnum) < 4, "only supporting splits [0,1,2,3]"
    if parse_qid_experiment(qid)=='planck':
        meta.Name = 'planck_npipe'
        meta.dm = DataModel.from_config(meta.Name)
        meta.splits = np.array([1,2])
        meta.nsplits = 2
        meta.calibration = 1.0
        meta.pol_eff = 1.0
        Beam = PlanckBeamHelper(meta.dm, args, qid, splitnum)
        meta.beam_fells = Beam.get_effective_beam()[1]
        meta.transfer_fells = Beam.get_effective_beam()[2]
        meta.inpaint_mask = None
        meta.kspace_mask = None
        meta.maptype = 'reprojected'
        meta.noisemodel = PlanckNoiseMetadata(qid, verbose=True,
                                              config_name=meta.Name,
                                              subproduct_name="noise_sims")
        # assigning ACT splits 0 + 1 to Planck split 1
        # and ACT splits 2 + 3 to Planck split 2
        isplit = None if coadd else (splitnum // 2 + 1)
    elif parse_qid_experiment(qid)=='act':
        meta.Name = 'act_dr6v4'
        meta.dm = DataModel.from_config(meta.Name)
        qid_dict = meta.dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
        
        meta.nsplits = qid_dict['num_splits']
        meta.splits = np.arange(meta.nsplits)
        meta.daynight = qid_dict['daynight']
   
        if meta.daynight == 'night':
            meta.calibration = meta.dm.read_calibration(qid, subproduct=args.cal_subproduct)
            meta.pol_eff = meta.dm.read_calibration(qid, subproduct=args.poleff_subproduct)
        else:
            meta.calibration = 1.
            meta.pol_eff = 1.

        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)
        meta.maptype = 'native'
        meta.noisemodel = ACTNoiseMetadata(qid, verbose=True)
        meta.nspecs = nspecs
        meta.specs = specs_weights['EB'] if args.pureEB else specs_weights['QU']
        isplit = None if coadd else splitnum
        
        Beam = ACTBeamHelper(meta.dm, args, qid, isplit, coadd=coadd, daynight=meta.daynight)
        meta.beam_fells = Beam.get_effective_beam()[1]
        meta.transfer_fells = Beam.get_effective_beam()[2]
    elif parse_qid_experiment(qid)=='pipe4_BN':
        meta.Name = 'so_lat_pipe4_BN' ##this should be passed as argument otherwise use default
        meta.dm = DataModel.from_config(meta.Name)
        qid_dict = meta.dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
        
        meta.nsplits = qid_dict['num_splits']
        meta.splits = np.arange(meta.nsplits)
        meta.calibration = 1.
        meta.pol_eff = 1.

        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)
        meta.maptype = 'native'
        meta.noisemodel = SOLATNoiseMetadata(qid, verbose=True) 
        meta.nspecs = nspecs
        meta.specs = specs_weights['EB'] if args.pureEB else specs_weights['QU']
        isplit = None if coadd else splitnum
        
        Beam = SOLATBeamHelper(meta.dm, args, qid, isplit, coadd=coadd)
        meta.beam_fells = Beam.get_effective_beam()[1]  #done
        meta.transfer_fells = Beam.get_effective_beam()[2] #done
    elif parse_qid_experiment(qid)=='so_sims':
        meta.Name = 'so_lat_mbs_mss0002' ##this should be passed as argument otherwise use default
        meta.dm = DataModel.from_config(meta.Name)
        qid_dict = meta.dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
        
        meta.nsplits = qid_dict['num_splits']
        meta.splits = np.arange(meta.nsplits)
        meta.calibration = 1.
        meta.pol_eff = 1.

        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)
        meta.maptype = 'native'
        meta.noisemodel = SOsimsNoiseMetadata(qid, verbose=True) 
        meta.nspecs = nspecs
        meta.specs = specs_weights['EB'] if args.pureEB else specs_weights['QU']
        isplit = None if coadd else splitnum
        
        Beam = SOsimsBeamHelper(meta.dm, args, qid, isplit, coadd=coadd)
        meta.beam_fells = Beam.get_effective_beam()[1]  #done
        meta.transfer_fells = 1.
    return meta, isplit

# The following 2 functions require:
# args.config_name: str, config name of the datamodel, e.g. "act_dr6v4"
# args.maps_subproduct: str, config name of the maps subproduct, e.g. "default"
def get_data_ivar(qid, splitnum=0, coadd=False, args=None):
    datamodel = DataModel.from_config(args.config_name)
    if is_planck(qid):
        splitnum+=1
    return datamodel.read_map(qid=qid, coadd=coadd,
                              split_num=splitnum,
                              subproduct=args.maps_subproduct,
                              maptag='ivar')

def get_data_map(qid, splitnum=0, coadd=False, args=None):
    datamodel = DataModel.from_config(args.config_name)
    if is_planck(qid):
        assert splitnum in [1,2], "Planck splits are either 1 or 2"
        maptag='srcfree'
    elif is_pipe4_BN(qid):
        maptag='map'
    else:
        maptag='map_srcfree'
    return datamodel.read_map(qid=qid, coadd=coadd,
                              split_num=splitnum,
                              subproduct=args.maps_subproduct,
                              maptag=maptag)

def process_beam(sofind_beam, norm=True):
    '''
    normalized beam if required and then interpolate
    '''
    ell_bells, bells = sofind_beam[0], sofind_beam[1]
    assert ell_bells[0] == 0

    if norm:
        bells /= bells[0]

    beam = maps.interp(ell_bells, bells, fill_value='extrapolate')
    return beam

class SOLATBeamHelper:
    def __init__(self, datamodel,args,qid, isplit=0, coadd=False, daynight=None):
        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.qid = qid
        self.isplit = isplit
        self.coadd = coadd
        self.beam_subproduct = args.beam_subproduct
        self.tf_subproduct = args.tf_subproduct
        self.daynight = daynight #not really needed here
    
    def get_beam(self):
        beam_map = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd)
        beam_map = process_beam(beam_map,True)
        return beam_map
        
    def get_tf(self):
        ells_tf, tf = self.datamodel.read_tf(subproduct=self.tf_subproduct, qid=self.qid)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')
    
    def get_effective_beam(self):
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        
        tf = self.get_tf()(np.arange(self.mlmax+1))
        
        beam = self.get_beam()(np.arange(self.mlmax+1))
        
        fkbeam[0] = beam * tf
        fkbeam[1] = beam
        fkbeam[2] = beam

        return fkbeam, beam, tf   

class SOsimsBeamHelper:
    def __init__(self, datamodel,args,qid, isplit=0, coadd=False, daynight=None):
        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.qid = qid
        self.isplit = isplit
        self.coadd = coadd
        self.beam_subproduct = args.beam_subproduct
        self.tf_subproduct = args.tf_subproduct
        self.daynight = daynight #not really needed here
    
    def get_beam(self):
        beam_map = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd)
        beam_map = process_beam(beam_map,True)
        return beam_map
        
    def get_tf(self):
        ells_tf, tf = self.datamodel.read_tf(subproduct=self.tf_subproduct, qid=self.qid)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')
    
    def get_effective_beam(self):
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        
        #tf = self.get_tf()(np.arange(self.mlmax+1))
        
        beam = self.get_beam()(np.arange(self.mlmax+1))
        
        fkbeam[0] = beam #* tf
        fkbeam[1] = beam
        fkbeam[2] = beam

        return fkbeam, beam #, tf  
        
class ACTBeamHelper:

    def __init__(self, datamodel, args, qid, isplit=0, coadd=False, daynight='night'):
        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.qid = qid
        self.isplit = isplit
        self.coadd = coadd
        self.beam_subproduct = args.beam_subproduct
        self.tf_subproduct = args.tf_subproduct
        self.daynight = daynight

    def get_beam(self):
        
        if self.beam_subproduct == 'beams_v4_20230130_snfit':
            
            beam_T = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd, tpol = 'T')
            beam_P = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd, tpol = 'POL')

            beam_T = process_beam(beam_T, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
            beam_P = process_beam(beam_P, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
        
            return beam_T, beam_P
        
        else:
            beam_map = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd)
            beam_map = process_beam(beam_map, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
        
            return beam_map

    def get_tf(self):
        
        if self.daynight == 'night':
            ells_tf, tf = self.datamodel.read_tf(subproduct=self.tf_subproduct, qid=self.qid)
        else: 
            ells_tf, tf = np.arange(2, 3000, dtype=float), np.ones(2998)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')

    def get_effective_beam(self):
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        
        tf = self.get_tf()(np.arange(self.mlmax+1))
        
        if self.beam_subproduct == 'beams_v4_20230130_snfit':
            
            beam_T, beam_P = self.get_beam()
            beam_T = beam_T(np.arange(self.mlmax+1))
            beam_P = beam_P(np.arange(self.mlmax+1))
            
            beam = beam_P
            fkbeam[0] = beam_T
            fkbeam[1] = beam_P
            fkbeam[2] = beam_P
            
        else:
            beam = self.get_beam()(np.arange(self.mlmax+1))
            
            fkbeam[0] = beam * tf
            fkbeam[1] = beam
            fkbeam[2] = beam

        return fkbeam, beam, tf


class PlanckNoiseMetadata:
    
    def __init__(self, qid, verbose=False,
                 config_name="planck_npipe",
                 subproduct_name="noise_sims"):
        self.qid = qid

        qid_dict_config_noise_name = {'p01': '030',
                                    'p02': '044',
                                    'p03': '070',
                                    'p04': '100',
                                    'p05': '143',
                                    'p06': '217',
                                    'p07': '353',
                                    'p08': '547',
                                    'p09': '857'}
        
        self.planck_config_name = config_name
        self.planck_noise_sims_subproduct = subproduct_name

        if verbose:
            print(f"Initializing NoiseMetadata with qid: {self.qid}")
        self.qid_freq = qid_dict_config_noise_name[qid]

    # moved Frank's residual noise alm function here...
    def noise_map_path(self, isplit, index):
        datamodel = DataModel.from_config(self.planck_config_name)
        maptag = str(index+200).zfill(4)
        assert isplit in [1,2], "Planck splits are either 1 or 2"
        split_num = "A" if isplit == 1 else "B"
        return datamodel.get_map_fn(qid=self.qid, coadd=False,
                                    split_num=split_num,
                                    subproduct="noise_sims",
                                    maptag=maptag)
    
    def read_in_sim(self, isplit, index, lmax=3000):
        # REQUIRES MODIFICATION TO PIXELL (ask Frank/Joshua)
        residual_map = hp.read_map(self.noise_map_path(isplit, index),
                                   field=(0,1,2))
        return reproject.healpix2map(residual_map, lmax=lmax,
                                     rot='gal,equ',save_alm=True)*10**6


class SOLATNoiseMetadata:
    def __init__(self, qid, verbose=False):
        self.qid = qid
        
        qid_dict_noise = {'ot_i1_f090': ["ot_i1_f090", "ot_i1_f150"],
                          'ot_i1_f150': ["ot_i1_f090", "ot_i1_f150"],
                          'ot_i3_f090': ["ot_i3_f090", "ot_i3_f150"],
                          'ot_i3_f150': ["ot_i3_f090", "ot_i3_f150"],
                          'ot_i4_f090': ["ot_i4_f090", "ot_i4_f150"],
                          'ot_i4_f150': ["ot_i4_f090", "ot_i4_f150"],
                          'ot_i6_f090': ["ot_i6_f090", "ot_i6_f150"],
                          'ot_i6_f150': ["ot_i6_f090", "ot_i6_f150"],
                          }
        config_name="so_lat_pipe4_BN"
        noise_model_name="tile_cmbmask"
        if verbose:
            print(f"Initializing NoiseMetadata with qid: {self.qid}")
            
        self.tnm = nm.BaseNoiseModel.from_config(config_name,
                                                 noise_model_name,
                                                 *qid_dict_noise[self.qid])
        
    def get_index_sim_qid(self,qid):

        # Define a mapping from order to index
        order_to_index = {'f090': 0, 'f150': 1}
        # Extract the order from the qid
        order = qid.split('_')[-1]
        # Get the index corresponding to the order
        index = order_to_index[order]

        return index

    def read_in_sim(self,split_num, sim_num, lmax=5400, alm=True):
        
        # grab a sim from disk, fail if does not exist on-disk
        my_sim = self.tnm.get_sim(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, generate=False)
        index = self.get_index_sim_qid(self.qid)
        my_sim = my_sim[index].squeeze()
        
        return my_sim           
class SOsimsNoiseMetadata:
    
    def __init__(self, qid, verbose=False):
        self.qid = qid
            
        # noise model qids
                # noise model qids
        qid_dict_noise = {'lfa': ['lfa', 'lfb'],
                    'lfb': ['lfa', 'lfb'],
                    'mfa': ['mfa', 'mfb'],
                    'mfb': ['mfa', 'mfb'],
                    'uhfa': ['uhfa', 'uhfb'],
                    'uhfb': ['uhfa', 'uhfb']}

        qid_dict_noise_model_name = {'lfa': 'fdw_lf',
                    'lfb': 'fdw_lf',
                    'mfa': 'fdw_mf',
                    'mfb': 'fdw_mf',
                    'uhfa': 'fdw_uhf',
                    'uhfb': 'fdw_uhf'}

        qid_dict_config_noise_name = {
                    'lfa': 'so_lat_mbs_mss0002',
                    'lfb': 'so_lat_mbs_mss0002',
                    'mfa': 'so_lat_mbs_mss0002',
                    'mfb': 'so_lat_mbs_mss0002',
                    'uhfa': 'so_lat_mbs_mss0002',
                    'uhfb': 'so_lat_mbs_mss0002'}
        
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
        order = qid.split('mf')[1] #[1] \\ok 
        # Get the index corresponding to the order
        index = order_to_index[order]

        return index

    def read_in_sim(self,split_num, sim_num, lmax=5400, alm=True):
        
        # grab a sim from disk, fail if does not exist on-disk
        my_sim = self.tnm.get_sim(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, generate=False)
        index = self.get_index_sim_qid(self.qid)
        my_sim = my_sim[index].squeeze()
        
        return my_sim   

class ACTNoiseMetadata:
    
    def __init__(self, qid, verbose=False):
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
        if verbose:
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

class PlanckBeamHelper:

    def __init__(self, datamodel, args, qid, isplit):
        """
        Initialize the PlanckBeamHelper.
        """
        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.qid = qid
        self.isplit = isplit
        self.beam_subproduct = args.beam_subproduct

    def get_beam(self, pixwin=True):
        """
        Retrieve the Planck beam for a given QID and split number.

        Parameters:
        - pixwin (bool): Apply pixel window function if True.

        Returns:
        - np.ndarray: The beam function array.
        """

        # Determine split letter
        assert self.isplit in [1,2], "Planck splits are either 1 or 2"
        sl = 'A' if self.isplit == 1 else 'B'

        # Load and interpolate the beam
        ell_b, bl = self.datamodel.read_beam(self.qid,
                        subproduct=self.beam_subproduct,
                        split_num=sl)
        beam_f = maps.interp(ell_b, bl, fill_value='extrapolate')

        # Generate the beam function values
        beam_fells = beam_f(np.arange(self.mlmax+1))
        if pixwin:
            beam_fells *= hp.pixwin(2048)[:self.mlmax+1]

        return beam_fells
    
    def get_tf(self):
        ells_tf, tf = np.arange(2, 3000, dtype=float), np.ones(2998)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')
    
    def get_effective_beam(self):
        
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        
        tf = self.get_tf()(np.arange(self.mlmax+1))
        beam = self.get_beam()[np.arange(self.mlmax+1)]
        
        fkbeam[0] = beam * tf
        fkbeam[1] = beam
        fkbeam[2] = beam

        return fkbeam, beam, tf
    
    def planck_processed_beam(self, qid, splitnum, pixwin=True):
        b_ell=self.get_beam(qid, splitnum, pixwin=pixwin)
        ell=np.arange(len(b_ell))
        sofind_beam = np.array([ell, b_ell])
        beam=process_beam(sofind_beam, norm=True)
        return beam


class ForegroundHandler:
    '''generates foregrounds'''

    def __init__(self, datamodel, args):
        
        '''
        datamodel: DataModel, sofind datamodel
        args.fg_type: str, type of foregrounds, 'sims' or 'theory'
        args.fgs_path: str, path to foreground sims
        args.is_noiseless: bool, if True, no foregrounds are generated
        args.lmax_signal: int, maximum ell of signal sims
        args.maps_subproduct: str, subproduct name for maps
        '''

        self.datamodel = datamodel
        self.args = args
        assert self.args.fg_type in ['sims', 'theory'] # 'sims' loads from foreground 2pt file (measured in sims), 'theory' estimates analytically
        self.fgcov_func = self._define_fgcov_func()

    def _define_fgcov_func(self):
        ''' load foreground covariance matrix (power spectra)'''
        if self.args.is_noiseless:
            return None
        if self.args.fg_type == 'sims':
            return lambda: self.generate_cov_fgs(self.args.fgs_path, self.args.lmax_signal) # lmax conditioned by max ell of signal sims (van Engelen)
        elif self.args.fg_type == 'theory':
            # currently unsupported
            # return lambda: self.get_fg_cov(qids)
            raise NotImplementedError
        return None

    def generate_cov_fgs(self, fgs_path, lmax):
        
        '''
        returns the foreground covariance matrix (power spectrum) for the given lmax 
        at two frequencies (90 and 150 GHz) and their cross-spectrum
        
        fgs_path: str, path to foreground sims
        lmax: int, maximum ell of fg power spectrum
        '''
        
        w_2_foreground = 0.27

        # Load foreground alms
        foreground_alms_93 = hp.read_alm(fgs_path + 'fg_nonoise_alms_0093.fits')
        foreground_alms_150 = hp.read_alm(fgs_path + 'fg_nonoise_alms_0145.fits')

        # Generate foreground Cls and smooth them
        cls_90 = simgen.smooth_rolling_cls(cs.alm2cl(foreground_alms_93)/ w_2_foreground, N=10) 
        cls_150 = simgen.smooth_rolling_cls(cs.alm2cl(foreground_alms_150)/ w_2_foreground, N=10) 
        cls_150x90 = simgen.smooth_rolling_cls(cs.alm2cl(foreground_alms_93, foreground_alms_150) / w_2_foreground, N=10)

        # Initialize the covariance matrix of the map (a.k.a. power spectrum)
        cov_matrix = np.zeros((2, 2, lmax + 1))

        # Assign values to the covariance matrix
        # 0 == f150, 1 == f090
        cov_matrix[0, 0] = cls_150[:lmax + 1]
        cov_matrix[0, 1] = cls_150x90[:lmax + 1]
        cov_matrix[1, 0] = cov_matrix[0, 1]
        cov_matrix[1, 1] = cls_90[:lmax + 1]

        return cov_matrix

    def get_fg_cov(self, qid):
        raise NotImplementedError

    def get_map_fgs(self, qid, alms_f):
        '''
        qid of the map (frequency it corresponds to) is mapped to component of alms_f
        '''
        qid_freq_dict = {'f150': 0, 'f090': 1}
        qid_dict = self.datamodel.get_qid_kwargs_by_subproduct(product='maps', subproduct=self.args.maps_subproduct, qid=qid)
        index_fg = qid_freq_dict[qid_dict['freq']]
        return alms_f[index_fg]

    def get_fg_alms(self, fgcov, qid, cmb_set, sim_indices):
        '''
        generate alm from cl
        
        fgcov: np.ndarray, foreground covariance matrix (2pt)
        qid: str, qid of the map
        cmb_set: int, set of cmb sim (for seed)
        sim_indices: int, index of the sim (for seed)
        '''
        if self.args.fg_type == 'sims':
            alms_f = cs.rand_alm(fgcov, seed=(0, cmb_set, sim_indices))
            return self.get_map_fgs(qid, alms_f)
        elif self.args.fg_type == 'theory':
            return cs.rand_alm(fgcov)
        return None

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


def preprocess_core(imap, mask,
                    calibration, pol_eff, ivar=None,
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
    For simulations ivar processing is redundant, we should probably set ivar as an optional argument

    Returns beam convolved (transfer uncorrected) T, Q, U maps.
    """
    if dfact!=1 and (dfact is not None):
        imap = enmap.downgrade(imap,dfact)
        if ivar is not None:
            ivar = enmap.downgrade(ivar,dfact,op=np.sum)

    if inpaint_mask is not None:
        # assert ivar is not None, "need ivar for inpainting" -- not true, random noise ivar
        imap = maps.gapfill_edge_conv_flat(imap, inpaint_mask, ivar=ivar)


    #for Planck, assert that we extract the RA DEC of the ACT footprint only
    if imap[0].shape != mask.shape:
        imap = enmap.extract(imap, (3,)+mask.shape, mask.wcs)
        if ivar is not None:
            ivar = enmap.extract(ivar, (3,)+mask.shape, mask.wcs)
    # Check that non-finite regions are in masked region; then set non-finite to zero
    if not(np.all((np.isfinite(imap[...,mask>1e-3])))): raise ValueError
    imap[~np.isfinite(imap)] = 0

    if foreground_cluster is not None:
        imap[0] = imap[0] - foreground_cluster        
        
    imap = imap * mask
    imap = depix_map(imap,maptype=maptype,dfact=dfact,kspace_mask=kspace_mask)

    imap = imap * calibration
    imap[1:] = imap[1:] / pol_eff
    
    if ivar is not None:
        ivar = ivar / calibration**2.
        ivar[1:] = ivar[1:] * pol_eff**2.
    
    return imap, ivar # ivar will be none if nothing happened to it



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
        if noise_mask is not None:
            lstitch = noise_lmax - 200
            mlmax = noise_lmax + 600
            nmap = maps.stitched_noise(shape,wcs,noise_alms,noise_mask,rms_uk_arcmin=rms_uk_arcmin,
                                       lstitch=lstitch,lcosine=lcosine,mlmax=mlmax)
        else:
            nmap = cs.alm2map(noise_alms,enmap.empty((3,)+shape,wcs))
            nmap[~np.isfinite(nmap)] = 0.
    else:
        nmap = 0.
    
    # notice how these are inverse of what's in preprocess
    # not applied to nmap because it is already based on data (which includes them)
    omap = omap / calibration  
    omap[1:] = omap[1:] * pol_eff
    
    omap = omap + nmap
    return omap


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

def get_name_cluster_fgmap(qid, isplit=1, coadd=False):
    
    if coadd:
        return f'{qid}_nemo' # !!!
    else:
        return f'{qid}_split{isplit}_nemo'

def get_name_run(args, split=None, coadd=False):

    name_run = f'{"_".join(args.qids)}_mnemo{args.cluster_subtraction}_{args.mask_tag}'
    
    if split is not None:
        name_run += f'_split{split}'
    
    if coadd:
        name_run += '_coadd'

    return name_run

def get_mask_tag(mask_fn, mask_subproduct):

    """
    Extracts and returns the tag from the mask filename.
    """

    assert 'lensing_masks' in mask_subproduct, 'mask tag only implemented for lensing masks'
    # find daynight tag "daydeep", "daywide" o "night"
    daynight = re.search(r'(daydeep|daywide|night)', mask_fn).group(0)
    # find skyfraction. it is 2 digit number. the string between the last "_" and ".fits"
    skyfrac  = re.search(r'_(\d{2})(?:_[^_]+)?\.fits$', mask_fn).group(1) # re.search(r'_([^_]+)\.fits$', mask_fn).group(1)

    return f'{daynight}_{skyfrac}'
def get_mask_tag_so(mask_fn, mask_subproduct):
    ##TBD how to implement this
    """
    Extracts and returns the tag from the mask filename.
    """

    assert mask_subproduct == 'so_lat_mbs_mss0002', 'mask tag only implemented for so_lat_mbs_mss0002 masks'
    # find daynight tag "daydeep", "daywide" o "night"
    daynight = re.search(r'(mss0002)', mask_fn).group(0)
    # find skyfraction. it is the string between the last "_" and ".fits"
    skyfrac  = re.search(r'_([^_]+)\.fits$', mask_fn).group(1)

    return f'{daynight}_{skyfrac}'
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
    
    elif stage == 'cluster_fgmap':
        fname += '_cluster_fgmap.fits'
        folder = 'stage_cluster_fgmap/'

    elif stage == 'kspace_coadd':
        fname  = 'kspace_coadd_' + fname + '.fits'
        if tag == 'sim':
            folder = 'stage_kspace_coadd_sims/'
        else:
            folder = 'stage_kspace_coadd/'

    elif stage == 'nilc_coadd':
        fname  = 'nilc_coadd_' + fname + '.fits'
        if tag == 'sim':
            folder = 'stage_nilc_coadd_sims/'
        else:
            folder = 'stage_nilc_coadd'

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
