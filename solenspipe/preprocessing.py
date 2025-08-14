from orphics import maps, io
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

specs_weights = {'EpureB': ['T','E','pureB'],
        'EB': ['T','E','B']}
nspecs = len(specs_weights['EB'])

def is_planck(qid):
    return (parse_qid_experiment(qid)=='planck')

def is_lat_iso(qid):
    return (parse_qid_experiment(qid)=='lat_iso')

def is_mss2(qid):
    return (parse_qid_experiment(qid)=='so_mss2')
          
def parse_qid_experiment(qid):
    if qid in ['p01','p02','p03','p04','p05','p06','p07','p08','p09']:
        return 'planck'
    elif qid[:3]=='sobs_':
        return 'sobs'
    elif qid in ["ot_i1_f150", "ot_i1_f090", "ot_i3_f150", "ot_i3_f090", "ot_i4_f150", "ot_i4_f090", "ot_i6_f150", "ot_i6_f090"]:
        return 'lat_iso'
    elif qid in ['lfa', 'lfb','mfa','mfb', 'uhfa', 'uhfb']:
        return 'so_mss2'
    else:
        return 'act'

class MetadataUnifier(object):
    """
    This class provides a binding between a YAML file
    and SOFind, providing seamless easily extensible
    access to data products under a unified interface.

    The following will be needed:
    1. maps (srcfull, srcfree, ivar)
    2. effective beams (beam * transfer * leakage * heapix_pixwin)
    3. calibration
    4. pol. eff.
    """
    def __init__(self,yaml_file='metadata.yaml'):
        self.c = io.config_from_yaml(yaml_file)

    def _get_class(self,qid):
        config_dict = self.c
        found = []
        for key in config_dict.keys():
            if qid in config_dict[key]['possible_qids']:
                found.append(key)
        if len(found)>1: raise ValueError(f"{qid} found in more than 1 class")
        if len(found)==0: raise ValueError(f"{qid} found in no class")
        return found[0]

    def get_args(self,qid):
        return bunch.Bunch(self.c[self._get_class(qid)])

    def get_cal(self,qid):
        args = self.get_args(qid)
        dm = DataModel.from_config(args.dm_name)
        # Calibration for T,Q,U. A single number.
        if args.cal_subproduct is None:
            return 1.
        else:
            return dm.read_calibration(qid, subproduct=args.cal_subproduct, which='cals')

    def get_poleff(self,qid):
        args = self.get_args(qid)
        dm = DataModel.from_config(args.dm_name)
        # Pol eff. for Q,U. A single number.
        if args.poleff_subproduct is None:
            return 1.
        else:
            return dm.read_calibration(qid, subproduct=args.poleff_subproduct, which='poleffs')
        
    def get_map_fname(self,qid,map_type='srcfree', # srcfull, srcfree, ivar
                      coadd=True,split_num=None):
        args = self.get_args(qid)
        dm = DataModel.from_config(args.dm_name)
        if coadd and split_num: raise ValueError
        if map_type=='srcfree':
            maptag = args.srcfree_name
        elif map_type=='srcfull':
            maptag = 'map'
        else:
            maptag = map_type
            
        f = dm.get_map_fn(qid=qid,
                      coadd=coadd,
                      split_num=(split_num+args.split_start) if not(coadd) else None,
                      subproduct=args.maps_subproduct,
                      maptag=maptag)

        if not(os.path.exists(f)): print(f"File {f} not found.")
        return f

    def get_map(self,qid,map_type='srcfree', # srcfull, srcfree, ivar
                coadd=True,split_num=None,pol=True,
                oshape=None,owcs=None):
        # TODO: To complete this function (use fname function for now)
        raise NotImplementedError
        return enmap.read_map(self.get_map_fname(qid,map_type=map_type,
                                          coadd=coadd,split_num=split_num),sel=sel)
        
    
    def get_effective_beam(self,qid,
                           simulation=False, # if this is for a simulation, skip sanitization of beam
                           ells=None,
                           get_breakdown=False):
        #if coadd and split_num: raise ValueError # TODO: Implement splits
        args = self.get_args(qid)
        dm = DataModel.from_config(args.dm_name)

        # TODO: Handle daytime
        lbeam,vbeam = dm.read_beam(subproduct=args.beam_subproduct, qid=qid, split_num=None, coadd=True)
        # The following normalizes the beam, and then "sanitizes" it if this is not a simulation
        if ells is not None:
            obeam = maps.sanitize_beam(ells,maps.interp(lbeam,vbeam)(ells),sval=1e-3 if not(simulation) else None,verbose=True)
        else:
            ells = lbeam.copy()
            obeam = maps.sanitize_beam(ells,vbeam,sval=1e-3 if not(simulation) else None,verbose=True)

        final_beam = np.ones((3,obeam.size))
        final_beam[0] = obeam.copy()
        final_beam[1] = obeam.copy()
        final_beam[2] = obeam.copy()

        if get_breakdown:
            breakdown = {}
            breakdown['raw'] = final_beam.copy()
            
        if args.tf_subproduct is not None:
            ltf,vtf = dm.read_tf(subproduct=args.tf_subproduct, qid=qid)
            otf = maps.interp(ltf,vtf,bounds_error=False,fill_value=1.0)(ells)
            final_beam[0] = final_beam[0] * otf
            if get_breakdown:
                breakdown['tf'] = otf.copy()
            
        else:
            pass
            

        try:
            pwin_nside = args.hp_pixwin_nside
            pwT,pwP = hp.pixwin(pwin_nside, pol=True, lmax=None)
            pls = np.arange(pwT.size)
            opwT = maps.interp(pls,pwT,bounds_error=False,fill_value=1.0)(ells)
            opwP = maps.interp(pls,pwP,bounds_error=False,fill_value=1.0)(ells)
            final_beam[0] = final_beam[0] * opwT
            final_beam[1] = final_beam[1] * opwP
            final_beam[2] = final_beam[2] * opwP
            if get_breakdown:
                breakdown['pwin_T'] = opwT.copy()
                breakdown['pwin_P'] = opwP.copy()

            # TODO: Add additional checks to make sure this really is Planck
            
        except AttributeError:
            pass

        # set monopole and dipole to 1
        final_beam[:,:2] = 1.0
        
        if get_breakdown:
            return ells,final_beam, breakdown
        else:
            return ells,final_beam

    

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
    Generates a mask where to inpaint the map.
    
    ### Parameters:
    - datamodel: DataModel object, used to read sofind products
    - args: argparse.Namespace(), must contain the following attributes: 
        args.inpaint: bool, if True, you want to inpaint
        args.inpaint_subproduct: str, subproduct name for inpainting catalog
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
        rdecs, rras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_regular_{args.cat_date}.csv', subproduct = args.inpaint_subproduct))
        ldecs, lras = np.rad2deg(datamodel.read_catalog(cat_fn = f'union_catalog_large_{args.cat_date}.csv', subproduct = args.inpaint_subproduct))

        # Make masks for gapfill
        mask1 = maps.mask_srcs(args.shape,args.wcs,np.asarray((ldecs,lras)),args.large_hole)
        mask2 = maps.mask_srcs(args.shape,args.wcs,np.asarray((rdecs,rras)),args.regular_hole)
        jmask = mask1 & mask2
        jmask = ~jmask
        
        # if args.not_srcfree:
        #     print('adding srcfull mask')
        #     planck70 = datamodel.read_mask(mask_fn='dr6v4_lensing_20240918_planck_galactic_mask_70.fits', subproduct=args.mask_subproduct)
        #     planck70 = enmap.extract( enmap.downgrade(planck70,2), args.shape, args.wcs)
            
        #     inpaint_mask = datamodel.read_mask(mask_fn='source_mask_15mJy_and_dust_rad5.fits', subproduct=args.mask_subproduct)
        #     inpaint_mask = enmap.enmap(enmap.downgrade(inpaint_mask,2), dtype=bool)
        #     inpaint_mask = ~inpaint_mask

        #     final_mask = enmap.enmap((~jmask & ~inpaint_mask) * planck70, dtype=bool)
        #     return final_mask
        
        return jmask
    
    else:
        print('not inpainting')
        return None



def get_metadata(qid, splitnum=0, coadd=False, args=None):
    """
    Retrieves metadata for a specific qid (split/coadd).
    
    ### Paramters:
    - qid: str, unique identifier for the data (must be in sofind)
    - splitnum: int, split of data to be used, will be ignored if coadd is True
    - coadd: bool, if True, coadd the data (ignore splitnum)
    - args: argparse.Namespace(), must contain the following attributes:
        * args needed for get_inpaint_mask()
        * args.maps_subproduct: str, subproduct name for maps (e.g. 'maps_srcfree', 'maps)
        * args.cal_subproduct: str, subproduct name for calibration
        * args.poleff_subproduct: str, subproduct name for polarization efficiency
        * args needed for BeamHelper()
        * args needed for NoiseMetadata()
        * args.khfilter: int, horizontal fourier strip width [used for ground-pickup removal]
        * args.kvfilter: int, vertical fourier strip width [used for ground-pickup removal]     
    """
    
    meta = bunch.Bunch({})
    #assert 0 <= splitnum < 4, "only supporting splits [0,1,2,3]"
    if parse_qid_experiment(qid)=='planck':
        meta.Name = 'planck_npipe'
        meta.dm = DataModel.from_config(meta.Name)
        meta.splits = np.array([1,2])
        meta.nsplits = 2
        isplit = None if coadd else (splitnum // 2 + 1)
        meta.calibration = meta.dm.read_calibration(qid, subproduct=args.cal_subproduct, which='cals')
        meta.pol_eff = meta.dm.read_calibration(qid, subproduct=args.poleff_subproduct, which='poleffs')
        meta.Beam = PlanckBeamHelper(meta.dm, args, qid, isplit)
        meta.beam_fells = meta.Beam.get_effective_beam()[1]
        meta.transfer_fells = meta.Beam.get_effective_beam()[2]
        meta.inpaint_mask = None
        meta.kspace_mask = None
        meta.maptype = 'reprojected'
        meta.noisemodel = PlanckNoiseMetadata(qid, verbose=True,
                                              config_name=meta.Name,
                                              subproduct_name="noise_sims")
        # assigning ACT splits 0 + 1 to Planck split 1
        # and ACT splits 2 + 3 to Planck split 2
        
    elif parse_qid_experiment(qid)=='act':
        
        meta.Name = 'act_dr6v4'
        meta.dm = DataModel.from_config(meta.Name)
        qid_dict = meta.dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
        
        meta.nsplits = qid_dict['num_splits']
        meta.splits = np.arange(meta.nsplits)
        meta.daynight = qid_dict['daynight']
   
        meta.calibration = meta.dm.read_calibration(qid, subproduct=args.cal_subproduct, which='cals')
        meta.pol_eff = meta.dm.read_calibration(qid.split('_')[0], subproduct=args.poleff_subproduct, which='poleffs')
        
        # if meta.daynight != 'night':
        #     meta.calibration /= meta.dm.read_calibration(qid.split('_')[0], subproduct='dr6v4_calday', which='cals')

        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)
        meta.maptype = 'native'
        meta.noisemodel = ACTNoiseMetadata(qid, verbose=True)
        meta.nspecs = nspecs
        meta.specs = specs_weights['EpureB'] if args.pureEB else specs_weights['EB']
        isplit = None if coadd else splitnum
        
        meta.Beam = ACTBeamHelper(meta.dm, args, qid, isplit, coadd=coadd)
        meta.beam_fells = meta.Beam.get_effective_beam()[1]
        meta.transfer_fells = meta.Beam.get_effective_beam()[2]
        
        meta.leakage_matrix = None
        meta.deconvolve_beam = False
        if args.deconvolve_beam:
            meta.deconvolve_beam = True
            if args.leakage_corr:
                meta.leakage_matrix = meta.Beam.get_invleakage_matrix()
        
    elif parse_qid_experiment(qid)=='lat_iso':
        meta.Name = 'so_lat_pipe4_BN' ##this should be passed as argument otherwise use default
        meta.dm = DataModel.from_config(meta.Name)
        qid_dict = meta.dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
        
        meta.nsplits = qid_dict['num_splits']
        meta.splits = np.arange(meta.nsplits)
        meta.calibration = meta.dm.read_calibration(qid, subproduct=args.cal_subproduct, which='cals')
        meta.pol_eff = meta.dm.read_calibration(qid, subproduct=args.poleff_subproduct, which='poleffs')
       

        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)
        meta.maptype = 'native'
        meta.noisemodel = SOLATNoiseMetadata(qid, verbose=True) 
        meta.nspecs = nspecs
        meta.specs = specs_weights['EpureB'] if args.pureEB else specs_weights['EB']
        isplit = None if coadd else splitnum
        
        meta.Beam = SOLATBeamHelper(meta.dm, args, qid, isplit, coadd=coadd)
        meta.beam_fells = meta.Beam.get_effective_beam()[1]  #done
        meta.transfer_fells = meta.Beam.get_effective_beam()[2] #done

    elif parse_qid_experiment(qid)=='so_mss2':
        meta.Name = 'so_lat_mbs_mss0002' ##this should be passed as argument otherwise use default
        meta.dm = DataModel.from_config(meta.Name)
        qid_dict = meta.dm.get_qid_kwargs_by_subproduct(product='maps', subproduct=args.maps_subproduct, qid=qid)
        
        meta.nsplits = qid_dict['num_splits']
        meta.splits = np.arange(meta.nsplits)
        meta.calibration = meta.dm.read_calibration(qid, subproduct=args.cal_subproduct, which='cals')
        meta.pol_eff = meta.dm.read_calibration(qid, subproduct=args.poleff_subproduct, which='poleffs')
       

        meta.inpaint_mask = get_inpaint_mask(args, meta.dm)
        meta.kspace_mask = np.array(maps.mask_kspace(args.shape, args.wcs, lxcut=args.khfilter, lycut=args.kvfilter), dtype=bool)
        meta.maptype = 'native'
        meta.noisemodel = SOsimsNoiseMetadata(qid, verbose=True) 
        meta.nspecs = nspecs
        meta.specs = specs_weights['EpureB'] if args.pureEB else specs_weights['EB']
        isplit = None if coadd else splitnum
        
        meta.Beam = SOsimsBeamHelper(meta.dm, args, qid, isplit, coadd=coadd)
        meta.beam_fells = meta.Beam.get_effective_beam()[1]  #done
        meta.transfer_fells = meta.Beam.get_effective_beam()[2] #done


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
    elif is_lat_iso(qid):
        maptag='map'
    else:
        maptag='map_srcfree'
    return datamodel.read_map(qid=qid, coadd=coadd,
                              split_num=splitnum,
                              subproduct=args.maps_subproduct,
                              maptag=maptag)

def process_beam(sofind_beam, norm=True, interp=True):
    '''
    normalized beam if required and then interpolate
    '''
    ell_bells, bells = sofind_beam[0], sofind_beam[1]
    assert ell_bells[0] == 0

    if norm:
        bells /= bells[0]
        
    if interp:
        return maps.interp(ell_bells, bells, fill_value='extrapolate')
    else:
        return ell_bells, bells

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
        
        tf = self.get_tf()(np.arange(self.mlmax+1))
        
        beam = self.get_beam()(np.arange(self.mlmax+1))
        
        fkbeam[0] = beam * tf
        fkbeam[1] = beam
        fkbeam[2] = beam

        return fkbeam, beam, tf           
class ACTBeamHelper:
    
    """
    Helper class to read and process the beam and transfer function for ACT data.
    
    ### Parameters:
    - datamodel: DataModel object, used to read sofind products
    - args: argparse.Namespace(), must contain the following attributes:
        * args.mlmax: int, maximum multipole for the beam. Will use default (4000) if unset.
        * args.beam_subproduct: str, subproduct name for beam (e.g. 'beams_v4_20230902'). Will use default ('dummy_beams') if unset.
        * args.tf_subproduct: str, subproduct name for transfer function (e.g. 'tf_v4_20230902'). Will use default ('dummy_tf') if unset.
    - qid: str, unique identifier for the data (must be in sofind)
    - isplit: int, split of data to be used, will be ignored if coadd is True
    - coadd: bool, if True, coadd the data (ignore splitnum)    
    
    ### Methods:
    
    - get_beam(): returns the beam 
    - get_tf(): returns the transfer function 
    - get_effective_beam(): returns the effective beam (effective beam, beam, transfer function)
    """

    def __init__(self, datamodel, args, qid, isplit=0, coadd=False):
        
        default_values = {'tf_subproduct': 'dummy',
                          'beam_subproduct': 'dummy',
                          'mlmax': 4000}
        for key, value in default_values.items():
            if not hasattr(args, key):
                print(f"Setting default value for '{key}': {value}")
            setattr(args, key, getattr(args, key, value))

        self.datamodel = datamodel
        self.mlmax = args.mlmax
        self.qid = qid
        self.isplit = isplit
        self.coadd = coadd
        self.beam_subproduct = args.beam_subproduct
        self.tf_subproduct = args.tf_subproduct

    def get_beam(self, interp=True):
        
        if self.beam_subproduct == 'beams_v4_20230130_snfit':
            
            beam_T = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd, tpol = 'T')
            beam_P = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd, tpol = 'POL')

            
            # if self.daynight != 'night':
            #     beam_T[1] = beam_T[1] * CAL_CORRECTION[self.qid.split('_')[0]]
            #     beam_P[1] = beam_P[1] * POL_CORRECTION[self.qid.split('_')[0]]

            beam_T = process_beam(beam_T, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
            beam_P = process_beam(beam_P, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
        
            return beam_T, beam_P
        
        else:
            beam_map = self.datamodel.read_beam(subproduct=self.beam_subproduct, qid=self.qid, split_num=self.isplit, coadd=self.coadd)
            
            # if self.daynight != 'night':
            #     print('removing tf from beam --day')
            #     tf_night = self.datamodel.read_tf(subproduct=self.tf_subproduct, qid=self.qid.split('_')[0])
            #     tf = maps.interp(tf_night[0], tf_night[1], fill_value='extrapolate')
            #     #beam_map_T = process_beam(beam_map, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
            #     # beam_map[1] *= self.datamodel.read_calibration(self.qid.split('_')[0], subproduct=self.poleff_subproduct)
            #     beam_map[1] /= tf(np.arange(len(beam_map[0]))) #  / self.datamodel.read_calibration(self.qid.split('_')[0], subproduct=self.poleff_subproduct))
            #     beam_map = process_beam(beam_map, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct))
            # # if self.daynight != 'night':
            # #     beam_map[1] = beam_map[1] * CAL_CORRECTION[self.qid.split('_')[0]]
                
            #     return beam_map # beam_map_T, beam_map_P

            # else:
            return process_beam(beam_map, self.datamodel.get_if_norm_beam(subproduct=self.beam_subproduct), interp=interp)
            
    def get_tf(self):
        ells_tf, tf = self.datamodel.read_tf(subproduct=self.tf_subproduct, qid=self.qid)
        return maps.interp(ells_tf, tf, fill_value='extrapolate')

    def get_effective_beam(self):
        fkbeam = np.empty((nspecs, self.mlmax+1)) + np.nan
        
        tf = self.get_tf()(np.arange(self.mlmax+1))
        
        # if self.daynight != 'night': #if self.beam_subproduct == 'beams_v4_20230130_snfit':
            
        #     beam_T, beam_P = self.get_beam()
        #     beam_T = beam_T(np.arange(self.mlmax+1))
        #     beam_P = beam_P(np.arange(self.mlmax+1))
            
        #     beam = beam_T # beam_P!
        #     fkbeam[0] = beam_T
        #     fkbeam[1] = beam_P
        #     fkbeam[2] = beam_P
            
        # else:
        beam = self.get_beam()(np.arange(self.mlmax+1))
        
        fkbeam[0] = beam * tf
        fkbeam[1] = beam # / self.datamodel.read_calibration(self.qid.split('_')[0], subproduct=self.poleff_subproduct)
        fkbeam[2] = beam # / self.datamodel.read_calibration(self.qid.split('_')[0], subproduct=self.poleff_subproduct)

        return fkbeam, beam, tf

    def get_invleakage_matrix(self):
        
        te = self.datamodel.read_beam(qid=self.qid, subproduct='beams_leakage', coadd=True, leakage_comp='e')
        ell = te[0]
        te = te[1]
        tb = self.datamodel.read_beam(qid=self.qid, subproduct='beams_leakage', coadd=True, leakage_comp='b')
        assert np.all(tb[0] == ell)
        tb = tb[1]
                
        # assert self.coadd is False
        
        sbeam = self.get_beam(interp=False)
        assert np.all(sbeam[0] == ell)
        sbeam = sbeam[1]
        
        self.coadd = True
        cbeam = self.get_beam(interp=False)
        self.coadd = False
        assert np.all(cbeam[0] == ell)
        cbeam = cbeam[1]
        
        array0 = np.zeros(len(ell))
        
        beam_matrix = np.array([[sbeam, array0, array0],
                                [te * cbeam * sbeam, sbeam, array0],
                                [tb * cbeam * sbeam,  array0, sbeam]])
        invmatrix = (np.linalg.inv(beam_matrix.T)).T
        
        return invmatrix

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
    
    def read_in_sim(self, isplit, index, lmax=4000):
        # REQUIRES MODIFICATION TO PIXELL (ask Frank/Joshua)
        try:
            residual_map = hp.read_map(self.noise_map_path(isplit, index),
                                       field=(0,1,2))
        except IndexError:
            residual_map = hp.read_map(self.noise_map_path(isplit, index),
                                       field=(0))
            print("No pol found, setting E/B to 0.")
            residual_map = np.array([residual_map,
                                     residual_map*0.,
                                     residual_map*0.])
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

    def read_in_sim(self,split_num, sim_num, lmax=5400, alm=True,  fwhm=1.6,  mask=None):
        
        # grab a sim from disk, fail if does not exist on-disk
        my_sim = self.tnm.get_sim(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, generate=False)
        index = self.get_index_sim_qid(self.qid)
        my_sim = my_sim[index].squeeze()

        if mask is not None:
            print('I am re-masking the noisy sim')
            bl = maps.gauss_beam(np.arange(lmax+1), fwhm)
            alm_con = cs.almxfl(my_sim,bl)
        
            new_map = cs.alm2map(alm_con, enmap.empty((3,) + mask.shape, mask.wcs)) * mask
            new_alm = cs.almxfl(cs.map2alm(new_map, lmax=lmax), 1/bl)
    
            return new_alm  

        else:
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

    def read_in_sim(self,split_num, sim_num, lmax=5400, alm=True,  fwhm=1.6, mask=None):
        
        # grab a sim from disk, fail if does not exist on-disk
        my_sim = self.tnm.get_sim(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, generate=False)
        index = self.get_index_sim_qid(self.qid)
        my_sim = my_sim[index].squeeze()

        if mask is not None:
            print('I am re-masking the noisy sim')
            bl = maps.gauss_beam(np.arange(lmax+1), fwhm)
            alm_con = cs.almxfl(my_sim,bl)
        
            new_map = cs.alm2map(alm_con, enmap.empty((3,) + mask.shape, mask.wcs)) * mask
            new_alm = cs.almxfl(cs.map2alm(new_map, lmax=lmax), 1/bl)
    
            return new_alm  

        else:
            return my_sim   
        
    
class ACTNoiseMetadata:
    
    '''
    A class to handle the noise metadata for ACT
    
    ### Initialization paramters:
    - qid: str, unique identifier for the data (must be in sofind)
    - verbose: bool, if True, print additional information during initialization

    ### Methods:
    - read_in_sim(split_num, sim_num, lmax=5400, alm=True): 
        Reads in a simulated noise realization (alm) for the given split and simulation number.
    - get_index_sim_qid(qid): 
        mnms sims from the same array (e.g. 'pa5a_pa5b') are stored in the same file.
        depending on the qid, it will return the corresponding component (e.g. 'pa5a' or 'pa5b').
    '''
    
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
    
    def get_index_sim_qid(self,qid):

        # Define a mapping from order to index
        order_to_index = {'a': 0, 'b': 1}
        # Extract the order from the qid
        order = qid.split('pa')[1][1]
        # Get the index corresponding to the order
        index = order_to_index[order]

        return index

    def read_in_sim(self,split_num, sim_num, lmax=5400, alm=True): #, fwhm=1.6, mask=None):
        
        # grab a sim from disk, fail if does not exist on-disk
        my_sim = self.tnm.get_sim(split_num=split_num, sim_num=sim_num, lmax=lmax, alm=alm, generate=False)
        index = self.get_index_sim_qid(self.qid)
        my_sim = my_sim[index].squeeze()
        return my_sim
        
        # if mask is not None:
        #     print('I am re-masking the noisy sim')
        #     bl = maps.gauss_beam(np.arange(lmax+1), fwhm)
        #     alm_con = cs.almxfl(my_sim,bl)
        
        #     new_map = cs.alm2map(alm_con, enmap.empty((3,) + mask.shape, mask.wcs)) * mask
        #     new_alm = cs.almxfl(cs.map2alm(new_map, lmax=lmax), 1/bl)
    
        #     return new_alm  

        # else:
        #     return my_sim        



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

        # Determine split letter (coadd case doesn't matter)
        sl = 'A' if self.isplit == 1 else 'B'

        # Load and interpolate the beam
        ell_b, bl = self.datamodel.read_beam(self.qid,
                        subproduct=self.beam_subproduct,
                        split_num=sl,
                        coadd=(self.isplit is None))
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
    '''
    A class to handle the generation and management of foregrounds for simulations
    
    ### Initialization parameters:
    - datamodel: DataModel object, used to read sofind products
    - args: argparse.Namespace(), must contain the following attributes:
        * args.fg_type: str, type of foregrounds, 'sims' or 'theory'
        * args.fgs_path: str, path to foreground sims
        * args.is_noiseless: bool, if True, no foregrounds are generated
        * args.lmax_signal: int, maximum ell of signal sims
    
    ### Methods:
    - generate_cov_fgs(fgs_path, lmax): 
        Generates the foreground covariance matrix (power spectrum) for the given lmax 
        at two frequencies (90 and 150 GHz) and their cross-spectrum.
    - get_map_fgs(qid, alms_f):
        Returns the foreground map corresponding to the given qid and alms_f. qid informs whether to use f150 or f090 from alms_f.
    - get_fg_alms(fgcov, qid, cmb_set, sim_indices):
        Generates alm from cl for the given qid, cmb_set, and sim_indices (the latter 2 are used in the seed).
    '''

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
                    foreground_cluster=None, deconvolve_beam_bool=False, beam=None, leakage=None, mlmax=5000):
    """
    This function will load a rectangular pixel map and pre-process it.
    This involves inpainting, masking in real and Fourier space
    and removing a pixel window function. It also removes a calibration
    and polarization efficiency.
    For simulations ivar processing is redundant, we should probably set ivar as an optional argument

    pass deconv_beam = True if you wanna do deconvolution
    Leakage is the inverse variance leakage matrix that needs to be applied to the alms

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
        
    if deconvolve_beam_bool:
        print('deconv beam')
        imap = deconvolve_beam(imap, mask, mlmax, beam=beam, leakage=leakage)

    imap = imap * mask
    imap = depix_map(imap,maptype=maptype,dfact=dfact,kspace_mask=kspace_mask)

    imap = imap * calibration
    imap[1:] = imap[1:] / pol_eff
    
    if ivar is not None:
        ivar = ivar / calibration**2.
        ivar[1:] = ivar[1:] * pol_eff**2.
    
    return imap, ivar # ivar will be none if nothing happened to it

def deconvolve_beam(imap, mask, mlmax, beam=None, leakage=None):
    
    alm_a = cs.map2alm(imap * mask, lmax = mlmax)
        
    if leakage is not None:
        info = cs.alm_info(nalm=alm_a.shape[-1])
        print('taking into account leakage')
        alm_T = info.lmul(alm_a[0], leakage[0,0,:])
        alm_E = info.lmul(alm_a[0], leakage[1,0,:]) + info.lmul(alm_a[1], leakage[1,1,:])
        alm_B = info.lmul(alm_a[0], leakage[2,0,:]) + info.lmul(alm_a[2], leakage[2,2,:])
        deconv = np.array([alm_T, alm_E, alm_B], dtype=np.complex128)

    else: 
        assert beam is not None
        deconv = cs.almxfl(alm_a, 1/beam) 

    imap = cs.alm2map(deconv, enmap.empty((3,) + mask.shape,mask.wcs))
    return imap

def get_signal_sim_core(shape,wcs,signal_alms,
                        beam_fells,transfer_fells,
                        calibration,pol_eff,
                        apod_y_arcmin = 0.,apod_x_arcmin = 0.,
                        maptype='native'):
    """
    This needs to only be called once if the beam, transfer, cal, poleff, etc. are the same for each split.
    """
    
    isignal_alms = signal_alms.copy()
    isignal_alms[0] = cs.almxfl(isignal_alms[0],beam_fells * transfer_fells)
    isignal_alms[1] = cs.almxfl(isignal_alms[1],beam_fells)
    isignal_alms[2] = cs.almxfl(isignal_alms[2],beam_fells)
    omap = cs.alm2map(isignal_alms,enmap.empty((3,)+shape,wcs,dtype=np.float32))
    if maptype=='native':
        if (apod_y_arcmin>1e-3) or (apod_x_arcmin>1e-3):
            res = maps.resolution(shape,wcs) / u.arcmin
            omap = enmap.apod(omap, (apod_y_arcmin/res,apod_x_arcmin/res))
        omap = enmap.apply_window(omap,pow=1)
    elif maptype=='reprojected':
        pass
    else:
        raise ValueError

    # notice how these are inverse of what's in preprocess
    # not applied to nmap because it is already based on data (which includes them)
    omap = omap / calibration  
    omap[1:] = omap[1:] * pol_eff
    
    return omap
    
def get_noise_sim_core(shape,wcs,
                       noise_alms=None,
                       noise_mask=None,
                       rms_uk_arcmin=None,
                       noise_lmax = 5400,
                       lcosine=80):
    """
    This needs to be called each time for each split.
    """
    
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

    return nmap
    
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

    You probably shouldn't be calling this function since it is wasteful when doing splits.

    """
    omap = get_signal_sim_core(shape,wcs,signal_alms,
                               beam_fells, transfer_fells,
                               calibration, pol_eff,
                               apod_y_arcmin, apod_x_arcmin,
                               maptype)
    nmap = get_noise_sim_core(shape,wcs,
                       noise_alms,
                       noise_mask,
                       rms_uk_arcmin,
                       noise_lmax,
                       lcosine)
    omap = omap + nmap
    return omap


def calculate_noise_power(nmap, mask, mlmax, nsplits, pureEB):    
    
    ''' 
    equation (9) Qu+24 paper
    '''
    
    cl_ab = []
    for k in range(nsplits):

        if pureEB:
            _,Balm=simgen.pureEB(nmap[k][1],nmap[k][2],mask,returnMask=0,lmax=mlmax,isHealpix=False)
            alm_T, alm_E, _ =cs.map2alm(nmap[k],lmax=mlmax)
            alm_a=np.array([alm_T,alm_E,Balm], dtype=np.complex128)
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
    
    '''
    the cluster finding algorithm has a corresponding beam.
    we take that beam out and apply in the beam of the experiment. 
    this could be the beam for the exact qid and split or all splits with the same (coadd) beam, hence the two options
    '''
    
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
    daynight = re.search(r'(daydeep|daywide|night|pipe4|mss0002)', mask_fn).group(0)
    # find skyfraction. it is 2 digit number. the string between the last "_" and ".fits"
    skyfrac  = re.search(r'_(\d{2})(?:_[^_]+)?\.fits$', mask_fn).group(1) # re.search(r'_([^_]+)\.fits$', mask_fn).group(1)

    return f'{daynight}_{skyfrac}'

def read_weights(args):
    
    '''
    reads in the fcoadd weights
    '''

    nqids = len(args.qids)
    noise_specs = np.zeros((nspecs, nqids, args.mlmax+1), dtype=np.float64)

    if args.pureEB:
        specs = specs_weights['EpureB']
    else:
        specs = specs_weights['EB']

    for i, qid in enumerate(args.qids):
        for ispec, spec in enumerate(specs):
            noise_specs[ispec, i] = np.loadtxt(get_fout_name(get_name_weights(qid, spec), args, stage='weights'))[:args.mlmax+1]
    
    return noise_specs

def get_fout_name(fname, args, stage, tag=None):

    '''
    fname: name of file to be saved
    args: argparse object containing args.output_dir
    stage: str
        ['weights', 'cluster_fgmap', 'kspace_coadd', 'nilc_coadd']
    tag: optional, used in kspace_coadd and nilc_coadd to distinguish between sim (tag='sim') and data (tag=None), because we store them in different folders
    '''
    fname = fname.split('.fits')[0]
    
    try:
        fcoadd_folder = f'{args.mask_tag}_fcoadd'
    except AttributeError:
        fcoadd_folder = "default_fcoadd"

    if stage == 'weights':
        fname += '_weights.txt'
        folder = f'../../{fcoadd_folder}/stage_compute_weights/'
    
    elif stage == 'cluster_fgmap':
        fname += '_cluster_fgmap.fits'
        folder = f'../../{fcoadd_folder}/stage_cluster_fgmap/'

    elif stage == 'kspace_coadd':
        fname  = 'kspace_coadd_' + fname + '.fits'
        if tag == 'sim':
            folder = '../stage_kspace_coadd_sims/'
        else:
            folder = '../stage_kspace_coadd/'

    elif stage == 'nilc_coadd':
        fname  = 'nilc_coadd_' + fname + '.fits'
        if tag == 'sim':
            folder = '../stage_nilc_coadd_sims/'
        else:
            folder = '../stage_nilc_coadd'

    output_dir = os.path.join(args.output_dir, folder)
    # create output folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    return os.path.join(output_dir, fname)

