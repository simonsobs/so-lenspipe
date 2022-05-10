#Script for making and testing ilc with Planck
from solenspipe.utility import kspace_mask
from pixell import curvedsky, enmap, reproject
from os.path import join as opj
import falafel.utils as futils
import healpy as hp
import astropy.io.fits as afits
from falafel import utils as futils
import numpy as np

NPIPE_VERSION="6v20"

PLANCK_MAP_DIR="/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe%s"%NPIPE_VERSION
PLANCK_SRCFREE_MAP_DIR="/global/project/projectdirs/act/data/synced_maps/NPIPE/"
PLANCK_SIM_MAP_DIR="/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe%s_sim"%NPIPE_VERSION
PLANCK_ALM_DIR="/global/cscratch1/sd/maccrann/cmb/planck_npipe6v20_alm_v0_lmax2000/"

ACT_DATA_DIR="/global/cscratch1/sd/maccrann/cmb/act_dr6/alm_v3"
ACT_SIM_DIR="/global/cscratch1/sd/maccrann/cmb/act_dr6/alm_v3/sim"

ACT_MASK_FILE="/global/project/projectdirs/act/data/maccrann/dr6/dr6v2_default_union_maskd1.fits"

h=6.62607004e-34
c=2.99792458e8
kb = 1.38064852e-23
T_cmb=2.7255

def cib_sed(freq, lmax=None, beta=1.2, T_cib=24.):
    freq = float(freq)*1.e9
    x = h*freq / kb / T_cmb
    #print(x)
    dBdT = 2*h * freq**3 * x * np.exp(x) / c**2 / T_cmb / (np.exp(x)-1)**2
    #print(dBdT)
    x_cib = h*freq / kb / T_cib
    #print(x_cib)
    return freq**(3+beta) / (np.exp(x_cib) - 1) / dBdT

def response_fnc_cib(qid, lmax=None):
    return cib_sed(qid)/cib_sed(145.)

def response_fnc_cmb(qid,lmax=None):
    return 1.

def y_to_deltaToverT(freq):
    x = float(freq)/56.8
    return 2.7255 * (x * (np.exp(x)+1)/(np.exp(x)-1) - 4)
    
def response_fnc_tsz(qid,lmax=None):
    return y_to_deltaToverT(float(qid))/y_to_deltaToverT(145.)


def generate_planck_Talm(freq, survey_mask, lmax, sim_seed=None,
                         kspace_mask_kwargs={"deconvolve":True},
                         npipe_version=NPIPE_VERSION,
                         planck_data_dir=PLANCK_MAP_DIR,
                         planck_sim_dir=PLANCK_SIM_MAP_DIR,
                         planck_srcfree_data_dir=PLANCK_SRCFREE_MAP_DIR,
                         act_cmb_seed=None,
                         srcfree=True):
    """
    If sim_seed is None, return the data.
    Return beam deconolved temperature alms
    """
    #get b(l) for temperature
    bl_file = opj(planck_data_dir, "quickpol",
                    "Bl_npipe%s_%03dGHzx%03dGHz.fits"%(
                        npipe_version, freq, freq)
                  )
    bl = ((afits.open(bl_file))[1].data)["TEMPERATURE"]
    is_car=False
    add_act_sim_cmb=False
    if sim_seed is None:
        if not srcfree:
            f = opj(planck_data_dir,
                    "npipe%s_%03d_map.fits"%(npipe_version, freq))
            print("reading planck map from %s"%f)
            t_map_hp = hp.read_map(f, field=0, hdu=1)
        else:
            #srcfree maps are already in car, in eq coords
            is_car=True
            f = opj(planck_srcfree_data_dir,
                    "npipe%sABcoadd_%03d_map_srcfree_enmap.fits"%(
                        npipe_version, freq)
                    )
            print("reading srcfree planck map from %s"%f)
            t_map = enmap.read_map(f)[0]
    else:
        #we want the option to  use act sim cmb
        #instead of the original Planck sim one
        #In this case, read in the noise only sim
        if act_cmb_seed is not None:
            add_act_sim_cmb = True
            f = opj(planck_sim_dir, "%04d"%sim_seed,
                    "residual",
                    "residual_npipe%s_%03d_%04d.fits"%(
                        npipe_version, freq, sim_seed))
        else:
            f = opj(planck_sim_dir, "%04d"%sim_seed,
                    "npipe%s_%03d_map.fits"%(npipe_version, freq))
        print("reading planck sim map from %s"%f)
        t_map_hp = hp.read_map(f, field=0, hdu=1)

    if not is_car:
        print("converting to car and rotating to"\
              "equatorial coordinates")
        #In this case we've read an original npipe product
        #so we need to i) convert to equatorial coordinates,
        #multiply by 10^6 to convert to muK
        t_map = reproject.healpix2map(t_map_hp*10**6,
                                      shape=survey_mask.shape,
                                      wcs=survey_mask.wcs,
                                      rot="gal,equ")
        if add_act_sim_cmb:
            #need to add in cmb
            print("adding cmb signal i=%d, set=%d"%(
                act_cmb_seed[0], act_cmb_seed[1])
                  )
            cmb_alm = futils.get_cmb_alm(*act_cmb_seed)[0] #just T
            #convolve beam
            cmb_alm = curvedsky.almxfl(cmb_alm, bl[:hp.Alm.getlmax(len(cmb_alm))])
            cmb_map = enmap.zeros(t_map.shape, t_map.wcs)
            curvedsky.alm2map(cmb_alm, cmb_map)
            #convolve pixel window
            cmb_map = enmap.apply_window(cmb_map)
            t_map += cmb_map

    #Ok, by hook or crook, we have the Planck car map
    #Apply survey mask, do k-space masking, deconvolve
    #pixel window, convert to alm, and deconvolve beam.
    print("applying survey mask")
    t_map_masked = t_map*survey_mask
    print("doing k-space masking")
    t_map_masked_kspace_masked = kspace_mask(
        t_map_masked, **kspace_mask_kwargs)
    print("transforming to alm")
    t_alm = curvedsky.map2alm(
        t_map_masked_kspace_masked, lmax=lmax,
        tweak=True)
    print("deconvolving beam")
    t_alm_deconvolved = curvedsky.almxfl(t_alm, 1./bl[:lmax+1])
    return t_alm_deconvolved

def get_planck_Talm(freq, alm_dir, sim_seed=None, lmax=None,
                    act_sim_seed=None):
    if sim_seed is None:
        alm_file = opj(alm_dir, "%03d_alm.fits"%freq)
    else:
        if act_sim_seed is not None:
            alm_file = opj(alm_dir, "sim%03d_%03d_set%02d_%05d_alm.fits"%(
                sim_seed, freq, act_sim_seed[0], act_sim_seed[1]))
        else:
            alm_file = opj(alm_dir, "sim%03d_%03d_alm.fits"%(
                sim_seed, freq))
    alm = hp.fitsfunc.read_alm(alm_file)
    if lmax is not None:
        alm = futils.change_alm_lmax(alm, lmax)
    return alm

def get_act_alm(freq,  sim_seed=None, nsplit=4, lmax=5000,
                act_data_dir=ACT_DATA_DIR,
                act_sim_dir=ACT_SIM_DIR):

    #If sim_seed is None, return data

    split_alms = []
    for isplit in range(nsplit):
        if sim_seed is None:
            f = opj(
                act_data_dir,
                "kcoadd_hack%dGHznotsz_split%d.fits"%(freq, isplit))
        else:
            f = opj(
                act_sim_dir,
		"kcoadd_hackalms%d_alm_set%02d_%05d_split_%d.fits"%(
                    freq, sim_seed[0], sim_seed[1], isplit)
                )
        print("loading alms from %s"%f)
        alm = hp.fitsfunc.read_alm(f, hdu=(1,2,3))
        alm = futils.change_alm_lmax(
            alm, lmax=lmax)
        split_alms.append(alm)
    return split_alms

