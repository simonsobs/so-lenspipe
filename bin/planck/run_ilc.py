from __future__ import print_function
import os,sys
sys.path.append("/global/homes/m/maccrann/cmb/lensing/code/HILC/")
from HILC.ilc import harmonic_ilc
from os.path import join as opj, abspath, dirname
from orphics import maps,io,cosmology,stats,pixcov
from pixell import enmap,curvedsky,utils,enplot,reproject
import numpy as np
import healpy as hp
import argparse
import falafel.utils as futils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orphics import maps
from planck_ilc_tools import get_act_alm, get_planck_Talm, response_fnc_cmb, response_fnc_cib, response_fnc_tsz
from cmbsky.utils import safe_mkdir
from scipy.signal import savgol_filter
import yaml
import copy

with open("run_ilc_defaults.yml",'rb') as f:
    DEFAULTS=yaml.load(f)
from prep_planck_alms import get_config

ACT_FREQS = [90, 150]
PLANCK_FREQS = [353, 545]

def covmat_from_websky_Cltot(cltot_file, qids, lmax,
                             sg_window_length=31, sg_order=2):
    #relate qids to labels in cltot_file
    cltot_data = np.load(cltot_file)
    n_qid = len(qids)
    qid_to_key = {90: "0093", 150 : "0145",
                  353: "0353", 545 : "0545"}
    covmat = np.zeros((lmax+1, n_qid, n_qid))
    for i in range(n_qid):
        qid_i = qids[i]
        freq_i = qid_to_key[qid_i]
        for j in range(i,n_qid):
            qid_j = qids[j]
            freq_j = qid_to_key[qid_j]
            try:
                cltot = cltot_data["%s-%s"%(freq_i, freq_j)]
            except KeyError:
                cltot = cltot_data["%s-%s"%(freq_j, freq_i)]
            covmat[:,i,j] = cltot[:lmax+1]
            covmat[:,j,i] = cltot[:lmax+1]
    return covmat
            
def main():

    config,_,_ = get_config(defaults=DEFAULTS)
    
    #First set up the ilc
    target_fwhm_arcmin = 2.
    ells = np.arange(config.lmax+1)
    target_beam = maps.gauss_beam(
        ells, target_fwhm_arcmin)

    #Set up the ilc
    qids = sorted(ACT_FREQS+PLANCK_FREQS)
    n_qids = len(qids)
    version, shape, wcs, mask_fncs, dataModel = None,None,None,None,None
    target_fwhm_arcmin=2.
    data_hilc = harmonic_ilc(qids, version, shape, wcs, mask_fncs,
                        target_fwhm_arcmin=target_fwhm_arcmin,
                        dataModel=dataModel,lmax=config.lmax)

    #Read the data alms
    planck_alms = [get_planck_Talm(freq, config.planck_alm_dir, lmax=config.lmax)
                   for freq in PLANCK_FREQS]
    act_alms = [get_act_alm(freq, lmax=config.lmax, act_data_dir=config.act_data_dir) for freq in ACT_FREQS]
    #average splits for act as this is the power we'll assume for
    #the ilc
    act_mean_alms = [(np.array(split_alms)).mean(axis=0) for split_alms in act_alms]

    for qid in qids:
        if qid in ACT_FREQS:
            alm = act_mean_alms[ACT_FREQS.index(qid)][0] #0th element for T
        else:
            alm = planck_alms[PLANCK_FREQS.index(qid)] #For Planck we only saved T
        print(alm.shape)
        data_hilc.alms_T[qid] = curvedsky.almxfl(alm, target_beam)

    #Now select the covariance we're using 
    if config.covmat_from_file:
        print("loading covmat from %s"%config.covmat_from_file)
        covmat_raw = np.load(config.covmat_from_file)
    elif config.use_websky_cltot_file is not None:
        covmat_raw = covmat_from_websky_Cltot(
            config.use_websky_cltot_file,
            qids, config.lmax, beam_fwhm=target_fwhm_arcmin)
        safe_mkdir(config.output_dir)
        np.save(opj(config.output_dir, "covmat_raw.npy"), covmat_raw)
    else:
        #compute covariance from data
        covmat_raw = data_hilc.compute_covMat_noDebias()
        covmat_debiased = data_hilc.compute_covMat_wBiasReduction()
        safe_mkdir(config.output_dir)
        np.save(opj(config.output_dir, "covmat_raw.npy"), covmat_raw)
        np.save(opj(config.output_dir, "covmat_debiased.npy"), covmat_debiased)

    #if smooth covmat
    if config.smooth_covmat:
        fig,ax = plt.subplots()
        covmat_smooth = np.zeros_like(covmat_raw)
        k=0
        for i in range(n_qids):
            for j in range(i,n_qids):
                cl = covmat_raw[:,i,j]
                cl_smooth = savgol_filter(
                    cl, config.sg_window_length,
                    config.sg_order)
                covmat_smooth[:,i,j] = cl_smooth
                covmat_smooth[:,j,i] = cl_smooth
                if i==j:
                    ax.plot(ells[10:], cl[10:], color='C%d'%k,
                            label="%d,%d"%(qids[i],qids[j]))
                    ax.plot(ells[10:], cl_smooth[10:],
                            linestyle='--', color='C%d'%k)
                    k+=1
        ax.set_yscale('log')
        ax.set_xlabel("l")
        ax.legend()
        fig.tight_layout()
        fig.savefig(opj(config.output_dir, "cls.png"))
        np.save(opj(config.output_dir, "covmat_smooth.npy"), covmat_smooth)
        data_hilc.covMat = covmat_smooth

    #Now we can run ilc
    #Do i) straight ilc
    #   i) with CIB deprojection
    #   ii) with CIB and tSZ deprojection
    def save_ilc_alms(hilc, output_dir, use_debiased_cov, lmax,
                      act_sim_seed=None, planck_sim_seed=None,
                      act_sim_dir=None, planck_alm_dir=None,
                      ):
        alm_size = hp.Alm.getsize(lmax)
        for isplit in range(4):
            print("split: %d"%isplit)
            if ((act_sim_seed is None) and (planck_sim_seed is None)):
                for qid in qids:
                    if qid in ACT_FREQS:
                        alm = act_alms[ACT_FREQS.index(qid)][isplit][0] #0th element for T
                    else:
                        alm = planck_alms[PLANCK_FREQS.index(qid)] #For Planck we only saved T
                    assert len(alm)==alm_size
                    hilc.alms_T[qid] = curvedsky.almxfl(alm, target_beam)
            else:
                assert act_sim_seed is not None
                assert planck_sim_seed is not None
                for qid in qids:
                    if qid in ACT_FREQS:
                        alm = get_act_alm(qid,  sim_seed=act_sim_seed, nsplit=4, lmax=lmax,
                                          act_sim_dir=config.act_sim_dir)[isplit][0]
                    else:
                        alm = get_planck_Talm(qid, planck_alm_dir, sim_seed=planck_sim_seed, lmax=lmax,
                                             act_sim_seed=act_sim_seed)
                    assert len(alm)==alm_size
                    hilc.alms_T[qid] = curvedsky.almxfl(alm, target_beam)

            if use_debiased_cov:
                apply_ilc, apply_ilc_deproj = hilc.apply_ilc_wBiasReduction, hilc.apply_ilc_deproj_wBiasReduction
            else:
                apply_ilc, apply_ilc_deproj = hilc.apply_ilc_noDebias, hilc.apply_ilc_deproj_noDebias
            ilc_alms = apply_ilc(response_fnc_cmb)
            f = opj(output_dir, "hilc_split%d.fits"%isplit)
            print("writing ilc alms to %s"%f)
            hp.fitsfunc.write_alm(f, ilc_alms, overwrite=True)
            cibd_alms = apply_ilc_deproj(response_fnc_cmb, [response_fnc_cib])
            f = opj(output_dir, "hilc-cibd_split%d.fits"%isplit)
            print("writing cib-deprojected alms to %s"%f)
            hp.fitsfunc.write_alm(f, cibd_alms, overwrite=True)
            tszd_alms = apply_ilc_deproj(response_fnc_cmb, [response_fnc_tsz])
            f = opj(output_dir, "hilc-tszd_split%d.fits"%isplit)
            print("writing tsz-deprojected alms to %s"%f)
            hp.fitsfunc.write_alm(f, tszd_alms, overwrite=True)
            cibandtszd_alms = apply_ilc_deproj(response_fnc_cmb, [response_fnc_tsz, response_fnc_cib])
            f = opj(output_dir, "hilc-tszandcibd_split%d.fits"%isplit)
            print("writing tsz and cib-deprojected alms to %s"%f)
            hp.fitsfunc.write_alm(f, cibandtszd_alms, overwrite=True)

    #Save data ilc

    print("running data ilc:")
    save_ilc_alms(data_hilc, config.output_dir, config.use_debiased_cov, config.lmax,
                  act_sim_seed=None, planck_sim_seed=None,
                  act_sim_dir=None, planck_alm_dir=config.planck_alm_dir)

    for sim_set in [0, 1]:
        planck_sim_seed = getattr(config, "planck_sim_start_set%02d"%sim_set)
        for isim in range(config.act_sim_start,
                          getattr(config, "nsim_set%02d"%sim_set)+config.act_sim_start):
            act_sim_seed = (sim_set, isim)
            output_dir = opj(config.output_dir, "sim_planck%03d_act%02d_%05d"%(planck_sim_seed, sim_set, isim))
            safe_mkdir(output_dir)
            save_ilc_alms(copy.deepcopy(data_hilc), output_dir, config.use_debiased_cov,
                          config.lmax, act_sim_seed=act_sim_seed, planck_sim_seed=planck_sim_seed,
                          act_sim_dir=config.act_sim_dir, planck_alm_dir=config.planck_alm_dir)
            planck_sim_seed += 1
        
if __name__=="__main__":
    main()
