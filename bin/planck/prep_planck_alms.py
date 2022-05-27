#Save Planck data and simulation alms ready to input
#into ilc
import os
from os.path import join as opj
from cmbsky.utils import safe_mkdir, get_disable_mpi
from pixell import enmap, curvedsky
import healpy as hp
from planck_ilc_tools import generate_planck_Talm
import yaml
import argparse

disable_mpi = get_disable_mpi()
if not disable_mpi:
    from mpi4py import MPI

if not disable_mpi:
    comm = MPI.COMM_WORLD
    rank,size = comm.Get_rank(), comm.Get_size()
else:
    rank,size = 0,1

with open("prep_planck_alm_defaults.yml",'rb') as f:
    DEFAULTS=yaml.load(f)
    
def get_config(defaults=DEFAULTS):
    parser = argparse.ArgumentParser(description='Prepare Planck alms')
    #only required arg is output_dir
    parser.add_argument("output_dir", type=str)
    #can also add a config file
    parser.add_argument("-c", "--config_file", type=str, default=None)
    #and otherwise variables are set to defaults,
    #or overwritten on command line
    updated_defaults = {}
    for key,val in defaults.items():
        nargs=None
        if val=="iNone":
            t,val = int, None
        elif val=="fNone":
            t,val = float, None
        elif val=="sNone":
            t,val = str, None
        elif val=="liNone":
            t,val,nargs = int,None,'*'
        elif val=="lfNone":
            t,val,nargs = float,None,'*'
        elif val=="lsNone":
            t,val,nargs = str,None,'*'
        elif isinstance(val, list):
            t = type(val[0])
            nargs='*'
        else:
            t = type(val)
        updated_defaults[key] = val
        parser.add_argument("--%s"%key, type=t, nargs=nargs)
        
    #This parser will have optional arguments set to
    #None if not set. So if that's the case, replace
    #with the default value
    args_dict=vars(parser.parse_args())
    config_file = args_dict.pop("config_file")
    output_dir = args_dict.pop("output_dir")

    config = {}
    if config_file is None:
        config_from_file = {}
    else:
        with open(config_file,"rb") as f:
            config_from_file = yaml.load(f)
        config.update(config_from_file)
        
    config['output_dir'] = output_dir

    for key,val in args_dict.items():
        if key in config:
            if val is not None:
                config[key] = val
        else:
            if val is not None:
                config[key] = val
            else:                
                config[key] = updated_defaults[key]
    #I think most convenient to return
    #a namespace
    from argparse import Namespace
    config_namespace = Namespace(**config)

    #also return the config from the file,
    #and the defaults
    return config_namespace, config_from_file, dict(DEFAULTS)

    
def main():
    config = get_config()[0]
    survey_mask = enmap.read_map(config.mask_file)
    npipe_version="6v20"
    alm_version="0"
    """
    if config.output_dir is None:
        config.output_dir = "/global/cscratch1/sd/maccrann/cmb/planck_npipe%s_alm_v%s_lmax%d"%(
        config.npipe_version, alm_version, config.lmax)
    """
    safe_mkdir(config.output_dir)
    
    def save_alm(freq, sim_seed, act_cmb_seed,
                 srcfree=None):
        print("getting Planck T_alm for %s, freq=%d, lmax=%d"%(
            ("data" if sim_seed is None else str(sim_seed)),
            freq, config.lmax))
        t_alm = generate_planck_Talm(freq, survey_mask, config.lmax, sim_seed=sim_seed,
                                     act_cmb_seed=act_cmb_seed, srcfree=srcfree)
        if sim_seed is not None:
            filename = opj(
                config.output_dir,
                "sim%03d_%s_set%02d_%05d_alm.fits"%(
                    sim_seed, freq, act_cmb_seed[1], act_cmb_seed[0])
                )
            print("writing alms for sim %d to %s"%(sim_seed, filename))
        else:
            filename = opj(
                config.output_dir,
                "%03d_alm.fits"%(
                    freq)
                )
            print("writing alms for data to %s"%filename)
        hp.fitsfunc.write_alm(filename, t_alm, overwrite=True)

    job_args = []
    for freq in config.freqs:
        #the real data
        job_args.append((freq, None, None, config.srcfree))
        #and the sims
        for i,planck_sim_index in enumerate(range(config.set00_sim_start, config.set00_sim_start+config.set00_nsim)):
            act_sim_index = i+config.act_sim_start
            job_args.append((freq, planck_sim_index, (act_sim_index, 0)))
        for i,planck_sim_index in enumerate(range(config.set01_sim_start, config.set01_sim_start+config.set01_nsim)):
            act_sim_index = i+config.act_sim_start
            job_args.append((freq, planck_sim_index, (act_sim_index, 1)))
    njobs=len(job_args)
    print(job_args)

    for i,job_arg in enumerate(job_args):
        if rank>=njobs:
            return
        if i%size != rank:
            continue
        print("rank %d running save_alm with args:"%rank,job_arg)
        save_alm(*job_arg)
        
if __name__=="__main__":
    main()
