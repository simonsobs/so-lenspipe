Staging and parallelization
===========================



Example run
-----------

As an example, let's say that we want to perform the following null test on ACT data:
check that the debiased lensing power spectrum on a split null for
a specific individual array is consistent with zero.

This requires the following:

1. A mask to define the geometry and region of the test
2. Simulations of the




pydpiper config.yaml --stages make_mask
pydpiper config.yaml --stages noise_sim_model 
pydpiper config.yaml --stages generate_noise_sims
pydpiper config.yaml --stages coadd_sims,rdn0,mcn1,mcmf
pydpiper config.yaml --stages coadd_data,qe,debiased_power

SCRIPTS that save to disk:
make_mask
noise_sim_model
generate_noise_sims
kspace_coadd
rdn0
mcn1
mcmf
qe_maps
qe_power
cov



  



