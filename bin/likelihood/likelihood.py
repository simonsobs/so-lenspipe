import sys
import os
import numpy as np
from orphics import io
from orphics import stats

#Planck 2pt+lensing+BAO likelihood 

l_max = 3000
ls=np.arange(0,l_max)
#path to the likelihood modules
modules_path ='/global/homes/j/jia_qu/likelihood/cosmology/modules'
#N0 correction path
likelihood_path ='/global/homes/j/jia_qu/maps/likelihood_data/'

fiducial_params = {
         'ombh2': 0.02219218, 'omch2':0.1203058, 'H0': 67.02393, 'tau': 0.6574325e-01,
        'nnu': 3.046,
        'As':2.15086031154146e-9 ,'ns':0.9625356E+00,}

info_fiducial = {
    'params': fiducial_params,
    'likelihood': {'one': None},
    'theory': {'camb': {'stop_at_error': True,'extra_args':{'kmax':10,"nonlinear":True,"accurate_massive_neutrino_transfers": True}}},#'sampler': {'mcmc':{'learn_proposal': True}},
    'modules': modules_path}

from cobaya.model import get_model
model_fiducial = get_model(info_fiducial)
model_fiducial.add_requirements({'Cl':{'pp': 10000}})
model_fiducial.logposterior({})
Cls = model_fiducial.provider.get_Cl(ell_factor=False)

#same binning function used to bin the auto bandpowers
def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return bls,cents
    
#same bin edges to the one used for auto bandpowers
bin_edges,errors=np.load("../data/likelihood_error.npy",allow_pickle=True)


#fiducial cls used in the likelihood corrections
fcltt=Cls['tt'][0:l_max]
fclpp=Cls['pp'][0:l_max]
fclee=Cls['ee'][0:l_max]
fclte=Cls['te'][0:l_max]
fclbb=Cls['bb'][0:l_max]
thetaclkk=fclpp*(ls*(ls+1))**2*0.25

#load the auto bandpowers
tclkk_b=np.loadtxt("../data/binnedauto_test.txt")



#load the binned covariance matrix
cov=np.loadtxt("../data/cov_clkk.txt")
#load the correction terms generate from the script n1so.py
N0cltt=np.loadtxt(likelihood_path+"n0MVderTT_lmin600_lmax3000.txt").transpose()
N0clte=np.loadtxt(likelihood_path+"n0MVderTE_lmin600_lmax3000.txt").transpose()
N0clee=np.loadtxt(likelihood_path+"n0MVderEE_lmin600_lmax3000.txt").transpose()
N0clbb=np.loadtxt(likelihood_path+"n0MVderBB_lmin600_lmax3000.txt").transpose()
n0=np.loadtxt(likelihood_path+"n0mv_lmin600_lmax3000.txt")[:3000]
n0[0:2]=1e30




def my_like(_self=None):

    
    Cl_theo = _self.provider.get_Cl(ell_factor=False)['pp'][0:l_max]  
    Cl_tt= _self.provider.get_Cl(ell_factor=False)['tt'][0:l_max] 
    Cl_ee= _self.provider.get_Cl(ell_factor=False)['ee'][0:l_max] 
    Cl_te= _self.provider.get_Cl(ell_factor=False)['te'][0:l_max] 
    Cl_bb= _self.provider.get_Cl(ell_factor=False)['bb'][0:l_max] 

    Clkk_theo=(ls*(ls+1))**2*Cl_theo*0.25
    Clkk_binned,cents=bandedcls(Clkk_theo,bin_edges) 

    correction=2*(Clkk_theo/n0)*(np.dot(N0cltt,Cl_tt-fcltt)+np.dot(N0clee,Cl_ee-fclee)+np.dot(N0clbb,Cl_bb-fclbb)+np.dot(N0clte,Cl_te-fclte))  
    correction,cents=bandedcls(correction,bin_edges)
    Clkk_x=Clkk_binned+correction
    
    return -0.5*np.dot(np.transpose(tclkk_b-Clkk_x),np.dot(np.linalg.inv(cov),tclkk_b-Clkk_x))
    


info = {
    'params': {
        # Fixed
        'nnu': 3.046,
        # Sampled
        'mnu': {'prior': {'min': 0, 'max': 10}, 'latex': 'mnu'},
        'H0': {'prior': {'min': 40, 'max': 100}, 'ref':{'min': 64, 'max': 80}, 'proposal':0.5, 'latex': 'H_0'},
        'ombh2':{"prior": {'min': 0.005, 'max': 0.1},'latex': 'ombh2'},
        'omch2':{"prior": {'min': 0.005, 'max': 0.99}, 'latex': 'omch2'},
        'tau':{"prior": {"dist": "norm", "loc": 0.6574325e-01, "scale": 0.012},'ref':0.066, 'proposal':0.001, 'latex': 'tau'},
        'ns':{"prior": {"min":0.8,'max':1.2}, 'latex': 'ns'},
        'logAs': {'prior': {'min': 2, 'max': 4}, 'latex': 'logAs', 'drop': True},
        'As': {'value': "lambda logAs: 10**(-10)*np.exp(logAs)", 'latex': 'As'},
        's8omegamp25':{"derived": 'lambda sigma8, omegam: sigma8*omegam**0.25','latex': '\sigma_8 \Omega_\mathrm{m}^{0.25}'},
        'sigma8':{'latex': '\sigma_8'},
        'omegam':{'latex': '\Omega_\mathrm{m}'}
        # Derived
        },
    'packages_path': modules_path,
    'likelihood': {'my_cl_like': {"external": my_like, "requires": {'Cl': {'pp': 10000}}} ,'planck_2018_lowl.TT':{},'planck_2018_highl_plik.TT':{},'bao.sixdf_2011_bao':{},'bao.sdss_dr7_mgs':{},'bao.sdss_dr12_consensus_bao':{}},
    'theory': {'camb': {'stop_at_error': True,'extra_args':{'kmax':10,"nonlinear":True,"accurate_massive_neutrino_transfers": True}}},
    'sampler': {'mcmc':{'learn_proposal': True}},
    'modules': modules_path,
    'output': 'lensplanckBAOt/planck',
    'resume': True,}

from cobaya.model import get_model
model = get_model(info)
from cobaya.run import run
updated_info, products = run(info)