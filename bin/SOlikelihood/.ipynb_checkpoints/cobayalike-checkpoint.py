#path to season 2 ACTPol likelihood
import sys
import os
import numpy as np
from orphics import io
from orphics import stats
#N0 Corrections only

#create the fiducial clkk
config = io.config_from_yaml("config.yml")

l_max = 3000
ls=np.arange(0,l_max)
f_sky=0.4
modules_path ='/global/homes/j/jia_qu/likelihood/cosmology/modules'
fiducial_params = {
         'ombh2': 0.02219218, 'omch2':0.1203058, 'H0': 67.02393, 'tau': 0.6574325E-01,
        'nnu': 3.046, 'num_massive_neutrinos': 1,
        'As':2.15086031154146e-9 ,'ns':0.9625356E+00,}

info_fiducial = {
    'params': fiducial_params,
    'likelihood': {'one': None},
    'theory': {'camb': {"extra_args": {'kmax':0.9}}},
    'modules': modules_path}

from cobaya.model import get_model
model_fiducial = get_model(info_fiducial)


model_fiducial.likelihood.theory.needs(Cl={'pp': 10000,'tt': 10000,'te': 10000,'ee': 10000,'bb':10000})
model_fiducial.logposterior({})
Cls = model_fiducial.likelihood.theory.get_Cl(ell_factor=False)





def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return bls,cents
    

#sonoise=np.loadtxt("/global/cscratch1/sd/jia_qu/so_covariance/noise_tt.txt")[:3000]
bin_edges = np.linspace(20,3000,10)
sonoise=np.loadtxt("/global/cscratch1/sd/jia_qu/cmblensfastdata/theorynoise1.txt")[0:l_max]
nls,cents=bandedcls(sonoise,bin_edges)

#load the observed clkk and the nls
fcltt=Cls['tt'][0:l_max]
fclpp=Cls['pp'][0:l_max]
fclee=Cls['ee'][0:l_max]
fclte=Cls['te'][0:l_max]
fclbb=Cls['bb'][0:l_max]

clttf_b,cents=bandedcls(fcltt,bin_edges) 

#tclkk_b=np.loadtxt("/global/cscratch1/sd/jia_qu/cmblensfastdata/theoryclkk.txt")[0:l_max]
tclkk_b=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/clkk.txt")[0:3000]
tclkk_b,cents=bandedcls(tclkk_b,bin_edges)
#correction use clkk obtained from clkk
thetaclkk=fclpp*(ls*(ls+1))**2*0.25
#alternatively load the covariance matrix
np.savetxt("/global/homes/j/jia_qu/so-lenspipe/data/cobclkk.txt",thetaclkk)
cov=np.ones(len(tclkk_b))
for i in range(len(tclkk_b)):
    cov[i]=(1/(cents[i]*311.11*f_sky))*(tclkk_b[i]+nls[i])**2

d=np.diag(cov)

#correction terms
"""
N1cltt=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/extn1TTdcltt1.txt").transpose()
N1clpp=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/n1mvdclkk1.txt").transpose()
"""
N0cltt=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/n0mvdcltt1.txt").transpose()
N0clte=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/n0mvdclte1.txt").transpose()
N0clee=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/n0mvdclee1.txt").transpose()
N0clbb=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/n0mvdclbb1.txt").transpose()
N1clpp=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/n1mvdclkk1.txt").transpose()

n0=np.loadtxt("/global/homes/j/jia_qu/so-lenspipe/data/n0mv.txt")


def my_like(
       
        _theory={'Cl': {'pp': 10000,'tt': 10000,'te': 10000,'ee': 10000,'bb':10000}},_self=None

         ):

    
    Cl_theo = _theory.get_Cl(ell_factor=False)['pp'][0:l_max]  # muK-2
    Cl_tt= _theory.get_Cl(ell_factor=False)['tt'][0:l_max]  # muK-2
    Cl_ee= _theory.get_Cl(ell_factor=False)['ee'][0:l_max]
    Cl_te= _theory.get_Cl(ell_factor=False)['te'][0:l_max]
    Cl_bb= _theory.get_Cl(ell_factor=False)['bb'][0:l_max]
    Clkk_theo=(ls*(ls+1))**2*Cl_theo*0.25
    Clkk_binned,cents=bandedcls(Clkk_theo,bin_edges) 
    Cltt_binned,cents=bandedcls(Cl_tt,bin_edges)
    correction=2*(thetaclkk/n0)*(np.dot(N0cltt,Cl_tt-fcltt)+np.dot(N0clee,Cl_ee-fclee)+np.dot(N0clbb,Cl_bb-fclbb)+np.dot(N0clte,Cl_te-fclte))+np.dot(N1clpp,Clkk_theo-thetaclkk)
    #correction=np.dot(N1cltt,Cl_tt-fcltt[:2992])+np.dot(N1clpp,Clkk_theo-thetaclkk)-2*thetaclkk*(np.dot(N0cltt,Cl_tt-fcltt[:2992]))
    #np.savetxt("/global/homes/j/jia_qu/so-lenspipe/data/saven1.txt",np.dot(N1cltt,thetaclkk))
    
    correction,cents=bandedcls(correction,bin_edges)
    
    Clkk_x=Clkk_binned+correction
    
    #print("%diff corrected vs uncorrected")
    #print(100*((Clkk_x/Clkk_binned)-1))
    #print(-0.5*np.dot(np.transpose(tclkk_b-Clkk_binned),np.dot(np.linalg.inv(d),tclkk_b-Clkk_binned)))        
    #return -0.5*np.dot(np.transpose(tclkk-Clkk_theo),np.dot(np.linalg.inv(d),tclkk-Clkk_theo))
    return -0.5*np.dot(np.transpose(tclkk_b-Clkk_x),np.dot(np.linalg.inv(d),tclkk_b-Clkk_x))
    


info = {
    'params': {
        # Fixed
        'nnu': 3.046, 'num_massive_neutrinos': 1,
        # Sampled
        'mnu': {'prior': {'min': 0, 'max': 2}, 'ref':0.06, 'proposal':0.001, 'latex': 'mnu'},
        'H0': {'prior': {'min': 40, 'max': 100}, 'ref':{'min': 64, 'max': 80}, 'proposal':0.5, 'latex': 'H_0'},
        'ombh2':{"prior": {"dist": "norm", "loc": 0.0222, "scale": 0.0009},'ref':{'min': 0.015, 'max': 0.03},'proposal':0.0001, 'latex': 'ombh2'},
        'omch2':{"prior": {'min': 0.005, 'max': 0.99},'ref':{'min': 0.1, 'max': 0.2},'proposal':0.001, 'latex': 'omch2'},
        'tau':{"prior": {"dist": "norm", "loc": 0.066, "scale": 0.012},'ref':0.066, 'proposal':0.001, 'latex': 'tau'},
        'ns':{"prior": {"dist": "norm", "loc": 0.96, "scale": 0.02},'ref':0.96,'proposal':0.01, 'latex': 'ns'},
        'As': {'prior': {'min': 7.39e-10, 'max': 5.46e-9}, 'ref':2.3e-9, 'proposal':0.1e-10, 'latex': 'As'},




        
        
        # Derived
        },
    'likelihood': {'my_cl_like': my_like ,'planck_2018_lowl.TT':{},'planck_2018_highl_plik.TT':{}},
    #'theory': {'camb': {'stop_at_error': True,'extra_args':{'kmax':0.9,'nonlinear':True}}}, 
    'theory': {'camb': {'stop_at_error': True,'extra_args':{'kmax':0.9}}},
    #'theory': {'camb': {'stop_at_error': True}},
    #'sampler': {'mcmc': {'covmat':'auto'}},  # or polychord...
    'sampler': {'mcmc':None},
    'modules': modules_path,
    'output': 'chains/my_imaginary_cmb',
    'resume': True,}

from cobaya.model import get_model
model = get_model(info)
from cobaya.run import run
updated_info, products = run(info)