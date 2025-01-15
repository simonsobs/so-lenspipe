from pixell import enmap, curvedsky as cs, utils, enplot,lensing as plensing,curvedsky as cs
from orphics import maps,io,cosmology,stats,pixcov,mpi
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from soapack import interfaces
from scipy.optimize import curve_fit
import pixell.powspec
from orphics import maps
from soapack import interfaces as sints
import os
from falafel.utils import get_cmb_alm,test_cmb_alm
import pytempura
import healpy as hp
import numpy as np

def eshow(x,fname): 
    ''' Define a function to quickly plot the maps '''
    #plots = enplot.get_plots(x, downgrade = 4,color="gray")
    plots = enplot.get_plots(x, downgrade = 4)
    enplot.write(fname,plots)

def stamp_plot(imap,ra,dec,boxwidth=10):
    dec,ra = np.deg2rad(np.array((dec, ra)))
    width = np.deg2rad(boxwidth)
    box = np.array([[dec-width/2.,ra-width/2.],[dec+width/2.,ra+width/2.]])
    stamp = imap.submap(box)
    return stamp



def project_mask(mask,shape,wcs,fname=None):
    sim_mask = enmap.project(mask,shape,wcs,order=1)
    if fname!=None:
        enmap.write_fits(fname, sim_mask, extra={})
    return sim_mask

#fitting functions

def rolloff(ell, ell_off=None, alpha=-4, patience=2.):

    """
    Adapted from mapsims
    Get a transfer function T(ell) to roll off red noise at ell <
    ell_off.  ell should be an ndarray.  Above the cut-off,
    T(ell>=ell_off) = 1.  For T(ell<ell_off) will roll off smoothly,
    approaching T(ell) \propto ell^-alpha.
    """
    if ell_off is None or ell_off <= 0:
        return np.ones(ell.shape)
    L2 = ell_off
    L1 = L2 * patience ** (2./alpha)
    x = -np.log(ell / L2) / np.log(L1 / L2)
    beta = alpha * np.log(L1 / L2)
    output = x*0
    output[x<0]  = (-x*x)[x<0]
    output[x<-1] = (1 + 2*x)[x<-1]
    return np.exp(output * beta)   


def rad_fit(x, l0, a):
    return ((l0/x)**-a + 1)

def get_fitting(c_ell):
    #input a power spectrum and get fitted power spectrum
    bounds = ((0, -5), (9000, 1))
    cents=np.arange(len(c_ell))
    #find the floor of the white noise
    w=c_ell[cents > 5000].mean()
    #define the fitting section i.e only l>500
    mask = cents < 500
    ell_fit = cents[~mask]
    c_ell_fit = c_ell[~mask]
    params=np.zeros(3)
    params[:2],_=curve_fit(rad_fit, ell_fit, c_ell_fit/w, p0 = [3000, -4], bounds = bounds)
    params[2]=w 
    fit=rad_fit(cents, params[0], params[1]) * params[2]
    fit[~np.isfinite(fit)]=0
    return rolloff(cents,200,alpha=params[1],patience=1.2)*fit
    #return fitted power spectrum parameters given input powere spectrum

def fit_noise_curve(cl,lmax):
    from scipy.optimize import curve_fit

    fit=np.zeros(len(cl[:lmax]))
    x=np.arange(len(cl[:lmax]))
    boundary = x[cl.argmax()]
    f1 = lambda x, m, c: m*x + c
    f2 = lambda x, d, k: d*np.power(x, k)
    t1=x[x<boundary]
    t2=x[x>=400]
    y1=cl[t1]
    y2=cl[t2]
    popt_1 ,pcov_1 = curve_fit(f1, t1, y1, p0=((y1[-1] - y1[0]) / (t1[-1] - t1[0]), y1[0])) #linear fit
    popt_2 ,pcov_2 = curve_fit(f2,t2,y2, p0=[0, 0], bounds=(-np.inf, np.inf)) #power law fit
    
    fit[t1]=f1(t1, *popt_1)
    #fit[np.arange(boundary,lmax)]=f2(np.arange(boundary,lmax), *popt_2)
    fit=f2(np.arange(0,lmax), *popt_2)
    a=rolloff(np.arange(lmax),ell_off=1.3*boundary,patience=1.2)
    return a*fit
    
def rolling_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


    
# def coadd_map(map_list,ivar_list,a):
#     """return coadded map from splits, the map in maplist contains I,Q,U 
#     a=0,1,2 selects one of I Q U """
#     wcs=map_list[0].wcs
#     map_list=np.array(map_list)
#     ivar_list=np.array(ivar_list)
#     coadd_map= np.sum(map_list[:,a] * ivar_list, axis = 0)
#     coadd_map/=((np.sum(ivar_list, axis = 0)))
#     coadd_map[~np.isfinite(coadd_map)] = 0
#     coadd_map = enmap.ndmap(coadd_map,wcs)
#     return coadd_map    



def coadd_mapnewTT(map_list,ivar_list):
    """return coadded map from splits, the map in maplist contains I,Q,U 
    a=0,1,2 selects one of I Q U """
    wcs=map_list[0].wcs
    map_list=np.array(map_list)
    ivar_list=np.array(ivar_list)
    coadd_map= np.sum(map_list * ivar_list, axis = 0)
    #coadd_map/=((np.sum(ivar_list*mask, axis = 0)))
    coadd_map/=((np.sum(ivar_list, axis = 0)))
    #coadd_map/=((np.sum(ivar_list, axis = 0)))
    coadd_map[~np.isfinite(coadd_map)] = 0
    coadd_map = enmap.ndmap(coadd_map,wcs)
    return coadd_map    

def ivar_eff(split,ivar_list):
    """return effective invers variance map for split i and a list of inverse variance maps.
    Inputs
    splits:integer
    ivar_list:list
    Output
    ndmap with same shape and wcs as individual inverse variance maps.
    """
    ivar_list=np.array(ivar_list)
    h_c=np.sum(ivar_list,axis=0)
    w=h_c-ivar_list[split]
    weight=1/(1/ivar_list[split]-1/h_c)
    weight[~np.isfinite(weight)] = 0
    weight[weight<0] = 0
    return enmap.samewcs(weight,ivar_list[0])

    
def coadd_map(map_list,ivar_list):
    """return coadded map from splits, the map in maplist contains I,Q,U """
    wcs=map_list[0].wcs
    map_list=np.array(map_list)
    ivar_list=np.array(ivar_list)
    if map_list.shape!=ivar_list.shape:
        ivar_list = ivar_list[:, np.newaxis, :, :]
    coadd_map= np.sum(map_list[:,] * ivar_list, axis = 0)
    coadd_map/=((np.sum(ivar_list, axis = 0)))
    coadd_map[~np.isfinite(coadd_map)] = 0
    coadd_map = enmap.ndmap(coadd_map,wcs)
    return coadd_map    


def get_power(map_list,ivar_list, a, b, mask,N=20):
    """
    Calculate the average coadded flattened power spectrum P_{ab} used to generate simulation for the splits.
    Inputs:
    map_list: list of source free splits
    ivar_list: list of the inverse variance maps splits
    a: 0,1,2 for I,Q,U respectively
    b:0,1,2 for I,Q,U, respectively
    N: window to smooth the power spectrum by in the rolling average.
    mask: apodizing mask

    Output:
    1D power spectrum accounted for w2 from 0 to 10000
    """
    pmap=enmap.pixsizemap(map_list[0].shape,map_list[0].wcs)

    cl_ab=[]
    n = len(map_list)
    #calculate the coadd maps
    if a!=b:
        coadd_a=coadd_map(map_list,ivar_list,a)
        coadd_b=coadd_map(map_list,ivar_list,b)
    else:
        coadd_a=coadd_map(map_list,ivar_list,a)

    for i in range(n):
        print(i)
        if a!=b:
            d_a=map_list[i][a]-coadd_a
            noise_a=d_a*np.sqrt(ivar_eff(i,ivar_list))*mask
            alm_a=cs.map2alm(noise_a,lmax=8000)
            d_b=map_list[i][b]-coadd_b
            noise_b=d_b*np.sqrt(ivar_eff(i,ivar_list))*mask
            alm_b=cs.map2alm(noise_b,lmax=8000)
            cls = hp.alm2cl(alm_a,alm_b)
            cl_ab.append(cls)
        else:
            d_a=map_list[i][a]-coadd_a
            noise_a=d_a*np.sqrt(ivar_eff(i,ivar_list))*mask
            print("generating noise alms")
            alm_a=cs.map2alm(noise_a,lmax=8000)
            alm_a=alm_a.astype(np.complex128) 
            cls = hp.alm2cl(alm_a)
            cl_ab.append(cls)
    cl_ab=np.array(cl_ab)
    sqrt_ivar=np.sqrt(ivar_eff(0,ivar_list))
    mask_ivar = sqrt_ivar*0 + 1
    mask_ivar[sqrt_ivar<=0] = 0
    mask=mask*mask_ivar
    mask[mask<=0]=0
    w2=np.sum((mask**2)*pmap) /np.pi / 4.
    power = 1/n/(n-1) * np.sum(cl_ab, axis=0)
    ls=np.arange(len(power))
    power[~np.isfinite(power)] = 0
    power=rolling_average(power, N)
    bins=np.arange(len(power))
    power=maps.interp(bins,power)(ls)
    return power / w2


def get_power_EB(map_list,ivar_list, a, b, mask,N=20):
    """
    Calculate the average coadded flattened power spectrum P_{ab} used to generate simulation for the splits.
    Inputs:
    map_list: list of source free splits
    ivar_list: list of the inverse variance maps splits
    a: 0,1,2 for I,Q,U respectively
    b:0,1,2 for I,Q,U, respectively
    N: window to smooth the power spectrum by in the rolling average.
    mask: apodizing mask

    Output:
    1D power spectrum accounted for w2 from 0 to 10000
    """
    pmap=enmap.pixsizemap(map_list[0].shape,map_list[0].wcs)

    cl_ab=[]
    n = len(map_list)
    #calculate the coadd maps
    if a!=b:
        coadd_a=coadd_map(map_list,ivar_list,a)
        coadd_b=coadd_map(map_list,ivar_list,b)
    else:
        coadd_a=coadd_map(map_list,ivar_list,a)

    for i in range(n):
        print(i)
        if a!=b:
            d_a=map_list[i][a]-coadd_a
            noise_a=d_a*np.sqrt(ivar_eff(i,ivar_list))*mask
            alm_a=cs.map2alm(noise_a,lmax=8000, spin=2 if a in [1, 2] else None)
            d_b=map_list[i][b]-coadd_b
            noise_b=d_b*np.sqrt(ivar_eff(i,ivar_list))*mask
            alm_b=cs.map2alm(noise_b,lmax=8000, spin=2 if b in [1, 2] else None)
            cls = cs.alm2cl(alm_a,alm_b)
            cl_ab.append(cls)
        else:
            d_a=map_list[i][a]-coadd_a
            noise_a=d_a*np.sqrt(ivar_eff(i,ivar_list))*mask
            print("generating noise alms with EB")
            alm_a=cs.map2alm(noise_a,lmax=8000, spin=2 if b in [1, 2] else None)
            alm_a=alm_a.astype(np.complex128) 
            cls = cs.alm2cl(alm_a)
            cl_ab.append(cls)
    cl_ab=np.array(cl_ab)
    sqrt_ivar=np.sqrt(ivar_eff(0,ivar_list))
    mask_ivar = sqrt_ivar*0 + 1
    mask_ivar[sqrt_ivar<=0] = 0
    mask=mask*mask_ivar
    mask[mask<=0]=0
    w2=np.sum((mask**2)*pmap) /np.pi / 4.
    power = 1/n/(n-1) * np.sum(cl_ab, axis=0)
    ls=np.arange(len(power))
    power[~np.isfinite(power)] = 0
    power=rolling_average(power, N)
    bins=np.arange(len(power))
    power=maps.interp(bins,power)(ls)
    return power / w2


def get_w(n,maps,mask):
    """compute w(n) due to the presence of mask"""
    pmap=enmap.pixsizemap(maps.shape,maps.wcs)
    wn=np.sum((mask**n)*pmap) /np.pi / 4.
    return wn



def get_datanoise(map_list,ivar_list, mask,beam,beam_deconvolve=True,lmax=6000):
    """
    Calculate the noise power of a coadded map given a list of maps and list of ivars.
    Inputs:
    map_list: list of source free splits
    ivar_list: list of the inverse variance maps splits
    mask: apodizing mask
    Output:
    1D power spectrum accounted for w2 in T,E,B 
    """
    
    cl_ab=[]
    n_splits = len(map_list)
    coadd=coadd_map(map_list,ivar_list)
    for i in range(n_splits):
        noise_a=map_list[i]-coadd
        noise_a=np.nan_to_num(noise_a)
        #do pure EB here to get pure E and B weights
        Ealm,Balm=pureEB(noise_a[1],noise_a[2],mask,returnMask=0,lmax=lmax,isHealpix=False)
        alm_T=cs.map2alm(noise_a[0],lmax=lmax)
        alm_a=np.array([alm_T,Ealm,Balm])
        alm_a=alm_a.astype(np.complex128)
        cls = cs.alm2cl(alm_a)
        cl_ab.append(cls)
    cl_ab=np.array(cl_ab)
    w2=w_n(mask,2)
    cl_sum = np.sum(cl_ab, axis=0)
    power = 1/n_splits/(n_splits-1) * cl_sum
    ls=np.arange(len(power[0]))
    power[~np.isfinite(power)] = 0
    power/=w2
    return power



def get_datanoise_fullresTT(map_list,ivar_list, a, b, mask,beam,N=20,beam_deconvolve=True,lmax=6000):
    """
    Calculate the noise power of a coadded map given a list of maps and list of ivars. Used for high resolution TT
    Inputs:
    map_list: list of source free splits
    ivar_list: list of the inverse variance maps splits
    a: 0,1,2 for I,Q,U respectively
    b:0,1,2 for I,Q,U, respectively
    N: window to smooth the power spectrum by in the rolling average.
    mask: apodizing mask

    Output:
    1D power spectrum accounted for w2 from 0 to 10000
    """
    
    pmap=enmap.pixsizemap(map_list[0].shape,map_list[0].wcs)


    cl_ab=[]
    n = len(map_list)
    print('shape of maplength')
    print(n)
    #calculate the coadd maps
  
    coadd_a=coadd_mapnewTT(map_list,ivar_list)
    #enmap.write_map('/home/r/rbond/jiaqu/scratch/DR6/maps/sims/ksz_4pt/coadd_a.fits',coadd_a)
    print(coadd_a)
    print(coadd_a.shape)

    for i in range(n):
   
        noise_a=map_list[i]-coadd_a

        alm_a=cs.map2alm(noise_a,spin=0,lmax=lmax)
        #cls=cs.alm2cl(alm_a)
        #np.savetxt('/home/r/rbond/jiaqu/scratch/DR6/maps/sims/ksz_4pt/cls.txt',cls)
        alm_a=alm_a.astype(np.complex128)
        if beam_deconvolve:
            alm_a = cs.almxfl(alm_a,lambda x: 1/beam(x)) 
        cls = hp.alm2cl(alm_a)
        cl_ab.append(cls)
    cl_ab=np.array(cl_ab)
    #sqrt_ivar=np.sqrt(ivar_eff(0,ivar_list))

    mask=mask
    mask[mask<=0]=0
    w2=np.sum((mask**2)*pmap) /np.pi / 4.
    print(w2)
    power = 1/n/(n-1) * np.sum(cl_ab, axis=0)
    ls=np.arange(len(power))
    power[~np.isfinite(power)] = 0
    power=rolling_average(power, N)
    bins=np.arange(len(power))
    power=maps.interp(bins,power)(ls)

    return power / w2

def generate_sim(ivar_list,cls,lmax):
    """
    Input: ivar_list: list of inverse variance maps
    cls: flattened 1D power spectrum Pab
    lmax:maximum multipole to generate the simulated maps
    seed: currently a number, need to fix this.
    Returns:
    list of simulated maps.
    """
    shape=ivar_list[0].shape
    wcs=ivar_list[0].wcs
    pmap=enmap.pixsizemap(shape,wcs)
    k=len(ivar_list)
    sim_maplist=[]
    for i in range(len(ivar_list)):
        sim_map=np.sqrt(k)*cs.rand_map(shape,wcs,cls,lmax,spin=0)/(np.sqrt(ivar_eff(i,ivar_list)))
        sim_map[~np.isfinite(sim_map)] = 0
        sim_maplist.append(sim_map)
    return sim_maplist
    
def generate_coaddsim(ivar_list,cls,lmax,sim_splits=False):
    """
    Input: ivar_list: list of inverse variance maps
    cls: flattened 1D power spectrum Pab
    lmax:maximum multipole to generate the simulated maps  
    seed: currently a number, need to fix this.
    Returns:
    coadded noise map.
    """
    shape=ivar_list[0].shape
    wcs=ivar_list[0].wcs
    k=len(ivar_list)
    sim_maplist=[]
   
    ivareffl =[]
    for i in range(len(ivar_list)):
        ivareffl.append(ivar_eff(i, ivar_list))

    
    for i in range(len(ivar_list)):
        sim_map=np.sqrt(k)*cs.rand_map(shape,wcs,cls,lmax,spin=0)/(np.sqrt(ivareffl[i]))
        sim_map[~np.isfinite(sim_map)] = 0
        sim_maplist.append(sim_map)
    coadd_sim=coadd_map(sim_maplist,ivar_list)
    if not(sim_splits):
        return enmap.samewcs(coadd_sim,ivar_list[0])
    else:
        output_list=[] 
        for i in range(len(sim_maplist)):
            output_list.append(sim_maplist[i])
        output_list.append(coadd_sim)
        output_list=np.array(output_list)
        output_list=enmap.samewcs(output_list,coadd_sim)
        return output_list

def save_noise_alm(nmap,i,iset,path="/global/cscratch1/sd/jia_qu/maps/"):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullskynoise2018pa6_90_alm_set%s_%s.fits" % (sstr,istr)
    return enmap.write_fits(fname,nmap) 
    
def save_coadded_alm(nmap,i,iset,path="/global/cscratch1/sd/jia_qu/maps/"):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullsky_noise_coadded19_alm_set%s_%s.fits" % (sstr,istr)
    return enmap.write_fits(fname,nmap) 


def EB_tranform(imap):
    wcs=imap.wcs

    # Define taper using enmap.apod and then plot the taper
    apod_pix = 100 
    taper = enmap.apod(imap*0+1,apod_pix) 
    # Apply taper to the smap by multiplying the two maps together
    imap *= taper
    imap = enmap.ndmap(imap,wcs)
    f_smap = enmap.map2harm(imap, normalize = "phys")
    # Apply filter that will supress modes with l<150 and lx<5
    ly, lx = f_smap.lmap(); l = np.sqrt(ly**2+lx**2)
    f = ((1 + (l/150)**-4)**-1) * ((1 + (lx/5)**-4)**-1)
    maps  = enmap.ifft(f_smap*f, normalize = "phys").real
    return maps
    
def sharpWrapperSpinS(sht,alm,spin):
    #code by Will Coulton
    if spin==0:
        assert(0)
    #SP,SM=sht.alm2map((-(alm[0]+alm[1]*(-1)**np.abs(spin))/2.,(-(alm[0]-alm[1]*(-1)**np.abs(spin))/(2.j))),spin=np.abs(spin))
    SP,SM=sht.alm2map([-alm,alm*0.],spin=np.abs(spin))
    #return SP,SM
    if spin>0:
        return SP+1j*SM
    return SP-1j*SM

def pixellWrapperSpinS(alm2map,alm,mp12,spin):
    #code by Will Coulton
    if spin==0:
        assert(0)
    #SP,SM=sht.alm2map((-(alm[0]+alm[1]*(-1)**np.abs(spin))/2.,(-(alm[0]-alm[1]*(-1)**np.abs(spin))/(2.j))),spin=np.abs(spin))
    SP,SM=alm2map(np.array([-alm,alm*0.]),mp12,spin=np.abs(spin))
    #return SP,SM
    if spin>0:
        return SP+1j*SM
    return SP-1j*SM


def pureEB(Q,U,mask_0,returnMask=0,lmax=None,isHealpix=True):
    #code by Will Coulton

    if isHealpix:
        nside=int((len(mask_0)/12.)**.5)
        nside=int((len(mask_0)/12.)**.5)   
        if lmax is None:
            lmax=int(3*nside-1)
        map2alm = cs.map2alm_healpix
        alm2map = cs.alm2map_healpix
        template = np.zeros([2,12*nside**2])
    else:
        if lmax is None:
            lmax = np.min(np.pi/(Q.pixshape()))
        map2alm = cs.map2alm
        alm2map = cs.alm2map
        template =enmap.enmap(np.zeros(np.shape([Q,U])),wcs=Q.wcs)
    ainfo = cs.alm_info(int(lmax))
    ells = np.arange(0., 10000.)
    fnc1 = np.ones(len(ells))
    wAlm_0=map2alm(mask_0,ainfo=ainfo,spin=0)
    fnc1*=0
    fnc1=np.sqrt((ells+1)*(ells))
    fnc1[0:1]=0
    #wAlm_1=hp.sphtfunc.almxfl(wAlm_0.copy(),fnc1)
    wAlm_1=ainfo.lmul(wAlm_0.copy(),fnc1)
    fnc1*=0
    fnc1[2:]=np.sqrt((ells[2:]+2)*(ells[2:]+1)*(ells[2:])*(ells[2:]-1))
    wAlm_2=ainfo.lmul(wAlm_0.copy(),fnc1)
    mask_1=pixellWrapperSpinS(alm2map,wAlm_1,template,1)#.conjugate()
    mask_2=pixellWrapperSpinS(alm2map,wAlm_2,template,2)#.conjugate()
    mask_1[mask_0==0]=0
    mask_2[mask_0==0]=0
    if returnMask:
        return mask_1,mask_2
    wAlm_0=wAlm_1=wAlm_2=None
    template[...] = np.array([Q*mask_0,U*mask_0])
    E2,B2=map2alm(template,ainfo=ainfo,spin=2)
    template[...] = np.array([Q*mask_1.real+U*mask_1.imag,U*mask_1.real-Q*mask_1.imag])
    E1,B1=map2alm(template,ainfo=ainfo,spin=1)
    E0=map2alm(Q*mask_2.real+U*mask_2.imag,ainfo=ainfo,spin=0)
    B0=map2alm(U*mask_2.real-Q*mask_2.imag,ainfo=ainfo,spin=0)
    fnc1=np.zeros(len(ells))
    fnc1[2:]=2.0*np.sqrt(1./(ells[2:]+2)/(ells[2:]-1))
    fnc2=np.zeros(len(ells))
    fnc2[2:]=np.sqrt(1.0/((ells[2:]+2)*(ells[2:]+1)*ells[2:]*(ells[2:]-1)))
    pureE=E2+ainfo.lmul(E1,fnc1)-ainfo.lmul(E0,fnc2) #
    pureB=B2+ainfo.lmul(B1,fnc1)-ainfo.lmul(B0,fnc2) #
    fnc1[:]=1.0
    fnc1[:2]=0
    pureE=ainfo.lmul(pureE,fnc1)
    pureB=ainfo.lmul(pureB,fnc1)
    return pureE,pureB

def check_simulation(a,b,map_list,sim_list,ivar_list,mask):
    """
    Check whether simulated power spectrum P_{ab} is consistent with data. Returns list of (split_sim-coadd,split_data-coadd)
    weighted by the mask*effective_ivar.
    """
    shape=ivar_list[0].shape
    wcs=ivar_list[0].wcs
    pmap=enmap.pixsizemap(shape,wcs)
    sim_coadd=[]
    data_coadd=[]
    for i in range(len(sim_list)):
        dsim=sim_list[i]-coadd_map(sim_list,ivar_list)
        dsim=dsim*mask*ivar_eff(i,ivar_list)
        testalm=cs.map2alm(dsim,lmax=10000)
        testalm=testalm.astype(np.complex128) 
        testcl=hp.alm2cl(testalm)
        sim_coadd.append(testcl)
    if a==b:
        for i in range(len(map_list)):
            dataco=map_list[i][a]-coadd_map(map_list,ivar_list,a)
            dataco=dataco*mask*ivar_eff(i,ivar_list)
            testalm=cs.map2alm(dataco,lmax=10000)
            testalm=testalm.astype(np.complex128) 
            testcl=hp.alm2cl(testalm)
            data_coadd.append(testcl)
    else:
            for i in range(len(map_list)):
                data_a=map_list[i][a]-coadd_map(map_list,ivar_list,a)
                data_a=data_a*mask*ivar_eff(i,ivar_list)
                data_b=map_list[i][b]-coadd_map(map_list,ivar_list,b)
                data_b=data_b*mask*ivar_eff(i,ivar_list)
                testalm_a=cs.map2alm(data_a,lmax=10000)
                testalm_a=testalm_a.astype(np.complex128)
                testalm_b=cs.map2alm(data_b,lmax=10000)
                testalm_b=testalm_b.astype(np.complex128)
                testcl=hp.alm2cl(testalm_a,testalm_b)
                data_coadd.append(testcl)
    sim_coadd=np.array(sim_coadd)
    data_coadd=np.array(data_coadd)
    return (sim_coadd,data_coadd)


def alm2map(alm,shape,wcs,ncomp=3):
    return cs.alm2map(alm,enmap.empty((ncomp,)+shape,wcs))

def deconvolve_maps(maps,mask,beam,lmax=6000):
    "deconvolve the beam of a map"
    shape=maps.shape
    wcs=maps.wcs
    alm_a=cs.map2alm(maps*mask,lmax=lmax)
    alm_a = cs.almxfl(alm_a,lambda x: 1/beam(x)) 
    reconvolved_map=cs.alm2map(alm_a,enmap.empty(shape,wcs))
    return reconvolved_map

def reconvolve_maps(maps,mask,beamdec,beamconv,lmax=6000):
    "deconvolve the beam of a map and return a map convolved with new beam"
    shape=maps.shape
    wcs=maps.wcs
    alm_a=cs.map2alm(maps*mask,lmax=lmax)
    alm_a = cs.almxfl(alm_a,lambda x: 1/beamdec(x)) 
    convolved_alm=cs.almxfl(alm_a,lambda x: beamconv(x)) 
    reconvolved_map=cs.alm2map(convolved_alm,enmap.empty(shape,wcs))
    return reconvolved_map


def convert_seeds(seed,nsims=2000,ndiv=4):
    # Convert the solenspipe convention to the Alex convention
    icov,cmb_set,i = seed
    assert icov==0, "Covariance from sims not yet supported."
    nstep = nsims//ndiv #changed this to access roght files
    if cmb_set==0 or cmb_set==1:
        s_i = i + cmb_set*nstep
        s_set = 0
        noise_seed = (icov,cmb_set,i)+(2,)
    elif cmb_set==2 or cmb_set==3:
        s_i = i + nstep*2
        s_set = cmb_set - 2
        noise_seed = (icov,cmb_set,i)+(2,)

    return s_i,s_set,noise_seed

def get_beamed_signal(s_i,s_set,beam,shape,wcs,unlensed=False,fixed_amp=False):
    print(s_i,s_set)
    s_i,s_set,_ = convert_seeds((0,s_set,s_i))
    print(f"set:{s_set}")
    print(f"s_i:{s_i}")
    #cmb_alm = test_cmb_alm(s_i,s_set,unlensed=unlensed,fixed_amp=fixed_amp).astype(np.complex128)
    cmb_alm=get_cmb_alm(s_i,s_set,unlensed=unlensed).astype(np.complex128)
    if beam is  not None:
        cmb_alm = cs.almxfl(cmb_alm,lambda x: beam(x)) 
    cmb_map = alm2map(cmb_alm,shape,wcs)
    return cmb_map

def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return cents,bls

def get_ACT_Noise(qid,i,iset,path="/global/cscratch1/sd/jia_qu/maps/2dsims/"):
    print("loading actnoise")
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    print("path")
    print(path)
    fname = path + f"noiseSim_{qid}alm_set%s_%s.fits" % (sstr,istr)
    print(fname)
    return enmap.read_map(fname) 

def get_newACT_Noise(qid,i,iset,split,path="/global/cscratch1/sd/jia_qu/maps/newsims/"):
    print("loading actnoise")
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + f"noiseSim_{qid}alm_set%s_%s_split{split}.fits" % (sstr,istr)
    print(fname)
    return enmap.read_map(fname) 

def kspace_coadd(map_alms,lbeams,noise,fkbeam=1):
    """map_alms is an array containing the coadded alms as arrays to be coadded. This is NOT beam deconvolved
       lbeams are the beam in harmonic space ordered the same way as the coadded alms in map_alms
       noise corresponds to the noise power of the coadded maps above. This is not beam deconvolved
       fkbeam is the common beam to be applied to the kspace coadd map """

    coalms=np.zeros(map_alms[0].shape)
    coalms=coalms.astype(complex)
    denom = np.sum(lbeams**2 / noise,axis=0)
    for i in range(len(noise)):
        weighted_alms=hp.almxfl(map_alms[i],lbeams[i]/noise[i])
        weighted_alms[~np.isfinite(weighted_alms)] = 0
        a=hp.almxfl(weighted_alms,1/(denom))
        a[~np.isfinite(a)] = 0
        coalms+=a
    return coalms


def smooth_cls(cl,points=300):
    """bin and interpolate a cl to smooth it"""
    bin_edges = np.linspace(2,len(cl),points).astype(int)
    cents,cls=bandedcls(cl,bin_edges)
    cls=maps.interp(cents,cls)(np.arange(len(cl)))
    return cls

def smooth_rolling_cls(cl,N=10):
    """bin and interpolate a cl to smooth it"""
    ells=np.arange(len(cl))
    a=rolling_average(cl, N)
    smooth=np.interp(ells,np.arange(len(a)),a)    
    return smooth

def apod(imap,width):
    # This apodization is for FFTs. We only need it in the dec-direction
    # since the AdvACT geometry should be periodic in the RA-direction.
    return enmap.apod(imap,[width,0]) 

def inpaint(omap,ivar,beam_fn,nsplits,qid,output_path,dataSet='DR5',null=False):
    #code by Will Coulton
    rmin = 15.0 * utils.arcmin
    width = 40. * utils.arcmin
    res = 2. * utils.arcmin
    N = int(width/res)

    theory = cosmology.default_theory()

    
    shape,wcs = omap.shape[-2:],omap.wcs

    print(shape,omap.shape,ivar.shape)
    ras,decs = sints.get_act_mr3f_union_sources(version='20210209_sncut_10_aggressive')

    zmap = omap.copy()
    gdicts = [{} for i in range(nsplits)]
    ind = 0

    """
    ras=np.delete(ras,202) 
    decs=np.delete(decs,202)
    ras=np.delete(ras,262) 
    decs=np.delete(decs,262)
    ras=np.delete(ras,375) 
    decs=np.delete(decs,375)
    ras=np.delete(ras,698) 
    decs=np.delete(decs,698)
    ras=np.delete(ras,833) 
    decs=np.delete(decs,833)  
    ras=np.delete(ras,880) 
    decs=np.delete(decs,880) 
    ras=np.delete(ras,995) 
    decs=np.delete(decs,995)   
    ras=np.delete(ras,999) 
    decs=np.delete(decs,999)  
    ras=np.delete(ras,1093) 
    decs=np.delete(decs,1093)
    ras=np.delete(ras,1094) 
    decs=np.delete(decs,1094)   
    """
    """
    ras=np.delete(ras,78) 
    decs=np.delete(decs,78) 
    ras=np.delete(ras,104) 
    decs=np.delete(decs,104) 
    """
   
    """
    ras=np.delete(ras,834) 
    decs=np.delete(decs,834) 
    ras=np.delete(ras,878) 
    decs=np.delete(decs,878) 
    ras=np.delete(ras,995) 
    decs=np.delete(decs,995) 
    ras=np.delete(ras,1093) 
    decs=np.delete(decs,1093) 
    ras=np.delete(ras,1095) 
    decs=np.delete(decs,1095)
    """
    ra=ras.copy()
    de=decs.copy()

    #loop through all the ras and dec and delete those which are singular
    ind=0
    fault=[]
    
    for i in range(len(ras)):
        print(i)
        for j in range(nsplits):
            print(f"split{j}")
            try:
                py,px = omap.sky2pix((decs[i]*utils.degree,ras[i]*utils.degree))
                pbox = [[int(py) - N//2,int(px) - N//2],[int(py) + N//2,int(px) + N//2]]
                thumb = enmap.extract_pixbox(omap[j], pbox)  
                modrmap = thumb.modrmap()
                thumb[:,modrmap<rmin] = 0
                enmap.insert(zmap[j],thumb)
                shape,wcs = thumb.shape,thumb.wcs
                thumb_ivar = enmap.extract_pixbox(ivar[j][0], pbox)
                pcov = pixcov.pcov_from_ivar(N,decs[i],ras[i],thumb_ivar,theory.lCl,beam_fn,iau=True,full_map=False)
            except:
                print("omiting singular matrix")
                fault.append(i)

    print(fault)
    ran = np.delete(ra, fault)
    den = np.delete(de, fault)



    inds = []
    k=0

    for ra,dec in zip(ran[k:],den[k:]):
        print(ind)
        for i in range(nsplits):
            py,px = omap.sky2pix((dec*utils.degree,ra*utils.degree))
            pbox = [[int(py) - N//2,int(px) - N//2],[int(py) + N//2,int(px) + N//2]]
            thumb = enmap.extract_pixbox(omap[i], pbox)  
            modrmap = thumb.modrmap()
            thumb[:,modrmap<rmin] = 0
            enmap.insert(zmap[i],thumb)
            shape,wcs = thumb.shape,thumb.wcs
            #modlmap = enmap.modlmap(shape,wcs)
            thumb_ivar = enmap.extract_pixbox(ivar[i][0], pbox)
            pcov = pixcov.pcov_from_ivar(N,dec,ra,thumb_ivar,theory.lCl,beam_fn,iau=True,full_map=False)
            gdicts[i][ind+k] = pixcov.make_geometry(shape,wcs,rmin,n=N,deproject=True,iau=True,res=res,pcov=pcov)
        inds.append(ind+k)
        ind = ind+1 
                       


    imap = omap.copy()

    if null:
        for i in range(nsplits):
            imap[i] = pixcov.inpaint(omap[i],np.asarray([den[k:],ran[k:]]),hole_radius_arcmin=rmin/utils.arcmin,npix_context=N,resolution_arcmin=res/utils.arcmin,
                        cmb2d_TEB=None,n2d_IQU=None,beam2d=None,deproject=True,iau=True,tot_pow2d=None,
                        geometry_tags=inds,geometry_dicts=gdicts[0],verbose=True)

    else:
        for i in range(nsplits):
            imap[i] = pixcov.inpaint(omap[i],np.asarray([den[:],ran[:]]),hole_radius_arcmin=rmin/utils.arcmin,npix_context=N,resolution_arcmin=res/utils.arcmin,
                        cmb2d_TEB=None,n2d_IQU=None,beam2d=None,deproject=True,iau=True,tot_pow2d=None,
                        geometry_tags=inds,geometry_dicts=gdicts[i],verbose=True)

    enmap.write_map(f'{output_path}/inpainted_{dataSet}_{qid}.fits',imap)

def get_mask(path):
    afname = path
    print(afname)
    mask = enmap.read_map(afname)  
    return mask

def w_n(mask,n):
    """wrapper for solenspipe's wfactor function"""
    pmap = enmap.pixsizemap(mask.shape,mask.wcs)
    return maps.wfactor(n,mask,sht=True,pmap=pmap)

def kspace_mask(imap, vk_mask=[-90,90], hk_mask=[-50,50], normalize="phys", deconvolve=False):

    """Filter the map in Fourier space removing modes in a horizontal and vertical band
    defined by hk_mask and vk_mask. This is a faster version that what is implemented in pspy
    We also include an option for removing the pixel window function. Stolen from Will C who stole it from PS group.
    
    Parameters
    ---------
    imap: ``so_map``
        the map to be filtered
    vk_mask: list with 2 elements
        format is fourier modes [-lx,+lx]
    hk_mask: list with 2 elements
        format is fourier modes [-ly,+ly]
    normalize: string
        optional normalisation of the Fourier transform
    inv_pixwin_lxly: 2d array
        the inverse of the pixel window function in fourier space
    """
    if vk_mask is None and hk_mask is None:
        imap=imap
        if deconvolve:
            pow=-1
            wy, wx = enmap.calc_window(imap.shape)
            ft = enmap.fft(imap, normalize=normalize)
            ft = ft* wy[:,None]**pow * wx[None,:]**pow
            
        imap[:,:] = np.real(enmap.ifft(ft, normalize=normalize))
        return imap
    print('performing FFT')
    lymap, lxmap = imap.lmap()
    ly, lx = lymap[:,0], lxmap[0,:]

   # filtered_map = map.copy()
    ft = enmap.fft(imap, normalize=normalize)
    
    if vk_mask is not None:
        id_vk = np.where((lx > vk_mask[0]) & (lx < vk_mask[1]))
    if hk_mask is not None:
        id_hk = np.where((ly > hk_mask[0]) & (ly < hk_mask[1]))

    ft[...,: , id_vk] = 0.
    ft[...,id_hk,:]   = 0.

    if deconvolve:
        pow=-1
        wy, wx = enmap.calc_window(imap.shape)
        ft = ft* wy[:,None]**pow * wx[None,:]**pow
        
    imap[:,:] = np.real(enmap.ifft(ft, normalize=normalize))
    return imap




def get_Dpower(X,U,mask,m=4):
    """Get split averaged data power

    Args:
        X (array of arrays): array containing m alms
        U (array of arrays): array containing m alms
        mask (arrays): analysis mask used
        m (int, optional): Number of splits used Defaults to 4.
    """
    cls = hp.alm2cl(X[0])*0

    for i in range(m):
        for j in range(m):
            if j!=i:
                cls+=hp.alm2cl(X[i],U[j])/(w_n(mask,2)*m*(m-1))
    #cls=cls/(w_n(mask,2)*m*(m-1))
    return cls

def get_Spower(X,U,mask):
    """Get signal only data power

    Args:
        X (array): array containing coadd alms
        U (array): array containing coadd alms
        mask (arrays): analysis mask used
    """
    cls = hp.alm2cl(X,U)/w_n(mask,2)
    return cls

def diagonal_RDN0cross(est1,X,U,coaddX,coaddU,filters,theory,theory_cross,mask,lmin,lmax,est2=None,cross=True,bh=False,nlpp=None,nlss=None,response=None,profile=None):
    """Generate beloved dumb N0s for both gradient and curl.

    Args:
        est1 (str): Polcomb used
        X (list): Array of splits of 'data' TEB unfiltered maps.
        U (list): Array of splits of 'data' TEB unfiltered maps.
        coaddX (_type_): Coadded signal sim used for the sim part
        coaddU (_type_): Coadded signal sim used for the sim part
        filters (array): List of TEB Cls used for the filter
        theory (_type_): _description_
        theory_cross (_type_): _description_
        mask (array): analysis mask
        lmin (int): minimum CMB multipole
        lmax (int): maximum CMB multipole
        est2 (_type_, optional): _description_. Defaults to None.
        cross (bool, optional): _description_. Defaults to True.
        bh (bool, optional): _description_. Defaults to False.
        nlpp (_type_, optional): _description_. Defaults to None.
        nlss (_type_, optional): _description_. Defaults to None.
        response (_type_, optional): _description_. Defaults to None.
        profile (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)
    QDO = [True,True,True,True,True,False]
    #consistency with tempura, put every lcl to theory_cross
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    if profile is None:
        profile=np.ones(Lmax+1)
    else:
        profile=profile[:Lmax+1]
        nlpp=nlpp[0][:Lmax+1]
        nlss=nlss[:Lmax+1]
        response=response[:Lmax+1]

    D_l=get_Dpower(X,U,mask,m=4)
    S_l=get_Spower(coaddX,coaddU,mask)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[0][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[0][:ls.size]])
    ocl=ffl
    ocl[np.where(ocl==0)] = 1e30
    if est2 is None:
        if est1=='TT':
            print("use TT")
            #get norm
            AgTT,AcTT=pytempura.norm_lens.qtt(Lmax, rlmin, rlmax, lcl[0,:],lcl[0,:],ocl[0,:])

            #dxd
            cl=ocl**2/(d_ocl)
            AgTT0,AcTT0=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],lcl[0,:],cl[0,:])
            AgTT0[np.where(AgTT0==0)] = 1e30
            AcTT0[np.where(AcTT0==0)] = 1e30

            #sxs
            cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
            AgTT1,AcTT1=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],lcl[0,:],cl[0,:])
            AgTT1[np.where(AgTT1==0)] = 1e30
            AcTT1[np.where(AcTT1==0)] = 1e30
            ng = AgTT**2*(1./AgTT0-1/AgTT1)
            nc = AcTT**2*(1./AcTT0-1/AcTT1)

            if bh:
                print("use TTBH")
                #second term get the source
                ocl= (ffl)/profile**2
                ocl[np.where(ocl==0)] = 1e30
                AsTT=pytempura.norm_src.qtt(Lmax, rlmin, rlmax,ocl[0,:])*profile**2
                ocl= ffl
                cl=ffl**2/(d_ocl*profile**2)
                AsTT0=pytempura.norm_src.qtt(Lmax, rlmin, rlmax,cl[0,:])*profile**2
                cl=ocl**2/((s_ocl-d_ocl)*profile**2)
                AsTT1=pytempura.norm_src.qtt(Lmax, rlmin, rlmax,cl[0,:])*profile**2
                n0TTs = AsTT**2*(1./AsTT0-1./AsTT1)*(AgTT*response)**2
                #get the cross term
                cl=ffl**2/(d_ocl*profile)
                AxTT0=pytempura.norm_lens.stt(lmax,rlmin,rlmax, lcl[0,:],cl[0,:])/profile
                #(data-sim) x (data-sim)
                cl=ffl**2/((s_ocl-d_ocl)*profile)
                AxTT1=pytempura.norm_lens.stt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:])/profile
                n0TTx = -1*AgTT*AsTT*(AxTT0-AxTT1)*2*nlpp*response
                prefactor=1/(1-nlpp*nlss*response**2)**2
                ng=prefactor*(ng+n0TTs+n0TTx)
                print("finished bh")

        elif est1=='TE':
            AgTE,AcTE=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:])
            #dxd
            cl=ocl**2/(d_ocl)
            AgTE0,AcTE0=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
            AgTE0[np.where(AgTE0==0)] = 1e30
            AcTE0[np.where(AcTE0==0)] = 1e30

            #sxs
            cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
            AgTE1,AcTE1=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
            AgTE1[np.where(AgTE1==0)] = 1e30
            AcTE1[np.where(AcTE1==0)] = 1e30

            ng = AgTE**2*(1./AgTE0-1/AgTE1)
            nc = AcTE**2*(1./AcTE0-1/AcTE1)

        elif est1=='EE':
            print("using EE")
            AgEE,AcEE=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:])
            #prepare the data total power spectrum
            #dxd
            cl=ocl**2/(d_ocl)
            AgEE0,AcEE0=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:] )
            AgEE0[np.where(AgEE0==0)] = 1e30
            AcEE0[np.where(AcEE0==0)] = 1e30

            #sxs
            cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
            AgEE1,AcEE1=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:])
            AgEE1[np.where(AgEE1==0)] = 1e30
            AcEE1[np.where(AcEE1==0)] = 1e30

            ng = AgEE**2*(1./AgEE0-1/AgEE1)
            nc = AcEE**2*(1./AcEE0-1/AcEE1)

        elif est1=='TB':
            AgTB,AcTB=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:])
            cl=ocl**2/(d_ocl)
            AgTB0,AcTB0=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:] )
            AgTB0[np.where(AgTB0==0)] = 1e30
            AcTB0[np.where(AcTB0==0)] = 1e30

            #sxs
            cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
            AgTB1,AcTB1=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:])
            AgTB1[np.where(AgTB1==0)] = 1e30
            AcTB1[np.where(AcTB1==0)] = 1e30

            ng = AgTB**2*(1./AgTB0-1/AgTB1)
            nc = AcTB**2*(1./AcTB0-1/AcTB1)

        elif est1=='EB':
            AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:])
            #dxd
            cl=ocl**2/(d_ocl)
            AgEB0,AcEB0=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
            AgEB0[np.where(AgEB0==0)] = 1e30
            AcEB0[np.where(AcEB0==0)] = 1e30

            #sxs
            cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
            AgEB1,AcEB1=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
            AgEB1[np.where(AgEB1==0)] = 1e30
            AcEB1[np.where(AcEB1==0)] = 1e30

            ng = AgEB**2*(1./AgEB0-1/AgEB1)
            nc = AcEB**2*(1./AcEB0-1/AcEB1)
        
        elif est1 =='MV':
            print("use mv")
            return diagonal_RDN0mv(X,U,coaddX,coaddU,filters,theory,theory_cross,mask,lmin,lmax,cross=cross,bh=bh,nlpp=nlpp,nlss=nlss,response=response,profile=profile)
        elif est1 == 'MVPOL':
            print("use mvpol")
            return diagonal_RDN0mvpol(X,U,coaddX,coaddU,filters,theory,theory_cross,mask,lmin,lmax,cross=True,bh=False,nlpp=None,nlss=None,response=None,profile=None)
    if est2 is not None:
        ("est 2 not none")
        if est1=='TT' and est2=='TE':

            AgTT,AcTT=pytempura.norm_lens.qtt(Lmax, rlmin, rlmax, lcl[0,:],ocl[0,:])
            AgTE,AcTE=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:])   
            #dxd
            cl=ocl**2/(d_ocl)
            AgTTTE0,AcTTTE0=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:], cl[0,:], ocl[1,:]*d_ocl[0,:]/ocl[0,:],d_ocl[3,:])
            #AgTTTE0[np.where(AgTTTE0==0)] = 1e30
            #AcTTTE0[np.where(AcTTTE0==0)] = 1e30

            #sxs
            cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
            AgTTTE1,AcTTTE1=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:],cl[0,:] ,(1-d_ocl[0,:]/ocl[0,:])*ocl[1,:] , s_ocl[3,:]-d_ocl[3,:])
            #AgTTTE1[np.where(AgTTTE1==0)] = 1e30
            #AcTTTE1[np.where(AcTTTE1==0)] = 1e30
            ng=AgTT*AgTE*(AgTTTE0+AgTTTE1)
            nc=AcTT*AcTE*(AcTTTE0+AcTTTE1)
        elif est1=='TT' and est2=='EE':
            AgTT,AcTT=pytempura.norm_lens.qtt(Lmax, rlmin, rlmax, lcl[0,:],ocl[0,:])
            AgEE,AcEE=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:])
            cl=ocl**2/(d_ocl)
            AgTTEE0,AcTTEE0=pytempura.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], d_ocl[3,:])
            cl=ocl**2/(s_ocl-d_ocl)
            AgTTEE1,AcTTEE1=pytempura.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], s_ocl[3,:]-d_ocl[3,:])
            ng=AgTT*AgEE*(AgTTEE0+AgTTEE1)
            nc=AcTT*AcEE*(AcTTEE0+AcTTEE1)
        elif (est1 == 'TB') and (est2 == 'EB'):
            AgTB,AcTB=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:])
            AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:])
            cl=ocl**2/(d_ocl)
            AgTBEB0,AcTBEB0=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], d_ocl[3,:])
            AgTBEB0[np.where(AgTBEB0==0)] = 1e30
            AcTBEB0[np.where(AcTBEB0==0)] = 1e30            
            cl=ocl**2/(s_ocl-d_ocl)
            AgTBEB1,AcTBEB1=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], s_ocl[3,:]-d_ocl[3,:])
            AgTBEB1[np.where(AgTBEB1==0)] = 1e30
            AcTBEB1[np.where(AcTBEB1==0)] = 1e30
            ng=AgTB*AgEB*(AgTBEB0+AgTBEB1)
            nc=AcTB*AcEB*(AgTBEB0+AcTBEB1)

    return ng*fac**2*0.25,nc*fac**2*0.25



def diagonal_RDN0_TE(X,U,coaddX,coaddU,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """Curvedsky dumb N0 for TT,EE,EB,TE,TB for cross estimator"""
    print('compute dumb N0')
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    ocl= fcl+noise
    ocl[np.where(ocl==0)] = 1e30
    AgTE,AcTE=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:])
    #prepare the data total power spectrum
    D_l=get_Dpower(X,U,m=4)
    S_l=get_Spower(coaddX,coaddU)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    ocl=ffl
    #dxd
    cl=ocl**2/(d_ocl)
    AgTE0,AcTE0=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
    AgTE0[np.where(AgTE0==0)] = 1e30
    AcTE0[np.where(AcTE0==0)] = 1e30

    #sxs
    cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
    AgTE1,AcTE1=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
    AgTE1[np.where(AgTE1==0)] = 1e30
    AcTE1[np.where(AcTE1==0)] = 1e30

    n0TEg = AgTE**2*(1./AgTE0-1/AgTE1)
    n0TEc = AcTE**2*(1./AcTE0-1/AcTE1)
    return n0TEg*fac**2*0.25,n0TEc*fac**2*0.25


def diagonal_RDN0_EE(X,U,coaddX,coaddU,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax,mask):
    filters=np.loadtxt("/home/r/rbond/jiaqu/DR6lensing/DR6lensing/4split_null/output/coaddTT4phoct9/stage_filter/filters.txt")

    """Curvedsky dumb N0 for TT,EE,EB,TE,TB for cross estimator"""
    print('compute dumb N0')
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    ocl= fcl+noise
    ocl[np.where(ocl==0)] = 1e30
    AgEE,AcEE=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:])
    #prepare the data total power spectrum
    D_l=get_Dpower(X,U,mask,m=4)
    S_l=get_Spower(coaddX,coaddU,mask)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    ocl=ffl
    #dxd
    cl=ocl**2/(d_ocl)
    AgEE0,AcEE0=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:] )
    AgEE0[np.where(AgEE0==0)] = 1e30
    AcEE0[np.where(AcEE0==0)] = 1e30

    #sxs
    cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
    AgEE1,AcEE1=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:])
    AgEE1[np.where(AgEE1==0)] = 1e30
    AcEE1[np.where(AcEE1==0)] = 1e30

    n0EEg = AgEE**2*(1./AgEE0-1/AgEE1)
    n0EEc = AcEE**2*(1./AcEE0-1/AcEE1)
    return n0EEg*fac**2*0.25,n0EEc*fac**2*0.25

def diagonal_RDN0_TB(X,U,coaddX,coaddU,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """Curvedsky dumb N0 for TT,EE,EB,TE,TB for cross estimator"""
    print('compute dumb N0')
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)

    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    ocl= fcl+noise
    ocl[np.where(ocl==0)] = 1e30
    AgTB,AcTB=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:])
    #prepare the data total power spectrum
    D_l=get_Dpower(X,U,m=4)
    S_l=get_Spower(coaddX,coaddU)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    ocl=ffl
    #dxd
    cl=ocl**2/(d_ocl)
    AgTB0,AcTB0=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:] )
    AgTB0[np.where(AgTB0==0)] = 1e30
    AcTB0[np.where(AcTB0==0)] = 1e30

    #sxs
    cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
    AgTB1,AcTB1=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:])
    AgTB1[np.where(AgTB1==0)] = 1e30
    AcTB1[np.where(AcTB1==0)] = 1e30

    n0TBg = AgTB**2*(1./AgTB0-1/AgTB1)
    n0TBc = AcTB**2*(1./AcTB0-1/AcTB1)
    return n0TBg*fac**2*0.25,n0TBc*fac**2*0.25


def diagonal_RDN0_EB(X,U,coaddX,coaddU,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """Curvedsky dumb N0 for TT,EE,EB,TE,TB for cross estimator"""
    print('compute dumb N0')
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)

    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    ocl= fcl+noise
    ocl[np.where(ocl==0)] = 1e30
    AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:])
    #prepare the data total power spectrum
    D_l=get_Dpower(X,U,m=4)
    S_l=get_Spower(coaddX,coaddU)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    ocl=ffl
    #dxd
    cl=ocl**2/(d_ocl)
    AgEB0,AcEB0=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
    AgEB0[np.where(AgEB0==0)] = 1e30
    AcEB0[np.where(AcEB0==0)] = 1e30

    #sxs
    cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
    AgEB1,AcEB1=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
    AgEB1[np.where(AgEB1==0)] = 1e30
    AcEB1[np.where(AcEB1==0)] = 1e30

    n0EBg = AgEB**2*(1./AgEB0-1/AgEB1)
    n0EBc = AcEB**2*(1./AcEB0-1/AcEB1)
    return n0EBg*fac**2*0.25,n0EBc*fac**2*0.25

def diagonal_RDN0_TTTE(X,U,coaddX,coaddU,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """Curvedsky dumb N0 for TT,EE,EB,TE,TB for cross estimator"""
    print('compute dumb N0')
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)

    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    ocl= fcl+noise
    ocl[np.where(ocl==0)] = 1e30
    AgTT,AcTT=pytempura.norm_lens.qtt(Lmax, rlmin, rlmax, lcl[0,:],ocl[0,:])
    AgTE,AcTE=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:])    #prepare the data total power spectrum
    D_l=get_Dpower(X,U,m=4)
    S_l=get_Spower(coaddX,coaddU)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    ocl=ffl
    #dxd
    cl=ocl**2/(d_ocl)
    AgTTTE0,AcTTTE0=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:], cl[0,:], ocl[1,:]*d_ocl[0,:]/ocl[0,:],d_ocl[3,:])
    #AgTTTE0[np.where(AgTTTE0==0)] = 1e30
    #AcTTTE0[np.where(AcTTTE0==0)] = 1e30

    #sxs
    cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
    AgTTTE1,AcTTTE1=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:],cl[0,:] ,(1-d_ocl[0,:]/ocl[0,:])*ocl[1,:] , s_ocl[3,:]-d_ocl[3,:])
    n0TTTEg=AgTT*AgTE*(AgTTTE0+AgTTTE1)
    n0TTTEc=AcTT*AcTE*(AcTTTE0+AcTTTE1)

    return n0TTTEg*fac**2*0.25,n0TTTEc*fac**2*0.25

def diagonal_RDN0_TBEB(X,U,coaddX,coaddU,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """Curvedsky dumb N0 for TT,EE,EB,TE,TB for cross estimator"""
    print('compute dumb N0')
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)

    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    ocl= fcl+noise
    ocl[np.where(ocl==0)] = 1e30
    AgTB,AcTB=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:])
    AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:])
    D_l=get_Dpower(X,U,m=4)
    S_l=get_Spower(coaddX,coaddU)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    ocl=ffl
    #dxd
    cl=ocl**2/(d_ocl)
    AgTBEB0,AcTBEB0=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], d_ocl[3,:])
    AgTBEB0[np.where(AgTBEB0==0)] = 1e30
    AcTBEB0[np.where(AcTBEB0==0)] = 1e30

    #sxs
    cl=ocl**2/(s_ocl-d_ocl) #the larger the difference, the smaller the actt1 and hence the larger 1/actt1
    AgTBEB1,AcTBEB1=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], s_ocl[3,:]-d_ocl[3,:])
    AgTBEB1[np.where(AgTBEB1==0)] = 1e30
    AcTBEB1[np.where(AcTBEB1==0)] = 1e30
    n0TBEBg=AgTB*AgEB*(AgTBEB0+AgTBEB1)
    n0TBEBc=AcTB*AcEB*(AgTBEB0+AcTBEB1)

    return n0TBEBg*fac**2*0.25,n0TBEBc*fac**2*0.25

def diagonal_RDN0mv(X,U,coaddX,coaddU,filters,theory,theory_cross,mask,lmin,lmax,cross=True,bh=False,nlpp=None,nlss=None,response=None,profile=None):
    """Curvedsky dumb N0 for MV"""
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)
    QDO = [True,True,True,True,True,False]
  
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    if profile is None:
        profile=np.ones(Lmax+1)
    else:
        profile=profile[:Lmax+1]
        nlpp=nlpp[:ls.size]
        nlss=nlss[:ls.size]
        response=response[:ls.size]

    #ocl= noise+fcl
    ocl=ffl
    ocl[np.where(ocl==0)] = 1e30
    AgTT,AcTT=pytempura.norm_lens.qtt(Lmax, rlmin, rlmax, lcl[0,:],lcl[0,:],ocl[0,:])
    AgTE,AcTE=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],ocl[0,:],ocl[1,:])
    AgTB,AcTB=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],ocl[0,:],ocl[2,:])
    AgEE,AcEE=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],ocl[1,:])
    AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],ocl[1,:],ocl[2,:])

    ocl=ffl

    #prepare the sim total power spectrum
    #prepare the data total power spectrum
    D_l=get_Dpower(X,U,mask,m=4)
    S_l=get_Spower(coaddX,coaddU,mask)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    #dataxdata

    cl=ocl**2/(d_ocl)
    AgTT0,AcTT0=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],lcl[0,:],cl[0,:] )
    AgTE0,AcTE0=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],cl[0,:],cl[1,:])
    AgTB0,AcTB0=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],cl[0,:],cl[2,:] )
    AgEE0,AcEE0=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:] )
    AgEB0,AcEB0=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:],cl[2,:] )

    AgTTTE0,AcTTTE0=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:], cl[0,:], ocl[1,:]*d_ocl[0,:]/ocl[0,:],d_ocl[3,:])
    AgTTEE0,AcTTEE0=pytempura.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], d_ocl[3,:])
    AgTEEE0,AcTEEE0=pytempura.norm_lens.qteee(lmax, rlmin, rlmax, lcl[1,:], lcl[3,:], ocl[0,:]*d_ocl[1,:]/ocl[1,:], cl[1,:], d_ocl[3,:])
    AgTBEB0,AcTBEB0=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], d_ocl[3,:])



    cl=ocl**2/(s_ocl-d_ocl)
    AgTT1,AcTT1=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],lcl[0,:],cl[0,:] )
    AgTE1,AcTE1=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],cl[0,:],cl[1,:])
    AgTB1,AcTB1=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],cl[0,:],cl[2,:])
    AgEE1,AcEE1=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:])
    AgEB1,AcEB1=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:],cl[2,:])
    AgTTTE1,AcTTTE1=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:],cl[0,:] ,(1-d_ocl[0,:]/ocl[0,:])*ocl[1,:] , s_ocl[3,:]-d_ocl[3,:])
    AgTTEE1,AcTTEE1=pytempura.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], s_ocl[3,:]-d_ocl[3,:])
    AgTEEE1,AcTEEE1=pytempura.norm_lens.qteee(lmax, rlmin, rlmax, lcl[1,:], lcl[3,:], (1-d_ocl[1,:]/ocl[1,:])*ocl[0,:],cl[1,:],s_ocl[3,:]-d_ocl[3,:])
    AgTBEB1,AcTBEB1=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], s_ocl[3,:]-d_ocl[3,:])

    nlist=[AgTT0,AgTT1,AgEE0,AgEE1,AgEB0,AgEB1,AgTE0,AgTE1,AgTTTE0,AgTTTE1,AgTTEE0,AgTTEE1,AgTEEE0,AgTEEE1,AgTBEB0,AgTBEB1]
    for i in range(len(nlist)):
        nlist[i][np.where(nlist[i]==0)] = 1e30
    n0TTg = AgTT**2*(1./AgTT0-1./AgTT1)
    n0TEg = AgTE**2*(1./AgTE0-1./AgTE1)
    n0TBg = AgTB**2*(1./AgTB0-1./AgTB1)  
    n0EEg = AgEE**2*(1./AgEE0-1./AgEE1)
    n0EBg = AgEB**2*(1./AgEB0-1./AgEB1)
    n0TTTE=AgTT*AgTE*(AgTTTE0+AgTTTE1)
    n0TTEE=AgTT*AgEE*(AgTTEE0+AgTTEE1)
    n0TEEE=AgTE*AgEE*(AgTEEE0+AgTEEE1)
    n0TBEB=AgTB*AgEB*(AgTBEB0+AgTBEB1)
    n0TTc = AcTT**2*(1./AcTT0-1./AcTT1)
    n0TEc = AcTE**2*(1./AcTE0-1./AcTE1)
    n0TBc = AcTB**2*(1./AcTB0-1./AcTB1)  
    n0EEc = AcEE**2*(1./AcEE0-1./AcEE1)
    n0EBc = AcEB**2*(1./AcEB0-1./AcEB1)
    n0TTTEc=AcTT*AcTE*(AcTTTE0+AcTTTE1)
    n0TTEEc=AcTT*AcEE*(AcTTEE0+AcTTEE1)
    n0TEEEc=AcTE*AcEE*(AcTEEE0+AcTEEE1)
    n0TBEBc=AcTB*AcEB*(AcTBEB0+AcTBEB1)

    if bh:
        #second term get the source
        ocl= (ffl)/profile**2
        ocl[np.where(ocl==0)] = 1e30
        AsTT=pytempura.norm_src.qtt(Lmax, rlmin, rlmax,ocl[0,:])*profile**2

        ocl= ffl
        cl=ffl**2/(d_ocl*profile**2)
        AsTT0=pytempura.norm_src.qtt(Lmax, rlmin, rlmax,cl[0,:])*profile**2

        cl=ocl**2/((s_ocl-d_ocl)*profile**2)
        AsTT1=pytempura.norm_src.qtt(Lmax, rlmin, rlmax,cl[0,:])*profile**2
        n0TTs = AsTT**2*(1./AsTT0-1./AsTT1)*(AgTT*response)**2


        #get the cross term
        cl=ffl**2/(d_ocl*profile)
        AxTT0=pytempura.norm_lens.stt(lmax,rlmin,rlmax, lcl[0,:],cl[0,:])/profile
        #need see and ste if possible
        #(data-sim) x (data-sim)
        cl=ffl**2/((s_ocl-d_ocl)*profile)
        AxTT1=pytempura.norm_lens.stt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:])/profile
        n0TTx = -1*AgTT*AsTT*(AxTT0-AxTT1)*2*nlpp*response
        prefactor=1/(1-nlpp*nlss*response**2)**2
        n0TTg=prefactor*(n0TTg+n0TTs+n0TTx)
        
        #n0TTEE=(n0TTEE)/(1-nlpp*nlss*response**2)
        n0TTTE=(n0TTTE-AgTT*AsTT*(AxTT0-AxTT1)*nlpp*response)/(1-nlpp*nlss*response**2) #normalization should be AgTE
        #n0TTEE=(n0TTEE-AgTT*AsTT*(AxTT0-AxTT1)*nlpp*response)/(1-nlpp*nlss*response**2)
        #n0TTEE=(n0TTEE)/(1-nlpp*nlss*response**2)
        #n0TTTE=(n0TTTE)/(1-nlpp*nlss*response**2)



    dumbn0g=[n0TTg,n0TEg,n0TBg,n0EBg,n0EEg,n0TTTE,n0TTEE,n0TEEE,n0TBEB]
    dumbn0c=[n0TTc,n0TEc,n0TBc,n0EBc,n0EEc,n0TTTEc,n0TTEEc,n0TEEEc,n0TBEBc]

    AgTTf,AcTTf=pytempura.norm_lens.qtt(Lmax, rlmin, rlmax, lcl[0,:],lcl[0,:],ocl[0,:])
    AgTEf,AcTEf=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],ocl[0,:],ocl[1,:])
    AgTBf,AcTBf=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],lcl[3,:],ocl[0,:],ocl[2,:])
    AgEEf,AcEEf=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],ocl[1,:])
    AgEBf,AcEBf=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],ocl[1,:],ocl[2,:])

    weights_NUMg=[1/AgTTf**2,1/AgTEf**2,1/AgTBf**2,1/AgEBf**2,1/AgEEf**2,2/(AgTTf*AgTEf),2/(AgTTf*AgEEf),2/(AgTEf*AgEEf),2/(AgTBf*AgEBf)]
    weights_NUMc=[1/AcTTf**2,1/AcTEf**2,1/AcTBf**2,1/AcEBf**2,1/AcEEf**2,2/(AcTTf*AcTEf),2/(AcTTf*AcEEf),2/(AcTEf*AcEEf),2/(AcTBf*AcEBf)]

    weights_deng=[1/AgTTf**2,1/AgTEf**2,1/AgTBf**2,1/AgEBf**2,1/AgEEf**2,2/(AgTTf*AgTEf),2/(AgTTf*AgTBf),2/(AgTTf*AgEBf),2/(AgTTf*AgEEf),
    2/(AgTE*AgTBf),2/(AgTEf*AgEBf),2/(AgTEf*AgEEf),2/(AgTBf*AgEBf),2/(AgTBf*AgEEf),2/(AgEBf*AgEEf)]
    
    weights_denc=[1/AcTTf**2,1/AcTEf**2,1/AcTBf**2,1/AcEBf**2,1/AcEEf**2,2/(AcTTf*AcTEf),2/(AcTTf*AcTBf),2/(AcTTf*AcEBf),2/(AcTTf*AcEEf),
    2/(AcTEf*AcTBf),2/(AcTEf*AcEBf),2/(AcTEf*AcEEf),2/(AcTBf*AcEBf),2/(AcTBf*AcEEf),2/(AcEBf*AcEEf)]

    mvdumbN0g=np.zeros(len(n0TTg))
    mvdumbN0c=np.zeros(len(n0TTg))
    sumcg=np.zeros(len(n0TTg))  
    sumcc=np.zeros(len(n0TTg)) 
    for i in range(len(weights_denc)):
        sumcg+=weights_deng[i]
        sumcc+=weights_denc[i]
    for i in range(len(weights_NUMc)):
        mvdumbN0g+=np.nan_to_num(weights_NUMg[i])*np.nan_to_num(dumbn0g[i])
        mvdumbN0c+=np.nan_to_num(weights_NUMc[i])*np.nan_to_num(dumbn0c[i])
    mvdumbN0g=mvdumbN0g/sumcg
    mvdumbN0c=mvdumbN0c/sumcc
    
    return mvdumbN0g*fac**2*0.25,mvdumbN0c*fac**2*0.25


def diagonal_RDN0mvpol(X,U,coaddX,coaddU,filters,theory,theory_cross,mask,lmin,lmax,cross=True,bh=False,nlpp=None,nlss=None,response=None,profile=None):
    """Curvedsky dumb N0 for MVPOL currently no T"""
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    fac=ls*(ls+1)
    QDO = [True,True,True,True,True,False]

    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])
    ffl=np.array([filters[0][:Lmax+1],filters[1][:Lmax+1],filters[2][:Lmax+1],filters[3][:Lmax+1]])
    if profile is None:
        profile=np.ones(Lmax+1)


    #ocl= noise+fcl
    ocl=ffl
    ocl[np.where(ocl==0)] = 1e30

    AgEE,AcEE=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:], lcl[1,:],ocl[1,:])
    AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:], lcl[1,:],ocl[1,:],ocl[2,:])

    ocl=ffl

    #prepare the sim total power spectrum
    #prepare the data total power spectrum
    D_l=get_Dpower(X,U,mask,m=4)
    S_l=get_Spower(coaddX,coaddU,mask)
    d_ocl=np.array([D_l[0][:ls.size],D_l[1][:ls.size],D_l[2][:ls.size],D_l[3][:ls.size]])
    s_ocl=np.array([S_l[0][:ls.size],S_l[1][:ls.size],S_l[2][:ls.size],S_l[3][:ls.size]])
    #dataxdata

    cl=ocl**2/(d_ocl)
    AgEE0,AcEE0=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:] )
    AgEB0,AcEB0=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:],cl[2,:] )

    cl=ocl**2/(s_ocl-d_ocl)
    AgEE1,AcEE1=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:])
    AgEB1,AcEB1=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],cl[1,:],cl[2,:])


    nlist=[AgEE0,AgEE1,AgEB0,AgEB1]
    for i in range(len(nlist)):
        nlist[i][np.where(nlist[i]==0)] = 1e30

    n0EEg = AgEE**2*(1./AgEE0-1./AgEE1)
    n0EBg = AgEB**2*(1./AgEB0-1./AgEB1)
    n0EEc = AcEE**2*(1./AcEE0-1./AcEE1)
    n0EBc = AcEB**2*(1./AcEB0-1./AcEB1)




    dumbn0g=[n0EBg,n0EEg]
    dumbn0c=[n0EBc,n0EEc]

    AgEEf,AcEEf=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],ocl[1,:])
    AgEBf,AcEBf=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],lcl[1,:],ocl[1,:],ocl[2,:])

    weights_NUMg=[1/AgEBf**2,1/AgEEf**2]
    weights_NUMc=[1/AcEBf**2,1/AcEEf**2]

    weights_deng=[1/AgEBf**2,1/AgEEf**2]
    
    weights_denc=[1/AcEBf**2,1/AcEEf**2]

    mvdumbN0g=np.zeros(len(n0EBg))
    mvdumbN0c=np.zeros(len(n0EBg))
    sumcg=np.zeros(len(n0EBg))  
    sumcc=np.zeros(len(n0EBg)) 
    for i in range(len(weights_denc)):
        sumcg+=weights_deng[i]
        sumcc+=weights_denc[i]
    for i in range(len(weights_NUMc)):
        mvdumbN0g+=np.nan_to_num(weights_NUMg[i])*np.nan_to_num(dumbn0g[i])
        mvdumbN0c+=np.nan_to_num(weights_NUMc[i])*np.nan_to_num(dumbn0c[i])
    mvdumbN0g=mvdumbN0g/sumcg
    mvdumbN0c=mvdumbN0c/sumcc
    
    return mvdumbN0g*fac**2*0.25,mvdumbN0c*fac**2*0.25




def create_binary_mask(ra_min, ra_max, dec_min, dec_max, nside=512):
    """
    Create a binary mask with the specified RA and Dec ranges.

    Parameters:
    ra_min (float): Minimum right ascension in degrees.
    ra_max (float): Maximum right ascension in degrees.
    dec_min (float): Minimum declination in degrees.
    dec_max (float): Maximum declination in degrees.
    nside (int): Resolution of the map (NSIDE parameter).

    Returns:
    np.ndarray: Binary mask.
    """
    # Create an empty mask
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)

    # Convert RA and Dec to theta and phi
    theta_min, theta_max = np.deg2rad(90 - dec_max), np.deg2rad(90 - dec_min)
    phi_min, phi_max = np.deg2rad(ra_min), np.deg2rad(ra_max)

    # Loop over all pixels and set the mask to True within the specified RA and Dec ranges
    for pix in range(len(mask)):
        theta, phi = hp.pix2ang(nside, pix)
        if theta_min <= theta <= theta_max and phi_min <= phi <= phi_max:
            mask[pix] = True

    return mask