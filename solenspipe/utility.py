import mapsims
from pixell import enmap, curvedsky as cs, utils, enplot,lensing as plensing,curvedsky as cs
from orphics import maps,io,cosmology,stats,pixcov
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from soapack import interfaces
from scipy.optimize import curve_fit
import pixell.powspec
from orphics import maps
from soapack import interfaces as sints
from pixell import sharp
import os
from falafel.utils import get_cmb_alm
from solenspipe import SOLensInterface,cmblensplus_norm,convert_seeds,get_kappa_alm,wfactor
from orphics import maps,io,cosmology,stats,mpi


def eshow(x,fname): 
    ''' Define a function to quickly plot the maps '''
    #plots = enplot.get_plots(x, downgrade = 4,color="gray")
    plots = enplot.get_plots(x, downgrade = 4)
    enplot.write(fname,plots)


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

def coadd_map(map_list,ivar_list):
    """return coadded map from splits, input list of maps, where each map only contains one of I,Q or U"""
    map_list=np.array(map_list)
    ivar_list=np.array(ivar_list)
    coadd_map= np.sum(map_list * ivar_list, axis = 0)
    coadd_map/=np.sum(ivar_list, axis = 0)
    coadd_map[~np.isfinite(coadd_map)] = 0

    return enmap.samewcs(coadd_map,map_list[0])
    
def coadd_mapnew(map_list,ivar_list,a):
    """return coadded map from splits, the map in maplist contains I,Q,U 
    a=0,1,2 selects one of I Q U """
    wcs=map_list[0].wcs
    map_list=np.array(map_list)
    ivar_list=np.array(ivar_list)
    coadd_map= np.sum(map_list[:,a] * ivar_list, axis = 0)
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
        coadd_a=coadd_mapnew(map_list,ivar_list,a)
        coadd_b=coadd_mapnew(map_list,ivar_list,b)
    else:
        coadd_a=coadd_mapnew(map_list,ivar_list,a)

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

def get_w(n,maps,mask):
    """compute w(n) due to the presence of mask"""
    pmap=enmap.pixsizemap(maps.shape,maps.wcs)
    wn=np.sum((mask**n)*pmap) /np.pi / 4.
    return wn


    
def get_datanoise(map_list,ivar_list, a, b, mask,beam,N=20,beam_deconvolve=True):
    """
    Calculate the noise power of a coadded map given a list of maps and list of ivars.
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
        coadd_a=coadd_mapnew(map_list,ivar_list,a)
        coadd_b=coadd_mapnew(map_list,ivar_list,b)
    else:
        coadd_a=coadd_mapnew(map_list,ivar_list,a)

    for i in range(n):
        print(i)
        if a!=b:
            d_a=map_list[i][a]-coadd_a
            noise_a=d_a*mask
            alm_a=cs.map2alm(noise_a,lmax=6000)
            d_b=map_list[i][b]-coadd_b
            noise_b=d_b*mask
            alm_b=cs.map2alm(noise_b,lmax=6000)
            cls = hp.alm2cl(alm_a,alm_b)
            cl_ab.append(cls)
        else:
            d_a=map_list[i][a]-coadd_a

            noise_a=d_a*mask
    
            print("generating alms")
            alm_a=cs.map2alm(noise_a,lmax=6000)
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
    from pixell import sharp

    if isHealpix:
        nside=int((len(mask_0)/12.)**.5)
        minfo=sharp.map_info_healpix(nside)
        nside=int((len(mask_0)/12.)**.5)   
        if lmax is None:
            lmax=int(3*nside-1)
        map2alm = cs.map2alm_healpix
        alm2map = cs.alm2map_healpix
        template = np.zeros([2,12*nside**2])
    else:
        minfo=cs.match_predefined_minfo(Q)
        if lmax is None:
            lmax = np.min(np.pi/(Q.pixshape()))
        map2alm = cs.map2alm
        alm2map = cs.alm2map
        template =enmap.enmap(np.zeros(np.shape([Q,U])),wcs=Q.wcs)
    ainfo = sharp.alm_info(int(lmax))
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
            dataco=map_list[i][a]-coadd_mapnew(map_list,ivar_list,a)
            dataco=dataco*mask*ivar_eff(i,ivar_list)
            testalm=cs.map2alm(dataco,lmax=10000)
            testalm=testalm.astype(np.complex128) 
            testcl=hp.alm2cl(testalm)
            data_coadd.append(testcl)
    else:
            for i in range(len(map_list)):
                data_a=map_list[i][a]-coadd_mapnew(map_list,ivar_list,a)
                data_a=data_a*mask*ivar_eff(i,ivar_list)
                data_b=map_list[i][b]-coadd_mapnew(map_list,ivar_list,b)
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

def reconvolve_maps(maps,mask,beamdec,beamconv):
    "deconvolve the beam of a map and return a map convolved with new beam"
    shape=maps.shape
    wcs=maps.wcs
    alm_a=cs.map2alm(maps*mask,lmax=8000)
    alm_a = cs.almxfl(alm_a,lambda x: 1/beamdec(x)) 
    convolved_alm=cs.almxfl(alm_a,lambda x: beamconv(x)) 
    reconvolved_map=cs.alm2map(convolved_alm,enmap.empty(shape,wcs))
    return reconvolved_map


def convert_seeds(seed,nsims=100,ndiv=4):
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

def get_beamed_signal(s_i,s_set,beam,shape,wcs):
    s_i,s_set,noise_seed = convert_seeds((0,s_set,s_i))
    cmb_alm = get_cmb_alm(s_i,s_set).astype(np.complex128)
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
    ras=np.delete(ras,78) 
    decs=np.delete(decs,78) 
    ras=np.delete(ras,104) 
    decs=np.delete(decs,104) 
    ras=np.delete(ras,375) 
    decs=np.delete(decs,375)

    ras=np.delete(ras,835) 
    decs=np.delete(decs,835) 
    ras=np.delete(ras,883) 
    decs=np.delete(decs,883) 
    ras=np.delete(ras,999) 
    decs=np.delete(decs,999) 
    ras=np.delete(ras,1099) 
    decs=np.delete(decs,1099) 
    ras=np.delete(ras,1101) 
    decs=np.delete(decs,1101) 
    """
    nmax=len(ras)
    inds = []
    for ra,dec in zip(ras[:nmax],decs[:nmax]):
        print(ind)
        for i in range(nsplits):
            py,px = omap.sky2pix((dec*utils.degree,ra*utils.degree))
            pbox = [[int(py) - N//2,int(px) - N//2],[int(py) + N//2,int(px) + N//2]]
            thumb = enmap.extract_pixbox(omap[i], pbox)  
            modrmap = thumb.modrmap()
            thumb[:,modrmap<rmin] = 0
            enmap.insert(zmap[i],thumb)
            shape,wcs = thumb.shape,thumb.wcs
            modlmap = enmap.modlmap(shape,wcs)
            thumb_ivar = enmap.extract_pixbox(ivar[i][0], pbox)
            pcov = pixcov.pcov_from_ivar(N,dec,ra,thumb_ivar,theory.lCl,beam_fn,iau=True,full_map=False)
            gdicts[i][ind] = pixcov.make_geometry(shape,wcs,rmin,n=N,deproject=True,iau=True,res=res,pcov=pcov)
        inds.append(ind)
        ind = ind + 1

    imap = omap.copy()

    if null:
        for i in range(nsplits):
            imap[i] = pixcov.inpaint(omap[i],np.asarray([decs[:nmax],ras[:nmax]]),hole_radius_arcmin=rmin/utils.arcmin,npix_context=N,resolution_arcmin=res/utils.arcmin,
                        cmb2d_TEB=None,n2d_IQU=None,beam2d=None,deproject=True,iau=True,tot_pow2d=None,
                        geometry_tags=inds,geometry_dicts=gdicts[0],verbose=True)

    else:
        for i in range(nsplits):
            imap[i] = pixcov.inpaint(omap[i],np.asarray([decs[:nmax],ras[:nmax]]),hole_radius_arcmin=rmin/utils.arcmin,npix_context=N,resolution_arcmin=res/utils.arcmin,
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
    return wfactor(n,mask,sht=True,pmap=pmap)

def kspace_mask(imap):
    """
    returns a real space map in which the kspace stripes of |lx|<90 and |ly|<50 are masked
    """
    k_mask=maps.mask_kspace(imap.shape,imap.wcs,lxcut=90,lycut=50)
    return maps.filter_map(imap,k_mask)