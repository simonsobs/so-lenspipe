from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap,lensing as enlensing
import numpy as np
import os,sys
from pixell.fft import fft,ifft

"""
This module implements classes
for simulations of inputs to lensing reconstruction.

"""



class CMBMap(object):

    def __init__(self,):
        pass

    def get_prepared_kmap(self,X,seed):
        icov,iset,i = seed

"""

CMBMap -> SingleFrequencyCMBMap, KSpaceCoadd, TILeCoadd

"""
        
        
class FlatLensingSims(object):
    def __init__(self,shape,wcs,theory,beam_arcmin,noise_uk_arcmin,noise_e_uk_arcmin=None,noise_b_uk_arcmin=None,pol=False,fixed_lens_kappa=None):
        # assumes theory in uK^2
        from orphics import cosmology
        if len(shape)<3 and pol: shape = (3,)+shape
        if noise_e_uk_arcmin is None: noise_e_uk_arcmin = np.sqrt(2.)*noise_uk_arcmin
        if noise_b_uk_arcmin is None: noise_b_uk_arcmin = noise_e_uk_arcmin
        self.modlmap = enmap.modlmap(shape,wcs)
        self.shape = shape
        self.wcs = wcs
        Ny,Nx = shape[-2:]
        lmax = self.modlmap.max()
        ells = np.arange(0,lmax,1)
        ps_cmb = cosmology.power_from_theory(ells,theory,lensed=False,pol=pol)
        self.mgen = MapGen(shape,wcs,ps_cmb)
        if fixed_lens_kappa is not None:
            self._fixed = True
            self.kappa = fixed_lens_kappa
            self.alpha = alpha_from_kappa(self.kappa)
        else:
            self._fixed = False
            ps_kk = theory.gCl('kk',self.modlmap).reshape((1,1,Ny,Nx))
            self.kgen = MapGen(shape[-2:],wcs,ps_kk)
            self.posmap = enmap.posmap(shape[-2:],wcs)
            self.ps_kk = ps_kk
        self.kbeam = gauss_beam(self.modlmap,beam_arcmin)
        ncomp = 3 if pol else 1
        ps_noise = np.zeros((ncomp,ncomp,Ny,Nx))
        ps_noise[0,0] = (noise_uk_arcmin*np.pi/180./60.)**2.
        if pol:
            ps_noise[1,1] = (noise_e_uk_arcmin*np.pi/180./60.)**2.
            ps_noise[2,2] = (noise_b_uk_arcmin*np.pi/180./60.)**2.
        self.ngen = MapGen(shape,wcs,ps_noise)
        self.ps_noise = ps_noise

    def get_unlensed(self,seed=None):
        return self.mgen.get_map(seed=seed)
    def get_kappa(self,seed=None):
        return self.kgen.get_map(seed=seed)
    def get_sim(self,seed_cmb=None,seed_kappa=None,seed_noise=None,lens_order=5,return_intermediate=False,skip_lensing=False,cfrac=None):
        unlensed = self.get_unlensed(seed_cmb)
        if skip_lensing:
            lensed = unlensed
            kappa = enmap.samewcs(lensed.copy()[0]*0,lensed)
        else:
            if not(self._fixed):
                kappa = self.get_kappa(seed_kappa)
                self.kappa = kappa
                self.alpha = alpha_from_kappa(kappa,posmap=self.posmap)
            else:
                kappa = None
                assert seed_kappa is None
            lensed = enlensing.displace_map(unlensed, self.alpha, order=lens_order)
        beamed = filter_map(lensed,self.kbeam)
        noise_map = self.ngen.get_map(seed=seed_noise)
        
        observed = beamed + noise_map
        if return_intermediate:
            return [ get_central(x,cfrac) for x in [unlensed,kappa,lensed,beamed,noise_map,observed] ]
        else:
            return get_central(observed,cfrac)
    
class MapGen(object):
        """
        Once you know the shape and wcs of an ndmap and the input power spectra, you can 
        pre-calculate some things to speed up random map generation.
        """
        
        def __init__(self,shape,wcs,cov=None,covsqrt=None,pixel_units=False,smooth="auto",ndown=None,order=1):
                self.shape = shape
                self.wcs = wcs
                assert cov.ndim>=3 , "Power spectra have to be of shape (ncomp,ncomp,lmax) or (ncomp,ncomp,Ny,Nx)."
                if covsqrt is not None:
                    self.covsqrt = covsqrt
                else:
                    if cov.ndim==4:
                            if not(pixel_units): cov = cov * np.prod(shape[-2:])/enmap.area(shape,wcs )
                            if ndown:
                                self.covsqrt = downsample_power(shape,wcs,cov,ndown,order,exp=0.5)
                            else:
                                self.covsqrt = enmap.multi_pow(cov, 0.5)
                    else:
                            self.covsqrt = enmap.spec2flat(shape, wcs, cov, 0.5, mode="constant",smooth=smooth)


        def get_map(self,seed=None,scalar=False,iau=True,real=False):
                if seed is not None: np.random.seed(seed)
                rand = enmap.fft(enmap.rand_gauss(self.shape, self.wcs)) if real else enmap.rand_gauss_harm(self.shape, self.wcs)
                data = enmap.map_mul(self.covsqrt, rand)
                kmap = enmap.ndmap(data, self.wcs)
                if scalar:
                        return enmap.ifft(kmap).real
                else:
                        return enmap.harm2map(kmap,iau=iau)

def gauss_beam(ell,fwhm):
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))

def filter_map(imap,kfilter):
    return enmap.enmap(np.real(ifft(fft(imap,axes=[-2,-1])*kfilter,axes=[-2,-1],normalize=True)) ,imap.wcs)

def get_central(img,fracy,fracx=None):
    if fracy is None and fracx is None: return img
    fracx = fracy if fracx is None else fracx
    Ny,Nx = img.shape[-2:]
    cropy = int(fracy*Ny)
    cropx = int(fracx*Nx)
    if cropy%2==0 and Ny%2==1:
        cropy -= 1
    else:
        if cropy%2==1 and Ny%2==0: cropy -= 1
    if cropx%2==0 and Nx%2==1:
        cropx -= 1
    else:
        if cropx%2==1 and Nx%2==0: cropx -= 1
    return crop_center(img,cropy,cropx)


def alpha_from_kappa(kappa=None,posmap=None,phi=None):
    if phi is None:
        phi,_ = kappa_to_phi(kappa,kappa.modlmap(),return_fphi=True)
        shape,wcs = phi.shape,phi.wcs
    else:
        shape,wcs = phi.shape,phi.wcs
    grad_phi = enmap.grad(phi)
    if posmap is None: posmap = enmap.posmap(shape,wcs)
    pos = posmap + grad_phi
    alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)
    return alpha_pix

def kappa_to_phi(kappa,modlmap,return_fphi=False):
    fphi = enmap.samewcs(kappa_to_fphi(kappa,modlmap),kappa)
    phi =  enmap.samewcs(ifft(fphi,axes=[-2,-1],normalize=True).real, kappa) 
    if return_fphi:
        return phi, fphi
    else:
        return phi

def kappa_to_fphi(kappa,modlmap):
    return fkappa_to_fphi(fft(kappa,axes=[-2,-1]),modlmap)

def fkappa_to_fphi(fkappa,modlmap):
    kmap = np.nan_to_num(2.*fkappa/modlmap/(modlmap+1.))
    kmap[modlmap<2.] = 0.
    return kmap

