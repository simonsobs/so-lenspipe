from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
from orphics import maps,io,cosmology,mpi # msyriac/orphics ; pip install -e . --user
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot

import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,Channel,SOStandalonePrecomputedCMB
import mapsims
from falafel import qe
import os
import glob
import traceback

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']
#mask path
mpath="/global/cscratch1/sd/msyriac/data/depot/solenspipe"

def get_mask(lmax=3000,car_deg=2,hp_deg=4,healpix=False,no_mask=False):
    if healpix:
        mask = np.ones((hp.nside2npix(2048),)) if no_mask else initialize_mask(2048,hp_deg)
    else:
        if no_mask:
            # CAR resolution is decided based on lmax
            res = np.deg2rad(2.0 *(3000/lmax) /60.)
            shape,wcs = enmap.fullsky_geometry(res=res)
            mask = enmap.ones(shape,wcs)
        else:
            
            afname = f'{mpath}/car_mask_lmax_{lmax}_apodized_{car_deg:.1f}_deg.fits'
            mask = enmap.read_map(afname)[0]
    return mask

def initialize_args(args):
    # Lensing reconstruction ell range
    lmin = args.lmin
    lmax = args.lmax
    use_cached_norm = args.use_cached_norm
    quicklens = not(args.flat_sky_norm)
    disable_noise = args.disable_noise
    debug_cmb = args.debug
    
    wnoise = args.wnoise
    beam = args.beam
    atmosphere = not(args.no_atmosphere)
    polcomb = args.polcomb

    # Number of sims
    nsims = args.nsims
    curl=args.curl
    sindex = args.sindex
    comm,rank,my_tasks = mpi.distribute(nsims)

    isostr = "isotropic_" if args.isotropic else "classical"

    config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
    opath = config['data_path']

    mask = get_mask(healpix=args.healpix,lmax=lmax,no_mask=args.no_mask,car_deg=2,hp_deg=4)

    # Initialize the lens simulation interface
    solint = SOLensInterface(mask=mask,data_mode=None,scanning_strategy="isotropic" if args.isotropic else "classical",fsky=0.4 if args.isotropic else None,white_noise=wnoise,beam_fwhm=beam,disable_noise=disable_noise,atmosphere=atmosphere,zero_sim=args.zero_sim)
    if rank==0: solint.plot(mask,f'{opath}/{args.label}_{args.polcomb}_{isostr}mask')
    
    # Choose the frequency channel
    channel = mapsims.SOChannel("LA", 145)

    # norm dict
    Als = {}
    ils,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv = solint.initialize_norm(channel,lmin,lmax,recalculate=not(use_cached_norm),quicklens=quicklens,curl=curl,label=args.label)    
    Als['mv'] = al_mv
    Als['mvpol'] = al_mv_pol
    al_mv = Als[polcomb]
    Nl = al_mv * ils*(ils+1.) / 4.
    return solint,ils,Als,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr

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

def get_cmb_alm(i,iset,path=config['signal_path']):
    sstr = str(iset).zfill(2)
    istr = str(i).zfill(5)
    fname = path + "fullskyLensedUnabberatedCMB_alm_set%s_%s.fits" % (sstr,istr)
    return hp.read_alm(fname,hdu=(1,2,3))


def get_kappa_alm(i,path=config['signal_path']):
    istr = str(i).zfill(5)
    fname = path + "fullskyPhi_alm_%s.fits" % istr
    return plensing.phi_to_kappa(hp.read_alm(fname))

def wfactor(n,mask,sht=True,pmap=None,equal_area=False):
    """
    Approximate correction to an n-point function for the loss of power
    due to the application of a mask.

    For an n-point function using SHTs, this is the ratio of 
    area weighted by the nth power of the mask to the full sky area 4 pi.
    This simplifies to mean(mask**n) for equal area pixelizations like
    healpix. For SHTs on CAR, it is sum(mask**n * pixel_area_map) / 4pi.
    When using FFTs, it is the area weighted by the nth power normalized
    to the area of the map. This also simplifies to mean(mask**n)
    for equal area pixels. For CAR, it is sum(mask**n * pixel_area_map) 
    / sum(pixel_area_map).

    If not, it does an expensive calculation of the map of pixel areas. If this has
    been pre-calculated, it can be provided as the pmap argument.
    
    """
    assert mask.ndim==1 or mask.ndim==2
    if pmap is None: 
        if equal_area:
            npix = mask.size
            pmap = 4*np.pi / npix if sht else enmap.area(mask.shape,mask.wcs) / npix
        else:
            pmap = enmap.pixsizemap(mask.shape,mask.wcs)
    return np.sum((mask**n)*pmap) /np.pi / 4. if sht else np.sum((mask**n)*pmap) / np.sum(pmap)

class SOLensInterface(object):
    def __init__(self,mask,data_mode=None,scanning_strategy="isotropic",fsky=0.4,white_noise=None,beam_fwhm=None,disable_noise=False,atmosphere=True,rolloff_ell=50,zero_sim=False):

        self.rolloff_ell = rolloff_ell
        self.mask = mask
        self._debug = False
        self.atmosphere = atmosphere
        self.zero_map = zero_sim
        if mask.ndim==1:
            self.nside = hp.npix2nside(mask.size)
            self.healpix = True
            self.mlmax = 2*self.nside
            self.npix = hp.nside2npix(self.nside)
            self.pmap = 4*np.pi / self.npix
            self.px = qe.pixelization(nside=self.nside)
        else:
            self.shape,self.wcs = mask.shape[-2:],mask.wcs
            self.nside = None
            self.healpix = False
            #self.beam = None
            res_arcmin = np.rad2deg(enmap.pixshape(self.shape, self.wcs)[0])*60.
            self.mlmax = int(4000 * (2.0/res_arcmin))
            self.pmap = enmap.pixsizemap(self.shape,self.wcs)
            self.px = qe.pixelization(shape=self.shape,wcs=self.wcs)
        self.disable_noise = disable_noise
        if (white_noise is None) and not(disable_noise):
            self.wnoise = None
            self.beam = None
            self.nsim = noise.SONoiseSimulator(telescopes=['LA'],nside=self.nside,
                                               shape=self.shape if not(self.healpix) else None,
                                               wcs=self.wcs if not(self.healpix) else None, 
                                               apply_beam_correction=False,scanning_strategy=scanning_strategy,
                                               fsky={'LA':fsky} if fsky is not None else None,rolloff_ell=rolloff_ell)    
        else:
            self.wnoise = white_noise
            self.beam = beam_fwhm
        thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
        theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
        ells,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])
        class T:
            def __init__(self):
                self.lCl = lambda p,x: maps.interp(ells,gt)(x)
        self.theory_cross = T()
        self.cltt = lambda x: theory.lCl('TT',x) 
        self.clee = lambda x: theory.lCl('EE',x) 
        self.clbb = lambda x: theory.lCl('BB',x) 
        self.cache = {}
        self.theory = theory
        self.set_data_map(data_mode)

    def wfactor(self,n):
        return wfactor(n,self.mask,sht=True,pmap=self.pmap)

    def set_data_map(self,data_mode=None):
        if data_mode is None:
            print('WARNING: No data mode specified. Defaulting to simulation iset=0,i=0 at 150GHz.')
            data_mode = 'sim'



    def alm2map(self,alm,ncomp=3):
        if self.healpix:
            hmap = hp.alm2map(alm.astype(np.complex128),nside=self.nside,verbose=False)
            return hmap[None] if ncomp==1 else hmap
        else:
            return curvedsky.alm2map(alm,enmap.empty((ncomp,)+self.shape,self.wcs))
        
    def map2alm(self,imap):
        if self.healpix:
            return hp.map2alm(imap,lmax=self.mlmax,iter=0)
        else:
            return curvedsky.map2alm(imap,lmax=self.mlmax)


    def get_kappa_alm(self,i):
        kalms = get_kappa_alm(i,path=config['signal_path'])
        return self.map2alm(self.alm2map(kalms,ncomp=1)[0]*self.mask)

    def rand_map(self,power,seed):
        if self.healpix:
            np.random.seed(seed)
            pmap = (4.*np.pi / self.npix)*((180.*60./np.pi)**2.)
            return (self.wnoise/np.sqrt(pmap))*np.random.standard_normal((3,self.npix,))
            #return hp.synfast(power,self.nside)
        else:
            return maps.white_noise((3,)+self.shape,self.wcs,self.wnoise,seed=seed)
            #return enmap.rand_map((3,)+self.shape,self.wcs,power)

    def get_noise_map(self,noise_seed,channel):
        if not(self.disable_noise):
            ls,nells,nells_P = self.get_noise_power(channel,beam_deconv=False)
            nseed = noise_seed+(int(channel.band),)
            
            if self.wnoise is None:
                noise_map = self.nsim.simulate(channel,seed=nseed,atmosphere=self.atmosphere,mask_value=np.nan)
                noise_map[np.isnan(noise_map)] = 0
            else:
                npower = np.zeros((3,3,ls.size))
                npower[0,0] = nells
                npower[1,1] = nells_P
                npower[2,2] = nells_P
                noise_map = self.rand_map(npower,nseed)
        else:
            noise_map = 0

        return noise_map


    def get_beamed_signal(self,channel,s_i,s_set):
        if self.beam is None:
            self.beam = self.nsim.get_beam_fwhm(channel)
        cmb_alm = get_cmb_alm(s_i,s_set).astype(np.complex128)
        cmb_alm = curvedsky.almxfl(cmb_alm,lambda x: maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else cmb_alm
        cmb_map = self.alm2map(cmb_alm)
        return cmb_map

    def plot(self,imap,name,**kwargs):
        if self.healpix:
            io.mollview(imap,f'{name}.png',**kwargs)
        else:
            io.hplot(imap,name,**kwargs)

    def prepare_map(self,channel,seed,lmin,lmax):
        """
        Generates a beam-deconvolved simulation.
        Filters it and caches it.
        """

        if not(self.zero_map):
            print("prepare map")
            # Convert the solenspipe convention to the Alex convention
            s_i,s_set,noise_seed = convert_seeds(seed)
            # Get a beamed CMB signal. Any foreground simulations should be beamed and added to this.
            cmb_map = self.get_beamed_signal(channel,s_i,s_set)
            # Get a noise map from the SO sim generator
            noise_map = self.get_noise_map(noise_seed,channel)
            noise_map=enmap.samewcs(noise_map,cmb_map)
            noise_oalms = self.map2alm(noise_map)


            # Sum and mask
            imap = (cmb_map + noise_map)
            imap = imap * self.mask

            if self._debug:
                for i in range(3): self.plot(imap[i],f'imap_{i}')
                for i in range(3): self.plot(noise_map[i],f'nmap_{i}',range=300)

            # Map -> alms, and deconvolve the beam
            oalms = self.map2alm(imap)


            oalms = curvedsky.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms
            #hp.fitsfunc.write_alm("/global/cscratch1/sd/jia_qu/maps/testTT.fits",oalms[0])
            oalms[~np.isfinite(oalms)] = 0

            # Isotropic filtering
            # load the noise powers
            ls,nells,nells_P = self.get_noise_power(channel,beam_deconv=True)
            nells_T = maps.interp(ls,nells) if not(self.disable_noise) else lambda x: x*0
            nells_P = maps.interp(ls,nells_P) if not(self.disable_noise) else lambda x: x*0
            # Make 1/(C+N) filter functions
            filt_t = lambda x: 1./(self.cltt(x) + nells_T(x))
            filt_e = lambda x: 1./(self.clee(x) + nells_P(x))
            filt_b = lambda x: 1./(self.clbb(x) + nells_P(x))

   
            # And apply the filters to the alms
            almt = qe.filter_alms(oalms[0].copy(),filt_t,lmin=lmin,lmax=lmax)
            alme = qe.filter_alms(oalms[1].copy(),filt_e,lmin=lmin,lmax=lmax)
            almb = qe.filter_alms(oalms[2].copy(),filt_b,lmin=lmin,lmax=lmax)


        else:
            nalms = hp.Alm.getsize(self.mlmax)
            almt = np.zeros((nalms,),dtype=np.complex128)
            alme = np.zeros((nalms,),dtype=np.complex128)
            almb = np.zeros((nalms,),dtype=np.complex128)
            oalms = []
            for i in range(3):
                oalms.append( np.zeros((nalms,),dtype=np.complex128) )
            
        # Cache the alms
        self.cache = {}
        self.cache[seed] = (almt,alme,almb,oalms[0],oalms[1],oalms[2])
        icov,s_set,i=seed
        
    def get_sim_power(self,channel,seed,lmin,lmax):
        """
        Generates the sim cmb+noise cls.
        """

        if not(self.zero_map):
            # Convert the solenspipe convention to the Alex convention
            s_i,s_set,noise_seed = convert_seeds(seed)
            cmb_map = self.get_beamed_signal(channel,s_i,s_set)
            noise_map = self.get_noise_map(noise_seed,channel)

            imap = (cmb_map + noise_map)
            imap = imap * self.mask

            oalms = self.map2alm(imap)
            oalms = curvedsky.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms
            oalms[~np.isfinite(oalms)] = 0
            clttsim=hp.alm2cl(oalms[0],oalms[0])/self.wfactor(2)
            cleesim=hp.alm2cl(oalms[1],oalms[1])/self.wfactor(2)
            clbbsim=hp.alm2cl(oalms[2],oalms[2])/self.wfactor(2)
            cltesim=hp.alm2cl(oalms[0],oalms[1])/self.wfactor(2)
        return clttsim,cleesim,clbbsim,cltesim
            

            
    def prepare_shearT_map(self,channel,seed,lmin,lmax):
        """For the shear estimator, obtain beam deconvolved T_F map filtered by inverse variance filter squared"""

        if not(self.zero_map):
            print("prepare map")
            # Convert the solenspipe convention to the Alex convention
            s_i,s_set,noise_seed = convert_seeds(seed)

            cmb_map = self.get_beamed_signal(channel,s_i,s_set)
            noise_map = self.get_noise_map(noise_seed,channel)
            noise_oalms = self.map2alm(noise_map[0])
            ls,nells,nells_P = self.get_noise_power(channel,beam_deconv=False)
            imap = (cmb_map + noise_map)
            imap = imap * self.mask

            oalms = self.map2alm(imap)

            beam=maps.gauss_beam(self.beam,ls)
            oalms = curvedsky.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms

            oalms[~np.isfinite(oalms)] = 0
            filt_t = lambda x: 1.

            almt = qe.filter_alms(oalms[0].copy(),filt_t,lmin=lmin,lmax=lmax)
            return almt
        

    def prepare_shear_map(self,channel,seed,lmin,lmax):
        """
        Generates a beam-deconvolved Tmap used for the shear estimator
        """
        print("loading shear map")
        s_i,s_set,noise_seed = convert_seeds(seed)


        cmb_map = self.get_beamed_signal(channel,s_i,s_set)
        noise_map = self.get_noise_map(noise_seed,channel)

        imap = (cmb_map + noise_map)
        imap = imap * self.mask

        ls,nells,nells_P = self.get_noise_power(channel,beam_deconv=True)
        nells_T = maps.interp(ls,nells) if not(self.disable_noise) else lambda x: x*0
        nells_P = maps.interp(ls,nells_P) if not(self.disable_noise) else lambda x: x*0
        
        oalms = self.map2alm(imap)
        print(self.disable_noise)
        oalms = curvedsky.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms
        oalms[~np.isfinite(oalms)] = 0

        ls,nells,nells_P = self.get_noise_power(channel,beam_deconv=True)
        #need to multiply by derivative cl
        der=lambda x: np.gradient(x)
        filt_t = lambda x: (1./(x*(self.cltt(x) + nells_T(x))**2))*der(self.cltt(x))

        almt = qe.filter_alms(oalms[0],filt_t,lmin=lmin,lmax=lmax)
        return almt

    def get_kmap(self,channel,seed,lmin,lmax,filtered=True):
        # Wrapper around self.prepare_map that uses caching
        if not(seed in self.cache.keys()): self.prepare_map(channel,seed,lmin,lmax)
        xs = {'T':0,'E':1,'B':2}
        return self.cache[seed][:3] if filtered else self.cache[seed][3:]

    def get_mv_kappa(self,polcomb,talm,ealm,balm):
        
        # Wrapper for qfunc
        return self.qfunc(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfunc(self,alpha,X,Y):
        # Wrapper for the core falafel full-sky lensing reconstruction function
        polcomb = alpha
        return qe.qe_all(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])[polcomb][0]

    def get_mv_curl(self,polcomb,talm,ealm,balm):
    
        return self.qfunc_curl(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfunc_curl(self,alpha,X,Y):
        polcomb = alpha
        return qe.qe_all(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])[polcomb][1]

    def get_mv_mask(self,polcomb,talm,ealm,balm):
        
        return self.qfuncmask(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfuncmask(self,alpha,X,Y):
        """mask reconstruction"""
        polcomb = alpha
        return qe.qe_mask(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])
    
    def get_pointsources(self,polcomb,talm,ealm,balm):
        return self.qfunc_ps(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfunc_ps(self,alpha,X,Y):
        """Point source reconstruction from Falafel"""
        polcomb = alpha
        return qe.qe_pointsources(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])

    def get_noise_power(self,channel=None,beam_deconv=False):
        if (self.wnoise is not None) or self.disable_noise:
            ls = np.arange(self.mlmax+1)
            if not(self.disable_noise):
                bfact = maps.gauss_beam(self.beam,ls)**2. if beam_deconv else np.ones(ls.size)
                nells = (self.wnoise*np.pi/180./60.)**2. / bfact
                nells_P = nells * 2.
            else:
                nells = ls*0
                nells_P = ls*0
        else:
            if self.atmosphere:
                ls,nells = self.nsim.ell,self.nsim.noise_ell_T[channel.telescope][int(channel.band)]
                ls,nells_P = self.nsim.ell,self.nsim.noise_ell_P[channel.telescope][int(channel.band)]
            else:
                ls = np.arange(self.mlmax+1)
                nells = ls*0 + self.nsim.get_white_noise_power(channel) 
                nells_P = 2 * nells
        assert ls[0]==0
        assert ls[1]==1
        assert ls[2]==2
        return ls,nells,nells_P
    

    def initialize_norm(self,ch,lmin,lmax,recalculate=False,quicklens=False,curl=False,label=None):
        lstr = "" if label is None else f"{label}_"
        wstr = "" if self.wnoise is None else "wnoise_"
        onormfname = opath+"norm_%s%slmin_%d_lmax_%d.txt" % (wstr,lstr,lmin,lmax)
        onormcurlfname = opath+"norm_%s%slmin_%d_lmax_%d.txt" % (wstr,lstr,lmin,lmax)
        try:
            assert not(recalculate), "Recalculation of norm requested."
            if curl:
                print("using curl norm")
                print(onormcurlfname)
                return np.loadtxt(onormcurlfname,unpack=True)
            else:
                return np.loadtxt(onormfname,unpack=True)
        except:
            print(traceback.format_exc())
            thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
            theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
            ells,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1])

            class T:
                def __init__(self):
                    self.lCl = lambda p,x: maps.interp(ells,gt)(x)
            theory_cross = T()
            ls,nells,nells_P = self.get_noise_power(ch,beam_deconv=True)
            cltt=theory.lCl('TT',ls)+nells

            if quicklens: #now cmblensplus
                Als = {}
                Acs={}
                ls,Ag,Ac = cmblensplus_norm(nells,nells_P,nells_P,theory,theory_cross,lmin,lmax)
                Als['TT']=Ag[0];Als['TE']=Ag[1];Als['EE']=Ag[2];Als['TB']=Ag[3];Als['EB']=Ag[4];al_mv =1/(1/Als['EB']+1/Als['TB']+1/Als['EE']+1/Als['TE']+1/Als['TT']);al_mv_pol = 1/(1/Als['EB']+1/Als['TB']);Al_te_hdv = Als['TT']*0 
                Acs['TT']=Ac[0];Acs['TE']=Ac[1];Acs['EE']=Ac[2];Acs['TB']=Ac[3];Acs['EB']=Ac[4];ac_mv =1/(1/Acs['EB']+1/Acs['TB']+1/Acs['EE']+1/Acs['TE']+1/Acs['TT']);ac_mv_pol = 1/(1/Acs['EB']+1/Acs['TB']);Ac_te_hdv = Acs['TT']*0 
            else:
                ells = np.arange(lmax+100)
                uctt = theory.lCl('TT',ells)
                ucee = theory.lCl('EE',ells)
                ucte = theory.lCl('TE',ells)
                ucbb = theory.lCl('BB',ells)
                tctt = uctt + maps.interp(ls,nells)(ells)
                tcee = ucee + maps.interp(ls,nells_P)(ells)
                tcte = ucte 
                tcbb = ucbb + maps.interp(ls,nells_P)(ells)
                ls,Als,al_mv_pol,al_mv,Al_te_hdv = qe.symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
            io.save_cols(onormfname,(ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv))
            io.save_cols(onormcurlfname,(ls,Acs['TT'],Acs['EE'],Acs['EB'],Acs['TE'],Acs['TB'],ac_mv_pol,ac_mv,Ac_te_hdv))
            if curl:
                print("curl norm")
                return ls,Acs['TT'],Acs['EE'],Acs['EB'],Acs['TE'],Acs['TB'],ac_mv_pol,ac_mv,Ac_te_hdv
            else: 
                return ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv
    
    def analytic_n1(self,ch,lmin,lmax,Lmin_out=2,Lmaxout=3000,Lstep=20,label=None):
        
        from solenspipe import biastheory as nbias
        lstr = "" if label is None else f"{label}_"
        wstr = "" if self.wnoise is None else "wnoise_"
        onormfname = opath+"norm_%s%slmin_%d_lmax_%d.txt" % (wstr,lstr,lmin,lmax)
        n1fname=opath+"analytic_n1_%s%slmin_%d_lmax_%d.txt"% (wstr,lstr,lmin,lmax)
        try:
            return np.loadtxt(n1fname,unpack=True)
        except:
            print(traceback.format_exc())        
            norms=np.loadtxt(onormfname)
            thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
            theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
            ls,nells,nells_P = self.get_noise_power(ch,beam_deconv=True)
            NOISE_LEVEL=nells[:lmax]
            polnoise=nells_P[:lmax]
            LMAX_TT=Lmaxout
            TMP_OUTPUT=config['data_path']
            LCORR_TT=0
            lens=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lenspotentialCls.dat",unpack=True)
            cls=np.loadtxt(config['data_path']+"cosmo2017_10K_acc3_lensedCls.dat",unpack=True)
            
            #arrays with l starting at l=2"
            #clphiphi array starting at l=2
            clpp=lens[5,:][:8249]
            #cls is an array containing [cltt,clee,clbb,clte] used for the filters
            cltt=cls[1]       
            clee=cls[2]
            clbb=cls[3]
            clte=cls[4]
            bins=norms[2:,0]
            ntt=norms[2:,1]
            nee=norms[2:,2]
            neb=norms[2:,3]
            nte=norms[2:,4]
            ntb=norms[2:,5]
            nbb=np.ones(len(ntb))
            norms=np.array([[ntt/bins**2],[nee/bins**2],[neb/bins**2],[nte/bins**2],[ntb/bins**2],[nbb]])
            n1tt,n1ee,n1eb,n1te,n1tb=nbias.compute_n1_py(clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,Lmaxout,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
            n1mv=nbias.compute_n1mv(clpp,norms,cls,cltt,clee,clbb,clte,NOISE_LEVEL,polnoise,lmin,Lmaxout,LMAX_TT,LCORR_TT,TMP_OUTPUT,Lstep,Lmin_out)
            n1bins=np.arange(Lmin_out,Lmaxout,Lstep)
            io.save_cols(n1fname,(n1bins,n1tt,n1ee,n1eb,n1te,n1tb,n1mv))

            
        return n1bins,n1tt,n1ee,n1eb,n1te,n1tb,n1mv   
    

def initialize_mask(nside,smooth_deg):
    omaskfname = "lensing_mask_nside_%d_apodized_%.1f.fits" % (nside,smooth_deg)
    try:
        return hp.read_map(opath + omaskfname)
    except:
        mask = hp.ud_grade(hp.read_map(opath + config['mask_name']),nside)
        mask[mask<0] = 0
        mask = hp.smoothing(mask,np.deg2rad(smooth_deg))
        mask[mask<0] = 0
        hp.write_map(opath + omaskfname,mask,overwrite=True)
        return mask


def cmblensplus_norm(nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    import curvedsky as cs
    print('compute cmblensplus norm')
    Tcmb = 2.726e6    # CMB temperature
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])/Tcmb**2
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    ocl= fcl+noise
    Ag, Ac, Wg, Wc = cs.norm_lens.qall(QDO,lmax,rlmin,rlmax,lcl,ocl)
    fac=ls*(ls+1)
    return ls,Ag*fac,Ac*fac

def diagonal_RDN0(get_sim_power,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax,simn):
    """Curvedsky dumb N0 for TT,EE,EB,TE,TB"""
    import curvedsky as cs
    print('compute dumb N0')
    Tcmb = 2.726e6    # CMB temperature
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])/Tcmb**2
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    ocl= fcl+noise
    ocl[np.where(ocl==0)] = 1e30
    AgTT,AcTT=cs.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],ocl[0,:])
    AgTE,AcTE=cs.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:])
    AgTB,AcTB=cs.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:])
    AgEE,AcEE=cs.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:])
    AgEB,AcEB=cs.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:])

    fac=ls*(ls+1)
    #prepare the sim total power spectrum
    cldata=get_sim_power((0,0,simn))
    sim_ocl=np.array([cldata[0][:ls.size],cldata[1][:ls.size],cldata[2][:ls.size],cldata[3][:ls.size]])/Tcmb**2
    #dataxdata
    cl=ocl**2/(sim_ocl)
    AgTT0,AcTT0=cs.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:])
    AgTE0,AcTE0=cs.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
    AgTB0,AcTB0=cs.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:])
    AgEE0,AcEE0=cs.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:])
    AgEB0,AcEB0=cs.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
    #(data-sim) x (data-sim)
    cl=ocl**2/(ocl-sim_ocl)
    AgTT1,AcTT1=cs.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:])
    AgTE1,AcTE1=cs.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
    AgTB1,AcTB1=cs.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:])
    AgEE1,AcEE1=cs.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:])
    AgEB1,AcEB1=cs.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
    AgTT0[np.where(AgTT0==0)] = 1e30
    AgTT1[np.where(AgTT1==0)] = 1e30
    AgEE0[np.where(AgEE0==0)] = 1e30
    AgEE1[np.where(AgEE1==0)] = 1e30
    AgEB0[np.where(AgEB0==0)] = 1e30
    AgEB1[np.where(AgEB1==0)] = 1e30
    AgTE0[np.where(AgTE0==0)] = 1e30
    AgTE1[np.where(AgTE1==0)] = 1e30

    n0TTg = AgTT**2*(1./AgTT0-1./AgTT1)
    n0TEg = AgTE**2*(1./AgTE0-1./AgTE1)
    n0TBg = AgTB**2*(1./AgTB0-1./AgTB1)
    n0EEg = AgEE**2*(1./AgEE0-1./AgEE1)
    n0EBg = AgEB**2*(1./AgEB0-1./AgEB1)
    n0=np.array([n0TTg,n0TEg,n0EEg,n0TBg,n0EBg])*fac

    return ls,n0TTg*fac,n0EEg*fac,n0EBg*fac,n0TEg*fac,n0TBg*fac

    
def diagonal_RDN0mv(get_sim_power,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax,simn):
    """Curvedsky dumb N0 for MV"""
    import curvedsky as cs
    print('compute dumb N0')
    Tcmb = 2.726e6    # CMB temperature
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])/Tcmb**2
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    ocl= fcl+noise
    AgTT,AcTT=cs.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],ocl[0,:] ,gtype='k')
    AgTE,AcTE=cs.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:] ,gtype='k')
    AgTB,AcTB=cs.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:], gtype='k')
    AgEE,AcEE=cs.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:] ,gtype='k')
    AgEB,AcEB=cs.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:], gtype='k')

    #prepare the sim total power spectrum
    cldata=get_sim_power((0,0,simn))
    sim_ocl=np.array([cldata[0][:ls.size],cldata[1][:ls.size],cldata[2][:ls.size],cldata[3][:ls.size]])/Tcmb**2
    #dataxdata
    sim_ocl[np.where(sim_ocl==0)] = 1e30
    cl=ocl**2/(sim_ocl)
    AgTT0,AcTT0=cs.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:] ,gtype='k')
    AgTE0,AcTE0=cs.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:], gtype='k')
    AgTB0,AcTB0=cs.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:], gtype='k')
    AgEE0,AcEE0=cs.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:] ,gtype='k')
    AgEB0,AcEB0=cs.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:], gtype='k')
    ATTTE0,__=cs.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:], cl[0,:], ocl[1,:]*sim_ocl[0,:]/ocl[0,:],sim_ocl[3,:],gtype='k')
    ATTEE0,__=cs.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], sim_ocl[3,:], gtype='k')
    ATEEE0,__=cs.norm_lens.qteee(lmax, rlmin, rlmax, lcl[1,:], lcl[3,:], ocl[0,:]*sim_ocl[1,:]/ocl[1,:], cl[1,:], sim_ocl[3,:], gtype='k')
    ATBEB0,__=cs.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], sim_ocl[3,:], gtype='k')


    #(data-sim) x (data-sim)
    cl=ocl**2/(ocl-sim_ocl)
    AgTT1,AcTT1=cs.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:] ,gtype='k')
    AgTE1,AcTE1=cs.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:],gtype='k')
    AgTB1,AcTB1=cs.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:],gtype='k')
    AgEE1,AcEE1=cs.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],gtype='k')
    AgEB1,AcEB1=cs.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:],gtype='k')
    ATTTE1,__=cs.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:],cl[0,:] ,(1-sim_ocl[0,:]/ocl[0,:])*ocl[1,:] , ocl[3,:]-sim_ocl[3,:],gtype='k')
    ATTEE1,__=cs.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], ocl[3,:]-sim_ocl[3,:],gtype='k')
    ATEEE1,__=cs.norm_lens.qteee(lmax, rlmin, rlmax, lcl[1,:], lcl[3,:], (1-sim_ocl[1,:]/ocl[1,:])*ocl[0,:],cl[1,:],ocl[3,:]-sim_ocl[3,:],gtype='k')
    ATBEB1,__=cs.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], ocl[3,:]-sim_ocl[3,:],gtype='k')


    AgTT0[np.where(AgTT0==0)] = 1e30
    AgTT1[np.where(AgTT1==0)] = 1e30
    AgEE0[np.where(AgEE0==0)] = 1e30
    AgEE1[np.where(AgEE1==0)] = 1e30
    AgEB0[np.where(AgEB0==0)] = 1e30
    AgEB1[np.where(AgEB1==0)] = 1e30
    AgTE0[np.where(AgTE0==0)] = 1e30
    AgTE1[np.where(AgTE1==0)] = 1e30
    ATTTE0[np.where(ATTTE0==0)] = 1e30
    ATTTE1[np.where(ATTTE1==0)] = 1e30
    ATTEE0[np.where(ATTEE0==0)] = 1e30
    ATTEE1[np.where(ATTEE1==0)] = 1e30
    ATEEE0[np.where(ATEEE0==0)] = 1e30
    ATEEE1[np.where(ATEEE1==0)] = 1e30
    ATBEB0[np.where(ATBEB0==0)] = 1e30
    ATBEB1[np.where(ATBEB0==0)] = 1e30

    n0TTg = AgTT**2*(1./AgTT0-1./AgTT1)
    n0TEg = AgTE**2*(1./AgTE0-1./AgTE1)
    n0TBg = AgTB**2*(1./AgTB0-1./AgTB1)  
    n0EEg = AgEE**2*(1./AgEE0-1./AgEE1)
    n0EBg = AgEB**2*(1./AgEB0-1./AgEB1)
    n0TTTE=AgTT*AgTE*(ATTTE0+ATTTE1)
    n0TTEE=AgTT*AgEE*(ATTEE0+ATTEE1)
    n0TEEE=AgTE*AgEE*(ATEEE0+ATEEE1)
    n0TBEB=AgTB*AgEB*(ATBEB0+ATBEB1)

    dumbn0=[n0TTg,n0TEg,n0TBg,n0EBg,n0EEg,n0TTTE,n0TTEE,n0TEEE,n0TBEB]
    weights_NUM=[1/AgTT**2,1/AgTE**2,1/AgTB**2,1/AgEB**2,1/AgEE**2,2/(AgTT*AgTE),2/(AgTT*AgEE)
    ,2/(AgTE*AgEE),2/(AgTB*AgEB)]
    weights_den=[1/AgTT**2,1/AgTE**2,1/AgTB**2,1/AgEB**2,1/AgEE**2,2/(AgTT*AgTE),2/(AgTT*AgTB),2/(AgTT*AgEB),2/(AgTT*AgEE),
    2/(AgTE*AgTB),2/(AgTE*AgEB),2/(AgTE*AgEE),2/(AgTB*AgEB),2/(AgTB*AgEE),2/(AgEB*AgEE)]
 
    mvdumbN0=np.zeros(len(n0TTg))
    sumc=np.zeros(len(n0TTg))  
    for i in range(len(weights_den)):
        sumc+=weights_den[i]
    for i in range(len(weights_NUM)):
        mvdumbN0+=np.nan_to_num(weights_NUM[i])*np.nan_to_num(dumbn0[i])
    mvdumbN0=mvdumbN0/sumc
    fac=ls*(ls+1)*0.25
    
    return ls,mvdumbN0/fac


        

def bias_hard_mask_norms(nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """return normalization for mask reconstruction"""
    import curvedsky as cs
    Tcmb = 2.726e6    # CMB temperature
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])/Tcmb**2
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    ocl= fcl+noise
    A_mask = cs.norm_tau.qtt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:])
    Alpp, __ = cs.norm_lens.qtt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:])
    Rlpt = cs.norm_lens.ttt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:]) #this is unnormalized
    fac=ls*(ls+1)*0.5
    detR=1-Alpp*A_mask*Rlpt**2
    bhmask=Alpp*Rlpt/detR
    bhp=1/detR
    bhclkknorm=fac**2*Alpp/detR
    return ls,bhp,bhmask,Alpp,A_mask,bhclkknorm

def bias_hard_ps_norms(nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """Normalizations for point source reconstruction"""
    import curvedsky as cs
    Tcmb = 2.726e6    # CMB temperature
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))
    noise=np.array([nltt,nlee,nlbb,nlte])/Tcmb**2
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    ocl= fcl+noise
    A_ps = cs.norm_src.qtt(lmax,rlmin,rlmax,ocl[0,:])
    Alpp, __ = cs.norm_lens.qtt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:])
    Rlps = cs.norm_lens.stt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:]) #this is unnormalized
    fac=ls*(ls+1)*0.5
    detR=1-Alpp*A_ps*Rlps**2
    bhps=Alpp*Rlps/detR
    bhp=1/detR
    bhclkknorm=fac**2*Alpp/detR

    return ls,bhp,bhps,Alpp,A_ps,bhclkknorm   
        

def cmblensplusreconstruction(solint,w2,w3,w4,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """example of reconstruction using Toshiya's cmblensplus pipeline"""
    import curvedsky as cs
    mlmax=lmax

    polcomb='TT'
    Tcmb = 2.726e6    # CMB temperature
    Lmax = lmax       # maximum multipole of output normalization
    rlmin = lmin
    rlmax = lmax      # reconstruction multipole range
    ls = np.arange(0,Lmax+1)
    QDO = [True,True,True,True,True,False]
    nltt=nltt[:ls.size]
    nlee=nlee[:ls.size]
    nlbb=nlbb[:ls.size]
    nlte=np.zeros(len(nltt))

    noise=np.array([nltt,nlee,nlbb,nlte])/Tcmb**2
    lcl=np.array([theory_cross.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    fcl=np.array([theory.lCl('TT',ls),theory.lCl('EE',ls),theory.lCl('BB',ls),theory.lCl('TE',ls)])/Tcmb**2
    ocl= fcl+noise
    Ag, Ac, Wg, Wc = cs.norm_lens.qall(QDO,lmax,rlmin,rlmax,lcl,ocl) #the Als used (same norm as used for theory) TT,TE,EE,TB,EB
    #load the beam deconvolved alms
    sTalm=hp.fitsfunc.read_alm(config['data_path']+'pipetest/talms.fits')
    sEalm=hp.fitsfunc.read_alm(config['data_path']+'pipetest/ealms.fits')
    sBalm=hp.fitsfunc.read_alm(config['data_path']+'pipetest/balms.fits')
    mlm=int(0.5*(-3+np.sqrt(9-8*(1-len(sTalm)))))
    sTalm = cs.utils.lm_healpy2healpix(len(sTalm), sTalm, mlm) 
    sEalm = cs.utils.lm_healpy2healpix(len(sEalm), sEalm, mlm) 
    sBalm = cs.utils.lm_healpy2healpix(len(sBalm), sBalm, mlm) 
    Talm=sTalm[:rlmax+1,:rlmax+1]/Tcmb
    Ealm=sEalm[:rlmax+1,:rlmax+1]/Tcmb
    Balm=sBalm[:rlmax+1,:rlmax+1]/Tcmb
    Talm[~np.isfinite(Talm)] = 0
    Ealm[~np.isfinite(Ealm)] = 0
    Balm[~np.isfinite(Balm)] = 0

    Fl = np.zeros((3,rlmax+1,rlmax+1))
    for l in range(rlmin,rlmax+1):
        Fl[:,l,0:l+1] = 1./ocl[:3,l,None]
        
    Talm *= Fl[0,:,:]
    Ealm *= Fl[1,:,:]
    Balm *= Fl[2,:,:]
    # compute unnormalized estimator
    glm, clm = {}, {}
    glm['TT'], clm['TT'] = cs.rec_lens.qtt(lmax,rlmin,rlmax,lcl[0,:],Talm,Talm)
    glm['TE'], clm['TE'] = cs.rec_lens.qte(lmax,rlmin,rlmax,lcl[3,:],Talm,Ealm)
    glm['EE'], clm['EE'] = cs.rec_lens.qee(lmax,rlmin,rlmax,lcl[1,:],Ealm,Ealm)
    glm['TB'], clm['TB'] = cs.rec_lens.qtb(lmax,rlmin,rlmax,lcl[3,:],Talm,Balm)
    glm['EB'], clm['EB'] = cs.rec_lens.qeb(lmax,rlmin,rlmax,lcl[1,:],Ealm,Balm)
    
    
    # normalized estimators
    ell=np.arange(lmax+1)
    fac=ell*(ell+1)/2
    for qi, q in enumerate(['TT','TE','EE','TB','EB']):
        glm[q] *= Ag[qi,:,None] 
    glm['MV']=0.
    for qi, q in enumerate(['TT','TE','EE','TB','EB']):
        glm['MV'] += Wg[qi,:,None]*glm[q]

    glm['MV']=glm['MV'] * Ag[5,:,None]
    istr = str(0).zfill(5)
    phifname = "/project/projectdirs/act/data/actsims_data/signal_v0.4/fullskyPhi_alm_%s.fits" % istr
    kalms=plensing.phi_to_kappa(hp.read_alm(phifname))
    phimap=hp.alm2map(kalms.astype(complex),2048)
    kalms=cs.utils.hp_map2alm(2048, rlmax, mlmax, phimap)
    kalms = solint.get_kappa_alm(0+0)
    lm=int(0.5*(-3+np.sqrt(9-8*(1-len(kalms)))))
    kalms = cs.utils.lm_healpy2healpix(len(kalms), kalms, lm) 
    kalms=kalms[:lmax+1,:lmax+1]
    macl=np.zeros(rlmax+1)
    micl=np.zeros(rlmax+1)
    mxcl=np.zeros(rlmax+1)
    micl+=cs.utils.alm2cl(rlmax,kalms,kalms)/w2
    acl=cs.utils.alm2cl(rlmax,glm[polcomb],glm[polcomb])/w4
    macl+= fac**2*acl
    xcl=cs.utils.alm2cl(rlmax,glm[polcomb],kalms)/w3
    mxcl+=xcl*fac
    normMV=Ag[5]*fac**2


class weighted_bin1D:
    '''
    * Takes data defined on x0 and produces values binned on x.
    * Assumes x0 is linearly spaced and continuous in a domain?
    * Assumes x is continuous in a subdomain of x0.
    * Should handle NaNs correctly.
    '''
    

    def __init__(self, bin_edges):

        self.update_bin_edges(bin_edges)

    def update_bin_edges(self,bin_edges):
        
        self.bin_edges = bin_edges
        self.numbins = len(bin_edges)-1
        self.cents = (self.bin_edges[:-1]+self.bin_edges[1:])/2.

        self.bin_edges_min = self.bin_edges.min()
        self.bin_edges_max = self.bin_edges.max()
        
    
    def bin(self,ix,iy,weights):
        #binning which allows to optimally weight for signal and noise. weights the same size as y
        x = ix.copy()
        y = iy.copy()
        # this just prevents an annoying warning (which is otherwise informative) everytime
        # all the values outside the bin_edges are nans
        y[x<self.bin_edges_min] = 0
        y[x>self.bin_edges_max] = 0
        bin_means=[]
        for i in range(1,len(self.bin_edges)):
            bin_means.append(np.nansum(weights[self.bin_edges[i-1]:self.bin_edges[i]]*iy[self.bin_edges[i-1]:self.bin_edges[i]])/np.nansum(weights[self.bin_edges[i-1]:self.bin_edges[i]]))
        bin_means=np.array(bin_means)
        return self.cents,bin_means
        
    def binning_matrix(self,ix,iy,weights):
        #return the binning matrix used for the data products
        x = ix.copy()
        y = iy.copy()
        y[x<self.bin_edges_min] = 0
        y[x>self.bin_edges_max] = 0
        #num columns
        matrix=[]
    
        #num rows
        nrows=len(self.bin_edges)
        for i in range(1,nrows):
            col=np.zeros(len(y))
            col[self.bin_edges[i-1]:self.bin_edges[i]]=weights[self.bin_edges[i-1]:self.bin_edges[i]]/np.sum(weights[self.bin_edges[i-1]:self.bin_edges[i]])
            matrix.append(col)
        matrix=np.array(matrix)
        return matrix 
        
        
        