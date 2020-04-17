from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
from orphics import maps,io,cosmology # msyriac/orphics ; pip install -e . --user
from pixell import enmap,lensing as plensing,curvedsky
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,Channel,SOStandalonePrecomputedCMB
from falafel import qe
from solenspipe._lensing_biases import lensingbiases as lensingbiases_f
from solenspipe._lensing_biases import checkproc as checkproc_f
import os
import glob


config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']


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
    def __init__(self,mask,data_mode=None,scanning_strategy="isotropic",fsky=0.4,white_noise=None,beam_fwhm=None,disable_noise=False,atmosphere=True):
        self.mask = mask
        self._debug = False
        self.atmosphere = atmosphere
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
                                               fsky={'LA':fsky} if fsky is not None else None)    
        else:
            self.wnoise = white_noise
            self.beam = beam_fwhm
        thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
        theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
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
        # if data_mode=='sim':
        #     self.


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
            #s_i,s_set,noise_seed = convert_seeds(seed)
            ls,nells,nells_P = self.get_noise_power(channel,beam_deconv=False)
            nseed = noise_seed+(int(channel.band),)
            
            if self.wnoise is None:
                noise_map = self.nsim.simulate(channel,seed=nseed,atmosphere=self.atmosphere)
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

    def prepare_map(self,channel,seed,lmin,lmax):
        """
        Generates a beam-deconvolved simulation.
        Filters it and caches it.
        """
        # Convert the solenspipe convention to the Alex convention
        s_i,s_set,noise_seed = convert_seeds(seed)

        if self.beam is None:
            self.beam = self.nsim.get_beam_fwhm(channel)
        cmb_alm = get_cmb_alm(s_i,s_set).astype(np.complex128)
        cmb_alm = curvedsky.almxfl(cmb_alm,lambda x: maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else cmb_alm
        cmb_map = self.alm2map(cmb_alm)

        noise_map = self.get_noise_map(noise_seed,channel)
        noise_map[1 : ]*= np.sqrt(2.)
  
        imap = (cmb_map + noise_map)
        imap = imap - imap.mean()  #what does this do?
        imap = imap * self.mask
    
        #self._debug = True
        if self._debug:
            for i in range(3): io.hplot(imap[i],f'imap_{i}')

        
        oalms = self.map2alm(imap)
        #hp.fitsfunc.write_alm("/global/homes/j/jia_qu/cmblensplus/example/cmblens/selm.fits", oalms[1])
        oalms = curvedsky.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms
        oalms[~np.isfinite(oalms)] = 0
        clte = hp.alm2cl(oalms[0],oalms[1])
        ecl = hp.alm2cl(oalms[1])
        np.savetxt("/global/homes/j/jia_qu/cmblensplus/example/cmblens/scltemap.txt",clte)
        np.savetxt("/global/homes/j/jia_qu/cmblensplus/example/cmblens/scleemap.txt",ecl)

        ls,nells,nells_P = self.get_noise_power(channel,beam_deconv=True)
        nells_T = maps.interp(ls,nells) if not(self.disable_noise) else lambda x: x*0
        nells_P = maps.interp(ls,nells_P) if not(self.disable_noise) else lambda x: x*0
        filt_t = lambda x: 1./(self.cltt(x) + nells_T(x))
        filt_e = lambda x: 1./(self.clee(x) + nells_P(x))
        filt_b = lambda x: 1./(self.clbb(x) + nells_P(x))


        almt = qe.filter_alms(oalms[0].copy(),filt_t,lmin=lmin,lmax=lmax)
        alme = qe.filter_alms(oalms[1].copy(),filt_e,lmin=lmin,lmax=lmax)
        almb = qe.filter_alms(oalms[2].copy(),filt_b,lmin=lmin,lmax=lmax)

        self.cache = {}
        self.cache[seed] = (almt,alme,almb,oalms[0],oalms[1],oalms[2])

    def prepare_shear_map(self,channel,seed,lmin,lmax):
        """
        Generates a beam-deconvolved simulation.
        Filters it and caches it.
        """
        # Convert the solenspipe convention to the Alex convention
        s_i,s_set,noise_seed = convert_seeds(seed)

        cmb_alm = get_cmb_alm(s_i,s_set).astype(np.complex128)
        cmb_map = self.alm2map(cmb_alm)
        
        noise_map = self.nsim.simulate(channel,seed=noise_seed+(int(channel.band),))
        noise_map[noise_map<-1e24] = 0
        noise_map[np.isnan(noise_map)] = 0

        imap = (cmb_map + noise_map)*self.mask
        nells_T = maps.interp(self.nsim.ell,self.nsim.noise_ell_T[channel.telescope][int(channel.band)])
        nells_P = maps.interp(self.nsim.ell,self.nsim.noise_ell_P[channel.telescope][int(channel.band)])
        
        oalms = self.map2alm(imap)
        #need to multiply by derivative cl
        der=lambda x: np.gradient(x)
        filt_t = lambda x: (1./(x*self.cltt(x)*(self.cltt(x) + nells_T(x))))*der(self.cltt(x))
        almt = qe.filter_alms(oalms[0],filt_t,lmin=lmin,lmax=lmax)
        return almt

    def get_kmap(self,channel,seed,lmin,lmax,filtered=True):
        if not(seed in self.cache.keys()): self.prepare_map(channel,seed,lmin,lmax)
        xs = {'T':0,'E':1,'B':2}
        return self.cache[seed][:3] if filtered else self.cache[seed][3:]

    def get_mv_kappa(self,polcomb,talm,ealm,balm):
        return self.qfunc(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfunc(self,alpha,X,Y):
        polcomb = alpha
        return qe.qe_all(self.px,lambda x,y: self.theory.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])[polcomb][0]

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
    

    def initialize_norm(self,ch,lmin,lmax,recalculate=False,quicklens=False,label=None):
        lstr = "" if label is None else f"{label}_"
        wstr = "" if self.wnoise is None else "wnoise_"
        onormfname = opath+"norm_%s%slmin_%d_lmax_%d.txt" % (wstr,lstr,lmin,lmax)
        try:
            if recalculate: raise
            return np.loadtxt(onormfname,unpack=True)
        except:
            thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
            theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)

            ls,nells,nells_P = self.get_noise_power(ch,beam_deconv=True)
            if quicklens:
                Als = {}
                for polcomb in ['TT','TE','EE','EB','TB']:
                    ls,Als[polcomb] = quicklens_norm(polcomb,nells,nells_P,nells_P,theory,lmin,lmax,lmin,lmax)
                al_mv_pol = 1/(1/Als['EB']+1/Als['TB']); al_mv = 1/(1/Als['EB']+1/Als['TB']+1/Als['EE']+1/Als['TE']+1/Als['TT']) ; Al_te_hdv = Als['TT']*0 
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
            return ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv
        

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


        

def quicklens_norm(polcomb,nltt,nlee,nlbb,theory,tellmin,tellmax,pellmin,pellmax,Lmax=None):
    import quicklens as ql
    fmap = {'TT':ql.qest.lens.phi_TT,
            'TE':ql.qest.lens.phi_TE,
            'EE':ql.qest.lens.phi_EE,
            'EB':ql.qest.lens.phi_EB,
            'TB':ql.qest.lens.phi_TB}
    rcl = {'TT':'TT',
            'TE':'TE',
            'EE':'EE',
            'EB':'EE',
            'TB':'TE'}
    nls = {'TT': nltt,'EE': nlee,'BB': nlbb}
    assert nltt.size > (tellmax+1)
    assert nlee.size > (pellmax+1)
    assert nlbb.size > (pellmax+1)

    if Lmax is None: Lmax = max(tellmax,pellmax)+1
    ls = np.arange(0,Lmax+1)
    qest = fmap[polcomb](theory.lCl(rcl[polcomb],ls))

    X,Y = polcomb

    clx = theory.lCl(X+X,ls) + nls[X+X][:ls.size]
    cly = theory.lCl(Y+Y,ls) + nls[Y+Y][:ls.size]
    
    flx        = np.zeros( Lmax+1 ); flx[2:] = 1./clx[2:]
    fly        = np.zeros( Lmax+1 ); fly[2:] = 1./cly[2:]

    if X=='T':
        flx[ls<tellmin] = 0
        flx[ls>tellmax] = 0
    else:
        flx[ls<pellmin] = 0
        flx[ls>pellmax] = 0

    if Y=='T':
        fly[ls<tellmin] = 0
        fly[ls>tellmax] = 0
    else:
        fly[ls<pellmin] = 0
        fly[ls>pellmax] = 0

    resp_fullsky = qest.fill_resp(qest, np.zeros(Lmax+1, dtype=np.complex), flx, fly)
    nlqq_fullsky = 1 / resp_fullsky
    if X!=Y: nlqq_fullsky = nlqq_fullsky / 2.

    return ls,(ls*(ls+1.)) * nlqq_fullsky.real

	
def checkproc_py():
    '''
    Routine to check the number of processors involved
    in the computation (Fortran routines use openmp).
    '''
    nproc = checkproc_f.get_threads()
    if nproc > 1:
        print ('You are using ', nproc, ' processors')
    else:
        print ('###################################')
        print ('You are using ', nproc, ' processor')
        print ('If you want to speed up the computation,')
        print ('set up correctly your number of task.')
        print ('e.g in bash, if you want to use n procs,')
        print ('add this line to your bashrc:')
        print ('export OMP_NUM_THREADS=n')
        print ('###################################')

def compute_n0_py(
    phifile=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lmin_out=None,
    Lstep=None):
    """returns derivatives of kappa N0 noise with respect to the Cls"""
    bins=np.arange(2,2992,20)
    n0tt,n0ee,n0eb,n0te,n0tb=lensingbiases_f.compute_n0(
        phifile,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,Lmin_out,Lstep)
    return n0tt,n0ee,n0eb,n0te,n0tb

	
def compute_n0mix_py(
    phifile=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lmin_out=None,
    Lstep=None):
    bins=np.arange(2,2992,20)
    n0ttee,n0ttte,n0eete,n0ebtb=lensingbiases_f.compute_n0mix(
        phifile,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,Lmin_out,Lstep)
        
    return n0ttee,n0ttte,n0eete,n0ebtb

def compute_n1_py(
    phifile=None,
    normarray=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lstep=None,
    Lmin_out=None
    ):

    n1tt,n1ee,n1eb,n1te,n1tb=lensingbiases_f.compute_n1(
        phifile,
        normarray,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,
        Lstep,
        Lmin_out)
    
    return n1tt,n1ee,n1eb,n1te,n1tb  
    
def compute_n1mix(
    phifile=None,
    normarray=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lstep=None,
    Lmin_out=None):

    n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb=lensingbiases_f.compute_n1mix(
        phifile,
        normarray,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,
        Lstep,
        Lmin_out)
    
    return n1ttee,n1tteb,n1ttte,n1tttb,n1eeeb,n1eete,n1eetb,n1ebte,n1ebtb,n1tetb  
    
	
def n1_derivatives(
    x,
    y,
    phifile=None,
    normarray=None,
    lensedcmbfile=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None,
    Lstep=None,
    Lmin_out=None
    ):
    #x= First set i.e 'TT'
    #y= Second set i.e 'EB'
    lensingbiases_f.compute_n1_derivatives(
        phifile,
        normarray,
        lensedcmbfile,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output,
        Lstep,
        Lmin_out)
    n1 = np.loadtxt(os.path.join(tmp_output,'N1_%s%s_analytical_matrix.dat'% (x, y))).T  
    #column L refer to N(L) being differenciated.
    #row L refer to the C_L(phi) values
    #Output already in convergence kappa format. No need for L**4/4 scaling.

    return n1
    
def covmatrix(N0,CMB_noise,polcomb):
    """load N0 and CMB_noise as array of arrays ['TT','TE','EE','TB','EB']"""
    #on diagonal elements only
    field_weight={}
    for i,pol in enumerate(polcomb):
        field_weight[pol]=1/(N0[i]+CMB_noise[i])
        
    weights={}
    weights['TTTT']=field_weight['TT']**2
    weights['TETE']=field_weight['TE']**2
    weights['EEEE']=field_weight['EE']**2
    weights['TBTB']=field_weight['TB']**2
    weights['EBEB']=field_weight['EB']**2
    weights['TTTE']=field_weight['TT']*field_weight['TE']
    weights['TTEE']=field_weight['TT']*field_weight['EE']
    weights['TTTB']=field_weight['TT']*field_weight['TB']
    weights['TTEB']=field_weight['TT']*field_weight['EB']
    weights['TEEE']=field_weight['EE']*field_weight['TE']
    weights['TETB']=field_weight['TE']*field_weight['TB']
    weights['TEEB']=field_weight['EB']*field_weight['TE']
    weights['EETB']=field_weight['EE']*field_weight['TB']
    weights['EEEB']=field_weight['EE']*field_weight['EB']
    weights['TBEB']=field_weight['TB']*field_weight['EB']
    return weights
