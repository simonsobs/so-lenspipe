from __future__ import print_function
from orphics import maps,io,cosmology
from pixell import enmap,lensing as plensing
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from mapsims import noise,Channel,SOStandalonePrecomputedCMB
from mapsims import SO_Noise_Calculator_Public_20180822 as sonoise
from falafel import qe
from solenspipe._lensing_biases import lensingbiases as lensingbiases_f
from solenspipe._lensing_biases import checkproc as checkproc_f
import os
import glob
import matplotlib
matplotlib.use("Agg")
import pylab as pl
pl.ioff()


config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']


def convert_seeds(seed,nsims=2000,ndiv=4):
    # Convert the solenspipe convention to the Alex convention
    icov,cmb_set,i = seed
    assert icov==0, "Covariance from sims not yet supported."
    nstep = nsims//ndiv
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


class SOLensInterface(object):
    def __init__(self,mask,data_mode=None):
        self.mask = mask
        self.nside = hp.npix2nside(mask.size)
        self.nsim = noise.SONoiseSimulator(nside=self.nside,apply_beam_correction=True)    
        thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
        theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
        self.cltt = lambda x: theory.lCl('TT',x) 
        self.clee = lambda x: theory.lCl('EE',x) 
        self.clbb = lambda x: theory.lCl('BB',x) 
        self.cache = {}
        self.theory = theory
        self.set_data_map(data_mode)

    def set_data_map(self,data_mode=None):
        if data_mode is None:
            print('WARNING: No data mode specified. Defaulting to simulation iset=0,i=0 at 150GHz.')
            data_mode = 'sim'
        # if data_mode=='sim':
        #     self.

    def prepare_map(self,channel,seed,lmin,lmax):
        """
        Generates a beam-deconvolved simulation.
        Filters it and caches it.
        """
        # Convert the solenspipe convention to the Alex convention
        s_i,s_set,noise_seed = convert_seeds(seed)

        cmb_alm = get_cmb_alm(s_i,s_set).astype(np.complex128)
        cmb_map = hp.alm2map(cmb_alm,nside=self.nside)
        noise_map = self.nsim.simulate(channel,seed=noise_seed+(int(channel.band),))
        noise_map[noise_map<-1e24] = 0
        # io.mollview(noise_map[0],"noisemap.png",lim=1000)
        # sys.exit()
        imap = (cmb_map + noise_map)*self.mask
        oalms = hp.map2alm(imap)

        nells_T = maps.interp(self.nsim.ell,self.nsim.noise_ell_T[channel.telescope][int(channel.band)])
        nells_P = maps.interp(self.nsim.ell,self.nsim.noise_ell_P[channel.telescope][int(channel.band)])
        filt_t = lambda x: 1./(self.cltt(x) + nells_T(x))
        filt_e = lambda x: 1./(self.clee(x) + nells_P(x))
        filt_b = lambda x: 1./(self.clbb(x) + nells_P(x))

        almt = qe.filter_alms(oalms[0],filt_t,lmin=lmin,lmax=lmax)
        alme = qe.filter_alms(oalms[1],filt_e,lmin=lmin,lmax=lmax)
        almb = qe.filter_alms(oalms[2],filt_b,lmin=lmin,lmax=lmax)

        self.cache = {}
        self.cache[seed] = (almt,alme,almb,oalms[0],oalms[1],oalms[2])


    def get_kmap(self,channel,X,seed,lmin,lmax,filtered=True):
        xs = {'T':0,'E':1,'B':2}
        assert X in xs.keys()
        if not(seed in self.cache.keys()): self.prepare_map(channel,seed,lmin,lmax)
        return self.cache[seed][xs[X]] if filtered else self.cache[seed][xs[X]+3]

    def get_mv_kappa(self,polcomb,talm,ealm,balm):
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(0.5*8192/self.nside/60.))
        mlmax = 2*self.nside
        res = qe.qe_all(shape,wcs,lambda x,y: self.theory.lCl(x,y),mlmax,talm,ealm,balm,estimators=[polcomb])
        return res[polcomb]
        

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


def initialize_norm(solint,ch,lmin,lmax):
    onormfname = opath+"norm_lmin_%d_lmax_%d.txt" % (lmin,lmax)
    try:
        return np.loadtxt(onormfname,unpack=True)
    except:
        thloc = os.path.dirname(os.path.abspath(__file__)) + "/../data/" + config['theory_root']
        theory = cosmology.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
        ells = np.arange(lmax+100)
        uctt = theory.lCl('TT',ells)
        ucee = theory.lCl('EE',ells)
        ucte = theory.lCl('TE',ells)
        ucbb = theory.lCl('BB',ells)
        ls,nells = solint.nsim.ell,solint.nsim.noise_ell_T[ch.telescope][int(ch.band)]
        ls,nells_P = solint.nsim.ell,solint.nsim.noise_ell_P[ch.telescope][int(ch.band)]
        tctt = uctt + maps.interp(ls,nells)(ells)
        tcee = ucee + maps.interp(ls,nells_P)(ells)
        tcte = ucte 
        tcbb = ucbb + maps.interp(ls,nells_P)(ells)
        ls,Als,al_mv_pol,al_mv,Al_te_hdv = qe.symlens_norm(uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,lmax=lmax,plot=False)
        io.save_cols(onormfname,(ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv))
        return ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv
        
	
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
    FWHM=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None):
    """returns derivatives of kappa N0 noise with respect to the Cls"""
    bins=np.arange(2,2992,20)
    n0tt,n0ee,n0eb,n0te,n0tb=lensingbiases_f.compute_n0(
        phifile,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        FWHM/60.,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output)
    return n0tt,n0ee,n0eb,n0te,n0tb

	
def loadn0(file_path,sample_size):
#prepare N0 to be used by N1 Fortran
#BB has 2990 rows
#sample size, the nth bin where n1 is calculated, determined by L_step in Fortran. Set to 20
    fname = file_path+"N0_analytical.txt" 
    try:
        return np.loadtxt(fname)
        print("NO file present")
    except:
        bb=file_path+"bb.txt" 
        Als = {}
        with bench.show("norm"):
            ls,Als['TT'],Als['EE'],Als['EB'],Als['TE'],Als['TB'],al_mv_pol,al_mv,Al_te_hdv = initialize_norm(solint,ch,lmin,lmax)
        bin=ls[2:][::sample_size]
        TT=Als['TT'][2:][::sample_size]
        EE=Als['EE'][2:][::sample_size]
        EB=Als['EB'][2:][::sample_size]
        TE=Als['TE'][2:][::sample_size]
        TB=Als['TB'][2:][::sample_size]
        BB=bb
        ar=[bin,TT/bin**2,EE/bin**2,EB/bin**2,TE/bin**2,TB/bin**2,BB/bin**2]
        y=np.transpose(ar)
        np.savetxt(fname,y)
        print("saved N0 in: "+fname)

def compute_n1_py(
    phifile=None,
    normarray=None,
    lensedcmbfile=None,
    cltt=None,
    clee=None,
    clbb=None,
    cleb=None,
    FWHM=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None):

    n1tt,n1ee,n1eb,n1te,n1tb=lensingbiases_f.compute_n1(
        phifile,
        normarray,
        lensedcmbfile,
        cltt,
        clee,
        clbb,
        cleb,
        FWHM/60.,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output)
    
    return n1tt,n1ee,n1eb,n1te,n1tb  
    
def compute_n1clphiphi_py(
    phifile=None,
    normarray=None,
    lensedcmbfile=None,
    FWHM=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None):

    n1dtt,n1dee,n1deb,n1dte,n1dtb=lensingbiases_f.compute_n1_derivatives(
        phifile,
        normarray,
        lensedcmbfile,
        FWHM/60.,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output)
    
    return n1dtt,n1dee,n1deb,n1dte,n1dtb #returns n1tt array

	
def n1_derivatives(
    x,
    y,
    phifile=None,
    normarray=None,
    lensedcmbfile=None,
    FWHM=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None):
    #x= First set i.e 'TT'
    #y= Second set i.e 'EB'
    lensingbiases_f.compute_n1_derivatives(
        phifile,
        normarray,
        lensedcmbfile,
        FWHM/60.,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output)
    n1 = np.loadtxt(os.path.join(tmp_output,'N1_%s%s_analytical_matrix.dat'% (x, y))).T  
    #column L refer to N(L) being differenciated.
    #row L refer to the C_L(phi) values
    #Output already in convergence kappa format. No need for L**4/4 scaling.

    return n1


def n0_TT(
    phifile=None,
    lensedcmbfile=None,
    FWHM=None,
    noise_level=None,
    noisep=None,
    lmin=None,
    lmaxout=None,
    lmax_TT=None,
    lcorr_TT=None,
    tmp_output=None):
    #x= First set i.e 'TT'
    #y= Second set i.e 'EB'
    lensingbiases_f.compute_n0_tt(
        phifile,
        lensedcmbfile,
        FWHM/60.,
        noise_level,
        noisep,
        lmin,
        lmaxout,
        lmax_TT,
        lcorr_TT,
        tmp_output)
    print("saved file to")
    print(tmp_output)
    
def plot_biases(bins, phiphi, MV_n1=None, N1_array=None):
    '''
    Quick plot for inspection
    '''
    tphi = lambda l: (l + 0.5)**4 / (2. * np.pi) # scaling to apply to cl_phiphi when plotting.
    colors = lambda i: matplotlib.cm.jet(i * 60)

    ## Plot lensing
    pl.loglog(bins, phiphi, color='grey', label='Lensing')


    ## Plot N1
    if MV_n1 is not None:
        pl.loglog(bins, MV_n1 * tphi(bins), color='black', lw=2, ls='--', label='N1 bias')
    if N1_array is not None:
        indices = ['TT','EE','EB','TE','TB','BB']
        for i in range(len(N1_array)):
            pl.loglog(
                bins,
                N1_array[i][i][:] * tphi(bins),
                color=colors(i),
                ls='--',
                lw=2,
                alpha=1,
                label=indices[i]+indices[i])

    pl.xlabel('$\ell$', fontsize=20)
    pl.ylabel(r"$[\ell(\ell+1)]^2/(2\pi)C_\ell^{\phi^{XY} \phi^{ZW}}$", fontsize=20)
    leg=pl.legend(loc='best', ncol=2, fontsize=12.5)
    leg.get_frame().set_alpha(0.0)
    pl.savefig('Biases.pdf')
    pl.clf()
    
    
def minimum_variance_n0(N0_array, N0_names, checkit=False):
	'''
	Compute the variance of the minimum variance estimator and the associated weights.
	Input:
		* N0_array: ndarray, contain the N0s to combine
		* N0_names: ndarray of string, contain the name of the N0s to combine (['TTTT', 'EEEE', etc.])
	Output:
		* minimum_variance_n0: 1D array, the MV N0
		* weights*minimum_variance_n0: ndarray, the weights for each spectrum (TT, EE, etc.)
		* N0_names_ordered: 1D array, contain the name of the spectra (TT, EE, etc.)
	'''
	N0_array = np.reshape(N0_array, (len(N0_array)**2, len(N0_array[0][0])))
	N0_names_full = ['%s%s'%(i, j) for i in N0_names for j in N0_names]

	## Fill matrix
	sub_vec = [[name, pos] for pos, name in enumerate(N0_names)]
	dic_mat = {'%s%s'%(XY, ZW):[i, j] for XY, i in sub_vec for ZW, j in sub_vec}

	## Build the inverse matrix for each ell
	def build_inv_submatrix(vector_ordered, names_ordered, dic_mat, nsub_element):
		mat = np.zeros((nsub_element, nsub_element))
		for pos, name in enumerate(names_ordered):
			mat[dic_mat[name][0]][dic_mat[name][1]] = mat[dic_mat[name][1]][dic_mat[name][0]] = vector_ordered[pos]
		return np.linalg.pinv(mat)

	inv_submat_array = np.array([
		build_inv_submatrix(
			vector_ordered,
			N0_names_full,
			dic_mat,
			len(N0_names)) for vector_ordered in np.transpose(N0_array)])
	inv_N0_array = np.array([ np.sum(submat) for submat in inv_submat_array ])
	minimum_variance_n0 = 1. / inv_N0_array

    
        
	weights = np.array([[np.sum(submat[i]) for submat in inv_submat_array] for i in range(6) ])
	#weights = np.array([np.sum(submat[i]) for submat in inv_submat_array])

	if checkit:
		print ('Sum of weights = ', np.sum(weights * minimum_variance_n0) / len(minimum_variance_n0))
		print ('Is sum of weights 1? ',np.sum(weights * minimum_variance_n0) / len(minimum_variance_n0) == 1.0)

	return minimum_variance_n0, weights * minimum_variance_n0

def minimum_variance_n1(bins, N1_array, weights_for_MV, spectra_names, bin_function=None):
	'''
	Takes all N1 and form the mimimum variance estimator.
	Assumes N1 structure is coming from Biases_n1mat.f90
	Input:
		* N1: ndarray, contain the N1 (output of Biases_n1mat.f90)
		* weights_for_MV: ndarray, contain the weights used for MV
		* spectra_names: ndarray of string, contain the name of the spectra ordered
	'''
	## Ordering: i_TT=0,i_EE=1,i_EB=2,i_TE=3,i_TB=4, i_BB=5 (from Frotran)
	names_N1 = ['%s%s'%(i, j) for i in spectra_names for j in spectra_names]

	if bin_function is not None:
		n1_tot = np.zeros_like(bin_centers)
	else:
		n1_tot = np.zeros_like(weights_for_MV[0])

	for estimator_name in names_N1:
		## Indices for arrays
		index_x = spectra_names.index(estimator_name[0:2])
		index_y = spectra_names.index(estimator_name[2:])

		## Interpolate N1 if necessary
		n1_not_interp = N1_array[index_x][index_y]
		if bin_function is not None:
			n1_interp = np.interp(bin_centers, bins, n1_not_interp)
		else:
			n1_interp = n1_not_interp

		## Weights
		wXY_index = spectra_names.index(estimator_name[0:2])
		wZW_index = spectra_names.index(estimator_name[2:4])

		## Update N1
		if bin_function is not None:
			n1_tot += bin_function(weights_for_MV[wXY_index]) * bin_function(weights_for_MV[wZW_index]) * n1_interp
		else:
			n1_tot += weights_for_MV[wXY_index] * weights_for_MV[wZW_index] * n1_interp
	return n1_tot

