from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
from orphics import maps,io,cosmology,mpi
from pixell import enmap,lensing as plensing,curvedsky as cs, utils, enplot,bunch
import pytempura
import numpy as np
import os,sys
import healpy as hp
from enlib import bench
from falafel import qe
import os
import glob
import traceback,warnings
from . import bias
from falafel.utils import get_cmb_alm, get_kappa_alm, \
    get_theory_dicts, get_theory_dicts_white_noise, \
    change_alm_lmax

from falafel import utils as futils

config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
opath = config['data_path']

def four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0=None,Xdatp_1=None,Xdatp_2=None,Xdatp_3=None,q_func1=None):
    """Return kappa_alms combinations required for the 4cross estimator in Eq. 38 of arXiv:2011.02475v1 .

    Args:
        Xdat_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0
        Xdat_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1
        Xdat_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2
        Xdat_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3
        q_func1 (function): function for quadratic estimator
        Xdatp_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0 used for RDN0 for different sim data combination
        Xdatp_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1 used for RDN0 for different sim data combination
        Xdatp_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2 used for RDN0 for different sim data combination
        Xdatp_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3 used for RDN0 for different sim data combination
        qfunc2 ([type], optional): [description]. Defaults to None.

    Returns:
        array: Combination of reconstructed kappa alms
    """
    q_bh_1=q_func1
    if Xdatp_0 is None:
        print("none")
        
        phi_xy00 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_0))
        phi_xy11 = plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_1))
        phi_xy22 = plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_2))
        phi_xy33 = plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_3))
        phi_xy01 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_1))+plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_0)))
        phi_xy02 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_0)))
        phi_xy03 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_0)))
        phi_xy10=phi_xy01
        phi_xy12= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_1)))
        phi_xy13= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_1)))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_2)))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4
    
    else:
       
        phi_xy00 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_0))
        phi_xy11 = plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_1))
        phi_xy22 = plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_2))
        phi_xy33 = plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_3))
        phi_xy01 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_1))+plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_0)))
        phi_xy02 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_0)))
        phi_xy03 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_0)))
        phi_xy10=phi_xy01
        phi_xy12= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_1)))
        phi_xy13= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_1)))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_2)))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4

    phi_xy=np.array([phi_xy_X,phi_xy01,phi_xy02,phi_xy03,phi_xy12,phi_xy13,phi_xy23,phi_xy_x0,phi_xy_x1,phi_xy_x2,phi_xy_x3])
    

    return phi_xy

def split_phi_to_cl(xy,uv,m=4,cross=False,ikalm=None):
    phi_x=xy[0];phi01=xy[1];phi02=xy[2];phi03=xy[3];phi12=xy[4];phi13=xy[5];phi23=xy[6];phi_x0=xy[7];phi_x1=xy[8];phi_x2=xy[9];phi_x3=xy[10]
    phi_xp=uv[0];phi01p=uv[1];phi02p=uv[2];phi03p=uv[3];phi12p=uv[4];phi13p=uv[5];phi23p=uv[6];phi_x0p=uv[7];phi_x1p=uv[8];phi_x2p=uv[9];phi_x3p=uv[10]
    if cross is False:
        tg1=m**4*cs.alm2cl(phi_x,phi_xp)
        tg2=-4*m**2*(cs.alm2cl(phi_x0,phi_x0p)+cs.alm2cl(phi_x1,phi_x1p)+cs.alm2cl(phi_x2,phi_x2p)+cs.alm2cl(phi_x3,phi_x3p))
        tg3=4*(cs.alm2cl(phi01,phi01p)+cs.alm2cl(phi02,phi02p)+cs.alm2cl(phi03,phi03p)+cs.alm2cl(phi12,phi12p)+cs.alm2cl(phi13,phi13p)+cs.alm2cl(phi23,phi23p))
    else:
        tg1=m**4*cs.alm2cl(phi_x,ikalm)
        tg2=-4*m**2*(cs.alm2cl(phi_x0,ikalm)+cs.alm2cl(phi_x1,ikalm)+cs.alm2cl(phi_x2,ikalm)+cs.alm2cl(phi_x3,ikalm))
        tg3=4*(cs.alm2cl(phi01,ikalm)+cs.alm2cl(phi02,ikalm)+cs.alm2cl(phi03,ikalm)+cs.alm2cl(phi12,ikalm)+cs.alm2cl(phi13,ikalm)+cs.alm2cl(phi23,ikalm))

    auto =(1/(m*(m-1)*(m-2)*(m-3)))*(tg1+tg2+tg3)
    return auto


def get_sim_pixelization(lmax,is_healpix,verbose=False):
    # Geometry
    if is_healpix:
        nside = futils.closest_nside(lmax)
        shape = None ; wcs = None
        if verbose: print(f"NSIDE: {nside}")
    else:
        px_arcmin = 2.0  / (lmax / 3000)
        shape,wcs = enmap.fullsky_geometry(res=np.deg2rad(px_arcmin/60.),proj='car')
        nside = None
        if verbose: print(f"shape,wcs: {shape}, {wcs}")
    return qe.pixelization(shape=shape,wcs=wcs,nside=nside)


def get_tempura_norms(est1,est2,ucls,tcls,lmin,lmax,mlmax):
    """
    Get norms for lensing potential, sources and cross


    Parameters
    ----------


    est1 : str
        The name of a pre-defined falafel estimator. e.g. MV,MVPOL,TT,
        EB,TE,EE,TB.
    est2 : str
        The name of a pre-defined falafel estimator. e.g. MV,MVPOL,TT,
        EB,TE,EE,TB.

    ucls : dict
        A dictionary mapping TT,TE,EE,BB to spectra used in the response
        of various estimators. Typically these are gradient-field spectra
        or lensed field spectra.

    tcls : dict
        A dictionary mapping TT,TE,EE,BB to spectra used in the filtering
        of various estimators. Typically these are gradient-field spectra
        or lensed field spectra added to the noise power spectra.
    
    lmin: int
        Minumum CMB multipole 
    lmax: int
        Maximum CMB multipole
    mlmax : int
        Maximum multipole for alm transforms

    Returns
    -------
    bh: bool
        Specify whether Bias hardened norm is calculated
    ls: ndarray
        A (mlmax+1,) shape numpy array for the ell range of the normalization
    Als: dict
        Dictionary with key 'est1' and values correspond to a (2, mlmax+1) array in which the first component is the 
        normalization for the gradient and the second component the curl normalizaton of the corresponding estimator.
        if bh==True
        key 'src' access the source normalization.
    R_src_tt: ndarray
        A (mlmax+1,) shape numpy array containing the unnormalized cross response between est1 and point sources. None if bh==False
    Nl_g: ndarray
        A (mlmax+1,) shape numpy array for the convergence N0 bias for est1
    Nl_c: ndarray
        A (mlmax+1,) shape numpy array for the curl N0 bias for est1
    Nl_g_bh: ndarray
        A (mlmax+1,) shape numpy array for the N0 bias for the bias hardened estimator. None if bh==False
    
    """
    est_norm_list = [est1]
    if est2!=est1:
        est_norm_list.append(est2)
    bh = False
    for e in est_norm_list:
        if e.upper()=='TT' or e.upper()=='MV':
            bh = True
    if bh and est2=='SRC':
        est_norm_list.append('src')
        R_src_tt = pytempura.get_cross('SRC','TT',ucls,tcls,lmin,lmax,k_ellmax=mlmax)
    else:
        R_src_tt = None
    Als = pytempura.get_norms(est_norm_list,ucls,tcls,lmin,lmax,k_ellmax=mlmax)
    ls = np.arange(Als[est1][0].size)

    # Convert to noise per mode on lensing convergence
    diag = est1==est2 
    e1 = est1.upper()
    e2 = est2.upper()
    if diag:
        Nl_g = Als[e1][0] * (ls*(ls+1.)/2.)**2.
        Nl_c = Als[e1][1] * (ls*(ls+1.)/2.)**2.
        if bh and est2=='SRC':
            Nl_g_bh = bias_hardened_n0(Als[e1][0],Als['src'],R_src_tt) * (ls*(ls+1.)/2.)**2.
        else:
            Nl_g_bh = None
    else:
        assert ('MV' not in [e1,e2]) and ('MVPOL' not in [e1,e2])
        R_e1_e2 = pytempura.get_cross(e1,e2,ucls,tcls,lmin,lmax,k_ellmax=mlmax)
        Nl_phi_g = Als[e1][0]*Als[e2][0]*R_e1_e2[0]
        Nl_phi_c = Als[e1][1]*Als[e2][1]*R_e1_e2[1]
        Nl_g = Nl_phi_g * (ls*(ls+1.)/2.)**2.
        Nl_c = Nl_phi_c * (ls*(ls+1.)/2.)**2.
        if bh and est2=='SRC':
            Nl_g_bh = bias_hardened_n0(Nl_phi_g,Als['src'],R_src_tt) * (ls*(ls+1.)/2.)**2.
        else:
            Nl_g_bh = None
    return bh,ls,Als,R_src_tt,Nl_g,Nl_c,Nl_g_bh


def get_qfunc(px,ucls,mlmax,est1,Al1=None,est2=None,Al2=None,Al3=None,R12=None,profile=None):
    """
    Prepares a qfunc lambda function for an estimator est1. Optionally,
    normalize it with Al1. Optionally, bias harden it (which
    results in a normalized estimator) against est2 with
    normalization Al2 and unnormalized cross-response R12.


    Parameters
    ----------

    px : object
        A falafal.qe.pixelization object that holds healpix or rectangular
        pixel information and associated common functions
    ucls : dict
        A dictionary mapping TT,TE,EE,BB to spectra used in the response
        of various estimators. Typically these are gradient-field spectra
        or lensed field spectra.
    mlmax : int
        Maximum multipole for alm transforms
    est1 : str
        The name of a pre-defined falafel estimator. e.g. MV,MVPOL,TT,
        EB,TE,EE,TB.
    Al1 : ndarray
        A (2,mlmax) shape numpy array containing the gradient-like (e.g. lensing
        potential) and curl-like normalization.
    est2 : str, optional
        The name of a pre-defined falafel estimator to bias harden against
    Al2 : ndarray, optional
        A (mlmax,) shape numpy array containing the normalization of the 
        estimator being hardened against.
    Al3 : ndarray, optional
        A (mlmax,) shape numpy array containing the normalization of the 
        TT estimator used when calculating BH estimator.
    R12 : ndarray, optional
        An (mlmax,) or (1,mlmax) or (2,mlmax) shape numpy array containing 
        the unnormalized cross-response of est1 and est2. If two components
        are present, then the curl of est1 is also bias hardened using the
        cross-response of est2 with curl specified through the second
        component.
    profile : (mlmax) array, default=None
        An array to use as the profile for profile-hardening, when est2="SRC".
        If not provided, will just do point-source hardening. 

    Returns
    -------
    qfunc : function
        Quadratic estimator lambda function
    
    """
    est1 = est1.upper()
    print(pytempura.est_list)
    print(est1)
    assert est1 in pytempura.est_list
    if Al1 is not None:
        assert Al1.ndim==2, "Both gradient and curl normalizations need to be present."
    if est2 is not None:
        bh = True
        assert est2 in pytempura.est_list
        assert Al1 is not None
        assert Al2 is not None
        if Al2.ndim==2:
            assert Al2.shape[0]==1
            Al2 = Al2[0]
        else:
            assert Al2.ndim==1
        assert R12 is not None
        if R12.ndim==1: 
            R12 = R12[None]
        else: 
            assert R12.ndim==2
    else:
        bh = False

    assert est1 in ['TT','TE','EE','EB','TB','MV','MVPOL','SHEAR'] # TODO: add other
    if est1=='SHEAR':
        qfunc1 = lambda X,Y: qe.qe_shear(px,mlmax,
                            Talm=X[0],fTalm=Y[1])
    else:
        qfunc1 = lambda X,Y: qe.qe_all(px,ucls,mlmax,
                                    fTalm=Y[0],fEalm=Y[1],fBalm=Y[2],
                                    estimators=[est1],
                                    xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])[est1]

    if bh:
        assert est2 in ['SRC','MASK'] # TODO: add mask
        if est2 == 'SRC':
            qfunc2 = lambda X,Y: qe.qe_source(px,mlmax,Y[0],profile=profile,xfTalm=X[0])
        elif est2 == 'mask':
            qfunc2 = lambda X,Y: qe.qe_mask(px,ucls,mlmax,fTalm=Y[0],xfTalm=X[0])
        # The bias-hardened estimator Eq 27 of arxiv:1209.0091
        if R12.shape[0]==1:

            if est1=='TT':
                # Bias harden only gradient e.g. source hardening
                def retfunc(X,Y):
                    q1 = qfunc1(X,Y)
                    q2 = qfunc2(X,Y)
                    g = cs.almxfl( \
                                (cs.almxfl(q1[0],Al1[0]) - \
                                    cs.almxfl(qfunc2(X,Y),Al1[0] * Al2 * R12[0])) , \
                                1. / (1. - Al1[0] * Al2 * R12[0]**2.) \
                    )
                    c = cs.almxfl(q1[1],Al1[1])
                    return np.asarray((g,c))
            else:
                def retfunc(X,Y):
                    print('test bh MV')
                    qfuncTT= lambda X,Y: qe.qe_all(px,ucls,mlmax,
                                        fTalm=Y[0],fEalm=Y[1],fBalm=Y[2],
                                        estimators=['TT'],
                                        xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])['TT']
                    q1=qfuncTT(X,Y)

                    q2 = qfunc2(X,Y)

                    qfuncmv=lambda X,Y: qe.qe_all(px,ucls,mlmax,
                                        fTalm=Y[0],fEalm=Y[1],fBalm=Y[2],
                                        estimators=['MV'],
                                        xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])['MV']
                    
                    qmv=qfuncmv(X,Y)
                    g_bh_TT = cs.almxfl( \
                                (cs.almxfl(q1[0],Al3[0]) - \
                                    cs.almxfl(qfunc2(X,Y),Al3[0] * Al2 * R12[0])) , \
                                1. / (1. - Al3[0] * Al2 * R12[0]**2.) \
                    )
                    g= cs.almxfl(qmv[0]-q1[0]+cs.almxfl(g_bh_TT,1/Al3[0]),Al1[0])
                    c = cs.almxfl(qmv[1],Al1[1])


                    return np.asarray((g,c))

        elif R12.shape[0]==2:
            # Bias harden both e.g. mask hardening
            def retfunc(X,Y):
                q1 = qfunc1(X,Y)
                q2 = qfunc2(X,Y)
                g = cs.almxfl( \
                               (cs.almxfl(q1[0],Al1[0]) - \
                                cs.almxfl(qfunc2(X,Y),Al1[0] * Al2 * R12[0])) , \
                               1. / (1. - Al1[0] * Al2 * R12[0]**2.) \
                )
                c = cs.almxfl( \
                               (cs.almxfl(q1[1],Al1[1]) - \
                                cs.almxfl(qfunc2(X,Y),Al1[1] * Al2 * R12[1])) , \
                               1. / (1. - Al1[1] * Al2 * R12[1]**2.) \
                )
                return np.asarray((g,c))

        return retfunc
                
    else:
        if Al1 is not None: 
            # TODO: Improve this construct by building a multi-dimensional almxfl
            def retfunc(X,Y):
                recon = qfunc1(X,Y)
                return np.asarray((cs.almxfl(recon[0],Al1[0]),cs.almxfl(recon[1],Al1[1])))
            return retfunc
        else: return qfunc1



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
            
            afname = f'{opath}/car_mask_lmax_{lmax}_apodized_{car_deg:.1f}_deg.fits'
            mask = enmap.read_map(afname)[0]
    return mask

def initialize_args(args):
    from mapsims import noise,Channel,SOStandalonePrecomputedCMB
    import mapsims
    # Lensing reconstruction ell range
    # We don't need to redefine all these variables!
    # just use the args.lmin etc. below instead of lmin
    lmin = args.lmin
    lmax = args.lmax
    use_cached_norm = args.use_cached_norm
    use_cmblensplus = not(args.flat_sky_norm)
    disable_noise = args.disable_noise
    debug_cmb = args.debug
    
    wnoise = args.wnoise
    beam = args.beam
    atmosphere = not(args.no_atmosphere)
    polcomb = args.polcomb

    # Number of sims
    nsims = args.nsims
    sindex = args.sindex
    comm,rank,my_tasks = mpi.distribute(nsims)

    isostr = "isotropic_" if args.isotropic else "classical"

    config = io.config_from_yaml(os.path.dirname(os.path.abspath(__file__)) + "/../input/config.yml")
    opath = config['data_path']

    mask = get_mask(healpix=args.healpix,lmax=lmax,no_mask=args.no_mask,car_deg=2,hp_deg=4)

    # Initialize the lens simulation interface
    solint = SOLensInterface(
        mask=mask, data_mode=None,
        scanning_strategy="isotropic" if args.isotropic else "classical",
        fsky=0.4 if args.isotropic else None,
        white_noise=wnoise, beam_fwhm=beam,
        disable_noise=disable_noise,
        atmosphere=atmosphere,zero_sim=args.zero_sim)
    if rank==0: solint.plot(mask,f'{opath}/{args.label}_{args.polcomb}_{isostr}mask')
    
    # Choose the frequency channel
    channel = mapsims.SOChannel("LA", 145)

    # norm dict
    Als,Als_curl = solint.initialize_norm(channel,lmin,lmax,
                                 recalculate=not(use_cached_norm),
                                 use_cmblensplus=use_cmblensplus,label=args.label)
    Nl = Als[polcomb]*Als['L']*(Als['L']+1.)/4.
    return solint,Als,Als_curl,Nl,comm,rank,my_tasks,sindex,debug_cmb,lmin,lmax,polcomb,nsims,channel,isostr

def convert_seeds(seed,nsims=2000,ndiv=4):
    # Convert the solenspipe convention to the Alex convention
    icov,cmb_set,i = seed
    assert icov==0, "Covariance from sims not yet supported."
    nstep = nsims//ndiv
    if i>=nstep: 
        warnings.warn("i>=nstep: If more than one CMB set is being used (e.g for RDN0 and MCN1), you might be re-using sims.")
    if cmb_set==0 or cmb_set==1:
        s_i = i + cmb_set*nstep
        s_set = 0
        noise_seed = (icov,cmb_set,i)+(2,)
    elif cmb_set==2 or cmb_set==3:
        s_i = i + nstep*2
        s_set = cmb_set - 2
        noise_seed = (icov,cmb_set,i)+(2,)

    return s_i,s_set,noise_seed



def wfactor(*args, **kwargs):
    warnings.warn("wfactor should be called directly from orphics.maps")
    return maps.wfactor(*args, **kwargs)

class SOLensInterface(object):
    def __init__(self,mask,data_mode=None,scanning_strategy="isotropic",fsky=0.4,white_noise=None,beam_fwhm=None,disable_noise=False,atmosphere=True,rolloff_ell=50,zero_sim=False):

        from mapsims import noise,Channel,SOStandalonePrecomputedCMB
        import mapsims
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
            return cs.alm2map(alm,enmap.empty((ncomp,)+self.shape,self.wcs))
        
    def map2alm(self,imap):
        if self.healpix:
            return hp.map2alm(imap,lmax=self.mlmax,iter=0)
        else:
            return cs.map2alm(imap,lmax=self.mlmax)


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
        cmb_alm = cs.almxfl(cmb_alm,lambda x: maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else cmb_alm
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
            #imap=noise_map
            imap = imap * self.mask

            if self._debug:
                for i in range(3): self.plot(imap[i],f'imap_{i}')
                for i in range(3): self.plot(noise_map[i],f'nmap_{i}',range=300)

            # Map -> alms, and deconvolve the beam
            oalms = self.map2alm(imap)


            oalms = cs.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms
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
            oalms = cs.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms
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
            oalms = cs.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms

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
        oalms = cs.almxfl(oalms,lambda x: 1./maps.gauss_beam(self.beam,x)) if not(self.disable_noise) else oalms
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

    def qfunc_bh(self,alpha,X,Y,ils,blens,bhps,Alpp,A_ps):
        """
        wrapper to compute normalized bias hardened temperature convergent alms. (See Eq. 27 of https://arxiv.org/pdf/1209.0091.pdf)

        Parameters
        ----------

            ils: lensing multipoles L, used for the conversion from phi_alms to kappa_alms
            blens: lensing response function
            bhps: point source response function
            Alpp: Lensing Phi Normalization 
            A_ps: Point source Normalization

        Returns
        -------
        recon_alms : ndarray
            A (mlmax,) shape numpy array containing the normalised BH quadratic estimator
     
        """

        
  
        
        # Frank: Some of the bias hardening normalization code requires
        # using functions from cmblensplus from Toshiya. The Tcmb factor
        # here is to address the fact in Toshiya's code the temperature is
        # dimensionless whereas in solenspipe it is in microKelvins. So we
        # hard code the conversion here since it shouldn't change.
        Tcmb = 2.726e6
        polcomb=alpha #Only TT used for bias hardening
        #point source reconstruction
        source=qe.qe_pointsources(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])
        #lensing reconstruction
        phi=qe.qe_all(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])[polcomb][0]
        #normalise the point sources alms
        s_alms=qe.filter_alms(source,maps.interp(ils,A_ps*bhps*Tcmb**2))
        #normalised lensing phi_alms
        phi_alms = qe.filter_alms(phi,maps.interp(ils,2*Alpp*blens))
        #bias hardened alms
        balms=phi_alms-s_alms
        recon_alms=hp.almxfl(balms,ils*(ils+1)*0.5)
        return recon_alms

    def get_mv_curl(self,polcomb,talm,ealm,balm):
        return self.qfunc_curl(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfunc_curl(self,alpha,X,Y):
        """
         Wrapper for the core falafel full-sky curl reconstruction function
        Calculates the unnormalised curl estimator. 
        By construction, the lensing field is given by the gradient of the deflection field which is irrotational. 
        Systematics mimicking lensing need not obey this symmetry and give a non zero curl."""
        polcomb = alpha
        return qe.qe_all(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])[polcomb][1]

    def get_mv_mask(self,polcomb,talm,ealm,balm):
        
        return self.qfuncmask(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfuncmask(self,alpha,X,Y):
        """Wrapper for the analysis mask reconstruction based on Eq 22 of https://arxiv.org/pdf/1209.0091.pdf"""
        polcomb = alpha
        return qe.qe_mask(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])
    
    def get_pointsources(self,polcomb,talm,ealm,balm):
        return self.qfunc_ps(polcomb,[talm,ealm,balm],[talm,ealm,balm])

    def qfunc_ps(self,alpha,X,Y):
        """Wrapper for point source reconstruction from Falafel"""
        polcomb = alpha
        return qe.qe_pointsources(self.px,lambda x,y: self.theory.lCl(x,y),lambda x,y:self.theory_cross.lCl(x,y),
                         self.mlmax,Y[0],Y[1],Y[2],estimators=[polcomb],
                         xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])

    def qfuncshear(self,Talm,fTalm):
        """Wrapper for full sky shear reconstruction from Falafel, the shear estimator is a foreground immune estimator."""
        return qe.qe_shear(self.px,self.mlmax,Talm=Talm,fTalm=fTalm)

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
    

    def initialize_norm(self, channel, lmin, lmax, recalculate=False,
                        use_cmblensplus=True, label=None):
        """
        Calculate the normalization factors A(L) to normalize
        the kappa a_lms (see e.g. https://arxiv.org/abs/astro-ph/0111606).
        Return a recarray of A(L) values, as well as writing these
        to file for subsequent use.
        
        Parameters
        ----------
        channel: mapsims.Channel instance
            the channel to calculate the A(L)s for
        lmin: int
            minimum l in reconstruced kappa
        lmax: int
            maxmimum l in rconstructed kappa
        recalculate: bool
            if True, perform the calculation, otherwise
        use read from file.
        cmblensplus: bool (default=True)
            if True, use Cmblensplus/Tempura to do the calculation,
        otherwise use qe.symlens_norm
        label: str
            An identifer string used in the output text file,
        (or in the input filename if recalculate is False)

        Returns
        -------
        Als: numpy recarray
            recarray of norm info, (including the L
        values). Columns are accessible by name e.g.
        Als["L"] or Als["TT"]
        Als_curl: numpy recarray or None
            same as Als but for the curl. In the case
        that no cached file is found, or we do the 
        calculation with qe.symlens_norm, we return
        None instead.
        """

        #Build the filenames.
        lstr = "" if label is None else f"{label}_"
        wstr = "" if self.wnoise is None else "wnoise_"
        als_fname = opath+"als_%s%slmin_%d_lmax_%d.txt" % (wstr,lstr,lmin,lmax)
        als_curl_fname = opath+"als_curl_%s%slmin_%d_lmax_%d.txt" % (wstr,lstr,lmin,lmax)

        #stuff for reading/writing the cached files. Assuming for now
        #they should be human readable, if not - we could simplify
        #these reading and writing steps quite a lot by using
        #.yaml or .npy files.
        AL_data_dtype = [("L",int), ("TT",float), ("TE",float), ("EE",float),
                         ("TB",float), ("EB",float), ("mv",float),
                         ("mv_pol",float), ("TE_hdv",float)]
        AL_data_names = [d[0] for d in AL_data_dtype]
        #format for savetxt - all floats execept L column
        savetxt_fmt = ["%.18e" if d[1]==float else "%d" for d in AL_data_dtype]
        def write_als(Als, filename):
            output_data = np.zeros(len(Als['L']), dtype=AL_data_dtype)
            for d in AL_data_dtype:
                output_data[d[0]] = Als[d[0]]
            np.savetxt(filename, output_data, fmt=savetxt_fmt,
                       header='#'+" ".join(AL_data_names)
                       )
        def read_als(filename):
            AL_data = np.genfromtxt(filename, names=True)
            #Make sure we have the correct columns
            assert list(AL_data.dtype.names) == AL_data_names
            return AL_data

        #If not calculating, just read in from files
        #and return the arrays. For now, we do not throw
        #an error if the curl file is not found.
        if not recalculate:
            print("reading A(L)s from %s, %s"%(als_fname, als_curl_fname))
            try:
                als_data = read_als(als_fname)
            except IOError as e:
                print("A(L)s file %s not found"%als_fname)
                raise(e)
            try:
                als_curl_data = read_als(als_curl_fname)
            except IOError as e:
                print("A(L)s file for curl %s not found"%als_curl_fname)
                print("Continuing for now in case you didn't need curl anyway")
                als_curl_data = None
                pass
            return als_data, als_curl_data        

        else:
            #In this case we do the calculation
            #First read in the theory
            thloc = (os.path.dirname(os.path.abspath(__file__))
                     + "/../data/" + config['theory_root'])
            theory = cosmology.loadTheorySpectraFromCAMB(
                thloc,get_dimensionless=False)
            ells,gt = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",
                                 unpack=True,usecols=[0,1])
            class T:
                def __init__(self):
                    self.lCl = lambda p,x: maps.interp(ells,gt)(x)
            theory_cross = T()
            ls,nells,nells_P = self.get_noise_power(channel,
                                                    beam_deconv=True)
            cltt=theory.lCl('TT',ls)+nells
            Als = np.zeros(lmax+1, dtype=AL_data_dtype)
            Als_curl = np.zeros_like(Als)
            if use_cmblensplus:
                ls, Ag, Ac = cmblensplus_norm(
                    nells, nells_P, nells_P, theory,
                    theory_cross, lmin,lmax)
                def fill_array(L, Ain, Aout):
                    Aout['L'] = L
                    Aout['TT'], Aout['TE'], Aout['EE'], Aout['TB'], Aout['EB'] = (
                        Ag[0], Ag[1], Ag[2], Ag[3], Ag[4])
                    Als['mv'] = 1/(1/Als['EB']+1/Als['TB']+
                              1/Als['EE']+1/Als['TE']+1/Als['TT'])
                    Als['mv_pol'] = 1/(1/Als['EB']+1/Als['TB'])
                    Als['TE_hdv'] = 0.
                fill_array(ls, Ag, Als)
                fill_array(ls, Ac, Als_curl)
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
                ls,als,al_mv_pol,al_mv,Al_te_hdv = qe.symlens_norm(
                    uctt,tctt,ucee,tcee,ucte,tcte,ucbb,tcbb,lmin=lmin,
                    lmax=lmax,plot=False)
                Als = np.zeros(len(ls), dtype=AL_data_dtype)
                Als['L'] = ls
                for key in ['TT','EE','EB','TE','TB']:
                    Als[key] = als[key][:lmax+1]
                Als['mv_pol'] = al_mv_pol[:lmax+1]
                Als['mv'] = al_mv[:lmax+1]
                Als['TE_hdv'] = Al_te_hdv[:lmax+1]
                #qe.symlens_norm doesn't do the curl so set this to None
                Als_curl = None

            #Now write the files which can be used in subsequent
            #calls with recalculate=False
            write_als(Als, als_fname)
            if Als_curl is not None:
                write_als(Als_curl, als_curl_fname)

            return Als, Als_curl
        
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
    print('compute norm from Tempura')
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
    Ag, Ac, Wg, Wc = pytempura.norm_lens.qall(QDO,lmax,rlmin,rlmax,lcl,ocl)
    fac=ls*(ls+1)
    return ls,Ag*fac,Ac*fac

def diagonal_RDN0(get_sim_power,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax,simn):
    """Curvedsky dumb N0 for TT,EE,EB,TE,TB"""
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
    AgTT,AcTT=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],ocl[0,:])
    AgTE,AcTE=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:])
    AgTB,AcTB=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:])
    AgEE,AcEE=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:])
    AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:])

    fac=ls*(ls+1)
    #prepare the sim total power spectrum
    cldata=get_sim_power((0,0,simn))
    sim_ocl=np.array([cldata[0][:ls.size],cldata[1][:ls.size],cldata[2][:ls.size],cldata[3][:ls.size]])/Tcmb**2
    #dataxdata
    cl=ocl**2/(sim_ocl)
    AgTT0,AcTT0=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:])
    AgTE0,AcTE0=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
    AgTB0,AcTB0=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:])
    AgEE0,AcEE0=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:])
    AgEB0,AcEB0=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
    #(data-sim) x (data-sim)
    cl=ocl**2/(ocl-sim_ocl)
    AgTT1,AcTT1=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:])
    AgTE1,AcTE1=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:])
    AgTB1,AcTB1=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:])
    AgEE1,AcEE1=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:])
    AgEB1,AcEB1=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:])
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
    AgTT,AcTT=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],ocl[0,:] ,gtype='k')
    AgTE,AcTE=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[1,:] ,gtype='k')
    AgTB,AcTB=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],ocl[0,:],ocl[2,:], gtype='k')
    AgEE,AcEE=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:] ,gtype='k')
    AgEB,AcEB=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],ocl[1,:],ocl[2,:], gtype='k')

    #prepare the sim total power spectrum
    cldata=get_sim_power((0,0,simn))
    sim_ocl=np.array([cldata[0][:ls.size],cldata[1][:ls.size],cldata[2][:ls.size],cldata[3][:ls.size]])/Tcmb**2
    #dataxdata
    sim_ocl[np.where(sim_ocl==0)] = 1e30
    cl=ocl**2/(sim_ocl)
    AgTT0,AcTT0=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:] ,gtype='k')
    AgTE0,AcTE0=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:], gtype='k')
    AgTB0,AcTB0=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:], gtype='k')
    AgEE0,AcEE0=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:] ,gtype='k')
    AgEB0,AcEB0=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:], gtype='k')
    ATTTE0,__=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:], cl[0,:], ocl[1,:]*sim_ocl[0,:]/ocl[0,:],sim_ocl[3,:],gtype='k')
    ATTEE0,__=pytempura.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], sim_ocl[3,:], gtype='k')
    ATEEE0,__=pytempura.norm_lens.qteee(lmax, rlmin, rlmax, lcl[1,:], lcl[3,:], ocl[0,:]*sim_ocl[1,:]/ocl[1,:], cl[1,:], sim_ocl[3,:], gtype='k')
    ATBEB0,__=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], sim_ocl[3,:], gtype='k')


    #(data-sim) x (data-sim)
    cl=ocl**2/(ocl-sim_ocl)
    AgTT1,AcTT1=pytempura.norm_lens.qtt(lmax, rlmin, rlmax, lcl[0,:],cl[0,:] ,gtype='k')
    AgTE1,AcTE1=pytempura.norm_lens.qte(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[1,:],gtype='k')
    AgTB1,AcTB1=pytempura.norm_lens.qtb(lmax, rlmin, rlmax, lcl[3,:],cl[0,:],cl[2,:],gtype='k')
    AgEE1,AcEE1=pytempura.norm_lens.qee(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],gtype='k')
    AgEB1,AcEB1=pytempura.norm_lens.qeb(lmax, rlmin, rlmax, lcl[1,:],cl[1,:],cl[2,:],gtype='k')
    ATTTE1,__=pytempura.norm_lens.qttte(lmax, rlmin, rlmax, lcl[0,:], lcl[3,:],cl[0,:] ,(1-sim_ocl[0,:]/ocl[0,:])*ocl[1,:] , ocl[3,:]-sim_ocl[3,:],gtype='k')
    ATTEE1,__=pytempura.norm_lens.qttee(lmax, rlmin, rlmax, lcl[0,:], lcl[1,:], cl[0,:], cl[1,:], ocl[3,:]-sim_ocl[3,:],gtype='k')
    ATEEE1,__=pytempura.norm_lens.qteee(lmax, rlmin, rlmax, lcl[1,:], lcl[3,:], (1-sim_ocl[1,:]/ocl[1,:])*ocl[0,:],cl[1,:],ocl[3,:]-sim_ocl[3,:],gtype='k')
    ATBEB1,__=pytempura.norm_lens.qtbeb(lmax, rlmin, rlmax, lcl[1,:], lcl[2,:], lcl[3,:], cl[0,:], cl[1,:], cl[2,:], ocl[3,:]-sim_ocl[3,:],gtype='k')


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
    A_mask = tempura.norm_tau.qtt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:])
    Alpp, __ = tempura.norm_lens.qtt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:])
    Rlpt = tempura.norm_lens.ttt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:]) #this is unnormalized
    fac=ls*(ls+1)*0.5
    detR=1-Alpp*A_mask*Rlpt**2
    bhmask=Alpp*Rlpt/detR
    bhp=1/detR
    bhclkknorm=fac**2*Alpp/detR
    return ls,bhp,bhmask,Alpp,A_mask,bhclkknorm

def bias_hard_ps_norms(nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """Normalizations for point source reconstruction"""
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
    A_ps = tempura.norm_src.qtt(lmax,rlmin,rlmax,ocl[0,:])
    Alpp, __ = tempura.norm_lens.qtt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:])
    Rlps = tempura.norm_lens.stt(lmax,rlmin,rlmax,lcl[0,:],ocl[0,:]) #this is unnormalized
    fac=ls*(ls+1)*0.5
    detR=1-Alpp*A_ps*Rlps**2
    bhps=Alpp*Rlps/detR
    bhp=1/detR
    bhclkknorm=fac**2*Alpp/detR

    return ls,bhp,bhps,Alpp,A_ps,bhclkknorm   
        

def cmblensplusreconstruction(solint,w2,w3,w4,nltt,nlee,nlbb,theory,theory_cross,lmin,lmax):
    """example of reconstruction using Toshiya's cmblensplus pipeline"""
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
    Ag, Ac, Wg, Wc = tempura.norm_lens.qall(QDO,lmax,rlmin,rlmax,lcl,ocl) #the Als used (same norm as used for theory) TT,TE,EE,TB,EB
    #load the beam deconvolved alms
    sTalm=hp.fitsfunc.read_alm(config['data_path']+'pipetest/talms.fits')
    sEalm=hp.fitsfunc.read_alm(config['data_path']+'pipetest/ealms.fits')
    sBalm=hp.fitsfunc.read_alm(config['data_path']+'pipetest/balms.fits')
    mlm=int(0.5*(-3+np.sqrt(9-8*(1-len(sTalm)))))
    sTalm = tempura.utils.lm_healpy2healpix(len(sTalm), sTalm, mlm) 
    sEalm = tempura.utils.lm_healpy2healpix(len(sEalm), sEalm, mlm) 
    sBalm = tempura.utils.lm_healpy2healpix(len(sBalm), sBalm, mlm) 
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
    glm['TT'], clm['TT'] = tempura.rec_lens.qtt(lmax,rlmin,rlmax,lcl[0,:],Talm,Talm)
    glm['TE'], clm['TE'] = tempura.rec_lens.qte(lmax,rlmin,rlmax,lcl[3,:],Talm,Ealm)
    glm['EE'], clm['EE'] = tempura.rec_lens.qee(lmax,rlmin,rlmax,lcl[1,:],Ealm,Ealm)
    glm['TB'], clm['TB'] = tempura.rec_lens.qtb(lmax,rlmin,rlmax,lcl[3,:],Talm,Balm)
    glm['EB'], clm['EB'] = tempura.rec_lens.qeb(lmax,rlmin,rlmax,lcl[1,:],Ealm,Balm)
    
    
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
    kalms=pytempura.utils.hp_map2alm(2048, rlmax, mlmax, phimap)
    kalms = solint.get_kappa_alm(0+0)
    lm=int(0.5*(-3+np.sqrt(9-8*(1-len(kalms)))))
    kalms = tempura.utils.lm_healpy2healpix(len(kalms), kalms, lm) 
    kalms=kalms[:lmax+1,:lmax+1]
    macl=np.zeros(rlmax+1)
    micl=np.zeros(rlmax+1)
    mxcl=np.zeros(rlmax+1)
    micl+=pytempura.utils.alm2cl(rlmax,kalms,kalms)/w2
    acl=pytempura.utils.alm2cl(rlmax,glm[polcomb],glm[polcomb])/w4
    macl+= fac**2*acl
    xcl=pytempura.utils.alm2cl(rlmax,glm[polcomb],kalms)/w3
    mxcl+=xcl*fac
    normMV=Ag[5]*fac**2

def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return cents,bls



def error_f(f_sky,n0,n1,clkk,bin_edges):
    """
    input: f_sky
    n0: unbinned numpy array
    n1:unbinned numpy array
    clkk: unbinned numpy array
    binned theory error
    
    """
    error=n0+clkk+n1
    cents,errorb=bandedcls(error,bin_edges)
    cov=np.ones(len(errorb))
    for i in range(len(errorb)):
        cov[i]=(1/(cents[i]*np.diff(bin_edges)[i]*f_sky))*(errorb[i])**2
    return np.sqrt(cov)


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
            print(np.nansum(weights[self.bin_edges[i-1]:self.bin_edges[i]]*iy[self.bin_edges[i-1]:self.bin_edges[i]]))
            bin_means.append(np.nansum(weights[self.bin_edges[i-1]:self.bin_edges[i]+1]*iy[self.bin_edges[i-1]:self.bin_edges[i]+1])/np.nansum(weights[self.bin_edges[i-1]:self.bin_edges[i]+1]))
            print(bin_means)
        bin_means=np.array(bin_means)
        return self.cents,bin_means
        
    def binning_matrix(self,ix,iy,weights):
        #return the binning matrix used for the data product ix,iy are length of the array we want to bin
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
            col[self.bin_edges[i-1]:self.bin_edges[i]+1]=weights[self.bin_edges[i-1]:self.bin_edges[i]+1]/np.sum(weights[self.bin_edges[i-1]:self.bin_edges[i]+1])
            matrix.append(col)
        matrix=np.array(matrix)
        return matrix 
        
        
def bias_hardened_n0(Nl,Nlbias,Cross):
    ret = Nl*0
    ret[1:] = Nl[1:] / (1.-Nl[1:]*Nlbias[1:]*Cross[1:]**2.)
    return ret


def get_labels():
    labs = bunch.Bunch()
    labs.clii = r'$C_L^{\kappa \kappa}$'
    labs.n1 = r'$N_L^{1,\kappa\kappa}$'
    labs.nlg = r'$N_L^{0,\kappa\kappa}$'
    labs.nlc = r'$N_L^{0,\omega\omega}$'
    labs.tclng = r'$C_L^{\kappa \kappa} + N_L^{0,\kappa\kappa}$'
    labs.rdn0g = r'$RDN_L^{0,\kappa\kappa}$'
    labs.rdn0c = r'$RDN_L^{0,\omega\omega}$'
    labs.mcn1g = r'$MCN_L^{1,\kappa\kappa}$'
    labs.mcn1c = r'$MCN_L^{1,\omega\omega}$'
    labs.tcrdn0g = r'$C_L^{\kappa \kappa} + RDN_L^{0,\kappa\kappa}$'
    labs.xcl = r'$C_L^{\hat{\kappa} \kappa}$'
    labs.acl = r'$C_L^{\hat{\kappa} \hat{\kappa}}$'
    return labs


# Load signal map, apply beam and add noise
class LensingSandbox(object):
    def __init__(self,fwhm_arcmin,noise_uk,dec_min,dec_max,res, # simulation
                 lmin,lmax,mlmax,ests, # reconstruction
                 include_te = False, # whether to include TE correlations
                 add_noise = False, mask = None,
                 verbose = False):  # whether to add noise (it will still be in the filters)
        self.fwhm = fwhm_arcmin
        self.noise = noise_uk
        self.no_te_corr = not include_te
        # Specify geometry
        if mask is None:
            if (dec_min is None) and (dec_max is None):
                self.shape,self.wcs = enmap.fullsky_geometry(res=res * utils.arcmin,variant='fejer1')
            else:
                if dec_min is None: dec_min = -90.
                if dec_max is None: dec_max = 90. 
                self.shape,self.wcs = enmap.band_geometry((dec_min * utils.degree, dec_max * utils.degree),res=res * utils.arcmin, variant='fejer1')
            mask = enmap.ones(self.shape,self.wcs)
        else:
            self.shape = mask.shape
            self.wcs = mask.wcs

        self.w2 = maps.wfactor(2,mask)
        self.w3 = maps.wfactor(3,mask)
        self.w4 = maps.wfactor(4,mask)
        if verbose:
            print(f"W2 factor: {self.w2:.5f}")
            print(f"W3 factor: {self.w3:.5f}")
            print(f"W4 factor: {self.w4:.5f}")

        self.ucls,self.tcls = futils.get_theory_dicts_white_noise(self.fwhm,self.noise,
                                                                  grad=True,lmax=mlmax)
        self.Als = pytempura.get_norms(ests, self.ucls, self.ucls, self.tcls, lmin, lmax,
                                       no_corr=self.no_te_corr)
        ls = np.arange(self.Als[ests[0]][0].size)
        self.Nls = {}
        px = qe.pixelization(self.shape,self.wcs)
        self.qfuncs = {}
        for est in ests:
            self.qfuncs[est] =  get_qfunc(px,self.ucls,mlmax,est,Al1=self.Als[est])
            self.Nls[est] = self.Als[est][0] * (ls*(ls+1.)/2.)**2.

        self.ests = ests
        self.mlmax = mlmax
        self.lmin = lmin
        self.lmax = lmax
        self.mask = mask
        self.add_noise = add_noise

    def get_observed_map(self,index,iset=0):
        shape,wcs = self.shape,self.wcs
        calm = futils.get_cmb_alm(index,iset)
        calm = cs.almxfl(calm,lambda x: maps.gauss_beam(x,self.fwhm))
        # ignoring pixel window function here
        omap = cs.alm2map(calm,enmap.empty((3,)+shape,wcs,dtype=np.float32),spin=[0,2])
        if self.add_noise:
            nmap = maps.white_noise((3,)+shape,wcs,self.noise)
            nmap[1:] *= np.sqrt(2.)
        else:
            nmap = 0.
        return (omap + nmap) * self.mask

    def kmap(self,stuple):
        icov,ip,i = stuple
        nstep = 500
        if i>nstep: raise ValueError
        if ip==0 or ip==1:
            iset = 0
            index = nstep*ip + i
        elif ip==2 or ip==3:
            iset = ip - 2
            index = 1000 + i
        dmap = self.get_observed_map(index,iset)
        X = self.prepare(dmap)
        return X

    def prepare(self,omap):
        alm = cs.map2alm(omap,lmax=self.mlmax,spin=[0,2])
        with np.errstate(divide='ignore', invalid='ignore'):
            alm = cs.almxfl(alm,lambda x: 1./maps.gauss_beam(x,self.fwhm))
        ftalm,fealm,fbalm = futils.isotropic_filter(alm,self.tcls,self.lmin,
                                                    self.lmax,ignore_te=self.no_te_corr)
        return [ftalm,fealm,fbalm]
        
    def reconstruct(self,omap,est):
        # You can derive from this class and overload this function with your own
        
        # e.g. do map-level pre-processing here
        # do coadding here
        # do optimal filtering here
        X = self.prepare(omap)
        return self.qfuncs[est](X,X)

    
    def get_rdn0(self,prepared_data_alms,est,nsims,comm):
        Xdata = prepared_data_alms
        return bias.simple_rdn0(0,est,est,lambda alpha,X,Y: self.qfuncs[alpha](X,Y),self.kmap,comm,cs.alm2cl,nsims,Xdata)

    def get_mcn1(self,est,nsims,comm):
        return bias.mcn1(0,self.kmap,cs.alm2cl,nsims,self.qfuncs[est],comm=comm,verbose=True).mean(axis=0)

    def get_mcmf_twosets(self,est,nsims,comm):
        return bias.mcmf_twosets(0,self.qfuncs[est],self.kmap,comm,nsims)
    
    def get_mcmf(self,est,nsims,comm):
        return bias.mcmf_pair(0,self.qfuncs[est],self.kmap,comm,nsims)
