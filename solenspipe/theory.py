import symlens

"""
Functions that use theoretical flat-sky expressions
for lensing predictions.

CMB theory is passed around using a dict with the following
convention.
theory['ells']
theory['uCl_TT']
theory['uCl_TE']
theory['uCl_EE']
theory['lCl_TT']
theory['lCl_TE']
theory['lCl_EE']
theory['lCl_BB']
theory['gCl_TT'] = C_gradT_T
theory['dCl_TT'] = d ln uC_TT / d ln l


"""


"""
N0 library

bias.py contains
1. RDN0, which combines data and simulation through raw 4-point evaluations.

Here we provide approximations based on the theoretical expression:

$$
N^0_{XYUV} = \frac{1}{4}A_{XY}A_{UV}\int \frac{\dl_1 }{ (2\pi)^2 }  F_{XY}(\bl_1,\bl_2) \
(F_{UV}(\bl_1,\bl_2) D_1({\ell_1},{\ell_2}) + F_{UV}(\bl_1,\bl_2) D_2({\ell_1},{\ell_2}) )
$$

2. Theory N0 for the co-add estimator
D_1({\ell_1},{\ell_2}) = C^{XU}_{\ell_1} C^{YV}_{\ell_2}
D_2({\ell_1},{\ell_2}) = C^{XV}_{\ell_1} C^{YU}_{\ell_2}

where the C are theoretical total power spectra of the map co-adds.

3. Theory N0 for the split estimator
where the C are theoretical power spectra that exclude beam-deconvolved noise 
since only cross-split terms are included.

4. Approximate theory RDN0 for the coadd estimator
D_1({\ell_1},{\ell_2}) = C^{XU}_data_{\ell_1} C^{YV}_theory_{\ell_2} + \
C^{XU}_theory_{\ell_1} C^{YV}_data_{\ell_2} - C^{XU}_theory_{\ell_1} C^{YV}_theory_{\ell_2}
D_2({\ell_1},{\ell_2}) = C^{XV}_data_{\ell_1} C^{YU}_theory_{\ell_2} + \
C^{XV}_theory_{\ell_1} C^{YU}_data_{\ell_2} - C^{XV}_theory_{\ell_1} C^{YU}_theory_{\ell_2}
where data refer to realized power spectra and theory to a theorey expectation.

4. Approximate theory RDN0 for the split estimator
D_1({\ell_1},{\ell_2}) = S_XUYV_data_theory(l1,l2) + S_XUYV_theory_data(l1,l2) - S_XUYV_theory_theory(l1,l2)
D_2({\ell_1},{\ell_2}) = S_XVYU_data_theory(l1,l2) + S_XVYU_theory_data(l1,l2) - S_XVYU_theory_theory(l1,l2)

where S is the O(m^2) split estimator, but with the quadratic estimator Q(X,Y) function replaced by a power spectrum
<XY>, and the power spectrum estimator P(X,Y) replaced by a multiplication after assigning the variable l1
to X and l2 to Y. The subscripts like "data_theory" tell us whether to use realized data or theory for variables
of l1 and l2, e.g "data_theory" means use data for l1 variables and theory for l2 variables.

5. Very approximate theory RDN0 for the coadd
D_1({\ell_1},{\ell_2}) = C^{XU}_data_{\ell_1} C^{YV}_data_{\ell_2} 
D_2({\ell_1},{\ell_2}) = C^{XV}_data_{\ell_1} C^{YU}_data_{\ell_2} 

6. Very approximate theory RDN0 for the coadd
D_1({\ell_1},{\ell_2}) = S_XUYV_data_data(l1,l2)
D_2({\ell_1},{\ell_2}) = S_XVYU_data_data(l1,l2)

"""

def get_feed_dict(shape,wcs,theory,noise_t,noise_p,fwhm,gradt=False,cross_fields=None):
    modlmap = enmap.modlmap(shape,wcs)
    ells = theory['ells']
    interp = lambda x: symlens.interp(ells,y)(modlmap)
    feed_dict = {}
    feed_dict['uC_T_T'] = interp(theory['lCl_TT']) if not(gradt) else interp(theory['gCl_TT'])
    feed_dict['uC_T_E'] = interp(theory['lCl_TE'])
    feed_dict['uC_E_E'] = interp(theory['lCl_EE'])
    feed_dict['tC_T_T'] = interp(theory['lCl_TT']) + (noise_t * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.
    feed_dict['tC_T_E'] = interp(theory['lCl_TE'])
    feed_dict['tC_E_E'] = interp(theory['lCl_EE']) + (noise_p * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.
    feed_dict['tC_B_B'] = interp(theory['lCl_BB']) + (noise_p * np.pi/180./60.)**2. / symlens.gauss_beam(modlmap,fwhm)**2.

    if cross_fields is not None:
    fields = ['x','y','u','v']
    nfields = len(fields)
    for pol in ['TT','EE','TE','BB']:
        for i in range(nfields):
            for j in range(i+1,nfields):
                a,b = pol
                f1 = fields[i]
                f2 = fields[j]
                if rdn0_spectra is None:
                    feed_dict[f'tC_{f1}_{a}_{f2}_{b}'] = theory.lCl(f'{a}{b}',modlmap)
                    feed_dict[f'tC_{f1}_{b}_{f2}_{a}'] = theory.lCl(f'{a}{b}',modlmap)
                else:
                    feed_dict[f'tC_{f1}_{a}_{f2}_{b}'] = rdn0_spectra[f'{a}{b}'].copy()
                    feed_dict[f'tC_{f1}_{b}_{f2}_{a}'] = rdn0_spectra[f'{a}{b}'].copy()



def N_l_split_cross_estimator(shape,wcs,theory,noise_t,noise_p,fwhm,estimator,XY,UV,kmask_t,AlXY=None,AlUV=None,kmask=None,rdn0_spectra=None):
    modlmap = enmap.modlmap(shape,wcs)


    return symlens.N_l_cross(shape,wcs,feed_dict,estimator,XY,estimator,UV,
                                        xmask=kmask_t,ymask=kmask_t,Aalpha=AlXY,Abeta=AlUV,
                                        field_names_alpha=['x','y'],field_names_beta=['u','v'],kmask=kmask,skip_filter_field_names=True)



def N0_coadd_theory(shape,wcs):
    Nl_coadd = symlens.N_l_cross(shape,wcs,feed_dict,estimator_type,XY,estimator_type,UV,
                                 xmask=kmask_t,ymask=kmask_t,
                                 Aalpha=AlXY,Abeta=AlUV,field_names_alpha=None,field_names_beta=None,kmask=kmask)


def N0_split_theory():
    Nl_cross = sutils.N_l_split_cross_estimator(shape,wcs,theory,
                                                noise_t,noise_p,fwhm,
                                                estimator_type,XY,UV,kmask_t,
                                                AlXY=AlXY,AlUV=AlUV,kmask=kmask)

def RDN0_approx_coadd():

    Dexpr1 = tCXU_data_l1*tCYV_theory_l2 + tCXU_theory_l1*tCYV_data_l2 - tCXU_theory_l1*tCYV_theory_l2
    Dexpr2 = tCad_l1*tCbc_l2

    generic_cross_integral(shape,wcs,feed_dict,alpha_XY,beta_XY,Falpha,Fbeta,Fbeta_rev,Dexpr1,Dexpr2,
                           xmask=xmask,ymask=ymask,
                           field_names_alpha=field_names_alpha,field_names_beta=field_names_beta,groups=groups)


def RDN0_simple_coadd():
    rdn0_spectra = {}
    rdn0_spectra['TT'] = ppow(tmap.mean(axis=0)[0],tmap.mean(axis=0)[0])
    rdn0_spectra['TE'] = ppow(tmap.mean(axis=0)[0],tmap.mean(axis=0)[1])
    rdn0_spectra['EE'] = ppow(tmap.mean(axis=0)[1],tmap.mean(axis=0)[1])
    rdn0_spectra['BB'] = ppow(tmap.mean(axis=0)[2],tmap.mean(axis=0)[2])

    rdn0_coadd = N_l_split_cross_estimator(shape,wcs,theory,noise_t,noise_p,fwhm,estimator_type,
                                                  XY,UV,kmask_t,AlXY=AlXY,AlUV=AlUV,kmask=kmask,
                                                  rdn0_spectra=rdn0_spectra)


def N_l_coadd(shape,wcs,theory,estimator,XY,UV,kmask_t,AlXY=None,AlUV=None,kmask=None):
    Nl_coadd = symlens.N_l_cross(shape,wcs,feed_dict,estimator_type,XY,estimator_type,UV,
                                 xmask=kmask_t,ymask=kmask_t,
                                 Aalpha=AlXY,Abeta=AlUV,field_names_alpha=None,field_names_beta=None,kmask=kmask)


