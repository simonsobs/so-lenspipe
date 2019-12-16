import symlens

"""
Functions that use theoretical flat-sky expressions
for lensing predictions.
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

def N0_coadd_theory(shape,wcs):
    Nl_coadd = symlens.N_l_cross(shape,wcs,feed_dict,estimator_type,XY,estimator_type,UV,
                                 xmask=kmask_t,ymask=kmask_t,
                                 Aalpha=AlXY,Abeta=AlUV,field_names_alpha=None,field_names_beta=None,kmask=kmask)


def N0_split_theory():
    Nl_cross = sutils.N_l_split_cross_estimator(shape,wcs,theory,
                                                noise_t,noise_p,fwhm,
                                                estimator_type,XY,UV,kmask_t,
                                                AlXY=AlXY,AlUV=AlUV,kmask=kmask)


def 
    rdn0_spectra = {}
    rdn0_spectra['TT'] = ppow(tmap.mean(axis=0)[0],tmap.mean(axis=0)[0])
    rdn0_spectra['TE'] = ppow(tmap.mean(axis=0)[0],tmap.mean(axis=0)[1])
    rdn0_spectra['EE'] = ppow(tmap.mean(axis=0)[1],tmap.mean(axis=0)[1])
    rdn0_spectra['BB'] = ppow(tmap.mean(axis=0)[2],tmap.mean(axis=0)[2])

    rdn0_coadd = sutils.N_l_split_cross_estimator(shape,wcs,theory,noise_t,noise_p,fwhm,estimator_type,
                                                  XY,UV,kmask_t,AlXY=AlXY,AlUV=AlUV,kmask=kmask,
                                                  rdn0_spectra=rdn0_spectra)

