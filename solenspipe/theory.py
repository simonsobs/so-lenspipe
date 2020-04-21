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




def RDN0_coadd():
    return symlens.RDN0_analytic(shape,wcs,feed_dict,alpha_estimator,alpha_XY,beta_estimator,beta_XY,
                  Aalpha=None,Abeta=None,xmask=None,ymask=None,kmask=None,
                  field_names_alpha=None,field_names_beta=None,skip_filter_field_names=False):
    


