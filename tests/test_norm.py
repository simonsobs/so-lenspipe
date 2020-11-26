#Test the SOLensPipe.initalize_norm
from solenspipe import SOLensInterface, get_mask
import mapsims
import numpy as np

#Test that caching of the norm is working by
#running initialize_norm twice - once with
#recalculate=True, and then once with
#recalculate=False, and make sure they give
#the same answer. Do this with cmblensplus
#True and False
def test_norm_caching():
    mask = get_mask()
    beam_fwhm = 1.4
    white_noise = 30.
    solint = SOLensInterface(mask=mask, beam_fwhm=beam_fwhm,
                               white_noise=white_noise)
    channel = mapsims.SOChannel("LA", 145)
    lmin,lmax=200,1000

    for use_cmblensplus in [True,False]:
        #Run once with recalculate=True
        ALs,ALs_curl = solint.initialize_norm(
            channel,lmin,lmax,
            recalculate=True,use_cmblensplus=use_cmblensplus)
        #Run again with recaclulate=False
        ALs_cached,ALs_curl_cached = solint.initialize_norm(
            channel,lmin,lmax,
            recalculate=False,use_cmblensplus=use_cmblensplus)
        for col in ALs.dtype.names:
            np.testing.assert_array_equal(ALs[col], ALs_cached[col])
            if ALs_curl is not None:
                np.testing.assert_array_equal(ALs_curl[col],
                                              ALs_curl_cached[col]
                                              )
    print("test passed!")

            
if __name__=="__main__":
    test_norm_caching()
