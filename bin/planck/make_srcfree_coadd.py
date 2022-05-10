#It seems we only have the srcfree maps
#as the A and B splits. We should run the
#full coadd, but in the meantime, just
#coadd weighted by ivar

import os
os.environ['DISABLE_MPI']="true"
from os.path import join as opj
from pixell import enmap,curvedsky,reproject

map_dir="/global/project/projectdirs/act/data/synced_maps/NPIPE/"
for freq in [217, 353, 545]:
    m_A = enmap.read_map(opj(map_dir, "npipe6v20A_%03d_map_srcfree_enmap.fits"%freq))
    m_B = enmap.read_map(opj(map_dir, "npipe6v20B_%03d_map_srcfree_enmap.fits"%freq))
    ivar_A = enmap.read_map(opj(map_dir, "npipe6v20A_%03d_ivar_enmap.fits"%freq))
    ivar_B = enmap.read_map(opj(map_dir, "npipe6v20B_%03d_ivar_enmap.fits"%freq))
    
    w_A = 1./ivar_A
    w_B = 1./ivar_B
    coadd = (w_A*m_A + w_B*m_B)/(w_A+w_B)
    coadd_ivar = w_A + w_B

    fcoadd = opj(map_dir,
                    "npipe6v20ABcoadd_%03d_map_srcfree_enmap.fits"%freq)
    print("writing coadd to %s"%fcoadd)
    coadd.write(fcoadd)
    fivar = opj(map_dir,
                    "npipe6v20ABcoadd_%03d_ivar_enmap.fits"%freq)
    print("writing ivar to %s"%fivar)
    coadd_ivar.write(fivar)
