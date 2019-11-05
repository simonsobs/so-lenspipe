=======
solenspipe
=======

Pipeline libraries and scripts for L3.1

* Free software: BSD license

Dependencies
------------

* simonsobs/mapsims, simonsobs/pixell, simonsobs/falafel, simonsobs/symlens
* simonsobs/mapsims requires simonsobs/so_pysm_models and healpy/pysm
* msyriac/orphics, amaurea/enlib (just bench.py for benchmarking)
* healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow (Python Image Library)

Installing
----------

To install, run:

.. code-block:: console
		
   $ pip install -e . --user


Then copy `input/config_template.yml` to `input/config.yml` and edit it to match paths on your system.


Demo
----

.. code-block:: console

				$ python bin/demo.py -h
				usage: demo.py [-h] [--nside NSIDE] [--smooth-deg SMOOTH_DEG] [--lmin LMIN]
				[--lmax LMAX] [--freq FREQ]
				polcomb

				Demo lensing pipeline.

				positional arguments:
				polcomb               Polarization combination. Possibilities include mv
                (all), mvpol (all pol), TT, EE, TE, EB or TB.

				optional arguments:
				-h, --help            show this help message and exit
				--nside NSIDE         nside
				--smooth-deg SMOOTH_DEG
                Gaussian smoothing sigma for mask in degrees.
				--lmin LMIN           lmin
				--lmax LMAX           lmax
				--freq FREQ           channel freq
				


Contributing
------------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. 
  
