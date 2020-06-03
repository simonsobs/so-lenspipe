=======
solenspipe
=======

Pipeline libraries and scripts for L3.1

* Free software: BSD license

Dependencies
------------

* simonsobs/mapsims, simonsobs/pixell, simonsobs/falafel, simonsobs/symlens
* simonsobs/mapsims requires simonsobs/so_pysm_models and healpy/pysm
* msyriac/quicklens (Python 3 fork of Duncan Hanson's code used to get
  normalization of lensing estimators)
* msyriac/orphics, amaurea/enlib (just bench.py for benchmarking)
* healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow (Python Image Library)

Installing
----------

To install, run:

.. code-block:: console
		
   $ python setup.py build_ext -i
   $ pip install -e . --user


Then copy `input/config_template.yml` to `input/config.yml` and edit it to match paths on your system.


Demo
----

.. code-block:: console

		$ python bin/simple.py -h
		usage: simple.py [-h] [-N NSIMS] [--sindex SINDEX] [--lmin LMIN] [--lmax LMAX]
						 [--isotropic] [--no-atmosphere] [--use-cached-norm]
						 [--wnoise WNOISE] [--beam BEAM] [--disable-noise]
						 [--zero-sim] [--healpix] [--no-mask] [--debug]
						 [--flat-sky-norm]
						 label polcomb

		Do a thing.

		positional arguments:
		  label                 Label.
		  polcomb               polcomb.

		optional arguments:
		  -h, --help            show this help message and exit
		  -N NSIMS, --nsims NSIMS
								Number of sims.
		  --sindex SINDEX       Declination band.
		  --lmin LMIN           Minimum multipole.
		  --lmax LMAX           Minimum multipole.
		  --isotropic           Isotropic sims.
		  --no-atmosphere       Disable atmospheric noise.
		  --use-cached-norm     Use cached norm.
		  --wnoise WNOISE       Override white noise.
		  --beam BEAM           Override beam.
		  --disable-noise       Disable noise.
		  --zero-sim            Just make a sim of zeros. Useful for benchmarking.
		  --healpix             Use healpix.
		  --no-mask             No mask. Use with the isotropic flag.
		  --debug               Debug plots.
		  --flat-sky-norm       Use flat-sky norm.


Contributing
------------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. 
  
