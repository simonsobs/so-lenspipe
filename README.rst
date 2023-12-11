==========
solenspipe
==========

.. image:: https://readthedocs.org/projects/so-lenspipe/badge/?version=latest
           :target: https://so-lenspipe.readthedocs.io/en/latest/?badge=latest
		   :alt: Documentation Status


Pipeline libraries and scripts for L3.1

-  Free software: BSD license

Dependencies
------------

NERSC tip: you may need the ``python/3.7-anaconda-2019.07`` module. You
can add ``module load python/3.7-anaconda-2019.07`` to your
``~/.bash_profile.ext``. Also, when running quick tests on the login
node (e.g. to test imports after setting up), you should run
``export DISABLE_MPI=true`` since MPI calls do not work on the cori
login node.

Here are all the pacakges you'll need before you can run this library
and scripts therein:

* `pixell <https://github.com/simonsobs/pixell/>`__ (if
  running on NERSC, run
  ``python setup.py build_ext -i --fcompiler=intelem --compiler=intelem``
  followed by adding the directory to your PYTHONPATH; else run
  ``python setup.py install --user``); test by running ``py.test -s``
* `falafel <https://github.com/simonsobs/falafel/>`__
  (``pip install -e . --user``) 
* `tempura <https://github.com/simonsobs/tempura>`__ 
  (Clone the repository then do ``python setup.py build_ext -i`` and then add to your PYTHONPATH)
* `camb <https://camb.readthedocs.io/en/latest/>`__
  (``pip install camb --user``) 
* `orphics <https://github.com/msyriac/orphics/>`__
  (``pip install -e . --user``) 
* `enlib <https://github.com/amaurea/enlib/>`__ (just need enlib/bench.py
  for benchmarking ; git clone the repo and add to PYTHONPATH) 
* `soapack <https://github.com/simonsobs/soapack>`__
  (``pip install -e . --user``)
* `pyfisher <https://github.com/msyriac/pyfisher>`__
  (``pip install -e . --user``)
* `actsims <https://github.com/ACTCollaboration/actsims>`__
  (clone repo, checkout `new_scheme` branch, and then ``pip install -e . --user``)
* Other miscellaneous packages:
  healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow
  (Python Image Library), toml
  

Installing
----------

To install, run:

::

    python setup.py build_ext -i
    pip install -e . --user

Then copy ``input/config_template.yml`` to ``input/config.yml`` and edit
it to match paths on your system (specifically, the ``data_path``
variable in the .yml file will need to be changed to be for a directory
of your own).

Demo
----

Run ``python examples/bias.py -h`` and if the installation is succesfull,
you should see

::

		usage: bias.py [-h] [--nsims-n0 NSIMS_N0] [--nsims-n1 NSIMS_N1] [--healpix]
					   [--new-scheme] [--lmax LMAX] [--lmin LMIN] [--biases BIASES]
					   version est1 est2

		Verify and benchmark RDN0 on the full noiseless sky.

		positional arguments:
		  version              Version name.
		  est1                 Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.
		  est2                 Estimator 2, same as above.

		optional arguments:
		  -h, --help           show this help message and exit
		  --nsims-n0 NSIMS_N0  Number of sims.
		  --nsims-n1 NSIMS_N1  Number of sims.
		  --healpix            Use healpix instead of CAR.
		  --new-scheme         New simulation scheme.
		  --lmax LMAX          Maximum multipole for lensing.
		  --lmin LMIN          Minimum multipole for lensing.
		  --biases BIASES      Maximum multipole for lensing.

For a test beyond the imports, you can run
``python examples/bias.py test TT TT --lmax 300 --nsims-n0 1 --nsims-n1 1`` but you'll need some files in your
``data_path`` directory to get going.

Note that if working on NERSC, you might have to run the scripts on an
interactive node.

Contributing
------------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and
proceed as described above.

