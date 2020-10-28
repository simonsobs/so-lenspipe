solenspipe
==========

|Documentation Status|

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
and scripts therein: \*
`so-pysm-models <https://github.com/simonsobs/so_pysm_models/>`__
(``python setup.py install --user``) required by mapsims \*
`pysm <https://github.com/healpy/pysm/>`__
(``pip install pysm3 --user``) required by mapsims \*
`mapsims <https://github.com/simonsobs/mapsims/>`__
(``git checkout car_fix ; python setup.py develop --user`` -- we
recommend a symbolic install link since you may have to periodically
update this repo, change branches, etc. since it is actively under
development) \* `pixell <https://github.com/simonsobs/pixell/>`__ (if
running on NERSC, run
``python setup.py build_ext -i --fcompiler=intelem --compiler=intelem``
followed by adding the directory to your PYTHONPATH; else run
``python setup.py install --user``); test by running ``py.test -s`` \*
`falafel <https://github.com/simonsobs/falafel/>`__
(``pip install -e . --user``) \*
`symlens <https://github.com/simonsobs/symlens/>`__
(``pip install -e . --user``) \*
`quicklens <https://github.com/msyriac/quicklens/>`__ (Python 3 fork of
Duncan Hanson's code used to get normalization of lensing estimators. if
running on NERSC, run
``python setup.py build_ext -i --fcompiler=intelem --compiler=intelem``;
else run ``python setup.py build_ext -i`` , and then add to PYTHONPATH)
\* `camb <https://camb.readthedocs.io/en/latest/>`__
(``pip install camb --user``) \*
`orphics <https://github.com/msyriac/orphics/>`__
(``pip install -e . --user``) \*
`enlib <https://github.com/amaurea/enlib/>`__ (just need enlib/bench.py
for benchmarking ; git clone the repo and add to PYTHONPATH) \*
`quaternionarray <https://pypi.org/project/quaternionarray/>`__:
(``pip install quaternionarray --user``) required by sotodlib \*
`sotodlib <https://github.com/simonsobs/sotodlib>`__
(``git checkout aca85843b70b0c6ebac031aa48fff47f93ed6661 ; python setup.py install --user``)
\* `sotoddb <https://github.com/simonsobs/sotoddb>`__
(``python setup.py install --user``) \*
`so\_noise\_models <https://github.com/simonsobs/so_noise_models>`__
(``python setup.py install --user``) \* Other miscellaneous packages:
healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow
(Python Image Library)

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

Run ``python bin/simple.py -h`` and if the installation is succesfull,
you should see

::

    usage: simple.py [-h] [-N NSIMS] [--sindex SINDEX] [--lmin LMIN] [--lmax LMAX]
                     [--isotropic] [--no-atmosphere] [--use-cached-norm]
                     [--wnoise WNOISE] [--beam BEAM] [--disable-noise]
                     [--zero-sim] [--write-meanfield] [--read-meanfield]
                     [--healpix] [--no-mask] [--debug] [--flat-sky-norm]
                     label polcomb

    Simple lensing reconstruction test.

    positional arguments:
      label                 Version label.
      polcomb               Polarizaiton combination: one of mv,TT,TE,EB,TB,EE.

    optional arguments:
      -h, --help            show this help message and exit
      -N NSIMS, --nsims NSIMS
                            Number of sims.
      --sindex SINDEX       Start index for sims.
      --lmin LMIN           Minimum multipole.
      --lmax LMAX           Minimum multipole.
      --isotropic           Isotropic sims.
      --no-atmosphere       Disable atmospheric noise.
      --use-cached-norm     Use cached norm.
      --wnoise WNOISE       Override white noise.
      --beam BEAM           Override beam.
      --disable-noise       Disable noise.
      --zero-sim            Just make a sim of zeros. Useful for benchmarking.
      --write-meanfield     Calculate and save mean-field map.
      --read-meanfield      Read and subtract mean-field map.
      --healpix             Use healpix instead of CAR.
      --no-mask             No mask. Use with the isotropic flag.
      --debug               Debug plots.
      --flat-sky-norm       Use flat-sky norm.

For a test beyond the imports, you can run
``python bin/simple.py test TT -N 1`` but you'll need some files in your
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

.. |Documentation Status| image:: https://readthedocs.org/projects/so-lenspipe/badge/?version=latest
   :target: https://so-lenspipe.readthedocs.io/en/latest/?badge=latest
