# solenspipe

Pipeline libraries and scripts for L3.1

* Free software: BSD license

## Dependencies
* [so-pysm-models](https://github.com/simonsobs/so_pysm_models/) (python setup.py install --user) and [pysm](https://github.com/healpy/pysm/) (pip install pysm3 --user) required by mapsims
* [mapsims](https://github.com/simonsobs/mapsims/) (python setup.py develop --user)
* [pixell](https://github.com/simonsobs/pixell/) (python setup.py install --user)
* [falafel](https://github.com/simonsobs/falafel/) (pip install -e . --user)
* [symlens](https://github.com/simonsobs/symlens/) (pip install -e . --user)
* [quicklens](https://github.com/msyriac/quicklens/) (Python 3 fork of Duncan Hanson's code used to get
  normalization of lensing estimators ; python setup.py build_ext -i ; and then
  add to PYTHONPATH)
* General utilities: [orphics](https://github.com/msyriac/orphics/) (pip install -e . --user), [enlib](https://github.com/amaurea/enlib/) (just
  enlib/bench.py for benchmarking ; git clone and add to PYTHONPATH)
* healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow (Python Image Library)

## Installing
To install, run:

```		
   $ python setup.py build_ext -i
   $ pip install -e . --user
```

Then copy `input/config_template.yml` to `input/config.yml` and edit it to match paths on your system.


## Demo
```
$ python bin/simple.py -h
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

```

## Contributing

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. 
