Testing
=======


Verification
------------

Verification tests do not involve any real-world data. The main aim
is to make sure that the pipeline is unbiased. These runs involve
treating some subset of the simulations as the data, and then
averaging the results to residuals with respect to the input. In
practice, we expect to find percent level residuals that constiute
the MC bias. We provide the following:

* Baseline MC with 1d sims
* Baseline MC with tiled sims
* Baseline MC with time-domain sim
* Extragalactic foreground test
* Galactic foreground test
* Varied cosmology runs

The last run is crucial in re-inforcing our confidence in the percent
level MC bias, when percent level precision is called for in the data.
We expect the MC bias to primarily be dictated by things like the mask
which our simulations accurately capture. The varied cosmology run
looks for a cosmology dependence in the MC bias, which should be ideally
be negligible.

Null tests
----------

We run a series of tests which should result in bandpowers we expect to
be consistent with zero. These come in two classes, curl tests and data
split tests. The latter involves taking two splittings of the data, differencing
them and running them through the lensing pipeline. In the absence of
systematics, we expect the CMB and foregrounds to cancel. If the debiasing
steps are working properly, the resulting bandpowers should be consistent
with zero. We provide:

* Curl tests
* 90 GHz - 150 GHz null
* Individual array split differences
* Co-added split difference
* Day vs. night difference
* Difference from Planck nulls

Bandpower consistency tests
---------------------------

We run the lensing pipeline on various splittings of the data. We then
examine the difference of bandpowers, which should be consistent with null.
We provide the following:

* Individual array consistency
* Individual array consistency (tiled sims)
* Polarization combination consistency
* Isotropy tests
* Minimum multipole and maximum multipole variations
* Night-only
* Cross-only split-based estimator
* Aggressive dust masks
* 353 GHz subtraction
* Samples from beam error
* Samples from calibration error

