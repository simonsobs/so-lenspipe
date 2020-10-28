.. Simons Observatory Lensing Pipeline documentation master file, created by
   sphinx-quickstart on Tue Oct 27 23:17:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simons Observatory Lensing Pipeline
===================================

This documentation will walk you through the process of transforming
microwave sky maps into a measurement of the CMB lensing power spectrum
and cosmological parameters. Broadly, this is achieved by passing
a co-added map through a quadratic estimator, calculating its
power spectrum, debiasing it with simulations and MCMC sampling for the
cosmological parameters.

That sounds simple enough, but with methods
that ensure insensitivity to the simulations used and robustness against
instrumental and astrophysical systematics, the detailed procedure can
be quite complex and computationally intensive. Moreover, a large array of
consistency and null tests
need to be done to ensure (and iterate on) the quality of the data that
is used. This cookbook-style documentaion will guide you through the process.

Note that parts of this documentation strongly overlap with parts of
the component separation pipeline, since they share a lot of the pre-processing.
The overlapping parts are fully documented here.



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :numbered:

   tasks
   staging
   testing




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
