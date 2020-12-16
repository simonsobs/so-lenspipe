Pipeline tasks
==============

Conventions
-----------

We will store one-dimensional power spectra in 1d numpy arrays whose
index corresponds to the multipole (so the first indices starting with
zero store multipoles 0,1,2...etc.). All CMB power spectra are
in units of :math:`(\mu K-{\rm rad})^2` and do not contain any
factors of :math:`2 \pi` or :math:`\ell (\ell+1)`.

We use lensing *convergence* everywhere (not potential).
The ``falafel`` code returns unnormalized quadratic estimators and ``tempura``
returns full-sky normalizations. Within ``solenspipe``, functions that
return quadratic estimator reconstructions will return *normalized* estimators.

We use the estimator normalization convention that results in the noise power 
:math:`N_L` for an optimal estimator being equal to its 
normalization :math:`A_L`.



Preparation
-----------

Data access
^^^^^^^^^^^

The ACT and SO map-makers provide sets of maps with mutually exclusive data;
each set consists of completely independent TOD samples. This constitutes some
splitting of the data. For historical reasons, we refer to each such set as
an `array'. This terminology is derived from the fact that the TODs are
primarily split by which detector array they originate from, though since 2015, ACTpol
and its successors (including Advanced ACT and SO) use multi-chroic arrays,
which means each hardware array will provide us multiple (almost always two) `array' map sets even
in the same season/year and region. We will now stop using quotes around `array'
under the understanding that it applies to some unit of splitting closely
related to what is used in ACT.

Within ACT, these arrays typically come from some region or scan (though since 2016 there
has primarily been just a wide scan each for day and night) for a particular season
and particular frequency band (since the ACTpol PA3 array, one of two within a dichroic hardware array).
For SO, under the current simulation design, there will be two array maps for each optics tube because
of the dichroic hardware array in the tube.

We will also be combining with Planck, for which we define a Planck array as a particular
frequency band, reprojected to the CAR pixelization and subtracted of sources (see :ref:`planck_reproj`).


Planck reprojection
^^^^^^^^^^^^^^^^^^^

Simulation
----------

Co-addition
-----------

Filtering and data spectra
--------------------------

This stage introduces a choice of cosmology.

In this stage, we also calculate the power spectra of the filtered maps
and store these. These will be used as inputs for approximate
bias subtraction. The spectra can also be used to verify accuracy of 
the simulations in a manner consistent with the blinding policy.

Normalization and theory N0
---------------------------

This stage also assumes a choice of cosmology (which is more important
downstream), but we constrain it to be the same cosmology used in the 
filtering stage.




Quadratic Estimator
-------------------




Multiplicative verification
---------------------------


Bias subtraction
----------------

Mean-field map
^^^^^^^^^^^^^^

Monte Carlo N1
^^^^^^^^^^^^^^

Realization-dependent N0
^^^^^^^^^^^^^^^^^^^^^^^^

Diagonal RDN0
^^^^^^^^^^^^^

MC bias
^^^^^^^

Covariance
----------

Exploration and validation
--------------------------

Cosmological constraints
------------------------

Mock external datasets
^^^^^^^^^^^^^^^^^^^^^^

