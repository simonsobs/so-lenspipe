Pipeline tasks
==============



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
which means each hardware array will provide us multiple `array' map sets even
in the same season/year and region. We will now stop using quotes around `array'
under the understanding that it applies to some unit of splitting closely
related to what is used in ACT.

Within ACT, these arrays typically come from some region or scan (though since 2016 there
has primarily been just a wide scan each for day and night) for a particular season
and particular frequency band (since the ACTpol PA3 array, one of two within a dichroic hardware array).
For SO, under the current simulation design, there will be one array for each

Planck reprojection
^^^^^^^^^^^^^^^^^^^

Simulation
----------

Co-addition
-----------

Filtering
---------

Normalization
-------------

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

