
POSEIDON's documentation
====================================

POSEIDON is a Python package designed to rapidly retrieve atmospheric properties 
from exoplanet spectra. POSEIDON has two main components: (1) a 'forward' model, 
`TRIDENT <https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_, that 
generates 1D, 2D, or 3D transmission spectra; and (2) a nested sampling retrieval 
framework that uses the sampling algorithm `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_,
wrapped around TRIDENT, to explore the range of atmospheric properties consistent
with an observed exoplanet transmission spectrum.

POSEIDON's official features currently include:

* Transmission spectra modelling for 1D, 2D, and 3D exoplanet atmospheres.
* Rapid atmospheric retrievals that can run on your laptop.
* Model support for planets ranging from ultra-hot Jupiters to temperate terrestrials.
* Parametric prescriptions for stellar contamination, multidimensional clouds, and more.
* High-resolution line-by-line models (:math:`R \sim 10^6`) for cross correlation analyses.

Beta features:

* Chemical equilibrium retrievals.
* Emission spectra modelling and retrievals for 1D, cloud-free atmospheres without scattering.

The initial public release of POSEIDON contains a range of tutorials on 
generating forward models and a tutorial on running atmospheric retrievals.
Tutorials on multidimensional retrievals will be added soon.


New in POSEIDON v1.1:
------------------------------------

To use these new features, you will need to re-download the POSEIDON input data.
See the installation instructions.

* Chemical equilibrium models and retrievals, demonstrated in two new tutorials.
* JWST proposal tutorial (PandExo + retrieving simulated JWST data).  
* Bayesian model comparison demonstration in first retrieval tutorial.
* Improved stellar contamination retrieval capabilities (e.g. spots + faculae).

See the POSEIDON `Release Notes 
<https://github.com/MartianColonist/POSEIDON/releases>`_ on GitHub for more details.


License:
------------------------------------

POSEIDON is available under the BSD 3-Clause License. If you use POSEIDON,
please cite `MacDonald & Madhusudhan (2017) 
<https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.1979M/abstract>`_ and 
`MacDonald (2023) <https://joss.theoj.org/papers/69710c0498d02fd1c6a0cfa4b01af7c5>`_.
Additionally, if you make use of the multidimensional transmission spectra 
modelling capabilities, we would appreciate a citation for the TRIDENT methods
paper: `MacDonald & Lewis (2022) 
<https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_.


Contributor Hall of Fame:
------------------------------------

Ryan MacDonald, Ruizhe Wang, Elijah Mullens


.. toctree::
   :maxdepth: 1
   :hidden:

   content/installation

.. toctree::
   :maxdepth: 2
   :caption: Guide:
   
   content/forward_model_tutorials
   content/retrieval_tutorials
   content/opacity_database
   
.. toctree::
   :maxdepth: 2
   :caption: Code Documentation:

   content/contributing
   autoapi/index

