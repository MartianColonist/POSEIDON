
POSEIDON's documentation
====================================

POSEIDON is a Python package designed to rapidly retrieve atmospheric properties 
from exoplanet spectra. POSEIDON has two main components: (1) a 'forward' model, 
`TRIDENT <https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_, that 
generates 1D, 2D, or 3D transmission spectra; and (2) a nested sampling retrieval 
framework that uses the sampling algorithm `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_,
wrapped around TRIDENT, to explore the range of atmospheric properties consistent
with an observed exoplanet spectrum.

POSEIDON's features currently include:

* Transmission spectra modelling for 1D, 2D, and 3D exoplanet atmospheres.
* Rapid atmospheric retrievals that can run on your laptop.
* Model support for planets ranging from ultra-hot Jupiters to temperate terrestrials.
* Parametric prescriptions for stellar contamination, multidimensional clouds, and more.
* High-resolution line-by-line models (:math:`R \sim 10^6`) for cross correlation analyses.

The initial public release of POSEIDON contains tutorials on generating model 
transmission spectra with TRIDENT and running atmospheric retrievals.
Tutorials on multidimensional retrievals will be added soon.

POSEIDON is available under the BSD 3-Clause License. If you use POSEIDON for retrievals, 
please cite `MacDonald & Madhusudhan (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.1979M/abstract>`_. 
If you only use the forward model, TRIDENT, please cite `MacDonald & Lewis (2022) 
<https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_.

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

   autoapi/index

