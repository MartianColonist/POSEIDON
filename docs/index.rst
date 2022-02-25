.. POSEIDON documentation master file, created by
   sphinx-quickstart on Thu Feb  3 13:15:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

POSEIDON's documentation
====================================

POSEIDON is a Python package designed to rapidly retrieve atmospheric properties from exoplanet spectra. POSEIDON has two main components: (1) a 'forward' model, `TRIDENT <https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_, which generates multidimensional transmission spectra; (2) a nested sampling retrieval framework that uses `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_ to explore the range of atmospheric properties consistent with an observed exoplanet spectrum.

POSEIDON's features currently include:

* Transmission spectra modelling for 1D, 2D, and 3D exoplanet atmospheres.
* Rapid atmospheric retrievals that can run on your laptop.
* Model ultra-hot Jupiters down to temperate terrestrial worlds.
* High-resolution line-by-line opacities (:math:`R \sim 10^6`) for cross correlation analyses.
* Stellar contamination from unocculted active regions.

The initial public release of POSEIDON contains tutorials on generating model transmission spectra with TRIDENT and running atmospheric retrievals for simulated HST data. Tutorials on multidimensional retrievals will be added soon.

POSEIDON is available under the BSD 3-Clause License. If you use POSEIDON for retrievals, please cite `MacDonald & Madhusudhan (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.1979M/abstract>`_. If you only use the forward model, TRIDENT, please cite `MacDonald & Lewis (2022) <https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_.



.. toctree::
   :maxdepth: 2
   :caption: Guide:
   
   content/forward_model_tutorials
   content/retrieval_tutorials
   
.. toctree::
   :maxdepth: 2
   :caption: Code Documentation:

   autoapi/index

