
POSEIDON's documentation
====================================

POSEIDON is a Python package designed to rapidly retrieve atmospheric properties 
from exoplanet spectra. POSEIDON includes modelling capabilities to 
generate 1D, 2D, or 3D exoplanet transmission spectra (`TRIDENT 
<https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_) alongside emission 
and/or reflection spectra for substellar atmospheres. For fitting observed spectra 
('atmospheric retrieval'), POSEIDON uses a nested sampling retrieval 
framework, via the sampling algorithm `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_,
to explore the range of allowed atmospheric properties.

POSEIDON's official features currently include:

* Transmission spectra modelling and retrievals for 1D, 2D, and 3D exoplanet atmospheres.
* Emission and reflection spectra, including scattering, for modelling and retrievals.
* Rapid atmospheric retrievals that can run on your laptop.
* Model planets ranging from ultra-hot Jupiters to temperate terrestrials.
* Parametric prescriptions for stellar contamination, multidimensional clouds, and more.
* High-resolution line-by-line models (:math:`R \sim 10^6`) for cross correlation analyses.

Beta features:

* Modelling and retrievals for brown dwarfs and directly imaged exoplanets.

POSEIDON contains an extensive range of tutorials covering the generation of
forward models and running atmospheric retrievals.


New in POSEIDON v1.2:
------------------------------------

* Mie scattering aerosols for exoplanet forward models and retrievals.
* An extensive database of pre-computed aerosol optical properties covering the full range of substellar atmospheres.
* New `opacity database <content/opacity_database.html>`_ with state-of-the-art molecular line lists.
* Emission and reflection spectra, including scattering, for modelling and retrievals.
* Many new tutorials.

To use these new features, you will need to re-download the POSEIDON input data.
Please see the installation instructions.

For a comprehensive list of new features, see the POSEIDON `Release Notes 
<https://github.com/MartianColonist/POSEIDON/releases>`_ on GitHub.


New in POSEIDON v1.1:
------------------------------------

* Chemical equilibrium models and retrievals, demonstrated in two new tutorials.
* JWST proposal tutorial (PandExo + retrieving simulated JWST data).  
* Bayesian model comparison demonstration in first retrieval tutorial.
* Improved stellar contamination retrieval capabilities (e.g. spots + faculae).


Contributor Hall of Fame:
------------------------------------

* Ryan MacDonald (Lead Developer)
* Elijah Mullens (Aerosols)
* Ruizhe Wang (Equilibrium Chemistry)
* Charlotte Fairman (Contributor)


.. toctree::
   :maxdepth: 1
   :caption: Installation:

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

   content/citations
   content/contributing
   autoapi/index


License:
------------------------------------

POSEIDON is available under the BSD 3-Clause License. If you use POSEIDON,
please cite `MacDonald & Madhusudhan (2017) 
<https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.1979M/abstract>`_ and 
`MacDonald (2023) <https://ui.adsabs.harvard.edu/abs/2023JOSS....8.4873M/abstract>`_. 
Guidance on specific papers to cite for individual features included within 
POSEIDON is provided on the page `"What to Cite" <content/citations.html>`_.