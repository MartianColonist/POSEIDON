What to Cite when Using POSEIDON
================================

General Citations
-----------------

When using POSEIDON in a paper, please cite `MacDonald & Madhusudhan (2017) 
<https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.1979M/abstract>`_  and 
`MacDonald (2023) <https://ui.adsabs.harvard.edu/abs/2023JOSS....8.4873M/abstract>`_.

For multidimensional transmission spectra forward models, please cite the TRIDENT
model methods paper, `MacDonald & Lewis (2022) 
<https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_.

For 1D thermal emission forward model or retrievals without scattering 
(POSEIDON v1.0), please cite `Coulombe et al. (2023) 
<https://ui.adsabs.harvard.edu/abs/2023Natur.620..292C/abstract>`_. For more 
comprehensive emission models including scattering (from POSEIDON v1.2), please 
cite `Mullens et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...977..105M/abstract>`_, 
as described below.

For high-resolution cross correlation retrievals (POSEIDON v1.3), please cite
`Wang et al. (2025) <https://ui.adsabs.harvard.edu/abs/2025AJ....169..328W/abstract>`_


Cross Sections (Opacities)
--------------------------

For each molecule included in a model, in your methods section it is good practice
to cite the specific molecular line list for each molecule's opacity. It takes 
our quantum colleagues *years* to make some of these line lists, so please do
recognise their hard work! 

For convenience, we have provided a table on the `opacity database <opacity_database.html>`_
page containing NASA ADS links to the relevant citation for every line list used 
in POSEIDON.


Mie Scattering Aerosols
-----------------------

POSEIDON v1.2 includes Mie scattering from compositionally-specific aerosols.
When using any Mie scattering prescription in POSEIDON, please cite 
`Mullens et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...977..105M/abstract>`_. 
POSEIDON's Mie scattering retrieval functionality was first used in 
`Grant et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract>`_, 
but since a full description of the methodology is provided in 
`Mullens et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...977..105M/abstract>`_
that is the preferred paper to cite for Mie scattering retrievals with POSEIDON.
If you are using the v1.2 version of the aerosol database, consider also citing the 
`LX-MIE algorithm <https://ui.adsabs.harvard.edu/abs/2018MNRAS.475...94K/abstract>`_  
and `PLATON <https://ui.adsabs.harvard.edu/abs/2019PASP..131c4501Z/abstract>`_, 
whose algorithms were adapted to precompute Mie-scattering cross sections. 
If you are using the v1.3.1 version of the aerosol database, consider citing 
`miepython v2.5.5 <https://github.com/scottprahl/miepython>`_.

The `opacity database <opacity_database.html>`_ page provides extensive
references for specific aerosols (e.g. refractive index sources). We also provide
a comprehensive guide to our aerosol database and sources in the file
`aerosol_database_readme.txt <../_static/Aerosol-Database-Readme.txt>`_.


Thermal Scattering and Reflection 
---------------------------------

When using POSEIDON for emission spectroscopy with scattering enabled or for 
reflection spectroscopy, please cite `Mullens et al. (2024) 
<https://ui.adsabs.harvard.edu/abs/2024ApJ...977..105M/abstract>`_. 
As described in this paper, POSEIDON uses adapted forward models from PICASO, 
so for reflection spectra please also cite `Batalha et al. (2019) 
<https://ui.adsabs.harvard.edu/abs/2019ApJ...878...70B/abstract>`_ 
and for emission spectra with scattering please cite `Mukherjee et al. (2023) 
<https://ui.adsabs.harvard.edu/abs/2023ApJ...942...71M/abstract>`_. 
The underlying multiple scattering radiative transfer technique used in all these papers is 
described in `Toon et al. (1989) <https://ui.adsabs.harvard.edu/abs/1989JGR....9416287T/abstract>`_.


Pressure-Temperature (P-T) Profiles
-----------------------------------

POSEIDON includes many P-T profile prescriptions, including:

* The 'Madhu' profile from `Madhusudhan & Seager (2009) <https://ui.adsabs.harvard.edu/abs/2009ApJ...707...24M/abstract>`_.
* The 'Guillot' and 'Dayside Guillot' profiles from `Guillot (2010) <https://ui.adsabs.harvard.edu/abs/2010A%26A...520A..27G/abstract>`_.
* The 'Line' profile from `Line et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...775..137L/abstract>`_.
* The 'Slope' profile from `Piette & Madhusudhan (2021) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.5136P/abstract>`_.
* The 'Pelletier' profile from `Pelletier et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021AJ....162...73P/abstract>`_.


Contribution Functions
----------------------

For contribution functions (transmission or emission), added in POSEIDON v1.2, 
please cite `Mullens et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...977..105M/abstract>`_.
