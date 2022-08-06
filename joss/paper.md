---
title: '`POSEIDON`: A Multidimensional Atmospheric Retrieval Code for Exoplanet Spectra'
tags:
  - Python
  - astronomy
  - exoplanets
  - spectroscopy
  - atmospheric retrieval
  - atmospheric models
  - JWST

authors:
  - name: Ryan J. MacDonald
    orcid: 0000-0003-4816-3469
    affiliation: 1
affiliations:
  - name: Department of Astronomy and Carl Sagan Institute, Cornell University, 122 Sciences Drive, Ithaca, NY 14853, USA
    index: 1
date: 05 August 2022
bibliography: paper.bib

aas-doi: 10.3847/1538-4357/ac47fe
aas-journal: Astrophysical Journal

---

# Summary

Exoplanet atmosphere research is a dynamic and fast-changing field at the frontier of modern astronomy. Telescope observations can reveal the chemical composition, temperature, cloud properties, and (potentially) the habitability of these remote worlds. Astronomers can measure these exoplanet atmosphere properties by observing how the fraction of starlight blocked by a planet passing in front of its host star changes with wavelength --- this technique is called transmission spectroscopy. Since the wavelengths where different atoms and molecules absorb are already known (from laboratory measurements or quantum mechanics), astronomers can compare models of exoplanet spectra to observations to infer the chemical composition of distant exoplanets.

`POSEIDON` is a Python package for the modelling and analysis of exoplanet spectra. `POSEIDON` has two main functions: (i) computation of model spectra for 1D, 2D, or 3D exoplanet atmospheres; and (ii) a Bayesian fitting routine (`atmospheric retrieval') that can infer the range of atmospheric properties consistent with an observed exoplanet spectrum.

# Exoplanet Modelling and Atmospheric Retrieval with `POSEIDON`

Architecture of POSEIDON. Users can use the code in two ways, for forward modelling or retrievals. (see Figure \autoref{fig:overview})

POSEIDON first described in 2017 [@MacDonald:2017]. The multi-dimensional forward model, TRIDENT, was described in 2022 [@MacDonald:2022]

![Schematic architecture of `POSEIDON`. PLACEHOLDER to be expanded. \label{fig:POSEIDON_architecture}](figures/POSEIDON_Architecture_2022){width=100%}

# Statement of Need

Multi-dimensional atmospheric models.

Explosion of data from JWST. Reduce the barrier to entry by lowering computational burden. Also serve as an educational resource to be integrated in courses, given friendly plotting routines out-of-the-box and continually added tutorials.

# Documentation

Documentation for `POSEIDON`, with step-by-step tutorials illustrating research applications, are available at [https://poseidon-retrievals.readthedocs.io/en/latest/](https://poseidon-retrievals.readthedocs.io/en/latest/). 


# Similar Tools

[`PLATON`](https://github.com/ideasrule/platon) [@Zhang:2019], [`petitRADTRANS`](https://gitlab.com/mauricemolli/petitRADTRANS) [@Molliere:2019], [`CHIMERA`](https://github.com/mrline/CHIMERA) [@Line:2013], [`TauRex`](https://github.com/ucl-exoplanets/TauREx3_public) [@Waldman2015; @Al-Refaie:2021], [`Pyrat Bay`](https://github.com/pcubillos/pyratbay) [@Cubillos:2021], [`BART`](https://github.com/exosports/BART) [@Harrington:2022]


# Acknowledgements

RJM expresses gratitude to the developers of many open source Python packages used by `POSEIDON`, in particular `Numba` [@Lam:2015], `numpy` [@Harris:2020], `Matplotlib` [@Hunter:2007], `SciPy` [@Virtanen:2020], and `Spectres` [@Carnall:2017].

RJM acknowledges support from the UK's Science and Technology Facilities Council (STFC) for financial support during the early development of `POSEIDON`. RJM thanks Nikole Lewis, Ishan Mishra, Jonathan Gomez Barrientos, John Kappelmeier, Ruizhe Wang, and Antonia Peters for helpful discussions.


# References
