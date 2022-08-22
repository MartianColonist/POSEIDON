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

Exoplanet atmospheres are a dynamic and fast-changing field at the frontier of modern astronomy. Telescope observations can reveal the chemical composition, temperature, cloud properties, and (potentially) the habitability of these remote worlds. Astronomers can measure these atmospheric properties by observing how the fraction of starlight blocked by a planet passing in front of its host star changes with wavelength --- a technique called transmission spectroscopy. Since the wavelengths where different atoms and molecules absorb are already known (from laboratory measurements or quantum mechanics), astronomers can compare models of exoplanet spectra to observations to infer the chemical composition of exoplanets.

`POSEIDON` is a Python package for the modelling and analysis of exoplanet spectra. `POSEIDON` has two main functions: (i) computation of model spectra for 1D, 2D, or 3D exoplanet atmospheres; and (ii) a Bayesian fitting routine (`atmospheric retrieval') that can infer the range of atmospheric properties consistent with an observed exoplanet spectrum.

# Exoplanet Modelling and Atmospheric Retrieval with `POSEIDON`

The first major use case for `POSEIDON` is 'forward modelling' --- illustrated on the left of \autoref{fig:POSEIDON_architecture}. A user can generate a model planet spectrum, for a given star-planet system, by providing a specific set of atmospheric properties (e.g. the chemical composition and temperature). The forward model mode allows users to explore how atmospheric properties alter an exoplanet spectrum and to produce predicted model spectra for observing proposals. The required input files (pre-computed stellar grids and an opacity database) are available to download from a Zenodo repository (linked in the documentation).

The second major use case for `POSEIDON` is atmospheric retrieval --- illustrated on the right of \autoref{fig:POSEIDON_architecture}. To initialise a retrieval, a user provides an observed exoplanet spectrum and the range of atmospheric properties to be explored (i.e. the prior ranges for a set of free parameters defining a model). A Bayesian statistical sampling algorithm --- nominally [` PyMultiNest`](https://github.com/JohannesBuchner/PyMultiNest) [@Buchner:2014] --- then repeatedly calls the forward model, comparing the generated spectrum to the observations, until the parameter space is fully explored and a convergence criteria reached. The main outputs of an atmospheric retrieval are the posterior probability distributions of the model parameters and the model's Bayesian evidence. The Bayesian evidences from multiple retrievals, in turn, can be subsequently compared to compute detection significances for each model component (e.g. the statistical confidence for a molecule being present in the planetary atmosphere).

![Schematic architecture of the `POSEIDON` atmospheric retrieval code. Users can call `POSEIDON` in two main ways: (i) to generate a model exoplanet spectrum for a specified planet atmosphere (green arrows); or (ii) to fit an observed exoplanet spectrum by statistical sampling of a model's atmospheric properties (purple arrows). The diagram highlights code inputs (circles), algorithm steps (rectangles), and code outputs (bottom green or purples boxes). \label{fig:POSEIDON_architecture}](figures/POSEIDON_Architecture_2022){width=100%}

`POSEIDON` was first described in the exoplanet literature by [@MacDonald:2017]. Since then, the code has been used in 17 peer-reviewed publications [@Sedaghati:2017; @Kaltenegger:2020; e.g., @Alam:2021]. Most recently, a detailed description of `POSEIDON`'s new multi-dimensional forward model, `TRIDENT`, was provided by [@MacDonald:2022].

# Statement of Need

Recent years have seen a substantial improvement in the number of high-quality exoplanet spectra. In particular, the newly operational James Webb Space Telescope (JWST) and a profusion of high-resolution ground-based spectrographs offer an abundance of exoplanet data. The accurate interpretation of such data requires a retrieval code that can rapidly explore complex parameter spaces describing a rich variety of atmospheric phenomena.

`POSEIDON` provides the capability to model and retrieve planets with inhomogeneous temperatures, compositions, and cloud properties (i.e. 2D or 3D models). Several studies have highlighted that not including these multidimensional effects can bias retrieval inferences [@Line:2016; e.g., @Caldas:2019; @MacDonald:2020; @Pluriel:2022]. However, existing open-source exoplanet retrieval codes assume 1D atmospheres for computational efficiency. `POSEIDON` therefore offers an open-source implementation of state-of-the-art multidimensional retrieval methods [see @MacDonald:2022 and MacDonald & Lewis (in prep.)] to aid the interpretation of high-quality exoplanet spectra.

# Documentation

Documentation for `POSEIDON`, with step-by-step tutorials illustrating research applications, is available at [https://poseidon-retrievals.readthedocs.io/en/latest/](https://poseidon-retrievals.readthedocs.io/en/latest/). 

# Similar Tools

[`PLATON`](https://github.com/ideasrule/platon) [@Zhang:2019], [`petitRADTRANS`](https://gitlab.com/mauricemolli/petitRADTRANS) [@Molliere:2019], [`CHIMERA`](https://github.com/mrline/CHIMERA) [@Line:2013], [`TauRex`](https://github.com/ucl-exoplanets/TauREx3_public) [@Waldmann:2015; @Al-Refaie:2021], [`Pyrat Bay`](https://github.com/pcubillos/pyratbay) [@Cubillos:2021], [`BART`](https://github.com/exosports/BART) [@Harrington:2022]

# Acknowledgements

RJM expresses gratitude to the developers of many open source Python packages used by `POSEIDON`, in particular `Numba` [@Lam:2015], `numpy` [@Harris:2020], `Matplotlib` [@Hunter:2007], `SciPy` [@Virtanen:2020], and `Spectres` [@Carnall:2017].

RJM acknowledges financial support from the UK's Science and Technology Facilities Council (STFC) during the early development of `POSEIDON`. RJM thanks Nikole Lewis, Ishan Mishra, Jonathan Gomez Barrientos, John Kappelmeier, Antonia Peters, Kath Landgren, and Ruizhe Wang for helpful discussions.

# References
