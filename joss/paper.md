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
    affiliation: "1, 2, 3"
affiliations:
  - name: Department of Astronomy, University of Michigan, 1085 S. University Ave., Ann Arbor, MI 48109, USA
    index: 1
  - name: NHFP Sagan Fellow
    index: 2
  - name: Department of Astronomy and Carl Sagan Institute, Cornell University, 122 Sciences Drive, Ithaca, NY 14853, USA
    index: 3
date: 04 January 2023
bibliography: paper.bib

aas-doi: 10.3847/1538-4357/ac47fe
aas-journal: Astrophysical Journal

--- 

# Summary

Exoplanet atmospheres are a dynamic and fast-changing field at the frontier of modern astronomy. Telescope observations can reveal the chemical composition, temperature, cloud properties, and (potentially) the habitability of these remote worlds. Astronomers can measure these atmospheric properties by observing how the fraction of starlight blocked by a planet passing in front of its host star changes with wavelength --- a technique called transmission spectroscopy. Since the wavelengths where different atoms and molecules absorb are already known (from laboratory measurements or quantum mechanics), astronomers can compare models of exoplanet spectra to observations to infer the chemical composition of exoplanets.

`POSEIDON` is a Python package for the modelling and analysis of exoplanet spectra. `POSEIDON` has two main functions: (i) computation of model spectra for 1D, 2D, or 3D exoplanet atmospheres; and (ii) a Bayesian fitting routine (`atmospheric retrieval') that can infer the range of atmospheric properties consistent with an observed exoplanet spectrum.

# Exoplanet Modelling and Atmospheric Retrieval with `POSEIDON`

The first major use case for `POSEIDON` is 'forward modelling' --- illustrated on the left of \autoref{fig:POSEIDON_architecture}. A user can generate a model planet spectrum, for a given star-planet system, by providing a specific set of atmospheric properties (e.g. the chemical composition and temperature). The forward model mode allows users to explore how atmospheric properties alter an exoplanet spectrum and to produce predicted model spectra for observing proposals. The required input files (pre-computed stellar grids and an opacity database) are available to download from an online repository (linked in the documentation).

The second major use case for `POSEIDON` is atmospheric retrieval --- illustrated on the right of \autoref{fig:POSEIDON_architecture}. To initialise a retrieval, a user provides an observed exoplanet spectrum and the range of atmospheric properties to be explored (i.e. the prior ranges for a set of free parameters defining a model). A Bayesian statistical sampling algorithm --- nominally [`PyMultiNest`](https://github.com/JohannesBuchner/PyMultiNest) [@Buchner:2014] --- then repeatedly calls the forward model, comparing the generated spectrum to the observations, until the parameter space is fully explored and a convergence criteria reached. The main outputs of an atmospheric retrieval are the posterior probability distributions of the model parameters and the model's Bayesian evidence. The Bayesian evidence from multiple retrievals, in turn, can be subsequently compared to compute detection significance for each model component (e.g. the statistical confidence for a molecule being present in the planetary atmosphere).

![Schematic architecture of the `POSEIDON` atmospheric retrieval code. Users can call `POSEIDON` in two main ways: (i) to generate a model exoplanet spectrum for a specified planet atmosphere (green arrows); or (ii) to fit an observed exoplanet spectrum by statistical sampling of a model's atmospheric properties (purple arrows). The diagram highlights code inputs (circles), algorithm steps (rectangles), and code outputs (bottom green or purple boxes). \label{fig:POSEIDON_architecture}](figures/POSEIDON_Architecture_2022){width=100%}

`POSEIDON` was first described in the exoplanet literature by [@MacDonald:2017]. Since then, the code has been used in 17 peer-reviewed publications [e.g., @Alam:2021; @Sedaghati:2017; @Kaltenegger:2020]. Most recently, a detailed description of `POSEIDON`'s new multidimensional forward model, `TRIDENT`, was provided by [@MacDonald:2022].

# Statement of Need

Recent years have seen a substantial improvement in the number of high-quality exoplanet spectra. In particular, the newly operational JWST and a profusion of high-resolution ground-based spectrographs offer an abundance of exoplanet data. The accurate interpretation of such data requires a retrieval code that can rapidly explore complex parameter spaces describing a rich variety of atmospheric phenomena.

`POSEIDON` provides the capability to model and retrieve transmission spectra of planets with inhomogeneous temperatures, compositions, and cloud properties (i.e. 2D or 3D models). Several studies have highlighted that not including these multidimensional effects can bias retrieval inferences [e.g., @Caldas:2019; @Line:2016; @MacDonald:2020; @Pluriel:2022]. However, existing open-source exoplanet retrieval codes assume 1D atmospheres for computational efficiency. `POSEIDON`, therefore, offers an open-source implementation of state-of-the-art multidimensional retrieval methods [see @MacDonald:2022 and MacDonald & Lewis, in prep.] to aid the interpretation of high-quality exoplanet spectra.

In a 1D configuration, `POSEIDON` compares well with other retrieval codes. When applied to Hubble Space Telescope observations, `POSEIDON` produces consistent retrieval results with the ATMO and NEMESIS retrieval codes [@Lewis:2020; @Rathcke:2021]. Recently, [@Barstow:2022] presented a comparison of five exoplanet retrieval codes, including `POSEIDON`, which demonstrated good agreement on simulated Ariel [@Tinetti:2020] transmission spectra. `POSEIDON` also offers exceptional computational performance: a single 1D forward model over a wavelength range sufficient for JWST analyses takes 70 ms [see @MacDonald:2022, Appendix D], while publication-quality 1D retrievals typically take an hour or less. `POSEIDON` also supports multi-core retrievals via `PyMultiNest`'s MPI implementation, which achieves a roughly linear speed-up in the number of cores. Therefore, `POSEIDON` allows users to readily explore 1D retrievals on personal laptops while scaling up to multidimensional retrievals on modest clusters.

# Future Developments

`POSEIDON` v1.0 officially supports the modelling and retrieval of exoplanet transmission spectra in 1D, 2D, and 3D. The initial release also includes a beta version of thermal emission spectra modelling and retrieval (for cloud-free, 1D atmospheres, with no scattering), which will be developed further in future releases. Suggestions for additional features are more than welcome.

# Documentation

Documentation for `POSEIDON`, with step-by-step tutorials illustrating research applications, is available at [https://poseidon-retrievals.readthedocs.io/en/latest/](https://poseidon-retrievals.readthedocs.io/en/latest/). 

# Similar Tools

The following exoplanet retrieval codes are open source: [`PLATON`](https://github.com/ideasrule/platon) [@Zhang:2019; @Zhang:2020], [`petitRADTRANS`](https://gitlab.com/mauricemolli/petitRADTRANS) [@Molliere:2019], [`CHIMERA`](https://github.com/mrline/CHIMERA) [@Line:2013], [`TauRex`](https://github.com/ucl-exoplanets/TauREx3_public) [@Waldmann:2015; @Al-Refaie:2021], [`NEMESIS`](https://github.com/nemesiscode/radtrancode) [@Irwin:2008] [`Pyrat Bay`](https://github.com/pcubillos/pyratbay) [@Cubillos:2021], and [`BART`](https://github.com/exosports/BART) [@Harrington:2022]

# Acknowledgements

RJM expresses gratitude to the developers of many open source Python packages used by `POSEIDON`, in particular `Numba` [@Lam:2015], `numpy` [@Harris:2020], `Matplotlib` [@Hunter:2007], `SciPy` [@Virtanen:2020], and `Spectres` [@Carnall:2017].

RJM acknowledges financial support from the UK's Science and Technology Facilities Council (STFC) during the early development of `POSEIDON` and support from NASA Grant 80NSSC20K0586 issued through the James Webb Space Telescope Guaranteed Time Observer Program. Most recently, RJM acknowledges support from NASA through the NASA Hubble Fellowship grant HST-HF2-51513.001 awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., for NASA, under contract NAS5-26555. RJM thanks Nikole Lewis, Ishan Mishra, Jonathan Gomez Barrientos, John Kappelmeier, Antonia Peters, Kath Landgren, and Ruizhe Wang for helpful discussions.

# References
