''' 
POSEIDON CORE ROUTINE.

Copyright 2023, Ryan J. MacDonald.

'''

import os

# Force a single core to be used by numpy (mpirun handles parallelisation)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CBLAS_NUM_THREADS'] = '1'

# These settings only used for GPU models (experimental)
os.environ['block'] = '128'
os.environ['thread'] = '128'

import numpy as np
from numba.core.decorators import jit
import scipy.constants as sc
from mpi4py import MPI
from spectres import spectres
from scipy.constants import parsec

from .constants import R_J, R_E
from .utility import create_directories, write_spectrum, read_data
from .stellar import planck_lambda, load_stellar_pysynphot, load_stellar_pymsg, \
                     open_pymsg_grid
from .supported_chemicals import supported_species, supported_cia, inactive_species, \
                                 fastchem_supported_species, aerosol_supported_species
from .parameters import assign_free_params, generate_state, \
                        unpack_geometry_params, unpack_cloud_params
from .absorption import opacity_tables, store_Rayleigh_eta_LBL, extinction,\
                        extinction_LBL, extinction_GPU, extinction_spectrum_contribution, extinction_spectrum_pressure_contribution
from .geometry import atmosphere_regions, angular_grids
from .atmosphere import profiles
from .instrument import init_instrument
from .transmission import TRIDENT
from .emission import emission_rad_transfer, determine_photosphere_radii, \
                      emission_rad_transfer_GPU, determine_photosphere_radii_GPU

from .clouds_aerosols import Mie_cloud, load_aerosol_grid
from .clouds_LX_MIE_emission import Mie_cloud_free

from .utility import mock_missing

try:
    import cupy as cp
except ImportError:
    cp = mock_missing('cupy')


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

block = int(os.environ['block'])
thread = int(os.environ['thread'])

import warnings

warnings.filterwarnings("ignore") # Suppress numba performance warning


def create_star(R_s, T_eff, log_g, Met, T_eff_error = 100.0, log_g_error = 0.1,
                stellar_grid = 'blackbody', stellar_contam = None, 
                f_het = None, T_het = None, log_g_het = None, 
                f_spot = None, f_fac = None, T_spot = None,
                T_fac = None, log_g_spot = None, log_g_fac = None,
                wl = [], interp_backend = 'pysynphot'):
    '''
    Initialise the stellar dictionary object used by POSEIDON.

    Stellar spectra are only required to compute transmission spectra if the
    star has unocculted stellar heterogeneities (spots/faculae) - since the 
    stellar intensity cancels out in the transit depth. Hence a stellar 
    spectrum is only added to the stellar dictionary if the user requests it.

    Args:
        R_s (float): 
            Stellar radius (m).
        T_eff (float):
            Stellar effective temperature (K).
        log_g (float):
            Stellar log surface gravity (log10(cm/s^2) by convention).
        Met (float):
            Stellar metallicity [log10(Fe/H_star / Fe/H_solar)].
        T_eff_error (float):
            A priori 1-sigma error on stellar effective temperature (K).
        T_eff_error (float):
            A priori 1-sigma error on stellar log g (log10(cm/s^2)).
        stellar_grid (string):
            Chosen stellar model grid
            (Options: blackbody / cbk04 [for pysynphot] / phoenix [for pysynphot] /
                      Goettingen-HiRes [for pymsg]).
        stellar_contam (str):
            Chosen prescription for modelling unocculted stellar contamination
            (Options: one_spot / one_spot_free_log_g / two_spots).
        f_het (float):
            For the 'one_spot' model, the fraction of stellar photosphere 
            covered by either spots or faculae.
        T_het (float):
            For the 'one_spot' model, the temperature of the heterogeneity (K).
        log_g_het (float):
            For the 'one_spot' model, the log g of the heterogeneity (log10(cm/s^2)).
        f_spot (float):
            For the 'two_spots' model, the fraction of stellar photosphere 
            covered by spots.
        f_fac (float):
            For the 'two_spots' model, the fraction of stellar photosphere 
            covered by faculae.
        T_spot (float):
            For the 'two_spots' model, the temperature of the spot (K).
        T_fac (float):
            For the 'two_spots' model, the temperature of the facula (K).
        log_g_spot (float):
            For the 'two_spots' model, the log g of the spot (log10(cm/s^2)).
        log_g_fac (float):
            For the 'two_spots' model, the log g of the facula (log10(cm/s^2)).
        wl (np.array of float):
            Wavelength grid on which to output the stellar spectra (μm). If not
            provided, a fiducial grid from 0.2 to 5.4 μm will be used.
        interp_backend (str):
            Stellar grid interpolation package for POSEIDON to use.
            (Options: pysynphot / pymsg).
    
    Returns:
        star (dict):
            Collection of stellar properties used by POSEIDON.

    '''

    # If the user did not specify a wavelength grid for the stellar spectrum 
    if (wl == []):

        # Create fiducial wavelength grid
        wl_min = 0.2  # μm
        wl_max = 5.4  # μm
        R = 20000     # Spectral resolution (wl / dwl)

        wl_star = wl_grid_constant_R(wl_min, wl_max, R)

    else:
        wl_star = wl

    # Compute stellar spectrum (not used for uncontaminated transmission spectra)
    if (stellar_grid == 'blackbody'):

        if (stellar_contam != None):
            raise Exception("Error: cannot use black bodies for a model " +
                            "with heterogeneities, please specify a stellar grid.")

        # Evaluate Planck function at stellar effective temperature
        I_phot = planck_lambda(T_eff, wl_star)

    else:

        if (interp_backend not in ['pysynphot', 'pymsg']):
            raise Exception("Error: supported stellar grid interpolater backends " +
                            "are 'pysynphot' or 'pymsg'.")

        # Obtain photosphere spectrum from pysynphot
        if (interp_backend == 'pysynphot'):

            # Load stellar model from Pysynphot
            I_phot = load_stellar_pysynphot(wl_star, T_eff, Met, log_g, stellar_grid)

        # Obtain photosphere spectrum from pymsg
        elif (interp_backend == 'pymsg'):

            # Open pymsg grid file
            specgrid = open_pymsg_grid(stellar_grid)

            # Interpolate stellar grid to compute photosphere intensity
            I_phot = load_stellar_pymsg(wl_star, specgrid, T_eff, Met, log_g)

    # For uniform stellar surfaces
    if (stellar_contam == None): 

        # Surface flux is pi * intensity
        F_star = np.pi * I_phot

        # No heterogeneity spectra to return 
        I_het = None
        I_spot = None
        I_fac = None 

    # For non-uniform stellar surfaces
    elif ('one_spot' in stellar_contam):

        # If log g not specified for the heterogeneities, set to photosphere
        if (log_g_het == None):
            log_g_het = log_g

        # Individual spot and faculae intensities not needed for one spot model
        I_spot = None
        I_fac = None

        # Obtain heterogeneity spectrum
        if (interp_backend == 'pysynphot'):
            I_het = load_stellar_pysynphot(wl_star, T_het, Met, log_g_het, stellar_grid)
        elif (interp_backend == 'pymsg'):
            I_het = load_stellar_pymsg(wl_star, specgrid, T_het, Met, log_g_het)

        # Evaluate total stellar flux as a weighted sum of each region 
        F_star = np.pi * ((f_het * I_het) + (1.0 - f_het) * I_phot)

    # For non-uniform stellar surfaces
    elif ('two_spots' in stellar_contam):

        # If log g not specified for the heterogeneities, set to photosphere
        if (log_g_spot == None):
            log_g_spot = log_g
        if (log_g_fac == None):
            log_g_fac = log_g

        # Check provided temperatures are physical
        if (T_spot > T_fac):
            raise Exception("Error: spots must be cooler than faculae")
        if (T_spot > T_eff):
            raise Exception("Error: spots must be cooler than the photosphere")
        if (T_fac < T_eff):
            raise Exception("Error: faculae must be hotter than the photosphere")

        # Single heterogeneity intensity not needed for two spot model
        I_het = None

        # Obtain spot and facula spectra
        if (interp_backend == 'pysynphot'):
            I_spot = load_stellar_pysynphot(wl_star, T_spot, Met, log_g_spot, stellar_grid)
            I_fac = load_stellar_pysynphot(wl_star, T_fac, Met, log_g_fac, stellar_grid)
        elif (interp_backend == 'pymsg'):
            I_spot = load_stellar_pymsg(wl_star, specgrid, T_spot, Met, log_g_spot)
            I_fac = load_stellar_pymsg(wl_star, specgrid, T_fac, Met, log_g_fac)

        # Evaluate total stellar flux as a weighted sum of each region 
        F_star = np.pi * ((f_spot * I_spot) + (f_fac * I_fac) + 
                            (1.0 - (f_spot + f_fac)) * I_phot)
        
    else:
        raise Exception("Error: unsupported heterogeneity type. Supported " +
                        "types are: None, 'one_spot', 'two_spots'")

    # Package stellar properties
    star = {'R_s': R_s, 'T_eff': T_eff, 'T_eff_error': T_eff_error,
            'log_g_error': log_g_error, 'Met': Met, 'log_g': log_g, 
            'F_star': F_star, 'wl_star': wl_star,
            'f_het': f_het, 'T_het': T_het, 'log_g_het': log_g_het, 
            'f_spot': f_spot, 'T_spot': T_het, 'log_g_spot': log_g_het,
            'f_fac': f_fac, 'T_fac': T_het, 'log_g_fac': log_g_het,
            'I_phot': I_phot, 'I_het': I_het, 'I_spot': I_spot, 'I_fac': I_fac,
            'stellar_grid': stellar_grid, 'stellar_interp_backend': interp_backend,
            'stellar_contam': stellar_contam,
           }

    return star


def create_planet(planet_name, R_p, mass = None, gravity = None, 
                  log_g = None, T_eq = None, d = None, d_err = None, b_p = 0.0):
    '''
    Initialise the planet dictionary object used by POSEIDON.

    The user only need provide one out of 'mass', 'gravity', or 'log_g'
    (depending on their unit of preference). Note that 'gravity' follows SI
    units (m/s^2), whilst 'log_g' follows the CGS convention (log_10 cm/s^2).

    Args:
        planet_name (str):
            Identifier for planet object (e.g. HD209458b).
        R_p (float): 
            Planetary radius (m).
        mass (float):
            Planetary mass (kg).
        gravity (float):
            Planetary gravity corresponding to observed radius (m/s^2).
        log_g (float):
            Instead of g, the user can provide log_10 (g / cm/s^2).
        T_eq (float):
            Planetary equilibrium temperature (zero albedo) (K). 
        d (float):
            Distance to system (m).
        d_err (float):
            Measured error on system distance (m).
        b_p (float):
            Impact parameter of planetary orbit (m) -- NOT in stellar radii!
    
    Returns:
        planet (dict):
            Collection of planetary properties used by POSEIDON.

    '''

    base_dir = './'

    # Create output directories (the first core handles this)
    if (rank == 0):
        create_directories(base_dir, planet_name)

    # Calculate g_p or M_p if the user only provided one of the pair
    if ((gravity is None) and (log_g is None) and (mass is None)):
        raise Exception("At least one of Mass or gravity must be specified.")

    if (gravity is None):
        if (log_g is not None):
            gravity = np.power(10.0, log_g)/100   # Convert log cm/s^2 to m/s^2
        elif ((log_g is None) and (mass is not None)):
            gravity = (sc.G * mass) / (R_p**2)    # Compute gravity from mass

    if ((mass is None) and (gravity is not None)):
        mass = (gravity * R_p**2) / sc.G

    # Package planetary properties
    planet = {'planet_name': planet_name, 'planet_radius': R_p, 
              'planet_mass': mass, 'planet_gravity': gravity, 
              'planet_T_eq': T_eq, 'planet_impact_parameter': b_p,
              'system_distance': d, 'system_distance_error': d_err
             }

    return planet


def define_model(model_name, bulk_species, param_species,
                 object_type = 'transiting', PT_profile = 'isotherm', 
                 X_profile = 'isochem', cloud_model = 'cloud-free', 
                 cloud_type = 'deck', opaque_Iceberg = False,
                 gravity_setting = 'fixed', stellar_contam = None, 
                 offsets_applied = None, error_inflation = None, 
                 radius_unit = 'R_J', distance_unit = 'pc',
                 PT_dim = 1, X_dim = 1, cloud_dim = 1, TwoD_type = None, 
                 TwoD_param_scheme = 'difference', species_EM_gradient = [], 
                 species_DN_gradient = [], species_vert_gradient = [],
                 surface = False, sharp_DN_transition = False,
                 reference_parameter = 'R_p_ref', disable_atmosphere = False,
                 aerosol_species = ['free']):
    '''
    Create the model dictionary defining the configuration of the user-specified 
    forward model or retrieval.

    Args:
        model_name (str):
            Identifier for model in output files and plots.
        bulk_species (list of str):
            The chemical species (or two for H2+He) filling most of the atmosphere.
        param_species (list of str):
            Chemical species with parametrised mixing ratios (trace species).
        object_type (str):
            Type of planet / brown dwarf the user wishes to model
            (Options: transiting / directly_imaged).
        PT_profile (str):
            Chosen P-T profile parametrisation 
            (Options: isotherm / gradient / two-gradients / Madhu / slope /
             file_read).
        X_profile (str):
            Chosen mixing ratio profile parametrisation
            (Options: isochem / gradient / two-gradients / file_read / chem_eq).
        cloud_model (str):
            Chosen cloud parametrisation 
            (Options: cloud-free / MacMad17 / Iceberg / Mie).
        cloud_type (str):
            Cloud extinction type to consider 
            (Options: deck / haze / deck_haze).
        opaque_Iceberg (bool):
            If using the Iceberg cloud model, True disables the kappa parameter.
        gravity_setting (str):
            Whether log_g is fixed or a free parameter.
            (Options: fixed / free).
        stellar_contam (str):
            Chosen prescription for modelling unocculted stellar contamination
            (Options: one_spot / one_spot_free_log_g / two_spots / 
             two_spots_free_log_g).
        offsets_applied (str):
            Whether a relative offset should be applied to a dataset 
            (Options: single_dataset).
        error_inflation (str):
            Whether to consider inflation of error bars in a retrieval
            (Options: Line15).
        radius_unit (str)
            Planet radius unit used to report retrieval results
            (Options: R_J / R_E)
        distance_unit (str):
            Distance to system unit used to report retrieval results
            (Options: pc)
        PT_dim (int):
            Dimensionality of the pressure-temperature field (uniform -> 1, 
            a day-night or evening-morning gradient -> 2, both day-night and 
            evening-morning gradients -> 3)
            (Options: 1 / 2 / 3).
        X_dim (int):
            Max dimensionality of the mixing ratio field (not all species need
            have gradients, this just specifies the highest dimensionality of 
            chemical gradients -- see the species_XX_gradient arguments)
            (Options: 1 / 2 / 3).
        cloud_dim (int):
            Dimensionality of the cloud model prescription (only the Iceberg
            cloud model supports 3D clouds)
            (Options: 1 / 2 / 3).
        TwoD_type (str):
            For 2D models, specifies whether the model considers day-night
            gradients or evening-morning gradients
            (Options: D-N / E-M).
        TwoD_param_scheme (str):
            For 2D models, specifies which quantities should be consider as
            free parameters (e.g. day & night vs. terminator & day-night diff.)
            (Options: absolute / difference).
        species_EM_gradient (list of str):
            List of chemical species with an evening-morning mixing ratio gradient.
        species_DN_gradient (list of str):
            List of chemical species with a day-night mixing ratio gradient.
        species_vert_gradient (list of str):
            List of chemical species with a vertical mixing ratio gradient.
        surface (bool):
            If True, model a surface via an opaque cloud deck.
        sharp_DN_transition (bool):
            For 2D / 3D models, sets day-night transition width (beta) to 0.
        reference_parameter (str):
            For retrievals, whether R_p_ref or P_ref will be a free parameter
            (Options: R_p_ref / P_ref).
        disable_atmosphere (bool):
            If True, returns a flat planetary transmission spectrum @ (Rp/R*)^2
        aerosol (string):
            If cloud_model = Mie and clod_type = specific_aerosol 

    Returns:
        model (dict):
            Dictionary containing the description of the desired POSEIDON model.

    '''

    #***** Create chemical species arrays *****#

    # Create array containing all chemical species in model
    bulk_species = np.array(bulk_species)
    param_species = np.array(param_species)

    # For chemical equilibrium models, find the necessary chemical species
    if (X_profile == 'chem_eq'):
        supported_chem_eq_species = np.intersect1d(supported_species, 
                                                    fastchem_supported_species)
        
        # If param_species = ['all'] then default to all species
        if ('all' in param_species):
            param_species = supported_chem_eq_species

        # Check all user-specified species are compatible with the fastchem grid
        else:
            if (np.any(~np.isin(param_species, supported_chem_eq_species)) == True):
                raise Exception("A chemical species you selected is not supported " +
                                "for equilibrium chemistry models.\n")
            
    # Combine bulk species with parametrised species
    chemical_species = np.append(bulk_species, param_species)

    # Identify chemical species with active spectral features
    active_species = chemical_species[~np.isin(chemical_species, inactive_species)]

    # Convert arrays specifying which species have gradients into numpy arrays
    species_vert_gradient = np.array(species_vert_gradient)

    # Check if cross sections are available for all the chemical species
    if (np.any(~np.isin(active_species, supported_species)) == True):
        raise Exception("A chemical species you selected is not supported.\n")
    
    # Check to make sure an aerosol is inputted if cloud_type = specific_aerosol
    if (np.any(~np.isin(aerosol_species, aerosol_supported_species)) == True) and aerosol_species != ['free'] and aerosol_species != ['file_read']:
        raise Exception('Please input supported aerosols (check supported_opac.py) or aerosol = [\'free\'] or [\'file_read\'].')

    # Create list of collisionally-induced absorption (CIA) pairs
    CIA_pairs = []
    for pair in supported_cia:
        pair_1, pair_2 = pair.split('-')   
        if (pair_1 in chemical_species) and (pair_2 in chemical_species):
            CIA_pairs.append(pair)     
    CIA_pairs = np.array(CIA_pairs)

    # Create list of free-free absorption pairs
    ff_pairs = []
    if ('H' in chemical_species) and ('e-' in chemical_species):  
        ff_pairs.append('H-ff')       # H- free-free    
    ff_pairs = np.array(ff_pairs)

    # Create list of bound-free absorption species
    bf_species = []
    if ('H-' in chemical_species):  
        bf_species.append('H-bf')      # H- bound-free    
    bf_species = np.array(bf_species)

    #***** Geometrical properties of background atmosphere *****#

    # Find dimensionality of background atmosphere
    Atmosphere_dimension = max(PT_dim, X_dim)

    #***** Finally, identify the free parameters defining this model *****#

    param_names, physical_param_names, \
    PT_param_names, X_param_names, \
    cloud_param_names, geometry_param_names, \
    stellar_param_names, \
    N_params_cum = assign_free_params(param_species, object_type, PT_profile,
                                      X_profile, cloud_model, cloud_type, 
                                      gravity_setting, stellar_contam, 
                                      offsets_applied, error_inflation, PT_dim, 
                                      X_dim, cloud_dim, TwoD_type, TwoD_param_scheme, 
                                      species_EM_gradient, species_DN_gradient, 
                                      species_vert_gradient, Atmosphere_dimension,
                                      opaque_Iceberg, surface, sharp_DN_transition,
                                      reference_parameter, disable_atmosphere, aerosol_species)
    
    # If cloud_model = Mie, load in the cross section 
    if cloud_model == 'Mie' and aerosol_species != ['free'] and aerosol_species != ['file_read']:
        aerosol_grid = load_aerosol_grid(aerosol_species)
    else:
        aerosol_grid = None
        

    # Package model properties
    model = {'model_name': model_name, 'object_type': object_type,
             'Atmosphere_dimension': Atmosphere_dimension,
             'PT_profile': PT_profile, 'X_profile': X_profile,
             'cloud_model': cloud_model, 'cloud_type': cloud_type,
             'gravity_setting': gravity_setting, 
             'chemical_species': chemical_species, 'bulk_species': bulk_species,
             'active_species': active_species, 'CIA_pairs': CIA_pairs,
             'ff_pairs': ff_pairs, 'bf_species': bf_species,
             'param_species': param_species, 
             'radius_unit': radius_unit, 'distance_unit': distance_unit,
             'species_EM_gradient': species_EM_gradient,
             'species_DN_gradient': species_DN_gradient,
             'species_vert_gradient': species_vert_gradient,
             'stellar_contam': stellar_contam, 
             'offsets_applied': offsets_applied, 
             'error_inflation': error_inflation, 'param_names': param_names,
             'physical_param_names': physical_param_names, 
             'PT_param_names': PT_param_names, 'X_param_names': X_param_names, 
             'cloud_param_names': cloud_param_names,
             'geometry_param_names': geometry_param_names, 
             'stellar_param_names': stellar_param_names, 
             'N_params_cum': N_params_cum, 'TwoD_type': TwoD_type, 
             'TwoD_param_scheme': TwoD_param_scheme, 'PT_dim': PT_dim,
             'X_dim': X_dim, 'cloud_dim': cloud_dim, 'surface': surface,
             'sharp_DN_transition': sharp_DN_transition,
             'reference_parameter': reference_parameter,
             'disable_atmosphere': disable_atmosphere,
             'aerosol_species': aerosol_species,
             'aerosol_grid': aerosol_grid}

    return model


def wl_grid_constant_R(wl_min, wl_max, R):
    '''
    Create a wavelength array with constant spectral resolution (R = wl/dwl).

    Args:
        wl_min (float):
            Minimum wavelength of grid (μm).
        wl_max (float): 
            Maximum wavelength of grid (μm).
        R (int or float):
            Spectral resolution of desired wavelength grid.
    
    Returns:
        wl (np.array of float):
            Model wavelength grid (μm).

    '''

    # Constant R -> uniform in log(wl)
    delta_log_wl = 1.0/R
    N_wl = (np.log(wl_max) - np.log(wl_min)) / delta_log_wl
    N_wl = np.around(N_wl).astype(np.int64)
    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), N_wl)    

    wl = np.exp(log_wl)
    
    return wl


def wl_grid_line_by_line(wl_min, wl_max, line_by_line_res = 0.01):
    '''
    Create a wavelength array with constant spectral resolution (R = wl/dwl).

    Args:
        wl_min (float):
            Minimum wavelength of grid (μm).
        wl_max (float): 
            Maximum wavelength of grid (μm).
        line_by_line_res (float):
            Wavenumber resolution of pre-computer opacity database (0.01 cm^-1).
    
    Returns:
        wl (np.array of float):
            Model wavelength grid (μm).

    '''

    nu_min = 1.0e4/wl_max   # Minimum wavenumber on output grid
    nu_max = 1.0e4/wl_min   # Maximum wavenumber on output grid
    
    # Round so wavenumber grid bounds match cross section resolution (0.01 cm^-1)
    nu_min = np.around(nu_min, np.abs(np.int(np.log10(line_by_line_res))))
    nu_max = np.around(nu_max, np.abs(np.int(np.log10(line_by_line_res))))
    
    # Find number of wavenumber points on grid
    N_nu = np.int((nu_max - nu_min)/line_by_line_res)
    
    # Initialise line-by-line model wavenumber grid
    nu = np.linspace(nu_max, nu_min, N_nu)                          # Decreasing wavenumber order
    nu = np.around(nu, np.abs(np.int(np.log10(line_by_line_res))))  # Remove floating point errors
    
    # Initialise corresponding wavelength grid 
    wl = 1.0e4/nu   # Convert from cm^-1 to um
    
    return wl
    

def read_opacities(model, wl, opacity_treatment = 'opacity_sampling', 
                   T_fine = None, log_P_fine = None, opacity_database = 'High-T',
                   device = 'cpu', wl_interp = 'sample', testing = False):
    '''
    Load the various cross sections required by a given model. When using 
    opacity sampling, the native high-resolution are pre-interpolated onto 
    'fine' temperature and pressure grids, then sampled onto the desired
    wavelength grid, and stored in memory. This removes the need to interpolate 
    opacities during a retrieval. For line-by-line models, this function only
    stores Rayleigh scattering cross sections in memory (cross section 
    interpolation is handled in other functions later).

    Args:
        model (dict):
            A specific description of a given POSEIDON model.
        wl (np.array of float):
            Model wavelength grid (μm).
        opacity_treatment (str):
            Choice between opacity sampling or line-by-line cross sections
            (Options: opacity_sampling / line_by_line).
        T_fine (np.array of float):
            Fine temperature grid for opacity pre-interpolation.
        log_P_fine (np.array of float):
            Fine pressure grid for opacity pre-interpolation.
        opacity_database (str):
            Choice between high-temperature or low-temperature opacity databases
            (Options: High-T / Temperate).
        wl_interp (str):
            When initialising cross sections, whether to use opacity sampling
            or linear interpolation over wavenumber
            (Options: sample / linear) .
        testing (bool):
            For GitHub Actions automated tests. If True, disables reading the 
            full opacity database (since GitHub Actions can't handle downloading 
            the full database - alas, 30+ GB is a little too large!).
    
    Returns:
        opac (dict):
            Collection of numpy arrays storing cross sections for the molecules, 
            atoms, and ions contained in the model. The separate arrays store 
            standard cross sections, CIA, free-free + bound-free opacity, 
            and Rayleigh scattering cross sections.
    
    '''

    # Unpack lists of chemical species for which we need opacities
    chemical_species = model['chemical_species'] 
    active_species = model['active_species']
    CIA_pairs = model['CIA_pairs']
    ff_pairs = model['ff_pairs']
    bf_species = model['bf_species']
    
    # For opacity sampling, pre-compute opacities
    if (opacity_treatment == 'opacity_sampling'):

        # Check that a fine temperature and pressure grid have been specified
        if ((T_fine is None) or (log_P_fine is None)):
            raise Exception("T_fine and log_P_fine must be provided for " +
                             "pre-interpolation when using opacity sampling.")

        # Read and interpolate cross sections in pressure, temperature and wavelength
        sigma_stored, CIA_stored, \
        Rayleigh_stored, eta_stored, \
        ff_stored, bf_stored = opacity_tables(rank, comm, wl, chemical_species, 
                                              active_species, CIA_pairs, 
                                              ff_pairs, bf_species, T_fine,
                                              log_P_fine, opacity_database, 
                                              wl_interp, testing)
                    
    elif (opacity_treatment == 'line_by_line'):   
        
        # For line-by-line case, we still compute Rayleigh scattering in advance
        Rayleigh_stored, eta_stored = store_Rayleigh_eta_LBL(wl, chemical_species)   
        
        # No need for pre-computed arrays for line-by-line, so keep empty arrays
        sigma_stored, CIA_stored, \
        ff_stored, bf_stored = (np.array([]) for _ in range(4))

    # Move cross sections to GPU memory to speed up later computations
    if (device == 'gpu'):
        sigma_stored = cp.asarray(sigma_stored)
        CIA_stored = cp.asarray(CIA_stored)
        Rayleigh_stored = cp.asarray(Rayleigh_stored)
        eta_stored = cp.asarray(eta_stored)
        ff_stored = cp.asarray(ff_stored)
        bf_stored = cp.asarray(bf_stored)

    # Package opacity data required by our model in memory
    opac = {'opacity_database': opacity_database, 
            'opacity_treatment': opacity_treatment, 'sigma_stored': sigma_stored, 
            'CIA_stored': CIA_stored, 'Rayleigh_stored': Rayleigh_stored, 
            'eta_stored': eta_stored, 'ff_stored': ff_stored, 
            'bf_stored': bf_stored, 'T_fine': T_fine, 'log_P_fine': log_P_fine
           }

    return opac


def make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params = [],
                    log_X_params = [], cloud_params = [], geometry_params = [],
                    log_g = None, T_input = [], X_input = [], P_surf = None,
                    P_param_set = 1.0e-2, He_fraction = 0.17, 
                    N_slice_EM = 2, N_slice_DN = 4, constant_gravity = False,
                    chemistry_grid = None):
    '''
    Generate an atmosphere from a user-specified model and parameter set. In
    full generality, this function generates 3D pressure-temperature and mixing 
    ratio fields, the radial extent of atmospheric columns, geometrical 
    properties of the atmosphere, and cloud properties.

    Args:
        planet (dict):
            Collection of planetary properties used POSEIDON.
        model (dict):
            A specific description of a given POSEIDON model.
        P (np.array of float):
            Model pressure grid (bar).
        P_ref (float):
            Reference pressure (bar).
        R_p_ref (float):
            Planet radius corresponding to reference pressure (m).
        PT_params (np.array of float):
            Parameters defining the pressure-temperature field.
        log_X_params (np.array of float):
            Parameters defining the log-mixing ratio field.
        cloud_params (np.array of float):
            Parameters defining atmospheric aerosols.
        geometry_params (np.array of float):
            Terminator opening angle parameters.
        log_g (float):
            Gravitational field of planet - only needed for free log_g parameter.
        T_input (np.array of float):
            Temperature profile (only if provided directly by the user).
        X_input (2D np.array of float):
            Mixing ratio profiles (only if provided directly by the user).
        P_surf (float):
            Surface pressure of the planet.
        P_param_set (float):
            Only used for the Madhusudhan & Seager (2009) P-T profile.
            Sets the pressure where the reference temperature parameter is 
            defined (P_param_set = 1.0e-6 corresponds to that paper's choice).
        He_fraction (float):
            Assumed H2/He ratio (0.17 default corresponds to the solar ratio).
        N_slice_EM (even int):
            Number of azimuthal slices in the evening-morning transition region.
        N_slice_DN (even int):
            Number of zenith slices in the day-night transition region.
        constant_gravity (bool):
            If True, disable inverse square law gravity (only for testing).
        chemistry_grid (dict):
            For models with a pre-computed chemistry grid only, this dictionary
            is produced in chemistry.py.
    
    Returns:
        atmosphere (dict):
            Collection of atmospheric properties required to compute the
            resultant spectrum of the planet.
    
    '''

    # Unpack model properties
    Atmosphere_dimension = model['Atmosphere_dimension']
    TwoD_type = model['TwoD_type']
    param_names = model['param_names']
    N_params_cum = model['N_params_cum']
    param_species = model['param_species']
    X_profile = model['X_profile']
    X_dim = model['X_dim']
    TwoD_param_scheme = model['TwoD_param_scheme']
    species_EM_gradient = model['species_EM_gradient']
    species_DN_gradient = model['species_DN_gradient']
    species_vert_gradient = model['species_vert_gradient']
    PT_dim = model['PT_dim']
    PT_profile = model['PT_profile']
    cloud_model = model['cloud_model']
    cloud_dim = model['cloud_dim']
    gravity_setting = model['gravity_setting']
    sharp_DN_transition = model['sharp_DN_transition']
    aerosol_species = model['aerosol_species']

    # Unpack planet properties
    R_p = planet['planet_radius']

    # Load planet gravity
    if (gravity_setting == 'fixed'):
        g_p = planet['planet_gravity']   # For fixed g, load from planet object
    elif (gravity_setting == 'free'):
        if (log_g is None):
            raise Exception("Must provide 'log_g' when log_g is a free parameter")
        else:
            g_p = np.power(10.0, log_g)/100   # Convert log cm/s^2 to m/s^2

    # Unpack lists of chemical species in this model
    chemical_species = model['chemical_species']
    active_species = model['active_species']
    bulk_species = model['bulk_species']
    CIA_pairs = model['CIA_pairs']
    ff_pairs = model['ff_pairs']
    bf_species = model['bf_species']

    # Checks for validity of user inputs
    if (((T_input != []) or (X_input != [])) and Atmosphere_dimension > 1):
        raise Exception("Only 1D P-T and mixing ratio profiles are currently " +
                        "supported as user inputs. Multidimensional inputs " +
                        "will be added soon!")
    if ((PT_profile == 'file_read') and (T_input == [])):
        raise Exception("No user-provided P-T profile. Did you read in a file?")
    if ((X_profile == 'file_read') and (X_input == [])):
        raise Exception("No user-provided composition profile. Did you read in a file?")
    if ((cloud_params == []) and (cloud_model != 'cloud-free')):
        raise Exception("Cloud parameters must be provided for cloudy models.")
    if ((geometry_params == []) and (Atmosphere_dimension > 1) and
        (sharp_DN_transition == False)):
            raise Exception("Geometry parameters must be provided for 2D or 3D models.")

    #***** Establish model geometry *****# 

    # If user specifies a sharp day-night transition, use no transition slices
    if (sharp_DN_transition == True):
        N_slice_DN = 0

    # Check that the number of azimuthal and zenith slices are even
    if ((N_slice_EM % 2 != 0) or (N_slice_DN % 2 != 0)):
        raise Exception("The number of slices resolving the day-night or " +
                        "morning-evening transition regions must be even.")

    # For 2D or 3D models, find the number of atmospheric sectors and zones
    N_sectors, N_zones = atmosphere_regions(Atmosphere_dimension, TwoD_type, 
                                            N_slice_EM, N_slice_DN)

    # Unpack terminator opening angles (for 2D or 3D models)
    alpha, beta = unpack_geometry_params(param_names, geometry_params, N_params_cum)

    # Compute discretised angular grids for multidimensional atmospheres
    phi, theta, phi_edge, \
    theta_edge, dphi, dtheta = angular_grids(Atmosphere_dimension, TwoD_type, 
                                             N_slice_EM, N_slice_DN, 
                                             alpha, beta, sharp_DN_transition)

    #***** Generate state arrays for the PT and mixing ratio profiles *****#

    # Recast PT and mixing ratio parameters as state arrays used by atmosphere.py
    PT_state, \
    log_X_state = generate_state(PT_params, log_X_params, param_species, 
                                 PT_dim, X_dim, PT_profile, X_profile, TwoD_type, 
                                 TwoD_param_scheme, species_EM_gradient, 
                                 species_DN_gradient, species_vert_gradient,
                                 alpha, beta)

    #***** Compute P-T, radial, mixing ratio, and other atmospheric profiles *****#

    T, n, r, r_up, r_low, \
    dr, mu, X, X_active, \
    X_CIA, X_ff, X_bf, \
    is_physical = profiles(P, R_p, g_p, PT_profile, X_profile, PT_state, P_ref, 
                           R_p_ref, log_X_state, chemical_species, bulk_species, 
                           param_species, active_species, CIA_pairs, 
                           ff_pairs, bf_species, N_sectors, N_zones, alpha, 
                           beta, phi, theta, species_vert_gradient, He_fraction,
                           T_input, X_input, P_param_set, constant_gravity,
                           chemistry_grid)

    #***** Store cloud / haze / aerosol properties *****#

    kappa_cloud_0, P_cloud, \
    f_cloud, phi_cloud_0, \
    theta_cloud_0, \
    a, gamma, \
    r_m, log_n_max, fractional_scale_height, \
    r_i_real, r_i_complex, log_X_Mie = unpack_cloud_params(param_names, cloud_params, cloud_model, cloud_dim, 
                                   N_params_cum, TwoD_type)
    
    # Temporary H for testing 
    g = g_p * (R_p_ref / r)**2
    H = (sc.k * T) / (mu * g)

    # Package atmosphere properties
    atmosphere = {'P': P, 'T': T, 'g': g_p, 'n': n, 'r': r, 'r_up': r_up,
                  'r_low': r_low, 'dr': dr, 'P_surf': P_surf, 'X': X, 
                  'X_active': X_active, 'X_CIA': X_CIA, 'X_ff': X_ff,
                  'X_bf': X_bf, 'mu': mu, 'N_sectors': N_sectors, 
                  'N_zones': N_zones, 'alpha': alpha, 'beta': beta, 'phi': phi, 
                  'theta': theta, 'phi_edge': phi_edge, 'theta_edge': theta_edge,
                  'dphi': dphi, 'dtheta': dtheta, 'kappa_cloud_0': kappa_cloud_0, 
                  'P_cloud': P_cloud, 'f_cloud': f_cloud, 'phi_cloud_0': phi_cloud_0, 
                  'theta_cloud_0': theta_cloud_0, 'a': a, 'gamma': gamma, 
                  'is_physical': is_physical,
                  'H': H, 'r_m': r_m, 'log_n_max': log_n_max, 'fractional_scale_height': fractional_scale_height,
                  'aerosol_species': aerosol_species, 'r_i_real': r_i_real, 'r_i_complex': r_i_complex, 'log_X_Mie': log_X_Mie
                 }

    return atmosphere


def check_atmosphere_physical(atmosphere, opac):
    '''
    Checks that a specific model atmosphere is physical.

    Args:
        atmosphere (dict):
            Collection of atmospheric properties.
        opac (dict):
            Collection of cross sections and other opacity sources.

    Returns:
        Bool:
            True if atmosphere physical, otherwise False.
    
    '''

    # Reject if atmosphere already internally identified as unphysical
    if (atmosphere['is_physical'] == False):
        return False

    # Also check if temperature field is within bounds of fine temperature grid
    else:

        # Unpack atmospheric temperature field
        T = atmosphere['T']
        T_min = np.min(T)
        T_max = np.max(T)

        # Only need to check fine temperature grid when using opacity sampling
        if (opac['opacity_treatment'] == 'opacity_sampling'):

            # Unpack fine temperature grid (opacity sampling only)
            T_fine = opac['T_fine']
            T_fine_min = np.min(T_fine)
            T_fine_max = np.max(T_fine)
    
            # Check if minimum or maximum temperatures are outside opacity range
            if ((T_max > T_fine_max) or (T_min < T_fine_min)): 
                return False

            else:
                return True

        # For line-by-line models, there is not fine temperature grid
        else:
            return True
            

def compute_spectrum(planet, star, model, atmosphere, opac, wl,
                     spectrum_type = 'transmission', save_spectrum = False,
                     disable_continuum = False, suppress_print = False,
                     Gauss_quad = 2, use_photosphere_radius = True,
                     device = 'cpu', y_p = np.array([0.0])):
    '''
    Calculate extinction coefficients, then solve the radiative transfer 
    equation to compute the spectrum of the model atmosphere.

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        star (dict):
            Collection of stellar properties used by POSEIDON.
        model (dict):
            A specific description of a given POSEIDON model.
        atmosphere (dict):
            Collection of atmospheric properties.
        opac (dict):
            Collection of cross sections and other opacity sources.
        wl (np.array of float):
            Model wavelength grid (μm).
        spectrum_type (str):
            The type of spectrum for POSEIDON to compute
            (Options: transmission / emission / direct_emission / 
                      transmission_time_average).
        save_spectrum (bool):
            If True, writes the spectrum to './POSEIDON_output/PLANET/spectra/'.
        disable_continuum (bool):
            If True, turns off CIA and Rayleigh scattering opacities.
        suppress_print (bool):
            if True, turn off opacity print statements (in line-by-line mode).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            * Only for emission spectra *
            (Options: 2 / 3).
        use_photosphere_radius (bool):
            If True, use R_p at tau = 2/3 for emission spectra prefactor.
        device (str):
            Experimental: use CPU or GPU (only for emission spectra)
            (Options: cpu / gpu)
        y_p (np.array of float):
            Coordinate of planet centre along orbit at the time the spectrum
            is computed (y_p = 0, the default, corresponds to mid-transit).
            For non-grazing transits of uniform stellar disks, the spectrum
            is identical at all times due to translational symmetry, so y_p = 0
            is good for all times post second contact and pre third contact.
            Units are in m, not in stellar radii.

    Returns:
        spectrum (np.array of float):
            The spectrum of the atmosphere (transmission or emission).
    
    '''

    # Check if the atmosphere is unphysical (e.g. temperature out of bounds)
    if (check_atmosphere_physical(atmosphere, opac) == False):
        spectrum = np.empty(len(wl))
        spectrum[:] = np.NaN
        return spectrum   # Unphysical => reject model

    # Unpack model properties
    PT_dim = model['PT_dim']
    X_dim = model['X_dim']
    cloud_dim = model['cloud_dim']

    # Check that the requested spectrum model is supported
    if (spectrum_type not in ['transmission', 'emission', 'direct_emission',
                              'dayside_emission', 'nightside_emission',
                              'transmission_time_average']):
        raise Exception("Only transmission spectra and emission " +
                        "spectra are currently supported.")
    elif (('emission' in spectrum_type) and 
         ((PT_dim or X_dim or cloud_dim) == 3)):
        raise Exception("Only 1D or 2D emission spectra currently supported.")

    # Unpack planet and star properties
    b_p = planet['planet_impact_parameter']
    R_p = planet['planet_radius']
    b_p = planet['planet_impact_parameter']
    d = planet['system_distance']

    if (star is not None):
        R_s = star['R_s']

    # Check that a distance is provided if user wants a direct spectrum
    if (d is None) and ('direct' in spectrum_type):
        raise Exception("Must provide a system distance when computing a " +
                        "direct emission spectrum.")

    # Unpack atmospheric properties needed for radiative transfer
    r = atmosphere['r']
    r_low = atmosphere['r_low']
    r_up = atmosphere['r_up']
    dr = atmosphere['dr']
    n = atmosphere['n']
    T = atmosphere['T']
    P = atmosphere['P']
    P_surf = atmosphere['P_surf']
    X = atmosphere['X']
    X_active = atmosphere['X_active']
    X_CIA = atmosphere['X_CIA']
    X_ff = atmosphere['X_ff']
    X_bf = atmosphere['X_bf']
    N_sectors = atmosphere['N_sectors']
    N_zones = atmosphere['N_zones']
    phi_edge = atmosphere['phi_edge']
    theta_edge = atmosphere['theta_edge']
    a = atmosphere['a']
    gamma = atmosphere['gamma']
    P_cloud = atmosphere['P_cloud']
    kappa_cloud_0 = atmosphere['kappa_cloud_0']
    f_cloud = atmosphere['f_cloud']
    phi_cloud_0 = atmosphere['phi_cloud_0']
    theta_cloud_0 = atmosphere['theta_cloud_0']
    H = atmosphere['H']
    r_m = atmosphere['r_m']
    log_n_max = atmosphere['log_n_max']
    fractional_scale_height = atmosphere['fractional_scale_height']
    aerosol_species = atmosphere['aerosol_species']
    r_i_real = atmosphere['r_i_real']
    r_i_complex = atmosphere['r_i_complex']
    log_X_Mie = atmosphere['log_X_Mie']


    # Check if haze enabled in the cloud model
    if ('haze' in model['cloud_type']):
        enable_haze = 1
    else:
        enable_haze = 0

    # Check if a cloud deck is enabled in the cloud model
    if ('deck' in model['cloud_type']):
        enable_deck = 1
    else:
        enable_deck = 0

    # Check if a surface is enabled
    if (P_surf != None):
        enable_surface = 1
    else:
        enable_surface = 0
        P_surf = 100.0      # Set surface pressure to 100 bar if not defined

    #***** Calculate extinction coefficients *****#

    # Unpack lists of chemical species in this model
    chemical_species = model['chemical_species'] 
    active_species = model['active_species']
    CIA_pairs = model['CIA_pairs']
    ff_pairs = model['ff_pairs']
    bf_species = model['bf_species']
            
    # If computing line-by-line radiative transfer, use lbl optimised functions 
    if (opac['opacity_treatment'] == 'line_by_line'):

        # Identify the opacity database being used
        opacity_database = opac['opacity_database']

        # Unpack pre-computed Rayleigh cross sections
        Rayleigh_stored = opac['Rayleigh_stored']

        # Calculate extinction coefficients in line-by-line mode        
        kappa_clear, kappa_cloud = extinction_LBL(chemical_species, active_species, 
                                                  CIA_pairs, ff_pairs, bf_species, 
                                                  n, T, P, wl, X, X_active, X_CIA, 
                                                  X_ff, X_bf, a, gamma, P_cloud,
                                                  kappa_cloud_0, Rayleigh_stored, 
                                                  enable_haze, enable_deck,
                                                  enable_surface, N_sectors, 
                                                  N_zones, P_surf, opacity_database, 
                                                  disable_continuum, suppress_print)
        
    # If using opacity sampling, we can use pre-interpolated cross sections
    elif (opac['opacity_treatment'] == 'opacity_sampling'):

        # Unpack pre-interpolated cross sections
        sigma_stored = opac['sigma_stored']
        CIA_stored = opac['CIA_stored']
        Rayleigh_stored = opac['Rayleigh_stored']
        ff_stored = opac['ff_stored']
        bf_stored = opac['bf_stored']

        # Also unpack fine temperature and pressure grids from pre-interpolation
        T_fine = opac['T_fine']
        log_P_fine = opac['log_P_fine']

        # Running POSEIDON on the CPU
        if (device == 'cpu'):

            if (model['cloud_model'] == 'Mie'):

                aerosol_grid = model['aerosol_grid']

                # If its a fuzzy deck run
                if (model['cloud_type'] == 'fuzzy_deck'):

                    if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                        n_aerosol_array, sigma_Mie_array = Mie_cloud_free(P,wl,r, H, n,
                                                                            r_m,
                                                                            r_i_real,
                                                                            r_i_complex,
                                                                            P_cloud = P_cloud,
                                                                            log_n_max = log_n_max, 
                                                                            fractional_scale_height = fractional_scale_height,)

                    else : 
                        n_aerosol_array, sigma_Mie_array = Mie_cloud(P,wl,r, H, n,
                                                                    r_m, 
                                                                    aerosol_species,
                                                                    cloud_type = model['cloud_type'],
                                                                    aerosol_grid = aerosol_grid,
                                                                    P_cloud = P_cloud,
                                                                    log_n_max = log_n_max, 
                                                                    fractional_scale_height = fractional_scale_height,)
                        
                          
                # If its a uniform X run
                elif( model['cloud_type'] == 'uniform_X'):

                    if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                        n_aerosol_array, sigma_Mie_array = Mie_cloud_free(P,wl,r, H, n,
                                                                            r_m,
                                                                            r_i_real,
                                                                            r_i_complex,
                                                                            log_X_Mie = log_X_Mie)

                    else : 
                        n_aerosol_array, sigma_Mie_array = Mie_cloud(P,wl,r, H, n,
                                                                    r_m, 
                                                                    aerosol_species,
                                                                    cloud_type = model['cloud_type'],
                                                                    aerosol_grid = aerosol_grid,
                                                                    log_X_Mie = log_X_Mie)
  
                enable_Mie = True
            
            else:
                # Generate empty arrays so numba doesn't break
                n_aerosol_array = []
                sigma_Mie_array = []
                    
                n_aerosol_array.append(np.zeros_like(r))
                sigma_Mie_array.append(np.zeros_like(wl))

                n_aerosol_array = np.array(n_aerosol_array)
                sigma_Mie_array = np.array(sigma_Mie_array)

                enable_Mie = False
            
            # Calculate extinction coefficients in standard mode
            kappa_clear, kappa_cloud = extinction(chemical_species, active_species,
                                                CIA_pairs, ff_pairs, bf_species,
                                                n, T, P, wl, X, X_active, X_CIA, 
                                                X_ff, X_bf, a, gamma, P_cloud, 
                                                kappa_cloud_0, sigma_stored, 
                                                CIA_stored, Rayleigh_stored, 
                                                ff_stored, bf_stored, enable_haze, 
                                                enable_deck, enable_surface,
                                                N_sectors, N_zones, T_fine, 
                                                log_P_fine, P_surf,
                                                enable_Mie, n_aerosol_array, sigma_Mie_array)
            

        # Running POSEIDON on the GPU
        elif (device == 'gpu'):

            N_wl = len(wl)     # Number of wavelengths on model grid
            N_layers = len(P)  # Number of layers

            # Define extinction coefficient arrays explicitly on GPU
            kappa_clear = cp.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
            kappa_cloud = cp.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))

            # Find index of deep pressure below which atmosphere is opaque
            P_deep = 1000       # Default value of P_deep (needs to be high for brown dwarfs)
            i_bot = np.argmin(np.abs(P - P_deep))

            # Store length variables for mixing ratio arrays 
            N_species = len(chemical_species)        # Number of chemical species
            N_species_active = len(active_species)   # Number of spectrally active species
            
            N_cia_pairs = len(CIA_pairs)             # Number of cia pairs included
            N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
            N_bf_species = len(bf_species)           # Number of bound-free species included
        
            # Calculate extinction coefficients in standard mode
            extinction_GPU[block, thread](kappa_clear, kappa_cloud, i_bot, N_species, 
                                          N_species_active, N_cia_pairs, N_ff_pairs, 
                                          N_bf_species, n, T, P, wl, X, X_active, 
                                          X_CIA, X_ff, X_bf, a, gamma, P_cloud, 
                                          kappa_cloud_0, sigma_stored, 
                                          CIA_stored, Rayleigh_stored, 
                                          ff_stored, bf_stored, enable_haze, 
                                          enable_deck, enable_surface,
                                          N_sectors, N_zones, T_fine, 
                                          log_P_fine, P_surf, P_deep)

    # Generate transmission spectrum        
    if (spectrum_type == 'transmission'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

        # Call the core TRIDENT routine to compute the transmission spectrum
        spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                           enable_deck, enable_haze, b_p, y_p[0], R_s, f_cloud,
                           phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

    # Generate time-averaged transmission spectrum 
    elif (spectrum_type == 'transmission_time_average'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

        N_y = len(y_p)   # Number of time steps

        spectrum_stored = np.zeros(shape=(len(y_p),len(wl)))

        # We only need to calculate spectrum once for inbound vs. outbound
        for i in range(0, (N_y//2 + 1)):   

            # Call TRIDENT at the given time step
            spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                               enable_deck, enable_haze, b_p, y_p[i], R_s, f_cloud,
                               phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

            # At mid-transit, only have one spectrum to store
            if (i == N_y//2):
                spectrum_stored[i,:] = spectrum

            # At other time steps, store identical spectra for inbound and outbound planet
            else:
                spectrum_stored[i,:] = spectrum
                spectrum_stored[(N_y-1-i),:] = spectrum

        # Average all time steps (trapezium rule to increase accuracy)
        spectrum_avg = 0.5*(np.mean(spectrum_stored[1:-1], axis=0) +
                            np.mean(spectrum_stored, axis=0))

        spectrum = spectrum_avg  # Output spectrum is the time-averaged spectrum

    # Generate emission spectrum
    elif ('emission' in spectrum_type):

        # Find zone index for the emission spectrum atmospheric region
        if ('dayside' in spectrum_type):
            zone_idx = 0
        elif ('nightside' in spectrum_type):
            zone_idx = -1
        else:
            zone_idx = 0

        # Use atmospheric properties from dayside/nightside
        kappa = kappa_clear[:,0,zone_idx,:]  # Only consider one region for 1D models
        dz = dr[:,0,zone_idx]   # Only consider one region for 1D/2D models
        T = T[:,0,zone_idx]     # Only consider one region for 1D/2D models

        # Compute planet flux (on CPU or GPU)
        if (device == 'cpu'):
            F_p, dtau = emission_rad_transfer(T, dz, wl, kappa, Gauss_quad)
        elif (device == 'gpu'):
            F_p, dtau = emission_rad_transfer_GPU(T, dz, wl, kappa, Gauss_quad)

        # Calculate effective photosphere radius at tau = 2/3
        if (use_photosphere_radius == True):    # Flip to start at top of atmosphere
            
            # Running POSEIDON on the CPU
            if (device == 'cpu'):
                R_p_eff = determine_photosphere_radii(np.flip(dtau, axis=0), np.flip(r_low[:,0,zone_idx]), wl, photosphere_tau = 2/3)
            
            # Running POSEIDON on the GPU
            elif (device == 'gpu'):

                # Initialise photosphere radius array
                R_p_eff = cp.zeros(len(wl))
                dtau_flipped = cp.flip(dtau, axis=0)
                r_low_flipped = np.ascontiguousarray(np.flip(r_low[:,0,zone_idx]))

                # Find cumulative optical depth from top of atmosphere down at each wavelength
                tau_lambda = cp.cumsum(dtau_flipped, axis=0)

                # Calculate photosphere radius using GPU
                determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)

                # Convert back to numpy array on CPU
                R_p_eff = cp.asnumpy(R_p_eff)          
        
        else:
            R_p_eff = R_p    # If photosphere calculation disabled, use observed planet radius
        
        # If distance not specified, use fiducial value
        if (d is None):
            d = 1        # This value only used for flux ratios, so it cancels

        # For direct emission spectra (brown dwarfs and directly imaged planets)        
        if ('direct' in spectrum_type):

            # Convert planet surface flux to observed flux at Earth
            F_p_obs = (R_p_eff / d)**2 * F_p

            # Direct spectrum is F_p observed at Earth
            spectrum = F_p_obs

        # For transiting planet emission spectra
        else:

            # Load stellar spectrum
            F_s = star['F_star']
            wl_s = star['wl_star']

            if (np.array_equiv(wl_s, wl) is False):
                raise Exception("Error: wavelength grid for stellar spectrum does " +
                                "not match wavelength grid of planet spectrum. " +
                                "Did you forget to provide 'wl' to create_star?")

            # Interpolate stellar spectrum onto planet spectrum wavelength grid
        #    F_s_interp = spectres(wl, wl_s, F_s)

            # Convert stellar surface flux to observed flux at Earth
            F_s_obs = (R_s / d)**2 * F_s

            # Convert planet surface flux to observed flux at Earth
            F_p_obs = (R_p_eff / d)**2 * F_p

            # Final spectrum is the planet-star flux ratio
            spectrum = F_p_obs / F_s_obs
        
    # Write spectrum to file
    if (save_spectrum == True):
        write_spectrum(planet['planet_name'], model['model_name'], spectrum, wl)

    return spectrum


def compute_spectrum_c(planet, star, model, atmosphere, opac, wl,
                     spectrum_type = 'transmission', save_spectrum = False,
                     disable_continuum = False, suppress_print = False,
                     Gauss_quad = 2, use_photosphere_radius = True,
                     device = 'cpu', y_p = np.array([0.0]),
                     contribution_molecule_list = [],
                     bulk = False):
    '''
    Calculate extinction coefficients, then solve the radiative transfer 
    equation to compute the spectrum of the model atmosphere.

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        star (dict):
            Collection of stellar properties used by POSEIDON.
        model (dict):
            A specific description of a given POSEIDON model.
        atmosphere (dict):
            Collection of atmospheric properties.
        opac (dict):
            Collection of cross sections and other opacity sources.
        wl (np.array of float):
            Model wavelength grid (μm).
        spectrum_type (str):
            The type of spectrum for POSEIDON to compute
            (Options: transmission / emission / direct_emission / 
                      transmission_time_average).
        save_spectrum (bool):
            If True, writes the spectrum to './POSEIDON_output/PLANET/spectra/'.
        disable_continuum (bool):
            If True, turns off CIA and Rayleigh scattering opacities.
        suppress_print (bool):
            if True, turn off opacity print statements (in line-by-line mode).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            * Only for emission spectra *
            (Options: 2 / 3).
        use_photosphere_radius (bool):
            If True, use R_p at tau = 2/3 for emission spectra prefactor.
        device (str):
            Experimental: use CPU or GPU (only for emission spectra)
            (Options: cpu / gpu)
        y_p (np.array of float):
            Coordinate of planet centre along orbit at the time the spectrum
            is computed (y_p = 0, the default, corresponds to mid-transit).
            For non-grazing transits of uniform stellar disks, the spectrum
            is identical at all times due to translational symmetry, so y_p = 0
            is good for all times post second contact and pre third contact.
            Units are in m, not in stellar radii.
        contribution_molecule_list = [] (np.array of strings)
            Has a list of (active) molecules you want the spectral contribution function of 
        bulk (bool)
            If true, also returns the spectral contribution of the bulk species 

    Returns:
        spectrum (np.array of float):
            The spectrum of the atmosphere (transmission or emission).
        If contribution_molecule_list != []
            Returns spectrum (np.array of float), ['Molecule',contribution_spectrum]
            The second is a list where the first element is a string of the molecule being considered
            And the second element is the contribution_spectrum [np.array of float]
    
    '''

    # Check if the atmosphere is unphysical (e.g. temperature out of bounds)
    if (check_atmosphere_physical(atmosphere, opac) == False):
        spectrum = np.empty(len(wl))
        spectrum[:] = np.NaN
        return spectrum   # Unphysical => reject model

    # Unpack model properties
    PT_dim = model['PT_dim']
    X_dim = model['X_dim']
    cloud_dim = model['cloud_dim']

    # Check that the requested spectrum model is supported
    if (spectrum_type not in ['transmission', 'emission', 'direct_emission',
                              'dayside_emission', 'nightside_emission',
                              'transmission_time_average']):
        raise Exception("Only transmission spectra and emission " +
                        "spectra are currently supported.")
    elif (('emission' in spectrum_type) and 
         ((PT_dim or X_dim or cloud_dim) == 3)):
        raise Exception("Only 1D or 2D emission spectra currently supported.")

    # Unpack planet and star properties
    b_p = planet['planet_impact_parameter']
    R_p = planet['planet_radius']
    b_p = planet['planet_impact_parameter']
    d = planet['system_distance']

    if (star is not None):
        R_s = star['R_s']

    # Check that a distance is provided if user wants a direct spectrum
    if (d is None) and ('direct' in spectrum_type):
        raise Exception("Must provide a system distance when computing a " +
                        "direct emission spectrum.")

    # Unpack atmospheric properties needed for radiative transfer
    r = atmosphere['r']
    r_low = atmosphere['r_low']
    r_up = atmosphere['r_up']
    dr = atmosphere['dr']
    n = atmosphere['n']
    T = atmosphere['T']
    P = atmosphere['P']
    P_surf = atmosphere['P_surf']
    X = atmosphere['X']
    X_active = atmosphere['X_active']
    X_CIA = atmosphere['X_CIA']
    X_ff = atmosphere['X_ff']
    X_bf = atmosphere['X_bf']
    N_sectors = atmosphere['N_sectors']
    N_zones = atmosphere['N_zones']
    phi_edge = atmosphere['phi_edge']
    theta_edge = atmosphere['theta_edge']
    a = atmosphere['a']
    gamma = atmosphere['gamma']
    P_cloud = atmosphere['P_cloud']
    kappa_cloud_0 = atmosphere['kappa_cloud_0']
    f_cloud = atmosphere['f_cloud']
    phi_cloud_0 = atmosphere['phi_cloud_0']
    theta_cloud_0 = atmosphere['theta_cloud_0']

    # Check if haze enabled in the cloud model
    if ('haze' in model['cloud_type']):
        enable_haze = 1
    else:
        enable_haze = 0

    # Check if a cloud deck is enabled in the cloud model
    if ('deck' in model['cloud_type']):
        enable_deck = 1
    else:
        enable_deck = 0

    # Check if a surface is enabled
    if (P_surf != None):
        enable_surface = 1
    else:
        enable_surface = 0
        P_surf = 100.0      # Set surface pressure to 100 bar if not defined

    #***** Calculate extinction coefficients *****#

    # Unpack lists of chemical species in this model
    chemical_species = model['chemical_species'] 
    active_species = model['active_species']
    CIA_pairs = model['CIA_pairs']
    ff_pairs = model['ff_pairs']
    bf_species = model['bf_species']
            
    # If computing line-by-line radiative transfer, use lbl optimised functions 
    if (opac['opacity_treatment'] == 'line_by_line'):

        # Identify the opacity database being used
        opacity_database = opac['opacity_database']

        # Unpack pre-computed Rayleigh cross sections
        Rayleigh_stored = opac['Rayleigh_stored']

        # Calculate extinction coefficients in line-by-line mode        
        kappa_clear, kappa_cloud = extinction_LBL(chemical_species, active_species, 
                                                  CIA_pairs, ff_pairs, bf_species, 
                                                  n, T, P, wl, X, X_active, X_CIA, 
                                                  X_ff, X_bf, a, gamma, P_cloud,
                                                  kappa_cloud_0, Rayleigh_stored, 
                                                  enable_haze, enable_deck,
                                                  enable_surface, N_sectors, 
                                                  N_zones, P_surf, opacity_database, 
                                                  disable_continuum, suppress_print)
        
    # If using opacity sampling, we can use pre-interpolated cross sections
    elif (opac['opacity_treatment'] == 'opacity_sampling'):

        # Unpack pre-interpolated cross sections
        sigma_stored = opac['sigma_stored']
        CIA_stored = opac['CIA_stored']
        Rayleigh_stored = opac['Rayleigh_stored']
        ff_stored = opac['ff_stored']
        bf_stored = opac['bf_stored']

        # Also unpack fine temperature and pressure grids from pre-interpolation
        T_fine = opac['T_fine']
        log_P_fine = opac['log_P_fine']

        # Running POSEIDON on the CPU
        if (device == 'cpu'):

            # Calculate extinction coefficients in standard mode (to get normal spectrum)
            kappa_clear, kappa_cloud = extinction(chemical_species, active_species,
                                                CIA_pairs, ff_pairs, bf_species,
                                                n, T, P, wl, X, X_active, X_CIA, 
                                                X_ff, X_bf, a, gamma, P_cloud, 
                                                kappa_cloud_0, sigma_stored, 
                                                CIA_stored, Rayleigh_stored, 
                                                ff_stored, bf_stored, enable_haze, 
                                                enable_deck, enable_surface,
                                                N_sectors, N_zones, T_fine, 
                                                log_P_fine, P_surf)


            # This is to store the contribution kappas
            kappa_clear_contribution_array = []
            kappa_cloud_contribution_array = []
            
            # If you want to see only the bulk species contribution, this runs first 
            if bulk == True:

                kappa_clear_temp, kappa_cloud_temp = extinction_spectrum_contribution(chemical_species, active_species,
                                                                                    CIA_pairs, ff_pairs, bf_species,
                                                                                    n, T, P, wl, X, X_active, X_CIA, 
                                                                                    X_ff, X_bf, a, gamma, P_cloud, 
                                                                                    kappa_cloud_0, sigma_stored, 
                                                                                    CIA_stored, Rayleigh_stored, 
                                                                                    ff_stored, bf_stored, enable_haze, 
                                                                                    enable_deck, enable_surface,
                                                                                    N_sectors, N_zones, T_fine, 
                                                                                    log_P_fine, P_surf,
                                                                                    contribution_molecule_list = [contribution_molecule_list[0]],
                                                                                    bulk = True)

                kappa_clear_contribution_array.append(kappa_clear_temp)
                kappa_cloud_contribution_array.append(kappa_cloud_temp)

                        
            # Then it runs the rest of the molecules that are provided 
            if (contribution_molecule_list != []):

                for i in range(len(contribution_molecule_list)):

                    kappa_clear_temp, kappa_cloud_temp = extinction_spectrum_contribution(chemical_species, active_species,
                                                        CIA_pairs, ff_pairs, bf_species,
                                                        n, T, P, wl, X, X_active, X_CIA, 
                                                        X_ff, X_bf, a, gamma, P_cloud, 
                                                        kappa_cloud_0, sigma_stored, 
                                                        CIA_stored, Rayleigh_stored, 
                                                        ff_stored, bf_stored, enable_haze, 
                                                        enable_deck, enable_surface,
                                                        N_sectors, N_zones, T_fine, 
                                                        log_P_fine, P_surf,
                                                        contribution_molecule_list = [contribution_molecule_list[i]],
                                                        bulk = False)

                    kappa_clear_contribution_array.append(kappa_clear_temp)
                    kappa_cloud_contribution_array.append(kappa_cloud_temp)



        # Running POSEIDON on the GPU
        elif (device == 'gpu'):

            N_wl = len(wl)     # Number of wavelengths on model grid
            N_layers = len(P)  # Number of layers

            # Define extinction coefficient arrays explicitly on GPU
            kappa_clear = cp.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
            kappa_cloud = cp.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))

            # Find index of deep pressure below which atmosphere is opaque
            P_deep = 1000       # Default value of P_deep (needs to be high for brown dwarfs)
            i_bot = np.argmin(np.abs(P - P_deep))

            # Store length variables for mixing ratio arrays 
            N_species = len(chemical_species)        # Number of chemical species
            N_species_active = len(active_species)   # Number of spectrally active species
            
            N_cia_pairs = len(CIA_pairs)             # Number of cia pairs included
            N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
            N_bf_species = len(bf_species)           # Number of bound-free species included
        
            # Calculate extinction coefficients in standard mode
            extinction_GPU[block, thread](kappa_clear, kappa_cloud, i_bot, N_species, 
                                          N_species_active, N_cia_pairs, N_ff_pairs, 
                                          N_bf_species, n, T, P, wl, X, X_active, 
                                          X_CIA, X_ff, X_bf, a, gamma, P_cloud, 
                                          kappa_cloud_0, sigma_stored, 
                                          CIA_stored, Rayleigh_stored, 
                                          ff_stored, bf_stored, enable_haze, 
                                          enable_deck, enable_surface,
                                          N_sectors, N_zones, T_fine, 
                                          log_P_fine, P_surf, P_deep)

    # Generate transmission spectrum        
    if (spectrum_type == 'transmission'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

       
       # Calculate normal spectrum 
        spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                           enable_deck, enable_haze, b_p, y_p[0], R_s, f_cloud,
                           phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)
       
        # If you have both contribution molecules and bulk molecules
        if contribution_molecule_list != [] and bulk == True:

            spectrum_contribution_list = []

            # First add the bulk spectrum 
            kappa_clear_temp = kappa_clear_contribution_array[0]
            kappa_cloud_temp = kappa_cloud_contribution_array[0]

            spectrum_temp = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear_temp, kappa_cloud_temp,
                                            enable_deck, enable_haze, b_p, y_p[0], R_s, f_cloud,
                                            phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

            spectrum_contribution_list.append(['Bulk Species',spectrum_temp])

            # Then add the rest 
            for i in range(len(contribution_molecule_list)):

                    kappa_clear_temp = kappa_clear_contribution_array[i+1]
                    kappa_cloud_temp = kappa_cloud_contribution_array[i+1]

                    spectrum_temp = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear_temp, kappa_cloud_temp,
                                            enable_deck, enable_haze, b_p, y_p[0], R_s, f_cloud,
                                            phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

                    spectrum_contribution_list.append([contribution_molecule_list[i],spectrum_temp])

        # If you just have contribution molecules 
        elif contribution_molecule_list != [] and bulk == False:

                spectrum_contribution_list = []

                for i in range(len(contribution_molecule_list)):

                    kappa_clear_temp = kappa_clear_contribution_array[i]
                    kappa_cloud_temp = kappa_cloud_contribution_array[i]

                    spectrum_temp = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear_temp, kappa_cloud_temp,
                                            enable_deck, enable_haze, b_p, y_p[0], R_s, f_cloud,
                                            phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

                    spectrum_contribution_list.append([contribution_molecule_list[i],spectrum_temp])

        # If you 'just' have bulk 
        elif contribution_molecule_list == [] and bulk == True:
            spectrum_contribution_list = []

            # First add the bulk spectrum 
            kappa_clear_temp = kappa_clear_contribution_array[0]
            kappa_cloud_temp = kappa_cloud_contribution_array[0]

            spectrum_temp = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear_temp, kappa_cloud_temp,
                                            enable_deck, enable_haze, b_p, y_p[0], R_s, f_cloud,
                                            phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

            spectrum_contribution_list.append(['Bulk Species',spectrum_temp])

    # Generate time-averaged transmission spectrum 
    elif (spectrum_type == 'transmission_time_average'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

        N_y = len(y_p)   # Number of time steps

        spectrum_stored = np.zeros(shape=(len(y_p),len(wl)))

        # We only need to calculate spectrum once for inbound vs. outbound
        for i in range(0, (N_y//2 + 1)):   

            # Call TRIDENT at the given time step
            spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                               enable_deck, enable_haze, b_p, y_p[i], R_s, f_cloud,
                               phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

            # At mid-transit, only have one spectrum to store
            if (i == N_y//2):
                spectrum_stored[i,:] = spectrum

            # At other time steps, store identical spectra for inbound and outbound planet
            else:
                spectrum_stored[i,:] = spectrum
                spectrum_stored[(N_y-1-i),:] = spectrum

        # Average all time steps (trapezium rule to increase accuracy)
        spectrum_avg = 0.5*(np.mean(spectrum_stored[1:-1], axis=0) +
                            np.mean(spectrum_stored, axis=0))

        spectrum = spectrum_avg  # Output spectrum is the time-averaged spectrum

    # Generate emission spectrum
    elif ('emission' in spectrum_type):

        # Find zone index for the emission spectrum atmospheric region
        if ('dayside' in spectrum_type):
            zone_idx = 0
        elif ('nightside' in spectrum_type):
            zone_idx = -1
        else:
            zone_idx = 0

        # Use atmospheric properties from dayside/nightside
        kappa = kappa_clear[:,0,zone_idx,:]  # Only consider one region for 1D models
        dz = dr[:,0,zone_idx]   # Only consider one region for 1D/2D models
        T = T[:,0,zone_idx]     # Only consider one region for 1D/2D models

        # Compute planet flux (on CPU or GPU)
        if (device == 'cpu'):
            F_p, dtau = emission_rad_transfer(T, dz, wl, kappa, Gauss_quad)
        elif (device == 'gpu'):
            F_p, dtau = emission_rad_transfer_GPU(T, dz, wl, kappa, Gauss_quad)

        # Calculate effective photosphere radius at tau = 2/3
        if (use_photosphere_radius == True):    # Flip to start at top of atmosphere
            
            # Running POSEIDON on the CPU
            if (device == 'cpu'):
                R_p_eff = determine_photosphere_radii(np.flip(dtau, axis=0), np.flip(r_low[:,0,zone_idx]), wl, photosphere_tau = 2/3)
            
            # Running POSEIDON on the GPU
            elif (device == 'gpu'):

                # Initialise photosphere radius array
                R_p_eff = cp.zeros(len(wl))
                dtau_flipped = cp.flip(dtau, axis=0)
                r_low_flipped = np.ascontiguousarray(np.flip(r_low[:,0,zone_idx]))

                # Find cumulative optical depth from top of atmosphere down at each wavelength
                tau_lambda = cp.cumsum(dtau_flipped, axis=0)

                # Calculate photosphere radius using GPU
                determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)

                # Convert back to numpy array on CPU
                R_p_eff = cp.asnumpy(R_p_eff)          
        
        else:
            R_p_eff = R_p    # If photosphere calculation disabled, use observed planet radius
        
        # If distance not specified, use fiducial value
        if (d is None):
            d = 1        # This value only used for flux ratios, so it cancels

        # For direct emission spectra (brown dwarfs and directly imaged planets)        
        if ('direct' in spectrum_type):

            # Convert planet surface flux to observed flux at Earth
            F_p_obs = (R_p_eff / d)**2 * F_p

            # Direct spectrum is F_p observed at Earth
            spectrum = F_p_obs

        # For transiting planet emission spectra
        else:

            # Load stellar spectrum
            F_s = star['F_star']
            wl_s = star['wl_star']

            # Interpolate stellar spectrum onto planet spectrum wavelength grid
            F_s_interp = spectres(wl, wl_s, F_s)

            # Convert stellar surface flux to observed flux at Earth
            F_s_obs = (R_s / d)**2 * F_s_interp

            # Convert planet surface flux to observed flux at Earth
            F_p_obs = (R_p_eff / d)**2 * F_p

            # Final spectrum is the planet-star flux ratio
            spectrum = F_p_obs / F_s_obs
        
    # Write spectrum to file
    if (save_spectrum == True):
        write_spectrum(planet['planet_name'], model['model_name'], spectrum, wl)

    if (contribution_molecule_list != [] or bulk==True):
        return spectrum, spectrum_contribution_list

    else:
        return spectrum

def compute_spectrum_p(planet, star, model, atmosphere, opac, wl,
                     spectrum_type = 'transmission', save_spectrum = False,
                     disable_continuum = False, suppress_print = False,
                     Gauss_quad = 2, use_photosphere_radius = True,
                     device = 'cpu', y_p = np.array([0.0]),
                     contribution_molecule = '',
                     layer_to_ignore = 0, 
                     total = False):
    '''
    Calculate extinction coefficients, then solve the radiative transfer 
    equation to compute the spectrum of the model atmosphere.

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        star (dict):
            Collection of stellar properties used by POSEIDON.
        model (dict):
            A specific description of a given POSEIDON model.
        atmosphere (dict):
            Collection of atmospheric properties.
        opac (dict):
            Collection of cross sections and other opacity sources.
        wl (np.array of float):
            Model wavelength grid (μm).
        spectrum_type (str):
            The type of spectrum for POSEIDON to compute
            (Options: transmission / emission / direct_emission / 
                      transmission_time_average).
        save_spectrum (bool):
            If True, writes the spectrum to './POSEIDON_output/PLANET/spectra/'.
        disable_continuum (bool):
            If True, turns off CIA and Rayleigh scattering opacities.
        suppress_print (bool):
            if True, turn off opacity print statements (in line-by-line mode).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            * Only for emission spectra *
            (Options: 2 / 3).
        use_photosphere_radius (bool):
            If True, use R_p at tau = 2/3 for emission spectra prefactor.
        device (str):
            Experimental: use CPU or GPU (only for emission spectra)
            (Options: cpu / gpu)
        y_p (np.array of float):
            Coordinate of planet centre along orbit at the time the spectrum
            is computed (y_p = 0, the default, corresponds to mid-transit).
            For non-grazing transits of uniform stellar disks, the spectrum
            is identical at all times due to translational symmetry, so y_p = 0
            is good for all times post second contact and pre third contact.
            Units are in m, not in stellar radii.
        contribution_molecule (string)
            The molecule that will be turned off in the layer to ignore
        layer_to_ignore (int)
            This is looped over in the contribution_func.py
            Indicates the pressure layer in which the molecule will be turned off 
        total (bool)
            If true, the total pressure contribution spectrum is returned

    Returns:
        spectrum (np.array of float):
            The spectrum of the atmosphere (transmission or emission).
        If contribution_molecule_list != [] or total = True
            Returns a spectrum where the molecule indicated is turned off for the pressure layer to ignore 
            If total = true, then a whole layer is turned off 
    
    
    '''

    # Check if the atmosphere is unphysical (e.g. temperature out of bounds)
    if (check_atmosphere_physical(atmosphere, opac) == False):
        spectrum = np.empty(len(wl))
        spectrum[:] = np.NaN
        return spectrum   # Unphysical => reject model

    # Unpack model properties
    PT_dim = model['PT_dim']
    X_dim = model['X_dim']
    cloud_dim = model['cloud_dim']

    # Check that the requested spectrum model is supported
    if (spectrum_type not in ['transmission', 'emission', 'direct_emission',
                              'dayside_emission', 'nightside_emission',
                              'transmission_time_average']):
        raise Exception("Only transmission spectra and emission " +
                        "spectra are currently supported.")
    elif (('emission' in spectrum_type) and 
         ((PT_dim or X_dim or cloud_dim) == 3)):
        raise Exception("Only 1D or 2D emission spectra currently supported.")

    # Unpack planet and star properties
    b_p = planet['planet_impact_parameter']
    R_p = planet['planet_radius']
    b_p = planet['planet_impact_parameter']
    d = planet['system_distance']

    if (star is not None):
        R_s = star['R_s']

    # Check that a distance is provided if user wants a direct spectrum
    if (d is None) and ('direct' in spectrum_type):
        raise Exception("Must provide a system distance when computing a " +
                        "direct emission spectrum.")

    # Unpack atmospheric properties needed for radiative transfer
    r = atmosphere['r']
    r_low = atmosphere['r_low']
    r_up = atmosphere['r_up']
    dr = atmosphere['dr']
    n = atmosphere['n']
    T = atmosphere['T']
    P = atmosphere['P']
    P_surf = atmosphere['P_surf']
    X = atmosphere['X']
    X_active = atmosphere['X_active']
    X_CIA = atmosphere['X_CIA']
    X_ff = atmosphere['X_ff']
    X_bf = atmosphere['X_bf']
    N_sectors = atmosphere['N_sectors']
    N_zones = atmosphere['N_zones']
    phi_edge = atmosphere['phi_edge']
    theta_edge = atmosphere['theta_edge']
    a = atmosphere['a']
    gamma = atmosphere['gamma']
    P_cloud = atmosphere['P_cloud']
    kappa_cloud_0 = atmosphere['kappa_cloud_0']
    f_cloud = atmosphere['f_cloud']
    phi_cloud_0 = atmosphere['phi_cloud_0']
    theta_cloud_0 = atmosphere['theta_cloud_0']

    # Check if haze enabled in the cloud model
    if ('haze' in model['cloud_type']):
        enable_haze = 1
    else:
        enable_haze = 0

    # Check if a cloud deck is enabled in the cloud model
    if ('deck' in model['cloud_type']):
        enable_deck = 1
    else:
        enable_deck = 0

    # Check if a surface is enabled
    if (P_surf != None):
        enable_surface = 1
    else:
        enable_surface = 0
        P_surf = 100.0      # Set surface pressure to 100 bar if not defined

    #***** Calculate extinction coefficients *****#

    # Unpack lists of chemical species in this model
    chemical_species = model['chemical_species'] 
    active_species = model['active_species']
    CIA_pairs = model['CIA_pairs']
    ff_pairs = model['ff_pairs']
    bf_species = model['bf_species']
            
    # If computing line-by-line radiative transfer, use lbl optimised functions 
    if (opac['opacity_treatment'] == 'line_by_line'):

        # Identify the opacity database being used
        opacity_database = opac['opacity_database']

        # Unpack pre-computed Rayleigh cross sections
        Rayleigh_stored = opac['Rayleigh_stored']

        # Calculate extinction coefficients in line-by-line mode        
        kappa_clear, kappa_cloud = extinction_LBL(chemical_species, active_species, 
                                                  CIA_pairs, ff_pairs, bf_species, 
                                                  n, T, P, wl, X, X_active, X_CIA, 
                                                  X_ff, X_bf, a, gamma, P_cloud,
                                                  kappa_cloud_0, Rayleigh_stored, 
                                                  enable_haze, enable_deck,
                                                  enable_surface, N_sectors, 
                                                  N_zones, P_surf, opacity_database, 
                                                  disable_continuum, suppress_print)
        
    # If using opacity sampling, we can use pre-interpolated cross sections
    elif (opac['opacity_treatment'] == 'opacity_sampling'):

        # Unpack pre-interpolated cross sections
        sigma_stored = opac['sigma_stored']
        CIA_stored = opac['CIA_stored']
        Rayleigh_stored = opac['Rayleigh_stored']
        ff_stored = opac['ff_stored']
        bf_stored = opac['bf_stored']

        # Also unpack fine temperature and pressure grids from pre-interpolation
        T_fine = opac['T_fine']
        log_P_fine = opac['log_P_fine']

        # Running POSEIDON on the CPU
        if (device == 'cpu'):

            if (contribution_molecule != '' and total != True):

                kappa_clear, kappa_cloud = extinction_spectrum_pressure_contribution(chemical_species, active_species,
                                                                                    CIA_pairs, ff_pairs, bf_species,
                                                                                    n, T, P, wl, X, X_active, X_CIA, 
                                                                                    X_ff, X_bf, a, gamma, P_cloud, 
                                                                                    kappa_cloud_0, sigma_stored, 
                                                                                    CIA_stored, Rayleigh_stored, 
                                                                                    ff_stored, bf_stored, enable_haze, 
                                                                                    enable_deck, enable_surface,
                                                                                    N_sectors, N_zones, T_fine, 
                                                                                    log_P_fine, P_surf,
                                                                                    contribution_molecule = contribution_molecule,
                                                                                    layer_to_ignore = layer_to_ignore)
            
            elif(contribution_molecule == '' and total == True):

                kappa_clear, kappa_cloud = extinction_spectrum_pressure_contribution(chemical_species, active_species,
                                                                                    CIA_pairs, ff_pairs, bf_species,
                                                                                    n, T, P, wl, X, X_active, X_CIA, 
                                                                                    X_ff, X_bf, a, gamma, P_cloud, 
                                                                                    kappa_cloud_0, sigma_stored, 
                                                                                    CIA_stored, Rayleigh_stored, 
                                                                                    ff_stored, bf_stored, enable_haze, 
                                                                                    enable_deck, enable_surface,
                                                                                    N_sectors, N_zones, T_fine, 
                                                                                    log_P_fine, P_surf,
                                                                                    layer_to_ignore = layer_to_ignore, 
                                                                                    total = True)

            elif contribution_molecule == '' and total == False:
                raise Exception('If total = False, must provide a molecule (str)')
            elif contribution_molecule != '' and total == True:
                raise Exception('If total = True, cannot provide a molecule (str)')




        # Running POSEIDON on the GPU
        elif (device == 'gpu'):

            N_wl = len(wl)     # Number of wavelengths on model grid
            N_layers = len(P)  # Number of layers

            # Define extinction coefficient arrays explicitly on GPU
            kappa_clear = cp.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
            kappa_cloud = cp.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))

            # Find index of deep pressure below which atmosphere is opaque
            P_deep = 1000       # Default value of P_deep (needs to be high for brown dwarfs)
            i_bot = np.argmin(np.abs(P - P_deep))

            # Store length variables for mixing ratio arrays 
            N_species = len(chemical_species)        # Number of chemical species
            N_species_active = len(active_species)   # Number of spectrally active species
            
            N_cia_pairs = len(CIA_pairs)             # Number of cia pairs included
            N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
            N_bf_species = len(bf_species)           # Number of bound-free species included
        
            # Calculate extinction coefficients in standard mode
            extinction_GPU[block, thread](kappa_clear, kappa_cloud, i_bot, N_species, 
                                          N_species_active, N_cia_pairs, N_ff_pairs, 
                                          N_bf_species, n, T, P, wl, X, X_active, 
                                          X_CIA, X_ff, X_bf, a, gamma, P_cloud, 
                                          kappa_cloud_0, sigma_stored, 
                                          CIA_stored, Rayleigh_stored, 
                                          ff_stored, bf_stored, enable_haze, 
                                          enable_deck, enable_surface,
                                          N_sectors, N_zones, T_fine, 
                                          log_P_fine, P_surf, P_deep)

    # Generate transmission spectrum        
    if (spectrum_type == 'transmission'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

        # Call the core TRIDENT routine to compute the transmission spectrum
        spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                           enable_deck, enable_haze, b_p, y_p[0], R_s, f_cloud,
                           phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

    # Generate time-averaged transmission spectrum 
    elif (spectrum_type == 'transmission_time_average'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

        N_y = len(y_p)   # Number of time steps

        spectrum_stored = np.zeros(shape=(len(y_p),len(wl)))

        # We only need to calculate spectrum once for inbound vs. outbound
        for i in range(0, (N_y//2 + 1)):   

            # Call TRIDENT at the given time step
            spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                               enable_deck, enable_haze, b_p, y_p[i], R_s, f_cloud,
                               phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

            # At mid-transit, only have one spectrum to store
            if (i == N_y//2):
                spectrum_stored[i,:] = spectrum

            # At other time steps, store identical spectra for inbound and outbound planet
            else:
                spectrum_stored[i,:] = spectrum
                spectrum_stored[(N_y-1-i),:] = spectrum

        # Average all time steps (trapezium rule to increase accuracy)
        spectrum_avg = 0.5*(np.mean(spectrum_stored[1:-1], axis=0) +
                            np.mean(spectrum_stored, axis=0))

        spectrum = spectrum_avg  # Output spectrum is the time-averaged spectrum

    # Generate emission spectrum
    elif ('emission' in spectrum_type):

        # Find zone index for the emission spectrum atmospheric region
        if ('dayside' in spectrum_type):
            zone_idx = 0
        elif ('nightside' in spectrum_type):
            zone_idx = -1
        else:
            zone_idx = 0

        # Use atmospheric properties from dayside/nightside
        kappa = kappa_clear[:,0,zone_idx,:]  # Only consider one region for 1D models
        dz = dr[:,0,zone_idx]   # Only consider one region for 1D/2D models
        T = T[:,0,zone_idx]     # Only consider one region for 1D/2D models

        # Compute planet flux (on CPU or GPU)
        if (device == 'cpu'):
            F_p, dtau = emission_rad_transfer(T, dz, wl, kappa, Gauss_quad)
        elif (device == 'gpu'):
            F_p, dtau = emission_rad_transfer_GPU(T, dz, wl, kappa, Gauss_quad)

        # Calculate effective photosphere radius at tau = 2/3
        if (use_photosphere_radius == True):    # Flip to start at top of atmosphere
            
            # Running POSEIDON on the CPU
            if (device == 'cpu'):
                R_p_eff = determine_photosphere_radii(np.flip(dtau, axis=0), np.flip(r_low[:,0,zone_idx]), wl, photosphere_tau = 2/3)
            
            # Running POSEIDON on the GPU
            elif (device == 'gpu'):

                # Initialise photosphere radius array
                R_p_eff = cp.zeros(len(wl))
                dtau_flipped = cp.flip(dtau, axis=0)
                r_low_flipped = np.ascontiguousarray(np.flip(r_low[:,0,zone_idx]))

                # Find cumulative optical depth from top of atmosphere down at each wavelength
                tau_lambda = cp.cumsum(dtau_flipped, axis=0)

                # Calculate photosphere radius using GPU
                determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)

                # Convert back to numpy array on CPU
                R_p_eff = cp.asnumpy(R_p_eff)          
        
        else:
            R_p_eff = R_p    # If photosphere calculation disabled, use observed planet radius
        
        # If distance not specified, use fiducial value
        if (d is None):
            d = 1        # This value only used for flux ratios, so it cancels

        # For direct emission spectra (brown dwarfs and directly imaged planets)        
        if ('direct' in spectrum_type):

            # Convert planet surface flux to observed flux at Earth
            F_p_obs = (R_p_eff / d)**2 * F_p

            # Direct spectrum is F_p observed at Earth
            spectrum = F_p_obs

        # For transiting planet emission spectra
        else:

            # Load stellar spectrum
            F_s = star['F_star']
            wl_s = star['wl_star']

            # Interpolate stellar spectrum onto planet spectrum wavelength grid
            F_s_interp = spectres(wl, wl_s, F_s)

            # Convert stellar surface flux to observed flux at Earth
            F_s_obs = (R_s / d)**2 * F_s_interp

            # Convert planet surface flux to observed flux at Earth
            F_p_obs = (R_p_eff / d)**2 * F_p

            # Final spectrum is the planet-star flux ratio
            spectrum = F_p_obs / F_s_obs
        
    # Write spectrum to file
    if (save_spectrum == True):
        write_spectrum(planet['planet_name'], model['model_name'], spectrum, wl)

    return spectrum

def load_data(data_dir, datasets, instruments, wl_model, offset_datasets = None,
              wl_unit = 'micron', bin_width = 'half', spectrum_unit = '(Rp/Rs)^2', 
              skiprows = None):
    '''
    Load the user provided datasets into the format expected by POSEIDON. 
    Also generate the functions required for POSEIDON to later calculate 
    the binned data for each instrument (e.g. the PSFs for each instrument)
    corresponding to model spectra.

    Args:
        data_dir (str):
            Path to the directory containing the user's data files.
        datasets (list of str):
            List containing file names of the user's data files.
        instruments (list of str):
            List containing the instrument names corresponding to each data file
            (e.g. WFC3_G141, JWST_NIRSpec_PRISM, JWST_NIRISS_SOSS_Ord2).
        wl_model (np.array of float):
            Model wavelength grid (μm).
        offset_datasets (list of str):
            If applying a relative offset to one or more datasets, this list
            gives the file names of the datasets that will have free offsets
            applied (note: currently only supports *one* offset dataset).
        wl_unit (str):
            Unit of wavelength column (first column in file)
            (Options: micron (or equivalent) / nm / A / m)
        bin_width (str):
            Whether bin width (second column) is half or full width
            (Options: half / full).
        spectrum_unit (str):
            Unit of spectrum (third column) and spectrum errors (fourth column)
            (Options: (Rp/Rs)^2 / Rp/Rs / Fp/Fs / Fp (or equivalent units)).
        skiprows (int):
            The number of rows to skip (e.g. use 1 if file has a header line).

    Returns:
        data (dict):
            Collection of data properties required for POSEIDON's instrument
            simulator (i.e. to create simulated binned data during retrievals).
    
    '''

    # If the user is running the retrieval tutorial, point to the reference data
    if (data_dir == 'Tutorial/WASP-999b'):
        data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
                                   '.', 'reference_data/observations/WASP-999b/'))

    # Convert lists to numpy arrays
    instruments = np.array(instruments)
    datasets = np.array(datasets)
    
    # Initialise arrays containing input properties of the data
    wl_data, half_bin, ydata, err_data, len_data = (np.array([]) for _ in range(5))
    
    # Initialise arrays containing instrument function properties
    psf_sigma, fwhm, sens, norm = (np.array([]) for _ in range(4))
    bin_left, bin_cent, bin_right, norm = (np.array([]).astype(np.int64) for _ in range(4))
    
    # For each dataset
    for i in range(len(datasets)):
        
        # Read data files
        wl_data_i, half_bin_i, \
        ydata_i, err_data_i = read_data(data_dir, datasets[i], wl_unit,
                                        bin_width, spectrum_unit, skiprows)
        
        # Combine datasets
        wl_data = np.concatenate([wl_data, wl_data_i])
        half_bin = np.concatenate([half_bin, half_bin_i])  
        ydata = np.concatenate([ydata, ydata_i])
        err_data = np.concatenate([err_data, err_data_i])
        
        # Length of each dataset (used for indexing the combined dataset, if necessary to extract one specific dataset later)
        len_data = np.concatenate([len_data, np.array([len(ydata_i)])])
        
        # Read instrument transmission functions, compute PSF std dev, and identify locations of each data bin on model grid
        psf_sigma_i, fwhm_i, sens_i, bin_left_i, \
        bin_cent_i, bin_right_i, norm_i = init_instrument(wl_model, wl_data_i, half_bin_i, instruments[i])
        
        # Combine instrument properties into single arrays for convenience (can index by len_data[i] to extract each later)
        psf_sigma = np.concatenate([psf_sigma, psf_sigma_i])  # Length for each dataset: len_data[i]
        fwhm = np.concatenate([fwhm, fwhm_i])                 # Length for each dataset: len_data[i]
        sens = np.concatenate([sens, sens_i])                 # Length for each dataset: N_wl
        bin_left = np.concatenate([bin_left, bin_left_i])     # Length for each dataset: len_data[i]
        bin_cent = np.concatenate([bin_cent, bin_cent_i])     # Length for each dataset: len_data[i]
        bin_right = np.concatenate([bin_right, bin_right_i])  # Length for each dataset: len_data[i]
        norm = np.concatenate([norm, norm_i])                 # Length for each dataset: len_data[i]
        
    # Cumulative sum of data lengths for indexing later
    len_data_idx = np.append(np.array([0]), np.cumsum(len_data)).astype(np.int64)       

    # For relative offsets, find which data indices the offset applies to
    if (offset_datasets is not None):
        offset_datasets = np.array(offset_datasets)
        if (len(offset_datasets) > 1):
            raise Exception("Error: only one dataset can have a free offset.")
        if (offset_datasets[0] in datasets):
            offset_dataset_idx = np.where(datasets == offset_datasets[0])[0][0]
            offset_data_start = len_data_idx[offset_dataset_idx]  # Data index of first point with offset
            offset_data_end = len_data_idx[offset_dataset_idx+1]  # Data index of last point with offset + 1
        else: 
            raise Exception("Dataset chosen for relative offset is not included.")
    else:
        offset_data_start = 0    # Dummy values when no offsets included
        offset_data_end = 0
        
    # Check that the model wavelength grid covers all the data bins
    if (np.any((wl_data - half_bin) < wl_model[0])):
        raise Exception("Some data lies below the lowest model wavelength, reduce wl_min.")
    elif (np.any((wl_data + half_bin) > wl_model[-1])):
        raise Exception("Some data lies above the highest model wavelength, increase wl_max.")
    
    # Package data properties
    data = {'datasets': datasets, 'instruments': instruments, 'wl_data': wl_data,
            'half_bin': half_bin, 'ydata': ydata, 'err_data': err_data, 
            'sens': sens, 'len_data_idx': len_data_idx, 'psf_sigma': psf_sigma,
            'norm': norm, 'bin_left': bin_left, 'bin_cent': bin_cent, 
            'bin_right': bin_right, 'offset_start': offset_data_start,
            'offset_end': offset_data_end, 'fwhm': fwhm
           }

    return data


def set_priors(planet, star, model, data, prior_types = {}, prior_ranges = {}):
    '''
    Initialise the priors for each free parameter for a POSEIDON retrieval.
    
    If the user does not provide a prior type or prior range for one or more
    of the parameters, this function will prescribe a default prior with
    a wide range. Thus the user can choose the degree to which they would
    like to 'micromanage' the assignment of priors.
    
    Disclaimer: while using default priors can be good for exploratory 
    retrievals, for a publication we *strongly* suggest you explicitly specify 
    your priors - you'll need to give your priors in a Table somewhere anyway, 
    so it's generally a good idea to know what they are ;)

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        star (dict):
            Collection of stellar properties used by POSEIDON.
        model (dict):
            A specific description of a given POSEIDON model.
        data (dict):
            Collection of data properties in POSEIDON format.
        prior_types (dict):
            User-provided dictionary containing the prior type for each 
            free parameter in the retrieval moel
            (Options: uniform, gaussian, sine, CLR).
        prior_ranges (dict):
            User-provided dictionary containing numbers defining the prior range
            for each free parameter in the retrieval model
            (Options: for 'uniform' [low, high], for 'gaussian' [mean, std],
                      for 'sine' [high] - only for 2D/3D angle parameters,
                      for 'CLR' [low] - only for mixing ratios).

    Returns:
        priors (dict):
            Collection of the prior types and ranges used by POSEIDON's
            retrieval module.
    
    '''

    # Unpack parameter names
    param_names = model['param_names']
    X_param_names = model['X_param_names']
    PT_profile = model['PT_profile']
    radius_unit = model['radius_unit']
    distance_unit = model['distance_unit']
    Atmosphere_dimension = model['Atmosphere_dimension']
    
    # Unpack planet and star properties
    R_p = planet['planet_radius']
    T_eq = planet['planet_T_eq']

    if (star != None):
        T_phot = star['T_eff']
        err_T_phot = star['T_eff_error']
        log_g_phot = star['log_g']
        err_log_g_phot = star['log_g_error']
    else:
        T_phot = None
        err_T_phot = None
        log_g_phot = None
        err_log_g_phot = None

    # Unpack data error bars (not error inflation parameter prior)
    err_data = data['err_data']    

    # Normalise retrieved planet radius parameter into Jupiter or Earth radii
    if (radius_unit == 'R_J'):
        R_p_norm = R_J
    elif (radius_unit == 'R_E'):
        R_p_norm = R_E
    if ('R_p_ref' in prior_ranges):
        prior_ranges['R_p_ref'] = [prior_ranges['R_p_ref'][0]/R_p_norm,
                                   prior_ranges['R_p_ref'][1]/R_p_norm]

    # Normalise retrieved system distance parameter into parsecs
    if (distance_unit == 'pc'):
        d_norm = parsec
    if ('d' in prior_ranges):
        prior_ranges['d'] = [prior_ranges['d'][0]/d_norm,
                             prior_ranges['d'][1]/d_norm]

    # Set default priors (used if user doesn't specify one or more priors)
    prior_ranges_defaults = {'R_p_ref': [0.85*R_p/R_p_norm, 1.15*R_p/R_p_norm],
                             'log_g': [2.0, 5.0], 'T': [400, 3000], 
                             'Delta_T': [0, 1000], 'Grad_T': [-200, 0],
                             'T_mid': [400, 3000], 'T_high': [400, 3000], 
                             'a1': [0.02, 2.00], 'a2': [0.02, 2.00], 
                             'log_P1': [-6, 2], 'log_P2': [-6, 2], 
                             'log_P3': [-2, 2], 'log_P_mid': [-5, 1], 
                             'log_P_surf': [-4, 1], 'log_P_ref': [-6, 2],
                             'log_X': [-12, -1], 
                             'Delta_log_X': [-10, 10], 'Grad_log_X': [-1, 1], 
                             'log_a': [-4, 8], 'gamma': [-20, 2], 
                             'log_P_cloud': [-6, 2], 'phi_cloud': [0, 1],
                             'log_kappa_cloud': [-10, -4], 'f_cloud': [0, 1],
                             'phi_0': [-180, 180], 'theta_0': [-35, 35],
                             'alpha': [0.1, 180], 'beta': [0.1, 70],
                             'f_het': [0.0, 0.5], 'T_het': [0.6*T_phot, 1.2*T_phot],
                             'f_spot': [0.0, 0.5], 'T_spot': [0.6*T_phot, T_phot],
                             'f_fac': [0.0, 0.5], 'T_fac': [T_phot, 1.2*T_phot],
                             'log_g_het': [log_g_phot-0.5, log_g_phot+0.5],
                             'log_g_spot': [log_g_phot-0.5, log_g_phot+0.5],
                             'log_g_fac': [log_g_phot-0.5, log_g_phot+0.5],
                             'T_phot': [T_phot, err_T_phot], 
                             'log_g_phot': [log_g_phot, err_log_g_phot], 
                             'delta_rel': [-1.0e-3, 1.0e-3],
                             'log_b': [np.log10(0.001*np.min(err_data**2)),
                                       np.log10(100.0*np.max(err_data**2))],
                             'C_to_O': [0.3,1.9],
                             'log_Met' : [-0.9,3.9],
                             'log_r_m': [-3,1],       
                             'log_n_max': [5.0,20.0],  
                             'fractional_scale_height': [0.1,1], 
                             'r_i_real': [0,10],
                             'r_i_complex': [1e-6,100], 
                             'log_X_Mie' : [-30,-1]
                            }   

    # Iterate through parameters, ensuring we have a full set of priors
    for parameter in param_names:

        # Check for parameters without a user-specified prior range
        if (parameter not in prior_ranges):
            
            # Special case for mixing ratio parameters
            if (parameter in X_param_names):

                # Set non-specified pressure of mid mixing ratio prior to that for 'log_P_mid'
                if ('log_P_' in parameter):
                    if ('log_P_X_mid' in prior_ranges):
                        prior_ranges[parameter] = prior_ranges['log_P_X_mid']
                    else:
                        prior_ranges[parameter] = prior_ranges_defaults['log_P_mid']

                # Set non-specified mixing ratio difference prior to that for 'Delta_log_X'
                elif ('Delta_log_' in parameter):
                    if ('Delta_log_X' in prior_ranges):
                        prior_ranges[parameter] = prior_ranges['Delta_log_X']
                    else:
                        prior_ranges[parameter] = prior_ranges_defaults['Delta_log_X']

                # Set non-specified mixing ratio gradient prior to that for 'Grad_log_X'
                elif ('Grad_' in parameter):
                    if ('Grad_log_X' in prior_ranges):
                        prior_ranges[parameter] = prior_ranges['Grad_log_X']
                    else:
                        prior_ranges[parameter] = prior_ranges_defaults['Grad_log_X']
                    
                # Set non-specified mixing ratio prior to that for 'log_X'
                elif ('log_' in parameter):
                    if ('log_X' in prior_ranges):
                        prior_ranges[parameter] = prior_ranges['log_X']
                    else:
                        prior_ranges[parameter] = prior_ranges_defaults['log_X']

            # Set non-specified temperature difference parameters to that for 'Delta_T'
            elif ('Delta_T_' in parameter):
                if ('Delta_T' in prior_ranges):
                    prior_ranges[parameter] = prior_ranges['Delta_T']
                else:
                    prior_ranges[parameter] = prior_ranges_defaults['Delta_T']

            # Set non-specified temperature gradient parameters to that for 'Grad_T'
            elif ('Grad_' in parameter):
                if ('Grad_T' in prior_ranges):
                    prior_ranges[parameter] = prior_ranges['Grad_T']
                else:
                    prior_ranges[parameter] = prior_ranges_defaults['Grad_T']

            # Set non-specified temperature parameters to that for 'T'
            elif ('T_' in parameter):
                if ('T' in prior_ranges):
                    prior_ranges[parameter] = prior_ranges['T']
                else:
                    prior_ranges[parameter] = prior_ranges_defaults['T']

            # Check if user didn't specify a distance prior for an imaged object 
            elif (parameter == 'd'):
                raise Exception("Error: no prior range provided for the system " +
                                " distance. Perhaps there is GAIA data you can " +
                                "use to prescribe a Gaussian prior for the " +
                                "'d' parameter?")
            
            # For all other non-specified parameters, use the default values
            else:
                prior_ranges[parameter] = prior_ranges_defaults[parameter]

        # Check for parameters without a user-specified prior type
        if (parameter not in prior_types):

            # Special case for mixing ratio parameters
            if (parameter in X_param_names):

                # Set non-specified pressure of mid mixing ratio prior to that for 'log_P_mid'
                if ('log_P_' in parameter):
                    if ('log_P_X_mid' in prior_types):
                        prior_types[parameter] = prior_types['log_P_X_mid']
                    else:
                        prior_types[parameter] = 'uniform'

                # Set non-specified mixing ratio difference prior to that for 'Delta_log_X'
                elif ('Delta_log_' in parameter):
                    if ('Delta_log_X' in prior_types):
                        prior_types[parameter] = prior_types['Delta_log_X']
                    else:
                        prior_types[parameter] = 'uniform'

                # Set non-specified mixing ratio gradient prior to that for 'Grad_log_X'
                elif ('Grad_' in parameter):
                    if ('Grad_log_X' in prior_types):
                        prior_types[parameter] = prior_types['Grad_log_X']
                    else:
                        prior_types[parameter] = 'uniform'

                # Set non-specified mixing ratio prior to that for 'log_X'
                elif ('log_' in parameter):
                    if ('log_X' in prior_types):
                        prior_types[parameter] = prior_types['log_X']
                    else:
                        if ('CLR' in prior_types.values()): # If any parameters CLR, set all to CLR
                            prior_types[parameter] = 'CLR'
                        else:
                            prior_types[parameter] = 'uniform'
                
            # Set non-specified temperature difference parameters to that for 'Delta_T'
            elif ('Delta_T_' in parameter):
                if ('Delta_T' in prior_types):
                    prior_types[parameter] = prior_types['Delta_T']
                else:
                    prior_types[parameter] = 'uniform'

            # Set non-specified temperature gradient parameters to that for 'Grad_T'
            elif ('Grad_' in parameter):
                if ('Grad_T' in prior_types):
                    prior_types[parameter] = prior_types['Grad_T']
                else:
                    prior_types[parameter] = 'uniform'

            # Set non-specified temperature parameters to that for 'T'
            elif ('T_' in parameter):
                if ('T' in prior_types):
                    prior_types[parameter] = prior_types['T']
                else:
                    prior_types[parameter] = 'uniform'

            # The stellar T_phot and log_g_phot default to a Gaussian prior
            elif (parameter in ['T_phot', 'log_g_phot']):
                prior_types[parameter] = 'gaussian'
            
            # All other parameters default to uniform priors
            else:
                prior_types[parameter] = 'uniform'

    # If the user provided a single prior for mixing ratios or temperature,
    # that parameter can be removed now that all parameters have separate priors

    # Remove group prior range for mixing ratio and temperature parameters
    if ('log_P_X_mid' in prior_ranges):
        del prior_ranges['log_P_X_mid']
    if ('log_X' in prior_ranges):
        del prior_ranges['log_X']
    if ('Delta_log_X' in prior_ranges):
        del prior_ranges['Delta_log_X']
    if ('Grad_log_X' in prior_ranges):
        del prior_ranges['Grad_log_X']
    if (('T' in prior_ranges) and (PT_profile != 'isotherm')):
        del prior_ranges['T']
    if (('Delta_T' in prior_ranges) and (PT_profile != 'gradient')):
        del prior_ranges['Delta_T']
    if (('Grad_T' in prior_ranges) and (PT_profile != 'gradient')):
        del prior_ranges['Grad_T']

    # Remove group prior types for mixing ratio and temperature parameters
    if ('log_P_X_mid' in prior_types):
        del prior_types['log_P_X_mid']
    if ('log_X' in prior_types):
        del prior_types['log_X']
    if ('Delta_log_X' in prior_types):
        del prior_types['Delta_log_X']
    if ('Grad_log_X' in prior_types):
        del prior_types['Grad_log_X']
    if (('T' in prior_types) and (PT_profile != 'isotherm')):
        del prior_types['T']
    if (('Delta_T' in prior_types) and (PT_profile != 'gradient')):
        del prior_types['Delta_T']
    if (('Grad_T' in prior_types) and (PT_profile != 'gradient')):
        del prior_types['Grad_T']

    CLR_limit_check = 0   # Tracking variable for CLR limit check below

    # Check that parameter types are all supported
    for parameter in param_names:

        # Check that sine parameters are only used for geometry parameters and theta_0_cloud
        if ((prior_types[parameter] == 'sine') and (parameter not in ['alpha', 'beta', 'theta_0'])):
            raise Exception("Unsupported prior for " + parameter)

        # Check that centred-log ratio prior is only used for mixing ratios
        if ((prior_types[parameter] == 'CLR') and (parameter not in X_param_names)):
            raise Exception("Unsupported prior for " + parameter)

        # Check that centred-log ratio is being employed in a 1D model
        if ((prior_types[parameter] == 'CLR') and (Atmosphere_dimension != 1)):
            raise Exception("CLR prior only supported for 1D models.")
        
        if ((parameter in ['T_spot', 'T_fac', 'log_g_spot', 'log_g_fac']) and 
            (prior_types[parameter] == 'gaussian')):
            raise Exception("Gaussian priors can only be used on T_phot or log_g_phot.")

        # Check mixing ratio parameter have valid settings
        if (parameter in X_param_names):

            if (prior_types[parameter] not in ['uniform', 'CLR']):
                raise Exception("Only uniform and CLR priors supported for mixing ratio parameters.")
            
            # Check that centred-log ratio prior is set for all mixing ratio parameters
            if (('CLR' in prior_types.values()) and (prior_types[parameter] != 'CLR')):
                raise Exception("When using a CLR prior, all mixing ratio parameters " + 
                                "must also have a CLR prior")
            
            # Check that all CLR variables have the same lower limit
            if (prior_types[parameter] == 'CLR'):

                CLR_limit = prior_ranges[parameter][0]
                
                if (CLR_limit_check == 0):       # First parameter passes check
                    CLR_limit_check = CLR_limit

                if (CLR_limit != CLR_limit_check):
                    raise Exception("When using a CLR prior, all mixing ratio " + 
                                    "parameters must have the same lower limit.")
                else:
                    CLR_limit_check = CLR_limit

    # Package prior properties
    priors = {'prior_ranges': prior_ranges, 'prior_types': prior_types}

    return priors


