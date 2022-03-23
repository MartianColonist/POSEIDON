''' 
POSEIDON CORE ROUTINE

Copyright 2022, Ryan J. MacDonald.

'''

import os

# Force a single core to be used by numpy (mpirun handles parallelisation)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CBLAS_NUM_THREADS'] = '1'

import numpy as np
from numba.core.decorators import jit
import scipy.constants as sc
from mpi4py import MPI
from spectres import spectres

from .constants import R_Sun, R_J, R_E

from .utility import create_directories, write_spectrum, read_data
from .stellar import planck_lambda, load_stellar_pysynphot, \
                     stellar_contamination_general
from .supported_opac import supported_species, supported_cia, inactive_species
from .parameters import assign_free_params, reformat_log_X, generate_state, \
                        unpack_geometry_params, unpack_cloud_params
from .absorption import opacity_tables, store_Rayleigh_eta_LBL, extinction, \
                        extinction_LBL
from .geometry import atmosphere_regions, angular_grids
from .atmosphere import profiles
from .instrument import init_instrument
from .transmission import TRIDENT
from .emission import emission_rad_transfer


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def create_star(R_s, T_eff, log_g, Met, T_eff_error = 100.0, 
                stellar_spectrum = True, grid = 'blackbody',
                heterogeneous = False, f_het = 0.0, T_het = None):
    '''
    Initialise the stellar dictionary object used by POSEIDON.

    Args:
        R_s (float): 
            Stellar radius (m).
        T_eff (float):
            Stellar effective temperature (K).
        err_T_s (float):
            A priori 1-sigma error on stellar effective temperature (K).
        Met_s (float),
            Stellar metallicity [log10(Fe/H_star / Fe/H_solar)].
        log_g_s (int):
            Stellar log surface gravity (log10(cm/s^2) by convention).
        stellar_spectrum (bool):
            If True, compute a stellar spectrum.
        grid (string):
            Stellar model grid to use if 'stellar_spectrum' is True.
            (Options: blackbody / cbk04 / phoenix).
    
    Returns:
        star (dict):
            Collection of stellar properties used by POSEIDON.

    '''

    # Compute stellar spectrum
    if (stellar_spectrum == True):

        if (grid == 'blackbody'):

            # Create fiducial wavelength grid for blackbody spectrum
            wl_min = 0.2
            wl_max = 20.0
            R = 10000

            # This grid should be broad enough for most applications
            wl_star = wl_grid_constant_R(wl_min, wl_max, R)

            # Evaluate Planck function at stellar effective temperature
            I_phot = planck_lambda(T_eff, wl_star)

        else:

            # Obtain photosphere spectrum by interpolating stellar grids
            wl_star, I_phot = load_stellar_pysynphot(T_eff, Met, log_g, 
                                                     grid = grid)

        # For uniform stellar surfaces
        if (heterogeneous == False):

            # No heterogeneity spectrum to return 
            I_het = np.zeros(len(wl_star))   

            # Surface flux is pi * intensity
            F_star = np.pi * I_phot

        # For non-uniform stellar surfaces
        elif (heterogeneous == True):

            # Obtain heterogeneity spectrum by interpolation
            _, I_het = load_stellar_pysynphot(T_het, Met, log_g, grid = grid)

            # Evaluate total stellar flux as a weighted sum of each region 
            F_star = np.pi * ((f_het * I_het) + (1.0 - f_het) * I_phot)

    # If user doesn't require a stellar spectrum
    else:

        F_star = None
        wl_star = None

    # Package stellar properties
    star = {'stellar_radius': R_s, 'stellar_T_eff': T_eff, 
            'stellar_T_eff_error': T_eff_error, 'stellar_metallicity': Met, 
            'stellar_log_g': log_g, 'F_star': F_star, 'wl_star': wl_star,
            'f_het': f_het, 'T_het': T_het, 'I_phot': I_phot, 'I_het': I_het
           }

    return star


def create_planet(planet_name, R_p, mass = None, gravity = None, 
                  log_g = None, T_eq = None, d = None, d_err = None, b_p = 0.0):
    '''
    Initialise the stellar dictionary object used by POSEIDON.

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
            Instead of g, can provide log_10 (g / cm/s^2).
        T_eq (float):
            Planetary equilibrium temperature (zero albedo) (K). 
        d (float):
            Distance to system (m).
        d_err (float):
            Measured error on system distance (m).
        b_p (float),
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
                 cloud_type = 'deck', gravity_setting = 'fixed',
                 stellar_contam = 'No', offsets_applied = 'No', 
                 error_inflation = 'No', radius_unit = 'R_J',
                 PT_dim = 1, X_dim = 1, cloud_dim = 1, TwoD_type = None, 
                 TwoD_param_scheme = 'difference', species_EM_gradient = [], 
                 species_DN_gradient = [], species_vert_gradient = []):
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
            (Options: isotherm / gradient / Madhu / slope).
        X_profile (str):
            Chosen mixing ratio profile parametrisation
            (Options: isochem / gradient).
        cloud_model (str):
            Chosen cloud parametrisation 
            (Options: cloud-free / MacMad17 / Iceberg).
        cloud_type (str):
            Cloud extinction type to consider 
            (Options: deck / haze / deck_haze).
        gravity_setting (str):
            Whether log_g is fixed or a free parameter.
            (Options: fixed / free).
        stellar_contam (str):
            Chosen prescription for modelling unocculted stellar contamination
            (Options: No / one-spot).
        offsets_applied (str):
            Whether a relative offset should be applied to a dataset 
            (Options: No / single-dataset).
        error_inflation (str):
            Whether to consider inflation of error bars in a retrieval
            (Options: No / Line15).
        radius_unit (str)
            Planet radius unit used to report retrieval results
            (Options: R_J / R_E)
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
            (Options: 1 / 2 / 3).object_type = 'transiting', 
        TwoD_type (str):
            For 2D models, specifies whether the model considers day-night
            gradients or evening-morning gradients
            (Options: D-N / E-M).
        TwoD_param_scheme (str):
            For 2D models, specifies which quantities should be consider as
            free parameters (e.g. day & night vs. terminator & day-night diff.)
            (Options: absolute / difference).
        species_EM_gradient (list of str):
            Chemical species with an evening-morning mixing ratio gradient.
        species_DN_gradient (list of str):
            Chemical species with a day-night mixing ratio gradient.
        species_vert_gradient (list of str):
            Chemical species with a vertical mixing ratio gradient.
    
    Returns:
        model (dict):
            A specific description of the desired POSEIDON model.

    '''

    #***** Create chemical species arrays *****#

    # Create array containing all chemical species in model
    bulk_species = np.array(bulk_species)
    param_species = np.array(param_species)
    chemical_species = np.append(bulk_species, param_species)

    # Identify chemical species with active spectral features
    active_species = chemical_species[~np.isin(chemical_species, inactive_species)]

    # Convert arrays specifying which species have gradients into numpy arrays
    species_vert_gradient = np.array(species_vert_gradient)

    # Check if cross sections are available for all the chemical species
    if (np.any(~np.isin(active_species, supported_species)) == True):
        raise Exception("A chemical species you selected is not supported.\n")

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

    # Create list of bound-free absorption pairs
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
                                      species_vert_gradient, Atmosphere_dimension)

    # Package model properties
    model = {'model_name': model_name, 'object_type': object_type,
             'Atmosphere_dimension': Atmosphere_dimension,
             'PT_profile': PT_profile, 'X_profile': X_profile,
             'cloud_model': cloud_model, 'cloud_type': cloud_type,
             'gravity_setting': gravity_setting, 
             'chemical_species': chemical_species, 'bulk_species': bulk_species,
             'active_species': active_species, 'CIA_pairs': CIA_pairs,
             'ff_pairs': ff_pairs, 'bf_species': bf_species,
             'param_species': param_species, 'radius_unit': radius_unit,
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
             'X_dim': X_dim, 'cloud_dim': cloud_dim
            }

    return model


def wl_grid_constant_R(wl_min, wl_max, R):
    '''
    Create a wavelength array with constant spectral resolution (R = wl/dwl).

    Args:
        wl_min (float):
            Minimum wavelength of grid (micron).
        wl_max (float): 
            Maximum wavelength of grid (micron).
        R (int or float):
            Spectral resolution of desired wavelength grid.
    
    Returns:
        wl (np.array of float):
            Model wavelength grid (micron).

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
            Minimum wavelength of grid (micron).
        wl_max (float): 
            Maximum wavelength of grid (micron).
        line_by_line_R (float):
            Wavenumber resolution of pre-computer opacity database (0.01 cm^-1).
    
    Returns:
        wl (np.array of float):
            Model wavelength grid (micron).

    '''

    nu_min = 1.0e4/wl_max   # Minimum wavenumber on output grid
    nu_max = 1.0e4/wl_min   # Maximum wavenumber on output grid
    
    # Round so wavenumber grid bounds to match cross section resolution (0.01 cm^-1)
    nu_min = np.around(nu_min, np.abs(np.int(np.log10(line_by_line_res))))
    nu_max = np.around(nu_max, np.abs(np.int(np.log10(line_by_line_res))))
    
    # Find number of wavenumber points on grid
    N_nu = np.int((nu_max - nu_min)/line_by_line_res)
    N_wl = N_nu
    
    # Initialise line-by-line model wavenumber grid
    nu = np.linspace(nu_max, nu_min, N_nu)                          # Decreasing wavenumber order
    nu = np.around(nu, np.abs(np.int(np.log10(line_by_line_res))))  # Remove floating point errors
    
    # Initialise corresponding wavelength grid 
    wl = 1.0e4/nu   # Convert from cm^-1 to um
    
    return wl
    

def read_opacities(model, wl, opacity_treatment = 'opacity_sampling', 
                   T_fine = None, log_P_fine = None, opacity_database = 'High-T'):
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
            Model wavelength grid (microns).
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
    
    Returns:
        opac (dict):
            Collection of numpy arrays storing cross sections for the molecules, 
            atoms, and ions contained in the model. Separate arrays store 
            standard cross sections, CIA, free-free and bound-free opacity, 
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
                                              log_P_fine, opacity_database)
                    
    elif (opacity_treatment == 'line_by_line'):   
        
        # For line-by-line case, we still compute Rayleigh scattering in advance
        Rayleigh_stored, eta_stored = store_Rayleigh_eta_LBL(wl, chemical_species)   
        
        # No need for pre-computed arrays for line-by-line, so keep empty arrays
        sigma_stored, CIA_stored, \
        ff_stored, bf_stored = (np.array([]) for _ in range(4))

    # Package opacity data required by our model in memory
    opac = {'opacity_database': opacity_database, 
            'opacity_treatment': opacity_treatment, 'sigma_stored': sigma_stored, 
            'CIA_stored': CIA_stored, 'Rayleigh_stored': Rayleigh_stored, 
            'eta_stored': eta_stored, 'ff_stored': ff_stored, 
            'bf_stored': bf_stored, 'T_fine': T_fine, 'log_P_fine': log_P_fine
           }

    return opac  


def make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, log_X_params, 
                    cloud_params = [], geometry_params = [], log_g = None,
                    He_fraction = 0.17, N_slice_EM = 2, N_slice_DN = 4,
                    retrieval_run = False):
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
        log_g (float):
            Gravitational field of planet - only needed for free log_g parameter
        R_p_ref (float):
            Planet radius corresponding to reference pressure (m).
        PT_params (np.array of float):
            Parameters defining the pressure-temperature field.
        log_X_params (2D np.array of float):
            Parameters defining the log-mixing ratio field.
        cloud_params (np.array of float):
            Parameters defining atmospheric aerosols.
        geometry_params (np.array of float):
            Terminator opening angle parameters.
        ignore_species (list):
            Any chemical species to ignore in generating atmosphere.
        He_fraction (float):
            Assumed H2/He ratio (0.17 default corresponds to the Sun).
        N_slice_EM (even int):
            Number of azimuthal slices in the evening-morning transition region.
        N_slice_DN (even int):
            Number of zenith slices in the day-night transition region.
        P_deep (float):
            For P-T gradient profile, P > P_deep is isothermal.
        P_high (float):
            For P-T gradient profile, P < P_high is isothermal.
        retrieval_run (bool):
            True if a retrieval is being run, False for forward models.
    
    Returns:
        atmosphere (dict):
            Collection of the atmospheric properties required to compute the
            resultant spectra of the planet.
    
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

    # Unpack planet properties
    R_p = planet['planet_radius']

    # Load planet gravity
    if (gravity_setting == 'fixed'):
        g_p = planet['planet_gravity']   # For fixed g, load from planet object
    elif (gravity_setting == 'free'):
        if (log_g is None):
            raise Exception("Must provide 'log_g' when log_g a free parameter")
        else:
            g_p = np.power(10.0, log_g)/100   # Convert log cm/s^2 to m/s^2

    # Unpack lists of chemical species in this model
    chemical_species = model['chemical_species']
    active_species = model['active_species']
    bulk_species = model['bulk_species']
    CIA_pairs = model['CIA_pairs']
    ff_pairs = model['ff_pairs']
    bf_species = model['bf_species']

    #***** First, establish model geometry *****# 

    # Check that the number of azimuthal and zenith slices are even
    if ((N_slice_EM % 2 != 0) or (N_slice_DN % 2 != 0)):
        raise Exception("The number of slices resolving the day-night or " +
                        "morning-evening transition regions must be even.")

    # For 2D or 3D models, find the number of atmospheric sectors and zones
    N_sectors, N_zones = atmosphere_regions(Atmosphere_dimension, TwoD_type, 
                                            N_slice_EM, N_slice_DN)

    # Unpack terminator opening angles (for 2D or 3D models)
    alpha, beta = unpack_geometry_params(param_names, N_params_cum, 
                                         geometry_params)

    # Compute discretised angular grids for multidimensional atmospheres
    phi, theta, phi_edge, \
    theta_edge, dphi, dtheta = angular_grids(Atmosphere_dimension, TwoD_type, 
                                             N_slice_EM, N_slice_DN, 
                                             alpha, beta)

    #***** Generate state arrays for the PT and mixing ratio profiles *****# 

    # For forward models, reformat mixing ratio parameters to flat array
    if (retrieval_run is False):
        log_X_flat_array = reformat_log_X(log_X_params, param_species, X_profile, 
                                          X_dim, TwoD_type, TwoD_param_scheme, 
                                          species_EM_gradient, species_DN_gradient, 
                                          species_vert_gradient)
    else:
        log_X_flat_array = log_X_params

    # Recast PT and mixing ratio parameters as state arrays used by atmosphere.py
    PT_state, \
    log_X_state = generate_state(PT_params, log_X_flat_array, param_species, 
                                 PT_dim, X_dim, PT_profile, X_profile, TwoD_type, 
                                 TwoD_param_scheme, species_EM_gradient, 
                                 species_DN_gradient, species_vert_gradient)

    #***** Compute P-T, radial, mixing ratio, and other atmospheric profiles *****#

    P, T, n, r, r_up, r_low, \
    dr, X, X_active, X_CIA, \
    X_ff, X_bf, mu, \
    is_physical = profiles(P, R_p, g_p, PT_profile, X_profile, PT_state, P_ref, 
                           R_p_ref, log_X_state, chemical_species, bulk_species, 
                           param_species, active_species, CIA_pairs, 
                           ff_pairs, bf_species, N_sectors, N_zones, alpha, 
                           beta, phi, theta, species_vert_gradient, He_fraction)

    #***** Store cloud / haze / aerosol properties *****#

    kappa_cloud_0, P_cloud, \
    f_cloud, phi_cloud_0, \
    theta_cloud_0, \
    a, gamma = unpack_cloud_params(param_names, cloud_params, cloud_model, cloud_dim, 
                                   N_params_cum, TwoD_type)

    # Package atmosphere properties
    atmosphere = {'P': P, 'T': T, 'g': g_p, 'n': n, 'r': r, 'r_up': r_up,
                  'r_low': r_low, 'dr': dr, 'X': X, 'X_active': X_active, 
                  'X_CIA': X_CIA, 'X_ff': X_ff, 'X_bf': X_bf, 'mu': mu, 
                  'N_sectors': N_sectors, 'N_zones': N_zones, 'alpha': alpha,
                  'beta': beta, 'phi': phi, 'theta': theta, 'phi_edge': phi_edge, 
                  'theta_edge': theta_edge, 'dphi': dphi, 'dtheta': dtheta,
                  'kappa_cloud_0': kappa_cloud_0, 'P_cloud': P_cloud, 
                  'f_cloud': f_cloud, 'phi_cloud_0': phi_cloud_0, 
                  'theta_cloud_0': theta_cloud_0, 'a': a, 'gamma': gamma, 
                  'is_physical': is_physical
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

        else:
            return True
            

def compute_spectrum(planet, star, model, atmosphere, opac, wl,
                     spectrum_type = 'transmission', save_spectrum = False):
    '''
    Solves the radiative transfer equation to compute the transmission
    spectrum of the model atmosphere.

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
            Model wavelength grid (microns)
        spectrum_type (str):
            The type of spectrum for POSEIDON to compute
            (Options: transmission / emission / direct_emission).
        write_spectrum (bool):
            If True, writes the spectrum to './POSEIDON_output/PLANET/spectra/'.

    Returns:
        spectrum (np.array of float):
            The desired atmospheric spectrum.
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
                              'dayside_emission', 'nightside_emission']):
        raise Exception("Only transmission spectra and emission " +
                        "spectra are currently supported.")
    elif (('emission' in spectrum_type) and 
         ((PT_dim + X_dim + cloud_dim) != 3)):
        raise Exception("Only 1D emission spectra currently supported.")

    # Unpack planet and star properties
    b_p = planet['planet_impact_parameter']
    R_p = planet['planet_radius']
    d = planet['system_distance']

    if (star is not None):
        R_s = star['stellar_radius']

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
                                                  N_sectors, N_zones, 
                                                  opacity_database)
        
    # If using opacity sampling, we can use pre-interpolated cross sections
    elif (opac['opacity_treatment'] == 'opacity_sampling'):

        # Unpack pre-interpolated cross sections
        sigma_stored = opac['sigma_stored']
        CIA_stored = opac['CIA_stored']
        Rayleigh_stored = opac['Rayleigh_stored']
        ff_stored = opac['ff_stored']
        bf_stored = opac['bf_stored']

        # Also unpack fine temeprature and pressure grids from pre-interpolation
        T_fine = opac['T_fine']
        log_P_fine = opac['log_P_fine']
        
        # Calculate extinction coefficients in standard mode
        kappa_clear, kappa_cloud = extinction(chemical_species, active_species,
                                              CIA_pairs, ff_pairs, bf_species,
                                              n, T, P, wl, X, X_active, X_CIA, 
                                              X_ff, X_bf, a, gamma, P_cloud, 
                                              kappa_cloud_0, sigma_stored, 
                                              CIA_stored, Rayleigh_stored, 
                                              ff_stored, bf_stored, enable_haze, 
                                              enable_deck, N_sectors, N_zones,
                                              T_fine, log_P_fine)

    # Generate transmission spectrum        
    if (spectrum_type == 'transmission'):

        # Place planet at mid-transit (spectrum identical due to translational symmetry)
        y_p = 0   # Coordinate of planet centre along orbit (y=0 at mid-transit)

        # Call the core TRIDENT routine to compute the transmission spectrum
        spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                           enable_deck, enable_haze, b_p, y_p, R_s, f_cloud,
                           phi_cloud_0, theta_cloud_0, phi_edge, theta_edge)

    # Generate emission spectrum
    elif ('emission' in spectrum_type):

        # If distance not specified, use fiducial value
        if (d is None):
            d = 1        # This value only used for flux ratios, so it cancels

        # Compute planet flux
        F_p = emission_rad_transfer(T, dr, wl, kappa_clear)

        # Convert planet surface flux to observed flux at Earth
        F_p_obs = (R_p / d)**2 * F_p

        # For direct emission spectra (brown dwarfs and directly imaged planets)        
        if (spectrum_type == 'direct_emission'):

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

            # Final spectrum is the planet-star flux ratio
            spectrum = F_p_obs / F_s_obs

    # Write spectrum to file
    if (save_spectrum == True):
        write_spectrum(planet['planet_name'], model['model_name'], spectrum, wl)

    return spectrum


def load_data(data_dir, datasets, instruments, wl_model, offset_datasets = None):
    '''
    ADD DOCSTRING
    '''

    # If the user is running the retrieval tutorial, point to the reference data
    if (data_dir == 'Tutorial'):
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
        wl_data_i, half_bin_i, ydata_i, err_data_i = read_data(data_dir, datasets[i])
        
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
        
    N_data = len(ydata)  # Total number of data points
    
    # Cumulative sum of data lengths for indexing later
    len_data_idx = np.append(np.array([0]), np.cumsum(len_data)).astype(np.int64)       

    # For relative offsets, find which data indices the offset applies to
    if (offset_datasets is not None):
        offset_datasets = np.array(offset_datasets)
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
    ADD DOCSTRING
    '''

    # Unpack parameter names
    param_names = model['param_names']
    X_param_names = model['X_param_names']
    PT_profile = model['PT_profile']
    radius_unit = model['radius_unit']
    Atmosphere_dimension = model['Atmosphere_dimension']
    
    # Unpack planet and star properties
    R_p = planet['planet_radius']
    T_eq = planet['planet_T_eq']
    T_s = star['stellar_T_eff']
    err_T_s = star['stellar_T_eff_error']

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

    # Set default priors (used if user doesn't specify one or more priors)
    prior_ranges_defaults = {'T': [400, T_eq+200], 'Delta_T': [0, 1000],
                             'a1': [0.02, 2.00], 'a2': [0.02, 2.00],
                             'log_P1': [-6, 2], 'log_P2': [-6, 2],
                             'log_P3': [-2, 2], 
                             'R_p_ref': [0.85*R_p/R_p_norm, 1.15*R_p/R_p_norm],
                             'log_X': [-12, -1], 'Delta_log_X': [-8, 8], 
                             'log_a': [-4, 8], 'gamma': [-20, 2], 
                             'log_P_cloud': [-6, 2], 'phi_cloud': [0, 1],
                             'log_kappa_cloud': [-10, -4], 'f_cloud': [0, 1],
                             'phi_0': [-180, 180], 'theta_0': [-35, 35],
                             'alpha': [0.1, 180], 'beta': [0.1, 70],
                             'f_het': [0.0, 0.5], 'T_het': [0.6*T_s, 1.2*T_s],
                             'T_phot': [T_s, err_T_s], 
                             'delta_rel': [-1.0e-3, 1.0e-3],
                             'log_b': [np.log10(0.001*np.min(err_data**2)),
                                       np.log10(100.0*np.max(err_data**2))]
                            }    

    # Iterate through parameters, ensuring we have a full set of priors
    for parameter in param_names:

        # Check for parameters without a user-specified prior range
        if (parameter not in prior_ranges):
            
            # Special case for mixing ratio parameters
            if (parameter in X_param_names):

                # Set non-specified mixing ratio prior to that for 'log_X'
                if ('log_' in parameter):
                    if ('log_X' in prior_ranges):
                        prior_ranges[parameter] = prior_ranges['log_X']
                    else:
                        prior_ranges[parameter] = prior_ranges_defaults['log_X']

                # Set non-specified mixing ratio difference prior to that for 'Delta_log_X'
                elif ('Delta_log_' in parameter):
                    if ('Delta_log_X' in prior_ranges):
                        prior_ranges[parameter] = prior_ranges['Delta_log_X']
                    else:
                        prior_ranges[parameter] = prior_ranges_defaults['Delta_log_X']
                
            # Set non-specified temperature parameters to that for 'T'
            elif ('T_' in parameter):
                if ('T' in prior_ranges):
                    prior_ranges[parameter] = prior_ranges['T']
                else:
                    prior_ranges[parameter] = prior_ranges_defaults['T']
            
            # For all other non-specified parameters, use the default values
            else:
                prior_ranges[parameter] = prior_ranges_defaults[parameter]

        # Check for parameters without a user-specified prior type
        if (parameter not in prior_types):

            # Special case for mixing ratio parameters
            if (parameter in X_param_names):

                # Set non-specified mixing ratio prior to that for 'log_X'
                if ('log_' in parameter):
                    if ('log_X' in prior_types):
                        prior_types[parameter] = prior_types['log_X']
                    else:
                        if ('CLR' in prior_types.values()): # If any parameters CLR, set all to CLR
                            prior_types[parameter] = 'CLR'
                        else:
                            prior_types[parameter] = 'uniform'

                # Set non-specified mixing ratio difference prior to that for 'Delta_log_X'
                elif ('Delta_log_' in parameter):
                    if ('Delta_log_X' in prior_types):
                        prior_types[parameter] = prior_types['Delta_log_X']
                    else:
                        prior_types[parameter] = 'uniform'
                
            # Set non-specified temperature parameters to that for 'T'
            elif ('T_' in parameter):
                if ('T' in prior_types):
                    prior_types[parameter] = prior_types['T']
                else:
                    prior_types[parameter] = 'uniform'

            # Only the stellar T_phot defaults to a Gaussian prior
            elif (parameter == 'T_phot'):
                prior_types[parameter] = 'gaussian'
            
            # All other parameters default to uniform priors
            else:
                prior_types[parameter] = 'uniform'

    # If the user provided a single prior for mixing ratios or temperature,
    # that parameter can be removed now that all parameters have separate priors

    # Remove group prior types for mixing ratio and temperature parameters
    if ('log_X' in prior_types):
        del prior_types['log_X']
    if ('Delta_log_X' in prior_types):
        del prior_types['Delta_log_X']
    if (('T' in prior_types) and (PT_profile != 'isotherm')):
        del prior_types['T']

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
                    raise Exception("When using a CLR prior, all mixing ratio parameters " + 
                                    "must have the same lower limit.")
                else:
                    CLR_limit_check = CLR_limit

    # Package prior properties
    priors = {'prior_ranges': prior_ranges, 'prior_types': prior_types}

    return priors










