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
import cmasher as cmr
import matplotlib.pyplot as plt

from .constants import R_J, R_E
from .utility import create_directories, write_spectrum, read_data, prior_index, prior_index_V2, closest_index, closest_index_GPU, \
                     shared_memory_array, mock_missing, bin_spectrum
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
from .emission import emission_single_stream, determine_photosphere_radii, \
                      emission_single_stream_GPU, determine_photosphere_radii_GPU, \
                      emission_Toon, reflection_Toon

from .species_data import polarisabilities

from .clouds import Mie_cloud, Mie_cloud_free, load_aerosol_grid

from .utility import mock_missing

from .core import compute_spectrum

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

    # Fix for numerical rounding error
    wl[0] = wl_min
    wl[-1] = wl_max
    
    return wl

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

#################################
# Spectral Contribution Functions
#################################

def extinction_spectral_contribution(chemical_species, active_species, cia_pairs, ff_pairs, bf_species, aerosol_species,
               n, T, P, wl, X, X_active, X_cia, X_ff, X_bf, a, gamma, P_cloud, 
               kappa_cloud_0, sigma_stored, cia_stored, Rayleigh_stored, ff_stored, 
               bf_stored, enable_haze, enable_deck, enable_surface, N_sectors, 
               N_zones, T_fine, log_P_fine, P_surf, enable_Mie, n_aerosol_array, 
               sigma_Mie_array, P_deep = 1000.0,
               contribution_species = '',
               bulk_species = False,
               cloud_contribution = False,
               cloud_species = '',
               cloud_total_contribution = False,
               put_one_in = False,
               take_one_out = False,
               fix_mu = True):
    
    ''' Main function to evaluate extinction coefficients for molecules / atoms,
        Rayleigh scattering, hazes, and clouds for parameter combination
        chosen in retrieval step.
        
        Takes in cross sections pre-interpolated to 'fine' P and T grids
        before retrieval run (so no interpolation is required at each step).
        Instead, for each atmospheric layer the extinction coefficient
        is simply kappa = n * sigma[log_P_nearest, T_nearest, wl], where the
        'nearest' values are the closest P_fine, T_fine points to the
        actual P, T values in each layer. This results in a large speed gain.
        
        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), terminator
        sector, and wavelength.
    
    '''
    
    # Store length variables for mixing ratio arrays 
    N_species = len(chemical_species)        # Number of chemical species
    N_species_active = len(active_species)   # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)             # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
    N_bf_species = len(bf_species)           # Number of bound-free species included
    
    # Set up to find the bulk species indices (these are always turned on)
    # Find the number and indices for the bulk species 
    N_bulk_species = N_species - N_species_active
    # The bulk species are also the first few indices in the chemical species list
    bulk_species_indices = range(N_bulk_species)
    bulk_species_names = chemical_species[:N_bulk_species]

    # Find the name of the bulk species and check to see if they are 
    # also in the CIA list 
    bulk_cia_indices = []
    for i in range(len(cia_pairs)):
        pair_1, pair_2 = cia_pairs[i].split('-')
        pair_1_bool = False
        pair_2_bool = False 
        for j in bulk_species_names:
            if pair_1 == j:
                pair_1_bool = True
            if pair_2 == j:
                pair_2_bool = True
        
        if pair_1_bool == True and pair_2_bool == True:
            bulk_cia_indices.append(i)

    # Else, we are trying to find the contribution from a species
    if bulk_species == False and cloud_contribution == False:

        # Find the index in the list of chemical species 
        for i in range(len(chemical_species)):
            if contribution_species == chemical_species[i]:
                contribution_molecule_species_index = i

        # Find the index in the list of active species 
        for i in range(len(active_species)):
            if contribution_species == active_species[i]:
                contribution_molecule_active_index = i

        # Find the idnex in the list of CIA pairs
        cia_indices = []
        for i in range(len(cia_pairs)):
            pair_1, pair_2 = cia_pairs[i].split('-')
            if contribution_species == pair_1 or contribution_species == pair_2:
                cia_indices.append(i)


    N_wl = len(wl)     # Number of wavelengths on model grid
    N_layers = len(P)  # Number of layers
    
    # Define extinction coefficient arrays
    kappa_gas = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    
    # Fine temperature grid (for pre-interpolating opacities)    
    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)
    
    # Find index of deep pressure below which atmosphere is opaque
    i_bot = np.argmin(np.abs(P - P_deep))
    
    # If haze is enabled in this model
    if (enable_haze == 1):
        
        # Generalised scattering slope for haze
        slope = np.power((wl/0.35), gamma)    # Reference wavelength at 0.35 um
        
    # For each terminator sector (terminator plane)
    for j in range(N_sectors):
            
        # For each terminator zone (along day-night transition)
        for k in range(N_zones):
            
            # For each layer, find closest pre-computed cross section to P_fine, T_fine
            for i in range(i_bot,N_layers):
                
                n_level = n[i,j,k]
                
                # Find closest index in fine temperature array to given layer temperature
                idx_T_fine = closest_index(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
                idx_P_fine = closest_index(np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
                
                # For each collisionally-induced absorption (CIA) pair
                for q in range(N_cia_pairs): 
                    
                    # If bulk is false, we include both the bulk + contribution species 
                    if bulk_species == False and cloud_contribution == False:
                        if q in bulk_cia_indices or q in cia_indices:
                            n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                            n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                            n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair

                    # If bulk is true or cloud contribution is true, we are only interested in the bulk species contribution 
                    if bulk_species == True or cloud_contribution == True:
                        if q in bulk_cia_indices:
                            n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                            n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                            n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_n_cia * cia_stored[q, idx_T_fine, l]
                        
                # For each free-free absorption pair
                for q in range(N_ff_pairs): 
                    
                    n_ff_1 = n_level*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
                    n_ff_2 = n_level*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
                    n_n_ff = n_ff_1*n_ff_2             # Product of number densities of ff pair
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_n_ff * ff_stored[q, idx_T_fine, l]
                        
                # For each source of bound-free absorption (photodissociation)
                for q in range(N_bf_species): 
                    
                    n_q = n_level*X_bf[q,i,j,k]   # Number density of dissociating species
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_q * bf_stored[q,l]
                
                # For each molecular / atomic species with active absorption features

                for q in range(N_species_active): 
                    
                    # If we are not doing the bulk species, we just want the species 
                    if bulk_species == False and cloud_contribution == False:
                        if q == contribution_molecule_active_index:
                            n_q = n_level*X_active[q,i,j,k]   # Number density of this active species
                
                        else:
                            n_q = 0
                    
                    else:
                        # If bulk is true or cloud is true, then everything in active is turned off 
                        n_q = 0
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]
                    
                # For each molecular / atomic species
                for q in range(N_species):  
                    
                    if bulk_species == False and cloud_contribution == False:
                        if q == contribution_molecule_species_index:
                            n_q = n_level*X[q,i,j,k]   # Number density of given species
                        elif q in bulk_species_indices:
                            n_q = n_level*X[q,i,j,k]   # Number density of given species
                        else:
                            n_q = 0

                    else:
                        # If bulk is true or cloud is true, only keep the bulk species on 
                        if q in bulk_species_indices:
                            n_q = n_level*X[q,i,j,k]   # Number density of given species
                        else:
                            n_q = 0
                    
                    # For each wavelength
                    for l in range(N_wl):
                                
                        # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_Ray[i,j,k,l] += n_q * Rayleigh_stored[q,l]
        

            
            # CLOUD CONTRIBUTION 
            # If haze is enabled in this model  
            if (enable_haze == 1):
                
                # For each layer
                for i in range(i_bot,N_layers):
                    
                    haze_amp = (n[i,j,k] * a * 5.31e-31)   # Final factor is H2 Rayleigh scattering cross section at 350 nm
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # If we are not looking at the cloud contribution 
                        if cloud_contribution == False:
                            haze_amp = 0
                    
                        # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                        kappa_cloud[i,j,k,l] += haze_amp * slope[l]
                        
            # If a cloud deck is enabled in this model
            if (enable_deck == 1):

                if cloud_contribution == False:
                    kappa_cloud_0 = 0

                # Set extinction inside cloud deck
                kappa_cloud[(P > P_cloud[0]),j,k,:] += kappa_cloud_0

            # If a surface is enabled in this model
            if (enable_surface == 1):

                # Set extinction to infinity below surface
                kappa_gas[(P > P_surf),j,k,:] = 1.0e250

            # If Mie clouds are turned on 
            if (enable_Mie == 1):

                # All deck, slab, aerosol information is stored in the n_aerosol_array
                # If its an opaque deck, then the length of sigma_Mie_array will be one more than n_aerosol_array
                # Since the opaque deck is being counted as an extra aerosol 
                # Otherwise, it should be the same 

                # If cloud contribution = True, we need to find the species we want the contribution of 
                if cloud_contribution == True and cloud_total_contribution == False:
                    for n in range(len(aerosol_species)):
                        if cloud_species == aerosol_species[n]:
                            aerosol_species_index = n

                # No opaque clouds 
                if len(n_aerosol_array) == len(sigma_Mie_array):
                    for aerosol in range(len(n_aerosol_array)):
                        for i in range(i_bot,N_layers):
                            for q in range(len(wl)):
                                
                                # If we don't want the cloud contribution 
                                if cloud_contribution == False:
                                    kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * 0

                                # If we want one species of cloud contribution 
                                elif cloud_contribution == True and cloud_total_contribution == False:

                                    if aerosol == aerosol_species_index:
                                        kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * sigma_Mie_array[aerosol][q]

                                    else:
                                        kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * 0

                                # If we want the total cloud contribution 
                                elif cloud_total_contribution == True:
                                    kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * sigma_Mie_array[aerosol][q]
                        

                # Opaque Deck is the first element in n_aerosol_array
                # I will decide for now to add the opaque deck to each species. I think that won't hurt to keep them together 
                else:
                    for aerosol in range(len(n_aerosol_array)):
                            
                            if cloud_contribution == False:
                                if aerosol == 0:
                                    kappa_cloud[(P > P_cloud[0]),j,k,:] += 0

                                else:
                                    for i in range(i_bot,N_layers):
                                        for q in range(len(wl)):
                                            kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* 0
                            
                            elif cloud_contribution == True and cloud_total_contribution == False:

                                if aerosol == 0:
                                    kappa_cloud[(P > P_cloud[0]),j,k,:] += 1.0e250

                                else:
                                    for i in range(i_bot,N_layers):
                                        for q in range(len(wl)):
                                            if aerosol - 1 == aerosol_species_index:
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* sigma_Mie_array[aerosol-1][q]
                                            else:
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* 0

                            elif cloud_total_contribution == True:
                                    
                                if aerosol == 0:
                                    kappa_cloud[(P > P_cloud[0]),j,k,:] += 1.0e250

                                else:
                                    for i in range(i_bot,N_layers):
                                        for q in range(len(wl)):
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* sigma_Mie_array[aerosol-1][q]

    return kappa_gas, kappa_Ray, kappa_cloud
        
def spectral_contribution(planet, star, model, atmosphere, opac, wl,
                     spectrum_type = 'transmission', save_spectrum = False,
                     disable_continuum = False, suppress_print = False,
                     Gauss_quad = 2, use_photosphere_radius = True,
                     device = 'cpu', y_p = np.array([0.0]),
                     contribution_species_list = [],
                     bulk_species = True, 
                     cloud_contribution = False, 
                     cloud_species_list = [],
                     cloud_total_contribution = False,
                     put_one_in = True,
                     take_one_out = False,
                     fix_mu = True,
                     scattering_contribution = False,
                     reflection_contribution = False):
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
    scattering = model['scattering']
    reflection = model['reflection']

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
    P_cloud_bottom = atmosphere['P_cloud_bottom']

    # Check if haze enabled in the cloud model
    if ('haze' in model['cloud_type']):
        enable_haze = 1
    else:
        enable_haze = 0

    # Check if a cloud deck is enabled in the cloud model
    # The cloud deck is handled differently for Mie calclations
    if ('deck' in model['cloud_type'] and 'Mie' not in model['cloud_model']):
        enable_deck = 1
    else:
        enable_deck = 0

    if ('Mie' in model['cloud_model']):
        enable_Mie = 1
    else:
        enable_Mie = 0

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
        kappa_gas, kappa_Ray, kappa_cloud = extinction_LBL(chemical_species, active_species, 
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

                wl_Mie = wl_grid_constant_R(wl[0], wl[-1], 1000)

                # If its a fuzzy deck run
                if (model['cloud_type'] == 'fuzzy_deck'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                          r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                          P_cloud = P_cloud,
                                                          log_n_max = log_n_max, 
                                                          fractional_scale_height = fractional_scale_height)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                     r_m, aerosol_species,
                                                     cloud_type = model['cloud_type'],
                                                     aerosol_grid = aerosol_grid,
                                                     P_cloud = P_cloud,
                                                     log_n_max = log_n_max, 
                                                     fractional_scale_height = fractional_scale_height)

                # If its a slab
                elif (model['cloud_type'] == 'slab'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                        r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                        log_X_Mie = log_X_Mie,
                                                        P_cloud = P_cloud,
                                                        P_cloud_bottom = P_cloud_bottom)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                    r_m, aerosol_species,
                                                    cloud_type = model['cloud_type'],
                                                    aerosol_grid = aerosol_grid,
                                                    log_X_Mie = log_X_Mie,
                                                    P_cloud = P_cloud,
                                                    P_cloud_bottom = P_cloud_bottom)
                            
                          
                # If its a uniform X run
                elif (model['cloud_type'] == 'uniform_X'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                          r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                          log_X_Mie = log_X_Mie)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                     r_m, aerosol_species,
                                                     cloud_type = model['cloud_type'],
                                                     aerosol_grid = aerosol_grid,
                                                     log_X_Mie = log_X_Mie)

                # If its a opaque_deck_plus_slab run 
                elif (model['cloud_type'] == 'opaque_deck_plus_slab'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                        r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                        log_X_Mie = log_X_Mie,
                                                        P_cloud = P_cloud,
                                                        P_cloud_bottom = P_cloud_bottom)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                    r_m, aerosol_species,
                                                    cloud_type = model['cloud_type'],
                                                    aerosol_grid = aerosol_grid,
                                                    log_X_Mie = log_X_Mie,
                                                    P_cloud = P_cloud,
                                                    P_cloud_bottom = P_cloud_bottom)
                        
                # If its a fuzzy_deck_plus_slab run 
                elif (model['cloud_type'] == 'fuzzy_deck_plus_slab'):

                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                     r_m, aerosol_species,
                                                     cloud_type = model['cloud_type'],
                                                     aerosol_grid = aerosol_grid,
                                                     P_cloud = P_cloud,
                                                     log_n_max = log_n_max, 
                                                     fractional_scale_height = fractional_scale_height,
                                                     log_X_Mie = log_X_Mie,
                                                     P_cloud_bottom = P_cloud_bottom)

            
            else:

                # Generate empty arrays so the dark god numba is satisfied
                n_aerosol = []
                sigma_ext_cloud = []
                    
                n_aerosol.append(np.zeros_like(r))
                sigma_ext_cloud.append(np.zeros_like(wl))

                n_aerosol = np.array(n_aerosol)
                sigma_ext_cloud = np.array(sigma_ext_cloud)

                w_cloud = np.zeros_like(wl)
                g_cloud = np.zeros_like(wl)


            # Calculate extinction coefficients in standard mode

            # Numba will get mad if P_cloud is not an array (because you can have more than one cloud now)
            # This line just makes sure that P_cloud is an array 
            if isinstance(P_cloud, np.ndarray) == False:
                P_cloud = np.array([P_cloud])

            kappa_gas, kappa_Ray, kappa_cloud = extinction(chemical_species, active_species,
                                                           CIA_pairs, ff_pairs, bf_species,
                                                           n, T, P, wl, X, X_active, X_CIA, 
                                                           X_ff, X_bf, a, gamma, P_cloud, 
                                                           kappa_cloud_0, sigma_stored, 
                                                           CIA_stored, Rayleigh_stored, 
                                                           ff_stored, bf_stored, enable_haze, 
                                                           enable_deck, enable_surface,
                                                           N_sectors, N_zones, T_fine, 
                                                           log_P_fine, P_surf, enable_Mie, 
                                                           n_aerosol, sigma_ext_cloud)
            
            
            # This is to store the contribution kappas
            kappa_gas_contribution_array = []
            kappa_cloud_contribution_array = []
            spectrum_contribution_list_names = []
            
            # If you want to see only the bulk species contribution, this runs first 
            if bulk_species == True:
    
                kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_spectral_contribution(chemical_species, active_species,
                                                                                    CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                                                                    n, T, P, wl, X, X_active, X_CIA, 
                                                                                    X_ff, X_bf, a, gamma, P_cloud, 
                                                                                    kappa_cloud_0, sigma_stored, 
                                                                                    CIA_stored, Rayleigh_stored, 
                                                                                    ff_stored, bf_stored, enable_haze, 
                                                                                    enable_deck, enable_surface,
                                                                                    N_sectors, N_zones, T_fine, 
                                                                                    log_P_fine, P_surf, enable_Mie, 
                                                                                    n_aerosol, sigma_ext_cloud,
                                                                                    bulk_species = True)

                kappa_gas_contribution_array.append(kappa_gas_temp)
                kappa_cloud_contribution_array.append(kappa_cloud_temp)

                spectrum_contribution_list_names.append('Bulk Species')
                        
            # Then it runs the rest of the molecules that are provided 
            if (contribution_species_list != []):

                for i in range(len(contribution_species_list)):

                    kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_spectral_contribution(chemical_species, active_species,
                                                                                    CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                                                                    n, T, P, wl, X, X_active, X_CIA, 
                                                                                    X_ff, X_bf, a, gamma, P_cloud, 
                                                                                    kappa_cloud_0, sigma_stored, 
                                                                                    CIA_stored, Rayleigh_stored, 
                                                                                    ff_stored, bf_stored, enable_haze, 
                                                                                    enable_deck, enable_surface,
                                                                                    N_sectors, N_zones, T_fine, 
                                                                                    log_P_fine, P_surf, enable_Mie, 
                                                                                    n_aerosol, sigma_ext_cloud,
                                                                                    contribution_species=contribution_species_list[i],
                                                                                    bulk_species = False,
                                                                                    cloud_contribution = False)

                    kappa_gas_contribution_array.append(kappa_gas_temp)
                    kappa_cloud_contribution_array.append(kappa_cloud_temp)

                    spectrum_contribution_list_names.append(contribution_species_list[i])

            # Cloud contribuiton 
            if cloud_contribution == True:

                # Do the total cloud contribution first 
                # Default for the non Mie clouds 
                if cloud_total_contribution == True or enable_Mie == False:

                    kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_spectral_contribution(chemical_species, active_species,
                                                                CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                                                n, T, P, wl, X, X_active, X_CIA, 
                                                                X_ff, X_bf, a, gamma, P_cloud, 
                                                                kappa_cloud_0, sigma_stored, 
                                                                CIA_stored, Rayleigh_stored, 
                                                                ff_stored, bf_stored, enable_haze, 
                                                                enable_deck, enable_surface,
                                                                N_sectors, N_zones, T_fine, 
                                                                log_P_fine, P_surf, enable_Mie, 
                                                                n_aerosol, sigma_ext_cloud,
                                                                cloud_contribution = True,
                                                                cloud_total_contribution = True)

                    kappa_gas_contribution_array.append(kappa_gas_temp)
                    kappa_cloud_contribution_array.append(kappa_cloud_temp)

                    spectrum_contribution_list_names.append('Total Clouds')
                
                # If you have cloud species, run this 
                if (cloud_species_list != []):
                    if enable_Mie == False:
                        raise Exception("Cloud species only available for Mie clouds")
                    
                    for i in range(len(cloud_species_list)):

                        kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_spectral_contribution(chemical_species, active_species,
                                            CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                            n, T, P, wl, X, X_active, X_CIA, 
                                            X_ff, X_bf, a, gamma, P_cloud, 
                                            kappa_cloud_0, sigma_stored, 
                                            CIA_stored, Rayleigh_stored, 
                                            ff_stored, bf_stored, enable_haze, 
                                            enable_deck, enable_surface,
                                            N_sectors, N_zones, T_fine, 
                                            log_P_fine, P_surf, enable_Mie, 
                                            n_aerosol, sigma_ext_cloud,
                                            cloud_contribution = True,
                                            cloud_species = cloud_species_list[i])

                        kappa_gas_contribution_array.append(kappa_gas_temp)
                        kappa_cloud_contribution_array.append(kappa_cloud_temp)

                        spectrum_contribution_list_names.append(cloud_species_list[i])


        # Running POSEIDON on the GPU
        elif (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

    # Generate transmission spectrum        
    if (spectrum_type == 'transmission'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")
        
        # Total Spectrum First 
        spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, (kappa_gas + kappa_Ray), 
                            kappa_cloud, enable_deck, enable_haze, b_p, y_p[0],
                            R_s, f_cloud, phi_cloud_0, theta_cloud_0, phi_edge, 
                            theta_edge)

        spectrum_contribution_list = []

        for i in range(len(kappa_gas_contribution_array)):

            kappa_gas_temp = kappa_gas_contribution_array[i]
            kappa_cloud_temp = kappa_cloud_contribution_array[i]
        
            # Call the core TRIDENT routine to compute the transmission spectrum
            spectrum_temp = TRIDENT(P, r, r_up, r_low, dr, wl, (kappa_gas_temp + kappa_Ray), 
                            kappa_cloud_temp, enable_deck, enable_haze, b_p, y_p[0],
                            R_s, f_cloud, phi_cloud_0, theta_cloud_0, phi_edge, 
                            theta_edge)
            
            spectrum_contribution_list.append(spectrum_temp)

    # Generate emission spectrum
    elif ('emission' in spectrum_type):

        # Find zone index for the emission spectrum atmospheric region
        if ('dayside' in spectrum_type):
            zone_idx = 0
        elif ('nightside' in spectrum_type):
            zone_idx = -1
        else:
            zone_idx = 0

        # Use atmospheric properties from dayside/nightside (only consider one region for 1D emission models)
        dz = dr[:,0,zone_idx]
        T = T[:,0,zone_idx]
        
        # First, compute the normal 
        # Compute total extinction (all absorption + scattering sources)
        kappa_tot = (kappa_gas[:,0,zone_idx,:] + kappa_Ray[:,0,zone_idx,:] +
                     kappa_cloud[:,0,zone_idx,:])

        # Store differential extinction optical depth across each layer
        dtau_tot = np.ascontiguousarray(kappa_tot * dz.reshape((len(P), 1)))

        # Without scattering, compute single steam radiative transfer
        if (scattering == False):

            # Compute planet flux (on CPU or GPU)
            if (device == 'cpu'):
                F_p, dtau = emission_single_stream(T, dz, wl, kappa_tot, Gauss_quad)
            elif (device == 'gpu'):
                F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)

        elif (scattering == True):
            
            # Else, we need to restructure w_cloud and g_cloud to span by layer 
            # For Mie models with 1 species, the g and w can be help constant with each layer since
            # Kappa cloud will encode where clouds are
            # For models that are cloud free, you still need a g and w thats just an array of 0s
            # For Mie models with more than one species, we need to be more careful with the g and w array
            if len(aerosol_species) == 1 or aerosol_species == []:
                w_cloud = np.ones_like(kappa_cloud)*w_cloud
                g_cloud = np.ones_like(kappa_cloud)*g_cloud

            # Need to make a g and w array that vary with pressure layer where aerosols actually are 
            else:
                raise Exception('Only 1 aerosol species supported for scattering')

            # Compute planet flux including scattering (PICASO implementation), see emission.py for details
            F_p, dtau = emission_Toon(P, T, wl, dtau_tot, 
                                        kappa_Ray, kappa_cloud, kappa_tot,
                                        w_cloud, g_cloud, zone_idx,
                                        hard_surface = 0, tridiagonal = 0, 
                                        Gauss_quad = 5, numt = 1)
            
            dtau = np.flip(dtau, axis=0)   # Flip optical depth pressure axis back

        else:
            raise Exception("Error: Invalid scattering option")

        # Add in the seperate reflection  
        # FOR DEBUGGING PURPOSES, THIS JUST RETURNS THE ALBEDO RIGHT NOW
        if (reflection == True):

            albedo = reflection_Toon(P, wl, dtau_tot,
                                    kappa_Ray, kappa_cloud, kappa_tot,
                                    w_cloud, g_cloud, zone_idx,
                                    single_phase = 3, multi_phase = 0,
                                    frac_a = 1, frac_b = -1, frac_c = 2, constant_back = -0.5, constant_forward = 1,
                                    Gauss_quad = 5, numt = 1,
                                    toon_coefficients=0, tridiagonal=0, b_top=0)
            
            return P, wl, dtau_tot, kappa_Ray, kappa_cloud, kappa_tot, w_cloud, g_cloud, zone_idx

        # Calculate effective photosphere radius at tau = 2/3
        if (use_photosphere_radius == True):    # Flip to start at top of atmosphere
            
            # Running POSEIDON on the CPU
            if (device == 'cpu'):
                R_p_eff = determine_photosphere_radii(np.flip(dtau, axis=0), np.flip(r_low[:,0,zone_idx]),
                                                      wl, photosphere_tau = 2/3)
            
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

        # Then do the contribution functions 
        spectrum_contribution_list = []

        for i in range(len(kappa_gas_contribution_array)):

            kappa_gas_temp = kappa_gas_contribution_array[i]
            kappa_cloud_temp = kappa_cloud_contribution_array[i]

            # Compute total extinction (all absorption + scattering sources)
            kappa_tot = (kappa_gas_temp[:,0,zone_idx,:] + kappa_Ray[:,0,zone_idx,:] +
                        kappa_cloud_temp[:,0,zone_idx,:])

            # Store differential extinction optical depth across each layer
            dtau_tot = np.ascontiguousarray(kappa_tot * dz.reshape((len(P), 1)))

            # Without scattering, compute single steam radiative transfer
            if (scattering == False):

                # Compute planet flux (on CPU or GPU)
                if (device == 'cpu'):
                    F_p, dtau = emission_single_stream(T, dz, wl, kappa_tot, Gauss_quad)
                elif (device == 'gpu'):
                    F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)

            elif (scattering == True):

                # Else, we need to restructure w_cloud and g_cloud to span by layer 
                # For Mie models with 1 species, the g and w can be help constant with each layer since
                # Kappa cloud will encode where clouds are
                # For models that are cloud free, you still need a g and w thats just an array of 0s
                # For Mie models with more than one species, we need to be more careful with the g and w array
                if len(aerosol_species) == 1 or aerosol_species == []:
                    w_cloud = np.ones_like(kappa_cloud)*w_cloud
                    g_cloud = np.ones_like(kappa_cloud)*g_cloud

                # Need to make a g and w array that vary with pressure layer where aerosols actually are 
                else:
                    raise Exception('Only 1 aerosol species supported for scattering')

                # Compute planet flux including scattering (PICASO implementation), see emission.py for details
                F_p, dtau = emission_Toon(P, T, wl, dtau_tot, 
                                            kappa_Ray, kappa_cloud, kappa_tot,
                                            w_cloud, g_cloud, zone_idx,
                                            hard_surface = 0, tridiagonal = 0, 
                                            Gauss_quad = 5, numt = 1)
            
                
                dtau = np.flip(dtau, axis=0)   # Flip optical depth pressure axis back

            else:
                raise Exception("Error: Invalid scattering option")

            # Calculate effective photosphere radius at tau = 2/3
            if (use_photosphere_radius == True):    # Flip to start at top of atmosphere
                
                # Running POSEIDON on the CPU
                if (device == 'cpu'):
                    R_p_eff = determine_photosphere_radii(np.flip(dtau, axis=0), np.flip(r_low[:,0,zone_idx]),
                                                        wl, photosphere_tau = 2/3)
                
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
                spectrum_temp = F_p_obs

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
                spectrum_temp = F_p_obs / F_s_obs

            spectrum_contribution_list.append(spectrum_temp)
  
    return spectrum, spectrum_contribution_list_names, spectrum_contribution_list

def plot_spectral_contribution(planet, wl, spectrum, spectrum_contribution_list_names, spectrum_contribution_list,
                               full_spectrum_first = True, y_unit='transit_depth',
                               brightness_temperature = False, star = [],
                               stellar_spectrum = False,
                               y_min = None, y_max = None,):

    from POSEIDON.utility import plot_collection
    from POSEIDON.visuals import plot_spectra

    spectra = []

    # If brightness_temperature is true, convert from FpFs (or Fp) to brightness temeperature 
    if brightness_temperature == True:

        y_unit = 'T_bright'
        h = sc.h
        c = sc.c 
        kb = sc.k
        wl_brightness = wl * 1e-6

        Rp = planet['planet_radius']

        # Check to see if its directly imaged or not 
        if star != []:
            F_star = star['F_star']
            Rs = star['R_s']
            Fp = spectrum * F_star/((Rp/Rs)**2.) # Planet Flux
        else:
            Fp = spectrum

        spectrum = (h*c)/(kb * (wl_brightness) * np.log(1 + (2 * h * c**2 / ((Fp/np.pi) * (wl_brightness)**5))))

        for s in range(len(spectrum_contribution_list)):

            if star != []:
                Fp = spectrum_contribution_list[s] * F_star/((Rp/Rs)**2.) # Planet Flux
            else:
                Fp = spectrum_contribution_list[s]

            spectrum_contribution_list[s] = (h*c)/(kb * (wl_brightness) * np.log(1 + (2 * h * c**2 / ((Fp/np.pi) * (wl_brightness)**5))))


    if full_spectrum_first == True:
        colour_list = ['black','dimgray', 'darkturquoise', 'green', 'darkorchid', 'salmon', '#ff7f00', 'hotpink', 'red', 'orange', 'green', 'blue', 'purple']

        spectra = plot_collection(spectrum, wl, collection = spectra)
        labels = spectrum_contribution_list_names.copy()
        labels.insert(0,'Full Spectrum')

        # Loop through the contribution spectra 
        for s in spectrum_contribution_list:
            spectra = plot_collection(s, wl, collection = spectra)

        # Plot the full spectrum last so its on top 

        colour_list = colour_list[:len(spectrum_contribution_list)+1]
        
        fig = plot_spectra(spectra, planet, R_to_bin = 100,
                    plt_label = 'Spectral Contribution Plot',
                    spectra_labels = labels,
                    plot_full_res = False, 
                    save_fig = False,
                    colour_list = colour_list,
                    y_unit = y_unit,
                    y_min = y_min,
                    y_max = y_max)
        
    else: 
        
        colour_list = ['dimgray', 'darkturquoise', 'green', 'darkorchid', 'salmon', '#ff7f00', 'hotpink', 'red', 'orange', 'green', 'blue', 'purple']

        spectra = []
        # Loop through the contribution spectra 
        for s in spectrum_contribution_list:
            spectra = plot_collection(s, wl, collection = spectra)

        # Plot the full spectrum last so its on top 
        spectra = plot_collection(spectrum, wl, collection = spectra)

        colour_list = colour_list[:len(spectrum_contribution_list)]
        colour_list.append('black')

        labels = spectrum_contribution_list_names.copy()
        labels.append('Full Spectrum')

        
        fig = plot_spectra(spectra, planet, R_to_bin = 100,
                    plt_label = 'Cloud Contribution Plot',
                    spectra_labels = labels,
                    plot_full_res = False, 
                    save_fig = False,
                    colour_list = colour_list,
                    y_unit = y_unit,
                    y_min = y_min,
                    y_max = y_max)
        
#################################
# Pressure Contribution Functions
#################################

def extinction_pressure_contribution(chemical_species, active_species, cia_pairs, ff_pairs, bf_species, aerosol_species,
               n, T, P, wl, X, X_active, X_cia, X_ff, X_bf, a, gamma, P_cloud, 
               kappa_cloud_0, sigma_stored, cia_stored, Rayleigh_stored, ff_stored, 
               bf_stored, enable_haze, enable_deck, enable_surface, N_sectors, 
               N_zones, T_fine, log_P_fine, P_surf, enable_Mie, n_aerosol_array, 
               sigma_Mie_array, P_deep = 1000.0,
               contribution_species = '',
               bulk_species = False,
               cloud_contribution = False,
               cloud_species = '',
               cloud_total_contribution = False,
               put_one_in = False,
               take_one_out = False,
               fix_mu = True,
               layer_to_ignore = 0,
               total_pressure_contribution = False):
    
    ''' Main function to evaluate extinction coefficients for molecules / atoms,
        Rayleigh scattering, hazes, and clouds for parameter combination
        chosen in retrieval step.
        
        Takes in cross sections pre-interpolated to 'fine' P and T grids
        before retrieval run (so no interpolation is required at each step).
        Instead, for each atmospheric layer the extinction coefficient
        is simply kappa = n * sigma[log_P_nearest, T_nearest, wl], where the
        'nearest' values are the closest P_fine, T_fine points to the
        actual P, T values in each layer. This results in a large speed gain.
        
        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), terminator
        sector, and wavelength.
    
    '''
    
    # Store length variables for mixing ratio arrays 
    N_species = len(chemical_species)        # Number of chemical species
    N_species_active = len(active_species)   # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)             # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
    N_bf_species = len(bf_species)           # Number of bound-free species included
    
    # Set up to find the bulk species indices (these are always turned on)
    # Find the number and indices for the bulk species 
    N_bulk_species = N_species - N_species_active
    # The bulk species are also the first few indices in the chemical species list
    bulk_species_indices = range(N_bulk_species)
    bulk_species_names = chemical_species[:N_bulk_species]

    # Find the name of the bulk species and check to see if they are 
    # also in the CIA list 
    bulk_cia_indices = []
    for i in range(len(cia_pairs)):
        pair_1, pair_2 = cia_pairs[i].split('-')
        pair_1_bool = False
        pair_2_bool = False 
        for j in bulk_species_names:
            if pair_1 == j:
                pair_1_bool = True
            if pair_2 == j:
                pair_2_bool = True
        
        if pair_1_bool == True and pair_2_bool == True:
            bulk_cia_indices.append(i)

    # Else, we are trying to find the contribution from a species
    if bulk_species == False and cloud_contribution == False:

        # Find the index in the list of chemical species 
        for i in range(len(chemical_species)):
            if contribution_species == chemical_species[i]:
                contribution_molecule_species_index = i

        # Find the index in the list of active species 
        for i in range(len(active_species)):
            if contribution_species == active_species[i]:
                contribution_molecule_active_index = i

        # Find the index in the list of CIA pairs
        cia_indices = []
        for i in range(len(cia_pairs)):
            pair_1, pair_2 = cia_pairs[i].split('-')
            if contribution_species == pair_1 or contribution_species == pair_2:
                cia_indices.append(i)


    N_wl = len(wl)     # Number of wavelengths on model grid
    N_layers = len(P)  # Number of layers
    
    # Define extinction coefficient arrays
    kappa_gas = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    
    # Fine temperature grid (for pre-interpolating opacities)    
    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)
    
    # Find index of deep pressure below which atmosphere is opaque
    i_bot = np.argmin(np.abs(P - P_deep))
    
    # If haze is enabled in this model
    if (enable_haze == 1):
        
        # Generalised scattering slope for haze
        slope = np.power((wl/0.35), gamma)    # Reference wavelength at 0.35 um
        
    # For each terminator sector (terminator plane)
    for j in range(N_sectors):
            
        # For each terminator zone (along day-night transition)
        for k in range(N_zones):
            
            # For each layer, find closest pre-computed cross section to P_fine, T_fine
            for i in range(i_bot,N_layers):
                
                n_level = n[i,j,k]
                
                # Find closest index in fine temperature array to given layer temperature
                idx_T_fine = closest_index(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
                idx_P_fine = closest_index(np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
                
                # For each collisionally-induced absorption (CIA) pair
                for q in range(N_cia_pairs): 

                    # We need to turn off either the bulk species CIA
                    # Or the species we are looking at CIA 
                    # In the layer we are ignoring 
                    
                    # If bulk_species == True, we only turn off the bulk cia species 
                    # for the layer we are ignoring
                    if (bulk_species == True) and (q in bulk_species_indices) and (i == layer_to_ignore):
                        n_n_cia = 0

                    # If its not a bulk species, we still need to turn off the contributing CIA
                    elif (bulk_species == False) and (cloud_contribution == False) and (total_pressure_contribution == False):
                        # We split the if statements because cia_indices is only defined if bulk_species = False
                        if (q in cia_indices) and (i == layer_to_ignore):
                            n_n_cia = 0
                        else:
                            n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                            n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                            n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair

                    # If its the total pressure contribution, we turn everything off no matter what 
                    elif (total_pressure_contribution == True) and (i == layer_to_ignore):
                        n_n_cia = 0

                    # If we are looking at cloud contribution, we keep all the cia on in every layer 
                    else:
                        n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                        n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                        n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_n_cia * cia_stored[q, idx_T_fine, l]
                        
                # For each free-free absorption pair
                for q in range(N_ff_pairs): 
                    
                    n_ff_1 = n_level*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
                    n_ff_2 = n_level*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
                    n_n_ff = n_ff_1*n_ff_2             # Product of number densities of ff pair
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_n_ff * ff_stored[q, idx_T_fine, l]
                        
                # For each source of bound-free absorption (photodissociation)
                for q in range(N_bf_species): 
                    
                    n_q = n_level*X_bf[q,i,j,k]   # Number density of dissociating species
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_q * bf_stored[q,l]
                
                # For each molecular / atomic species with active absorption features
                for q in range(N_species_active): 

                    # If we are not doing bulk species, or cloud species
                    # We just want the molecule and the layer to turn off 
                    
                    # If we are not doing the bulk species or cloud contribution
                    # We just want to turn off one molecule / atom in the layer we are ignoring 

                    if (bulk_species == False) and (cloud_contribution == False) and (total_pressure_contribution == False):
                        # Second if statement since the index is only defined if the above is true 
                        if (q == contribution_molecule_active_index) and (i == layer_to_ignore):
                            n_q = 0
                        else:
                            n_q = n_level*X_active[q,i,j,k] 
                    
                    # if we are doing the total contribution, turn everything off 

                    elif (total_pressure_contribution == True) and (i == layer_to_ignore):
                        n_q = 0

                    else:
                        n_q = n_level*X_active[q,i,j,k] 
                
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i,j,k,l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]
                    
                # For each molecular / atomic species
                for q in range(N_species):  

                    if (bulk_species == False) and (cloud_contribution == False) and (total_pressure_contribution == False):
                        if (q == contribution_molecule_species_index) and (i == layer_to_ignore):
                            n_q = 0
                        else:
                            n_q = n_level*X[q,i,j,k] 
                    
                    # if we are doing the total contribution, turn everything off 
                    elif (total_pressure_contribution == True) and (i == layer_to_ignore):
                        n_q = 0

                    else:
                        n_q = n_level*X[q,i,j,k] 
                    
                    # For each wavelength
                    for l in range(N_wl):
                                
                        # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_Ray[i,j,k,l] += n_q * Rayleigh_stored[q,l]
        

            # CLOUD CONTRIBUTION 
            # If haze is enabled in this model  
            if (enable_haze == 1):
                
                # For each layer
                for i in range(i_bot,N_layers):
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # If we are looking at the cloud contribution in that layer 
                        if (cloud_contribution == True) and (i == layer_to_ignore):
                            haze_amp = 0
                        elif (total_pressure_contribution == True) and (i == layer_to_ignore):
                            haze_amp = 0
                        else:
                            haze_amp = (n[i,j,k] * a * 5.31e-31) 
                    
                        # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                        kappa_cloud[i,j,k,l] += haze_amp * slope[l]
                        
            # If a cloud deck is enabled in this model
            if (enable_deck == 1):
                
                kappa_cloud[(P > P_cloud[0]),j,k,:] += kappa_cloud_0

                # Loop through and find the layer to turn off 
                for i in range(i_bot,N_layers):

                    if (cloud_contribution == True) and (i == layer_to_ignore):
                        kappa_cloud[i,j,k,:] = 0
                    if (total_pressure_contribution == True) and (i == layer_to_ignore):
                        kappa_cloud[i,j,k,:] = 0

            # If a surface is enabled in this model
            if (enable_surface == 1):

                # Set extinction to infinity below surface
                kappa_gas[(P > P_surf),j,k,:] = 1.0e250

            # If Mie clouds are turned on 
            if (enable_Mie == 1):

                # All deck, slab, aerosol information is stored in the n_aerosol_array
                # If its an opaque deck, then the length of sigma_Mie_array will be one more than n_aerosol_array
                # Since the opaque deck is being counted as an extra aerosol 
                # Otherwise, it should be the same 

                # If cloud contribution = True, we need to find the species we want the contribution of 
                if cloud_contribution == True and cloud_total_contribution == False:
                    for n in range(len(aerosol_species)):
                        if cloud_species == aerosol_species[n]:
                            aerosol_species_index = n

                # No opaque clouds 
                if len(n_aerosol_array) == len(sigma_Mie_array):
                    for aerosol in range(len(n_aerosol_array)):
                        for i in range(i_bot,N_layers):
                            for q in range(len(wl)):

                                # If we want one species of cloud contribution 
                                if (cloud_contribution == True) and (cloud_total_contribution == False) and (i == layer_to_ignore):

                                    if aerosol != aerosol_species_index:
                                        kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * sigma_Mie_array[aerosol][q]

                                    else:
                                        kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * 0

                                # If we want the total cloud contribution 
                                elif (cloud_total_contribution == True) and (i == layer_to_ignore):
                                    kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * 0

                                # If we just want total pressure contribution 
                                elif (total_pressure_contribution == True) and (i == layer_to_ignore):
                                    kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * 0

                                # Else, don't turn off clouds
                                else:
                                    kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k] * sigma_Mie_array[aerosol][q]
                        

                # Opaque Deck is the first element in n_aerosol_array
                # I will decide for now to add the opaque deck to each species. I think that won't hurt to keep them together 
                else:
                    for aerosol in range(len(n_aerosol_array)):
                            
                            if (cloud_contribution == True) and (cloud_total_contribution == True):

                                # Turn off the deck first 
                                if aerosol == 0:
                                    kappa_cloud[(P > P_cloud[0]),j,k,:] += kappa_cloud_0
                                    # Loop through and find the layer to turn off 
                                    for i in range(i_bot,N_layers):
                                        if i == layer_to_ignore:
                                            kappa_cloud[i,j,k,:] = 0

                                else:

                                    for i in range(i_bot,N_layers):
                                        for q in range(len(wl)):
                                            if i == layer_to_ignore:
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* 0
                                            else:
                                                 kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* sigma_Mie_array[aerosol-1][q]
                            
                            elif (cloud_contribution == True) and (cloud_total_contribution == False):
                                
                                # Deck doesn't count 
                                if aerosol == 0:
                                    kappa_cloud[(P > P_cloud[0]),j,k,:] += kappa_cloud_0

                                else:
                                    for i in range(i_bot,N_layers):
                                        for q in range(len(wl)):
                                            if (aerosol - 1 == aerosol_species_index) and (i == layer_to_ignore):
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* 0
                                            else:
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* sigma_Mie_array[aerosol-1][q]

                            elif (total_pressure_contribution == True):

                                # Turn off the deck first 
                                if aerosol == 0:
                                    kappa_cloud[(P > P_cloud[0]),j,k,:] += kappa_cloud_0
                                    # Loop through and find the layer to turn off 
                                    for i in range(i_bot,N_layers):
                                        if i == layer_to_ignore:
                                            kappa_cloud[i,j,k,:] -= kappa_cloud_0

                                else:

                                    for i in range(i_bot,N_layers):
                                        for q in range(len(wl)):
                                            if i == layer_to_ignore:
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* 0
                                            else:
                                                 kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* sigma_Mie_array[aerosol-1][q]


                            else:
                                    
                                if aerosol == 0:
                                    kappa_cloud[(P > P_cloud[0]),j,k,:] += 1.0e250

                                else:
                                    for i in range(i_bot,N_layers):
                                        for q in range(len(wl)):
                                                kappa_cloud[i,j,k,q] += n_aerosol_array[aerosol][i,j,k]* sigma_Mie_array[aerosol-1][q]

    return kappa_gas, kappa_Ray, kappa_cloud

def pressure_contribution_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                     spectrum_type = 'transmission', save_spectrum = False,
                     disable_continuum = False, suppress_print = False,
                     Gauss_quad = 2, use_photosphere_radius = True,
                     device = 'cpu', y_p = np.array([0.0]),
                     contribution_species_list = [],
                     bulk_species = True, 
                     cloud_contribution = False, 
                     cloud_species_list = [],
                     cloud_total_contribution = False,
                     put_one_in = True,
                     take_one_out = False,
                     fix_mu = True,
                     scattering_contribution = False,
                     reflection_contribution = False,
                     total_pressure_contribution = False,
                     layer_to_ignore = 0):
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
    scattering = model['scattering']

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
    P_cloud_bottom = atmosphere['P_cloud_bottom']

    # Check if haze enabled in the cloud model
    if ('haze' in model['cloud_type']):
        enable_haze = 1
    else:
        enable_haze = 0

    # Check if a cloud deck is enabled in the cloud model
    # The cloud deck is handled differently for Mie calclations
    if ('deck' in model['cloud_type'] and 'Mie' not in model['cloud_model']):
        enable_deck = 1
    else:
        enable_deck = 0

    if ('Mie' in model['cloud_model']):
        enable_Mie = 1
    else:
        enable_Mie = 0

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
        kappa_gas, kappa_Ray, kappa_cloud = extinction_LBL(chemical_species, active_species, 
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

                wl_Mie = wl_grid_constant_R(wl[0], wl[-1], 1000)

                # If its a fuzzy deck run
                if (model['cloud_type'] == 'fuzzy_deck'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                          r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                          P_cloud = P_cloud,
                                                          log_n_max = log_n_max, 
                                                          fractional_scale_height = fractional_scale_height)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                     r_m, aerosol_species,
                                                     cloud_type = model['cloud_type'],
                                                     aerosol_grid = aerosol_grid,
                                                     P_cloud = P_cloud,
                                                     log_n_max = log_n_max, 
                                                     fractional_scale_height = fractional_scale_height)

                # If its a slab
                elif (model['cloud_type'] == 'slab'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                        r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                        log_X_Mie = log_X_Mie,
                                                        P_cloud = P_cloud,
                                                        P_cloud_bottom = P_cloud_bottom)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                    r_m, aerosol_species,
                                                    cloud_type = model['cloud_type'],
                                                    aerosol_grid = aerosol_grid,
                                                    log_X_Mie = log_X_Mie,
                                                    P_cloud = P_cloud,
                                                    P_cloud_bottom = P_cloud_bottom)
                            
                          
                # If its a uniform X run
                elif (model['cloud_type'] == 'uniform_X'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                          r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                          log_X_Mie = log_X_Mie)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                     r_m, aerosol_species,
                                                     cloud_type = model['cloud_type'],
                                                     aerosol_grid = aerosol_grid,
                                                     log_X_Mie = log_X_Mie)

                # If its a opaque_deck_plus_slab run 
                elif (model['cloud_type'] == 'opaque_deck_plus_slab'):

                    if ((aerosol_species == ['free']) or (aerosol_species == ['file_read'])):
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud_free(P, wl, wl_Mie, r, H, n,
                                                        r_m, r_i_real, r_i_complex, model['cloud_type'],
                                                        log_X_Mie = log_X_Mie,
                                                        P_cloud = P_cloud,
                                                        P_cloud_bottom = P_cloud_bottom)

                    else: 
                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                    r_m, aerosol_species,
                                                    cloud_type = model['cloud_type'],
                                                    aerosol_grid = aerosol_grid,
                                                    log_X_Mie = log_X_Mie,
                                                    P_cloud = P_cloud,
                                                    P_cloud_bottom = P_cloud_bottom)
                        
                # If its a fuzzy_deck_plus_slab run 
                elif (model['cloud_type'] == 'fuzzy_deck_plus_slab'):

                        n_aerosol, sigma_ext_cloud, \
                        g_cloud, w_cloud = Mie_cloud(P, wl, r, H, n,
                                                     r_m, aerosol_species,
                                                     cloud_type = model['cloud_type'],
                                                     aerosol_grid = aerosol_grid,
                                                     P_cloud = P_cloud,
                                                     log_n_max = log_n_max, 
                                                     fractional_scale_height = fractional_scale_height,
                                                     log_X_Mie = log_X_Mie,
                                                     P_cloud_bottom = P_cloud_bottom)

            
            else:

                # Generate empty arrays so the dark god numba is satisfied
                n_aerosol = []
                sigma_ext_cloud = []
                    
                n_aerosol.append(np.zeros_like(r))
                sigma_ext_cloud.append(np.zeros_like(wl))

                n_aerosol = np.array(n_aerosol)
                sigma_ext_cloud = np.array(sigma_ext_cloud)

                w_cloud = np.zeros_like(wl)
                g_cloud = np.zeros_like(wl)

            # Calculate extinction coefficients in standard mode

            # Numba will get mad if P_cloud is not an array (because you can have more than one cloud now)
            # This line just makes sure that P_cloud is an array 
            if isinstance(P_cloud, np.ndarray) == False:
                P_cloud = np.array([P_cloud])

            kappa_gas, kappa_Ray, kappa_cloud = extinction(chemical_species, active_species,
                                                           CIA_pairs, ff_pairs, bf_species,
                                                           n, T, P, wl, X, X_active, X_CIA, 
                                                           X_ff, X_bf, a, gamma, P_cloud, 
                                                           kappa_cloud_0, sigma_stored, 
                                                           CIA_stored, Rayleigh_stored, 
                                                           ff_stored, bf_stored, enable_haze, 
                                                           enable_deck, enable_surface,
                                                           N_sectors, N_zones, T_fine, 
                                                           log_P_fine, P_surf, enable_Mie, 
                                                           n_aerosol, sigma_ext_cloud)
            
            
            # This is to store the contribution kappas
            kappa_gas_contribution_array = []
            kappa_cloud_contribution_array = []
            spectrum_contribution_list_names = []
            
            # If you want to see only the bulk species contribution, this runs first 
            if bulk_species == True:
    
                kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_pressure_contribution(chemical_species, active_species,
                                                                                    CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                                                                    n, T, P, wl, X, X_active, X_CIA, 
                                                                                    X_ff, X_bf, a, gamma, P_cloud, 
                                                                                    kappa_cloud_0, sigma_stored, 
                                                                                    CIA_stored, Rayleigh_stored, 
                                                                                    ff_stored, bf_stored, enable_haze, 
                                                                                    enable_deck, enable_surface,
                                                                                    N_sectors, N_zones, T_fine, 
                                                                                    log_P_fine, P_surf, enable_Mie, 
                                                                                    n_aerosol, sigma_ext_cloud,
                                                                                    bulk_species = True,
                                                                                    layer_to_ignore = layer_to_ignore)

                kappa_gas_contribution_array.append(kappa_gas_temp)
                kappa_cloud_contribution_array.append(kappa_cloud_temp)

                spectrum_contribution_list_names.append('Bulk Species')
                        
            # Then it runs the rest of the molecules that are provided 
            if (contribution_species_list != []):

                for i in range(len(contribution_species_list)):

                    kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_pressure_contribution(chemical_species, active_species,
                                                                                    CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                                                                    n, T, P, wl, X, X_active, X_CIA, 
                                                                                    X_ff, X_bf, a, gamma, P_cloud, 
                                                                                    kappa_cloud_0, sigma_stored, 
                                                                                    CIA_stored, Rayleigh_stored, 
                                                                                    ff_stored, bf_stored, enable_haze, 
                                                                                    enable_deck, enable_surface,
                                                                                    N_sectors, N_zones, T_fine, 
                                                                                    log_P_fine, P_surf, enable_Mie, 
                                                                                    n_aerosol, sigma_ext_cloud,
                                                                                    contribution_species=contribution_species_list[i],
                                                                                    bulk_species = False,
                                                                                    cloud_contribution = False,
                                                                                    layer_to_ignore = layer_to_ignore)

                    kappa_gas_contribution_array.append(kappa_gas_temp)
                    kappa_cloud_contribution_array.append(kappa_cloud_temp)

                    spectrum_contribution_list_names.append(contribution_species_list[i])

            # Cloud contribuiton 
            if cloud_contribution == True:

                # Do the total cloud contribution first 
                # Default for the non Mie clouds 
                if cloud_total_contribution == True or enable_Mie == False:

                    kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_pressure_contribution(chemical_species, active_species,
                                                                CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                                                n, T, P, wl, X, X_active, X_CIA, 
                                                                X_ff, X_bf, a, gamma, P_cloud, 
                                                                kappa_cloud_0, sigma_stored, 
                                                                CIA_stored, Rayleigh_stored, 
                                                                ff_stored, bf_stored, enable_haze, 
                                                                enable_deck, enable_surface,
                                                                N_sectors, N_zones, T_fine, 
                                                                log_P_fine, P_surf, enable_Mie, 
                                                                n_aerosol, sigma_ext_cloud,
                                                                cloud_contribution = True,
                                                                cloud_total_contribution = True,
                                                                layer_to_ignore = layer_to_ignore)

                    kappa_gas_contribution_array.append(kappa_gas_temp)
                    kappa_cloud_contribution_array.append(kappa_cloud_temp)

                    spectrum_contribution_list_names.append('Total Clouds')
                
                # If you have cloud species, run this 
                if (cloud_species_list != []):
                    if enable_Mie == False:
                        raise Exception("Cloud species only available for Mie clouds")
                    
                    for i in range(len(cloud_species_list)):

                        kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_pressure_contribution(chemical_species, active_species,
                                            CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                            n, T, P, wl, X, X_active, X_CIA, 
                                            X_ff, X_bf, a, gamma, P_cloud, 
                                            kappa_cloud_0, sigma_stored, 
                                            CIA_stored, Rayleigh_stored, 
                                            ff_stored, bf_stored, enable_haze, 
                                            enable_deck, enable_surface,
                                            N_sectors, N_zones, T_fine, 
                                            log_P_fine, P_surf, enable_Mie, 
                                            n_aerosol, sigma_ext_cloud,
                                            cloud_contribution = True,
                                            cloud_species = cloud_species_list[i],
                                            layer_to_ignore=layer_to_ignore)

                        kappa_gas_contribution_array.append(kappa_gas_temp)
                        kappa_cloud_contribution_array.append(kappa_cloud_temp)

                        spectrum_contribution_list_names.append(cloud_species_list[i])

            # If you want to run the total pressure contribuiton 
            if total_pressure_contribution == True:
                
                kappa_gas_temp, kappa_Ray_temp, kappa_cloud_temp = extinction_pressure_contribution(chemical_species, active_species,
                                        CIA_pairs, ff_pairs, bf_species, aerosol_species,
                                        n, T, P, wl, X, X_active, X_CIA, 
                                        X_ff, X_bf, a, gamma, P_cloud, 
                                        kappa_cloud_0, sigma_stored, 
                                        CIA_stored, Rayleigh_stored, 
                                        ff_stored, bf_stored, enable_haze, 
                                        enable_deck, enable_surface,
                                        N_sectors, N_zones, T_fine, 
                                        log_P_fine, P_surf, enable_Mie, 
                                        n_aerosol, sigma_ext_cloud,
                                        total_pressure_contribution=True,
                                        layer_to_ignore = layer_to_ignore)

                kappa_gas_contribution_array.append(kappa_gas_temp)
                kappa_cloud_contribution_array.append(kappa_cloud_temp)

                spectrum_contribution_list_names.append('Total Pressure Contribution')



        # Running POSEIDON on the GPU
        elif (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

    # Generate transmission spectrum        
    if (spectrum_type == 'transmission'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")
        
        # Total Spectrum First 
        spectrum = TRIDENT(P, r, r_up, r_low, dr, wl, (kappa_gas + kappa_Ray), 
                            kappa_cloud, enable_deck, enable_haze, b_p, y_p[0],
                            R_s, f_cloud, phi_cloud_0, theta_cloud_0, phi_edge, 
                            theta_edge)

        spectrum_contribution_list = []

        for i in range(len(kappa_gas_contribution_array)):

            kappa_gas_temp = kappa_gas_contribution_array[i]
            kappa_cloud_temp = kappa_cloud_contribution_array[i]
        
            # Call the core TRIDENT routine to compute the transmission spectrum
            spectrum_temp = TRIDENT(P, r, r_up, r_low, dr, wl, (kappa_gas_temp + kappa_Ray), 
                            kappa_cloud_temp, enable_deck, enable_haze, b_p, y_p[0],
                            R_s, f_cloud, phi_cloud_0, theta_cloud_0, phi_edge, 
                            theta_edge)
            
            spectrum_contribution_list.append(spectrum_temp)

    # Generate emission spectrum
    elif ('emission' in spectrum_type):

        # Find zone index for the emission spectrum atmospheric region
        if ('dayside' in spectrum_type):
            zone_idx = 0
        elif ('nightside' in spectrum_type):
            zone_idx = -1
        else:
            zone_idx = 0

        # Use atmospheric properties from dayside/nightside (only consider one region for 1D emission models)
        dz = dr[:,0,zone_idx]
        T = T[:,0,zone_idx]
        
        # First, compute the normal 
        # Compute total extinction (all absorption + scattering sources)
        kappa_tot = (kappa_gas[:,0,zone_idx,:] + kappa_Ray[:,0,zone_idx,:] +
                     kappa_cloud[:,0,zone_idx,:])

        # Store differential extinction optical depth across each layer
        dtau_tot = np.ascontiguousarray(kappa_tot * dz.reshape((len(P), 1)))

        # Without scattering, compute single steam radiative transfer
        if (scattering == False):

            # Compute planet flux (on CPU or GPU)
            if (device == 'cpu'):
                F_p, dtau = emission_single_stream(T, dz, wl, kappa_tot, Gauss_quad)
            elif (device == 'gpu'):
                F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)

        elif (scattering == True):

            # Else, we need to restructure w_cloud and g_cloud to span by layer 
            # For Mie models with 1 species, the g and w can be help constant with each layer since
            # Kappa cloud will encode where clouds are
            # For models that are cloud free, you still need a g and w thats just an array of 0s
            # For Mie models with more than one species, we need to be more careful with the g and w array
            if len(aerosol_species) == 1 or aerosol_species == []:
                w_cloud = np.ones_like(kappa_cloud)*w_cloud
                g_cloud = np.ones_like(kappa_cloud)*g_cloud

            # Need to make a g and w array that vary with pressure layer where aerosols actually are 
            else:
                raise Exception('Only 1 aerosol species supported for scattering')

            # Compute planet flux including scattering (PICASO implementation), see emission.py for details
            F_p, dtau = emission_Toon(P, T, wl, dtau_tot, 
                                        kappa_Ray, kappa_cloud, kappa_tot,
                                        w_cloud, g_cloud, zone_idx,
                                        hard_surface = 0, tridiagonal = 0, 
                                        Gauss_quad = 5, numt = 1)
        
            
            dtau = np.flip(dtau, axis=0)   # Flip optical depth pressure axis back
        

        else:
            raise Exception("Error: Invalid scattering option")

        # Calculate effective photosphere radius at tau = 2/3
        if (use_photosphere_radius == True):    # Flip to start at top of atmosphere
            
            # Running POSEIDON on the CPU
            if (device == 'cpu'):
                R_p_eff = determine_photosphere_radii(np.flip(dtau, axis=0), np.flip(r_low[:,0,zone_idx]),
                                                      wl, photosphere_tau = 2/3)
            
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

        # Then do the contribution functions 
        spectrum_contribution_list = []

        for i in range(len(kappa_gas_contribution_array)):

            kappa_gas_temp = kappa_gas_contribution_array[i]
            kappa_cloud_temp = kappa_cloud_contribution_array[i]

            # Compute total extinction (all absorption + scattering sources)
            kappa_tot = (kappa_gas_temp[:,0,zone_idx,:] + kappa_Ray[:,0,zone_idx,:] +
                        kappa_cloud_temp[:,0,zone_idx,:])

            # Store differential extinction optical depth across each layer
            dtau_tot = np.ascontiguousarray(kappa_tot * dz.reshape((len(P), 1)))

            # Without scattering, compute single steam radiative transfer
            if (scattering == False):

                # Compute planet flux (on CPU or GPU)
                if (device == 'cpu'):
                    F_p, dtau = emission_single_stream(T, dz, wl, kappa_tot, Gauss_quad)
                elif (device == 'gpu'):
                    F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)

            elif (scattering == True):

                # Else, we need to restructure w_cloud and g_cloud to span by layer 
                # For Mie models with 1 species, the g and w can be help constant with each layer since
                # Kappa cloud will encode where clouds are
                # For models that are cloud free, you still need a g and w thats just an array of 0s
                # For Mie models with more than one species, we need to be more careful with the g and w array
                if len(aerosol_species) == 1 or aerosol_species == []:
                    w_cloud = np.ones_like(kappa_cloud)*w_cloud
                    g_cloud = np.ones_like(kappa_cloud)*g_cloud

                # Need to make a g and w array that vary with pressure layer where aerosols actually are 
                else:
                    raise Exception('Only 1 aerosol species supported for scattering')

                # Compute planet flux including scattering (PICASO implementation), see emission.py for details
                F_p, dtau = emission_Toon(P, T, wl, dtau_tot, 
                                            kappa_Ray, kappa_cloud, kappa_tot,
                                            w_cloud, g_cloud, zone_idx,
                                            hard_surface = 0, tridiagonal = 0, 
                                            Gauss_quad = 5, numt = 1)
            
                
                dtau = np.flip(dtau, axis=0)   # Flip optical depth pressure axis back

            else:
                raise Exception("Error: Invalid scattering option")

            # Calculate effective photosphere radius at tau = 2/3
            if (use_photosphere_radius == True):    # Flip to start at top of atmosphere
                
                # Running POSEIDON on the CPU
                if (device == 'cpu'):
                    R_p_eff = determine_photosphere_radii(np.flip(dtau, axis=0), np.flip(r_low[:,0,zone_idx]),
                                                        wl, photosphere_tau = 2/3)
                
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
                spectrum_temp = F_p_obs

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
                spectrum_temp = F_p_obs / F_s_obs

            spectrum_contribution_list.append(spectrum_temp)
  
    return spectrum, spectrum_contribution_list_names, spectrum_contribution_list

def pressure_contribution(planet, star, model, atmosphere, opac, wl,
                            spectrum_type = 'transmission', save_spectrum = False,
                            disable_continuum = False, suppress_print = False,
                            Gauss_quad = 2, use_photosphere_radius = True,
                            device = 'cpu', y_p = np.array([0.0]),
                            contribution_species_list = [],
                            bulk_species = True, 
                            cloud_contribution = False, 
                            cloud_species_list = [],
                            cloud_total_contribution = False,
                            put_one_in = True,
                            take_one_out = False,
                            fix_mu = True,
                            scattering_contribution = False,
                            reflection_contribution = False,
                            total_pressure_contribution = False):


    '''
    Computes the pressure contribution function 
    
    Returns:
        Contribution (np.array)
            Array. [i,j,k] i = molecule number (or total if total = True), j = Pressure layer, k = Wavelength 
        Norm (np.array)
            Array [i,j] where i = molecule number and j = wavelength. If user wants to normalize them   
    '''
    
    # Warning message
    if len(wl) > 10000:
        print('Given current resolution (R), this will take more than a few hours to run. We reccomend to lower the resolution to 1000.')

    P = atmosphere['P']

    # Find how many elements you need 
    contribution_length = 0 

    # Need to figure out if cloud contribution for non aerosols counts here or not(it doesnt)
    if bulk_species == True:
        contribution_length += 1
    # This only matters of aerosol models with multiple cloud species 
    if model['cloud_type'] == 'Mie':
        if cloud_total_contribution == True:
            contribution_length += 1
    else:
        if cloud_contribution == True:
            contribution_length += 1
        
    if total_pressure_contribution == True:
        contribution_length += 1
    
    contribution_length += len(contribution_species_list)
    contribution_length += len(cloud_species_list)

    # Define arrays where pressure contribution functions will live 
    Contribution = np.zeros(shape=(contribution_length,len(P), len(wl)))

    # For denominator of contribution function 
    norm = np.zeros(shape=(contribution_length,len(wl)))   # Running sum for contribution

    for i in range(len(P)):

        spectrum, spectrum_contribution_list_names, spectrum_contribution_list = pressure_contribution_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                                                                                                        spectrum_type = spectrum_type, save_spectrum = save_spectrum,
                                                                                                                        disable_continuum = disable_continuum, suppress_print = suppress_print,
                                                                                                                        Gauss_quad = Gauss_quad, use_photosphere_radius = use_photosphere_radius,
                                                                                                                        device = device, y_p = y_p,
                                                                                                                        contribution_species_list = contribution_species_list,
                                                                                                                        bulk_species = bulk_species, 
                                                                                                                        cloud_contribution = cloud_contribution, 
                                                                                                                        cloud_species_list = cloud_species_list,
                                                                                                                        cloud_total_contribution = cloud_total_contribution,
                                                                                                                        put_one_in = put_one_in,
                                                                                                                        take_one_out = take_one_out,
                                                                                                                        fix_mu = fix_mu,
                                                                                                                        scattering_contribution = scattering_contribution,
                                                                                                                        reflection_contribution = reflection_contribution,
                                                                                                                        total_pressure_contribution = total_pressure_contribution,
                                                                                                                        layer_to_ignore = i)
        
                                                                                                            
        for j in range(len(spectrum_contribution_list)):
        
            # Find the difference between spectrums
            diff = spectrum - spectrum_contribution_list[j]

            # Add to contribution function (not yet normalized)
            Contribution[j,i,:] = np.abs(diff)

            # Increment normalization factor 
            norm[j,:] += diff
    

    return Contribution, norm, spectrum_contribution_list_names

def plot_pressure_contribution(wl,P,
                               Contribution,
                               spectrum_contribution_list_names, 
                               R = 100):

    # Plots out the pressure contribution functions. Only displays them, doesn't save them.
    
    for i in range(len(spectrum_contribution_list_names)):

            fig, ax = plt.subplots(figsize=(10, 10))

            a = ax.contourf(wl, np.log10(P), Contribution[i,:,:],cmap='plasma')

            ax.set_ylabel('Log Pressure (bar)')
            ax.invert_yaxis()
            ax.set_xlabel('Wavelength ($\mu$m)')

            title = 'Contribution Function : ' + str(spectrum_contribution_list_names[i])
            
            ax.set_title(title)
            plt.colorbar(a, label='Contribution')
            plt.show()

            # Trying Ryan's Binning 

            fig = plt.figure()  
            fig.set_size_inches(14, 7)
            ax = plt.gca()

            ax.set_yscale("log")

            # Bin the wavelengths using the first pressure layer of the spectrum 
            # This is because bin_spectrum returns both a wl binned and spectrum grid and we want the wl binned for now 
            wl_binned, _ , _ = bin_spectrum(wl, Contribution[i,0,:], R)

            # Now to create the contribution function but binned 
            Contribution_binned = np.zeros(shape=(len(P), len(wl_binned)))

            # Now loop over all pressure layers 
            for j in range(len(P)):
                    _, Contribution_binned[j,:], _ = bin_spectrum(wl, Contribution[i,j,:], R)

            X_bin, Y_bin = np.meshgrid(wl_binned, P)
            
            # Plot binned contribution function
            contour_plot = plt.contourf(X_bin, Y_bin, Contribution_binned[:,:], 100, cmap=cmr.swamp_r)
            #contour_plot = plt.contourf(wl_binned, P, Contribution_binned[:,:], 100, cmap=cmr.swamp_r)

            ax.invert_yaxis()    

            ax.set_xlim([wl[0], wl[-1]])
            ax.set_ylim([P[0], P[-1]])        
            
            ax.set_ylabel(r'P (bar)', fontsize = 15, labelpad=0.5)
            ax.set_xlabel(r'Wavelength ' + r'(μm)', fontsize = 15)
            ax.set_title(title)

            plt.colorbar()
            plt.show()

def photometric_contribution_function(wl, P, Contribution, 
                                      spectrum_contribution_list_names,
                                      ):

    wl_min = np.min(wl)
    wl_max = np.max(wl)

    # Bin Stuff from minimum wavelength to maximum wavelength by 0.1 
    bins = np.arange(wl_min,wl_max+0.1,0.1)

    for b in range(len(bins)):
        bins[b] = round(bins[b],1)

    # Make it so the last bin includes the max wavelength (if not it will be a seperate bin)
    bins[-1] += 0.1
    bin_indices = np.digitize(wl, bins)
    bins[-1] -= 0.1

    bincount = np.bincount(bin_indices)

    # Finds the indices to loop over in the wavelength ranges
    indices_for_loop = []
    for n in range(len(bincount)):
        if n == 0:
            indices_for_loop.append(n)
        else:
            indices_for_loop.append(np.sum(bincount[0:n+1])-1)


    # Now to find photometric contribution 

    # [molecule][photometric conitrbution for each bin]
    photometric_contribution = []

    # Loop over each molecule
    for i in range(len(spectrum_contribution_list_names)):

        median_array_one_molecule = []
        # Loop over each wavelength range 
        for j in range(len(indices_for_loop)-1):
            # Loop over each pressure range to get the median 
            temp_row = []
            for p in range(len(P)):

                temp_row.append(np.nanmedian(Contribution[i,p,indices_for_loop[j]:indices_for_loop[j+1]]))

            median_array_one_molecule.append(temp_row)

        photometric_contribution.append(median_array_one_molecule)

    # Finding the total photometric contribution for each molecule by adding everything up    
    photometric_total = []
    for i in range(len(photometric_contribution)):
        temp_row = np.zeros(len(photometric_contribution[i][0]))
        for j in range(len(photometric_contribution[i])):
            temp_row += photometric_contribution[i][j]
            
        # Now to normalize the area to 1 
        temp_row = temp_row / np.trapz(temp_row)
        photometric_total.append(temp_row)


    return photometric_contribution, photometric_total

def plot_photometric_contribution(wl,P,
                                  photometric_contribution, photometric_total,
                                  spectrum_contribution_list_names):

    # Loop over each molecule

    labels = []
    for i in spectrum_contribution_list_names:
        labels.append(i)
    labels.append('Total')

    for i in range(len(spectrum_contribution_list_names)):

        fig, ax = plt.subplots(figsize=(10, 10))

        for b in range(len(photometric_contribution[i])):
            ax.plot(photometric_contribution[i][b],np.log10(P))

        ax.set_ylabel('Log Pressure (bar)')
        ax.invert_yaxis()
        ax.set_xlabel('Contribution')
        title = 'Photometric Contribution Function : ' + str(labels[i])
        ax.set_title(title)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylabel('Log Pressure (bar)')
        ax.invert_yaxis()
        ax.set_xlabel('Contribution')
        title = 'Photometric Contribution Function All Wavelength : ' + str(labels[i])
        ax.set_title(title)
        ax.plot(photometric_total[i],np.log10(P))
        plt.show()

    # Plots all of them together, and the log version as well 
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylabel('Log Pressure (bar)')
    ax.invert_yaxis()
    ax.set_xlabel('Contribution')
    title = 'Photometric Contribution Function All Wavelength All Molecules:'
    ax.set_title(title)
    for i in range(len(spectrum_contribution_list_names)):
        ax.plot(photometric_total[i],np.log10(P), label = labels[i])
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylabel('Log Pressure (bar)')
    ax.invert_yaxis()
    ax.set_xlabel('Log Contribution')
    title = 'Photometric Contribution Function All Wavelength All Molecules:'
    ax.set_title(title)
    for i in range(len(spectrum_contribution_list_names)):
        ax.plot(np.log10(photometric_total[i]),np.log10(P), label = labels[i])
    ax.legend()
    plt.show()