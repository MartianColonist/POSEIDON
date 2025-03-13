'''
Functions related to atmospheric retrieval.

'''

import numpy as np
import time
import os
import pymultinest
from mpi4py import MPI
from scipy.special import ndtri
from numba.core.decorators import jit
from scipy.special import erfcinv
from scipy.special import lambertw as W
from scipy.constants import parsec

from .constants import R_J, R_E, M_J, M_E

from .parameters import split_params
from .instrument import bin_spectrum_to_data
from .utility import write_MultiNest_results, round_sig_figs, closest_index, \
                     write_retrieved_spectrum, write_retrieved_PT, \
                     write_retrieved_log_X, confidence_intervals
from .core import make_atmosphere, compute_spectrum
from .parameters import unpack_stellar_params
from .stellar import precompute_stellar_spectra, stellar_contamination_general
from .high_res import loglikelihood_high_res
from .chemistry import load_chemistry_grid
from .transmission import area_overlap_circles

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create global variable needed for centred log-ratio prior function
allowed_simplex = 1


def run_retrieval(planet, star, model, opac, data, priors, wl, P, 
                  P_ref = None, R_p_ref = None, P_param_set = 1.0e-2, 
                  R = None, retrieval_name = None, He_fraction = 0.17, 
                  N_slice_EM = 2, N_slice_DN = 4, constant_gravity = False,
                  spectrum_type = 'transmission', y_p = np.array([0.0]),
                  stellar_T_step = 20, stellar_log_g_step = 0.1, 
                  N_live = 400, ev_tol = 0.5, sampling_algorithm = 'MultiNest', 
                  resume = False, verbose = True, sampling_target = 'parameter',
                  chem_grid = 'fastchem', N_output_samples = 1000,
                  ):
    '''
    ADD DOCSTRING (will hopefully be done before the heat death of the Universe)
    '''

    # Unpack planet name
    planet_name = planet['planet_name']
    
    # Unpack prior types and ranges
    prior_types = priors['prior_types']
    prior_ranges = priors['prior_ranges']

    # Unpack model properties
    model_name = model['model_name']
    chemical_species = model['chemical_species']
    param_species = model['param_species']
    param_names = model['param_names']
    stellar_contam = model['stellar_contam']
    reference_parameter = model['reference_parameter']
    disable_atmosphere = model['disable_atmosphere']
    X_profile = model['X_profile']
    high_res_method = model['high_res_method']

    # Unpack stellar properties
    if (star is not None):
        R_s = star['R_s']
        stellar_interp_backend = star['stellar_interp_backend']

    # Check that one of the two reference parameters has been provided by the user
    if ((reference_parameter == 'R_p_ref') and (P_ref is None)):
        raise Exception("Error: Must provide P_ref when R_p_ref is a free parameter.")
    if ((reference_parameter == 'P_ref') and (R_p_ref is None)):
        raise Exception("Error: Must provide R_p_ref when P_ref is a free parameter.")
    
    if ((disable_atmosphere == True) and ('transmission' not in spectrum_type)):
        raise Exception("Error: An atmosphere can only be disabled for transmission spectra.")

    N_params = len(param_names)

    if retrieval_name is None:
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'

    # Load chemistry grid (e.g. equilibrium chemistry) if option selected
    if X_profile == "chem_eq":
        chemistry_grid = load_chemistry_grid(param_species, chem_grid, comm, rank)
    else:
        chemistry_grid = None

    # Pre-compute stellar spectra for models with unocculted spots / faculae
    if (stellar_contam != None):

        if (rank == 0):
            print("Pre-computing stellar spectra before starting retrieval...")

        # Interpolate and store stellar photosphere and heterogeneity spectra
        T_phot_grid, T_het_grid, \
        log_g_phot_grid, log_g_het_grid, \
        I_phot_grid, I_het_grid = precompute_stellar_spectra(comm, wl, star, prior_types, 
                                                             prior_ranges, stellar_contam,
                                                             stellar_T_step, stellar_log_g_step,
                                                             stellar_interp_backend)

    # No stellar grid precomputation needed for models with uniform star
    else:

        T_phot_grid, T_het_grid = None, None
        log_g_phot_grid, log_g_het_grid = None, None
        I_phot_grid, I_het_grid = None, None

    # Interpolate stellar spectrum onto planet wavelength grid (one-time operation)
    if (('transmission' not in spectrum_type) and (star != None)):

        # Load stellar spectrum
        F_s = star['F_star']
        d = planet['system_distance']

        # Distance only used for flux ratios, so set it to 1 since it cancels
        if (d is None):
            planet['system_distance'] = 1
            d = planet['system_distance']

        # Convert stellar surface flux to observed flux at Earth
        F_s_obs = (R_s / d)**2 * F_s

    # Skip for directly imaged planets or brown dwarfs
    else:

        # Stellar flux not needed for transmission spectra
        F_s_obs = None

    if rank == 0:
        print("POSEIDON now running '" + retrieval_name + "'")

    # Run POSEIDON retrieval using PyMultiNest
    if (sampling_algorithm == 'MultiNest'):

        # Change directory into MultiNest result file folder
        os.chdir(output_dir + 'MultiNest_raw/')

        # Set basename for MultiNest output files
        basename = retrieval_name + '-'

        # Begin retrieval timer
        if (rank == 0):
            t0 = time.perf_counter()

        # Run MultiNest
        PyMultiNest_retrieval(planet, star, model, opac, data, prior_types, 
                               prior_ranges, spectrum_type, wl, P, P_ref,
                               R_p_ref, P_param_set, He_fraction, N_slice_EM, 
                               N_slice_DN, N_params, T_phot_grid, T_het_grid, 
                               log_g_phot_grid, log_g_het_grid, I_phot_grid, 
                               I_het_grid, y_p, F_s_obs, constant_gravity,
                               chemistry_grid, resume = resume, verbose = verbose,
                               outputfiles_basename = basename, 
                               n_live_points = N_live, multimodal = False,
                               evidence_tolerance = ev_tol, log_zero = -1e90,
                               importance_nested_sampling = False, 
                               sampling_efficiency = sampling_target, 
                               const_efficiency_mode = False)

        # Write retrieval results to file
        if (rank == 0):

            # Write retrieval runtime to terminal
            t1 = time.perf_counter()
            total = round_sig_figs((t1-t0)/3600.0, 2)  # Round to 2 significant figures
            
            print('POSEIDON retrieval finished in ' + str(total) + ' hours')

            # Compute samples of retrieved P-T, mixing ratio profiles, and spectrum
            T_low2, T_low1, T_median, \
            T_high1, T_high2, \
            log_X_low2, log_X_low1, \
            log_X_median, log_X_high1, \
            log_X_high2, \
            spec_low2, spec_low1, \
            spec_median, spec_high1, \
            spec_high2, T_best, \
            spectrum_best, ymodel_best, \
            ymodel_samples = retrieved_samples(planet, star, model, opac, data,
                                               retrieval_name, wl, P, P_ref, R_p_ref,
                                               P_param_set, He_fraction, N_slice_EM, 
                                               N_slice_DN, spectrum_type, T_phot_grid, 
                                               T_het_grid, log_g_phot_grid,
                                               log_g_het_grid, I_phot_grid, 
                                               I_het_grid, y_p, F_s_obs,
                                               constant_gravity, chemistry_grid,
                                               N_output_samples)
            
            # Write POSEIDON retrieval output files 
            write_MultiNest_results(planet, model, data, retrieval_name,
                                    N_live, ev_tol, sampling_algorithm, wl, R,
                                    ymodel_best, spectrum_type)
                        
            # Save sampled spectrum
            write_retrieved_spectrum(retrieval_name, wl, spec_low2, 
                                     spec_low1, spec_median, spec_high1, spec_high2)
            
            # Save ymodel samples
            ymodel_samples_object = np.array(ymodel_samples).T
            np.savetxt('../samples/' + retrieval_name + '_ymodel_samples.txt', ymodel_samples_object.T)
            
            # Only write retrieved P-T profile and mixing ratio arrays if atmosphere enabled
            if (disable_atmosphere == False):

                # Save sampled P-T profile
                write_retrieved_PT(retrieval_name, P, T_low2, T_low1, 
                                   T_median, T_high1, T_high2)

                # Save sampled mixing ratio profiles
                write_retrieved_log_X(retrieval_name, chemical_species, P, 
                                      log_X_low2, log_X_low1, log_X_median, 
                                      log_X_high1, log_X_high2)

            print("All done! Output files can be found in " + output_dir + "results/")

    comm.Barrier()

    # Change directory back to directory where user's python script is located
    os.chdir('../../../../')


def forward_model(param_vector, planet, star, model, opac, data, wl, P, P_ref_set,
                  R_p_ref_set, P_param_set, He_fraction, N_slice_EM, N_slice_DN, 
                  spectrum_type, T_phot_grid, T_het_grid, log_g_phot_grid,
                  log_g_het_grid, I_phot_grid, I_het_grid, y_p, F_s_obs,
                  constant_gravity, chemistry_grid):
    '''
    ADD DOCSTRING
    '''

    # Unpack number of free parameters
    param_names = model['param_names']
    physical_param_names = model['physical_param_names']

    # Unpack model properties
    radius_unit = model['radius_unit']
    mass_unit = model['mass_unit']
    distance_unit = model['distance_unit']
    N_params_cum = model['N_params_cum']
    surface = model['surface']
    stellar_contam = model['stellar_contam']
    nightside_contam = model['nightside_contam']
    disable_atmosphere = model['disable_atmosphere']
    PT_penalty = model['PT_penalty']
    high_res_method = model['high_res_method']

    # Unpack planet and star properties
    R_p = planet['planet_radius']
    b_p = planet['planet_impact_parameter']

    if (star is not None):
        R_s = star['R_s']

    # For a retrieval we do not have user provided P-T or chemical profiles
    T_input = []
    X_input = []

    #***** Step 1: unpack parameter values from prior sample *****#
    
    physical_params, PT_params, \
    log_X_params, cloud_params, \
    geometry_params, stellar_params, \
    offset_params, err_inflation_params, \
    high_res_params = split_params(param_vector, N_params_cum)

    # If the atmosphere is disabled
    if ((disable_atmosphere == True) and ('transmission' in spectrum_type)):

        R_p_ref = physical_params[np.where(physical_param_names == 'R_p_ref')[0][0]]

        # Convert normalised radius drawn by MultiNest back into SI
        if (radius_unit == 'R_J'):
            R_p_ref *= R_J
        elif (radius_unit == 'R_E'):
            R_p_ref *= R_E

        # Calculate distance between planet and star centres
        d = np.sqrt((b_p**2 + np.min(y_p)**2))

        # Check if planet fully overlaps the star
        if (d <= (R_s - R_p_ref)):
            spectrum = (R_p_ref / R_s)**2 * np.ones_like(wl)

        # Partial overlap case
        elif (d > (R_s - R_p_ref)) and (d < (R_s + R_p_ref)):
            spectrum = area_overlap_circles(d, R_p_ref, R_s) / (np.pi * (R_s**2)) * np.ones_like(wl)

        # No overlap
        else:
            spectrum = np.zeros_like(wl)

        # No atmosphere dictionary needed if atmosphere disabled
        atmosphere = None

        ln_prior_TP = 0   # Not needed for flat lines, so set to zero

    else:

        # Unpack reference pressure if set as a free parameter
        if ('log_P_ref' in physical_param_names):
            P_ref = np.power(10.0, physical_params[np.where(physical_param_names == 'log_P_ref')[0][0]])
        else:
            P_ref = P_ref_set

        # Unpack reference radius if set as a free parameter
        if ('R_p_ref' in physical_param_names):
            R_p_ref = physical_params[np.where(physical_param_names == 'R_p_ref')[0][0]]

            # Convert normalised radius drawn by MultiNest back into SI
            if (radius_unit == 'R_J'):
                R_p_ref *= R_J
            elif (radius_unit == 'R_E'):
                R_p_ref *= R_E

        else:
            R_p_ref = R_p_ref_set

        # Unpack planet mass if set as a free parameter
        if ('M_p' in physical_param_names):
            M_p = physical_params[np.where(physical_param_names == 'M_p')[0][0]]

            # Convert normalised mass drawn by MultiNest back into SI
            if (mass_unit == 'M_J'):
                M_p *= M_J
            elif (mass_unit == 'M_E'):
                M_p *= M_E

        else:
            M_p = None

        # Unpack log(gravity) if set as a free parameter
        if ('log_g' in physical_param_names):
            log_g = physical_params[np.where(physical_param_names == 'log_g')[0][0]]
        else:
            log_g = None

        # Unpack system distance if set as a free parameter
        if ('d' in physical_param_names):
            d_sampled = physical_params[np.where(physical_param_names == 'd')[0][0]]

            # Convert distance drawn by MultiNest (in parsec) back into SI
            if (distance_unit == 'pc'):
                d_sampled *= parsec

            # Redefine object distance to sampled value 
            planet['system_distance'] = d_sampled

        else:
            d_sampled = planet['system_distance']

        # Unpack surface pressure if set as a free parameter
        if (surface == True):
            P_surf = np.power(10.0, physical_params[np.where(physical_param_names == 'log_P_surf')[0][0]])
        else:
            P_surf = None

        # Unpack background gas molecular mass if set as a free parameter
        if ('mu_back' in physical_param_names):
            mu_back = physical_params[np.where(physical_param_names == 'mu_back')[0][0]]
        else:
            mu_back = None

        #***** Step 2: generate atmosphere corresponding to parameter draw *****#

        atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, 
                                     log_X_params, cloud_params, geometry_params, 
                                     log_g, M_p, T_input, X_input, P_surf, P_param_set,
                                     He_fraction, N_slice_EM, N_slice_DN, 
                                     constant_gravity, chemistry_grid, mu_back)
        
        # If PT_penalty is true, then you compute the PT penalty
        # Only for Pelletier 2021 profiles
        if PT_penalty == True:

            # Unpack P_top and P_bottom 
            log_P_min = np.min(np.log10(P))
            log_P_max = np.max(np.log10(P))

            # Unpack the number of knots, T points, and sigma smooth
            num_of_knots = len(PT_params) - 1
            T_points = np.array(PT_params[:-1])
            sigma_s = PT_params[-1]

            # Delta log p goes in denominator of the sum
            # re-step here just returns the spacing 
            delta_log_p = np.linspace(log_P_min, log_P_max, num=num_of_knots, retstep=True)[1]

            # Sum in Equation 11, with the addition of a 1/2 ln (2 pi sigma_smooth**2) from Line et al 2011
            sum = np.sum(((T_points[2:] - 2*T_points[1:-1] + T_points[:-2])**2)/(delta_log_p**3) ) - 0.5 * np.log(2*np.pi*sigma_s**2)
            
            # Prefix remains the same from Equation 11
            ln_prior_TP = (-1.0/(2.0*sigma_s**2)) * (1/(log_P_max-log_P_min)) * sum

        else:
            ln_prior_TP = 0

        #***** Step 3: generate spectrum of atmosphere ****#

        # For emission spectra retrievals we directly compute Fp (instead of Fp/F*)
        # so we can convolve and bin Fp and F* separately when comparing to data
        if (('emission' in spectrum_type) and (spectrum_type != 'direct_emission')):
            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                        spectrum_type = ('direct_' + spectrum_type))   # Always Fp (even for secondary eclipse)

        # For transmission spectra
        else:
            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                        spectrum_type, y_p = y_p)

        # Reject unphysical spectra (forced to be NaN by function above)
        if (np.any(np.isnan(spectrum))):
            
            # Quit if given parameter combination is unphysical
            return 0, spectrum, atmosphere, ln_prior_TP

    # For high-resolution retrievals, we skip the rest of the forward model,
    # since only the spectrum and atmosphere are needed
    if (high_res_method is not None):
        return 0, spectrum, atmosphere, ln_prior_TP  # No y_model needed for high-res retrievals here

    #***** Step 4: stellar contamination *****#
    
    # Stellar contamination is only relevant for transmission spectra
    if ('transmission' in spectrum_type):

        if (stellar_contam != None):

            if ('one_spot' in stellar_contam):

                # Unpack stellar contamination parameters
                f_het, _, _, T_het, _, \
                _, T_phot, log_g_het, \
                _, _, log_g_phot = unpack_stellar_params(param_names, star, 
                                                         stellar_params, 
                                                         stellar_contam, 
                                                         N_params_cum)
                
                # Find stellar intensities at closest T and log g
                I_het = I_het_grid[closest_index(T_het, T_het_grid[0], 
                                                 T_het_grid[-1], len(T_het_grid)),
                                   closest_index(log_g_het, log_g_het_grid[0], 
                                                 log_g_het_grid[-1], len(log_g_het_grid)),
                                   :]
                I_phot = I_phot_grid[closest_index(T_phot, T_phot_grid[0], 
                                                   T_phot_grid[-1], len(T_phot_grid)),
                                     closest_index(log_g_phot, log_g_phot_grid[0], 
                                                   log_g_phot_grid[-1], len(log_g_phot_grid)),
                                     :]
                
                # Package heterogeneity properties for general contamination formula
                f_het = np.array([f_het])
                I_het = np.array([I_het])
                
            elif ('two_spots' in stellar_contam):

                # Unpack stellar contamination parameters
                _, f_spot, f_fac, _, \
                T_spot, T_fac, T_phot, \
                _, log_g_spot, log_g_fac, \
                log_g_phot = unpack_stellar_params(param_names, star, 
                                                   stellar_params, 
                                                   stellar_contam, 
                                                   N_params_cum)
                
                # Find stellar intensities at closest T and log g
                I_spot = I_het_grid[closest_index(T_spot, T_het_grid[0], 
                                                  T_het_grid[-1], len(T_het_grid)),
                                    closest_index(log_g_spot, log_g_het_grid[0], 
                                                  log_g_het_grid[-1], len(log_g_het_grid)),
                                    :]
                I_fac = I_het_grid[closest_index(T_fac, T_het_grid[0], 
                                                 T_het_grid[-1], len(T_het_grid)),
                                   closest_index(log_g_fac, log_g_het_grid[0], 
                                                 log_g_het_grid[-1], len(log_g_het_grid)),
                                   :]
                I_phot = I_phot_grid[closest_index(T_phot, T_phot_grid[0], 
                                                   T_phot_grid[-1], len(T_phot_grid)),
                                     closest_index(log_g_phot, log_g_phot_grid[0], 
                                                   log_g_phot_grid[-1], len(log_g_phot_grid)),
                                     :]
                
                # Package spot and faculae properties for general contamination formula
                f_het = np.array([f_spot, f_fac])
                I_het = np.vstack((I_spot, I_fac))
            
            # Compute wavelength-dependant stellar contamination factor
            epsilon = stellar_contamination_general(f_het, I_het, I_phot)

            # Apply multiplicative stellar contamination to spectrum
            spectrum = epsilon * spectrum

    #***** Step 5: nightside contamination (credit to John Kappelmeier) *****#
    
    # Nightside contamination is only relevant for transmission spectra
    if ('transmission' in spectrum_type):

        if (nightside_contam == True):

            # Calculate nightside thermal emission spectrum
            Fp_Fs_night = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                           spectrum_type = 'nightside_emission')
            
            # Compute wavelength-dependent nightside contamination factor
            psi = (1.0 / (1.0 + Fp_Fs_night))

            # Apply multiplicative nightside contamination to spectrum
            spectrum = psi * spectrum
            
    #***** Step 6: convolve spectrum with instrument PSF and bin to data resolution ****#

    if ('transmission' in spectrum_type):
        ymodel = bin_spectrum_to_data(spectrum, wl, data)
    else:
        F_p_binned = bin_spectrum_to_data(spectrum, wl, data)
        if ('direct' in spectrum_type):
            ymodel = F_p_binned
        else:
            F_s_binned = bin_spectrum_to_data(F_s_obs, wl, data)
            ymodel = F_p_binned/F_s_binned

    return ymodel, spectrum, atmosphere, ln_prior_TP


@jit(nopython = True)
def CLR_Prior(chem_params_drawn, limit = -12.0):
    
    ''' Implements the centred-log-ratio (CLR) prior for chemical mixing ratios.
    
        CLR[i] here is the centred log-ratio transform of the mixing ratio, X[i]
       
    '''
    
    n = len(chem_params_drawn)     # Number of species free parameters

    # Limits correspond to condition that all X_i > 10^(-12)
    prior_lower_CLR = ((n-1.0)/n) * (limit * np.log(10.0) + np.log(n-1.0))      # Lower limit corresponds to species under-abundant
    prior_upper_CLR = ((1.0-n)/n) * (limit * np.log(10.0))                      # Upper limit corresponds to species dominant

    CLR = np.zeros(shape=(n+1))   # Vector of CLR variables
    X = np.zeros(shape=(n+1))     # Vector of mixing ratio parameters
    
    # Evaluate centred log-ratio parameters by uniformly sampling between limits
    for i in range(n):
 
        CLR[1+i] = ((chem_params_drawn[i] * (prior_upper_CLR - prior_lower_CLR)) + prior_lower_CLR) 
          
    if (np.abs(np.sum(CLR[1:n])) <= prior_upper_CLR):   # Impose same prior on X_0
            
        CLR[0] = -1.0*np.sum(CLR[1:n])   # CLR variables must sum to 0, so that X_i sum to 1
        
        if ((np.max(CLR) - np.min(CLR)) <= (-1.0 * limit * np.log(10.0))):      # Necessary for all X_i > 10^(-12)    
        
            normalisation = np.sum(np.exp(CLR))
        
            for i in range(n+1):
                
                # Map log-ratio parameters to mixing ratios
                X[i] = np.exp(CLR[i]) / normalisation   # Vector of mixing ratios (should sum to 1!)
                
                # One final check that all X_i > 10^(-12)
                if (X[i] < 1.0e-12): 
                    return (np.ones(n+1)*(-50.0))    # Fails check -> return dummy array of log values
            
            return np.log10(X)   # Return vector of log-mixing ratios
        
        elif ((np.max(CLR) - np.min(CLR)) > (-1.0 * limit * np.log(10.0))):
        
            return (np.ones(n+1)*(-50.0))   # Fails check -> return dummy array of log values
    
    elif (np.abs(np.sum(CLR[1:n])) > prior_upper_CLR):   # If falls outside of allowed triangular subspace
        
        return (np.ones(n+1)*(-50.0))    # Fails check -> return dummy array of log values


def PyMultiNest_retrieval(planet, star, model, opac, data, prior_types, 
                          prior_ranges, spectrum_type, wl, P, P_ref_set, 
                          R_p_ref_set, P_param_set, He_fraction, N_slice_EM, 
                          N_slice_DN, N_params, T_phot_grid, T_het_grid, 
                          log_g_phot_grid, log_g_het_grid, I_phot_grid, 
                          I_het_grid, y_p, F_s_obs, constant_gravity,
                          chemistry_grid, **kwargs):
    ''' 
    Main function for conducting atmospheric retrievals with PyMultiNest.
    
    '''

    # Unpack model properties
    param_names = model['param_names']
    param_species = model['param_species']
    X_params = model['X_param_names']
    N_params_cum = model['N_params_cum']
    Atmosphere_dimension = model['Atmosphere_dimension']
    species_EM_gradient = model['species_EM_gradient']
    species_DN_gradient = model['species_DN_gradient']
    error_inflation = model['error_inflation']
    offsets_applied = model['offsets_applied']
    stellar_contam = model['stellar_contam']
    PT_penalty = model['PT_penalty']
    high_res_method = model['high_res_method']
    high_res_param_names = model['high_res_param_names']
    
   # R_p = planet["planet_radius"]
   # d = planet["system_distance"]
   # R_s = star["R_s"]

    # Unpack number of free mixing ratio parameters for prior function  
    N_species_params = len(X_params)

    # Assign PyMultiNest keyword arguments
    n_dims = N_params

    # Pre-compute normalisation for log-likelihood
    if (high_res_method is None):   # Not needed for high-res retrievals
        err_data = data["err_data"]
        norm_log_default = (-0.5 * np.log(2.0 * np.pi * err_data * err_data)).sum()

    # Create variable governing if a mixing ratio parameter combination lies in 
    # the allowed CLR simplex space (X_i > 10^-12 and sum to 1)
    global allowed_simplex    # Needs to be global, as prior function has no return

    allowed_simplex = 1    # Only changes to 0 for CLR variables outside prior

    # Define the prior transformation function
    def Prior(cube, ndim, nparams):
        ''' 
        Transforms the unit cube provided by MultiNest into the values of 
        each free parameter used by the forward model.
        
        '''

        # Assign prior distribution to each free parameter
        for i, parameter in enumerate(param_names):

            # First deal with all parameters besides mixing ratios 
            if (parameter not in X_params) or (parameter in ['C_to_O', 'log_Met']):

                # Uniform priors
                if (prior_types[parameter] == 'uniform'):

                    min_value = prior_ranges[parameter][0]
                    max_value = prior_ranges[parameter][1]

                    cube[i] = ((cube[i] * (max_value - min_value)) + min_value)

                # Gaussian priors
                elif (prior_types[parameter] == 'gaussian'):

                    mean = prior_ranges[parameter][0]
                    std = prior_ranges[parameter][1]

                    cube[i] = mean + (std * ndtri(cube[i]))

                # Sine priors
                elif (prior_types[parameter] == 'sine'):

                    max_value = prior_ranges[parameter][1]

                    if parameter in ['alpha', 'beta']:
                        cube[i] = (180.0/np.pi)*2.0*np.arcsin(cube[i] * np.sin((np.pi/180.0)*(max_value/2.0)))

                    elif parameter in ['theta_0']:
                        cube[i] = (180.0/np.pi)*np.arcsin((2.0*cube[i] - 1) * np.sin((np.pi/180.0)*(max_value/2.0)))

            # Draw mixing ratio parameters with uniform priors
            elif ((parameter in X_params) and (prior_types[parameter] == 'uniform')):

                # Find which chemical species this parameter represents
                for species_q in param_species:
                    phrase = '_' + species_q
                    if ((phrase + '_' in parameter) or (parameter[-len(phrase):] == phrase)):
                        species = species_q

                # For 1D models, prior just given by mixing ratio prior range
                if (Atmosphere_dimension == 1):

                    min_value = prior_ranges[parameter][0]
                    max_value = prior_ranges[parameter][1]

                    cube[i] = ((cube[i] * (max_value - min_value)) + min_value)

                # For 2D models, the prior range for 'Delta' parameters can change to satisfy mixing ratio priors
                elif (Atmosphere_dimension == 2):

                    # Absolute mixing ratio parameter comes first
                    if ('Delta' not in parameter):

                        min_value = prior_ranges[parameter][0]
                        max_value = prior_ranges[parameter][1]

                        last_value = ((cube[i] * (max_value - min_value)) + min_value)

                        cube[i] = last_value

                        # Store name of previous parameter for delta prior
                        prev_parameter = parameter
                
                    # Mixing ratio gradient parameter comes second
                    elif ('Delta' in parameter):

                        # Mixing ratio gradient parameters dynamically update allowed range
                        min_prior_abs = prior_ranges[prev_parameter][0]
                        max_prior_abs = prior_ranges[prev_parameter][1]

                        min_prior_delta = prior_ranges[parameter][0]
                        max_prior_delta = prior_ranges[parameter][1]

                        # Load chosen abundance from previous parameter
                        sampled_abundance = last_value

                        # Find largest gradient such that the abundances in all
                        # atmospheric regions satisfy the absolute abundance constraint
                        largest_delta = 2*min((sampled_abundance - min_prior_abs), 
                                              (max_prior_abs - sampled_abundance))   # This is |Delta|_max

                        # Max / min values governed by the most restrictive of
                        # delta prior or absolute prior, such that both are satisfied
                        max_value_delta = min(max_prior_delta, largest_delta)
                        min_value_delta = max(min_prior_delta, -largest_delta)
                        
                        cube[i] = ((cube[i] * (max_value_delta - min_value_delta)) + min_value_delta)
                    
                # For 3D models, the prior ranges for 'Delta' parameters can change to satisfy mixing ratio priors
                elif (Atmosphere_dimension == 3):
                        
                    # For species with 3D gradients, sample such that highest and lowest values still satisfy mixing ratio prior
                    if ((species in species_EM_gradient) and (species in species_DN_gradient)):

                        # Absolute mixing ratio parameter comes first
                        if ('Delta' not in parameter):

                            min_value = prior_ranges[parameter][0]
                            max_value = prior_ranges[parameter][1]

                            cube[i] = ((cube[i] * (max_value - min_value)) + min_value)

                            # Store name of previous parameter for next delta prior
                            prev_parameter = parameter
                        
                        # Terminator mixing ratio gradient parameter comes second
                        elif (parameter == ('Delta_log_' + species + '_term')):

                            # Mixing ratio gradient parameters dynamically update allowed range
                            min_prior_abs = prior_ranges[prev_parameter][0]
                            max_prior_abs = prior_ranges[prev_parameter][1]

                            min_prior_delta = prior_ranges[parameter][0]
                            max_prior_delta = prior_ranges[parameter][1]

                            # Load chosen abundance from previous parameter
                            sampled_abundance = cube[i-1]

                            # Find largest gradient such that the abundances in all
                            # atmospheric regions satisfy the absolute abundance constraint
                            largest_delta = 2*min((sampled_abundance - min_prior_abs), 
                                                  (max_prior_abs - sampled_abundance))   # This is |Delta|_max

                            # Max / min values governed by the most restrictive of
                            # delta prior or absolute prior, such that both are satisfied
                            max_value_delta = min(max_prior_delta, largest_delta)
                            min_value_delta = max(min_prior_delta, -largest_delta)

                            cube[i] = ((cube[i] * (max_value_delta - min_value_delta)) + min_value_delta)

                            # Store name of previous parameters for next delta prior
                            prev_prev_parameter = prev_parameter
                            prev_parameter = parameter

                        # Day-night mixing ratio gradient parameter comes third
                        elif (parameter == ('Delta_log_' + species + '_DN')):

                            # Mixing ratio gradient parameters dynamically update allowed range
                            min_prior_abs = prior_ranges[prev_prev_parameter][0]
                            max_prior_abs = prior_ranges[prev_prev_parameter][1]

                            min_prior_delta_DN = prior_ranges[parameter][0]
                            max_prior_delta_DN = prior_ranges[parameter][1]

                            # Find minimum and maximum mixing ratio in terminator plane (i.e. evening/morning)
                            max_term_abundance = cube[i-2] + abs(cube[i-1]/2.0)  # log_X_term_bar + |delta_log_X_term|/2
                            min_term_abundance = cube[i-2] - abs(cube[i-1]/2.0)  # log_X_term_bar - |delta_log_X_term|/2

                            # Find largest gradient such that the abundances in all
                            # atmospheric regions satisfy the absolute abundance constraint
                            largest_delta = 2*min((min_term_abundance - min_prior_abs), 
                                                    (max_prior_abs - max_term_abundance))   # This is |Delta|_max

                            # Max / min values governed by the most restrictive of
                            # delta priors or absolute prior, such that both are satisfied
                            max_value_delta_DN = min(max_prior_delta_DN, largest_delta)
                            min_value_delta_DN = max(min_prior_delta_DN, -largest_delta)

                            cube[i] = ((cube[i] * (max_value_delta_DN - min_value_delta_DN)) + min_value_delta_DN)

                    # Species with a 2D gradient (or no gradient) within a 3D model reduces to the 2D case above
                    else:

                        # Absolute mixing ratio parameter comes first
                        if ('Delta' not in parameter):

                            min_value = prior_ranges[parameter][0]
                            max_value = prior_ranges[parameter][1]

                            last_value = ((cube[i] * (max_value - min_value)) + min_value)

                            cube[i] = last_value

                            # Store name of previous parameter for delta prior
                            prev_parameter = parameter
                    
                        # Mixing ratio gradient parameter comes second
                        elif ('Delta' in parameter):

                            # Mixing ratio gradient parameters dynamically update allowed range
                            min_prior_abs = prior_ranges[prev_parameter][0]
                            max_prior_abs = prior_ranges[prev_parameter][1]

                            min_prior_delta = prior_ranges[parameter][0]
                            max_prior_delta = prior_ranges[parameter][1]

                            # Load chosen abundance from previous parameter
                            sampled_abundance = last_value

                            # Find largest gradient such that the abundances in all
                            # atmospheric regions satisfy the absolute abundance constraint
                            largest_delta = 2*min((sampled_abundance - min_prior_abs), 
                                                  (max_prior_abs - sampled_abundance))   # This is |Delta|_max

                            # Max / min values governed by the most restrictive of
                            # delta prior or absolute prior, such that both are satisfied
                            max_value_delta = min(max_prior_delta, largest_delta)
                            min_value_delta = max(min_prior_delta, -largest_delta)
                            
                            cube[i] = ((cube[i] * (max_value_delta - min_value_delta)) + min_value_delta)
                    
        # If mixing ratio parameters have centred-log ratio prior, treat separately 
        if ('CLR' in prior_types.values()):

            # Random numbers from 0 to 1 corresponding to mixing ratio parameters
            chem_drawn = np.array(cube[N_params_cum[1]:N_params_cum[2]])

            # Load Lower limit on log mixing ratios specified by user
            limit = prior_ranges[X_params[0]][0]   # Same for all CLR variables, so choose first one

            # Map random numbers to CLR variables, than transform to mixing ratios
            log_X = CLR_Prior(chem_drawn, limit)
            
            # Check if this random parameter draw lies in the allowed simplex space (X_i > 10^-12 and sum to 1)
            global allowed_simplex     # Needs a global, as prior function has no return

            if (log_X[1] == -50.0): 
                allowed_simplex = 0       # Mixing ratios outside allowed simplex space -> model rejected by likelihood
            elif (log_X[1] != -50.0): 
                allowed_simplex = 1       # Likelihood will be computed for this parameter combination
                
            # Pass the mixing ratios corresponding to the sampled CLR variables to MultiNest
            for i in range(N_species_params):
                
                i_prime = N_params_cum[1] + i
                
                cube[i_prime] = log_X[(1+i)]   # log_X[0] is not a free parameter


    # Define the log-likelihood function
    def LogLikelihood(cube, ndim, nparams):
        ''' 
        Evaluates the log-likelihood for a given point in parameter space.
        
        Works by generating a PT profile, calculating the opacity in the
        model atmosphere, computing the resulting spectrum and finally 
        convolving and integrating the spectrum to produce model data 
        points for each instrument.
        
        The log-likelihood is then evaluated using the difference between 
        the binned spectrum and the actual data points. 
        
        '''
        
        #***** Check for non-allowed parameter values *****#

        # Immediately reject samples falling outside of mixing ratio simplex (CLR prior only)
        global allowed_simplex
        if (allowed_simplex == 0):
            loglikelihood = -1.0e100   
            return loglikelihood

        # Unpack stellar parameters
        _, _, _, _, _, \
        stellar_params, _, _, _ = split_params(cube, N_params_cum)

        # Reject models with spots hotter than faculae (by definition)
        if ((stellar_contam != None) and ('two_spots' in stellar_contam)):

            # Unpack stellar contamination parameters
            _, _, _, _, \
            T_spot, T_fac, T_phot, \
            _, _, _, _ = unpack_stellar_params(param_names, star, stellar_params, 
                                               stellar_contam, N_params_cum)
                            
            if ((T_spot > T_phot) or (T_fac < T_phot) or (T_spot > T_fac)):
                loglikelihood = -1.0e100   
                return loglikelihood

        #***** For valid parameter combinations, run forward model *****#

        ymodel, spectrum, _, ln_prior_TP = forward_model(cube, planet, star, model, opac, data, 
                                                        wl, P, P_ref_set, R_p_ref_set, P_param_set, 
                                                        He_fraction, N_slice_EM, N_slice_DN, 
                                                        spectrum_type, T_phot_grid, T_het_grid, 
                                                        log_g_phot_grid, log_g_het_grid,
                                                        I_phot_grid, I_het_grid, y_p, F_s_obs,
                                                        constant_gravity, chemistry_grid)

        # Reject unphysical spectra (forced to be NaN by function above)
        if (np.any(np.isnan(spectrum))):
            
            # Assign penalty to likelihood => point ignored in retrieval
            loglikelihood = -1.0e100
            
            # Quit if given parameter combination is unphysical
            return loglikelihood

        if (high_res_method is not None):

            # Unpack high-resolution retrieval parameters
            _, _, _, _, _, \
            _, _, _, high_res_params = split_params(cube, N_params_cum)

            # Compute the log-likelihood relevant for high-resolution retrievals
            loglikelihood = loglikelihood_high_res(wl, spectrum, F_s_obs, data,
                                                   spectrum_type, high_res_method,
                                                   high_res_params, high_res_param_names)
            
            return loglikelihood

        #***** Handle error bar inflation and offsets (if optionally enabled) *****#

        # Unpack offset and error inflation parameters
        _, _, _, _, _, _, \
        offset_params, err_inflation_params, _ = split_params(cube, N_params_cum)
        
        # Load error bars specified in data files
        err_data = data['err_data']
        
        # Compute effective error, if unknown systematics included
        if (error_inflation == 'Line15'):
            err_eff_sq = (err_data*err_data + np.power(10.0, err_inflation_params[0]))
            norm_log = (-0.5*np.log(2.0*np.pi*err_eff_sq)).sum()
        elif (error_inflation == 'Piette20'):
            err_eff_sq = (err_data*err_data + (err_inflation_params[0]*ymodel)**2)
            norm_log = (-0.5*np.log(2.0*np.pi*err_eff_sq)).sum()
        else: 
            err_eff_sq = err_data*err_data
            norm_log = norm_log_default

        # Load transit depth data points and indices of any offset ranges
      #  ydata = data['ydata']
      #  offset_start = data['offset_start']
      #  offset_end = data['offset_end']

        # Apply relative offset between datasets
      #  if (offsets_applied == 'single_dataset'):
      #      ydata_adjusted = ydata.copy()
      #      ydata_adjusted[offset_start:offset_end] -= offset_params[0]*1e-6  # Convert from ppm to transit depth
      #  else: 
      #      ydata_adjusted = ydata

        # Load transit depth data points and indices of any offset ranges
        ydata = data['ydata']
        offset_start = data['offset_start']
        offset_end = data['offset_end']

        offset_1_start = data['offset_1_start']
        offset_1_end = data['offset_1_end']
        offset_2_start = data['offset_2_start']
        offset_2_end = data['offset_2_end']
        offset_3_start = data['offset_3_start']
        offset_3_end = data['offset_3_end']

        # Apply relative offset between datasets
        if (offsets_applied == 'single_dataset'):

            ydata_adjusted = ydata.copy()

            # One offset for one dataset
            if offset_1_start == 0:
                ydata_adjusted[offset_start:offset_end] -= offset_params[0]*1e-6  # Convert from ppm to transit depth
            
            # Else, you have multiple datasets lumped together with a single offset
            else:
                for n in range(len(offset_1_start)):
                    ydata_adjusted[offset_1_start[n]:offset_1_end[n]] -= offset_params[0]*1e-6 

        elif (offsets_applied == 'two_datasets'):

            ydata_adjusted = ydata.copy()

            # Two offsets for two datasets
            if offset_1_start == 0:
                ydata_adjusted[offset_start[0]:offset_end[0]] -= offset_params[0]*1e-6
                ydata_adjusted[offset_start[1]:offset_end[1]] -= offset_params[1]*1e-6
            
            # Else, you have multiple datasets lumped together in both or either offset
            else:
                for n in range(len(offset_1_start)):
                    ydata_adjusted[offset_1_start[n]:offset_1_end[n]] -= offset_params[0]*1e-6 
                for m in range(len(offset_2_start)):
                    ydata_adjusted[offset_2_start[m]:offset_2_end[m]] -= offset_params[1]*1e-6 

        elif (offsets_applied == 'three_datasets'):

            ydata_adjusted = ydata.copy()

            # Three offsets for three datasets
            if offset_1_start == 0:
                ydata_adjusted[offset_start[0]:offset_end[0]] -= offset_params[0]*1e-6
                ydata_adjusted[offset_start[1]:offset_end[1]] -= offset_params[1]*1e-6
                ydata_adjusted[offset_start[2]:offset_end[2]] -= offset_params[2]*1e-6

            # Else, you have multiple datasets lumped together in both or either offset
            else:
                for n in range(len(offset_1_start)):
                    ydata_adjusted[offset_1_start[n]:offset_1_end[n]] -= offset_params[0]*1e-6 
                for m in range(len(offset_2_start)):
                    ydata_adjusted[offset_2_start[m]:offset_2_end[m]] -= offset_params[1]*1e-6 
                for s in range(len(offset_3_start)):
                    ydata_adjusted[offset_3_start[s]:offset_3_end[s]] -= offset_params[2]*1e-6 
            
        else: 
            ydata_adjusted = ydata

        #***** Calculate ln(likelihood) ****#
    
        loglikelihood = (-0.5*((ymodel - ydata_adjusted)**2)/err_eff_sq).sum()
        loglikelihood += norm_log

        # Add the PT penalty 
        loglikelihood += ln_prior_TP
                    
        return loglikelihood
    
    # Run PyMultiNest
    pymultinest.run(LogLikelihood, Prior, n_dims, **kwargs)


def retrieved_samples(planet, star, model, opac, data, retrieval_name, wl, P, 
                      P_ref_set, R_p_ref_set, P_param_set, He_fraction, 
                      N_slice_EM, N_slice_DN, spectrum_type, T_phot_grid, 
                      T_het_grid, log_g_phot_grid, log_g_het_grid, I_phot_grid, 
                      I_het_grid, y_p, F_s_obs, constant_gravity, 
                      chemistry_grid, N_output_samples):
    '''
    ADD DOCSTRING
    '''

    # Load relevant output directory
    output_prefix = retrieval_name + '-'

    # Unpack number of free parameters
    param_names = model['param_names']
    n_params = len(param_names)

    # Unpack model properties
    disable_atmosphere = model['disable_atmosphere']
    
    # Run PyMultiNest analyser to extract posterior samples
    analyzer = pymultinest.Analyzer(n_params, outputfiles_basename = output_prefix,
                                    verbose = False)
    samples = analyzer.get_equal_weighted_posterior()[:,:-1]

    # Store best-fitting set of parameters
    best_fit = analyzer.get_best_fit()
    best_fit_params = best_fit['parameters']

    # Find total number of available posterior samples from MultiNest 
    N_samples_total = len(samples[:,0])
    
    # Randomly draw parameter samples from posterior
    N_sample_draws = min(N_samples_total, N_output_samples)
    sample = np.random.choice(len(samples), N_sample_draws, replace=False)

    print("Now generating " + str(N_sample_draws) + " sampled spectra and " + 
          "P-T profiles from the posterior distribution...")
    
    # Calculate best-fitting spectrum and PT profile
    ymodel_best, spectrum_best, \
    atmosphere_best, _ = forward_model(best_fit_params, planet, star, model, opac, data, 
                                       wl, P, P_ref_set, R_p_ref_set, P_param_set, 
                                       He_fraction, N_slice_EM, N_slice_DN, 
                                       spectrum_type, T_phot_grid, T_het_grid, 
                                       log_g_phot_grid, log_g_het_grid,
                                       I_phot_grid, I_het_grid, y_p, F_s_obs,
                                       constant_gravity, chemistry_grid)
    
    # Store temperature field and mixing ratios for the best model
    if (disable_atmosphere == False):

        T_best = atmosphere_best['T']
        log_X_best = np.log10(atmosphere_best['X'])

    else:
        T_best = 0.0
                    
    # For all the samples, generate spectra and PT profiles
    for i in range(N_sample_draws):

        # Estimate run time for this function based on one model evaluation
        if (i == 0):
            t0 = time.perf_counter()   # Time how long one model takes

        param_vector = samples[sample[i],:]

        ymodel, spectrum, \
        atmosphere, _ = forward_model(param_vector, planet, star, model, opac, data, 
                                      wl, P, P_ref_set, R_p_ref_set, P_param_set, 
                                      He_fraction, N_slice_EM, N_slice_DN, 
                                      spectrum_type, T_phot_grid, T_het_grid, 
                                      log_g_phot_grid, log_g_het_grid,
                                      I_phot_grid, I_het_grid, y_p, F_s_obs,
                                      constant_gravity, chemistry_grid)

        # Based on first model, create arrays to store retrieved temperature, spectrum, and mixing ratios
        if (i == 0):

            # Estimate run time for this function based on one model evaluation
            t1 = time.perf_counter()
            total = round_sig_figs((N_sample_draws * (t1-t0)/60.0), 2)  # Round to 2 significant figures
            
            print('This process will take approximately ' + str(total) + ' minutes')

            # Only store T and log X if an atmosphere enabled
            if (disable_atmosphere == False):

                # Find size of mixing ratio field (same as temperature field)
                N_species, N_D, N_sectors, N_zones = np.shape(atmosphere['X'])

                # Create arrays to store sampled retrieval outputs
                T_stored = np.zeros(shape=(N_sample_draws, N_D, N_sectors, N_zones))
                log_X_stored = np.zeros(shape=(N_sample_draws, N_species, N_D, N_sectors, N_zones))

            spectrum_stored = np.zeros(shape=(N_sample_draws, len(wl)))

            if model['high_res_method'] is None:
                ymodel_samples = np.zeros(shape=(N_sample_draws, len(ymodel)))

        if (disable_atmosphere == False):

            # Store temperature field and mixing ratios in sample arrays
            T_stored[i,:,:,:] = atmosphere['T']
            log_X_stored[i,:,:,:,:] = np.log10(atmosphere['X'])

        # Store spectrum in sample array
        spectrum_stored[i,:] = spectrum

        if model['high_res_method'] is None:
            ymodel_samples[i,:] = ymodel
            
    # Compute 1 and 2 sigma confidence intervals for P-T and mixing ratio profiles and spectrum
        
    # P-T profile
    if (disable_atmosphere == False):
        _, T_low2, T_low1, T_median, \
        T_high1, T_high2, _ = confidence_intervals(N_sample_draws, 
                                                T_stored[:,:,0,0], N_D)
    else:
        T_low2, T_low1, T_median, T_high1, T_high2 = None, None, None, None, None

    # Mixing ratio profiles
    if (disable_atmosphere == False):

        log_X_low2 = np.zeros(shape=(N_species, N_D))
        log_X_low1 = np.zeros(shape=(N_species, N_D))
        log_X_median = np.zeros(shape=(N_species, N_D))
        log_X_high1 = np.zeros(shape=(N_species, N_D))
        log_X_high2 = np.zeros(shape=(N_species, N_D))

        for q in range(N_species):

            _, log_X_low2[q,:], log_X_low1[q,:], \
            log_X_median[q,:], log_X_high1[q,:], \
            log_X_high2[q,:], _ = confidence_intervals(N_sample_draws, 
                                                    log_X_stored[:,q,:,0,0], N_D)
            
    else:

        log_X_low2, log_X_low1, log_X_median, \
        log_X_high1, log_X_high2 = None, None, None, None, None
    
    # Spectrum
    _, spec_low2, spec_low1, spec_median, \
    spec_high1, spec_high2, _ = confidence_intervals(N_sample_draws, 
                                                     spectrum_stored, len(wl))
    
    return T_low2, T_low1, T_median, T_high1, T_high2, \
           log_X_low2, log_X_low1, log_X_median, log_X_high1, log_X_high2, \
           spec_low2, spec_low1, spec_median, spec_high1, spec_high2, \
           T_best, spectrum_best, ymodel_best, ymodel_samples


def get_retrieved_atmosphere(planet, model, P, P_ref_set = 10, R_p_ref_set = None, 
                             median = False, best_fit = True,
                             P_param_set = 1.0e-2, He_fraction = 0.17,
                             N_slice_EM = 2, N_slice_DN = 4, 
                             constant_gravity = False, chemistry_grid = None,
                             specific_param_values = [],
                             verbose = False):
    '''
    Creates the atmosphere dictionary for the median or best fit spectrum of a retrieval.

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
        median (optional, bool)
            Option to create an atmosphere from the median retrieved spectrum.
        best_fit (optional, bool)
            Option to create an atmosphere from the best fit retrieved spectrum
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
        specific_param_values (list):
            If a specific parameter combination is provided, this will be used
            instead of the median or best fit parameters.

    Returns:
        atmosphere (dict):
                Collection of atmospheric properties required to compute the
                resultant spectrum of the planet.
    '''

    # unpack planet
    planet_name = planet['planet_name']

    # unpack model
    param_names = model['param_names']
    N_params_cum = model['N_params_cum']
    physical_param_names = model['physical_param_names']

    radius_unit = model['radius_unit']
    mass_unit = model['mass_unit']
    surface = model['surface']

    # no user defined T_input and X_input in retrievals
    T_input = []
    X_input = []

    # Use specific parameter combination if provided
    if (len(specific_param_values) != 0):
        param_values = specific_param_values

    # Or load the median or best-fitting parameters from the MultiNest output
    else:

        # Identify output directory location
        output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'

        # Load relevant output directory
        output_prefix = model['model_name'] + '-'

        # Change directory into MultiNest result file folder
        os.chdir(output_dir + 'MultiNest_raw/')

        # Run PyMultiNest analyser to extract posterior samples
        analyzer = pymultinest.Analyzer(len(param_names), outputfiles_basename = output_prefix,
                                        verbose = False)
        
        # get parameter values for either the median or best fit model
        if median == True and best_fit == False:

            # get stats 
            stats = analyzer.get_stats()

            # create empty list to store parameter values
            param_values = []
            
            # loop over all parameters in model
            for i in range(len(param_names)):
                param_values.append(stats['marginals'][i]['median'])

        
        elif median == False and best_fit == True:

            # get best fit parameter values
            best_fit = analyzer.get_best_fit()
            param_values = best_fit['parameters']


        # catch cases where median and best fit are both true or both false
        elif (median == False and best_fit == False) or (median == True and best_fit == True):
            # Change directory back to directory where user's python script is located
            os.chdir('../../../../')
            raise Exception("Please specify either median or best fit model as True")
        

        # Change directory back to directory where user's python script is located
        os.chdir('../../../../')


    # split parameters into each atmosphere category
    physical_params, PT_params, log_X_params, \
    cloud_params, geometry_params, _, _, _, _ = split_params(param_values, N_params_cum)
    
    # Unpack reference pressure if set as a free parameter
    if ('log_P_ref' in physical_param_names):
        P_ref = np.power(10.0, physical_params[np.where(physical_param_names == 'log_P_ref')[0][0]])
    else:
        P_ref = P_ref_set

    # Unpack reference radius if set as a free parameter
    if ('R_p_ref' in physical_param_names):
        R_p_ref = physical_params[np.where(physical_param_names == 'R_p_ref')[0][0]]

        # Convert normalised radius drawn by MultiNest back into SI
        if (radius_unit == 'R_J'):
            R_p_ref *= R_J
        elif (radius_unit == 'R_E'):
            R_p_ref *= R_E
    else:
        R_p_ref = R_p_ref_set

    # Unpack planet mass if set as a free parameter
    if ('M_p' in physical_param_names):
        M_p = physical_params[np.where(physical_param_names == 'M_p')[0][0]]

        # Convert normalised mass drawn by MultiNest back into SI
        if (mass_unit == 'M_J'):
            M_p *= M_J
        elif (mass_unit == 'M_E'):
            M_p *= M_E

    else:
        M_p = None

    # Unpack log(gravity) if set as a free parameter
    if ('log_g' in physical_param_names):
        log_g = physical_params[np.where(physical_param_names == 'log_g')[0][0]]
    else:
        log_g = None

    # Unpack surface pressure if set as a free parameter
    if ((surface == True) and ('log_P_surf' in physical_param_names)):
        P_surf = np.power(10.0, physical_params[np.where(physical_param_names == 'log_P_surf')[0][0]])
    else:
        P_surf = None

    # Unpack background gas molecular mass if set as a free parameter
    if ('mu_back' in physical_param_names):
        mu_back = physical_params[np.where(physical_param_names == 'mu_back')[0][0]]
    else:
        mu_back = None

    if verbose == True:
        print('R_p_ref = ', physical_params[np.where(physical_param_names == 'R_p_ref')[0][0]], '* ', radius_unit)
        print('PT_params = np.array(', PT_params,')')
        print('log_X_params = np.array(', log_X_params,')')
        print('cloud_params = np.array(', cloud_params,')')
        print('geometry_params = np.array(', geometry_params,')')
    
    # make atmosphere 
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, 
                                 log_X_params, cloud_params, geometry_params,
                                 log_g = log_g, M_p = M_p, T_input = T_input,
                                 X_input = X_input, P_surf = P_surf,
                                 P_param_set = P_param_set, He_fraction = He_fraction, 
                                 N_slice_EM = N_slice_EM, N_slice_DN = N_slice_DN, 
                                 constant_gravity = constant_gravity,
                                 chemistry_grid = chemistry_grid,
                                 mu_back = mu_back)
    
    return atmosphere


#***** Compute Bayes factors, sigma significance etc *****#

def Z_to_sigma(ln_Z1, ln_Z2):
    ''' 
    Convert the log-evidences of two models to a sigma confidence level.
    
    '''
    
    np.set_printoptions(precision=50)

    B = np.exp(ln_Z1 - ln_Z2)                        # Bayes factor
    p = np.real(np.exp(W((-1.0/(B*np.exp(1))),-1)))  # p-value

    sigma = np.sqrt(2)*erfcinv(p)    # Equivalent sigma
    
    #print "p-value = ", p
    #print "n_sigma = ", sigma
    
    return B, sigma   


def Bayesian_model_comparison(planet_name, model_1, model_2,
                              ln_Z_format = '{:.2f}', B_format = '{:.2e}',
                              ln_B_format = '{:.2f}', sigma_format = '{:.1f}'):
    '''    
    Conduct Bayesian model comparison between the outputs of two retrievals. 
    This function outputs the Bayes factor and equivalent sigma significance 
    comparing the two models.
        
    '''

    # Unpack model properties
    model_1_name = model_1['model_name']
    model_2_name = model_2['model_name']
    n_params_1 = len(model_1['param_names'])
    n_params_2 = len(model_2['param_names'])

    # Access directory containing raw MultiNest files
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/MultiNest_raw/'

    # Change directory into MultiNest result file folder
    os.chdir(output_dir)

    # Load relevant output directory
    model_1_prefix = model_1_name + '-'
    model_2_prefix = model_2_name + '-'
    
    # Run PyMultiNest analyser
    analyzer_1 = pymultinest.Analyzer(n_params_1, outputfiles_basename = model_1_prefix,
                                      verbose = False)
    analyzer_2 = pymultinest.Analyzer(n_params_2, outputfiles_basename = model_2_prefix,
                                      verbose = False)  

    # Extract Bayesian evidence of each model
    stats_1 = analyzer_1.get_stats()
    stats_2 = analyzer_2.get_stats()
    
    ln_Z_1 = stats_1['nested sampling global log-evidence']
    ln_Z_2 = stats_2['nested sampling global log-evidence']
    ln_Z_1_err = stats_1['nested sampling global log-evidence error']
    ln_Z_2_err = stats_2['nested sampling global log-evidence error']

    # Compute Bayes factor and equivalent sigma significance
    Bayes_factor, n_sigma = Z_to_sigma(ln_Z_1, ln_Z_2)

    print("Bayesian evidences:\n")
    print("Model " + model_1_name + ": ln Z = " + ln_Z_format.format(ln_Z_1) +
          " +/- " + ln_Z_format.format(ln_Z_1_err))
    print("Model " + model_2_name + ": ln Z = " + ln_Z_format.format(ln_Z_2) +
          " +/- " + ln_Z_format.format(ln_Z_2_err) + "\n")

    print("Bayes factor:\n")
    print("B = " + B_format.format(Bayes_factor))
    print("ln B = " + ln_B_format.format(np.log(Bayes_factor)) + "\n")

    if (Bayes_factor > 1):
        print("'Equivalent' detection significance:\n")
        print(sigma_format.format(n_sigma) + " σ\n")
    else:
        print("No detection of the reference model, model " + model_2_name + 
              " is preferred.\n")

    # Change directory back to directory where user's python script is located
    os.chdir('../../../../')
    
    return