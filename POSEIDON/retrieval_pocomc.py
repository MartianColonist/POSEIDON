
# %%
# Functions related to atmospheric retrieval

import numpy as np
import time
import os
from scipy.special import ndtri
from spectres import spectres
from numba.core.decorators import jit
from scipy.special import erfcinv
from scipy.special import lambertw as W
from scipy.constants import parsec
import matplotlib.pyplot as plt

from POSEIDON.constants import R_J, R_E

from POSEIDON.parameters import split_params
from POSEIDON.instrument import bin_spectrum_to_data
from POSEIDON.utility import write_MultiNest_results, round_sig_figs, closest_index, \
                     write_retrieved_spectrum, write_retrieved_PT, \
                     write_retrieved_log_X, confidence_intervals
from POSEIDON.core import make_atmosphere, compute_spectrum
from POSEIDON.stellar import precompute_stellar_spectra, stellar_contamination_single_spot
import pocomc as pc
from POSEIDON.high_res import log_likelihood, sysrem, fast_filter, fit_uncertainties

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# Create global variable needed for centred log-ratio prior function
allowed_simplex = 1


def run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref = 10.0, 
                  P_param_set = 1.0e-2, R = None, retrieval_name = None,
                  He_fraction = 0.17, N_slice_EM = 2, N_slice_DN = 4, 
                  spectrum_type = 'transmission', y_p = np.array([0.0]),
                  N_live = 400, ev_tol = 0.5,
                  sampling_algorithm = 'pocoMC', resume = False, 
                  verbose = True, sampling_target = 'parameter',
                  N_output_samples = 1000):

    # Unpack planet name
    planet_name = planet['planet_name']
    
    # Unpack prior types and ranges
    prior_types = priors['prior_types']
    prior_ranges = priors['prior_ranges']

    # Unpack model properties
    model_name = model['model_name']
    chemical_species = model['chemical_species']
    param_names = model['param_names']
    N_params = len(param_names)

    if (retrieval_name is None):
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'

    # Pre-compute stellar spectra for models with unocculted spots / faculae
    if (model['stellar_contam'] != 'No'):

        # if (rank == 0):
        print("Pre-computing stellar spectra before starting retrieval...")

        # Interpolate and store stellar photosphere and heterogeneity spectra
        T_phot_grid, T_het_grid, \
        I_phot_grid, I_het_grid = precompute_stellar_spectra(wl, star, prior_types, 
                                                             prior_ranges,
                                                             T_step_interp = 10)

    # No stellar grid precomputation needed for models with uniform star
    else:

        T_phot_grid, T_het_grid = None, None
        I_phot_grid, I_het_grid = None, None

    # if (rank == 0):
    print("POSEIDON now running '" + retrieval_name + "'")

    # Run POSEIDON retrieval using PyMultiNest
    if (sampling_algorithm == 'pocoMC'):
        os.makedirs(output_dir + 'pocoMC_raw/', exist_ok=True)
        # Change directory into MultiNest result file folder
        os.chdir(output_dir + 'pocoMC_raw/')

        # Set basename for MultiNest output files
        basename = retrieval_name + '-'

        # Begin retrieval timer
        # if (rank == 0):
        t0 = time.perf_counter()

        # Run MultiNest
        pocoMC_retrieval(planet, star, model, opac, data, prior_types, 
                    prior_ranges, spectrum_type, wl, P, P_ref, P_param_set, 
                    He_fraction, N_slice_EM, N_slice_DN, N_params, T_phot_grid,
                    T_het_grid, I_phot_grid, I_het_grid, y_p, n_live_points=N_live)

    # Change directory back to directory where user's python script is located
    os.chdir('../../../../')


@jit(nopython = True)
def CLR_Prior(chem_params_drawn, limit = -12.0):
    
    ''' Implements the centred-log-ratio (CLR) prior for chemical mixing ratios.
    
        CLR[i] here is the centred log-ratio transform of the mixing ratio, X[i]
       
    '''
    
    n = len(chem_params_drawn)     # Number of species free parameters

    # Limits correspond to condition that all X_i > 10^(-12)
    prior_lower_CLR = ((n-1.0)/n) * (limit * np.log(10.0) + np.log(n-1.0))      # Lower limit corresponds to species underabundant
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

# Define the prior transformation function
def log_prior(cube, model, prior_types, prior_ranges):
    ''' 
    Transforms the unit cube provided by MultiNest into the values of 
    each free parameter used by the forward model.
    
    '''
    # Unpack model properties
    param_names = model['param_names']
    param_species = model['param_species']
    Atmosphere_dimension = model['Atmosphere_dimension']
    X_params = model['X_param_names']
    species_EM_gradient = model['species_EM_gradient']
    species_DN_gradient = model['species_DN_gradient']

    log_prior = 0

    # Assign prior distribution to each free parameter
    for i, parameter in enumerate(param_names):

        # First deal with all parameters besides mixing ratios
        if (parameter not in X_params) or (parameter in ['C_to_O', 'log_Met']):

            # Uniform priors
            if (prior_types[parameter] == 'uniform'):

                min_value = prior_ranges[parameter][0]
                max_value = prior_ranges[parameter][1]
                
                if (cube[i] < min_value) or (cube[i] > max_value):
                    return -np.inf
                else:
                    log_prior -= np.log(max_value-min_value)

            # Gaussian priors
            elif (prior_types[parameter] == 'gaussian'):

                mean = prior_ranges[parameter][0]
                std = prior_ranges[parameter][1]

                log_prior -= 0.5 * ((cube[i] - mean) ** 2) / (std ** 2)

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

                if (cube[i] < min_value) or (cube[i] > max_value):
                    return -np.inf
                else:
                    log_prior -= np.log(max_value-min_value)

            # For 2D models, the prior range for 'Delta' parameters can change to satisfy mixing ratio priors
            elif (Atmosphere_dimension == 2):

                # Absolute mixing ratio parameter comes first
                if ('Delta' not in parameter):

                    min_value = prior_ranges[parameter][0]
                    max_value = prior_ranges[parameter][1]

                    last_value = ((cube[i] * (max_value - min_value)) + min_value)

                    # Store name of previous parameter for delta prior
                    prev_parameter = parameter
                    
                    if (cube[i] < min_value) or (cube[i] > max_value):
                        return -np.inf
                    else:
                        log_prior -= np.log(max_value-min_value)
            
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
                    
                    # cube[i] = ((cube[i] * (max_value_delta - min_value_delta)) + min_value_delta)

                    if (cube[i] < min_value_delta) or (cube[i] > max_value_delta):
                        return -np.inf
                    else:
                        log_prior -= np.log(max_value_delta-min_value_delta)

                
            # For 3D models, the prior ranges for 'Delta' parameters can change to satisfy mixing ratio priors
            elif (Atmosphere_dimension == 3):
                    
                # For species with 3D gradients, sample such that highest and lowest values still satisfy mixing ratio prior
                if ((species in species_EM_gradient) and (species in species_DN_gradient)):

                    # Absolute mixing ratio parameter comes first
                    if ('Delta' not in parameter):

                        min_value = prior_ranges[parameter][0]
                        max_value = prior_ranges[parameter][1]

                        if (cube[i] < min_value) or (cube[i] > max_value):
                            return -np.inf
                        else:
                            log_prior -= np.log(max_value-min_value)

                        # Store name of previous parameter for next delta prior
                        prev_parameter = parameter
                    
                    # Terminator mixing ratio gradient parameter comes second
                    elif (parameter is ('Delta_log_' + species + '_term')):

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

                        if (cube[i] < min_value_delta) or (cube[i] > max_value_delta):
                            return -np.inf
                        else:
                            log_prior -= np.log(max_value_delta-min_value_delta)

                        # Store name of previous parameters for next delta prior
                        prev_prev_parameter = prev_parameter
                        prev_parameter = parameter

                    # Day-night mixing ratio gradient parameter comes third
                    elif (parameter is ('Delta_log_' + species + '_DN')):

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

                        if (cube[i] < min_value_delta_DN) or (cube[i] > max_value_delta_DN):
                            return -np.inf
                        else:
                            log_prior -= np.log(max_value_delta_DN-min_value_delta_DN)

                # Species with a 2D gradient (or no gradient) within a 3D model reduces to the 2D case above
                else:

                    # Absolute mixing ratio parameter comes first
                    if ('Delta' not in parameter):

                        min_value = prior_ranges[parameter][0]
                        max_value = prior_ranges[parameter][1]

                        if (cube[i] < min_value) or (cube[i] > max_value):
                            return -np.inf
                        else:
                            log_prior -= np.log(max_value-min_value)

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
                        sampled_abundance = cube[i-1]

                        # Find largest gradient such that the abundances in all
                        # atmospheric regions satisfy the absolute abundance constraint
                        largest_delta = 2*min((sampled_abundance - min_prior_abs), 
                                                (max_prior_abs - sampled_abundance))   # This is |Delta|_max

                        # Max / min values governed by the most restrictive of
                        # delta prior or absolute prior, such that both are satisfied
                        max_value_delta = min(max_prior_delta, largest_delta)
                        min_value_delta = max(min_prior_delta, -largest_delta)
                        
                        if (cube[i] < min_value_delta) or (cube[i] > max_value_delta):
                            return -np.inf
                        else:
                            log_prior -= np.log(max_value_delta-min_value_delta)
    return log_prior

    '''
    TODO: CLR and sine prior not supported yet.

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

    '''

# Define the log-likelihood function
def LogLikelihood(cube, model, planet, star, data, F_s_obs, P, P_ref, P_param_set, spectrum_type,
                He_fraction, N_slice_EM, N_slice_DN, opac, wl, T_phot_grid,
                T_het_grid, I_phot_grid, I_het_grid, y_p):

    ''' 
    Evaluates the log-likelihood for a given point in parameter space.
    
    Works by generating a PT profile, calculating the opacity in the
    model atmosphere, computing the resulting spectrum and finally 
    convolving and integrating the spectrum to produce model data 
    points for each instrument.
    
    The log-likelihood is then evaluated using the difference between 
    the binned spectrum and the actual data points. 
    
    '''
    # Unpack model properties
    physical_param_names = model['physical_param_names']
    N_params_cum = model['N_params_cum']
    error_inflation = model['error_inflation']
    offsets_applied = model['offsets_applied']
    radius_unit = model['radius_unit']
    distance_unit = model['distance_unit']
    surface = model['surface']
    high_res = model['high_res']
    high_res_param_names = model['high_res_param_names']
    
    # Pre-compute normalisation for log-likelihood 
    if not high_res:
        err_data = data['err_data']
        norm_log_default = (-0.5*np.log(2.0*np.pi*err_data*err_data)).sum()
    else:
        wl_grid = data['wl_grid']
        V_bary = data['V_bary']
        Phi = data['Phi']
        V_sin_i = planet['V_sin_i']        
        data_scale = data['data_scale']
        data_arr = data['data_arr']

    # Immediately reject samples falling outside of mixing ratio simplex (CLR prior only)
    global allowed_simplex
    if (allowed_simplex == 0):
        loglikelihood = -1.0e100   
        return loglikelihood

    # For a retrieval we do not have user provided P-T or chemical profiles
    T_input = []
    log_X_input = []

    #***** Step 1: unpack parameter values from prior sample *****#
    
    physical_params, PT_params, \
    log_X_params, cloud_params, \
    geometry_params, stellar_params, \
    offset_params, err_inflation_params, \
    high_res_params = split_params(cube, N_params_cum)

    # Unpack reference radius parameter
    R_p_ref = physical_params[np.where(physical_param_names == 'R_p_ref')[0][0]]

    # Convert normalised radius drawn by MultiNest back into SI
    if (radius_unit == 'R_J'):
        R_p_ref *= R_J
    elif (radius_unit == 'R_E'):
        R_p_ref *= R_E

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

    #***** Step 2: generate atmosphere corresponding to parameter draw *****#

    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, 
                                    log_X_params, cloud_params, geometry_params, 
                                    log_g, T_input, log_X_input, P_surf, # P_param_set
                                    He_fraction, N_slice_EM, N_slice_DN)

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
        
        # Assign penalty to likelihood => point ignored in retrieval
        loglikelihood = -1.0e100
        
        # Quit if given parameter combination is unphysical
        return loglikelihood

    if high_res:
        if high_res == 'sysrem':
            # loglikelihood = log_likelihood(F_s_obs, spectrum, wl, wl_grid, V_bary, Phi, V_sin_i, model, high_res_params, 
            #                                 high_res_param_names, residuals=residuals, uncertainties=uncertainties, Bs=Bs)
            return
        elif high_res == 'pca':
            loglikelihood = log_likelihood(F_s_obs, spectrum, wl, wl_grid, V_bary, Phi, V_sin_i, model, high_res_params, 
                                            high_res_param_names, data_arr=data_arr, data_scale=data_scale)

        return loglikelihood
    #***** Step 4: stellar contamination *****#
    
    # Stellar contamination is only relevant for transmission spectra
    if ('transmission' in spectrum_type):

        # Model with a single spot / facula population
        if (model['stellar_contam'] == 'one-spot'):

            # Unpack stellar contamination parameters
            f, T_het, T_phot = stellar_params
            
            # Find photosphere and spot / faculae intensities at relevant effective temperatures
            I_het = I_het_grid[closest_index(T_het, T_het_grid[0], 
                                                T_het_grid[-1], len(T_het_grid)),:]
            I_phot = I_phot_grid[closest_index(T_phot, T_phot_grid[0], 
                                                T_phot_grid[-1], len(T_phot_grid)),:]
            
            # Compute wavelength-dependant stellar contamination factor
            epsilon = stellar_contamination_single_spot(f, I_het, I_phot)

            # Apply multiplicative stellar contamination to spectrum
            spectrum = epsilon * spectrum
        
    #***** Step 5: convolve spectrum with instrument PSF and bin to data resolution ****#

    if ('transmission' in spectrum_type):
        ymodel = bin_spectrum_to_data(spectrum, wl, data)
    else:
        F_p_binned = bin_spectrum_to_data(spectrum, wl, data)
        if ('direct' in spectrum_type):
            ymodel = F_p_binned
        else:
            F_s_binned = bin_spectrum_to_data(F_s_obs, wl, data)
            ymodel = F_p_binned/F_s_binned
                                
    #***** Step 6: inflate error bars (optional) ****#
    
    # Compute effective error, if unknown systematics included
    err_data = data['err_data']
    
    if (error_inflation == 'Line_2015'):
        err_eff_sq = (err_data*err_data + np.power(10.0, err_inflation_params[0]))
        norm_log = (-0.5*np.log(2.0*np.pi*err_eff_sq)).sum()
    else: 
        err_eff_sq = err_data*err_data
        norm_log = norm_log_default

    #***** Step 7: apply relative offset between datasets (optional) ****#
    
    ydata = data['ydata']
    offset_start = data['offset_start']
    offset_end = data['offset_end']

    if (offsets_applied == 'relative'): 
        ydata_adjusted = ydata.copy()
        ydata_adjusted[offset_start:offset_end] += offset_params[0]
    else: 
        ydata_adjusted = ydata

    #***** Step 8: evaluate ln(likelihood) ****#

    loglikelihood = (-0.5*((ymodel - ydata_adjusted)**2)/err_eff_sq).sum()
    loglikelihood += norm_log
                
    return loglikelihood


def pocoMC_retrieval(planet, star, model, opac, data, prior_types, 
                    prior_ranges, spectrum_type, wl, P, P_ref, P_param_set, 
                    He_fraction, N_slice_EM, N_slice_DN, N_params, T_phot_grid,
                    T_het_grid, I_phot_grid, I_het_grid, y_p, n_live_points):

    ''' 
    Main function for conducting atmospheric retrievals with pocoMC.
    
    '''

    # Unpack model properties
    param_names = model['param_names']
    X_params = model['X_param_names']
    high_res = model['high_res']
    d = planet['system_distance']

    # Unpack number of free mixing ratio parameters for prior function  
    N_species_params = len(X_params)

    # Assign PyMultiNest keyword arguments
    n_dims = N_params
    # Pre-compute normalisation for log-likelihood
    if not high_res:
        err_data = data['err_data']
        norm_log_default = (-0.5*np.log(2.0*np.pi*err_data*err_data)).sum()

    bounds = np.empty((n_dims, 2))
    
    # TODO: parameters concerning Delta
    for i, parameter in enumerate(param_names):
        if (prior_types[parameter] == 'uniform') and ('Delta' not in parameter):
            min_value = prior_ranges[parameter][0]
            max_value = prior_ranges[parameter][1]
            bounds[i, 0] = min_value
            bounds[i, 1] = max_value
        else:
            bounds[i, 0] = None
            bounds[i, 1] = None
    
    # Create variable governing if a mixing ratio parameter combination lies in 
    # the allowed CLR simplex space (X_i > 10^-12 and sum to 1)
    global allowed_simplex    # Needs to be global, as prior function has no return

    allowed_simplex = 1    # Only changes to 0 for CLR variables outside prior

    # Interpolate stellar spectrum onto planet wavelength grid (one-time operation)
    if (('transmission' not in spectrum_type) and (star != None)):

        # Load stellar spectrum
        F_s = star['F_star']
        wl_s = star['wl_star']
        R_s = star['stellar_radius']

        # Distance only used for flux ratios, so set it to 1 since it cancels
        if (d is None):
            planet['system_distance'] = 1
            d = planet['system_distance']

        # Interpolate stellar spectrum onto planet spectrum wavelength grid
        F_s_interp = spectres(wl, wl_s, F_s)

        # Convert stellar surface flux to observed flux at Earth
        F_s_obs = (R_s / d)**2 * F_s_interp

    # Skip for directly imaged planets or brown dwarfs
    else:

        # Stellar flux not needed for transmission spectra
        F_s_obs = None

    prior_samples = np.zeros(shape = (n_live_points, n_dims))

    # TODO: parameters concerning Delta
    for i, parameter in enumerate(param_names):
        if (prior_types[parameter] == 'uniform') and ('Delta' not in parameter):
            min_value = prior_ranges[parameter][0]
            max_value = prior_ranges[parameter][1]
            prior_samples[:, i] = np.random.uniform(low=min_value, high=max_value, size=n_live_points)
        elif (prior_types[parameter] == 'gaussian'):
            mean = prior_ranges[parameter][0]
            std = prior_ranges[parameter][1]
            prior_samples[:, i] = np.random.normal(mean, std, size=n_live_points)

    from multiprocessing import Pool

    # n_cpus = 9
    

    # with Pool(n_cpus) as pool:

    sampler = pc.Sampler(
        n_live_points,
        n_dims,
        log_likelihood=LogLikelihood,
        log_likelihood_args=[model, planet, star, data, F_s_obs, P, P_ref, P_param_set, spectrum_type,
        He_fraction, N_slice_EM, N_slice_DN, opac, wl, T_phot_grid,
        T_het_grid, I_phot_grid, I_het_grid, y_p], 
        log_prior=log_prior,
        log_prior_args=[model, prior_types, prior_ranges],
        vectorize_likelihood=False,
        bounds=bounds,
        random_state=0,
        infer_vectorization=False,
        # pool=pool,
        # parallelize_prior=True
    )

    # Problem: looks like each cube has three samples
    sampler.run(prior_samples)
    results = sampler.results
    pc.plotting.corner(results, dims = np.arange(n_dims))
    plt.show()