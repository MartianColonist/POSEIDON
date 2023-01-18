# Functions related to atmospheric retrieval

import numpy as np
import time
import os
import pymultinest
from mpi4py import MPI
from scipy.special import ndtri
from spectres import spectres   
from numba.core.decorators import jit
from scipy.special import erfcinv
from scipy.special import lambertw as W
from scipy.constants import parsec

from .constants import R_J, R_E

from .parameters import split_params
from .instrument import bin_spectrum_to_data
from .utility import write_MultiNest_results, round_sig_figs, closest_index, \
                     write_retrieved_spectrum, write_retrieved_PT, \
                     write_retrieved_log_X, confidence_intervals
from .core import make_atmosphere, compute_spectrum
from .stellar import precompute_stellar_spectra, stellar_contamination_single_spot
from .high_res import log_likelihood, sysrem, fast_filter, fit_uncertainties

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create global variable needed for centred log-ratio prior function
allowed_simplex = 1



def run_retrieval(planet, star, model, opac, data, priors, 
                  wl, P, P_ref = 10.0, R = None, retrieval_name = None,
                  He_fraction = 0.17, N_slice_EM = 2, N_slice_DN = 4, 
                  spectrum_type = 'transmission', N_live = 400, ev_tol = 0.5,
                  sampling_algorithm = 'MultiNest', resume = False, 
                  verbose = True, sampling_target = 'parameter',
                  N_output_samples = 1000):
    '''
    ADD DOCSTRING
    '''

    # Unpack planet name
    planet_name = planet['planet_name']
    
    # Unpack prior types and ranges
    prior_types = priors['prior_types']
    prior_ranges = priors['prior_ranges']

    # Unpack model properties
    model_name = model['model_name']
    chemical_species = model['chemical_species']
    param_names = model['param_names']
    high_res = model['high_res']
    N_params = len(param_names)
    
    if (retrieval_name is None):
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'

    # Pre-compute stellar spectra for models with unocculted spots / faculae
    if (model['stellar_contam'] != 'No'):

        if (rank == 0):
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

    if (rank == 0):
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
                              He_fraction, N_slice_EM, N_slice_DN, N_params, 
                              T_phot_grid, T_het_grid, I_phot_grid,
                              I_het_grid, high_res, resume = resume, verbose = verbose,
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
      
            # Write POSEIDON retrieval output files
            if not high_res:
                write_MultiNest_results(planet, model, data, retrieval_name,
                                    N_live, ev_tol, sampling_algorithm, wl, R)

            # Compute samples of retrieved P-T, mixing ratio profiles, and spectrum
            T_low2, T_low1, T_median, \
            T_high1, T_high2, \
            log_X_low2, log_X_low1, \
            log_X_median, log_X_high1, \
            log_X_high2, \
            spec_low2, spec_low1, \
            spec_median, spec_high1, \
            spec_high2 = retrieved_samples(planet, star, model, opac,
                                           retrieval_name, wl, P, P_ref, 
                                           He_fraction, N_slice_EM, N_slice_DN, 
                                           spectrum_type, T_phot_grid, 
                                           T_het_grid, I_phot_grid, I_het_grid, 
                                           N_output_samples)
                                            
            # Save sampled P-T profile
            write_retrieved_PT(retrieval_name, P, T_low2, T_low1, 
                               T_median, T_high1, T_high2)

            # Save sampled mixing ratio profiles
            write_retrieved_log_X(retrieval_name, chemical_species, P, 
                                  log_X_low2, log_X_low1, log_X_median, 
                                  log_X_high1, log_X_high2)

            # Save sampled spectrum
            write_retrieved_spectrum(retrieval_name, wl, spec_low2, 
                                     spec_low1, spec_median, spec_high1, spec_high2)

            print("All done! Output files can be found in " + output_dir + "results/")
         
    comm.Barrier()

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


def PyMultiNest_retrieval(planet, star, model, opac, data, prior_types, 
                          prior_ranges, spectrum_type, wl, P, P_ref, He_fraction, 
                          N_slice_EM, N_slice_DN, N_params, T_phot_grid,
                          T_het_grid, I_phot_grid, I_het_grid, high_res, **kwargs):
    ''' 
    Main function for conducting atmospheric retrievals with PyMultiNest.
    
    '''

    # Unpack model properties
    param_names = model['param_names']
    physical_param_names = model['physical_param_names']
    param_species = model['param_species']
    X_params = model['X_param_names']
    N_params_cum = model['N_params_cum']
    Atmosphere_dimension = model['Atmosphere_dimension']
    species_EM_gradient = model['species_EM_gradient']
    species_DN_gradient = model['species_DN_gradient']
    error_inflation = model['error_inflation']
    offsets_applied = model['offsets_applied']
    radius_unit = model['radius_unit']
    distance_unit = model['distance_unit']
    surface = model['surface']
    high_res = model['high_res']
    high_res_param_names = model['high_res_param_names']
    R_p = planet['planet_radius']
    d = planet['system_distance']

    # Unpack number of free mixing ratio parameters for prior function  
    N_species_params = len(X_params)

    # Assign PyMultiNest keyword arguments
    n_dims = N_params
    
    # Pre-compute normalisation for log-likelihood 
    if not high_res:
        err_data = data['err_data']
        norm_log_default = (-0.5*np.log(2.0*np.pi*err_data*err_data)).sum()
    else:
        wl_grid = data['wl_grid']
        V_bary = data['V_bary']
        Phi = data['Phi']
        V_sin_i = planet['V_sin_i']

        if high_res == 'sysrem':
            data_raw = data['data_raw']
            data_norm = np.zeros(data_raw.shape)
            for i in range(len(data_raw)):
                order = data_raw[i]
                order_norm = (order.T / np.median(order, axis=1)).T
                data_norm[i] = order_norm
            residuals, Us, Ws = fast_filter(data_norm, iter=15)
            uncertainties = fit_uncertainties(data_raw, NPC=5)
        elif high_res == 'pca':
            data_scale = data['data_scale']
            data_arr = data['data_arr']

    # Create variable governing if a mixing ratio parameter combination lies in 
    # the allowed CLR simplex space (X_i > 10^-12 and sum to 1)
    global allowed_simplex    # Needs to be global, as prior function has no return

    allowed_simplex = 1    # Only changes to 0 for CLR variables outside prior

    # Interpolate stellar spectrum onto planet wavelength grid (one-time operation)
    if ((spectrum_type != 'transmission') and (star != None)):

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
    
    # Define the prior transformation function
    def Prior(cube, ndim, nparams):
        ''' 
        Transforms the unit cube provided by MultiNest into the values of 
        each free parameter used by the forward model.
        
        '''

        # Assign prior distribution to each free parameter
        for i, parameter in enumerate(param_names):

            # First deal with all parameters besides mixing ratios 
            if (parameter not in X_params):

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

                            cube[i] = ((cube[i] * (max_value_delta - min_value_delta)) + min_value_delta)

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

                            cube[i] = ((cube[i] * (max_value_delta_DN - min_value_delta_DN)) + min_value_delta_DN)

                    # Species with a 2D gradient (or no gradient) within a 3D model reduces to the 2D case above
                    else:

                        # Absolute mixing ratio parameter comes first
                        if ('Delta' not in parameter):

                            min_value = prior_ranges[parameter][0]
                            max_value = prior_ranges[parameter][1]

                            cube[i] = ((cube[i] * (max_value - min_value)) + min_value)

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
        if ('R_p_ref' in physical_param_names):
            R_p_ref = physical_params[np.where(physical_param_names == 'R_p_ref')[0][0]]

            # Convert normalised radius drawn by MultiNest back into SI
            if (radius_unit == 'R_J'):
                R_p_ref *= R_J
            elif (radius_unit == 'R_E'):
                R_p_ref *= R_E
        else:
            R_p_ref = R_p

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
                                     log_g, T_input, log_X_input, P_surf,
                                     He_fraction, N_slice_EM, N_slice_DN)

        #***** Step 3: generate spectrum of atmosphere ****#

        # For emission spectra retrievals we directly compute Fp (instead of Fp/F*)
        # so we can convolve and bin Fp and F* separately when comparing to data
        if (('emission' in spectrum_type) and (spectrum_type != 'direct_emission')):
            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                        spectrum_type = ('direct_' + spectrum_type))   # Always Fp (even for secondary eclipse)

        # For transmission spectra and direct emission spectra
        else:
            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                        spectrum_type)

        # Reject unphysical spectra (forced to be NaN by function above)
        if (np.any(np.isnan(spectrum))):
            
            # Assign penalty to likelihood => point ignored in retrieval
            loglikelihood = -1.0e100
            
            # Quit if given parameter combination is unphysical
            return loglikelihood

        if high_res:
            if high_res == 'sysrem':
                loglikelihood = log_likelihood(F_s_obs, spectrum, wl, wl_grid, V_bary, Phi, V_sin_i, model, high_res_params, 
                                                high_res_param_names, residuals=residuals, uncertainties=uncertainties, Us=Us)
            elif high_res == 'pca':
                loglikelihood = log_likelihood(F_s_obs, spectrum, wl, wl_grid, V_bary, Phi, V_sin_i, model, high_res_params, 
                                                high_res_param_names, data_arr=data_arr, data_scale=data_scale)

            return loglikelihood
            
        
        #***** Step 4: stellar contamination *****#
        # Stellar contamination is only relevant for transmission spectra
        if (spectrum_type == 'transmission'):

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

        if (spectrum_type == 'transmission'):
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
    
    # Run PyMultiNest
    pymultinest.run(LogLikelihood, Prior, n_dims, **kwargs)
	

def retrieved_samples(planet, star, model, opac, retrieval_name,
                      wl, P, P_ref, He_fraction, N_slice_EM, N_slice_DN, 
                      spectrum_type, T_phot_grid, T_het_grid, I_phot_grid,
                      I_het_grid, N_output_samples):
    '''
    ADD DOCSTRING
    '''

    # Unpack number of free parameters
    param_names = model['param_names']
    physical_param_names = model['physical_param_names']
    n_params = len(param_names)

    # Unpack model properties
    radius_unit = model['radius_unit']
    distance_unit = model['distance_unit']
    N_params_cum = model['N_params_cum']
    surface = model['surface']

    R_p = planet['planet_radius']

    # Load relevant output directory
    output_prefix = retrieval_name + '-'

    # For a retrieval we do not have user provided P-T or chemical profiles
    T_input = []
    log_X_input = []
    
    # Run PyMultiNest analyser to extract posterior samples
    analyzer = pymultinest.Analyzer(n_params, outputfiles_basename = output_prefix,
                                    verbose = False)
    samples = analyzer.get_equal_weighted_posterior()[:,:-1]

    # Find total number of available posterior samples from MultiNest 
    N_samples_total = len(samples[:,0])
    
    # Randomly draw parameter samples from posterior
    N_sample_draws = min(N_samples_total, N_output_samples)
    sample = np.random.choice(len(samples), N_sample_draws, replace=False)

    print("Now generating " + str(N_sample_draws) + " sampled spectra and " + 
          "P-T profiles from the posterior distribution...")
                    
    # Generate spectrum and PT profiles from selected samples
    for i in range(N_sample_draws):

        # Estimate run time for this function based on one model evaluation
        if (i == 0):
            t0 = time.perf_counter()   # Time how long one model takes

        # Convert MultiNest parameter samples into POSEIDON function inputs
        physical_params, PT_params, \
        log_X_params, cloud_params, \
        geometry_params, stellar_params, \
        offset_params, err_inflation_params, \
        high_res_params = split_params(samples[sample[i],:], N_params_cum)

        # Unpack reference radius parameter
        if ('R_p_ref' in physical_param_names):
            R_p_ref = physical_params[np.where(physical_param_names == 'R_p_ref')[0][0]]

            # Convert normalised radius drawn by MultiNest back into SI
            if (radius_unit == 'R_J'):
                R_p_ref *= R_J
            elif (radius_unit == 'R_E'):
                R_p_ref *= R_E
        else:
            R_p_ref = R_p


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

        # Generate atmosphere corresponding to parameter draw
        atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, 
                                     log_X_params, cloud_params, geometry_params, 
                                     log_g, T_input, log_X_input, P_surf,
                                     He_fraction, N_slice_EM, N_slice_DN)

        # Generate spectrum of atmosphere
        spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                    spectrum_type)

        # Stellar contamination is only relevant for transmission spectra
        if (spectrum_type == 'transmission'):

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

        # Based on first model, create arrays to store retrieved temperature, spectrum, and mixing ratios
        if (i == 0):

            # Estimate run time for this function based on one model evaluation
            t1 = time.perf_counter()
            total = round_sig_figs((N_sample_draws * (t1-t0)/60.0), 2)  # Round to 2 significant figures
            
            print('This process will take approximately ' + str(total) + ' minutes')

            # Find size of mixing ratio field (same as temperature field)
            N_species, N_D, N_sectors, N_zones = np.shape(atmosphere['X'])

            # Create arrays to store sampled retrieval outputs
            T_stored = np.zeros(shape=(N_sample_draws, N_D, N_sectors, N_zones))
            log_X_stored = np.zeros(shape=(N_sample_draws, N_species, N_D, N_sectors, N_zones))
            spectrum_stored = np.zeros(shape=(N_sample_draws, len(wl)))

        # Store temperature field and spectrum in sample arrays
        T_stored[i,:,:,:] = atmosphere['T']
        log_X_stored[i,:,:,:,:] = np.log10(atmosphere['X'])
        spectrum_stored[i,:] = spectrum
            
    # Compute 1 and 2 sigma confidence intervals for P-T and mixing ratio profiles and spectrum
        
    # P-T profile
    _, T_low2, T_low1, T_median, \
    T_high1, T_high2, _ = confidence_intervals(N_sample_draws, 
                                               T_stored[:,:,0,0], N_D)

    # Mixing ratio profiles
    log_X_low2 = np.zeros(shape=(N_species, N_D))
    log_X_low1 = np.zeros(shape=(N_species, N_D))
    log_X_median = np.zeros(shape=(N_species, N_D))
    log_X_high1 = np.zeros(shape=(N_species, N_D))
    log_X_high2 = np.zeros(shape=(N_species, N_D))

    # Loop over each chemical species
    for q in range(N_species):

        _, log_X_low2[q,:], log_X_low1[q,:], \
        log_X_median[q,:], log_X_high1[q,:], \
        log_X_high2[q,:], _ = confidence_intervals(N_sample_draws, 
                                                   log_X_stored[:,q,:,0,0], N_D)
    
    # Spectrum
    _, spec_low2, spec_low1, spec_median, \
    spec_high1, spec_high2, _ = confidence_intervals(N_sample_draws, 
                                                     spectrum_stored, len(wl))
    
    return T_low2, T_low1, T_median, T_high1, T_high2, \
           log_X_low2, log_X_low1, log_X_median, log_X_high1, log_X_high2, \
           spec_low2, spec_low1, spec_median, spec_high1, spec_high2


#***** Compute Bayes factors, sigma significance etc *****#

def Z_to_sigma(ln_Z1, ln_Z2):
    
    ''' Convert log-evidences of two models to a sigma confidence level.
    
    '''
    
    np.set_printoptions(precision=50)

    B = np.exp(ln_Z1 - ln_Z2)                        # Bayes factor
    p = np.real(np.exp(W((-1.0/(B*np.exp(1))),-1)))  # p-value

    sigma = np.sqrt(2)*erfcinv(p)    # Equivalent sigma
    
    #print "p-value = ", p
    #print "n_sigma = ", sigma
    
    return B, sigma   


def Bayesian_model_comparison(model_feature, base_name_1, base_name_2, 
                              planet_name, err_data, n_params, n_removed,
                              provide_min_chi_square = True):
    
    ''' Conduct Bayesian model comparison between the outputs of two
        (nested) retrievals. This function outputs the Bayes factor and
        equivalent sigma significance for two models.
        
    '''
    
    # Find file locations for each retrieval
    base_name_present = ('../../output/retrievals/' + planet_name + 
                         '/MultiNest_raw/' + base_name_1)
    base_name_absent = ('../../output/retrievals/' + planet_name + 
                         '/MultiNest_raw/' + base_name_2)
    
    # Compute the likelihood normalisation for this dataset 
    N_data = len(err_data)
    norm_log = (-0.5*np.log(2.0*np.pi*err_data*err_data)).sum()

    # Open the retrieval outputs
    retrieval_present = pymultinest.Analyzer(n_params = n_params, 
                                             outputfiles_basename = base_name_present)
    retrieval_absent = pymultinest.Analyzer(n_params = (n_params - n_removed), 
                                             outputfiles_basename = base_name_absent)
    
    # Load the stats summaries
    s_present = retrieval_present.get_stats()
    s_absent = retrieval_absent.get_stats()

    # Extract Bayesian evidences of models with and without the feature   
    ln_Z_present = s_present['global evidence']
    ln_Z_absent = s_absent['global evidence']
    
    print("Evidence with " + model_feature + " = " + str(ln_Z_present))
    print("Evidence without " + model_feature + " = " + str(ln_Z_absent))

    # Compute Bayes factor and equivalent sigma
    Bayes_factor, n_sigma = Z_to_sigma(ln_Z_present, ln_Z_absent)
    
    print("ln(Bayes) of " + model_feature + " = " + str(np.log(Bayes_factor)))
    print("Sigma significance of " + model_feature + " = " + str(n_sigma))

    if (n_sigma < 3.0):
        print("No Detection of " + model_feature)
        
    if (n_sigma >= 3.0):
        print("Detection of " + model_feature + "!")

    # Calculate minimum reduced chi-square of each model
    if (provide_min_chi_square == True):
        
        # Find bet-fitting parameter combinations
        best_fit_present = retrieval_present.get_best_fit()
        best_fit_absent = retrieval_absent.get_best_fit()

        # Find maximum corresponding likelihoods
        max_likelihood_present = best_fit_present['log_likelihood']
        max_likelihood_absent = best_fit_absent['log_likelihood']

        # Compute corresponding minimum chi-square
        min_chi_square_present = -2.0 * (max_likelihood_present - norm_log)
        min_chi_square_absent = -2.0 * (max_likelihood_absent - norm_log)

        # Calculate reduced chi-squares of the models
        reduced_chi_square_present = min_chi_square_present/(N_data - n_params)
        reduced_chi_square_absent = min_chi_square_absent/(N_data - (n_params - n_removed))
        
        print("Minimum reduced Chi-square with " + model_feature + " = " + str(reduced_chi_square_present))
        print("Minimum reduced Chi-square no " + model_feature + " = " + str(reduced_chi_square_absent))
    
    return