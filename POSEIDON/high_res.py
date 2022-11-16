# Functions related to atmospheric retrieval

import numpy as np
import time
import os
import pymultinest
from mpi4py import MPI
from scipy.special import ndtri
from spectres import spectres
from numba.core.decorators import jit
from scipy.constants import parsec

from .constants import R_J, R_E

from .parameters import split_params
from .instrument import bin_spectrum_to_data
from .utility import write_MultiNest_results, round_sig_figs, closest_index, \
                     write_retrieved_spectrum, write_retrieved_PT, \
                     write_retrieved_log_X, confidence_intervals
from .core import make_atmosphere, compute_spectrum
from .stellar import precompute_stellar_spectra, stellar_contamination_single_spot
from .cross_correlate import log_likelihood

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create global variable needed for centred log-ratio prior function
allowed_simplex = 1



def run_high_res_retrieval(planet, star, model, opac, data, priors, 
                  wl, P, P_ref = 10.0, R = None, retrieval_name = None,
                  He_fraction = 0.17, N_slice_EM = 2, N_slice_DN = 4, 
                  spectrum_type = 'direct_emission', N_live = 400, ev_tol = 0.5,
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
    N_params = len(param_names)

    if (retrieval_name is None):
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'

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
        high_res_retrieval(planet, star, model, opac, data, prior_types, 
                              prior_ranges, spectrum_type, wl, P, P_ref, 
                              He_fraction, N_slice_EM, N_slice_DN, N_params, 
                              resume = resume, verbose = verbose,
                              outputfiles_basename = basename, 
                              n_live_points = N_live, multimodal = False,
                              evidence_tolerance = ev_tol, log_zero = -1e90,
                              importance_nested_sampling = False, 
                              sampling_efficiency = sampling_target, 
                              const_efficiency_mode = False)

        # Write retrieval results to file
        if (rank == 0):
            
            # pdb.set_trace()

            # Write retrieval runtime to terminal
            t1 = time.perf_counter()
            total = round_sig_figs((t1-t0)/3600.0, 2)  # Round to 2 significant figures
            
            print('POSEIDON retrieval finished in ' + str(total) + ' hours')
      
            # Write POSEIDON retrieval output files 
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
                                           spectrum_type, N_output_samples)
                                            
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


from .retrieval import CLR_Prior


def high_res_retrieval(planet, star, model, opac, data, prior_types, 
                        prior_ranges, spectrum_type, wl, P, P_ref, He_fraction,
                        N_slice_EM, N_slice_DN, N_params, **kwargs):
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
    offsets_applied = model['offsets_applied']
    radius_unit = model['radius_unit']
    distance_unit = model['distance_unit']
    surface = model['surface']
    R_p = planet['planet_radius']
    d = planet['system_distance']

    # Unpack number of free mixing ratio parameters for prior function  
    N_species_params = len(X_params)

    # Assign PyMultiNest keyword arguments
    n_dims = N_params

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
        offset_params, err_inflation_params, high_res_params = split_params(cube, N_params_cum)

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
                                     log_g, T_input, log_X_input, P_surf,
                                     He_fraction, N_slice_EM, N_slice_DN)

        #***** Step 3: generate spectrum of atmosphere ****#

        # For emission spectra retrievals we directly compute Fp (instead of Fp/F*)
        # so we can convolve and bin Fp and F* separately when comparing to data
        if (('emission' in spectrum_type) and (spectrum_type != 'direct_emission')):
            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                        spectrum_type = ('direct_' + spectrum_type))   # Always Fp (even for secondary eclipse)

        else:
            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                        spectrum_type)

        # Reject unphysical spectra (forced to be NaN by function above)
        if (np.any(np.isnan(spectrum))):
            
            # Assign penalty to likelihood => point ignored in retrieval
            loglikelihood = -1.0e100
            
            # Quit if given parameter combination is unphysical
            return loglikelihood

        #***** Step 7: Get the data properties from data dictionary ****#
        wl_grid = data['wl_grid']
        data_arr = data['data_arr']
        data_scale = data['data_scale']
        V_bary = data['V_bary']
        Phi = data['Phi']
        K_p = high_res_params[0]
        V_sys = high_res_params[1]

        loglikelihood = log_likelihood(F_s_obs, spectrum, wl, K_p, V_sys, wl_grid, data_arr, data_scale, V_bary, Phi)
        
        return loglikelihood[0]
    
    # Run PyMultiNest
    pymultinest.run(LogLikelihood, Prior, n_dims, **kwargs)
	

def retrieved_samples(planet, star, model, opac, retrieval_name,
                      wl, P, P_ref, He_fraction, N_slice_EM, N_slice_DN, 
                      spectrum_type, N_output_samples):
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
        offset_params, err_inflation_params, high_res_params = split_params(samples[sample[i],:], 
                                                           N_params_cum)

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

        # Generate atmosphere corresponding to parameter draw
        atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, 
                                     log_X_params, cloud_params, geometry_params, 
                                     log_g, T_input, log_X_input, P_surf,
                                     He_fraction, N_slice_EM, N_slice_DN)

        # Generate spectrum of atmosphere
        spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                    spectrum_type)

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

from .retrieval import Z_to_sigma

from .retrieval import Bayesian_model_comparison