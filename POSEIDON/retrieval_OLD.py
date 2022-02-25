# Functions related to atmospheric retrieval

import numpy as np
import pymultinest
from scipy.special import ndtri
from numba.core.decorators import jit
from scipy.special import erfcinv
from scipy.special import lambertw as W

# Import model settings from config.py
from config import PT_profile, chemistry_prior, cloud_model, stellar_contam, \
                   prior_lower_PT, prior_upper_PT, prior_lower_R_p_ref, \
                   prior_upper_R_p_ref, prior_lower_X, prior_upper_X, \
                   prior_lower_clouds, prior_upper_clouds, prior_lower_geometry, \
                   prior_upper_geometry, prior_lower_stellar, prior_upper_stellar, \
                   prior_lower_offsets, prior_upper_offsets, prior_gauss_T_phot, \
                   spectrum_type, error_inflation, offsets_applied
                   
from parameters import load_state
from transmission import TRIDENT

# Create global variable needed for centred log-ratio prior function
allowed_simplex = 1 

@jit(nopython = True)
def CLR_Prior(chem_params_drawn, N_species_params):
    
    ''' Impliments the centred-log-ratio (CLR) prior for chemical mixing ratios.
    
        CLR[i] here is the centred log-ratio transform of the mixing ratio, X[i]
       
    '''
    
    n = N_species_params    # Number of species free paramters
    limit = -12.0           # Lowest (log) mixing ratio considered

    # Limits correspond to condition that all X_i > 10^(-12)
    prior_lower_CLR = ((n-1.0)/n) * (limit * np.log(10.0) + np.log(n-1.0))      # Lower limit corresponds to species underabundant
    prior_upper_CLR = ((1.0-n)/n) * (limit * np.log(10.0))                      # Upper limit corresponds to species dominant

    CLR = np.zeros(shape=(n+1))   # Vector of CLR variables
    X = np.zeros(shape=(n+1))     # Vector of mixing ratio parameters
    
    # Evalaute centred log-ratio parameters by uniformly sampling between limits
    for i in range(n):
 
        CLR[1+i] = ((chem_params_drawn[i] * (prior_upper_CLR - prior_lower_CLR)) + prior_lower_CLR) 
          
    if (np.abs(np.sum(CLR[1:n])) <= prior_upper_CLR):   # Impose same prior on X_0
            
        CLR[0] = -1.0*np.sum(CLR[1:n])   # CLR_n (corresponding to mixing ratio of first species) must equal 0, so that X_i sum to 1
        
        if ((np.max(CLR) - np.min(CLR)) <= (-1.0 * limit * np.log(10.0))):      # Necessary for all X_i > 10^(-12)    
        
            normalisation = np.sum(np.exp(CLR))
        
            for i in range(n+1):
                
                # Map log-ratio parameters to mixing ratios
                X[i] = np.exp(CLR[i]) / normalisation   # Vector of mixing ratios (should sum to 1!)
                
                # One final check that all X_i > 10^(-12)
                if (X[i] < 1.0e-12): 
                    return (np.ones(n)*(-50.0))    # Fails check -> return dummy array of log values
            
            return np.log10(X[1:])   # Return vector of log-mixing ratios
        
        elif ((np.max(CLR) - np.min(CLR)) > (-1.0 * limit * np.log(10.0))):
        
            return (np.ones(n)*(-50.0))   # Fails check -> return dummy array of log values
    
    elif (np.abs(np.sum(CLR[1:n])) > prior_upper_CLR):   # If falls outside of allowed triangular subspace
        
        return (np.ones(n)*(-50.0))    # Fails check -> return dummy array of log values


def PyMultiNest_retrieval(param_names, N_params_cumulative, chemical_species,
                          bulk_species, param_species, active_species, 
                          cia_pairs, ff_pairs, bf_species, wl, sigma_stored, 
                          cia_stored, Rayleigh_stored, eta_stored, ff_stored, 
                          bf_stored, species_EM_gradient, species_DN_gradient, 
                          species_vert_gradient, T_phot_grid, I_phot_grid, 
                          T_het_grid, I_het_grid, enable_haze, enable_deck, 
                          Atmosphere_dimension, N_sectors, N_zones, 
                          prior_lower_err_inflation, prior_upper_err_inflation, 
                          data_properties, base_name, planet_name, **kwargs):

    ''' Main function for conducting atmospheric retrievals with PyMultiNest.
    
    '''

    # Unpack number of free parameters of each type for convience in prior function  
    N_PT_params = N_params_cumulative[0]   # No need to extract R_p_ref ([1] - [0])
    N_species_params = N_params_cumulative[2] - N_params_cumulative[1]
    N_cloud_params = N_params_cumulative[3] - N_params_cumulative[2]
    N_geometry_params = N_params_cumulative[4] - N_params_cumulative[3]
    N_stellar_params = N_params_cumulative[5] - N_params_cumulative[4]
    N_offset_params = N_params_cumulative[6] - N_params_cumulative[5]
    N_error_params = N_params_cumulative[7] - N_params_cumulative[6]

    # Unpack PyMultiNest keyword arguments
    n_params = len(param_names)
    kwargs['n_dims'] = n_params
    kwargs['outputfiles_basename'] = ('../../output/retrievals/' + planet_name + 
                                      '/MultiNest_raw/' + base_name)
    
    # Pre-compute normalisation for log-likelihood 
    err_data = data_properties['err_data']
    norm_log_default = (-0.5*np.log(2.0*np.pi*err_data*err_data)).sum()
    
    # Define the prior transformation function
    def Prior(cube, ndim, nparams):
        
        ''' Tranforms the unit cube provided by MultiNest into the values
            of each free parameter used by the forward model. The various
            prior ranges for each paramter are defined in config.py.
        
        '''       
           
        #***** Draw PT profile parameters *****#
    
        # For isotherms, only have one parameter to deal with
        if (PT_profile == 'isotherm'):
            
            cube[0] = ((cube[0] * (prior_upper_PT[0] - prior_lower_PT[0])) + prior_lower_PT[0])
            
        # For Madhusudhan & Seager (2009) profile, loop over priors specified in config.py
        elif (PT_profile == 'Madhu'):
            
            for i in range(N_PT_params):   
                cube[i] = ((cube[i] * (prior_upper_PT[i] - prior_lower_PT[i])) + prior_lower_PT[i])
            
        # For temperature gradient profile, assign priors based on parameter name (also supports 2D and 3D models)
        elif (PT_profile == 'gradient'):
            
            for i in range(N_PT_params):   
                if ('Delta' in param_names[i]):
                    cube[i] = ((cube[i] * (prior_upper_PT[2] - prior_lower_PT[2])) + prior_lower_PT[2])
                elif ('deep' in param_names[i]):
                    cube[i] = ((cube[i] * (prior_upper_PT[1] - prior_lower_PT[1])) + prior_lower_PT[1])
                else:
                    cube[i] = ((cube[i] * (prior_upper_PT[0] - prior_lower_PT[0])) + prior_lower_PT[0])

        # Drawn R_p_ref parameter (same in all regions)
        cube[N_PT_params] = ((cube[N_PT_params] * (prior_upper_R_p_ref - prior_lower_R_p_ref)) + prior_lower_R_p_ref)
        
        #***** Draw composition parameters *****#
        
        # For uniform-in-the-logarithm priors on chemical abundances
        if (chemistry_prior == 'log-uniform'): 
                    
            for i in range(N_species_params):
                
                i_new = N_params_cumulative[1] + i
                
                # Find which chemical species this parameter represents
                for species_q in param_species:
                    if (species_q in param_names[i_new]):
                        species = species_q 
                
                # For 1D models, prior just given by mixing ratio prior range
                if (Atmosphere_dimension == 1):
                    
                    cube[i_new] = ((cube[i_new] * (prior_upper_X[0] - prior_lower_X[0])) + prior_lower_X[0])  
                
                # For 2D models, the prior range for 'Delta' parameters can change to satisfy mixing ratio priors
                elif (Atmosphere_dimension == 2):
                
                    # Mixing ratio gradient parameters dynamically update priors
                    if ('Delta' in param_names[i_new]):
                        prior_upper_delta_X = min(prior_upper_X[1], 2*(prior_upper_X[0] - cube[i_new-1]))  
                        prior_lower_delta_X = max(prior_lower_X[1], 2*(prior_lower_X[0] - cube[i_new-1]))
                        cube[i_new] = ((cube[i_new] * (prior_upper_delta_X - prior_lower_delta_X)) + prior_lower_delta_X) 
                    else:
                        cube[i_new] = ((cube[i_new] * (prior_upper_X[0] - prior_lower_X[0])) + prior_lower_X[0])  
                               
                # For 3D models, the prior ranges for 'Delta' parameters can change to satisfy mixing ratio priors
                elif (Atmosphere_dimension == 3):
                        
                    # For species with 3D gradients, sample such that highest and lowest values still satisfy mixing ratio prior
                    if ((species in species_EM_gradient) and (species in species_DN_gradient)):
                        
                        # Check that evening / morning / day night mixing ratios do not exceed mixing ratio priors
                        if (param_names[i_new] == ('Delta_log_' + species + '_term')):
                            prior_upper_delta_X = min(prior_upper_X[1], 2*(prior_upper_X[0] - cube[i_new-1])) 
                            prior_lower_delta_X = max(prior_lower_X[1], 2*(prior_lower_X[0] - cube[i_new-1]))
                            cube[i_new] = ((cube[i_new] * (prior_upper_delta_X - prior_lower_delta_X)) + prior_lower_delta_X) 
                        elif (param_names[i_new] == ('Delta_log_' + species + '_DN')):
                            prior_upper_delta_X = min(prior_upper_X[1], min(2*(prior_upper_X[0] - (cube[i_new-2] + 0.5*cube[i_new-1])),
                                                                            2*(prior_upper_X[0] - (cube[i_new-2] - 0.5*cube[i_new-1]))))
                            prior_lower_delta_X = max(prior_lower_X[1], max(2*(prior_lower_X[0] - (cube[i_new-2] + 0.5*cube[i_new-1])),
                                                                            2*(prior_lower_X[0] - (cube[i_new-2] - 0.5*cube[i_new-1]))))
                            cube[i_new] = ((cube[i_new] * (prior_upper_delta_X - prior_lower_delta_X)) + prior_lower_delta_X) 
                        else:
                            cube[i_new] = ((cube[i_new] * (prior_upper_X[0] - prior_lower_X[0])) + prior_lower_X[0])  

                    # Species with a 2D gradient within a 3D model reduce to the 2D case above
                    elif ('Delta' in param_names[i_new]):
                        
                        prior_upper_delta_X = min(prior_upper_X[1], 2*(prior_upper_X[0] - cube[i_new-1]))  
                        prior_lower_delta_X = max(prior_lower_X[1], 2*(prior_lower_X[0] - cube[i_new-1]))
                        cube[i_new] = ((cube[i_new] * (prior_upper_delta_X - prior_lower_delta_X)) + prior_lower_delta_X) 
                    
                    # Species with a uniform abundance reduce to the 1D case above
                    else:
                        cube[i_new] = ((cube[i_new] * (prior_upper_X[0] - prior_lower_X[0])) + prior_lower_X[0])  
                    
        # For centred log-ratio prior on chemical abundances
        elif (chemistry_prior == 'CLR'):  
                
            chem_drawn = np.array(cube[N_params_cumulative[1]:N_params_cumulative[2]])
            log_X = CLR_Prior(chem_drawn)                # Note: X[0] is not a free parameter, as total must sum to unity
                
            global allowed_simplex     # Needs to be global, as prior function has no return option
            
            if (log_X[1] == -50.0): 
                allowed_simplex = 0       # If mixing ratios outside allowed simplical space (X_i > 10^-12 and sum to 1)
            elif (log_X[1] != -50.0): 
                allowed_simplex = 1       # If satisfied, likelihood will be computed for this parameter combination
            
            for i in range(N_species_params):
                
                i_new = N_params_cumulative[1] + i
                
                # [0] is not a free parameter, as total must sum to unity
                cube[i_new] = log_X[(1+i)]   
                   
        #***** Draw cloud parameters *****#   
    
        # For MacDonald & Madhusudhan (2017) cloud model
        if (cloud_model == 'MacMad17'):
            
            for i in range(N_cloud_params):   
                    
                i_new = N_params_cumulative[2] + i
    
                # Assign appropriate parameter priors depending on specified cloud type (e.g. cloud only, haze only, etc.)
                if (param_names[i_new] == 'log_a'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[0] - prior_lower_clouds[0])) + prior_lower_clouds[0])
                elif (param_names[i_new] == 'gamma'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[1] - prior_lower_clouds[1])) + prior_lower_clouds[1])
                elif (param_names[i_new] == 'log_P_cloud'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[2] - prior_lower_clouds[2])) + prior_lower_clouds[2])
                elif (param_names[i_new] == 'phi_c'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[3] - prior_lower_clouds[3])) + prior_lower_clouds[3])
                elif (param_names[i_new] == 'phi_0'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[4] - prior_lower_clouds[4])) + prior_lower_clouds[4])
          
        # For 3D 'Iceberg' patchy cloud model from MacDonald & Lewis (2021)
        elif (cloud_model == 'Iceberg'):
            
            for i in range(N_cloud_params):   
                    
                i_new = N_params_cumulative[2] + i
    
                # Assign appropriate parameter priors depending on specified cloud type (e.g. cloud only, haze only, etc.)
                if (param_names[i_new] == 'log_kappa_cloud'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[0] - prior_lower_clouds[0])) + prior_lower_clouds[0])
                elif (param_names[i_new] == 'log_P_cloud'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[1] - prior_lower_clouds[1])) + prior_lower_clouds[1])
                elif (param_names[i_new] == 'f_cloud'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[2] - prior_lower_clouds[2])) + prior_lower_clouds[2])
                elif (param_names[i_new] == 'phi_0'):
                    cube[i_new] = ((cube[i_new] * (prior_upper_clouds[3] - prior_lower_clouds[3])) + prior_lower_clouds[3])
                elif (param_names[i_new] == 'theta_0'):
                    cube[i_new] = (180.0/np.pi)*np.arcsin((2.0*cube[i_new] - 1) * np.sin((np.pi/180.0)*(prior_upper_geometry[1]/2.0)))
                   # cube[i_new] = ((cube[i_new] * (prior_upper_clouds[4] - prior_lower_clouds[4])) + prior_lower_clouds[4])
        
        #***** Draw geometry parameters *****#   
        
        for i in range(N_geometry_params):
            
            i_new = N_params_cumulative[3] + i
            
            # Assign priors for alpha and beta, since either (or both) can be free depending on dimensionality of model atmosphere
            if (param_names[i_new] == 'alpha'):
                cube[i_new] = ((cube[i_new] * (prior_upper_geometry[0] - prior_lower_geometry[0])) + prior_lower_geometry[0])
            elif (param_names[i_new] == 'beta'):
                cube[i_new] = (180.0/np.pi)*2.0*np.arcsin(cube[i_new] * np.sin((np.pi/180.0)*(prior_upper_geometry[1]/2.0)))
                # cube[i_new] = ((cube[i_new] * (prior_upper_geometry[1] - prior_lower_geometry[1])) + prior_lower_geometry[1])
          
        #***** Draw stellar parameters *****#      
        
        # First deal with uniform priors (f_het, T_het)
        for i in range(N_stellar_params-1):     
            
            i_new = N_params_cumulative[4] + i
            cube[i_new] = ((cube[i_new] * (prior_upper_stellar[i] - prior_lower_stellar[i])) + prior_lower_stellar[i])
          
        if (stellar_contam != 'No'):
            
            # Second, deal with Gaussian prior on T_phot
            i_new = N_params_cumulative[5] - 1
            cube[i_new] = prior_gauss_T_phot[0] + prior_gauss_T_phot[1] * ndtri(cube[i_new])
        
        #***** Draw offset parameters *****#  
        
        for i in range(N_offset_params):
            
            i_new = N_params_cumulative[5] + i
            cube[i_new] = ((cube[i_new] * (prior_upper_offsets[i] - prior_lower_offsets[i])) + prior_lower_offsets[i]) 
              
        #***** Draw error adjustment parameters *****#  
        
        for i in range(N_error_params):
            
            i_new = N_params_cumulative[6] + i
            cube[i_new] = ((cube[i_new] * (prior_upper_err_inflation - prior_lower_err_inflation)) + prior_lower_err_inflation) 
            
            
    # Define the log-likelihood function
    def LogLike(cube, ndim, nparams):
        
        ''' Evaluates the log-likelihood for a given point in parameter space.
        
            Works by generating a PT profile, calculating the opacity in the
            model atmsophere, computing the resulting transmission spectrum and
            finally convolving and integrating the spectrum to produce model
            data points for each instrument.
           
            The log-likelihood is then evaluated using the difference between 
            the binned model and the actual data points. 
        
        '''
           
        # FIXING TO MID-TRANSIT FOR RETRIEVALS (for now...)
        y_p = 0.0
        
        # Immediately reject samples falling outside of simplex (mixing ratio) parameter space
        global allowed_simplex
        if (allowed_simplex == 0):
            loglikelihood = -1.0e100   
            return loglikelihood
        
        #***** Step 1: load parameter values from prior sample *****#
        
        PT_state, R_p_ref, \
        log_X_state, clouds, \
        geometry, stellar, \
        offsets, err_inflation = load_state(cube, param_names, 
                                            N_params_cumulative, param_species)
            
        #***** Step 2: compute forward model ****#
        
        if (spectrum_type == 'transmission'):
            
            # Compute transmission spectrum
            spectrum, ymodel, \
            P, T, r, X, mu, \
            is_physical = TRIDENT(wl, PT_state, R_p_ref, log_X_state, clouds, 
                                  geometry, stellar, offsets, sigma_stored, 
                                  cia_stored, Rayleigh_stored, eta_stored, 
                                  ff_stored, bf_stored, T_phot_grid, 
                                  I_phot_grid, T_het_grid, I_het_grid, 
                                  y_p, chemical_species, bulk_species, 
                                  param_species, active_species, cia_pairs,
                                  ff_pairs, bf_species, species_vert_gradient, 
                                  enable_haze, enable_deck, N_sectors, 
                                  N_zones, Atmosphere_dimension, param_names, 
                                  N_params_cumulative, data_properties)
        
      #  print(is_physical)
                                                                                         
        #***** Step 3: evalaute ln(likelihood) ****#
        
        # Check that given parameter combination is physical
        if (is_physical == False):
            
            # If model is not physical, assign massive penalty to likelihood => point ignored in retrieval
            loglikelihood = -1.0e100
            
            # Quit if given parameter combination is unphysical
            return loglikelihood
            
        # Compute effective error, if unknown systematics included
        err_data = data_properties['err_data']
        
        if (error_inflation == 'Line_2015'):
            err_eff_sq = (err_data*err_data + np.power(10.0, err_inflation))
            norm_log = (-0.5*np.log(2.0*np.pi*err_eff_sq)).sum()
        else: 
            err_eff_sq = err_data*err_data
            norm_log = norm_log_default
        
        # To check for relative systematics between data sets, add relative offset to one dataset's transit depth values
        ydata = data_properties['ydata']
        offset_start = data_properties['offset_start']
        offset_end = data_properties['offset_end']

        if (offsets_applied == 'relative'): 
            ydata_adjusted = ydata.copy()
            ydata_adjusted[offset_start:offset_end] += offsets[0]
        else: 
            ydata_adjusted = ydata
    
        # Compute log-likelihood for given parameter combination
        loglikelihood = (-0.5*((ymodel - ydata_adjusted)*(ymodel - ydata_adjusted))/err_eff_sq).sum()
        loglikelihood += norm_log
                    
        return loglikelihood
    
        
    # Assign prior and loglikelihood functions into PyMultiNest key word arguments
    kwargs['Prior'] = Prior
    kwargs['LogLikelihood'] = LogLike
    
    # Run PyMultiNest
    pymultinest.run(**kwargs)
	
    		
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
                                
