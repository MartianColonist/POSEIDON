import pytest


def test_retrieval():
    '''
    Test POSEIDON's retrieval functionality by running the 'retrieval_basic'
    tutorial and checking the median parameters are within 5% of expectations.
    
    '''

    from POSEIDON.constants import R_Sun, R_J
    from POSEIDON.core import create_star, create_planet, define_model, \
                              read_opacities, wl_grid_constant_R, \
                              load_data, set_priors
    from POSEIDON.retrieval import run_retrieval
    import pymultinest
    import numpy as np
    import os

    #***** Define stellar properties *****#

    R_s = 1.155*R_Sun     # Stellar radius (m)
    T_s = 6071.0          # Stellar effective temperature (K)
    Met_s = 0.0           # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.38        # Stellar log surface gravity (log10(cm/s^2) by convention)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s)

    #***** Define planet properties *****#

    planet_name = 'WASP-999b'  # Planet name used for plots, output files etc.

    R_p = 1.359*R_J     # Planetary radius (m)
    g_p = 9.186         # Gravitational field of planet (m/s^2)
    T_eq = 1400.0       # Equilibrium temperature (K)

    # Create the planet object
    planet = create_planet(planet_name, R_p, gravity = g_p, T_eq = T_eq)

    #***** Model wavelength grid *****#

    wl_min = 0.4      # Minimum wavelength (um)
    wl_max = 1.8      # Maximum wavelength (um)
    R = 4000          # Spectral resolution of grid      

    # We need to provide a model wavelength grid to initialise instrument properties
    wl = wl_grid_constant_R(wl_min, wl_max, R)

    #***** Specify data location and instruments  *****#

    # Specify the STIS and WFC3 Hubble data
    data_dir = 'Tutorial/WASP-999b'                   
    datasets = ['WASP-999b_STIS_G430.dat', 
                'WASP-999b_STIS_G750.dat', 
                'WASP-999b_WFC3_G141.dat']  
    instruments = ['STIS_G430', 'STIS_G750', 'WFC3_G141']

    # Load dataset, pre-load instrument PSF and transmission function
    data = load_data(data_dir, datasets, instruments, wl)

    #***** Define model *****#

    model_name = 'My_first_retrieval'  # Model name used for plots, output files etc.

    bulk_species = ['H2', 'He']     # H2 + He comprises the bulk atmosphere
    param_species = ['H2O']         # The only trace gas is H2O

    # Create the model object
    model = define_model(model_name, bulk_species, param_species, 
                         PT_profile = 'isotherm', cloud_model = 'cloud-free')

    #***** Set priors for retrieval *****#

    # Initialise prior type dictionary
    prior_types = {}

    # Specify whether priors are linear, Gaussian, etc.
    prior_types['T'] = 'uniform'
    prior_types['R_p_ref'] = 'uniform'
    prior_types['log_H2O'] = 'uniform'

    # Initialise prior range dictionary
    prior_ranges = {}

    # Specify prior ranges for each free parameter
    prior_ranges['T'] = [400, 1600]
    prior_ranges['R_p_ref'] = [0.85*R_p, 1.15*R_p]
    prior_ranges['log_H2O'] = [-12, -1]

    # Create prior object for retrieval
    priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

    #***** Read opacity data *****#

    opacity_treatment = 'opacity_sampling'

    # Define fine temperature grid (K)
    T_fine_min = 400     # Same as prior range for T
    T_fine_max = 1600    # Same as prior range for T
    T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
    log_P_fine_max = 2.0    # 100 bar is the highest pressure in the opacity database
    log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                        log_P_fine_step)

    # Pre-interpolate the opacities
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

    #***** Specify fixed atmospheric settings for retrieval *****#

    # Atmospheric pressure grid
    P_min = 1.0e-7    # 0.1 ubar
    P_max = 100       # 100 bar
    N_layers = 100    # 100 layers

    # Let's space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure
    P_ref = 10.0   # Retrieved R_p_ref parameter will be the radius at 10 bar

    #***** Run atmospheric retrieval *****#

    run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R = R, 
                  spectrum_type = 'transmission', sampling_algorithm = 'MultiNest', 
                  N_live = 400, verbose = True)

    #***** Read MultiNest retrieval results *****#

    true_medians = [1.316, 1317, -4.57]  # Expected results from a 2,000 live point retrieval

    # Change directory into MultiNest result file folder
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'
    os.chdir(output_dir + 'MultiNest_raw/')

    n_params = len(model['param_names'])

    # Run PyMultiNest analyser to extract posterior samples and model evidence
    analyzer = pymultinest.Analyzer(n_params, verbose = False,
                                    outputfiles_basename = model_name + '-')

    stats = analyzer.get_stats()

    # Load retrieved median values for the planet radius, temperature, and log(H2O)
    for i in range(n_params):

        parameter = model['param_names'][i]
        true_median = true_medians[i]

        m = stats['marginals'][i]
        retrieved_median = m['median']

        # Check relative difference between solutions < 0.5%
        relative_diff = np.abs((retrieved_median - true_median)/true_median)

        print('Relative diff for ' + parameter + ' = ' + 
              str(round(np.max(relative_diff*1e2),3)) + ' %')

        assert relative_diff < 0.005

    print("Retrieval test passed!")


test_retrieval()