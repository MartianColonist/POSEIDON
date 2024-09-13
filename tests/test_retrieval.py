import pytest


def test_continuum_retrieval():
    '''
    Test POSEIDON's retrieval functionality on a synthetic WASP-121b 
    transmission spectrum only including H-, H2-H2 and H2-He CIA, and Rayleigh
    scattering (i.e. a continuum-only retrieval).
    
    '''

    from POSEIDON.constants import R_Sun, R_J
    from POSEIDON.core import create_star, create_planet, define_model, \
                            make_atmosphere, read_opacities, wl_grid_constant_R, \
                            compute_spectrum, load_data, set_priors
    from POSEIDON.instrument import generate_syn_data_from_user
    from POSEIDON.visuals import plot_spectra, plot_spectra_retrieved
    from POSEIDON.utility import plot_collection, read_retrieved_spectrum
    from POSEIDON.retrieval import run_retrieval
    from POSEIDON.corner import generate_cornerplot
    import scipy.constants as sc
    import pymultinest
    import numpy as np
    import os

    #***** Define stellar properties *****#

    R_s = 1.46*R_Sun      # Stellar radius (m)
    T_s = 6776.0          # Stellar effective temperature (K)
    Met_s = 0.13          # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.24        # Stellar log surface gravity (log10(cm/s^2) by convention)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s)

    #***** Define planet properties *****#

    planet_name = 'WASP-121b'  # Planet name used for plots, output files etc.

    R_p = 1.753*R_J      # Planetary radius (m)
    log_g_p = 2.97       # Gravitational field of planet (cgs)
    T_eq = 2450          # Equilibrium temperature (K)

    # Create the planet object
    planet = create_planet(planet_name, R_p, log_g = log_g_p, T_eq = T_eq)

    #***** Model wavelength grid *****#

    wl_min = 0.4    # Minimum wavelength (um)
    wl_max = 1.8    # Maximum wavelength (um)
    R = 1000   # We can get away with R = 1k for this test, since we only have continuum opacity

    # We need to provide a model wavelength grid to initialise instrument properties
    wl = wl_grid_constant_R(wl_min, wl_max, R)

    #***** Define model *****#

    model_name = 'H-_retrieval_test'

    bulk_species = ['H2', 'He']
    param_species = ['H-']

    # Create the model object
    model = define_model(model_name, bulk_species, param_species,
                         PT_profile = 'isotherm')

    # Specify the pressure grid of the atmosphere
    P_min = 1.0e-7    # 0.1 ubar
    P_max = 100       # 100 bar
    N_layers = 100    # 100 layers

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure and radius
    P_ref = 10.0   # Reference pressure (bar)
    R_p_ref = R_p   # Radius at reference pressure

    # Provide a specific set of model parameters for the atmosphere 
    PT_params = np.array([T_eq])         
    log_X_params = np.array([-9.0])

    # Generate the atmosphere
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                PT_params, log_X_params)

    #***** Read opacity data *****#

    opacity_treatment = 'opacity_sampling'

    # Define fine temperature grid (K)
    T_fine_min = 800
    T_fine_max = 3000
    T_fine_step = 10

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -6.0
    log_P_fine_max = 2.0
    log_P_fine_step = 0.2

    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                        log_P_fine_step)

    # Read cross sections
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine,
                        testing = True)

    # Generate transmission spectrum
    spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                spectrum_type = 'transmission')

    #***** Generate synthetic data *****#

    os.mkdir('./data')
    os.mkdir('./data/WASP-121b')
    
    data_dir = './data/WASP-121b'

    generate_syn_data_from_user(planet, wl, spectrum, data_dir, instrument = 'dummy',
                                R_data = 30, err_data = 50, wl_start = 0.45, 
                                wl_end = 1.7, Gauss_scatter = False)

    # Load synthetic data file
    datasets = ['WASP-121b_SYNTHETIC_dummy.dat']
    instruments = ['dummy']

    # Load dataset, pre-load instrument PSF and transmission function
    data = load_data(data_dir, datasets, instruments, wl, wl_unit = 'micron',
                     bin_width = 'half', spectrum_unit = 'transit_depth', skiprows = None)

    #***** Set priors for retrieval *****#

    # Initialise prior type dictionary
    prior_types = {}

    # Specify whether priors are linear, Gaussian, etc.
    prior_types['T'] = 'uniform'
    prior_types['R_p_ref'] = 'uniform'
    prior_types['log_H-'] = 'uniform'

    # Initialise prior range dictionary
    prior_ranges = {}

    # Specify prior ranges for each free parameter
    prior_ranges['T'] = [800, 3000]
    prior_ranges['R_p_ref'] = [0.85*R_p, 1.15*R_p]
    prior_ranges['log_H2O'] = [-14, -2]

    # Create prior object for retrieval
    priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

    #***** Run atmospheric retrieval *****#

    run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R = R, 
                  spectrum_type = 'transmission', sampling_algorithm = 'MultiNest', 
                  N_live = 400, verbose = True)

    #***** Read MultiNest retrieval results *****#

    true_vals = [1.753, T_eq, log_X_params[0]]

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
        true_median = true_vals[i]

        m = stats['marginals'][i]
        retrieved_median = m['median']

        # Check relative difference between solutions < 1%
        relative_diff = np.abs((retrieved_median - true_median)/true_median)

        print('Relative diff for ' + parameter + ' = ' + 
                str(round(np.max(relative_diff*1e2),3)) + ' %')

        assert relative_diff < 0.01

    os.chdir('../../../../')

    print("Retrieval test passed!")


#test_continuum_retrieval()