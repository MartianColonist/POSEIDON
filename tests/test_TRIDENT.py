import pytest


def test_Rayleigh():
    '''
    Test POSEIDON's forward model, TRIDENT, against an analytic expression
    for a pure H2 Rayleigh scattering atmosphere.
    
    '''

    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import create_star, create_planet, define_model, \
                              make_atmosphere, read_opacities, compute_spectrum, \
                              wl_grid_constant_R
    from POSEIDON.absorption import Rayleigh_cross_section
    import scipy.constants as sc
    from scipy.special import expn
    import numpy as np

    #***** Define stellar properties *****#

    R_s = R_Sun      # Stellar radius (m)
    T_s = 5000.0     # Stellar effective temperature (K)
    Met_s = 0.0      # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.0    # Stellar log surface gravity (log10(cm/s^2) by convention)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s)

    #***** Define planet properties *****#

    planet_name = 'Example Planet'  # Planet name used for plots, output files etc.

    R_p = R_J         # Planetary radius (m)
    M_p = M_J         # Planet mass (kg)
    T_eq = 1000.0     # Equilibrium temperature (K)

    # Create the planet object
    planet = create_planet(planet_name, R_p, mass = M_p, T_eq = T_eq)

    #***** Define model *****#

    model_name = 'Only_Rayleigh'  # Model name used for plots, output files etc.

    bulk_species = ['H2']      # Only H2
    param_species = []         # No other gases for this test

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
    P_ref = 100.0   # Reference pressure (bar)
    R_p_ref = R_p   # Radius at reference pressure

    # Provide a specific set of model parameters for the atmosphere 
    PT_params = np.array([T_eq])         
    log_X_params = np.array([])

    # Generate the atmosphere
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                PT_params, log_X_params,
                                constant_gravity = True)

    #***** Wavelength grid *****#
    
    wl_min = 0.2      # Minimum wavelength (um)
    wl_max = 10.0     # Maximum wavelength (um)
    R = 10000         # Spectral resolution of grid

    wl = wl_grid_constant_R(wl_min, wl_max, R)

    #***** Read opacity data *****#

    opacity_treatment = 'opacity_sampling'

    # Define fine temperature grid (K)
    T_fine_min = 900
    T_fine_max = 1100
    T_fine_step = 10

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -6.0
    log_P_fine_max = 2.0
    log_P_fine_step = 0.2

    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                        log_P_fine_step)

    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine,
                          testing = True)

    # Remove H2-H2 CIA for this test
    opac['CIA_stored'] *= 0.0

    # Calculate numerical spectrum with POSEIDON
    spectrum_TRIDENT = compute_spectrum(planet, star, model, atmosphere, opac,
                                        wl, spectrum_type = 'transmission')

    #***** Analytic calculation *****#

    # Using the analytic expression for an atmosphere with constant scale height,
    # constant cross section with altitude, and H/Rp << 1

    # Load H2 Rayleigh scattering cross section from POSEIDON 
    sigma_H2, _ = Rayleigh_cross_section(wl, 'H2')

    # Load mean molecular mass from POSEIDON
    mu = atmosphere['mu'][0,0,0]

    # Calculate required quantities for analytic expression
    g = sc.G * M_p / R_p**2
    H = sc.k * T_eq / (mu * g)
    gamma = 0.5772156649
    kappa = sigma_H2 / mu

    # Surface pressure and radius
    P_surface = 1e7                            # 100 bar in Pa
    R_surface = atmosphere['r_low'][0,0,0]     # Bottom of model atmosphere

    # Calculate equivalent surface optical depth
    tau_surface = P_surface/g * np.sqrt(2*np.pi*R_surface/H) * kappa

    # Calculate effective planet radius as function of wavelength
    analytic_R = R_surface + H*(gamma + np.log(tau_surface) + expn(1, tau_surface))

    # Convert units to transit depth
    spectrum_analytic = analytic_R**2 / R_s**2

    # Check relative difference between solutions < 0.01%
    ratios = spectrum_analytic  / spectrum_TRIDENT
    relative_diffs = np.abs(ratios - 1)

    print('Max relative difference = ' + str(round(np.max(relative_diffs*1e2),3)) + ' %')

    assert np.all(relative_diffs < 0.0001)

    # Check absolute difference between solutions < 1 ppm
    diffs = np.abs(spectrum_analytic - spectrum_TRIDENT)

    print('Max difference = ' + str(round(np.max(diffs*1e6),3)) + ' ppm')

    assert np.all(diffs < 1.0e-6)

    print("Forward model test passed - TRIDENT is ready for action!")


test_Rayleigh()