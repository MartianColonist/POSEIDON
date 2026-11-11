import pytest


def test_blackbody_star():
    '''
    Test POSEIDON's black body stellar spectrum against the analytic Planck
    function and the Stefan-Boltzmann law for the bolometric surface flux.

    '''

    from POSEIDON.constants import R_Sun
    from POSEIDON.core import create_star, wl_grid_constant_R
    from POSEIDON.utility import trapezoid
    import scipy.constants as sc
    import numpy as np

    #***** Define stellar properties *****#

    R_s = R_Sun      # Stellar radius (m)
    T_s = 5778.0     # Stellar effective temperature (K)
    Met_s = 0.0      # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.44   # Stellar log surface gravity (log10(cm/s^2) by convention)

    #***** Wavelength grid *****#

    # Wide enough to contain essentially all of the bolometric flux
    wl_min = 0.05     # Minimum wavelength (μm)
    wl_max = 1000.0   # Maximum wavelength (μm)
    R = 1000          # Spectral resolution of grid

    wl = wl_grid_constant_R(wl_min, wl_max, R)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid = 'blackbody',
                       wl = wl)

    I_phot = star['I_phot']
    F_star = star['F_star']

    #***** Analytic calculations *****#

    # Planck function spectral radiance (W/m^2/sr/m)
    wl_m = wl * 1.0e-6
    B_lambda = ((2.0 * sc.h * sc.c**2) / wl_m**5) / \
               (np.exp((sc.h * sc.c) / (wl_m * sc.k * T_s)) - 1.0)

    # Check intensity matches the Planck function to < 1e-10 relative
    relative_diffs = np.abs(I_phot / B_lambda - 1.0)

    print('Max relative difference from Planck function = ' +
          str(np.max(relative_diffs)))

    assert np.all(relative_diffs < 1.0e-10)

    # Check the surface flux of a uniform star is pi times the intensity
    assert np.allclose(F_star, np.pi * I_phot, rtol = 1.0e-12, atol = 0.0)

    # Integrate the surface flux over wavelength to obtain the bolometric flux
    F_bol = trapezoid(F_star, wl_m)

    # Check against the Stefan-Boltzmann law to < 0.01%
    F_bol_analytic = sc.Stefan_Boltzmann * T_s**4
    relative_diff_bol = np.abs(F_bol / F_bol_analytic - 1.0)

    print('Relative difference from Stefan-Boltzmann law = ' +
          str(relative_diff_bol))

    assert (relative_diff_bol < 1.0e-4)

    print("Black body star test passed!")
