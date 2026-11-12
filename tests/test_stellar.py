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


def test_pymsg_grid_splicing():
    '''
    Test that load_stellar_pymsg combines several PyMSG grids correctly (as
    for stellar_grid = 'newera'), using mock grids so neither PyMSG nor the
    grid files are needed. Bins fully inside the first grid (NewEra-LowRes,
    0.25-2.5 μm) must use it, later bins up to 28.5 μm the second grid
    (NewEra-JWST), and bins outside both grids a black body.

    '''

    from POSEIDON.core import wl_grid_constant_R
    from POSEIDON.stellar import load_stellar_pymsg, planck_lambda
    import numpy as np

    #***** Define mock PyMSG grids *****#

    class MockSpecGrid:
        '''
        Minimal stand-in for a PyMSG SpecGrid, returning a constant flux.

        '''

        def __init__(self, lam_min, lam_max, flux_value):

            self.lam_min = lam_min   # Å
            self.lam_max = lam_max   # Å
            self.flux_value = flux_value
            self.axis_labels = ['Teff', '[Fe/H]', 'log(g)']

        def flux(self, x, deriv, lam_edges):

            # Check the requested wavelengths lie within the grid
            assert (lam_edges[0] >= self.lam_min)
            assert (lam_edges[-1] <= self.lam_max)

            return self.flux_value * np.ones(len(lam_edges) - 1)

    grid_LowRes = MockSpecGrid(2500.0, 25000.0, 1.0)
    grid_JWST = MockSpecGrid(6000.0, 285000.0, 2.0)

    #***** Define stellar properties *****#

    T_s = 4500.0     # Stellar effective temperature (K)
    Met_s = 0.0      # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.5    # Stellar log surface gravity (log10(cm/s^2) by convention)

    #***** Wavelength grid *****#

    wl_min = 0.2      # Minimum wavelength (μm)
    wl_max = 30.0     # Maximum wavelength (μm)
    R = 1000          # Spectral resolution of grid

    wl = wl_grid_constant_R(wl_min, wl_max, R)

    # Bin edges (as computed in load_stellar_pymsg)
    wl_edges = np.zeros(len(wl)+1)
    wl_edges[0] = wl[0] - 0.5*(wl[1] - wl[0])
    wl_edges[-1] = wl[-1] + 0.5*(wl[-1] - wl[-2])
    wl_edges[1:-1] = 0.5*(wl[1:] + wl[:-1])

    wl_lower = wl_edges[:-1]
    wl_upper = wl_edges[1:]

    #***** Combined grids *****#

    I_grid = load_stellar_pymsg(wl, [grid_LowRes, grid_JWST], T_s, Met_s,
                                log_g_s, 'newera')

    I_LowRes = 1.0e7 / np.pi   # Mock flux converted to W/m^2/sr/m
    I_JWST = 2.0e7 / np.pi
    B_lambda = planck_lambda(T_s, wl)

    # Expected source of each bin
    in_LowRes = (wl_lower >= 0.25) & (wl_upper <= 2.5)
    in_JWST = (~in_LowRes) & (wl_lower >= 0.6) & (wl_upper <= 28.5)
    in_BB = (~in_LowRes) & (~in_JWST)

    # Check every region is present in this test
    assert (np.any(in_LowRes) and np.any(in_JWST) and np.any(in_BB))

    # The bin straddling 2.5 μm must come from NewEra-JWST
    i_join = np.where((wl_lower < 2.5) & (wl_upper > 2.5))[0][0]
    assert in_JWST[i_join]

    assert np.all(I_grid[in_LowRes] == I_LowRes)
    assert np.all(I_grid[in_JWST] == I_JWST)
    assert np.allclose(I_grid[in_BB], B_lambda[in_BB], rtol = 1.0e-12,
                       atol = 0.0)

    # Black body only below 0.25 μm and above 28.5 μm
    assert np.all((wl[in_BB] < 0.25) | (wl[in_BB] > 28.5))

    #***** Single grid *****#

    # A single grid object gives the same result as a one element list
    I_single = load_stellar_pymsg(wl, grid_JWST, T_s, Met_s, log_g_s,
                                  'newera-jwst')
    I_list = load_stellar_pymsg(wl, [grid_JWST], T_s, Met_s, log_g_s,
                                'newera-jwst')

    assert np.array_equal(I_single, I_list)

    in_JWST_only = (wl_lower >= 0.6) & (wl_upper <= 28.5)
    assert np.all(I_single[in_JWST_only] == I_JWST)
    assert np.allclose(I_single[~in_JWST_only], B_lambda[~in_JWST_only],
                       rtol = 1.0e-12, atol = 0.0)

    print("PyMSG grid splicing test passed!")
