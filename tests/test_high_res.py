import numpy as np


def test_cross_correlate_recovers_transmission_signal():
    '''
    Test the high-resolution cross-correlation function by injecting a
    Doppler-shifted synthetic absorption line into noiseless residuals and
    checking that the CCF peak recovers the injected Kp and Vsys.

    '''

    from POSEIDON.high_res import cross_correlate

    #***** Define synthetic planet spectrum *****#

    # Wavelength grid (only relative Doppler shifts matter here)
    wl = np.linspace(5000.0, 5100.0, 200)

    # Single box-shaped absorption line
    line = np.zeros_like(wl)
    line[np.abs(wl - 5050.0) < 2.0] = 1.0

    # Transmission spectra are transit depths, so absorption raises the depth
    planet_spectrum = 0.2 * line

    #***** Define planet orbital properties *****#

    phi = np.linspace(-0.05, 0.05, 9)   # Orbital phases
    Kp_true = 100.0                     # km/s
    Vsys_true = 20.0                    # km/s

    #***** Generate synthetic residuals *****#

    residuals = np.ones((1, len(phi), len(wl)))

    # Doppler shift the planet spectrum at each phase
    for i, phase in enumerate(phi):

        RV = Kp_true * np.sin(2.0 * np.pi * phase) + Vsys_true
        wl_shifted = wl * (1.0 + RV / 3.0e5)

        # Absorption appears as a dip in the flux residuals
        residuals[0, i] = -np.interp(wl, wl_shifted, planet_spectrum)

    # Package into a high-resolution data dictionary
    data = {
        'residuals': residuals,
        'uncertainties': np.ones_like(residuals),
        'phi': phi,
        'wl_grid': np.tile(wl[None, :], (1, 1)),
        'V_bary': np.zeros_like(phi),
        'transit_weight': np.full_like(phi, 0.99),
    }

    #***** Cross correlate *****#

    Kp_range = np.arange(0.0, 201.0, 5.0)        # km/s
    Vsys_range = np.arange(-50.0, 51.0, 5.0)     # km/s
    RV_range = np.arange(-250.0, 251.0, 1.0)     # km/s

    CCF_Kp_Vsys, _ = cross_correlate(Kp_range, Vsys_range, RV_range, wl,
                                     planet_spectrum, data, Print = False)

    # Locate the CCF peak
    peak_idx = np.unravel_index(np.argmax(CCF_Kp_Vsys), CCF_Kp_Vsys.shape)

    # The broad line and narrow phase range limit the Kp precision
    assert abs(Kp_range[peak_idx[0]] - Kp_true) < 20.0
    assert abs(Vsys_range[peak_idx[1]] - Vsys_true) < 20.0
