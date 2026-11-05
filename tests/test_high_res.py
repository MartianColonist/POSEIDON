import numpy as np

from POSEIDON.high_res import cross_correlate


def test_cross_correlate_recovers_transmission_signal():
    wl = np.linspace(5000.0, 5100.0, 200)
    line = np.zeros_like(wl)
    line[np.abs(wl - 5050.0) < 2.0] = 1.0
    planet_spectrum = 1.0 - 0.2 * line

    phi = np.linspace(-0.05, 0.05, 9)
    kp_true = 100.0
    vsys_true = 20.0

    residuals = np.ones((1, len(phi), len(wl)))
    for i, phase in enumerate(phi):
        rv = kp_true * np.sin(2 * np.pi * phase) + vsys_true
        wl_shifted = wl * (1.0 + rv / 3e5)
        residuals[0, i] = np.interp(wl, wl_shifted, planet_spectrum)

    data = {
        "residuals": residuals,
        "uncertainties": np.ones_like(residuals),
        "phi": phi,
        "wl_grid": np.tile(wl[None, :], (1, 1)),
        "V_bary": np.zeros_like(phi),
        "transit_weight": np.zeros_like(phi),
    }

    kp_range = np.arange(0.0, 201.0, 5.0)
    vsys_range = np.arange(-50.0, 51.0, 5.0)
    rv_range = np.arange(-250.0, 251.0, 1.0)

    ccf, _ = cross_correlate(kp_range, vsys_range, rv_range, wl, planet_spectrum, data, Print=False)
    peak_idx = np.unravel_index(np.argmax(ccf), ccf.shape)

    assert abs(kp_range[peak_idx[0]] - kp_true) < 20.0
    assert abs(vsys_range[peak_idx[1]] - vsys_true) < 20.0
