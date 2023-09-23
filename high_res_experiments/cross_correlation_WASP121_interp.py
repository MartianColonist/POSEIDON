from __future__ import absolute_import, unicode_literals, print_function
from POSEIDON.high_res import loglikelihood_sysrem
import math, os
import numpy as np
import pickle
import pickle
from scipy import constants
from numba import jit
from astropy.io import fits
from scipy import interpolate
from POSEIDON.core import (
    create_star,
    create_planet,
    define_model,
    make_atmosphere,
    read_opacities,
    wl_grid_constant_R,
    compute_spectrum,
)
from POSEIDON.constants import R_Sun, R_J, M_J
import numpy as np
from spectres import spectres
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load
import time
from scipy.ndimage import gaussian_filter1d, median_filter
from POSEIDON.utility import read_high_res_data
import multiprocessing

K_p = -200
N_K_p = 200
d_K_p = 2
K_p_arr = (
    np.arange(N_K_p) - (N_K_p - 1) // 2
) * d_K_p + K_p  # making K_p_arr (centered on published or predicted K_p)
# K_p_arr = [92.06 , ..., 191.06, 192.06, 193.06, ..., 291.06]

V_sys = 0
N_V_sys = 200
d_V_sys = 2
V_sys_arr = (
    np.arange(N_V_sys) - (N_V_sys - 1) // 2
) * d_V_sys + V_sys  # making V_sys_arr (centered on published or predicted V_sys (here 0 because we already added V_sys in V_bary))


def cross_correlate(Kp_arr, Vsys_arr, wl, planet_spectrum, data):
    a = 10
    b = 1
    uncertainties = data["uncertainties"]
    residuals = data["residuals"]
    phi = data["phi"]
    Bs = data["Bs"]
    wl_grid = data["wl_grid"]
    transit_weight = data["transit_weight"]
    max_transit_depth = np.max(1 - transit_weight)

    N_order, N_phi, N_wl = residuals.shape
    loglikelihood_array_final = np.zeros((len(Kp_arr), len(Vsys_arr)))
    CCF_array_final = np.zeros((len(Kp_arr), len(Vsys_arr)))

    RV_min = min(
        [
            np.min(Kp_arr * np.sin(2 * np.pi * phi[i])) + np.min(Vsys_arr)
            for i in range(len(phi))
        ]
    )
    RV_max = max(
        [
            np.max(Kp_arr * np.sin(2 * np.pi * phi[i])) + np.max(Vsys_arr)
            for i in range(len(phi))
        ]
    )

    RV_range = np.arange(RV_min, RV_max + 1)
    CCF_array_per_phase = np.zeros((len(phi), len(RV_range)))
    models_shifted = np.zeros((len(RV_range), N_order, N_wl))
    for i, RV in enumerate(RV_range):
        # Looping through each order and computing total log-L by summing logLs for each obvservation/order
        for j in range(N_order):
            wl_slice = wl_grid[j]  # Cropped wavelengths
            delta_lambda = RV * 1e3 / constants.c
            wl_shifted = wl * (1.0 + delta_lambda)
            # wl_shifted_p = wl_slice * np.sqrt((1.0 - dl_p[j]) / (1 + dl_p[j]))
            F_p = np.interp(wl_slice, wl_shifted, planet_spectrum)
            # Fp = interpolate.splev(wl_shifted_p, cs_p, der=0) # linear interpolation, einsum
            models_shifted[i, j] = F_p  # choose not to filter

    for i in range(N_phi):
        print(i)
        loglikelihoods = np.zeros_like(RV_range)
        CCFs = np.zeros_like(RV_range)
        for k in range(N_order):
            f = residuals[k, i] / uncertainties[k, i]
            f2 = f.dot(f)
            for j, RV in enumerate(RV_range):
                # model = (-models_shifted[j, k]) * (
                #     1 - transit_weight[i]
                # ) / max_transit_depth + 1  # change to -model instead?
                model = -models_shifted[j, k]
                # divide by the median over wavelength to mimic blaze correction
                # model = model / np.median(model)  # keep or not?

                m = model / uncertainties[k, i] * a
                m2 = m.dot(m)
                CCF = f.dot(m)
                # loglikelihood = -N_wl / 2 * np.log((m2 + f2 - 2.0 * CCF) / N_wl)
                loglikelihood = -0.5 * (m2 + f2 - 2.0 * CCF) / (b**2)
                loglikelihoods[j] += loglikelihood

                CCFs[j] += CCF
        CCF_array_per_phase[i, :] = CCFs
        for j, Kp in enumerate(Kp_arr):
            RV = Kp * np.sin(2 * np.pi * phi[i]) + Vsys_arr
            loglikelihood_array_final[j] += np.interp(RV, RV_range, loglikelihoods)
            CCF_array_final[j] += np.interp(RV, RV_range, CCFs)
    cross_correlation_result = [
        Kp_arr,
        Vsys_arr,
        RV_range,
        loglikelihood_array_final,
        CCF_array_final,
        CCF_array_per_phase,
    ]

    return cross_correlation_result


# The code below will only be run on one core to get the model spectrum.
# if __name__ == "__main__":
R_s = 1.458 * R_Sun  # Stellar radius (m)
T_s = 6776  # Stellar effectsive temperature (K)
Met_s = 0.13  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.24  # Stellar log surface gravity (log10(cm/s^2) by convention)

# ***** Define planet properties *****#

planet_name = "WASP-121b"  # Planet name used for plots, output files etc.

R_p = 1.753 * R_J  # Planetary radius (m)
M_p = 1.157 * M_J  # Mass of planet (kg)
g_p = 10 ** (2.97 - 2)  # Gravitational field of planet (m/s^2)
T_eq = 2450  # Equilibrium temperature (K)

# Create the planet object
planet = create_planet(planet_name, R_p, mass=M_p, gravity=g_p, T_eq=T_eq)

# If distance not specified, use fiducial value
if planet["system_distance"] is None:
    planet["system_distance"] = 1  # This value only used for flux ratios, so it cancels
d = planet["system_distance"]

# ***** Define model *****#

model_name = "Injection-retrieval-Fe-6"  # Model name used for plots, output files etc.
bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere

for species in ["Fe", "Li", "Mg", "Ti"]:
    param_species = [species]

    model = define_model(model_name, bulk_species, param_species, PT_profile="isotherm")

    # Check the free parameters defining this model
    print("Free parameters: " + str(model["param_names"]))

    # ***** Wavelength grid *****#

    wl_min = 0.37  # Minimum wavelength (um) 0.37
    wl_max = 0.51  # Maximum wavelength (um) 1.05
    R = 250000  # Spectral resolution of grid
    model["R"] = R
    model["R_instrument"] = 66000  # Resolution of instrument

    wl = wl_grid_constant_R(wl_min, wl_max, R)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="phoenix")

    # ***** Read opacity data *****#

    opacity_treatment = "opacity_sampling"

    # Define fine temperature grid (K)
    T_fine_min = 400  # 400 K lower limit suffices for a typical hot Jupiter
    T_fine_max = 3500  # 2000 K upper limit suffices for a typical hot Jupiter
    T_fine_step = 20  # 20 K steps are a good tradeoff between accuracy and RAM

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -12.0  # 1 ubar is the lowest pressure in the opacity database
    log_P_fine_max = 2  # 100 bar is the highest pressure in the opacity database
    log_P_fine_step = 0.2  # 0.2 dex steps are a good tradeoff between accuracy and RAM

    log_P_fine = np.arange(
        log_P_fine_min, (log_P_fine_max + log_P_fine_step), log_P_fine_step
    )

    # Now we can pre-interpolate the sampled opacities (may take up to a minute)
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

    # Specify the pressure grid of the atmosphere
    P_min = 1.0e-12  # 0.1 ubar
    P_max = 100  # 100 bar
    N_layers = 100  # 100 layers

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure and radius
    P_ref = 1e-5  # Reference pressure (bar)
    R_p_ref = R_p  # Radius at reference pressure

    params = (-6, 2500)
    log_species, T = params

    # Provide a specific set of model parameters for the atmosphere
    PT_params = np.array([T])  # a1, a2, log_P1, log_P2, log_P3, T_deep
    log_X_params = np.array([[log_species]])

    atmosphere = make_atmosphere(
        planet, model, P, P_ref, R_p_ref, PT_params, log_X_params
    )

    # Generate planet surface flux
    spectrum = compute_spectrum(
        planet,
        star,
        model,
        atmosphere,
        opac,
        wl,
        spectrum_type="transmission",
    )

    # param_species = []

    # # Create the model object
    # model = define_model(model_name, bulk_species, param_species, PT_profile="isotherm")

    # # Provide a specific set of model parameters for the atmosphere
    # PT_params = np.array([T])
    # log_X_params = np.array([[log_Fe]])

    # atmosphere = make_atmosphere(
    #     planet, model, P, P_ref, R_p_ref, PT_params, log_X_params
    # )

    # # Generate planet surface flux
    # continuum = compute_spectrum(
    #     planet, star, model, atmosphere, opac, wl, spectrum_type="transmission"
    # )

    # Passing stellar spectrum, planet spectrum, wavelenght grid to each core, thus saving time for reading the opacity again
    data_path = f"./data/WASP-121b/"
    output_path = f"./CC_output/WASP-121b/"

    # data_path = f"./data/WASP-121b-injection/"
    # output_path = f"./CC_output/WASP-121b-injection/"

    os.makedirs(output_path, exist_ok=True)
    data = read_high_res_data(data_path, method="sysrem", spectrum_type="transmission")

    time_1 = time.time()
    cross_correlation_result = cross_correlate(K_p_arr, V_sys_arr, wl, spectrum, data)
    time_2 = time.time()
    print(time_2 - time_1)

    pickle.dump(
        cross_correlation_result,
        open(output_path + f"/{species}_cross_correlation_results.pic", "wb"),
    )
