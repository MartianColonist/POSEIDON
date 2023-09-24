from __future__ import absolute_import, unicode_literals, print_function
from POSEIDON.high_res import loglikelihood_sysrem, loglikelihood_PCA, get_rot_kernel
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

K_p = -200  # True value 192.06
N_K_p = 100
d_K_p = 2
K_p_arr = (
    np.arange(N_K_p) - (N_K_p - 1) // 2
) * d_K_p + K_p  # making K_p_arr (centered on published or predicted K_p)
# K_p_arr = [92.06 , ..., 191.06, 192.06, 193.06, ..., 291.06]

V_sys = 0
N_V_sys = 100
d_V_sys = 2
V_sys_arr = (
    np.arange(N_V_sys) - (N_V_sys - 1) // 2
) * d_V_sys + V_sys  # making V_sys_arr (centered on published or predicted V_sys (here 0 because we already added V_sys in V_bary))


N_jobs = -1  # Change how many cores to use here.


def cross_correlate(Kp_arr, Vsys_arr, wl, F_p_obs, F_s_obs, data):
    a = 1
    data_arr = data["data_arr"]
    data_scale = data["data_scale"]
    phi = data["phi"]
    wl_grid = data["wl_grid"]
    V_bary = data["V_bary"]

    N_order, N_phi, N_wl = data_arr.shape
    loglikelihood_array_final = np.zeros((len(Kp_arr), len(Vsys_arr)))
    CCF_array_final = np.zeros((len(Kp_arr), len(Vsys_arr)))

    RV_min = min(
        [
            np.min(Kp_arr * np.sin(2 * np.pi * phi[i]) + V_bary[i]) + np.min(Vsys_arr)
            for i in range(len(phi))
        ]
    )
    RV_max = max(
        [
            np.max(Kp_arr * np.sin(2 * np.pi * phi[i]) + V_bary[i]) + np.max(Vsys_arr)
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
            dl_p = RV * 1e3 / constants.c
            wl_shifted_p = wl_slice * (1.0 - dl_p)
            F_p = np.interp(wl_shifted_p, wl, F_p_obs)
            # F_s = np.interp(wl_slice, wl, F_s_obs)
            # models_shifted[i, j, :] = F_p / F_s * data_scale[j, :]
            models_shifted[i, j, :] = F_p

    for i in range(N_phi):
        loglikelihoods = np.zeros_like(RV_range)
        CCFs = np.zeros_like(RV_range)
        for k in range(N_order):
            for j, RV in enumerate(RV_range):
                m = models_shifted[j, k, :] * a
                m -= m.mean()  # mean subtracting here...
                m2 = m.dot(m)
                f = data_arr[k, i]  # already mean-subtracted
                f2 = f.dot(f)
                R = m.dot(f)  # cross-covariance
                CCF = R / np.sqrt(m2 * f2)  # cross-correlation
                loglikelihood = (
                    -0.5 * N_wl * np.log((m2 + f2 - 2.0 * R) / N_wl)
                )  # Equation 9 in paper
                loglikelihoods[j] += loglikelihood
                CCFs[j] += CCF

        CCF_array_per_phase[i, :] = CCFs
        for j, Kp in enumerate(Kp_arr):
            RV = Kp * np.sin(2 * np.pi * phi[i]) + Vsys_arr + V_bary[i]
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
if __name__ == "__main__":
    data_path = "./data/WASP-77Ab-injection-6/"
    output_path = "./CC_output/WASP-77Ab-injection-6/"
    os.makedirs(output_path, exist_ok=True)
    data = read_high_res_data(data_path, method="pca", spectrum_type="emission")
    data["data_raw"] = None
    data["V_sin_i"] = 4.5

    R_s = 1.21 * R_Sun  # Stellar radius (m)
    T_s = 5605.0  # Stellar effective temperature (K)
    Met_s = -0.04  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.56  # Stellar log surface gravity (log10(cm/s^2) by convention)
    # ***** Define planet properties *****#

    planet_name = "WASP-77Ab"  # Planet name used for plots, output files etc.

    R_p = 1.21 * R_J  # Planetary radius (m)
    M_p = 0.07 * M_J  # Mass of planet (kg)
    g_p = 4.3712  # Gravitational field of planet (m/s^2)
    T_eq = 1043.8  # Equilibrium temperature (K)

    # Create the planet object
    planet = create_planet(planet_name, R_p, mass=M_p, gravity=g_p, T_eq=T_eq)

    # If distance not specified, use fiducial value
    if planet["system_distance"] is None:
        planet[
            "system_distance"
        ] = 1  # This value only used for flux ratios, so it cancels
    d = planet["system_distance"]

    # ***** Define model *****#

    model_name = "High-res retrieval"  # Model name used for plots, output files etc.

    bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere

    for species in ["H2O", "CO", "CH4", "CO2"]:
        # for species in ["H2O"]:
        param_species = [species]

        model = define_model(
            model_name, bulk_species, param_species, PT_profile="Madhu"
        )

        # Check the free parameters defining this model
        print("Free parameters: " + str(model["param_names"]))

        # ***** Wavelength grid *****#
        wl_min = 1.3  # Minimum wavelength (um)
        wl_max = 2.6  # Maximum wavelength (um)
        R = 250000  # Spectral resolution of grid

        model["R"] = R
        model["R_instrument"] = 66000  # Resolution of instrument

        wl = wl_grid_constant_R(wl_min, wl_max, R)

        # Create the stellar object
        star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="phoenix")

        F_s = star["F_star"]
        wl_s = star["wl_star"]

        # ***** Read opacity data *****#

        opacity_treatment = "opacity_sampling"

        # Define fine temperature grid (K)
        T_fine_min = 2000  # 400 K lower limit suffices for a typical hot Jupiter
        T_fine_max = 4000  # 2000 K upper limit suffices for a typical hot Jupiter
        T_fine_step = 20  # 20 K steps are a good tradeoff between accuracy and RAM

        T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

        # Define fine pressure grid (log10(P/bar))
        log_P_fine_min = -5.0  # 1 ubar is the lowest pressure in the opacity database
        log_P_fine_max = 2  # 100 bar is the highest pressure in the opacity database
        log_P_fine_step = 0.2

        log_P_fine = np.arange(
            log_P_fine_min, (log_P_fine_max + log_P_fine_step), log_P_fine_step
        )

        # Now we can pre-interpolate the sampled opacities (may take up to a minute)
        opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

        # Specify the pressure grid of the atmosphere
        P_min = 1.0e-5  # 0.1 ubar
        P_max = 100  # 100 bar
        N_layers = 100  # 100 layers

        # We'll space the layers uniformly in log-pressure
        P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

        # Specify the reference pressure and radius
        P_ref = 1e-5  # Reference pressure (bar)
        R_p_ref = R_p  # Radius at reference pressure

        params = (-6, 0.3, 0.3, -1, -2, 1, 3000)
        log_H2O, a1, a2, log_P1, log_P2, log_P3, T_ref = params

        # Provide a specific set of model parameters for the atmosphere
        PT_params = np.array([a1, a2, log_P1, log_P2, log_P3, T_ref])

        log_X_params = np.array([[log_H2O]])

        atmosphere = make_atmosphere(
            planet, model, P, P_ref, R_p_ref, PT_params, log_X_params
        )

        # Generate planet surface flux
        F_p_obs = compute_spectrum(
            planet, star, model, atmosphere, opac, wl, spectrum_type="direct_emission"
        )

        F_s_interp = spectres(wl, wl_s, F_s)
        F_s_obs = (R_s / d) ** 2 * F_s_interp  # observed flux of star on earth

        # instrument profile convolution
        R_instrument = model["R_instrument"]
        R = model["R"]
        V_sin_i = data["V_sin_i"]
        rot_kernel = get_rot_kernel(V_sin_i, wl, 401)
        F_p_rot = np.convolve(
            F_p_obs, rot_kernel, mode="same"
        )  # calibrate for planetary rotation
        xker = np.arange(-20, 21)
        sigma = (R / R_instrument) / (
            2 * np.sqrt(2.0 * np.log(2.0))
        )  # model is right now at R=250K.  IGRINS is at R~45K. We make gaussian that is R_model/R_IGRINS ~ 5.5
        yker = np.exp(
            -0.5 * (xker / sigma) ** 2.0
        )  # instrumental broadening kernel; not understand
        yker /= yker.sum()
        F_p_conv = np.convolve(F_p_rot, yker, mode="same")
        F_s_conv = np.convolve(
            F_s_obs, yker, mode="same"
        )  # no need to times (R)^2 because F_p, F_s are already observed value on Earth

        time_1 = time.time()
        # results = Parallel(n_jobs=N_jobs, max_nbytes=1e7, verbose=10)(
        #     delayed(cross_correlate)(
        #         x, K_p_arr, V_sys_arr, wl, F_p_conv, F_s_conv, data
        #     )
        #     for x in coordinate_list
        # )
        cross_correlation_result = cross_correlate(
            K_p_arr, V_sys_arr, wl, F_p_conv, F_s_conv, data
        )
        time_2 = time.time()
        print(time_2 - time_1)

        pickle.dump(
            cross_correlation_result,
            open(output_path + f"/{species}_cross_correlation_results.pic", "wb"),
        )
