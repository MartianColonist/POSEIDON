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

K_p = -200
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


N_jobs = 16  # Change how many cores to use here.


def get_coordinate_list(x_values, y_values):
    x, y = np.meshgrid(x_values, y_values)
    coordinates = np.dstack([x, y]).reshape(-1, 2)
    return [tuple(coord) for coord in coordinates]


x_values = np.arange(N_K_p)
y_values = np.arange(N_V_sys)

coordinate_list = get_coordinate_list(x_values, y_values)

data_path = "./data/WASP-121b-injection/"
output_path = "./CC_output/WASP-121b-injection/"
# data_path = "./data/WASP-121b/"
# output_path = "./CC_output/WASP-121b/"
os.makedirs(output_path, exist_ok=True)
data = read_high_res_data(data_path, method="sysrem")


def cross_correlate(coord, K_p_arr, V_sys_arr, wl, spectrum, data):
    print(coord)
    K_p = K_p_arr[coord[0]]
    V_sys = V_sys_arr[coord[1]]
    loglikelihood, CCF = loglikelihood_sysrem(
        V_sys, K_p, 0, 1, None, wl, spectrum, data
    )
    return (loglikelihood, CCF)


# The code below will only be run on one core to get the model spectrum.
if __name__ == "__main__":
    R_s = 1.458 * R_Sun  # Stellar radius (m)
    T_s = 6776  # Stellar effectsive temperature (K)
    Met_s = 0.13  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.24  # Stellar log surface gravity (log10(cm/s^2) by convention)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="phoenix")

    F_s = star["F_star"]
    wl_s = star["wl_star"]
    R_s = star["R_s"]

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
        planet[
            "system_distance"
        ] = 1  # This value only used for flux ratios, so it cancels
    d = planet["system_distance"]

    # ***** Define model *****#

    model_name = "High-res retrieval"  # Model name used for plots, output files etc.

    bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere
    param_species = ["Fe"]  # H2O, CO as in Brogi & Line

    # Create the model object
    model = define_model(model_name, bulk_species, param_species, PT_profile="Madhu")

    # Check the free parameters defining this model
    print("Free parameters: " + str(model["param_names"]))

    # ***** Wavelength grid *****#

    wl_min = 0.37  # Minimum wavelength (um)
    wl_max = 0.51  # Maximum wavelength (um)
    R = 250000  # Spectral resolution of grid

    # wl = wl_grid_line_by_line(wl_min, wl_max)
    wl = wl_grid_constant_R(wl_min, wl_max, R)
    # ***** Read opacity data *****#

    opacity_treatment = "opacity_sampling"

    # Define fine temperature grid (K)
    T_fine_min = 400  # 400 K lower limit suffices for a typical hot Jupiter
    T_fine_max = 4000  # 2000 K upper limit suffices for a typical hot Jupiter
    T_fine_step = 20  # 20 K steps are a good tradeoff between accuracy and RAM

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -9.0  # 1 ubar is the lowest pressure in the opacity database
    log_P_fine_max = 2  # 100 bar is the highest pressure in the opacity database
    log_P_fine_step = 0.2  # 0.2 dex steps are a good tradeoff between accuracy and RAM

    log_P_fine = np.arange(
        log_P_fine_min, (log_P_fine_max + log_P_fine_step), log_P_fine_step
    )

    # Now we can pre-interpolate the sampled opacities (may take up to a minute)
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

    # Specify the pressure grid of the atmosphere
    P_min = 1.0e-9  # 0.1 ubar
    P_max = 100  # 100 bar
    N_layers = 100  # 100 layers

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure and radius
    P_ref = 1e-5  # Reference pressure (bar)
    R_p_ref = R_p  # Radius at reference pressure

    params = (-3, 0.3, 0.3, -1, -2, 1, 3000)
    log_Fe, a1, a2, log_P1, log_P2, log_P3, T_ref = params

    # Provide a specific set of model parameters for the atmosphere
    PT_params = np.array(
        [a1, a2, log_P1, log_P2, log_P3, T_ref]
    )  # a1, a2, log_P1, log_P2, log_P3, T_deep
    log_X_params = np.array([[log_Fe]])

    atmosphere = make_atmosphere(
        planet, model, P, P_ref, R_p_ref, PT_params, log_X_params
    )

    # Generate planet surface flux
    spectrum = compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="transmission"
    )

    param_species = []

    # Create the model object
    model = define_model(
        model_name,
        bulk_species,
        param_species,
        PT_profile="Madhu",
    )

    params = (0.3, 0.3, -1, -2, 1, 3000)
    a1, a2, log_P1, log_P2, log_P3, T_ref = params

    # Provide a specific set of model parameters for the atmosphere
    PT_params = np.array(
        [a1, a2, log_P1, log_P2, log_P3, T_ref]
    )  # a1, a2, log_P1, log_P2, log_P3, T_deep

    atmosphere = make_atmosphere(
        planet, model, P, P_ref, R_p_ref, PT_params, log_X_params
    )

    # Generate planet surface flux
    continuum = compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="transmission"
    )

    time_1 = time.time()
    results = Parallel(
        n_jobs=N_jobs, max_nbytes=1e7, verbose=10
    )(  # wl, spectrum need to be memory mapped for optimum efficiency
        delayed(lambda x: cross_correlate(x, K_p_arr, V_sys_arr, wl, spectrum, data))(x)
        for x in coordinate_list
    )
    time_2 = time.time()
    print(time_2 - time_1)

    loglikelihood_array = np.zeros((N_K_p, N_V_sys))
    CCF_array = np.zeros((N_K_p, N_V_sys))
    for i, coord in enumerate(coordinate_list):
        loglikelihood_array[coord[0], coord[1]] = results[i][0]
        CCF_array[coord[0], coord[1]] = results[i][1]

    pickle.dump(
        [K_p_arr, V_sys_arr, loglikelihood_array, CCF_array],
        open(output_path + "/cross_correlation_results.pic", "wb"),
    )
