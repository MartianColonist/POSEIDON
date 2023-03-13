from __future__ import absolute_import, unicode_literals, print_function
from POSEIDON.high_res import fast_filter, log_likelihood_sysrem, get_rot_kernel, fit_uncertainties
import math, os
import numpy as np
import pickle
import pickle
from scipy import constants
from numba import jit
from astropy.io import fits
from scipy import interpolate
from POSEIDON.core import create_star, create_planet, define_model, make_atmosphere, read_opacities, wl_grid_constant_R, wl_grid_line_by_line, compute_spectrum
from POSEIDON.constants import R_Sun
from POSEIDON.visuals import plot_stellar_flux
from POSEIDON.constants import R_J, M_J
import numpy as np
from spectres import spectres
from tqdm import tqdm
from multiprocessing import Pool
import time
from scipy.ndimage import gaussian_filter1d, median_filter
from POSEIDON.utility import read_high_res_data

def cross_correlate(spectrum, continuum, wl, K_p_arr, V_sys_arr, wl_grid, residuals, Bs, V_bary, Phi, uncertainties):
    dPhi = 0.0
    a = 1.57
    b = 0.7652
    W_conv = 4.44
    # F_p_conv = gaussian_filter1d((spectrum - continuum) * a + continuum, W_conv)
    # F_p_conv = (spectrum - continuum) * a + continuum
    F_p_conv = gaussian_filter1d(spectrum - continuum, W_conv) * a + continuum
    
    cs_p = interpolate.splrep(wl, F_p_conv, s=0.0) # no need to times (R)^2 because F_p_obs, F_s_obs are already observed value on Earth

    loglikelihood_arr = np.zeros((len(K_p_arr), len(V_sys_arr)))
    CCF_arr = np.zeros((len(K_p_arr), len(V_sys_arr)))

    for i in range(len(K_p_arr)):
        for j in range(len(V_sys_arr)):
            loglikelihood, CCF_sum = log_likelihood_sysrem(V_sys_arr[j], K_p_arr[i], dPhi, cs_p, wl_grid, residuals, Bs, V_bary, Phi, uncertainties, a, b)
            loglikelihood_arr[i, j] = loglikelihood
            CCF_arr[i, j] = CCF_sum

    return loglikelihood_arr, CCF_arr

K_p = 200
N_K_p = 200
d_K_p = 2
K_p_arr = (np.arange(N_K_p) - (N_K_p-1)//2) * d_K_p + K_p # making K_p_arr (centered on published or predicted K_p)
# K_p_arr = [92.06 , ..., 191.06, 192.06, 193.06, ..., 291.06]

V_sys = 0
N_V_sys = 200
d_V_sys = 2
V_sys_arr = (np.arange(N_V_sys) - (N_V_sys-1)//2) * d_V_sys + V_sys # making V_sys_arr (centered on published or predicted V_sys (here 0 because we already added V_sys in V_bary))

N_cores = 10 # Change how many cores to use here.
core_indices = np.arange(N_cores)
V_sys_arr_split = np.array_split(V_sys_arr, N_cores)

def batch_cross_correlate(core_index, args):
    spectrum, continuum, wl = args
    for i in range(core_index, len(V_sys_arr_split), N_cores):
        print("Core "+str(i)+" is running.")
        output_path = './CC_output/WASP-121b' # Could modify output path here.
        data_path = './reference_data/observations/WASP-121b'
        os.makedirs(output_path, exist_ok=True)

        data = read_high_res_data(data_path, high_res='sysrem')
        data_raw = data['data_raw']
        wl_grid = data['wl_grid']
        Phi = data['Phi']
        V_bary = data['V_bary']
        data_raw[data_raw < 0] = 0
        Ndet, Nphi, Npix = data_raw.shape
        data_norm = np.zeros(data_raw.shape)

        # uncertainties = fit_uncertainties(data_raw, NPC=5)
        uncertainties = pickle.load(open(data_path+'/uncertainties.pic', 'rb'))

        for k in range(len(data_raw)):
            order = data_raw[k]
            
            median = np.median(order, axis=0)
            median[median == 0] = np.mean(median)
            order_norm = order / median

            uncertainty = uncertainties[k]

            uncertainty_norm = uncertainty / median
            
            uncertainties[k] = uncertainty_norm
            data_norm[k] = order_norm

        residuals, Us = fast_filter(data_norm, uncertainties, iter=15)
        Bs = np.zeros((Ndet, Nphi, Nphi))

        for j in range(Ndet):
            U = Us[j]
            L = np.diag(1 / np.mean(uncertainties[j], axis=-1))
            # B = U @ np.linalg.inv((L @ U).T @ (L @ U)) @ (L @ U).T @ L
            B = U @ np.linalg.pinv(L @ U) @ L
            Bs[j] = B

        log_L_arr, CCF_arr = cross_correlate(spectrum, continuum, wl, K_p_arr, V_sys_arr_split[i], wl_grid, residuals, Bs, V_bary, Phi, uncertainties)
        pickle.dump([V_sys_arr_split[i], K_p_arr, log_L_arr, CCF_arr], open(output_path+'/test_sysrem_'+str(i)+'.pic','wb')) # N_cores of these produced. Will be read in by plot_CCF.py

# The code below will only be run on one core to get the model spectrum.
if __name__ == '__main__':
    R_s = 1.458*R_Sun     # Stellar radius (m)
    T_s = 6776            # Stellar effectsive temperature (K)
    Met_s = 0.13          # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.24        # Stellar log surface gravity (log10(cm/s^2) by convention)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s, stellar_spectrum = True, stellar_grid = 'phoenix')

    F_s = star['F_star']
    wl_s = star['wl_star']
    R_s = star['stellar_radius']


    #***** Define planet properties *****#

    planet_name = 'WASP-121b'  # Planet name used for plots, output files etc.

    R_p = 1.753*R_J      # Planetary radius (m)
    M_p = 1.157*M_J      # Mass of planet (kg)
    g_p = 10**(2.97-2) # Gravitational field of planet (m/s^2)
    T_eq = 2450          # Equilibrium temperature (K)

    # Create the planet object
    planet = create_planet(planet_name, R_p, mass = M_p, gravity = g_p, T_eq = T_eq)

    # If distance not specified, use fiducial value
    if (planet['system_distance'] is None):
        planet['system_distance'] = 1    # This value only used for flux ratios, so it cancels
    d = planet['system_distance']
    
    #***** Define model *****#

    model_name = 'High-res retrieval'  # Model name used for plots, output files etc.

    bulk_species = ['H2', 'He']     # H2 + He comprises the bulk atmosphere
    param_species = ['Fe']  # H2O, CO as in Brogi & Line

    # Create the model object
    model = define_model(model_name, bulk_species, param_species,
                        PT_profile = 'Madhu', high_res = 'sysrem', R_p_ref_enabled=False)

    # Check the free parameters defining this model
    print("Free parameters: " + str(model['param_names']))


    #***** Wavelength grid *****#

    wl_min = 3.7      # Minimum wavelength (um)
    wl_max = 5.1      # Maximum wavelength (um)
    R = 200000        # Spectral resolution of grid

    # wl = wl_grid_line_by_line(wl_min, wl_max)
    wl = wl_grid_constant_R(wl_min, wl_max, R)
    #***** Read opacity data *****#

    opacity_treatment = 'opacity_sampling'

    # Define fine temperature grid (K)
    T_fine_min = 2000     # 400 K lower limit suffices for a typical hot Jupiter
    T_fine_max = 4500    # 2000 K upper limit suffices for a typical hot Jupiter
    T_fine_step = 20     # 20 K steps are a good tradeoff between accuracy and RAM

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
    log_P_fine_max = 2.5    # 100 bar is the highest pressure in the opacity database
    log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step),
                        log_P_fine_step)

    # Now we can pre-interpolate the sampled opacities (may take up to a minute)
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

    # Specify the pressure grid of the atmosphere
    P_min = 1.0e-5    # 0.1 ubar
    P_max = 100       # 100 bar
    N_layers = 100    # 100 layers

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure and radius
    P_ref = 1e-5   # Reference pressure (bar)
    R_p_ref = R_p  # Radius at reference pressure

    
    params = (-3.28, 2, 1, -2.5, -1.5, 1, 3000)
    log_Fe, a1, a2, log_P1, log_P2, log_P3, T_ref = params

    # Provide a specific set of model parameters for the atmosphere
    PT_params = np.array([a1, a2, log_P1, log_P2, log_P3, T_ref])     # a1, a2, log_P1, log_P2, log_P3, T_deep
    log_X_params = np.array([[log_Fe]])
    
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, log_X_params)

    # Generate planet surface flux
    spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl, spectrum_type='transmission')

    param_species = []

    # Create the model object
    model = define_model(model_name, bulk_species, param_species,
                        PT_profile = 'Madhu', high_res = 'sysrem', R_p_ref_enabled=False)
    params = (2, 1, -2.5, -1.5, 1, 3000)
    a1, a2, log_P1, log_P2, log_P3, T_ref = params

    # Provide a specific set of model parameters for the atmosphere
    PT_params = np.array([a1, a2, log_P1, log_P2, log_P3, T_ref])     # a1, a2, log_P1, log_P2, log_P3, T_deep
    log_X_params = np.array([[log_Fe]])
    
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, log_X_params)

    # Generate planet surface flux
    continuum = compute_spectrum(planet, star, model, atmosphere, opac, wl, spectrum_type='transmission')

    # Passing stellar spectrum, planet spectrum, wavelenght grid to each core, thus saving time for reading the opacity again
    from itertools import repeat
    pool = Pool(processes=N_cores)
    time_1 = time.time()
    pool.starmap(batch_cross_correlate, zip(core_indices, repeat((spectrum, continuum, wl))))
    time_2 = time.time()
    print(time_2-time_1)