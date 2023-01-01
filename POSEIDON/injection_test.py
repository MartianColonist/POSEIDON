# %%
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J

#***** Define stellar properties *****#

R_s = 0.91*R_Sun      # Stellar radius (m)
T_s = 5605.0          # Stellar effective temperature (K)
Met_s = -0.04         # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.48        # Stellar log surface gravity (log10(cm/s^2) by convention)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, stellar_spectrum = True, stellar_grid = 'phoenix')

F_s = star['F_star']
wl_s = star['wl_star']
R_s = star['stellar_radius']

#***** Define planet properties *****#

planet_name = 'WASP-77Ab_injection_test'  # Planet name used for plots, output files etc.

R_p = 1.21*R_J      # Planetary radius (m)
M_p = 1.76*M_J      # Mass of planet (kg)
g_p = 10**(3.4765-2)# Gravitational field of planet (m/s^2)
T_eq = 1740         # Equilibrium temperature (K)

# Create the planet object
planet = create_planet(planet_name, R_p, mass = M_p, gravity = g_p, T_eq = T_eq)

# If distance not specified, use fiducial value
if (planet['system_distance'] is None):
    planet['system_distance'] = 1    # This value only used for flux ratios, so it cancels
d = planet['system_distance']

planet['V_sin_i'] = 4.5

# %%
from POSEIDON.core import define_model, wl_grid_constant_R
from POSEIDON.utility import read_high_res_data
#***** Define model *****#

model_name = 'High-res retrieval'  # Model name used for plots, output files etc.

bulk_species = ['H2', 'He']     # H2 + He comprises the bulk atmosphere
param_species = ['H2O', 'CO']  # H2O, CO as in Brogi & Line

# Create the model object
model = define_model(model_name, bulk_species, param_species,
                    PT_profile = 'Madhu', high_res = 'pca', R_p_ref_enabled=False)

# Check the free parameters defining this model
print("Free parameters: " + str(model['param_names']))

#***** Wavelength grid *****#

wl_min = 1.3      # Minimum wavelength (um)
wl_max = 2.6      # Maximum wavelength (um)
R = 250000          # Spectral resolution of grid

# wl = wl_grid_line_by_line(wl_min, wl_max)
wl = wl_grid_constant_R(wl_min, wl_max, R)

data_dir = './reference_data/observations/WASP-77Ab_injection_test'         # Special directory for this tutorial

data = read_high_res_data(data_dir, high_res='pca')


from POSEIDON.core import set_priors

#***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types['T_ref'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['log_H2O'] = 'uniform'
prior_types['log_CO'] = 'uniform'
prior_types['log_CH4'] = 'uniform'
prior_types['log_H2S'] = 'uniform'
prior_types['log_NH3'] = 'uniform'
prior_types['log_HCN'] = 'uniform'
prior_types['a1'] = 'uniform'
prior_types['a2'] = 'uniform'
prior_types['log_P1'] = 'uniform'
prior_types['log_P2'] = 'uniform'
prior_types['log_P3'] = 'uniform'
prior_types['K_p'] = 'uniform'
prior_types['V_sys'] = 'uniform'
prior_types['log_a'] = 'gaussian'
prior_types['dPhi'] = 'gaussian'

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges['T_ref'] = [500, 2000]
prior_ranges['R_p_ref'] = [0.5*R_p, 1.5*R_p]
prior_ranges['log_H2O'] = [-12, 0]
prior_ranges['log_CO'] = [-12, 0]
prior_ranges['log_CH4'] = [-12, 0]
prior_ranges['log_H2S'] = [-12, 0]
prior_ranges['log_NH3'] = [-12, 0]
prior_ranges['log_HCN'] = [-12, 0]
prior_ranges['a1'] = [0.02, 1]
prior_ranges['a2'] = [0.02, 1]
prior_ranges['log_P1'] = [-5.5, 2.5]
prior_ranges['log_P2'] = [-5.5, 2.5]
prior_ranges['log_P3'] = [-2, 2]
prior_ranges['K_p'] = [180, 220]
prior_ranges['V_sys'] = [-20, 20]
prior_ranges['log_a'] = [-1, 1]
prior_ranges['dPhi'] = [-0.01, 0.01]

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

from POSEIDON.core import read_opacities
import numpy as np

#***** Read opacity data *****#

opacity_treatment = 'opacity_sampling'

# First, specify limits of the fine temperature and pressure grids for the 
# pre-interpolation of cross sections. These fine grids should cover a
# wide range of possible temperatures and pressures for the model atmosphere.

# Define fine temperature grid (K)
T_fine_min = 500     # 400 K lower limit suffices for a typical hot Jupiter
T_fine_max = 3000    # 2000 K upper limit suffices for a typical hot Jupiter
T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2.0    # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                       log_P_fine_step)

# Now we can pre-interpolate the sampled opacities (may take up to a minute)
opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

# # %%
# from POSEIDON.core import compute_spectrum

# # Generate our first transmission spectrum
# F_p_obs = compute_spectrum(planet, star, model, atmosphere, opac, wl,
#                             spectrum_type = 'direct_emission')


# from scipy import interpolate, constants

# V_sys = 0
# K_p = 100
# V_bary = data['V_bary']
# K_s = 0
# Phi = data['Phi']
# wl_grid = data['wl_grid']
# dPhi = 0

# from spectres import spectres
# # Interpolate stellar spectrum onto planet spectrum wavelength grid
# F_s_interp = spectres(wl, wl_s, F_s)

# # Convert stellar surface flux to observed flux at Earth
# F_s_obs = (R_s / d)**2 * F_s_interp

# cs_p = interpolate.splrep(wl, F_p_obs, s=0.0) # no need to times (R)^2 because F_p, F_s are already observed value on Earth
# cs_s = interpolate.splrep(wl, F_s_obs, s=0.0)

# RV_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (Phi + dPhi))  # V_sys is an additive term around zero
# dl_p = RV_p * 1e3 / constants.c # delta lambda, for shifting
# RV_s = (V_sys + V_bary - K_s * np.sin(2 * np.pi * Phi)) * 0  # Velocity of the star is very small compared to planet's velocity and it's already be corrected
# dl_s = RV_s * 1e3 / constants.c # delta lambda, for shifting

# Ndet, Nphi, Npix = 44, 79, 1848
# Fp_Fs = np.zeros((Ndet, Nphi, Npix)) # Fp + Fs
# for j in range(Ndet):
#     wl_slice = wl_grid[j, ]
#     for i in range(Nphi):
#         wl_shifted_p = wl_slice * (1.0 - dl_p[i])
#         Fp = interpolate.splev(wl_shifted_p, cs_p, der=0)
#         wl_shifted_s = wl_slice * (1.0 - dl_s[i])
#         Fs = interpolate.splev(wl_shifted_s, cs_s, der=0)
#         Fp_Fs[j, i, :] = Fp + Fs

# import pickle, os
# os.makedirs(data_dir+'_injection_test', exist_ok = True)
# pickle.dump([wl_grid, Fp_Fs], open(data_dir+'_injection_test/data_RAW.pic','wb'))

# %%
from POSEIDON.retrieval import run_retrieval

#***** Specify fixed atmospheric settings for retrieval *****#

# Atmospheric pressure grid
P_min = 1e-5    # 0.1 ubar
P_max = 100       # 100 bar
N_layers = 100    # 100 layers

# Let's space the layers uniformly in log-pressure
P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure and radius
# P_ref = 10.0   # Reference pressure (bar)
P_ref = 1e-5   # Reference pressure (bar)

#***** Run atmospheric retrieval *****#

run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R = R, 
                spectrum_type = 'direct_emission', sampling_algorithm = 'MultiNest', 
                N_live = 400, verbose = True, N_output_samples = 1000, resume = False, ev_tol=5)
# %%
from POSEIDON.utility import read_retrieved_PT, read_retrieved_log_X
from POSEIDON.visuals import plot_PT, plot_PT_retrieved, plot_chem_retrieved
from POSEIDON.corner import generate_cornerplot

# Read retrieved spectrum confidence regions

P, T_low2, T_low1, T_median, T_high1, T_high2 = read_retrieved_PT(planet_name, model_name, retrieval_name = None)
PT_median = [(T_median, P)]
PT_low2 = [(T_low2, P)]
PT_low1 = [(T_low1, P)]
PT_high1 = [(T_high1, P)]
PT_high2 = [(T_high2, P)]

plot_PT_retrieved(planet_name, PT_median, PT_low2, PT_low1, PT_high1,
                    PT_high2, T_true = None, Atmosphere_dimension = 1, 
                    TwoD_type = None, plt_label = None, show_profiles = [],
                    PT_labels = [], colour_list = [], log_P_min = None,
                    log_P_max = None, T_min = None, T_max = None,
                    legend_location = 'lower left')

P, chemical_species, log_X_low2, log_X_low1, log_X_median, log_X_high1, log_X_high2 = read_retrieved_log_X(planet_name, model_name, retrieval_name = None)
log_Xs_median = [(log_X_median, P)]
log_Xs_low2 = [(log_X_low2, P)]
log_Xs_low1 = [(log_X_low1, P)]
log_Xs_high1 = [(log_X_high1, P)]
log_Xs_high2 = [(log_X_high2, P)]

plot_chem_retrieved(planet_name, chemical_species, log_Xs_median, log_Xs_low2, log_Xs_low1, log_Xs_high1, log_Xs_high2)
#***** Make corner plot *****#

fig_corner = generate_cornerplot(planet, model)