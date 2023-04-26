# %%
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J

#***** Define stellar properties *****#

R_s = 1.458*R_Sun     # Stellar radius (m)
T_s = 6776            # Stellar effective temperature (K)
Met_s = 0.13          # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.24        # Stellar log surface gravity (log10(cm/s^2) by convention)

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

# %%
from POSEIDON.core import define_model, wl_grid_constant_R
from POSEIDON.utility import read_high_res_data
#***** Define model *****#

model_name = 'High-res retrieval'  # Model name used for plots, output files etc.

bulk_species = ['H2', 'He']         # H2 + He comprises the bulk atmosphere
# param_species = ['Mg', 'Fe', 'Ti']
param_species = ['Fe']

method = 'sysrem'
# high_res_params = ['a', 'b', 'dPhi', 'K_p', 'V_sys', 'W_conv']
high_res_params = ['a', 'b', 'K_p', 'V_sys', 'W_conv']

# Create the model object
# model = define_model(model_name, bulk_species, param_species, 
#                     PT_profile = 'Madhu', high_res = high_res,
#                     high_res_params = high_res_params, R_p_ref_enabled=False)
model = define_model(model_name, bulk_species, param_species, 
                    PT_profile = 'isotherm', high_res_params = high_res_params)

# Check the free parameters defining this model
print("Free parameters: " + str(model['param_names']))

#***** Wavelength grid *****#

wl_min = 0.37      # Minimum wavelength (um)
wl_max = 0.51      # Maximum wavelength (um)
R = 500000         # Spectral resolution of grid
model['R'] = R
model['R_instrument'] = 80000 # Resolution of instrument
# wl = wl_grid_line_by_line(wl_min, wl_max)
wl = wl_grid_constant_R(wl_min, wl_max, R)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid = 'phoenix')

data_dir = './reference_data/observations/WASP-121b'

data = read_high_res_data(data_dir, method='sysrem', spectrum_type='transmission')
data['V_sin_i'] = 14.5
data['uncertainties'] = None
data_raw = data['data_raw']
data_raw[data_raw < 0] = 0
data['data_raw'] = data_raw
# %%
from POSEIDON.core import set_priors

#***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types['T_ref'] = 'uniform'
prior_types['T'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['log_Ti'] = 'uniform'
prior_types['log_Fe'] = 'uniform'
prior_types['log_Mg'] = 'uniform'
prior_types['a1'] = 'uniform'
prior_types['a2'] = 'uniform'
prior_types['log_P1'] = 'uniform'
prior_types['log_P2'] = 'uniform'
prior_types['log_P3'] = 'uniform'
prior_types['K_p'] = 'uniform'
prior_types['V_sys'] = 'uniform'
prior_types['a'] = 'uniform'
prior_types['b'] = 'uniform'
prior_types['dPhi'] = 'uniform'
prior_types['W_conv'] = 'uniform'

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges['T_ref'] = [2000, 4500]
prior_ranges['T'] = [2000, 4500]
prior_ranges['R_p_ref'] = [0.5*R_p, 2*R_p]
prior_ranges['log_Ti'] = [-15, 0]
prior_ranges['log_Fe'] = [-15, 0]
prior_ranges['log_Mg'] = [-15, 0]
prior_ranges['a1'] = [0.02, 1]
prior_ranges['a2'] = [0.02, 1]
prior_ranges['log_P1'] = [-5.5, 2.5]
prior_ranges['log_P2'] = [-5.5, 2.5]
prior_ranges['log_P3'] = [-2, 2]
prior_ranges['K_p'] = [170, 230]
prior_ranges['V_sys'] = [-10, 10]
prior_ranges['a'] = [0.1, 10]
prior_ranges['b'] = [0.1, 10]
prior_ranges['dPhi'] = [-0.01, 0.01]
prior_ranges['W_conv'] = [1, 50]

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

# %%
from POSEIDON.core import read_opacities
import numpy as np

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

# %%
from POSEIDON.retrieval import run_retrieval

#***** Specify fixed atmospheric settings for retrieval *****#

# Atmospheric pressure grid
P_min = 1e-12    # 0.1 ubar
P_max = 100       # 100 bar
N_layers = 100    # 100 layers

# Let's space the layers uniformly in log-pressure
P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure and radius
P_ref = 1e-5   # Reference pressure (bar)

#***** Run atmospheric retrieval *****#

run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R = R, 
                spectrum_type = 'transmission', sampling_algorithm = 'MultiNest', 
                N_live = 800, verbose = True, N_output_samples = 1000, resume = False, ev_tol=0.05)


# %% [markdown]
# Now that the retrieval is finished, you're eager and ready to see what WASP-999b's atmosphere is hiding. 
# 
# You first plot confidence intervals of the retrieved spectrum from this model compared to WASP-999b's observed transmission spectrum. You also generate a corner plot showing the retrieved probability distributions of the model parameters.

# %%
from POSEIDON.utility import read_retrieved_PT, read_retrieved_log_X
from POSEIDON.visuals import plot_PT, plot_PT_retrieved, plot_chem_retrieved
from POSEIDON.corner import generate_cornerplot

#***** Plot retrieved transmission spectrum *****#

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