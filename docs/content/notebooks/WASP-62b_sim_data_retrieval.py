from POSEIDON.constants import R_Sun, R_J
from POSEIDON.core import create_star, create_planet, load_data, define_model, \
                          wl_grid_constant_R, set_priors, read_opacities
from POSEIDON.visuals import plot_data, plot_spectra_retrieved, plot_PT_retrieved, \
                             plot_chem_retrieved
from POSEIDON.retrieval import run_retrieval
from POSEIDON.utility import read_retrieved_spectrum, read_retrieved_PT, \
                             read_retrieved_log_X, plot_collection
from POSEIDON.corner import generate_cornerplot

import numpy as np

do_retrieval = True

#***** Define stellar properties *****#

R_s = 1.23*R_Sun     # Stellar radius (m)
T_s = 6230.0         # Stellar effective temperature (K)
Met_s = 0.04         # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.45       # Stellar log surface gravity (log10(cm/s^2) by convention)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s)

#***** Define planet properties *****#

planet_name = 'WASP-62b'  # Planet name used for plots, output files etc.

R_p = 1.32*R_J     # Planetary radius (m)
g_p = 7.3978       # Gravitational field of planet (m/s^2)
T_eq = 1394        # Equilibrium temperature (K)

# Create the planet object
planet = create_planet(planet_name, R_p, gravity = g_p, T_eq = T_eq)

#***** Model wavelength grid *****#

wl_min = 2.8      # Minimum wavelength (um)
wl_max = 5.3      # Maximum wavelength (um)
R = 10000         # Spectral resolution of grid      

# We need to provide a model wavelength grid to initialise instrument properties
wl = wl_grid_constant_R(wl_min, wl_max, R)

#***** Define model *****#

model_name = 'CH4_Search_With_JWST_multi-core'

bulk_species = ['H2', 'He']
param_species = ['H2O', 'CH4']   # Only H2O and CH4 in this model

# Create the model object
model = define_model(model_name, bulk_species, param_species, 
                     PT_profile = 'isotherm', cloud_model = 'cloud-free')

#***** Specify data location and instruments *****#

data_dir = './data/' + planet_name         

instruments = ['JWST_NIRSpec_G395H_NRS1', 'JWST_NIRSpec_G395H_NRS2']

datasets = [planet_name + '_SYNTHETIC_JWST_NIRSpec_G395H_NRS1_' + model_name + '_N_trans_1.dat',
            planet_name + '_SYNTHETIC_JWST_NIRSpec_G395H_NRS2_' + model_name + '_N_trans_1.dat']

# Load dataset, pre-load instrument PSF and transmission function
data = load_data(data_dir, datasets, instruments, wl)

#***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types['T'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['log_X'] = 'uniform'

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges['T'] = [400, 2000]
prior_ranges['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges['log_X'] = [-12, -1]                # Same for H2O and CH4

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

#***** Read opacity data *****#

opacity_treatment = 'opacity_sampling'

# Define fine temperature grid (K)
T_fine_min = 400     # Same as prior range for T
T_fine_max = 2000    # Same as prior range for T
T_fine_step = 10     # 10 K steps are a good tradeoff between accuracy and RAM

T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2.0    # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step), 
                       log_P_fine_step)

#***** Specify fixed atmospheric settings for retrieval *****#

# Atmospheric pressure grid
P_min = 1.0e-7   # 0.1 ubar
P_max = 100      # 100 bar
N_layers = 100   # 100 layers

# Let's space the layers uniformly in log-pressure
P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure
P_ref = 1.0e-2   # Retrieved R_p_ref parameter will be the radius at 10 mbar

#***** Run atmospheric retrieval *****#

# Run atmospheric retrieval
if (do_retrieval == True):

    # Pre-interpolate the opacities
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

    run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R = R, 
                  spectrum_type = 'transmission', sampling_algorithm = 'MultiNest', 
                  N_live = 1000, verbose = True, resume = False)

#***** Make corner plot *****#

fig_corner = generate_cornerplot(planet, model, true_vals = [R_p/R_J, T_eq, -3.62, -7.46])

#***** Plot retrieved transmission spectrum *****#

# Read retrieved spectrum confidence regions
wl, spec_low2, spec_low1, spec_median, \
spec_high1, spec_high2 = read_retrieved_spectrum(planet_name, model_name)

# Create composite spectra objects for plotting
spectra_median = []
spectra_low2 = []
spectra_low1 = []
spectra_high1 = []
spectra_high2 = []

# Add retrieved spectra to composite objects
spectra_median = plot_collection(spec_median, wl, collection = spectra_median)
spectra_low1 = plot_collection(spec_low1, wl, collection = spectra_low1) 
spectra_low2 = plot_collection(spec_low2, wl, collection = spectra_low2) 
spectra_high1 = plot_collection(spec_high1, wl, collection = spectra_high1) 
spectra_high2 = plot_collection(spec_high2, wl, collection = spectra_high2)

# Plot retrieved spectra
fig_spec = plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1, 
                                  spectra_high1, spectra_high2, planet_name,
                                  data, R_to_bin = 100,
                                  data_labels = ['NIRSpec G395H NRS1', 'NIRSpec G395H NRS2'],
                                  data_colour_list = ['orange', 'crimson'],
                                  y_min = 1.18e-2, y_max = 1.32e-2, 
                                  figure_shape = 'wide', wl_axis = 'linear',
                                  plt_label = 'Simulated JWST Retrieval')

#***** Plot retrieved P-T profile *****#

# Read retrieved P-T profile confidence regions
P, T_low2, T_low1, \
T_median, T_high1, T_high2 = read_retrieved_PT(planet_name, model_name)

# Create composite P-T objects for plotting
PT_median = []
PT_low2 = []
PT_low1 = []
PT_high1 = []
PT_high2 = []

# Add retrieved spectra to composite objects
PT_median = plot_collection(T_median, P, collection = PT_median)
PT_low1 = plot_collection(T_low1, P, collection = PT_low1) 
PT_low2 = plot_collection(T_low2, P, collection = PT_low2) 
PT_high1 = plot_collection(T_high1, P, collection = PT_high1) 
PT_high2 = plot_collection(T_high2, P, collection = PT_high2)

# Produce figure
fig_PT = plot_PT_retrieved(planet_name, PT_median, PT_low2, PT_low1, PT_high1,
                           PT_high2, plt_label = model_name,
                           PT_labels = ['P-T profile'])
