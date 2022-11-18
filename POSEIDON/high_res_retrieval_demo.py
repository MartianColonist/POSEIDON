# %% [markdown]
# # Atmospheric Retrievals with POSEIDON
# 
# At long last, your proposal to observe the newly discovered hot Jupiter WASP-999b with the *Hubble Space Telescope* has been accepted. Congratulations! 
# 
# ### Loading Data
# 
# Months later, after carefully reducing the observations, you are ready to gaze in awe at your transmission spectrum.
# 
# First, you load all the usual stellar and planetary properties for this system.

# %%
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J

#***** Define stellar properties *****#

R_s = 1.21*R_Sun      # Stellar radius (m)
T_s = 5605.0          # Stellar effective temperature (K)
Met_s = -0.04         # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.56        # Stellar log surface gravity (log10(cm/s^2) by convention)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, stellar_spectrum = True, stellar_grid = 'phoenix')

F_s = star['F_star']
wl_s = star['wl_star']
R_s = star['stellar_radius']


#***** Define planet properties *****#

planet_name = 'WASP-77Ab'  # Planet name used for plots, output files etc.

R_p = 1.21*R_J      # Planetary radius (m)
M_p = 0.07*M_J      # Mass of planet (kg)
g_p = 4.3712        # Gravitational field of planet (m/s^2)
T_eq = 1043.8       # Equilibrium temperature (K)

# Create the planet object
planet = create_planet(planet_name, R_p, mass = M_p, gravity = g_p, T_eq = T_eq)

# If distance not specified, use fiducial value
if (planet['system_distance'] is None):
    planet['system_distance'] = 1    # This value only used for flux ratios, so it cancels
d = planet['system_distance']

# %% [markdown]
# ### Creating a Retrieval Model
# 
# Now comes the creative part: what model do you try first to fit WASP-999b's transmission spectrum?
# 
# Given the a priori known low density of the planet, you conclude it is reasonable to assume this is a giant planet dominated by $\rm{H}_2$ and $\rm{He}$. Looking at your data above, especially the **huge** absorption feature in the infrared around 1.4 μm, you guess that $\rm{H}_2 \rm{O}$ could be present (based on theoretical predictions or after looking up its cross section).
# 
# So for a first attempt, you consider a model with $\rm{H}_2 \rm{O}$, an isothermal temperature profile, and no clouds.

# %%
from POSEIDON.core import define_model, wl_grid_constant_R
from POSEIDON.utility import read_high_res_data
#***** Define model *****#

model_name = 'High-res retrieval'  # Model name used for plots, output files etc.

bulk_species = ['H2', 'He']     # H2 + He comprises the bulk atmosphere
param_species = ['H2O', 'CO']  # H2O, CO as in Brogi & Line

# Create the model object
model = define_model(model_name, bulk_species, param_species,
                    PT_profile = 'Madhu', high_res=True)

# Check the free parameters defining this model
print("Free parameters: " + str(model['param_names']))

#***** Wavelength grid *****#

wl_min = 1.3      # Minimum wavelength (um)
wl_max = 2.6      # Maximum wavelength (um)
R = 250000          # Spectral resolution of grid

# wl = wl_grid_line_by_line(wl_min, wl_max)
wl = wl_grid_constant_R(wl_min, wl_max, R)

data_dir = './reference_data/observations/WASP77-Ab'         # Special directory for this tutorial

data = read_high_res_data(data_dir)
# %% [markdown]
# ### Setting Retrieval Priors
# 
# One of the most important aspects in any Bayesian analysis is deciding what priors to use for the free parameters. Specifying a prior has two steps: (i) choosing the type of probability distribution; and (ii) choosing the allowable range.
# 
# Most free parameters in atmospheric retrievals with POSEIDON use the following prior types:
# 
# - Uniform: you provide the minimum and maximum values for the parameter.
# - Gaussian: you provide the mean and standard deviation for the parameter.
# 
# <div class="alert alert-info">
# 
#   **Note:**
# 
#   If you do not specify a prior type or range for a given parameter, POSEIDON will ascribe a default prior type (generally uniform) and a 'generous' range.
# 
# </div>
# 
# **Priors for WASP-999b**
# 
# Your first retrieval is defined by three free parameters: (1) the isothermal atmospheric temperature; (2) the radius at the (fixed) reference pressure; and (3) the log-mixing ratio of $\rm{H}_2 \rm{O}$. Since you don't have any *a priori* information on WASP-999b's atmosphere, you decide to use uniform priors for the three parameters. 
# 
# You think a reasonable prior range for the temperature of this hot Jupiter is $400 \, \rm{K}$ to $(T_{\rm{eq}} + 200 \, \rm{K}) = 1600 \, \rm{K}$. For the reference radius, you choose a wide range from 85\% to 115% of the observed white light radius. Finally, for the $\rm{H}_2 \rm{O}$ abundance you ascribe a very broad range from $10^{-12}$ to 0.1. 

# %%
from POSEIDON.core import set_priors

#***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types['T_ref'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['log_H2O'] = 'uniform'
prior_types['log_CO'] = 'uniform'
prior_types['a1'] = 'uniform'
prior_types['a2'] = 'uniform'
prior_types['log_P1'] = 'uniform'
prior_types['log_P2'] = 'uniform'
prior_types['log_P3'] = 'uniform'
prior_types['K_p'] = 'uniform'
prior_types['V_sys'] = 'uniform'

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges['T_ref'] = [500, 2000]
prior_ranges['R_p_ref'] = [0.85*R_p, 1.15*R_p]
prior_ranges['log_H2O'] = [-12, 0]
prior_ranges['log_CO'] = [-12, 0]
prior_ranges['a1'] = [0.02, 1]
prior_ranges['a2'] = [0.02, 1]
prior_ranges['log_P1'] = [-5.5, 2.5]
prior_ranges['log_P2'] = [-5.5, 2.5]
prior_ranges['log_P3'] = [-2, 2]
prior_ranges['K_p'] = [180, 220]
prior_ranges['V_sys'] = [-20, 20]

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

# %% [markdown]
# ### Pre-load Opacities
# 
# The last step before running a retrieval is to pre-interpolate the cross sections for our model and store them in memory. For more details on this process, see the forward model tutorial.
# 
# <div class="alert alert-warning">
# 
#   **Warning:**
# 
#   Ensure the range of $T_{\rm{fine}}$ used for opacity pre-interpolation is at least as large as the desired prior range for temperatures to be explored in the retrieval. Any models with layer temperatures falling outside the range of $T_{\rm{fine}}$ will be automatically rejected (for retrievals with non-isothermal P-T profiles, this prevents unphysical profiles with negative temperatures etc.)
# 
# </div>
# 
# 
# 

# %%
from POSEIDON.core import read_opacities
import numpy as np

#***** Read opacity data *****#

opacity_treatment = 'opacity_sampling'

# Define fine temperature grid (K)
T_fine_min = 500     # 400 K lower limit suffices for a typical hot Jupiter
T_fine_max = 3000    # 2000 K upper limit suffices for a typical hot Jupiter
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

# %% [markdown]
# ### Run Retrieval
# 
# You are now ready to run your first atmospheric retrieval!
# 
# Here we will use the nested sampling algorithm MultiNest to explore the parameter space. The key input quantity you need to provide to MultiNest is called the *number of live points*, $N_{\rm{live}}$, which determines how finely the parameter space will be sampled (and hence the number of computed spectra). For exploratory retrievals, $N_{\rm{live}} = 400$ usually suffices. For publication-quality results, $N_{\rm{live}} = 2000$ is reasonable. 
# 
# This high_resolutional POSEIDON retrieval should take about 3.3 hours on 24 core.
# 
# <div class="alert alert-info">
# 
#   **Tip:**
# 
#   Retrievals run faster on multiple cores. When running the cells in this Jupyter notebook, only a single core will be used. You can run a multi-core retrieval on 4 cores by converting this Jupyter notebook into a python script, then calling mpirun on the .py file:
# 
#   ```
#   mpirun -n 24 python -u YOUR_RETRIEVAL_SCRIPT.py
#   ```
#   
# </div>

# %%
from POSEIDON.retrieval import run_retrieval

#***** Specify fixed atmospheric settings for retrieval *****#

# Atmospheric pressure grid
P_min = 1.0e-7    # 0.1 ubar
P_max = 100       # 100 bar
N_layers = 100    # 100 layers

# Let's space the layers uniformly in log-pressure
P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure and radius
P_ref = 10.0   # Reference pressure (bar)

#***** Run atmospheric retrieval *****#

run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R = R, 
                spectrum_type = 'direct_emission', sampling_algorithm = 'MultiNest', 
                N_live = 500, verbose = True, N_output_samples = 400, resume = 'True', high_res = True)


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