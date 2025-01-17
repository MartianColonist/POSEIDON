# %% [markdown]
# # Retrieval: High Resolution Transmission Spectroscopy on WASP-121b
# 
# 
# This tutorial covers how to run a retrieval with high resolution data using POSEIDON. Before you run this notebook, you should run transimission_cross_correlate.ipynb first to preprocess the data. If you have data_processed.hdf5 saved in your planet directory, you are all set!

# %% [markdown]
# ### Loading Data
# 
# First, we will load the processed data for your planet (WASP-121b). For more information about this dataset and to learn the basics of high-resolution cross correlation spectroscopy, see transimission_cross_correlate.ipynb.

# %%
from POSEIDON.high_res import read_high_res_data

planet_name = 'WASP-121b'

data_dir = '../../../POSEIDON/reference_data/observations/' + planet_name # the directory where you've put the data


data = read_high_res_data(data_dir, names=["blue"])  # only use blue arm for faster retrieval
data = read_high_res_data(data_dir, names=["blue", "redl", "redu"])

# %% [markdown]
# ## Setting up retrieval model
# 
# Now, let's provide the wavelength grid and properties of the host star and your planet. The wavelength range should match the range of your data. The blue arm spans 0.37 microns to 0.51 microns. If you decide to use both blue and red arms, you should increase the range to 0.37 microns to 0.87 microns.
# 
# We use R=250,000 as a tradeoff between computational speed and accuracy. For more discussion, see the previous tutorial.

# %%
from POSEIDON.core import define_model, wl_grid_constant_R
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J

wl_min = 0.37  # Minimum wavelength (um)
wl_max = 0.51  # Maximum wavelength (um) for blue arm
wl_max = 0.87  # change to include red arm
R = 250000  # Spectral resolution of grid
wl = wl_grid_constant_R(wl_min, wl_max, R)

# ***** Define stellar properties *****#

R_s = 1.458 * R_Sun  # Stellar radius (m)
T_s = 6776  # Stellar effective temperature (K)
Met_s = 0.13  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.24  # Stellar log surface gravity (log10(cm/s^2) by convention)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, wl=wl, stellar_grid="phoenix")
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

# %% [markdown]
# Existing literature have shown detection of $\rm{Fe}$ in the atmosphere of WASP-121b. There are strong $\rm{Fe}$ absorption features in the wavelength range as well.
# 
# So for a first attempt, we consider a model with $\rm{Fe}$, an isothermal temperature profile, and no clouds.
# 
# For additional parameters used in high resolution retrieval, we include: $a$ (the scale parameter), $b$ (the scale parameter for noise), $K_p$ (the Keplerian orbital velocity), $V_{sys}$ (the systematic velocity), and $W_{conv}$ (the broadening kernel width). You can opt to use the MLE estimator of $\beta$ and not include it as a free parameter, which we are going to do here. [Gibson et al. 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.4618G/abstract) contains a discussion on this choice. An additional parameter available is $\Delta \phi$, which offsets the ephemeris. However $\Delta \phi$ is very degenerate with $V_{sys}$ if the range of covered orbital phase is small.
# 
# Be sure to reference [Gibson et al. 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.4618G/abstract) if you want a more detailed description of these parameters.

# %%
from POSEIDON.core import define_model, wl_grid_constant_R

# ***** Define model *****#

model_name = "High-res retrieval 2"  # Model name used for plots, output files etc.

bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere
# param_species = ["Fe", "Cr", "Mg", "V", "Ti"] # Add more chemical species to the model here
param_species = ["Fe"]

method = "sysrem"
high_res_params = ["log_alpha", "K_p", "V_sys", "W_conv"]

# Create the model object
model = define_model(
    model_name,
    bulk_species,
    param_species,
    PT_profile="isotherm",
    high_res_params=high_res_params,
    reference_parameter="R_p_ref",
    high_res_method="sysrem",
)

# Check the free parameters defining this model
print("Free parameters: " + str(model["param_names"]))

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
# 
# Your first retrieval is defined by seven free parameters printed above: (1) the isothermal atmospheric temperature; (2) the radius at the (fixed) reference pressure; (3) the log-mixing ratio of $\rm{Fe}$; and the four high resolution parameters. Since you don't have any *a priori* information on WASP-121b's atmosphere, you decide to use uniform priors for all the parameters.

# %%
from POSEIDON.core import set_priors

# ***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types["T"] = "uniform"
prior_types["R_p_ref"] = "gaussian"
prior_types["log_X"] = "uniform"
prior_types["K_p"] = "uniform"
prior_types["V_sys"] = "uniform"
prior_types["log_alpha"] = "uniform"
prior_types["b"] = "uniform"
prior_types["W_conv"] = "uniform"

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges["T"] = [1000, 4000]
prior_ranges["R_p_ref"] = [R_p, 0.05 * R_J]
prior_ranges["log_X"] = [-15, 0]
prior_ranges["K_p"] = [170, 230]
prior_ranges["V_sys"] = [-10, 10]
prior_ranges["log_alpha"] = [-1, 2]
prior_ranges["b"] = [0.1, 10]
prior_ranges["W_conv"] = [1, 50]

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

# %%
from POSEIDON.core import read_opacities
import numpy as np

# ***** Read opacity data *****#

opacity_treatment = "opacity_sampling"

# Define fine temperature grid (K)
T_fine_min = 1000  # 400 K lower limit suffices for a typical hot Jupiter
T_fine_max = 4000  # 2000 K upper limit suffices for a typical hot Jupiter
T_fine_step = 50  # 20 K steps are a good tradeoff between accuracy and RAM

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

# %% [markdown]
# ## Run Retrieval
# 
# You are now ready to run your first high resolution atmospheric retrieval with SYSREM!
# 
# Here we will use the nested sampling algorithm MultiNest to explore the parameter space. The key input quantity you need to provide to MultiNest is called the *number of live points*, $N_{\rm{live}}$, which determines how finely the parameter space will be sampled (and hence the number of computed spectra). For exploratory retrievals, $N_{\rm{live}} = 400$ usually suffices. For publication-quality results, $N_{\rm{live}} = 2000$ is reasonable. 
# 
# <div class="alert alert-info">
# 
#   **Tip:**
# 
#   Retrievals run faster on multiple cores. When running the cells in this Jupyter notebook, only a single core will be used. You can run a multi-core retrieval on 4 cores by converting this Jupyter notebook into a python script, then calling mpirun on the .py file:
# 
#   ```
#   mpirun -n 4 python -u YOUR_RETRIEVAL_SCRIPT.py
#   ```
#   
# </div>
# 
# 
# <div class="alert alert-info">
# 
#   **Important Note:**
#   A high resolution forward model is computationally expensive (~1 second per model). With 400 live points, it takes ~100,000 evalutations for the model to converge. With 36 cores, this amounts to ~1 hour. Therefore, instead of waiting for the next cell to finish, you could convert the notebook into a '.py' file and run it with multiple cores in command line. Just make sure you run the cell below and wait for a couple of minutes. Once it says "live points generated" and still no error, you are good to run it on multiple cores!
#   
# </div>

# %%
from POSEIDON.retrieval import run_retrieval

# ***** Specify fixed atmospheric settings for retrieval *****#

# Atmospheric pressure grid
P_min = 1e-12  # 0.1 ubar
P_max = 100  # 100 bar
N_layers = 100  # 100 layers

# Let's space the layers uniformly in log-pressure
P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure and radius
P_ref = 1e-2  # Reference pressure (bar)

# ***** Run atmospheric retrieval *****#

run_retrieval(
    planet,
    star,
    model,
    opac,
    data,
    priors,
    wl,
    P,
    P_ref,
    R=R,
    spectrum_type="transmission",
    sampling_algorithm="MultiNest",
    N_live=400,
    verbose=True,
    N_output_samples=1000,
    resume=False,
    ev_tol=0.05,
)

# %%
# Generate a corner plot after the retrieval is finished
from POSEIDON.corner import generate_cornerplot

fig_corner = generate_cornerplot(planet, model)

# %%
# Read retrieved PT profile and plot it
from POSEIDON.utility import read_retrieved_PT
from POSEIDON.visuals import plot_PT_retrieved

P, T_low2, T_low1, T_median, T_high1, T_high2 = read_retrieved_PT(
    planet_name, model_name, retrieval_name=None
)
PT_median = [(T_median, P)]
PT_low2 = [(T_low2, P)]
PT_low1 = [(T_low1, P)]
PT_high1 = [(T_high1, P)]
PT_high2 = [(T_high2, P)]


plot_PT_retrieved(
    planet_name,
    PT_median,
    PT_low2,
    PT_low1,
    PT_high1,
    PT_high2,
    # T_true=None, # Uncomment this line if you have a PT profile to compare to
    Atmosphere_dimension=1,
    TwoD_type=None,
    plt_label=None,
    show_profiles=[],
    PT_labels=[],
    # colour_list=[], # Uncomment this line if you want to specify colors
    log_P_min=None,
    log_P_max=None,
    T_min=2000,
    T_max=4000,
    legend_location="lower left",
)


