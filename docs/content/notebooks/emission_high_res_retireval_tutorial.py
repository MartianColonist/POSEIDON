# %%
from POSEIDON.high_res import read_high_res_data, fit_uncertainties
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
import os

data_dir = "/home/rwang/POSEIDON_high_res/final_experiments/data/WASP-77Ab/"  # Special directory for this tutorial

name = "IGRINS"
data = read_high_res_data(data_dir, names=[name])

# %%
wl_grid = data[name]["wl_grid"]
flux = data[name]["flux"]
phi = data[name]["phi"]
residuals = data[name]["residuals"]
V_bary = data[name]["V_bary"]
uncertainties = fit_uncertainties(flux, initial_guess=[0.01, np.mean(flux)])

# %%
cmap = cmr.get_sub_cmap("cmr.sepia", 0.1, 0.9)
# %% [markdown]
# ### Define Wavelength Grid, Stellar and Planet Properties
#
# Then, let's provide the wavelength grid and properties of the host star and your planet. The wavelength range should match the range of your data. This observation spans 0.37 microns to 0.51 microns.

# %%
from POSEIDON.core import (
    create_star,
    create_planet,
    make_atmosphere,
    read_opacities,
    compute_spectrum,
    define_model,
    wl_grid_constant_R,
)

from POSEIDON.constants import R_Sun, R_J, M_J

# ***** Wavelength grid *****#

wl_min = 1.3  # Minimum wavelength (um)
wl_max = 2.6  # Maximum wavelength (um)
R = 250000  # Spectral resolution of grid

wl = wl_grid_constant_R(wl_min, wl_max, R)

# ***** Define stellar properties *****#

R_s = 0.91 * R_Sun  # Stellar radius (m)
T_s = 5605.0  # Stellar effective temperature (K)
Met_s = -0.04  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.48  # Stellar log surface gravity (log10(cm/s^2) by convention)

star = create_star(R_s, T_s, log_g_s, Met_s, wl=wl, stellar_grid="phoenix")

# ***** Define planet properties *****#

planet_name = "WASP-77Ab"  # Planet name used for plots, output files etc.

R_p = 1.21 * R_J  # Planetary radius (m)
M_p = 1.76 * M_J  # Mass of planet (kg)

# Create the planet object
planet = create_planet(planet_name, R_p, mass=M_p)

# If distance not specified, use fiducial value
if planet["system_distance"] is None:
    planet["system_distance"] = 1  # This value only used for flux ratios, so it cancels
d = planet["system_distance"]

# %% [markdown]
# ### Creating a Retrieval Model
#
# Existing literature have shown detection of $\rm{H}_2\rm{O}$ and $\rm{C}\rm{O}_2$ in the atmosphere of WASP-77Ab.
#
# So for a first attempt, we consider a model with $\rm{H}_2\rm{O}$ and $\rm{C}\rm{O}_2$, an isothermal temperature profile, and no clouds.
#
# For additional parameters used in high resolution retrieval, we include: $log_alpha$ (the scaling parameter), $K_p$ (the Keplerian orbital velocity), $V_{sys}$ (the systematic velocity), and $W_{conv}$ (width of the gaussian convolution kernel used for line broadening). Additional parameters available are $\Delta \phi$ (offseting the ephemeris) and b (the scaling parameter for noise).

# %%
model_name = "Tutorial"  # Model name used for plots, output files etc.
bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, wl=wl, stellar_grid="phoenix")
param_species = ["H2O", "CO"]
high_res_params = ["K_p", "V_sys", "log_alpha", "W_conv"]

model = define_model(
    model_name,
    bulk_species,
    param_species,
    PT_profile="Madhu",
    high_res_method="sysrem",
    high_res_params=high_res_params,
    reference_parameter="R_p_ref",
)

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
# ***** Wavelength grid *****#

wl_min = 1.3  # Minimum wavelength (um)
wl_max = 2.6  # Maximum wavelength (um)
R = 250000  # Spectral resolution of grid
wl = wl_grid_constant_R(wl_min, wl_max, R)

opacity_treatment = "opacity_sampling"

# Define fine temperature grid (K)
T_fine_min = 400  # 400 K lower limit suffices for a typical hot Jupiter
T_fine_max = 4000  # 2000 K upper limit suffices for a typical hot Jupiter
T_fine_step = 50  # 20 K steps are a good tradeoff between accuracy and RAM

T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -5.0  # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2  # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2  # 0.2 dex steps are a good tradeoff between accuracy and RAM

log_P_fine = np.arange(
    log_P_fine_min, (log_P_fine_max + log_P_fine_step), log_P_fine_step
)

opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

# %% [markdown]
# ### Generating an emission spectrum and cross-correlating with data
# To do so, you need to specify the atmosphere setting and provide values for your model parameters. We can the cross-correlate the spectrum we created with the processed data we obtained from running "prepare_high_res_data". We can plot the cross-correlating function as a function of $K_p$ and $V_{sys}$ and see the peak at the expected location.
# If you want to run retrieval only, you can remove the cells from this section.

# %%
# ***** Specify fixed atmospheric settings for retrieval *****#
# Specify the pressure grid of the atmosphere
P_min = 1.0e-5  # 0.1 ubar
P_max = 100  # 100 bar
N_layers = 100  # 100 layers

# We'll space the layers uniformly in log-pressure
P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

# Specify the reference pressure and radius
P_ref = 1e-2  # Reference pressure (bar)
R_p_ref = R_p  # Radius at reference pressure

log_species = [-4, -4]

# Provide a specific set of model parameters for the atmosphere
PT_params = np.array(
    [0.2, 0.1, 0.17, -1.39, 1, 1500]
)  # a1, a2, log_P1, log_P2, log_P3, T_top
log_X_params = np.array([log_species])

atmosphere = make_atmosphere(
    planet, model, P, P_ref, R_p_ref, PT_params, log_X_params, P_param_set=1e-5
)

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
# Your first retrieval is defined by three free parameters: (1) the isothermal atmospheric temperature; (2) the radius at the (fixed) reference pressure; (3) the log-mixing ratio of $\rm{Fe}$; and the four high resolution parameters. Since you don't have any *a priori* information on WASP-121b's atmosphere, you decide to use uniform priors for all the parameters.

# %%
from POSEIDON.core import set_priors

# ***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types["T_ref"] = "uniform"
prior_types["R_p_ref"] = "uniform"
prior_types["log_X"] = "uniform"
prior_types["a1"] = "uniform"
prior_types["a2"] = "uniform"
prior_types["log_P1"] = "uniform"
prior_types["log_P2"] = "uniform"
prior_types["log_P3"] = "uniform"
prior_types["K_p"] = "uniform"
prior_types["V_sys"] = "uniform"
prior_types["log_alpha"] = "uniform"
prior_types["W_conv"] = "uniform"

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges["T_ref"] = [500, 2000]
prior_ranges["R_p_ref"] = [0.5 * R_p, 1.5 * R_p]
prior_ranges["log_X"] = [-15, 0]
prior_ranges["a1"] = [0, 1]
prior_ranges["a2"] = [0, 1]
prior_ranges["log_P1"] = [-5, 2]
prior_ranges["log_P2"] = [-5, 2]
prior_ranges["log_P3"] = [-2, 2]
prior_ranges["K_p"] = [150, 250]
prior_ranges["V_sys"] = [-50, 50]
prior_ranges["log_alpha"] = [-2, 2]
prior_ranges["W_conv"] = [0, 50]

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

# %% [markdown]
# ### Run Retrieval
#
# You are now ready to run your first high resolution atmospheric retrieval with SYSREM!
#
# Here we will use the nested sampling algorithm MultiNest to explore the parameter space. The key input quantity you need to provide to MultiNest is called the *number of live points*, $N_{\rm{live}}$, which determines how finely the parameter space will be sampled (and hence the number of computed spectra). For exploratory retrievals, $N_{\rm{live}} = 400$ usually suffices. For publication-quality results, $N_{\rm{live}} = 2000$ is reasonable.
#
# <div class="alert alert-info">
#
#   **Tip:**
#
#   Retrievals run faster on multiple cores. When running the cells in this Jupyter notebook, only a single core will be used. You can run a multi-core retrieval on 24 cores by converting this Jupyter notebook into a python script, then calling mpirun on the .py file:
#
#   ```
#   mpirun -n 24 python -u YOUR_RETRIEVAL_SCRIPT.py
#   ```
#
# </div>
#
#
# <div class="alert alert-info">
#
#   **Important Note:**
#   A high resolution forward model is computationally expensive (~1 second per model). With 400 live points, it took > $10^6$ evalutations for the model to converge. This retrieval could be finished with ~8 hours on 24 cores. You should consider convert the notebook into a '.py' file and run it with multiple cores in command line.
#
# </div>

# %%
from POSEIDON.retrieval import run_retrieval

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
    R_p_ref=R_p,
    R=R,
    spectrum_type="emission",
    sampling_algorithm="MultiNest",
    N_live=400,
    verbose=True,
    N_output_samples=1000,
    resume=False,
    ev_tol=0.5,
)
