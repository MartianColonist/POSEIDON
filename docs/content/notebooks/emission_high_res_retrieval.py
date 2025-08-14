from POSEIDON.high_res import read_high_res_data

planet_name = 'WASP-77Ab'

data_dir = '../../../POSEIDON/reference_data/observations/' + planet_name # the directory where you've put the data

data = read_high_res_data(data_dir, names=["IGRINS"])  # we named the dataset IGRINS in the previous notebook

# %%
from POSEIDON.core import define_model, wl_grid_constant_R
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J

# ***** Wavelength grid *****#

wl_min = 1.3  # Minimum wavelength (um)
wl_max = 2.6  # Maximum wavelength (um)
R = 250000  # Change the spectral resolution of grid here.
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

model_name = "Retrieval"  # Model name used for plots, output files etc.
bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere

param_species = ["H2O", "CO"]

model = define_model(
    model_name,
    bulk_species,
    param_species,
    PT_profile="Madhu",
    reference_parameter="R_p_ref",
    high_res_method="sysrem", # should be the same as the method used to preprocess the data
    alpha_high_res_option = 'log',
    fix_alpha_high_res = False, 
    fix_W_conv_high_res = False, 
    fix_beta_high_res = True, 
    fix_Delta_phi_high_res = True,
)

# Check the free parameters defining this model
print("Free parameters: " + str(model["param_names"]))

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
prior_types["log_alpha_HR"] = "uniform"
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
prior_ranges["log_alpha_HR"] = [-2, 2]
prior_ranges["W_conv"] = [0, 50]

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)


from POSEIDON.core import read_opacities
import numpy as np

# ***** Read opacity data *****#
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

from POSEIDON.retrieval import run_retrieval

# ***** Specify fixed atmospheric settings for retrieval *****#

# Atmospheric pressure grid
P_min = 1e-5  # 0.1 ubar
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
