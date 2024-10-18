# %%
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J

evaluate = False
# ***** Define stellar properties *****#

R_s = 1.458 * R_Sun  # Stellar radius (m)
T_s = 6776  # Stellar effective temperature (K)
Met_s = 0.13  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.24  # Stellar log surface gravity (log10(cm/s^2) by convention)

# Create the stellar object
star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="phoenix")

F_s = star["F_star"]
wl_s = star["wl_star"]

# ***** Define planet properties *****#

planet_name = "WASP-121b"  # Planet name used for plots, output files etc.

R_p = 0.118 * R_s  # Planetary radius (m)
M_p = 1.157 * M_J  # Mass of planet (kg)
g_p = 10 ** (2.97 - 2)  # Gravitational field of planet (m/s^2)
T_eq = 2450  # Equilibrium temperature (K)

# Create the planet object
planet = create_planet(planet_name, R_p, mass=M_p, gravity=g_p, T_eq=T_eq)

# If distance not specified, use fiducial value
if planet["system_distance"] is None:
    planet["system_distance"] = 1  # This value only used for flux ratios, so it cancels
d = planet["system_distance"]

# %%
from POSEIDON.core import define_model, wl_grid_constant_R, make_atmosphere

# ***** Define model *****#

model_name = "Test"  # Model name used for plots, output files etc.

bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere
param_species = []

# model_name = "Fe Cr V Mg deck_haze 2D"  # Model name used for plots, output files etc.

# bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere
# param_species = ["Fe", "Cr", "V", "Mg"]
# DN_vary = ["Fe", "Cr", "V", "Mg"]

high_res = "sysrem"
high_res_params = ["K_p", "V_sys", "W_conv", "log_alpha"]

model = define_model(
    model_name,
    bulk_species,
    param_species,
    PT_profile="isotherm",
    high_res_params=high_res_params,
    reference_parameter="R_p_ref",
    high_res_method="sysrem",
    # cloud_model="MacMad17",
    # cloud_type="deck_haze",
    # TwoD_type="D-N",
    # PT_dim=2,
    # X_dim=2,
    # species_DN_gradient=DN_vary,
    # sharp_DN_transition=True,
)

model["spectrum_type"] = "transmission"
# Check the free parameters defining this model
print("Free parameters: " + str(model["param_names"]))

# ***** Wavelength grid *****#
wl_min = 0.37  # Minimum wavelength (um)
wl_max = 0.87  # Maximum wavelength (um)
R = 250000  # Spectral resolution of grid

# wl = wl_grid_line_by_line(wl_min, wl_max)
wl = wl_grid_constant_R(wl_min, wl_max, R)

from POSEIDON.high_res import read_high_res_data

names = ["blue", "redu", "redl"]
# ***** Read in data *****#
data = read_high_res_data("./data/WASP-121b", names)
# %%
from POSEIDON.core import set_priors

# ***** Set priors for retrieval *****#
prior_types = {}
# Specify whether priors are linear, Gaussian, etc.
prior_types["T_bar_DN_high"] = "uniform"
prior_types["T"] = "uniform"
prior_types["Delta_T_DN_high"] = "uniform"
prior_types["T_deep"] = "uniform"
prior_types["R_p_ref"] = "gaussian"
prior_types["log_X"] = "uniform"
prior_types["Delta_log_X"] = "uniform"
prior_types["log_P_cloud"] = "uniform"
prior_types["K_p"] = "uniform"
prior_types["V_sys"] = "uniform"
prior_types["log_alpha"] = "uniform"
prior_types["b"] = "uniform"
prior_types["alpha"] = "uniform"
prior_types["W_conv"] = "uniform"

prior_ranges = {}
# Specify prior ranges for each free parameter
prior_ranges["T_bar_DN_high"] = [1000, 4000]
prior_ranges["Delta_T_DN_high"] = [-2000, 2000]
prior_ranges["T"] = [1000, 4000]
prior_ranges["T_deep"] = [2000, 4000]
prior_ranges["R_p_ref"] = [R_p, 0.05 * R_J]
prior_ranges["log_X"] = [-15, 0]
prior_ranges["Delta_log_X"] = [-5, 5]
prior_ranges["log_P_cloud"] = [-4, 2]
prior_ranges["K_p"] = [0, 400]
prior_ranges["V_sys"] = [-50, 50]
prior_ranges["log_alpha"] = [-2, 2]
prior_ranges["b"] = [0.00001, 10]
prior_ranges["alpha"] = [1, 10]
prior_ranges["W_conv"] = [0.1, 20]

# Create prior object for retrieval
priors = set_priors(planet, star, model, data, prior_types, prior_ranges)

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
# R_p_ref = R_p

# ***** Run atmospheric retrieval *****#
if not evaluate:
    # Now we can pre-interpolate the sampled opacities (may take up to a minute)
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)
    run_retrieval(
        planet,
        star,
        model,
        opac,
        data,
        priors,
        wl,
        P,
        P_ref=P_ref,
        R_p_ref=R_p,
        R=R,
        spectrum_type="transmission",
        sampling_algorithm="MultiNest",
        N_live=400,
        verbose=True,
        N_output_samples=1000,
        resume=False,
    )


# from POSEIDON.visuals import *

# plot_histograms(
#     planet_name,
#     [model],
#     ["log_Fe", "log_Cr", "log_V", "log_Ti", "log_Mg"],
#     N_bins=[10, 10, 10, 10, 10],
#     N_rows=1,
#     N_columns=5,
# )

# %%
from POSEIDON.utility import read_retrieved_PT, read_retrieved_log_X
from POSEIDON.visuals import plot_PT, plot_PT_retrieved, plot_chem_retrieved
from POSEIDON.corner import generate_cornerplot


# params = (-6, -6, -6, 0.1, 0.3, -1, -2, 2, 3000)
# log_Fe, log_Cr, log_Mg, a1, a2, log_P1, log_P2, log_P3, T_ref = params

# # Provide a specific set of model parameters for the atmosphere
# PT_params = np.array([a1, a2, log_P1, log_P2, log_P3, T_ref])
# log_X_params = np.array([[log_Fe, log_Cr, log_Mg]])

# atmosphere = make_atmosphere(
#     planet,
#     define_model(
#         model_name,
#         bulk_species,
#         param_species,
#         PT_profile="Madhu",
#         high_res_params=high_res_params,
#     ),
#     P,
#     P_ref,
#     R_p,
#     PT_params,
#     log_X_params,
# )

# # ***** Plot retrieved transmission spectrum *****#

# # Read retrieved spectrum confidence regions

# P, T_low2, T_low1, T_median, T_high1, T_high2 = read_retrieved_PT(
#     planet_name, model_name, retrieval_name=None
# )
# PT_median = [(T_median, P)]
# PT_low2 = [(T_low2, P)]
# PT_low1 = [(T_low1, P)]
# PT_high1 = [(T_high1, P)]
# PT_high2 = [(T_high2, P)]


import colormaps as cmaps
import cmasher as cmr

cmap = cmr.get_sub_cmap("cmr.sapphire", 0.1, 0.9)  # cmaps.lapaz

colors = cmr.take_cmap_colors(
    "cmr.sapphire", 10, cmap_range=(0.1, 0.9), return_fmt="hex"
)

color = colors[5]

# plot_PT_retrieved(
#     planet_name,
#     PT_median,
#     PT_low2,
#     PT_low1,
#     PT_high1,
#     PT_high2,
#     T_true=atmosphere["T"].reshape(-1),
#     # T_true=None,
#     Atmosphere_dimension=1,
#     TwoD_type=None,
#     plt_label=None,
#     show_profiles=[],
#     PT_labels=[],
#     colour_list=[color],
#     log_P_min=None,
#     log_P_max=None,
#     T_min=2000,
#     T_max=4000,
#     legend_location="lower left",
# )

# ***** Make corner plot *****#
import matplotlib.pyplot as plt

fig_corner = generate_cornerplot(
    planet,
    model,
    # params_to_plot=["log_Fe", "log_Cr", "log_V", "log_Ti", "log_Mg"],
    # true_vals=[2000, 1000, 3000, -6, 3, -200, -20, 1, 0, None],
    # span=[
    #     (1.5, 1.9),
    #     # (0, 2),
    #     # (0, 2),
    #     # (-5, 2),
    #     # (-5, 2),
    #     # (-2, 2),
    #     (2000, 4000),
    #     (-8, -3),
    #     (-4, 0),
    #     (-210, -190),
    #     (-22, -18),
    #     (0, 4),
    #     (0, 10),
    #     (0.8464, 0.8476),
    # ],
    # span=[(-8, -4), (-12, -5), (-12, -8), (-15, -9), (-8, -2)],
    colour_scheme=color,
)
