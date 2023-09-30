# %%
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J
import pickle
from POSEIDON.visuals import *

# ***** Define stellar properties *****#
R_s = 1.21 * R_Sun  # Stellar radius (m)
T_s = 5605.0  # Stellar effective temperature (K)
Met_s = -0.04  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.56  # Stellar log surface gravity (log10(cm/s^2) by convention)
# ***** Define planet properties *****#

planet_name = "WASP-77Ab"  # Planet name used for plots, output files etc.

R_p = 1.21 * R_J  # Planetary radius (m)
M_p = 0.07 * M_J  # Mass of planet (kg)
g_p = 4.3712  # Gravitational field of planet (m/s^2)
T_eq = 1043.8  # Equilibrium temperature (K)

# Create the planet object
planet = create_planet(planet_name, R_p, mass=M_p, gravity=g_p, T_eq=T_eq)

from POSEIDON.core import define_model

bulk_species = ["H2", "He"]  # H2 + He comprises the bulk atmosphere
param_species = ["H2O", "CO"]  # H2O, CO as in Brogi & Line

method = "pca"
# high_res_params = ['a', 'b', 'dPhi', 'K_p', 'V_sys', 'W_conv']
high_res_params = ["K_p", "V_sys", "a"]

model_1 = define_model(
    "H2O, CO retrieval",
    bulk_species,
    param_species,
    PT_profile="Madhu",
    high_res_params=high_res_params,
    reference_parameter="R_p_ref",
)


model_2 = define_model(
    "H2O, CO retrieval 400",
    bulk_species,
    param_species,
    PT_profile="Madhu",
    high_res_params=high_res_params,
    reference_parameter="R_p_ref",
)
# %%
params_to_plot = ["R_p_ref", "log_H2O", "log_CO", "K_p", "V_sys", "a"]

from POSEIDON.corner import generate_cornerplot

fig_1 = generate_cornerplot(
    planet,
    model_1,
    params_to_plot=params_to_plot,
    colour_scheme="blue",
)

# %%
fig_2 = generate_cornerplot(
    planet, model_2, params_to_plot=params_to_plot, existing_fig=fig_1
)
# %%
