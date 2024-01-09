#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forward.py

Run the forward model to simulate multiple visits with stellar contamination and a planetary atmosphere.

Adapted from https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/transmission_basic.html

@author: MartianColonist, lim
"""
# Imports
import matplotlib.pyplot as plt
from astropy import constants as astrocst
from POSEIDON.core import load_data, create_star
from POSEIDON.core import create_planet
from POSEIDON.constants import R_Sun, R_J, M_J, R_E, M_E
from POSEIDON.core import define_model
from POSEIDON.core import make_atmosphere
import numpy as np
from POSEIDON.core import read_opacities, wl_grid_constant_R
from POSEIDON.core import compute_spectrum
from POSEIDON.stellar import stellar_contamination
from POSEIDON.instrument import generate_syn_data_from_user


# =====================================================================================================================
# Define variables
# =====================================================================================================================
# ***** Define stellar properties *****#
R_s = 0.1192 * R_Sun  # Stellar radius (m) (Agol+21)
T_s = 2566  # Stellar effective temperature (K) (Agol+21)
Met_s = 0.04  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)] (Delrez+18)
log_g_s = 5.2396  # Stellar log surface gravity (log10(cm/s^2) by convention) (Agol+21)

# ***** Define planet properties *****#
planet_name = 'forward_TRAPPIST-1f'  # Planet name used for plots, output files etc.
R_p = 1.045 * R_E  # Planetary radius (m) (Agol+21)
M_p = 1.039 * M_E  # Planetary mass (kg) (Agol+21)
g_p = astrocst.G * M_p / R_p ** 2  # Gravitational field of planet (m/s^2)
T_eq = 214.5  # Equilibrium temperature (K) (Delrez+18)

# ***** Define model *****#
model_name = 'Simple_model'  # Model name used for plots, output files etc.
bulk_species = ['N2']  # N2 comprises the bulk atmosphere
# param_species = ['H2O', 'CH4']  # The trace gases are H2O and CH4
param_species = ['CO2', 'H2']  # Trace gas(es)

# Specify the pressure grid of the atmosphere
P_min = 1.0e-7  # 0.1 ubar
P_max = 100  # 100 bar
N_layers = 100  # 100 layers

# Specify the reference pressure and radius
P_ref = 1.0  # Reference pressure (bar)
R_p_ref = R_p  # Radius at reference pressure

# Provide a specific set of model parameters for the atmosphere
PT_params = np.array([T_eq])  # T (K)
# log_X_params = np.array([-3.3, -5.0])  # log(H2O), log(CH4)
log_X_params = np.array([-3.3, -5.0])

# ***** Wavelength grid *****#
wl_min = 0.6  # Minimum wavelength (um)
wl_max = 3.0  # Maximum wavelength (um)
R = 10000  # Spectral resolution of grid

# ***** Read opacity data *****#
opacity_treatment = 'opacity_sampling'

# First, specify limits of the fine temperature and pressure grids for the
# pre-interpolation of cross sections. These fine grids should cover a
# wide range of possible temperatures and pressures for the model atmosphere.

# Define fine temperature grid (K)
T_fine_min = 100  # 400 K lower limit suffices for a typical hot Jupiter
T_fine_max = 500  # 2000 K upper limit suffices for a typical hot Jupiter
T_fine_step = 10  # 10 K steps are a good tradeoff between accuracy and RAM

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6.0  # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2.0  # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2  # 0.2 dex steps are a good tradeoff between accuracy and RAM

# Stellar contamination setup
stellar_contam = ['one_spot_free_log_g']
shared_stellar_contam = {0: 0}
f_het = .05
T_het = T_s + 150.
log_g_het = 5.0

# Gaussian scatter to add to the synthetic data (something small to simulate nearly noiseless data)
gauss_scatter = 10.0  # [ppm]
respow_data = 30  # resolving power of the synthetic data

use_pymsg = True

# Info on the output file
outdir = ('/home/olivia/projet/jwst/retrieval/POSEIDON/tests/POSEIDON_output/forward_TRAPPIST-1f/')
instruments = ['JWST_NIRISS_SOSS_Ord1']
label = 'bulk'
for bulk_species_i in bulk_species:
    label = label + '-' + bulk_species_i
label = label + '_param'
for param_species_i in param_species:
    label = label + '-' + param_species_i
label = label + '_fhet-{:.3f}_Thet-{:.1f}_logghet-{:.5f}'.format(f_het, T_het, log_g_het)
label = label + '_gaussscat-{:.1f}ppm'.format(gauss_scatter)
label = label + '_respow-{}'.format(respow_data)
label = label + '_pymsg{}'.format(int(use_pymsg))
outfname = planet_name + '_SYNTHETIC_' + instruments[0] + '_' + label + '.dat'

# =====================================================================================================================
# Define functions
# =====================================================================================================================


# =====================================================================================================================
# Start of code
# =====================================================================================================================
if __name__ == '__main__':
    # Create the stellar object
    if not use_pymsg:
        star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="phoenix", stellar_contam=stellar_contam[0],
                           f_het=f_het, T_het=T_het, log_g_het=log_g_het, interp_backend="pysynphot")
    else:
        star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="Goettingen-HiRes", stellar_contam=stellar_contam[0],
                           f_het=f_het, T_het=T_het, log_g_het=log_g_het, interp_backend="pymsg")
    print("Created star.")

    # Create the planet object
    planet = create_planet(planet_name, R_p, mass=M_p, gravity=g_p, T_eq=T_eq)
    print("Created planet.")

    # Create the model object
    model = define_model(model_name, bulk_species, param_species,
                         PT_profile='isotherm', cloud_model='cloud-free',
                         stellar_contam=stellar_contam, shared_stellar_contam=shared_stellar_contam,
                         radius_unit='R_E')
    print("Defined model.")

    # Check the free parameters defining this model
    print("Free parameters: " + str(model['param_names']))

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Generate the atmosphere
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref,
                                 PT_params, log_X_params)
    print("Made atmosphere.")

    # Create wavelength grid
    wl = wl_grid_constant_R(wl_min, wl_max, R)

    # Define fine temperature grid (K)
    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step),
                           log_P_fine_step)

    # Now we can pre-interpolate the sampled opacities (may take up to a minute)
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)
    print("Read opacities.")

    # Generate our first transmission spectrum
    spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                spectrum_type='transmission')
    print("Computed spectrum.")

    # Set up figure
    fig, ax = plt.subplots(nrows=3, figsize=(12, 10), sharex="col")

    # Set up first axis
    ax[0].set(ylabel="Transit depth [ppm]")
    ax[0].plot(wl, spectrum * 1e6, alpha=.5)  # Non-stellar-contaminated spectrum

    # Compute stellar contamination multiplicative factor and apply it to the spectrum
    epsilon = stellar_contamination(star, wl)
    spectrum *= epsilon
    print("Computed stellar contamination factor and contaminated spectrum.")

    # Set up second axis for contaminated spectrum
    ax[1].set(ylabel="Transit depth [ppm]")

    # Plot stellar contamination factor
    ax[2].set(xlabel="Wavelength [um]", ylabel=r"Stellar contamination factor $\epsilon$")
    ax[2].plot(wl, epsilon, alpha=.5)

    # Generate synthetic data (binned down to instrument resolution) using forward model
    generate_syn_data_from_user(planet, wl, spectrum, data_dir=outdir, instrument=instruments[0],
                                R_data=respow_data, err_data=gauss_scatter, wl_start=0.85, wl_end=2.8, label=label,
                                Gauss_scatter=True)
    print("Generated synthetic data based on instrument {}.".format(instruments[0]))

    data_syn = load_data(data_dir=outdir, datasets=[outfname], instruments=instruments, wl_model=wl,
                         offset_datasets=None, wl_unit='micron', bin_width='half', spectrum_unit='(Rp/Rs)^2',
                         skiprows=None)

    # Plot synthetic data and model (contaminated spectrum)
    ax[1].plot(data_syn['wl_data'][0], data_syn['ydata'][0] * 1e6, 'o', color='black', markersize=2, zorder=99)
    ax_1_ylim = ax[1].get_ylim()
    ax_1_xlim = ax[1].get_xlim()
    ax[1].plot(wl, spectrum * 1e6, alpha=.5)
    ax[1].set_ylim(ax_1_ylim)
    ax[1].set_xlim(ax_1_xlim)

    plt.show()

# =====================================================================================================================
# End of code
# =====================================================================================================================
