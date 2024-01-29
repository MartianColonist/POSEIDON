#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval_t1f_multivisit.py

Adapted from:
https://poseidon-retrievals.readthedocs.io/en/latest/content/notebooks/retrieval_basic.html

@author: MartianColonist, lim
"""
# Imports
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy import constants as astrocst
from POSEIDON.core import create_star, create_planet
from POSEIDON.constants import R_Sun, R_J, M_J, R_E, M_E
from POSEIDON.core import load_data, wl_grid_constant_R
from POSEIDON.visuals import plot_data
from POSEIDON.core import define_model
from POSEIDON.core import set_priors
from POSEIDON.core import read_opacities
import numpy as np
from POSEIDON.retrieval import run_retrieval
from POSEIDON.utility import read_retrieved_spectrum, plot_collection
from POSEIDON.visuals import plot_spectra_retrieved
from POSEIDON.corner import generate_cornerplot

# For uniform fonts in plots
fontsize = 12
markersize = 12
capsize = 2
plt.rcParams["font.size"] = fontsize
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
# plt.style.use('dark_background')


# =====================================================================================================================
# Define variables
# =====================================================================================================================
# ***** Define stellar properties *****#
R_s = 0.1192 * R_Sun  # Stellar radius (m) (Agol+21)
T_s = 2566  # Stellar effective temperature (K) (Agol+21)
sigma_T_s = 50  # Stellar effective temperature uncertainty (K) (Agol+21)
Met_s = 0.04  # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)] (Delrez+18)
log_g_s = 5.2396  # Stellar log surface gravity (log10(cm/s^2) by convention) (Agol+21)
sigma_log_g_s = 0.01  # Stellar log surface gravity uncertainty (log10(cm/s^2) by convention) (Agol+21)

# ***** Define planet properties *****#
planet_name = 'forward_TRAPPIST-1f'  # Planet name used for plots, output files etc.
R_p = 1.045 * R_E  # Planetary radius (m) (Agol+21)
M_p = 1.039 * M_E  # Planetary mass (kg) (Agol+21)
g_p = astrocst.G * M_p / R_p ** 2  # Gravitational field of planet (m/s^2)
T_eq = 214.5  # Equilibrium temperature (K) (Delrez+18)

# ***** Model wavelength grid *****#
wl_min = 0.58  # Minimum wavelength (um)
wl_max = 3.0  # Maximum wavelength (um)
R = 4000  # Spectral resolution of grid  # 20000 in Lim et al. (2023)

# ***** Specify data location and instruments  ***** #
# Specify the data
data_dir = ('/home/olivia/projet/jwst/retrieval/POSEIDON/tests/POSEIDON_output/forward_TRAPPIST-1f')
datasets = ['forward_TRAPPIST-1f_SYNTHETIC_JWST_NIRISS_SOSS_Ord1_bulk-N2_param-CO2-H2_fhet-0.100_Thet-2466.0_logghet-4.50000_gaussscat-10.0ppm_respow-30_pymsg1.dat',
            'forward_TRAPPIST-1f_SYNTHETIC_JWST_NIRISS_SOSS_Ord2_bulk-N2_param-CO2-H2_fhet-0.100_Thet-2466.0_logghet-4.50000_gaussscat-10.0ppm_respow-30_pymsg1.dat',
            'forward_TRAPPIST-1f_SYNTHETIC_JWST_NIRISS_SOSS_Ord1_bulk-N2_param-CO2-H2_fhet-0.050_Thet-2716.0_logghet-5.00000_gaussscat-10.0ppm_respow-30_pymsg1.dat',
            'forward_TRAPPIST-1f_SYNTHETIC_JWST_NIRISS_SOSS_Ord2_bulk-N2_param-CO2-H2_fhet-0.050_Thet-2716.0_logghet-5.00000_gaussscat-10.0ppm_respow-30_pymsg1.dat'
            ]  # Found in reference_data/observations
instruments = ['JWST_NIRISS_SOSS_Ord1',  # dataset 0
               'JWST_NIRISS_SOSS_Ord2',  # dataset 1
               'JWST_NIRISS_SOSS_Ord1',  # dataset 2
               'JWST_NIRISS_SOSS_Ord2'  # dataset 3
               ]  # Instruments corresponding to the data

# ***** Define model *****#
#model_name = 'vis-1_ord-1-2_full_r30'  # Model name used for plots, output files etc. Other strings will be added.
#model_name = 'vis-2_ord-1-2_full_r30'  # Model name used for plots, output files etc. Other strings will be added.
model_name = 'vis-1-2_ord-1-2_full_r30'  # Model name used for plots, output files etc. Other strings will be added.

bulk_species = ['N2']  # N2 comprises the bulk atmosphere
param_species = ['CO2', 'H2']

pt_profile = "isotherm"  # Isothermal temperature profile
cloud_model = "cloud-free"  # No clouds
cloud_type = "deck"  # Not used because no clouds

stellar_contam = ["two_spots_free_log_g",  # dataset 0
                  "two_spots_free_log_g",  # dataset 1
                  "two_spots_free_log_g",  # dataset 2
                  "two_spots_free_log_g"  # dataset 3
                  ]  # Stellar contamination prescription for each dataset

shared_stellar_contam = {0: 0,  # dataset 0 has its own stellar contamination parameters
                         1: 0,  # dataset 1 shares stellar contamination parameters with dataset 0
                         2: 2,  # dataset 2 has its own stellar contamination parameters
                         3: 2  # dataset 3 shares stellar contamination parameters with dataset 2
                         }

# ***** Set priors for retrieval *****#

# Initialise prior type dictionary
prior_types = {}

# Specify whether priors are linear, Gaussian, etc.
prior_types['T'] = 'uniform'
prior_types['R_p_ref'] = 'uniform'
prior_types['log_CO2'] = 'CLR'
prior_types['log_H2'] = 'CLR'
prior_types['T_phot'] = 'gaussian'
prior_types['log_g_phot'] = 'gaussian'
prior_types['f_spot_set0'] = 'uniform'
prior_types['T_spot_set0'] = 'uniform'
prior_types['log_g_spot_set0'] = 'uniform'
prior_types['f_fac_set0'] = 'uniform'
prior_types['T_fac_set0'] = 'uniform'
prior_types['log_g_fac_set0'] = 'uniform'
prior_types['f_spot_set2'] = 'uniform'
prior_types['T_spot_set2'] = 'uniform'
prior_types['log_g_spot_set2'] = 'uniform'
prior_types['f_fac_set2'] = 'uniform'
prior_types['T_fac_set2'] = 'uniform'
prior_types['log_g_fac_set2'] = 'uniform'

# Initialise prior range dictionary
prior_ranges = {}

# Specify prior ranges for each free parameter
prior_ranges['T'] = [100, 500]
prior_ranges['R_p_ref'] = [0.85 * R_p, 1.15 * R_p]
prior_ranges['log_CO2'] = [-12, 0.]
prior_ranges['log_H2'] = [-12, 0.]
prior_ranges['T_phot'] = [T_s, sigma_T_s]
prior_ranges['log_g_phot'] = [log_g_s, sigma_log_g_s]
prior_ranges['f_spot_set0'] = [0., 0.5]
prior_ranges['T_spot_set0'] = [2300., T_s + 3. * sigma_T_s]
prior_ranges['log_g_spot_set0'] = [3., 5.4]
prior_ranges['f_fac_set0'] = [0., 0.5]
prior_ranges['T_fac_set0'] = [T_s - 3 * sigma_T_s, 1.2 * T_s]
prior_ranges['log_g_fac_set0'] = [3., 5.4]
prior_ranges['f_spot_set2'] = [0., 0.5]
prior_ranges['T_spot_set2'] = [2300., T_s + 3. * sigma_T_s]
prior_ranges['log_g_spot_set2'] = [3., 5.4]
prior_ranges['f_fac_set2'] = [0., 0.5]
prior_ranges['T_fac_set2'] = [T_s - 3 * sigma_T_s, 1.2 * T_s]
prior_ranges['log_g_fac_set2'] = [3., 5.4]

# ***** Read opacity data *****#
opacity_treatment = 'opacity_sampling'

# Define fine temperature grid (K)
T_fine_min = 100  # Same as prior range for T
T_fine_max = 500  # Same as prior range for T
T_fine_step = 10  # 10 K steps are a good tradeoff between accuracy and RAM

# Define fine pressure grid (log10(P/bar))
log_P_fine_min = -6.0  # 1 ubar is the lowest pressure in the opacity database
log_P_fine_max = 2.0  # 100 bar is the highest pressure in the opacity database
log_P_fine_step = 0.2  # 0.2 dex steps are a good tradeoff between accuracy and RAM

# ***** Specify fixed atmospheric settings for retrieval *****#
# Atmospheric pressure grid
P_min = 1.0e-7  # 0.1 ubar
P_max = 100  # 100 bar
N_layers = 100  # 100 layers

# Specify the reference pressure
P_ref = 1.0  # Retrieved R_p_ref parameter will be the radius at 10 bar

# Specify retrieval setup
spectrum_type = "transmission"  # Transmission spectrum
sampling_algorithm = 'MultiNest'  # Use MultiNest for sampling
n_live_pts = 400  # Number of live points for nested sampling
verbose = True  # Verbose MultiNest output
use_pymsg = True  # Use pymsg as interpolation backend

# Plotting info
# Specify which dataset(s) to plots together on the retrieved spectrum figures
retrieved_spec_figs = {0: [0, 1],  # In figure 0, I want datasets 0 and 1
                       1: [2, 3]  # In figure 1, I want datasets 2 and 3
                       }
data_labels = ['Vis. 1, SOSS Ord. 1',
               'Vis. 1, SOSS Ord. 2',
               'Vis. 2, SOSS Ord. 1',
               'Vis. 2, SOSS Ord. 2'
               ]  # Labels for data in plots
data_colour_list = ['#1f77b4',
                    '#ff7f0e',
                    '#2ca02c',
                    '#d62728'
                    ]  # Colours for data in plots
r_to_bin = 100  # Bin the model spectrum to this resolution for plotting


# =====================================================================================================================
# Define functions
# =====================================================================================================================
def fprint(output, fname):
    """
    Print output in terminal (or IDE) and save to file

    :param output:  List of strings to print and save
    :param fname:   Name of the file where the output will be saved
    :return:
    """
    print(*output)
    if fname is None:
        return 0
    with open(fname, "a") as f:
        for i in range(len(output)):
            f.write(str(output[i]))
        f.write("\n")


# =====================================================================================================================
# Start of code
# =====================================================================================================================
if __name__ == '__main__':
    if not os.path.isdir("POSEIDON_output/{}".format(planet_name)):
        os.makedirs("POSEIDON_output/{}".format(planet_name))
    model_name = "{}_pymsg{}".format(model_name, int(use_pymsg))
    outfname = "POSEIDON_output/{}/{}.out".format(planet_name, model_name)

    # CELL 1 ----------------------------------------------------------------------------------------------------------
    # Create the stellar object
    if not use_pymsg:
        star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="phoenix")
    else:
        star = create_star(R_s, T_s, log_g_s, Met_s, stellar_grid="Goettingen-HiRes", interp_backend="pymsg")
    fprint(["Created star object."], fname=outfname)

    # Create the planet object
    planet = create_planet(planet_name, R_p, gravity=g_p, T_eq=T_eq)
    fprint(["Created planet object."], fname=outfname)

    # CELL 2 ----------------------------------------------------------------------------------------------------------
    # We need to provide a model wavelength grid to initialise instrument properties
    wl = wl_grid_constant_R(wl_min, wl_max, R)
    fprint(["Created wavelength grid."], fname=outfname)

    # Load dataset, pre-load instrument PSF and transmission function
    data = load_data(data_dir, datasets, instruments, wl)
    fprint(["Loaded data, instrument PSF and transmission function."], fname=outfname)

    # Plot our data
    fig_data = plot_data(data, planet_name)
    fprint(["Plotted data."], fname=outfname)

    # CELL 4 ----------------------------------------------------------------------------------------------------------
    # Create the model object
    model = define_model(model_name, bulk_species, param_species, PT_profile=pt_profile, cloud_model=cloud_model,
                         cloud_type=cloud_type, stellar_contam=stellar_contam,
                         shared_stellar_contam=shared_stellar_contam, radius_unit='R_E', mass_unit='M_E')
    fprint(["Created model object."], fname=outfname)

    # Check the free parameters defining this model
    fprint(["Free parameters: " + str(model['param_names'])], fname=outfname)

    # CELL 5 ----------------------------------------------------------------------------------------------------------
    # Create prior object for retrieval
    priors = set_priors(planet, star, model, data, prior_types, prior_ranges)
    fprint(["Created priors object."], fname=outfname)

    # CELL 6 ----------------------------------------------------------------------------------------------------------
    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)
    fprint(["Created fine temperature grid."], fname=outfname)

    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step),
                           log_P_fine_step)
    fprint(["Created fine pressure grid."], fname=outfname)

    # Pre-interpolate the opacities
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)
    fprint(["Pre-interpolated opacities."], fname=outfname)

    # CELL 7 ----------------------------------------------------------------------------------------------------------
    # Let's space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)
    fprint(["Created uniform-log pressure grid."], fname=outfname)

    # ***** Run atmospheric retrieval *****#
    run_retrieval(planet, star, model, opac, data, priors, wl, P, P_ref, R=R,
                  spectrum_type=spectrum_type, sampling_algorithm=sampling_algorithm,
                  N_live=n_live_pts, verbose=verbose, ev_tol=.5)
    fprint(["Ran retrieval."], fname=outfname)

    # CELL 8 ----------------------------------------------------------------------------------------------------------
    # ***** Plot retrieved transmission spectrum *****#

    # Read retrieved spectrum confidence regions
    wl, spec_low2, spec_low1, spec_median, \
        spec_high1, spec_high2 = read_retrieved_spectrum(planet_name, model_name, concatenate_wl=False)
    fprint(["Read retrieved spectrum."], fname=outfname)

    # Iterate over figures
    for i_fig in range(len(retrieved_spec_figs)):
        # Initialise collections for plotting
        spectra_median, spectra_low1, spectra_low2, spectra_high1, spectra_high2 = ([] for _ in range(5))
        for dataset_j in retrieved_spec_figs[i_fig]:
            # Append composite spectra objects to appropriate collections
            plot_collection(spec_median[dataset_j], wl[dataset_j], collection=spectra_median, concatenate_new=False)
            plot_collection(spec_low1[dataset_j], wl[dataset_j], collection=spectra_low1, concatenate_new=False)
            plot_collection(spec_low2[dataset_j], wl[dataset_j], collection=spectra_low2, concatenate_new=False)
            plot_collection(spec_high1[dataset_j], wl[dataset_j], collection=spectra_high1, concatenate_new=False)
            plot_collection(spec_high2[dataset_j], wl[dataset_j], collection=spectra_high2, concatenate_new=False)
        fprint(["Created composite spectra objects."], fname=outfname)

        # Produce figures
        # Make temporary 'data', 'data_labels' and 'data_colour' objects with only the datasets that we want to plot
        data_tmp = {}
        data_labels_tmp = []
        data_colour_list_tmp = []
        # Iterate over datasets we want to plot in current figure
        for j, dataset_j in enumerate(retrieved_spec_figs[i_fig]):
            if j == 0:
                # Initialise key-value pairs in dictionary
                for key in data.keys():
                    if data[key] is not None:  # Check if this key is empty
                        data_tmp[key] = []
            # Add info from this dataset to the temporary data, data_labels, and data_colour objects
            for key in data_tmp.keys():
                data_tmp[key].append(data[key][dataset_j])
            data_labels_tmp.append(data_labels[dataset_j])
            data_colour_list_tmp.append(data_colour_list[dataset_j])

        fig_spec = plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1,
                                          spectra_high1, spectra_high2, planet_name,
                                          data_tmp, R_to_bin=r_to_bin,
                                          data_labels=data_labels_tmp,
                                          data_colour_list=data_colour_list_tmp,
                                          plt_label=model_name + '_fig{}'.format(i_fig))
        plt.close(fig_spec)
    fprint(["Plotted retrieved spectrum/spectra."], fname=outfname)

    # ***** Make corner plot *****#

    fig_corner = generate_cornerplot(planet, model)
    fprint(["Plotted corner plot."], fname=outfname)

# =====================================================================================================================
# End of code
# =====================================================================================================================
