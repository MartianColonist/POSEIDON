''' 
POSEIDON CORE ROUTINE

V 2.0 (3D Retrieval Fully Operational)

Copyright 2021, Ryan MacDonald, All rights reserved.

'''

import numpy as np
import pymultinest
from numba.core.decorators import jit
import scipy.constants as sc
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
#size = comm.Get_size()
rank = comm.Get_rank()

# Load settings and external functions called by this program
# (visit config.py for common settings)
from config import planet_name, N_D, N_wl, R, wl_min, wl_max, wl_grid, PT_dim, \
                   X_dim, cloud_dim, single_model, do_retrieval, sim_retrieval, \
                   load_observations, TwoD_type, PT_profile, produce_sim_data, \
                   check_best_fit, plot_retrieved_spectrum, analyse_Bayes, \
                   spectrum_type, skip_preload, stellar_contam, bulk_species, \
                   param_species, species_EM_gradient, species_DN_gradient, \
                   species_vert_gradient, offsets_applied, opacity_treatment, \
                   offset_datasets, line_by_line_resolution, std_data, R_data, \
                   wl_data_min, wl_data_max, syn_instrument, syn_data_tag, \
                   instruments, datasets, N_live, sampling_target, base_name, \
                   N_slice_DN, N_slice_EM, term_transition, sampling_algorithm, \
                   ev_tol, make_corner, model_tag, sim_data_from_file

from atmosphere import profiles, compute_metallicity, compute_C_to_O
from absorption import store_Rayleigh_eta_LBL, opacity_tables
from instrument import init_instrument, generate_syn_data, \
                       generate_syn_data_from_file
from transmission import TRIDENT
from geometry import atmosphere_regions, angular_grids
from parameters import assign_free_params, load_state, unpack_geometry_params, \
                       load_parameters_from_config
                       
from stellar import precompute_stellar_spectra
from retrieval import PyMultiNest_retrieval, Bayesian_model_comparison
from corner import generate_cornerplot
from supported_features import supported_species, supported_cia, \
                               inactive_species
from visuals import plot_PT_profiles, plot_X_profiles, \
                    plot_spectra_retrieved, plot_PT_retrieved, plot_geometry, \
                    plot_spectrum_chem_removed, plot_spectrum_time_resolved, \
                    plot_opacity, plot_evening_morning_opening_angle, \
                    plot_day_night_opening_angle, plot_spectra

from utility import create_directories, read_data, read_spectrum, \
                    write_output, write_geometry, confidence_intervals, \
                    write_retrieval_results, spectrum_plot_collection


#******************************#
#***** INITIALISE PROGRAM *****#
#******************************#       

# Start clock for timing program
t0 = time.perf_counter() 

#***** Step (0): Create any mising directories (e.g. for output files) *****#

if (rank == 0):
    create_directories(planet_name, datasets, load_observations, 
                       do_retrieval, sampling_algorithm)   
    
comm.Barrier()

#***** Step (1): Create model wavelength grid *****#

# If the user has one setting indicating line-by-line mode, check both keys are turned.
if ((wl_grid == 'line-by-line') or (opacity_treatment == 'line-by-line')):
    
    if (wl_grid != 'line-by-line'):
        raise Exception("A line-by-line model also requires wavelength grid to be handled in line-by-line mode.\n"
                        "Please modify config.py to set wl_grid = 'line-by-line'.") 
        
    if (opacity_treatment != 'line-by-line'):
        raise Exception("A line-by-line model also requires opacities to be handled in line-by-line mode.\n"
                        "Please modify config.py to set opacity_treatment = 'line-by-line'.") 
        
if ((spectrum_type == 'emission') and (int(PT_dim) + int(X_dim) + int(cloud_dim) != 3)):
    
    raise Exception("Only 1D emission spectra currently supported.\n" 
                    "Please set X_dim, PT_dim, and cloud_dim to 1 in config.py.")
    
# For runs with multiple cores, only print to terminal once
if (rank == 0):    
    print("******")
    print("POSEIDON stirs in the deep")
    print("******")

# Create model grid, according to user choice in config.py
if (wl_grid == 'uniform'):  
    
    wl = np.linspace(wl_min, wl_max, N_wl)   # Units: micron (um)
    
    if (rank == 0):
        print("Wavelength grid is: uniform")  
    
elif (wl_grid == 'constant R'):  
     
    # Constant R -> uniform in log(wl)
    delta_log_wl = 1.0/R
    N_wl = (np.log(wl_max) - np.log(wl_min)) / delta_log_wl
    N_wl = np.around(N_wl).astype(np.int64)
    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), N_wl)    
    
    wl = np.exp(log_wl)    # Units: micron (um)
    
    if (rank == 0):
        print("Wavelength grid is: R = " + str(R))

# For line-by-line case, wavelengths are linearly spaced in wavenumber to match native cross section resolution
elif (wl_grid == 'line-by-line'):
            
    # Quick check to make sure retrieval mode is not on (unwise for line-by-line mode!)
    if (do_retrieval == True):
        raise Exception("Retrieval functionality is not available for line-by-line models!\n"
                        "Please modify config.py to set do_retrieval = False.\n"
                        "Perhaps you meant to run a single model via single_model = True?")
    else:
        if (rank != 0):
            raise Exception("Line-by-line mode not supported on multiple cores!")
        else:
            print("******")
            print("Running POSEIDON forward model in line-by-line mode")
            print("******")
            
            nu_min = 1.0e4/wl_max   # Minimum wavenumber on output grid
            nu_max = 1.0e4/wl_min   # Maximum wavenumber on output grid
            
            # Need wavenumber grid bounds to match cross section grid (0.01 cm^-1 spacing, so round to 2 decimal places)
            nu_min = np.around(nu_min, np.abs(np.int(np.log10(line_by_line_resolution))))
            nu_max = np.around(nu_max, np.abs(np.int(np.log10(line_by_line_resolution))))
            
            # Find number of wavenumber points on grid
            N_nu = np.int((nu_max - nu_min)/line_by_line_resolution)
            N_wl = N_nu
            
            # Initialise line-by-line model wavenumber grid
            nu = np.linspace(nu_max, nu_min, N_nu)   # Decreasing wavenumber order
            nu = np.around(nu, np.abs(np.int(np.log10(line_by_line_resolution))))    # Removed floating point errors
            
            # Initialise corresponding wavelength grid 
            wl = 1.0e4/nu   # Convert cm^-1 to um
        
#***** Step (2): Load observations *****#

# Observations can be real or synthetic, specified by user in config.py
if (load_observations == True):
    
    # Convert lists in config file to numpy arrays
    instruments = np.array(instruments)
    datasets = np.array(datasets)
    offset_datasets = np.array(offset_datasets)
    
    # Initialise arrays containing input properties of the data
    wl_data, half_bin, ydata, err_data, len_data = (np.array([]) for _ in range(5))
    
    # Initialise arrays containing instrument function properties
    psf_sigma, sens, norm = (np.array([]) for _ in range(3))
    bin_left, bin_cent, bin_right, norm = (np.array([]).astype(np.int64) for _ in range(4))
    
    # For each dataset (defined in config.py)
    for i in range(len(datasets)):
        
        # Read data files (defined in config.py)
        wl_data_i, half_bin_i, ydata_i, err_data_i = read_data(planet_name, datasets[i])
        
        # Combine datasets
        wl_data = np.concatenate([wl_data, wl_data_i])
        half_bin = np.concatenate([half_bin, half_bin_i])  
        ydata = np.concatenate([ydata, ydata_i])
        err_data = np.concatenate([err_data, err_data_i])
        
        # Length of each dataset (used for indexing the combined dataset, if necessary to extract one specific dataset later)
        len_data = np.concatenate([len_data, np.array([len(ydata_i)])])
        
        # Read instrument transmission functions, compute PSF std dev, and identify locations of each data bin on model grid
        psf_sigma_i, sens_i, bin_left_i, \
        bin_cent_i, bin_right_i, norm_i = init_instrument(wl, wl_data_i, half_bin_i, instruments[i])
        
        # Combine instrument properties into single arrays for convience (can index by len_data[i] to extract each later)
        psf_sigma = np.concatenate([psf_sigma, psf_sigma_i])              # Length for each dataset: len_data[i]
        sens = np.concatenate([sens, sens_i])                 # Length for each dataset: N_wl
        bin_left = np.concatenate([bin_left, bin_left_i])     # Length for each dataset: len_data[i]
        bin_cent = np.concatenate([bin_cent, bin_cent_i])     # Length for each dataset: len_data[i]
        bin_right = np.concatenate([bin_right, bin_right_i])  # Length for each dataset: len_data[i]
        norm = np.concatenate([norm, norm_i])                 # Length for each dataset: len_data[i]
        
    N_data = len(ydata)  # Total number of data points
    
    # Commulative sum of data lengths for indexing later
    len_data_idx = np.append(np.array([0]), np.cumsum(len_data)).astype(np.int64)       

    # For relative offsets, find which data indices the offset applies to
    if (offsets_applied == 'relative'):
        if (offset_datasets[0] in datasets):
            offset_dataset_idx = np.where(datasets == offset_datasets[0])[0][0]
            offset_data_start = len_data_idx[offset_dataset_idx]  # Data index of first point with offset
            offset_data_end = len_data_idx[offset_dataset_idx+1]  # Data index of last point with offset + 1
        else: 
            raise Exception("Dataset chosen for relative offset is not included.")
    else:
        offset_data_start = 0    # Dummy values when no offsets included
        offset_data_end = 0
        
    # Check that the model wavelength grid covers all the data bins
    if (np.any((wl_data - half_bin) < wl[0])):
        raise Exception("Some data lies below the lowest model wavelength, reduce wl_min in config.py.")
    elif (np.any((wl_data + half_bin) > wl[-1])):
        raise Exception("Some data lies above the highest model wavelength, increase wl_max in config.py.")
    
    # Set prior limits for error inflation parameter
    prior_lower_err_inflation = np.log10(0.001*np.min(err_data*err_data))  
    prior_upper_err_inflation = np.log10(100.0*np.max(err_data*err_data))  
 
# When no observations are included, initialise dummy arrays 
else:
    wl_data, half_bin, ydata, err_data, len_data = (np.array([]) for _ in range(5))
    psf_sigma, sens, norm = (np.array([]) for _ in range(3))
    bin_left, bin_cent, bin_right, norm = (np.array([]).astype(np.int64) for _ in range(4))
    N_data = 0        # Total number of data points
    len_data_idx = 0
    offset_data_start = 0    # Dummy values when no offsets included
    offset_data_end = 0
   
# Store the various data properties in a dictionary to avoid passing them all into functions separately
data_properties = dict(datasets = datasets, instruments = instruments, wl_data = wl_data,
                       half_bin = half_bin, ydata = ydata, err_data = err_data, sens = sens,
                       len_data_idx = len_data_idx, psf_sigma = psf_sigma, norm = norm,
                       bin_left = bin_left, bin_cent = bin_cent, bin_right = bin_right,
                       offset_start = offset_data_start, offset_end = offset_data_end)
    

#***** Step (3): Create chemical species arrays *****#

# Create array containing all chemical species in model
bulk_species = np.array(bulk_species)
param_species = np.array(param_species)
chemical_species = np.append(bulk_species, param_species)

# Identify chemical species with active spectral features
active_species = chemical_species[~np.isin(chemical_species, inactive_species)]

# Convert arrays specifying which species have gradients into numpy arrays
species_vert_gradient = np.array(species_vert_gradient)

# Check if cross sections are available for all the selected chemical species
if (np.any(~np.isin(active_species, supported_species)) == True):
    raise Exception("One or more chemical species you selected is not supported.\n"
                    "Please check supported_features.py.")

# Create list of collisionally-induced absorption (CIA) pairs
cia_pairs = []
for pair in supported_cia:
    pair_1, pair_2 = pair.split('-')   
    if (pair_1 in chemical_species) and (pair_2 in chemical_species):
        cia_pairs.append(pair)     
cia_pairs = np.array(cia_pairs)

# Create list of free-free absorption pairs
ff_pairs = []
if ('H' in chemical_species) and ('e-' in chemical_species):  
    ff_pairs.append('H-ff')       # H- free-free    
ff_pairs = np.array(ff_pairs)

# Create list of free-free absorption pairs
bf_species = []
if ('H-' in chemical_species):  
    bf_species.append('H-bf')      # H- bound-free    
bf_species = np.array(bf_species)

# Store numbers of chemical species, active species, and pair processes
N_species = len(chemical_species)        # Total number of chemical species
N_bulk_species = len(bulk_species)       # Total number of bulk species
N_param_species = len(param_species)     # Total number of parametrised species
N_species_active = len(active_species)   # Total number of spectrally active species
N_cia_pairs = len(cia_pairs)             # Total number of CIA pairs
N_ff_pairs = len(ff_pairs)               # Total number of free-free pairs
N_bf_species = len(bf_species)           # Total number of bound-free species

#***** Step (4): Geometrical properties of background atmosphere *****#

Atmosphere_dimension, \
N_sectors, N_zones = atmosphere_regions(PT_dim, X_dim, TwoD_type, 
                                        term_transition, N_slice_EM, N_slice_DN)

#***** Step (5): Identify free parameters *****#

param_names, N_params, \
N_PT_params, N_species_params, \
N_cloud_params, N_geometry_params, \
N_stellar_params, N_offset_params, \
N_error_params, enable_deck, \
enable_haze = assign_free_params(param_species, Atmosphere_dimension)

# The cumulative sum of the number of each type of parameter saves time indexing later 
N_params_cumulative = np.cumsum([N_PT_params, 1, N_species_params, N_cloud_params, 
                                 N_geometry_params, N_stellar_params, N_offset_params, 
                                 N_error_params])
    
#***** Step (5): Pre-compute opacity tables *****#

# Only need to pre-compute opacities when not in line-by-line mode
if (opacity_treatment != 'line-by-line'):   
    if (skip_preload == False):     # This option is only true for debugging

        # Read and interpolate cross sections to model pressure, wavelength, and fine temperature grids
        sigma_stored, cia_stored, \
        Rayleigh_stored, eta_stored, \
        ff_stored, bf_stored = opacity_tables(rank, comm, wl, chemical_species, 
                                              active_species, cia_pairs, 
                                              ff_pairs, bf_species)
                
else:   
    
    # For line-by-line case, still need to compute refractive indices
    Rayleigh_stored, eta_stored = store_Rayleigh_eta_LBL(wl, chemical_species)   
    
    # No need for pre-computed arrays in this case, but keep as empty dummies for function inputs
    sigma_stored, cia_stored,  \
    ff_stored, bf_stored = (np.array([]) for _ in range(4))  
    
#***** Step (6): Pre-compute stellar spectra *****#

# If running a model including unocculted stellar spots / faculae
if (stellar_contam != 'No'):
    if (skip_preload == False):    # This option is only true for debugging purposes
    
        if (rank == 0):
            print("Pre-computing stellar spectra...")
        
        # Compute photosphere and heterogeneity spectra across a grid of T_eff (fixed Met_s, log_g_s)
        T_phot_grid, I_phot_grid = precompute_stellar_spectra(wl, 'photosphere')
        T_het_grid, I_het_grid = precompute_stellar_spectra(wl, 'unocculted')
    
    enable_stellar_contam = 1   # Flag for radiative transfer functions
    
else:
    
    # No need for pre-initialised arrays in this case, but keep as empty dummy for function inputs
    T_phot_grid, T_het_grid, I_phot_grid, I_het_grid = (np.array([]) for _ in range(4)) 
    
    enable_stellar_contam = 0   # Flag for radiative transfer functions

#***** Step (7): Miscellaneous *****#

# Print model wavelength grid characteristics to terminal
if (rank == 0):
    print("Number of model wavelengths: " + str(N_wl))
    print("Datasets: " + ', '.join(instruments))
    
    if (single_model == True): 
        print("Generating spectrum of planet " + planet_name)
    elif (do_retrieval == True): 
        print("Retrieving atmosphere of planet " + planet_name)
    
    print("******")


#******************************#
#******** MAIN PROGRAM ********#
#******************************#   

#***** Single forward model spectrum / synthetic data generation *****#

if (single_model == True):
        
    # Load free parameters from user choices in config.py
    params_set = load_parameters_from_config(param_species, Atmosphere_dimension)
    
    # Organise parameter state vector into inputs required by forward model
    PT_set, R_p_ref_set, log_X_set, \
    clouds_set, geometry_set, \
    stellar_set, offsets_set, \
    err_inflation_set = load_state(params_set, param_names, N_params_cumulative, 
                                   param_species)
        
    # To check for relative systematics between data sets, add relative offset 
    # to one dataset's transit depth values
    if (offsets_applied == 'relative'): 
        ydata_adjusted = ydata.copy()
        ydata_adjusted[offset_data_start:offset_data_end] += offsets_set[0]
    else: 
        ydata_adjusted = ydata   

    # Generate transmission spectrum        
    if (spectrum_type == 'transmission'):
                    
        spectrum_set, ymodel_set, \
        P_set, T_set, r_set, X_set, \
        mu_set, is_physical_set = TRIDENT(wl, PT_set, R_p_ref_set, log_X_set, 
                                          clouds_set, geometry_set, stellar_set, 
                                          offsets_set, sigma_stored, cia_stored, 
                                          Rayleigh_stored, eta_stored, ff_stored, 
                                          bf_stored, T_phot_grid, I_phot_grid, 
                                          T_het_grid, I_het_grid, 0.0, 
                                          chemical_species, bulk_species, 
                                          param_species, active_species, 
                                          cia_pairs, ff_pairs, bf_species,
                                          species_vert_gradient, enable_haze, 
                                          enable_deck, N_sectors, N_zones, 
                                          Atmosphere_dimension, param_names, 
                                          N_params_cumulative, data_properties,
                                          ignore_species = [])

                                                                                                                                                                                                              
    # Check fiducial model is physical
    print("Model physical? " + str(is_physical_set))  
    
    if (is_physical_set == False):
        raise Exception("Chosen model parameters are not physical, please check config.py")
        
    print("Generated spectrum")
    
    # Create composite spectra object for plotting
    spectra = spectrum_plot_collection(spectrum_set, wl) 
    
    # Add a second spectrum to plotting collection
    # spectra = spectrum_plot_collection(spectrum_2, wl_2, spectra)
        
    M = compute_metallicity(X_set, chemical_species)
    C_to_O = compute_C_to_O(X_set, chemical_species)
    
    print("mu = " + str(mu_set[0,:,0]/sc.u))
    print("M = " + str(M[0,0,:,0]))
    print("C/O = " + str(C_to_O[0,0,:,0]))
    
    #***** Write / read spectra to file *****# 

    write_output(planet_name, spectrum_set, wl, P_set, T_set, model_tag)

 #   wl_2, spectrum_2 = read_spectrum('../../output/spectra/' + planet_name + '_2D_EM_test_2.dat')
    
#    wl_2, spectrum_1 = read_spectrum('../../output/spectra/' + planet_name + '_ME_0_X_gradient.dat')
#    wl_2, spectrum_2 = read_spectrum('../../output/spectra/' + planet_name + '_ME_5_X_gradient.dat')
#    wl_2, spectrum_3 = read_spectrum('../../output/spectra/' + planet_name + '_ME_10_X_gradient.dat')
#    wl_2, spectrum_4 = read_spectrum('../../output/spectra/' + planet_name + '_ME_20_X_gradient.dat')
#    wl_2, spectrum_5 = read_spectrum('../../output/spectra/' + planet_name + '_ME_40_X_gradient.dat')
#    wl_2, spectrum_6 = read_spectrum('../../output/spectra/' + planet_name + '_ME_80_X_gradient.dat')
#    wl_2, spectrum_7 = read_spectrum('../../output/spectra/' + planet_name + '_ME_120_X_gradient.dat')
#    wl_2, spectrum_8 = read_spectrum('../../output/spectra/' + planet_name + '_ME_180_X_gradient.dat')
#    wl_2, spectrum_9 = read_spectrum('../../output/spectra/' + planet_name + '_1D_X_gradient_log_average.dat')

#    wl_2, spectrum_1 = read_spectrum('../../output/spectra/' + planet_name + '_DN_0_X_gradient.dat')
#    wl_2, spectrum_2 = read_spectrum('../../output/spectra/' + planet_name + '_DN_5_X_gradient.dat')
#    wl_2, spectrum_3 = read_spectrum('../../output/spectra/' + planet_name + '_DN_10_X_gradient.dat')
#    wl_2, spectrum_4 = read_spectrum('../../output/spectra/' + planet_name + '_DN_20_X_gradient.dat')
#    wl_2, spectrum_5 = read_spectrum('../../output/spectra/' + planet_name + '_DN_40_X_gradient.dat')
#    wl_2, spectrum_6 = read_spectrum('../../output/spectra/' + planet_name + '_DN_80_X_gradient.dat')
#    wl_2, spectrum_7 = read_spectrum('../../output/spectra/' + planet_name + '_DN_120_X_gradient.dat')

    #***** Synthetic data generation *****#
    
    if (produce_sim_data == True):
        
        if (sim_data_from_file == True):
            syn_data, syn_err, \
            syn_wl_data, syn_half_bin, \
            syn_ymodel = generate_syn_data_from_file(wl, spectrum_set, instruments, 
                                                     datasets, syn_data_tag, 
                                                     Gauss_scatter = False) 
        else:
            syn_data, syn_err, \
            syn_wl_data, syn_half_bin, \
            syn_ymodel = generate_syn_data(wl, spectrum_set, [wl_data_min], 
                                           [wl_data_max], [R_data], [std_data], 
                                           [syn_instrument], syn_data_tag, 
                                           Gauss_scatter = True)
    
    #***** Plot output from model *****#
    
    # (1) 2D geometry slice plots
    
    # Unpack alpha and beta from geometry state vector
    alpha, beta = unpack_geometry_params(param_names, N_params_cumulative,
                                         geometry_set)
    
    # Compute discretised angular grids for multidimensional atmospheres
    phi, theta, phi_edge, \
    theta_edge, dphi, dtheta = angular_grids(Atmosphere_dimension, TwoD_type, 
                                             term_transition, N_slice_EM, 
                                             N_slice_DN, alpha, beta)
        
    # Generate 2D geometry slice plots
    
    plot_geometry(r_set, T_set, phi, phi_edge, dphi, theta, theta_edge, dtheta,
                  plt_tag = model_tag)
    
    
    # Write geometry to file for 3D plotting and animation (external application)
    write_geometry(planet_name, r_set, T_set, theta_edge, phi_edge, model_tag)
    
    # (2) Plot spectrum of user specified model atmosphere
    plot_spectra(spectra, ymodel_set, wl_data, ydata, err_data, half_bin,
                 plot_full_res = True, bin_spectra = True, R_bin = 100, show_ymodel=True,
                 colour_list = [], model_labels = ['2D true model'], plt_tag = model_tag,
                 transit_depth_min = 4.4e-3, transit_depth_max=7.6e-3)
    
    # (3) Plot P-T profiles    
    plot_PT_profiles(P_set, T_set, Atmosphere_dimension, TwoD_type, 
                     show_profiles=['morning', 'evening', 'average'],
                     plt_tag = model_tag)
    
    # (4) Plot mixing ratio profiles
    plot_X_profiles(P_set, np.log10(X_set), Atmosphere_dimension, TwoD_type, 
                    chemical_species, plot_species=['CH4', 'H2O', 'CO', 'CO2'],
                    show_profiles=['morning', 'evening', 'average'], 
                    plt_tag = model_tag)
    
  
# If retrieving atmosphere
if (do_retrieval == True):
    
    # Run POSEIDON retrieval using PyMultiNest
    if (sampling_algorithm == 'MultiNest'):

        PyMultiNest_retrieval(param_names, N_params_cumulative, chemical_species,
                              bulk_species, param_species, active_species, 
                              cia_pairs, ff_pairs, bf_species, wl, sigma_stored, 
                              cia_stored, Rayleigh_stored, eta_stored, ff_stored, 
                              bf_stored, species_EM_gradient, species_DN_gradient, 
                              species_vert_gradient, T_phot_grid, I_phot_grid, 
                              T_het_grid, I_het_grid, enable_haze, enable_deck, 
                              Atmosphere_dimension, N_sectors, N_zones, 
                              prior_lower_err_inflation, prior_upper_err_inflation, 
                              data_properties, base_name, planet_name, 
                              n_live_points = N_live, evidence_tolerance = ev_tol, 
                              log_zero = -1e90, importance_nested_sampling = False,
                              resume = False,  sampling_efficiency = sampling_target, 
                              verbose = True, multimodal = False, n_clustering_params = N_params)
    
    # Cast retrieval results into human readable format
    if (rank == 0):   # Only need to do this once, so use first core
    
        # Write POSEIDON retrieval output files 
        write_retrieval_results(planet_name, param_names, base_name, err_data, 
                                ydata, sampling_algorithm, N_live, ev_tol,
                                wl, R, instruments, datasets)   
        
    comm.Barrier()
        
# Generate corner plot (if option selected in config.py)
if (make_corner == True):
    
    if (rank == 0):
            
        # If running a simulated retrieval with known parameters
        if (sim_retrieval == True):
        
            # Load free parameters from user choices in config.py
            params_set = load_parameters_from_config(param_species, 
                                                     Atmosphere_dimension)
        
        # For retrievals with unknown parameters (i.e. real data)
        else:
        
            # The true parameter values are unknown, so we don't plot any
            params_set = None
        
        # Make corner plot and save in results directory
        generate_cornerplot(planet_name, param_names, base_name, 
                            sampling_algorithm, true_vals = params_set)

         
# Extract the minimum chi-square spectrum from a previous retrieval 
if (check_best_fit == True):
     
    # Load output from a previous MultiNest run
    full_base_name = ('../../output/retrievals/' + planet_name + 
                      '/MultiNest_raw/' + base_name)
    a = pymultinest.Analyzer(n_params = N_params, 
                             outputfiles_basename = full_base_name)
    s = a.get_stats()  
    ln_Z = s['global evidence']
    ln_Z_err = s['global evidence error']         
    best_fit = a.get_best_fit()
    max_likelihood = best_fit['log_likelihood']
    best_fit_params = best_fit['parameters']
    norm_log = (-0.5*np.log(2.0*np.pi*err_data*err_data)).sum()
    best_chi_square = -2.0 * (max_likelihood - norm_log)
    reduced_chi_square = best_chi_square/(len(ydata) - N_params)
        
    print('max log likelihood = ' + str(max_likelihood))
    print('reduced chi-square = ' + str(reduced_chi_square))
    print('ln Z = %.3f +- %.3f' % (ln_Z, ln_Z_err))
        
    # Generate state vector containing best fit parameters
    PT_ret, R_p_ref_ret, log_X_ret, \
    clouds_ret, geometry_ret, \
    stellar_ret, offsets_ret, \
    err_inflation_ret = load_state(best_fit_params, param_names, 
                                   N_params_cumulative, param_species)
    
    # Generate spectrum and PT profile for best fit parameters
    spectrum_ret, ymodel_ret, \
    P_ret, T_ret, r_ret, X_ret, \
    mu_ret, is_physical_ret = TRIDENT(wl, PT_ret, R_p_ref_ret, log_X_ret, 
                                      clouds_ret, geometry_ret, stellar_ret, 
                                      offsets_ret, sigma_stored, cia_stored, 
                                      Rayleigh_stored, eta_stored, ff_stored, 
                                      bf_stored, T_phot_grid, I_phot_grid, 
                                      T_het_grid, I_het_grid, 0.0, 
                                      chemical_species, bulk_species, 
                                      param_species, active_species, 
                                      cia_pairs, ff_pairs, bf_species,
                                      species_vert_gradient, enable_haze, 
                                      enable_deck, N_sectors, N_zones, 
                                      Atmosphere_dimension, param_names, 
                                      N_params_cumulative, data_properties)    

    if (offsets_applied == 'relative'): 
        ydata_adjusted = ydata.copy()
        ydata_adjusted[offset_data_start:offset_data_end] += offsets_ret[0]
    else: ydata_adjusted = ydata
        
    # Sanity check
    print('Best-fitting model physical? ' + str(is_physical_ret))
    
    # Create composite spectra object for plotting
    spectra = spectrum_plot_collection(spectrum_ret, wl) 
    
    # Plot best-fitting retrieved spectrum
    plot_spectra(spectra, ymodel_ret, wl_data, ydata, err_data, half_bin,
                 plot_unbinned = True, bin_spectra = True, R_bin = 100, 
                 colour_list = [], model_labels = [], plt_tag = model_tag)
    
    # Plot best-fitting retrieved PT profile
    plot_PT_profiles(P_ret, T_ret, Atmosphere_dimension, TwoD_type, 
                     plt_tag = model_tag)

#***** Plot results of a retrieval run visually with median profile + sigma contours *****#
    
if (plot_retrieved_spectrum == True):
    
    if (rank == 0):
        
        # Activate PyMultiNest analyser
        full_base_name = ('../../output/retrievals/' + planet_name + 
                          '/MultiNest_raw/' + base_name)
        a = pymultinest.Analyzer(n_params = N_params, 
                                 outputfiles_basename = full_base_name)
        s = a.get_stats()
            
        print('ln Z = %.1f +- %.1f' % (s['global evidence'], 
                                       s['global evidence error']))
                
        # Load equally weighted posterior parameter samples
        values = a.get_equal_weighted_posterior()
            
        N_values = len(values[:,0])
        print('Number of posterior samples: ' + str(N_values))
        
        # Randomly draw up to 5000 samples from posterior
        sample_draws = min(N_values, 5000)  # Number of random samples from posterior
        sample = np.random.choice(len(values), sample_draws, replace=False)
               
        # Define arrays storing sampled atmospheric properties in each region
        T_stored = np.zeros(shape=(sample_draws, N_D, N_sectors, N_zones))
        r_stored = np.zeros(shape=(sample_draws, N_D, N_sectors, N_zones))
        mu_stored = np.zeros(shape=(sample_draws, N_D, N_sectors, N_zones))
        X_stored = np.zeros(shape=(sample_draws, N_species, N_D, N_sectors, N_zones))
    
        # Define arrays storing sample transit depth and binned model at each wavelength
        spectrum_stored = np.zeros(shape=(sample_draws, N_wl))
        ymodel_stored = np.zeros(shape=(sample_draws, N_data))
            
        # Most P-T state vectors have 4 parameters
        if ((PT_dim == '1') and (PT_profile == 'Madhu')):  
            len_PT = 6    # Except for the Madhusudhan & Seager (2009) profile
        else:
            len_PT = 4
        
        # All abundance state arrays are (N_species_params x 4)
        len_X = 4    # log_X_bar_term, Delta_log_X_term, Delta_log_X_DN, log_X_deep)
        
        # For making histograms of model parameters
        PT_state_stored = np.zeros(shape=(N_values, len_PT))  
        R_p_ref_stored = np.zeros(shape=(N_values))
        log_X_state_stored = np.zeros(shape=(N_values, N_param_species, len_X))
        clouds_stored = np.zeros(shape=(N_values, N_cloud_params))
        geometry_stored = np.zeros(shape=(N_values, N_geometry_params))
        stellar_stored = np.zeros(shape=(N_values, N_stellar_params))
        offsets_stored = np.zeros(shape=(N_values, N_offset_params))
        err_inflation_stored = np.zeros(shape=(N_values, N_error_params))
        
        # Load full set of samples for posterior histograms
        for i in range(N_values):
                
            PT_state_stored[i,:], R_p_ref_stored[i], \
            log_X_state_stored[i,:,:], clouds_stored[i,:], \
            geometry_stored[i,:], stellar_stored[i,:], \
            offsets_stored[i,:], \
            err_inflation_stored[i,:] = load_state(values[i,:], param_names, 
                                                   N_params_cumulative, 
                                                   param_species)
                        
        # Generate spectrum and PT profiles from selected samples
        for i in range(sample_draws):
            
            if (i%100 == 0):
                print("Generating model " + str(i) + " sample = " + str(sample[i]))
                    
            # Load state vectors for sample i      
            PT_state_i, R_p_ref_i, \
            log_X_i, clouds_i, stellar_i, \
            geometry_i, offsets_i, \
            err_inflation_i = PT_state_stored[sample[i],:], \
                              R_p_ref_stored[sample[i]], \
                              log_X_state_stored[sample[i],:,:], \
                              clouds_stored[sample[i],:], \
                              stellar_stored[sample[i],:], \
                              geometry_stored[sample[i],:], \
                              offsets_stored[sample[i],:], \
                              err_inflation_stored[i,:]
            
            # Generate spectrum, P-T, and abundance profiles for parameter set 'i'
            spectrum_stored[i,:], ymodel_stored[i,:], \
            P_i, T_stored[i,:,:,:], r_stored[i,:,:,:], \
            X_stored[i,:,:,:,:], mu_stored[i,:,:,:], \
            is_physical_i = TRIDENT(wl, PT_state_i, R_p_ref_i, log_X_i, 
                                    clouds_i, geometry_i, stellar_i, 
                                    offsets_i, sigma_stored, cia_stored, 
                                    Rayleigh_stored, eta_stored, ff_stored, 
                                    bf_stored, T_phot_grid, I_phot_grid, 
                                    T_het_grid, I_het_grid, 0.0, 
                                    chemical_species, bulk_species, 
                                    param_species, active_species, 
                                    cia_pairs, ff_pairs, bf_species,
                                    species_vert_gradient, enable_haze, 
                                    enable_deck, N_sectors, N_zones, 
                                    Atmosphere_dimension, param_names, 
                                    N_params_cumulative, data_properties)  
            

            
        # If this retrieval used a relative offset parameter, shift relevant dataset
        if (offsets_applied == 'relative'): 
            
            # Compute median and +/- 1 sigma for relative offset parameter
            _, _, delta_rel_low1, \
            delta_rel_median, \
            delta_rel_high1, _, _ = confidence_intervals(sample_draws, 
                                                         offsets_stored[:,0], 0)
            
            # Apply retrieved relative osset to data
            ydata_adjusted = ydata.copy()
            ydata_adjusted[offset_data_start:offset_data_end] += delta_rel_median[0]
            
        # If no offset used, data remains unaltered
        else: 
            ydata_adjusted = ydata
                
        #***** Compute +/- 1 sigma and 2 sigma confidence intervals for: *****#
            
        # P-T profile
        _, T_low2, T_low1, T_median, \
        T_high1, T_high2, _ = confidence_intervals(sample_draws, 
                                                   T_stored[:,:,0,0], N_D)
        
        # Mean molecular mass
        _, _, mu_low1, mu_median, \
        mu_high1, _, _ = confidence_intervals(sample_draws, mu_stored[:,:,0,0], N_D)
        
        # Transmission sectrum
        _, spectrum_low2, spectrum_low1, spectrum_median, \
        spectrum_high1, spectrum_high2, _ = confidence_intervals(sample_draws, 
                                                                 spectrum_stored, N_wl)
        
        # Also extract the median retrieved model, binned to the data resolution
        _, _, _, ymodel_median, \
        _, _, _ = confidence_intervals(sample_draws, ymodel_stored, N_data)
                        
        # Create composite spectra objects for plotting
        spectra_median = spectrum_plot_collection(spectrum_median, wl, spectra = []) 
        spectra_low1 = spectrum_plot_collection(spectrum_low1, wl, spectra = []) 
        spectra_low2 = spectrum_plot_collection(spectrum_low2, wl, spectra = []) 
        spectra_high1 = spectrum_plot_collection(spectrum_high1, wl, spectra = []) 
        spectra_high2 = spectrum_plot_collection(spectrum_high2, wl, spectra = []) 
    
        # Finally, plot the retrieved spectrum    
        plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1, 
                               spectra_high1, spectra_high2, ymodel_median, 
                               wl_data, ydata, err_data, half_bin, R_bin = 100, 
                               colour_list = [], transit_depth_min = 1.40e-2, 
                               transit_depth_max=1.80e-2, plt_tag = model_tag)
        
        # Also plot the retrieved P-T profile
 #       plot_PT_retrieved(P_i, T_median, T_low2, T_low1, T_high1, T_high2, 
 #                         Atmosphere_dimension, plt_tag = model_tag)
    
    
# Conduct a Bayesian model comparison
if (analyse_Bayes == True):
    
    Bayesian_model_comparison('2D_atm', (planet_name + '_2000_3D_retrieval_results_' + model_tag + '_full-'), 
                                     (planet_name + '_1000_3D_retrieval_results_' + model_tag + '_1D_atm-'), 
                                      planet_name, err_data, N_params, 4,
                                      provide_min_chi_square = False)


# Print program run time
t1 = time.perf_counter()
total1 = t1-t0


plot_opacity(1.0, 3000, wl)


print('Total run time = ' + str((total1)/3600.0) + ' hr')



