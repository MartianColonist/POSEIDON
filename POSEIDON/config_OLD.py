# POSEIDON configuraton settings

import numpy as np

model_tag = '2D_EM_cloudy'

#***** Define physical constants *****#

R_J = 7.1492e7     # Radius of Jupiter (m)
M_J = 1.898e27     # Mass of Jupiter (kg)
R_E = 6.371e6      # Radius of Earth (m)
R_Sun = 6.957e8    # Radius of Sun (m)

#***** System parameters *****#

planet_name = 'HATP26b'

# Stellar properties
Band = 'J'            # Spectral band stellar magnitude measured in (options: 'J', 'H', 'K')
App_mag = 10.08       # Apparant magnitude of the star
R_s = 0.87*R_Sun      # Radius of star (m)
T_s = 5079.0          # Stellar effective temperature (K)
err_T_s = 88.0       # Error on known stellar effective temperature (K)
Met_s = -0.04          # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
log_g_s = 4.56        # Stellar log surface gravity (cgs by convention)

# Planet properties
R_p = 0.63*R_J      # Radius of planet (m)
M_p = 0.07*M_J      # Mass of planet (kg)
g_0 = 4.3712        # Gravitational field of planet: g = GM/R_p^2 (m/s^2)
a_p = 11.839*R_s      # Semi major axis of planetary orbit (m)
b_p = -0.3*R_s     # Impact parameter of planet orbit (m)
T_eq = 1043.8       # Equilibrium temperature (currently unused)

Trans_dur = 2.4552  # Transit duration (hours)

#***** Atmosphere Setup *****#

N_D = 100           # Number of depths (layer centres) in atmosphere
P_max = 1.0e2       # Pressure at lowest altitude considered (bar)
P_min = 1.0e-9      # Pressure at highest altitude considered (bar)

#***** Wavelength grid *****#
 
wl_grid = 'constant R'   # Options: 'uniform' / 'constant R' / 'line-by-line'
wl_min = 0.599             # Minimum wavelength (um)
wl_max = 5.3             # Maximum wavelength (um)
N_wl = 1000              # Number of wavelength points for evalauting spectrum (uniform grid) - value here ignored if uniform not chosen
R = 4000                 # Spectral resolution for evalauting spectrum (constant R)      

#***** Enable or disable program features *****#

mode = 'plot'   # forward_model / retrieve / plot

if (mode == 'forward_model'):
    
    single_model = True
    load_observations = True
    do_retrieval = False
    sim_retrieval = False
    produce_sim_data = False
    sim_data_from_file = False
    check_best_fit = False
    plot_retrieved_spectrum = False
    analyse_Bayes = False 
    make_corner = False
    
    skip_preload = True   # Ignore initialisation step (only for debugging!)

elif (mode == 'retrieve'):
    
    single_model = False
    load_observations = True
    do_retrieval = True
    sim_retrieval = True
    produce_sim_data = False
    sim_data_from_file = False
    check_best_fit = False
    plot_retrieved_spectrum = False
    analyse_Bayes = False
    make_corner = True
    
    skip_preload = False   # Ignore initialisation step (only for debugging!)
    
elif (mode == 'plot'):
    
    single_model = False
    load_observations = False
    do_retrieval = False
    sim_retrieval = True
    produce_sim_data = False
    sim_data_from_file = False
    check_best_fit = False
    plot_retrieved_spectrum = False
    analyse_Bayes = False
    make_corner = True
    
    skip_preload = False   # Ignore initialisation step (only for debugging!)

#***** Model settings *****#

spectrum_type = 'transmission'    # Options: transmission / emission
rad_transfer = 'geometric'        # Options: geometric (transmission only)

X_dim = '2'                       # Options: 1 (uniform) / 2 (Evening-Morning or Day-Night) / 3 (Evening-Morning-Day-Night)
PT_dim = '2'                      # Options: 1 (uniform) / 2 (Evening-Morning or Day-Night) / 3 (Evening-Morning-Day-Night)
cloud_dim = '2'                   # Options: 1 (uniform) / 2 (Patchy terminator) / 3 (Patchy terminator + Day-Night)
TwoD_type = 'E-M'                 # Options: E-M (2D Evening-Morning model) / D-N (2D Day-Night model)
TwoD_param_scheme = 'difference'    # Options: absolute / difference
term_transition = 'linear'        # Options: linear (linear terminator transition - 2D / 3D models only)
PT_profile = 'gradient'           # Options: isotherm / gradient / Madhu
X_profile = 'isochem'             # Options: isochem / gradient
cloud_model = 'Iceberg'           # Options: cloud-free / MacMad17 / Iceberg
cloud_type = 'deck'               # Options: deck_haze / haze / deck (only applies if cloud_model != cloud-free)
He_fraction_setting = 'fixed'     # Options: fixed / free (if set to free, then X_He becomes a free parameter)
chemistry_prior = 'log-uniform'   # Options: log-uniform (Bulk component known) / CLR (a priori unknown, max 8 gases)
stellar_contam = 'No'             # Options: No / one-spot
offsets_applied = 'No'            # Options: No / relative
error_inflation = 'No'            # Options: No / Line_2015

# Now specify which chemical species to include in this model
bulk_species = ['H2', 'He']           # Species filling most of atmosphere

param_species = ['H2O', 'CO', 'CH4', 'CO2']    # Chemical species with parametrised mixing ratios

# Specify the chemical species for which we consider non-uniform gradients 
species_EM_gradient = ['CO', 'CH4', 'CO2']
species_DN_gradient = []
species_vert_gradient = []

# If fixing He/H2 ratio, use this value
He_fraction = 0.17

#***** Single model run parameters *****#

# (1) Mixing ratios (corresponding to included chemical species)

# Note: H2 and He not included here, since they are given by sum to unity condition.
#       For other bulk components, first species in chemical_species is omitted in X_set

# 1D model (global average)
if (X_dim == '1'):  
    if (X_profile == 'isochem'):  
        log_X_set = np.array([[-6.0, -7.0, -3.3]])     # Log abundances
    elif (X_profile == 'gradient'):  
        log_X_set = np.array([[-5.0, -7.0],      # High log abundances
                              [-5.0, -7.0]])     # Deep log abundances
        
# 2D model (asymmetric terminator or day-night transition)
elif (X_dim == '2'):
    if (TwoD_param_scheme == 'absolute'):
        if (TwoD_type == 'E-M'):
            if (X_profile == 'isochem'):  
                log_X_set = np.array([[-2.3, -2.5, -4.0, -5.0],        # Evening log abundances
                                      [-2.3, -6.5, -2.5, -8.0]])       # Morning log abundances
            elif (X_profile == 'gradient'):
                log_X_set = np.array([[-6.0, -7.0],        # Evening log abundances
                                      [-4.0, -7.0],        # Morning log abundances
                                      [-5.0, -7.0]])       # Deep log abundances
        if (TwoD_type == 'D-N'):
            if (X_profile == 'isochem'):  
                log_X_set = np.array([[-6.0,  -5.3, -3.0],      # Day log abundances
                                      [-10.0, -3.3, -3.0]])    # Night log abundances
            elif (X_profile == 'gradient'):  
                log_X_set = np.array([[-6.0, -7.0],        # Day log abundances
                                      [-4.0, -7.0],        # Night log abundances
                                      [-5.0, -7.0]])       # Deep log abundances
        
    elif (TwoD_param_scheme == 'difference'):
        if (TwoD_type == 'E-M'):
            if (X_profile == 'isochem'):  
                log_X_set = np.array([[-2.3, -4.5, -3.25, -6.5],        # Average log abundances
                                      [-0.0, +4.0, -1.5, +3.0]])       # Terminator log differences (M to E)
            elif (X_profile == 'gradient'):  
                log_X_set = np.array([[-5.0, -7.0],        # High (average) log abundances
                                      [-2.0, -0.0],        # Terminator log differences (M to E)
                                      [-5.0, -7.0]])       # Deep log abundances
        if (TwoD_type == 'D-N'):
            if (X_profile == 'isochem'):  
                log_X_set = np.array([[-8.0, -4.3, -3.0],        # Average log abundances
                                      [+4.0, -2.0, -0.0]])       # Day-Night log differences (N to D)
   #             log_X_set = np.array([[-9.41, -4.53, -3.0],        # Average log abundances
   #                                   [+0.24, -3.89, -0.0]])       # Day-Night log differences (N to D)
            elif (X_profile == 'gradient'):  
                log_X_set = np.array([[-5.0, -7.0],        # High (average) log abundances
                                      [-2.0, -0.0],        # Day-Night log differences (N to D)
                                      [-5.0, -7.0]])       # Deep log abundances
    
# 3D model (asymmetric terminator + day-night transition)
elif (X_dim == '3'):
    if (X_profile == 'isochem'):  
        log_X_set = np.array([[-5.0, -7.0],        # High (average) log abundances
                              [+0.0, +0.0],        # Terminator log differences (E to M)
                              [+2.0, +0.0]])       # Day-Night log differences (D to N)
    elif (X_profile == 'gradient'):  
        log_X_set = np.array([[-5.0, -7.0],        # High (average) log abundances
                              [+0.0, +0.0],        # Terminator log differences (E to M)
                              [+2.0, +0.0],        # Day-Night log differences (D to N)
                              [-5.0, -7.0]])       # Deep log abundances

# (2) PT profile parameters
    
P_deep = 10.0                 # 'Anchor' pressure below which the atmosphere is homogenous
P_high = 1.0e-5               # Pressure where temperature parameters are defined ('gradient' PT profile)
P_ref_set = 10.0              # Reference pressure (bar)

R_p_ref_set = 0.84*(R_p/R_J)  # Radius at reference pressure
#R_p_ref_set = 0.9617647*(R_p/R_J)  # Radius at reference pressure

# 1D model (global average)
if (PT_dim == '1'):  
    if (PT_profile == 'isotherm'):  
        PT_set = [1200]                   # T
    elif (PT_profile == 'gradient'):  
        PT_set = [800, 1600]            # T_high, T_deep
    elif (PT_profile == 'Madhu'):     
        PT_set = [1.2, 1.0, -2.0, -5.0, 0.0, 2200.0]  # a1, a2, log(P1,2,3), T_deep

# 2D model (asymmetric terminator or day-night transition)
elif (PT_dim == '2'):
    if (TwoD_param_scheme == 'absolute'):
        if (TwoD_type == 'E-M'):
            PT_set = [1000, 600, 1400]          # T_Even, T_Morn, T_deep
        elif (TwoD_type == 'D-N'):
            PT_set = [3000, 1000, 2500]         # T_Day, T_Night, T_deep
    elif (TwoD_param_scheme == 'difference'):
        if (TwoD_type == 'E-M'):
            PT_set = [800, 400, 1400]          # T_bar_term, Delta_T_term, T_deep
        elif (TwoD_type == 'D-N'):
            PT_set = [2000, 2000, 2500]         # T_bar_DN, Delta_T_DN, T_deep
    #        PT_set = [2140, 2207, 2577]         # T_bar_DN, Delta_T_DN, T_deep


# 3D model (asymmetric terminator + day-night transition)
elif (PT_dim == '3'):
    PT_set = [1900, 600, 1590, 2500]     # T_bar_term, Delta_T_term, Delta_T_DN, T_deep

# (3) Cloud profile parameters
    
# 1D model (uniform clouds)
if (cloud_dim == '1'):  
    if (cloud_model == 'cloud-free'):   # No cloud parameters for clear atmosphere
        clouds_set = []
    else:            
        if (cloud_model == 'MacMad17'):
            if (cloud_type == 'deck_haze'):
                clouds_set = [2.0, -7.0, -3.0]    # log(a), gamma, log(P_cloud)
            elif (cloud_type == 'haze'):      
                clouds_set = [2.0, -8.0]          # log(a), gamma
            elif (cloud_type == 'deck'):      
                clouds_set = [-1.6]               # log(P_cloud)
        elif (cloud_model == 'Iceberg'):
            if (cloud_type == 'deck'):      
                clouds_set = [-7.5, -2.0]        # log(kappa_cloud), log(P_cloud)
            
# 2D model (patchy clouds)
elif (cloud_dim == '2'):
    if (cloud_model == 'cloud-free'):   # No cloud parameters for clear atmosphere
        clouds_set = []
    else:
        if (TwoD_type == 'E-M'):
            if (cloud_model == 'MacMad17'):
                if (cloud_type == 'deck_haze'):
                    clouds_set = [2.0, -7.0, -3.0, 0.5, 45.0]    # log(a), gamma, log(P_cloud), phi_c, phi_0
                elif (cloud_type == 'haze'):      
                    clouds_set = [2.0, -8.0, 0.5, 45.0]          # log(a), gamma, phi_c, phi_0
                elif (cloud_type == 'deck'):      
                    clouds_set = [-3.0, 0.5, 45.0]               # log(P_cloud), phi_c, phi_0
            elif (cloud_model == 'Iceberg'):
                if (cloud_type == 'deck'):      
                    clouds_set = [-6.0, -3.0, 0.4, 20.0]        # log(kappa_cloud), log(P_cloud), f_cloud, phi_0
        elif (TwoD_type == 'D-N'):       
            if (cloud_model == 'Iceberg'):
                if (cloud_type == 'deck'):      
                    clouds_set = [-7.0, -3.0, 2.0]               # log(kappa_cloud), log(P_cloud), theta_0

# 3D model (Iceberg cloud)
elif (cloud_dim == '3'):  
    if (cloud_model == 'cloud-free'):   # No cloud parameters for clear atmosphere
        clouds_set = []
    else:
        if (cloud_model == 'Iceberg'):
            if (cloud_type == 'deck'):      
                clouds_set = [100.0, -3.0, 0.5, 45.0, 5.0]       # log(kappa_cloud), log(P_cloud), f_cloud, phi_0, theta_0
        
# (4) Stellar contamination parameters
            
if (stellar_contam == 'No'):    # No stellar contamination
    stellar_set = []
elif (stellar_contam == 'one-spot'):
    stellar_set = [0.05, 6800.0, T_s]     # f_het, T_het, T_phot
    
# (5) Geometry parameters (2D and 3D models)

alpha_set = 40.0   # Angular width of Evening-Morning terminator transition (2D E-M and 3D only)
beta_set = 10.0    # Angular width of Day-Night terminator transition (2D D-N and 3D only)
#beta_set = 35

# (6) Data offset parameters
            
if (offsets_applied == 'No'):            # No offset
    offsets_set = []            
elif (offsets_applied == 'relative'):    # Relative offset between two data sets
    offsets_set = [0.000116]  

# (7) Data errorbar adjustment parameters
    
if (error_inflation == 'No'):          # No errorbar inflation
    err_inflation_set = []    
elif (error_inflation == 'Line2015'):  # Error inflation accroding to Line+2015
    err_inflation_set = [2.0]

#***** Absorption.py settings *****#

opacity_treatment = 'Opacity-sample'  # Options: 'line-by-line' / 'Opacity-sample'
opacity_database = 'High-T'        # Options: 'High-T' / 'Temperate'

line_by_line_resolution = 0.01    # If using line-by-line mode, specify wavenumber resolution of cross sections calculation

T_fine_step = 10.0      # Temperature resolution for pre-interpolation of opacities (K)
T_fine_min = 200.0      # Minimum temperature on fine temperature grid (K)
T_fine_max = 2000.0     # Maximum temperature on fine temperature grid (K)

N_D_pre_inp = 40       # Pressure resolution for pre-interpolation of opacities (number of points over atmosphere grid)

#***** Stellar.py settings *****#

T_phot_min = T_s - 10.0*err_T_s     # Minimum T_phot on pre-computed grid (-10 sigma)
T_phot_max = T_s + 10.0*err_T_s     # Maximum T_phot on pre-computed grid (+10 sigma)
T_phot_step = err_T_s/10.0          # T_phot pre-computed grid resolution (0.1 sigma)

T_het_min = 0.6*T_s     # Minimum T_het on pre-computed grid
T_het_max = 1.2*T_s     # Maximum T_het on pre-computed grid
T_het_step = 10.0       # T_het pre-computed grid resolution (K)

#***** Geometry.py settings *****#

# Resolution for integration along / across terminator (if term_transition = 'linear')
N_slice_EM = 4    # Number of azimuthal slices across Evening-Morning terminator transition (Even number !)
N_slice_DN = 4     # Number of angular slices along Day-Night terminator transition(Even number !)

#***** Synthetic data generation propeties *****#

# Settings for simple constant precision constant resolution run

syn_data_tag = '3D_Retrieval_' + model_tag    # Specify prefix for output data files



syn_instrument = 'STIS_G430'     # Specify instrument to simulate

std_data = 100       # Standard deviation of synthetic data (ppm)
R_data = 20          # Spectral resolution of synthetic data (wl/delta_wl)
wl_data_min = 0.32   # Starting wavelength of synthetic data range (um)
wl_data_max = 0.54   # Ending wavelength of synthetic data range (um)

'''
syn_instrument = 'STIS_G750'     # Specify instrument to simulate

std_data = 100       # Standard deviation of synthetic data (ppm)
R_data = 20          # Spectral resolution of synthetic data (wl/delta_wl)
wl_data_min = 0.557   # Starting wavelength of synthetic data range (um)
wl_data_max = 0.97   # Ending wavelength of synthetic data range (um)
'''
'''
syn_instrument = 'WFC3_G141'     # Specify instrument to simulate

std_data = 50        # Standard deviation of synthetic data (ppm)
R_data = 60          # Spectral resolution of synthetic data (wl/delta_wl)
wl_data_min = 1.1    # Starting wavelength of synthetic data range (um)
wl_data_max = 1.7    # Ending wavelength of synthetic data range (um)
'''

#***** Data sources and settings *****#

# Specify file locations of each data file
#niriss_SOSS_Ord2_data = planet_name + '_2D_EM_N_trans_1_NIRISS SOSS_Ord2_R_100.dat'
#niriss_SOSS_Ord1_data = planet_name + '_2D_EM_N_trans_1_NIRISS SOSS_Ord1_R_100.dat'
#nirspec_G395H_data = planet_name + '_2D_EM_N_trans_1_NIRSpec G395H_R_100.dat'
niriss_SOSS_Ord2_data = planet_name + '_SYNTHETIC_3D_Retrieval_' + model_tag + '_JWST_NIRISS_SOSS_Ord2.dat'
niriss_SOSS_Ord1_data = planet_name + '_SYNTHETIC_3D_Retrieval_' + model_tag + '_JWST_NIRISS_SOSS_Ord1.dat'
nirspec_G395H_data = planet_name + '_SYNTHETIC_3D_Retrieval_' + model_tag + '_JWST_NIRSpec_G395H.dat'

# Provide full list of instruments to use
instruments = np.array(['JWST_NIRISS_SOSS_Ord2', 'JWST_NIRISS_SOSS_Ord1', 'JWST_NIRSpec_G395H']) 

# Specify datasets above to use
datasets = [niriss_SOSS_Ord2_data, niriss_SOSS_Ord1_data, nirspec_G395H_data]


# If considering a relative offset
offset_datasets = []    # Identify dataset on which to apply relative offset 

#***** Retrieval settings *****#

# Set lower prior limits for parameters
prior_lower_X = np.array([-14.0, -8.0])                    # Lower prior for log(X_i) and delta log(X_i)
prior_lower_geometry = np.array([0.1, 0.1])                # Lower prior for alpha and beta (degrees)
prior_lower_offsets = np.array([-1.0e-3, -1.0e-3])         # Lower priors for linear DC offsets and relative offset 
prior_lower_R_p_ref = 0.85*(R_p/R_J)                       # Lower prior for R_p_ref (in Jupiter radii)
prior_lower_stellar = np.array([0.0, 0.6*T_s])             # Lower priors for f, T_het

# Set upper prior limits for parameters
prior_upper_X = np.array([-1.0, 8.0])                      # Upper prior for log(X_i) and delta log(X_i)
prior_upper_geometry = np.array([180.0, 70.0])             # Upper prior for alpha and beta (degrees)
prior_upper_offsets = np.array([1.0e-3, 1.0e-3])           # Upper priors for linear DC offsets and relative offset 
prior_upper_R_p_ref = 1.15*(R_p/R_J)                       # Upper prior for R_p_ref (in Jupiter radii)
prior_upper_stellar = np.array([0.5, 1.2*T_s])             # Upper priors for f, T_het

# Set mean and std for Gaussian parameter priors
prior_gauss_T_phot = np.array([T_s, err_T_s])              # Gaussian prior for T_phot

# Cloud parameter priors
if (cloud_model == 'MacMad17'):
    prior_lower_clouds = np.array([-4.0, -20.0, -6.0, 0.0, -180.0])     # Lower priors for log(a), gamma, log(P_cloud), phi_c, phi_0
    prior_upper_clouds = np.array([8.0, 2.0, 2.0, 1.0, 180.0])          # Upper priors for log(a), gamma, log(P_cloud), phi_c, phi_0
elif (cloud_model == 'Iceberg'):
    prior_lower_clouds = np.array([-10.0, -6.0, 0.0, -180.0, -35.0])    # Lower priors for log(kappa_cloud), log(P_cloud), f_cloud, phi_0, theta_0
    prior_upper_clouds = np.array([-4.0, 2.0, 1.0, 180.0, 35.0])        # Upper priors for log(kappa_cloud), log(P_cloud), f_cloud, phi_0, theta_0
else:
    prior_lower_clouds = np.array([])
    prior_upper_clouds = np.array([])

# P-T profile priors
if (PT_profile == 'isotherm'):
    prior_lower_PT = np.array([T_fine_min])      # Lower prior for T
    prior_upper_PT = np.array([T_fine_max])      # Upper prior for T
elif (PT_profile == 'gradient'):
    prior_lower_PT = np.array([T_fine_min, T_fine_min, 0.0])           # Lower priors for T_high, T_deep, delta_T
    prior_upper_PT = np.array([T_fine_max, T_fine_max, 1000.0])      # Upper priors for T_high, T_deep, delta_T    
elif (PT_profile == 'Madhu'):
    prior_lower_PT = np.array([0.02, 0.02, -6.0, -6.0, -2.0, T_fine_min])   # Lower priors for a1, a2, log(P1,2,3), T_deep
    prior_upper_PT = np.array([2.00, 2.00, 1.0, 1.0, 1.0, T_fine_max])     # Upper priors for a1, a2, log(P1,2,3), T_deep

# Specify sampling algorithm and properties
sampling_algorithm = 'MultiNest'    # Options: MultiNest

sampling_target = 'parameter'       # Options: parameter / model 

ev_tol = 0.5                        # Evidence tolerance factor

N_live = 2000                       # Number of MultiNest live points

base_name = planet_name + '_' + str(N_live) + '_3D_retrieval_' + model_tag + '_full-'    # Output file base name
#base_name = planet_name + '_' + str(N_live) + '_' + model_tag + '_test-'    # Output file base name
