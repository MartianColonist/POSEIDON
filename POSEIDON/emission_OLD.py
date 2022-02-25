# Radiative transfer calculations for generating emission spectra

import numpy as np
from numba.core.decorators import jit
from utility import prior_index, prior_index_V2, closest_index
from config import N_D, R_p, R_s
import scipy.constants as sc
from atmosphere import profiles
from absorption import extinction

import matplotlib.pyplot as plt


def Forward_model_emission(PT_state, R_p_ref, clouds_state, offsets, X_state, X_active,
                           X_cia, X_ff, X_bf, wl, sigma_stored, cia_stored, Rayleigh_stored, 
                           eta_stored, ff_stored, bf_stored):
    
    ''' Generates a PT profile, spectrum, and model data points from a given
        parameter state vector.
       
    '''
    
    N_regions = 1   # Only 1D models supported currently for emission spectra
        
    #***** Calculate radial profiles *****#
     
    # Check if the P-T profile parameters are physical
    if (PT_param == 'Madhu'):
        
        for j in range(PT_regions):
        
            log_P1 = PT_state[2,j]
            log_P2 = PT_state[3,j]
            log_P3 = PT_state[4,j]
        
            # Madhu & Seager 2009 profile requires P3 > P2 and P3 > P1
            if ((log_P3 < log_P2) or (log_P3 < log_P1)):
                is_physical = False
                
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, is_physical   # Unphysical => reject model
    
    #***** Unpack additional model parameters from state vector *****#
    
    # If haze is enabled in this model
    if (enable_haze == 1):
        a = clouds_state[np.where(cloud_params == 'a')][0]
        gamma = clouds_state[np.where(cloud_params == 'gamma')][0]
    else:
        a, gamma = 1.0, -4.0   # Dummy values, not used for models without hazes
    
    # If cloud deck enabled
    if (enable_deck == 1):
        P_cloud = clouds_state[np.where(cloud_params == 'P_cloud')][0]
    else:
        P_cloud = 100.0   # Set to 100 bar for models without a cloud deck
           
    #***** Calculate extinction coefficients *****#
            
    # If computing line-by-line radiative transfer, use special optimised functions 
    if (opacity_treatment == 'line-by-line'):  
        
        # Compute Rayleigh scattering cross sections and refractive indicies
        Rayleigh_stored, eta_stored = store_Rayleigh_eta_LBL(wl, chemical_species)   # Only need to do this step here for line-by-line case
        
        # If profile parameters are physical, then proceed to calculate profiles
        P, T, n, r, r_up, r_low, dr, mu, \
        eta, wl_eta, dlneta_dr, is_physical = profiles(PT_state, R_p_ref, X_state, 
                                                       wl, eta_stored, chemical_species)
        
        # If P-T profile is found to have unphysical temperatures
        if (is_physical == False):
                
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, is_physical   # Unphysical => reject model
        
        # Calculate extinction coefficients in line-by-line mode        
        kappa_chem, kappa_Rayleigh, \
        kappa_haze, kappa_cloud = extinction_LBL(chemical_species, active_species, 
                                                 cia_pairs, ff_pairs, bf_species, 
                                                 n, T, P, wl, X_state, X_active, 
                                                 X_cia, X_ff, X_bf, a, gamma, P_cloud,
                                                 Rayleigh_stored, enable_haze, 
                                                 enable_deck, N_regions)
        
        print("Beginning line-by-line radiative transfer")
       
    # If not doing line-by-line calculation (e.g. retrieval at lower-R than native cross section resolution)
    else:
            
        # If profile parameters are physical, then proceed to calculate profiles
        P, T, n, r, r_up, r_low, dr, mu, \
        eta, wl_eta, dlneta_dr, is_physical = profiles(PT_state, R_p_ref, X_state, 
                                                       wl, eta_stored, chemical_species)
            
        # If P-T profile is found to have unphysical temperatures
        if (is_physical == False):
                    
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, is_physical   # Unphysical => reject model
        
        # Calculate extinction coefficients in standard mode
        kappa_chem, kappa_Rayleigh, \
        kappa_haze, kappa_cloud = extinction(N_species, N_species_active, N_cia_pairs,
                                             N_ff_pairs, N_bf_species, n, T, P, wl,
                                             X_state, X_active, X_cia, X_ff, X_bf,
                                             a, gamma, P_cloud, sigma_stored, cia_stored,
                                             Rayleigh_stored, ff_stored, bf_stored,
                                             enable_haze, enable_deck, N_regions)
    
    #***** Solve radiative transfer *****#
        
    spectrum = emission_radiative_transfer_1st_order(P, T, dr, wl, kappa_chem)
    
    #***** Finally, convolve and bin model to resolution of the data *****# 
            
    # Initialise combined array of binned model points (all instruments)
    ymodel = np.array([])
    
    if (1 == 2):
            
        # Generate binned model points for each instrument
        for i in range(len(datasets)):
                    
            if (instruments[i] in ['IRAC1', 'IRAC2']): photometric = True
            else: photometric = False
                    
            if (i==0): ymodel_i = make_model_data(spectrum, wl, sigma_ALL[:len_data_sum[i]], 
                                                  sens_ALL[i*N_wl:(i+1)*N_wl], 
                                                  bin_left_ALL[:len_data_sum[i]],
                                                  bin_cent_ALL[:len_data_sum[i]], 
                                                  bin_right_ALL[:len_data_sum[i]],
                                                  half_bin[:len_data_sum[i]],
                                                  norm_ALL[:len_data_sum[i]],
                                                  photometric)
                        
            else: ymodel_i = make_model_data(spectrum, wl, sigma_ALL[len_data_sum[i-1]:len_data_sum[i]], 
                                             sens_ALL[i*N_wl:(i+1)*N_wl], 
                                             bin_left_ALL[len_data_sum[i-1]:len_data_sum[i]],
                                             bin_cent_ALL[len_data_sum[i-1]:len_data_sum[i]], 
                                             bin_right_ALL[len_data_sum[i-1]:len_data_sum[i]],
                                             half_bin[len_data_sum[i-1]:len_data_sum[i]],
                                             norm_ALL[len_data_sum[i-1]:len_data_sum[i]],
                                             photometric)
                                                  
            # Combine binned model points for each instrument
            ymodel = np.concatenate([ymodel, ymodel_i])    
        
    return spectrum, ymodel, P, T, r, mu, is_physical

@jit(nopython = True)
def planck_lambda(T, wl):
    
    ''' Compute the Planck spectral radiance for a range of model wavelengths 
        and atmospheric temperatures.
        
        Inputs: 
            
        wl => array of model wavelengths (m)
        T => array giving temperatures in each atmospheric layer (K)
        
        Outputs:
            
        B_lambda => Planck function array [T, wl] W m^-2 sr^-1 m^-1)
    
    '''
    
    # Define Planck function array
    B_lambda = np.zeros(shape=(len(T),len(wl)))  # (Temperature, wavelength)
    
    # Second radiative constant
    c_2 = (sc.h * sc.c) / sc.k
    
    # For each wavelength
    for k in range(len(wl)):
        
        # Precompute Planck function coefficient prefactor
        coeff = (2.0 * sc.h * sc.c**2) / (wl[k]**5)
        
        # For each atmospheric layer
        for i in range(len(T)):
            
            # Evalaute Planck function spectral radiance
            B_lambda[i,k] = coeff * (1.0 / (np.exp(c_2 / (wl[k] * T[i])) - 1.0))
            
    return B_lambda * 1.0e-6

@jit(nopython = True)
def emission_radiative_transfer_1st_order(P, T_all, dr, wl, kappa_chem):
    
    ''' Descriptive text.
    '''

    Gauss_order = 2
    
    if (Gauss_order == 2):
        
        W = np.array([0.5, 0.5])
        mu = np.array([0.5 - 0.5*np.sqrt(1.0/3.0), 0.5 + 0.5*np.sqrt(1.0/3.0)])
        
    elif (Gauss_order == 3):
        
        W = np.array([5.0/18.0, 4.0/9.0, 5.0/18.0])
        mu = np.array([0.5 - 0.5*np.sqrt(3.0/5.0), 0.5, 0.5 + 0.5*np.sqrt(3.0/5.0)])
    
    # The total extinction coefficient (without scattering) comes only from chemical absorption    
    kappa_tot = kappa_chem[:,0,:]  # 0 index to only consider one region for 1D models
    
    # Remove region dependance from layer thickness array
    dz = dr[:,0]   # 0 index to only consider one region for 1D models
    T = T_all[:,0]
    
    # Calculate Planck function in each layer and each wavelength
    B = planck_lambda(T, wl)
    
    # Initialise intensity (at base of atmosphere for each wavelength and mu) as Planck function 
    I = np.ones(shape=(len(mu),len(wl))) * B[0,:]
    
    # Initialise flux array
    F = np.zeros(len(wl))
    
    # For each wavelength
    for k in range(len(wl)):
    
        # For each ray travelling at mu = cos(theta)
        for j in range(len(mu)):

            # For each atmospheric layer
            for i in range(len(T)):
    
                # Compute vertical optical depth in this layer at given wavelength
                tau = kappa_tot[i,k] * dz[i]
                
                # Compute transmissivity of this layer at given wavelength
                Transmission = np.exp((-1.0 * tau)/mu[j])
                
                # Solve for emergent intensity from the layer top at given wavelength
                I[j,k] = Transmission * I[j,k] + (1.0 - Transmission) * B[i,k]
                
            F[k] += 2.0 * np.pi * mu[j] * I[j,k] * W[j]
    
    return F

