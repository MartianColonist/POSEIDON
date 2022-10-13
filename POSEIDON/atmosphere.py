# ***** Handles P-T and mixing ratio profiles *****

import numpy as np
import scipy.constants as sc
from scipy.ndimage import gaussian_filter1d as gauss_conv
from scipy.interpolate import pchip_interpolate
from numba.core.decorators import jit

from .supported_opac import inactive_species
from .species_data import masses
from .utility import prior_index


@jit(nopython = True)
def compute_T_Madhu(P, a1, a2, log_P1, log_P2, log_P3, T_deep, P_set = 1.0e-2):
    
    ''' Computes the temperature profile for an atmosphere using a re-arranged
        form of the P-T profile in Madhusudhan & Seager (2009).
       
        Inputs:
        
        P => pressure of each layer (bar)
        PT_state => P-T profile parameters defined in Madhu & Seager (2009)
                    in each atmospheric region

        Outputs:
           
        T => temperature of each layer (K)
       
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Initialise temperature array
    T = np.zeros(shape=(N_layers, 1, 1)) # 1D profile => N_sectors = N_zones = 1
    
    # Find index of pressure closest to the set pressure
    i_set = np.argmin(np.abs(P - P_set))
    P_set_i = P[i_set]
    
    # Store logarithm of various pressure quantities
    log_P = np.log10(P)
    log_P_min = np.log10(np.min(P))
    log_P_set_i = np.log10(P_set_i)

    # By default (P_set = 10 bar), so T(P_set) should be in layer 3
    if (log_P_set_i >= log_P3):
        
        T3 = T_deep  # T_deep is the isothermal deep temperature T3 here
        
        # Use the temperature parameter to compute boundary temperatures
        T2 = T3 - ((1.0/a2)*(log_P3 - log_P2))**2    
        T1 = T2 + ((1.0/a2)*(log_P1 - log_P2))**2    
        T0 = T1 - ((1.0/a1)*(log_P1 - log_P_min))**2   
        
    # If a different P_deep has been chosen, solve equations for layer 2...
    elif (log_P_set_i >= log_P1):   # Temperature parameter in layer 2
        
        # Use the temperature parameter to compute the boundary temperatures
        T2 = T_deep - ((1.0/a2)*(log_P_set_i - log_P2))**2  
        T1 = T2 + ((1.0/a2)*(log_P1 - log_P2))**2   
        T3 = T2 + ((1.0/a2)*(log_P3 - log_P2))**2
        T0 = T1 - ((1.0/a1)*(log_P1 - log_P_min))**2   
        
    # ...or for layer 1
    elif (log_P_set_i < log_P1):  # Temperature parameter in layer 1
    
        # Use the temperature parameter to compute the boundary temperatures
        T0 = T_deep - ((1.0/a1)*(log_P_set_i - log_P_min))**2
        T1 = T0 + ((1.0/a1)*(log_P1 - log_P_min))**2   
        T2 = T1 - ((1.0/a2)*(log_P1 - log_P2))**2  
        T3 = T2 + ((1.0/a2)*(log_P3 - log_P2))**2
        
    # Compute temperatures within each layer
    for i in range(N_layers):
        
        if (log_P[i] >= log_P3):
            T[i,0,0] = T3
        elif ((log_P[i] < log_P3) and (log_P[i] > log_P1)):
            T[i,0,0] = T2 + np.power(((1.0/a2)*(log_P[i] - log_P2)), 2.0)
        elif (log_P[i] <= log_P1):
            T[i,0,0] = T0 + np.power(((1.0/a1)*(log_P[i] - log_P_min)), 2.0)

    return T

#@jit(nopython = True)
def compute_T_slope(P, T_phot, Delta_T_arr, log_P_phot = 0.5, 
                    log_P_arr = [-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0]):
    '''
    ADD DOCSTRING
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Initialise temperature and pressure points arrays
    T_points = np.zeros(len(log_P_arr) + 1)
    log_P_points = np.sort(np.append(log_P_arr, log_P_phot))
    log_P_arr = np.array(log_P_arr)
    
    # Store number of temperature points defining slope parametrisation
    N_T_points = len(T_points)

    # Find index of layer containing the photosphere pressure parameter
    i_phot = prior_index(log_P_phot, log_P_arr, 0)

    # Work from top of atmosphere down to photosphere
    for i in range(0, i_phot+1):

        if (i == 0):
            T_points[i] = T_phot - np.sum(Delta_T_arr[i_phot::-1])
        else:
            T_points[i] = T_phot - np.sum(Delta_T_arr[i_phot:i-1:-1])

    # Add photosphere temperature
    T_points[i_phot+1] = T_phot

    # Work down from photosphere to bottom of atmosphere
    for i in range(i_phot+2, N_T_points):

        T_points[i] = T_phot + np.sum(Delta_T_arr[i_phot+1:i])

    # Initialise interpolated temperature array
    T = np.zeros(shape=(N_layers, 1, 1)) # 1D profile => N_sectors = N_zones = 1

    # Apply monotonic cubic interpolation to compute P-T profile from T points
    T[:,0,0] = pchip_interpolate(log_P_points, T_points, np.log10(P))

    return T


@jit(nopython = True)
def compute_T_field_gradient(P, T_bar_term, Delta_T_term, Delta_T_DN, T_deep,
                             N_sectors, N_zones, alpha, beta, phi, theta,
                             P_deep = 10.0, P_high = 1.0e-5):
    
    ''' Creates 3D temperature profile array storing T(P, phi, theta).
    
        For each atmospheric column, the temperature profile has a constant
        vertical gradient across the observable atmosphere. For pressures
        above and below the considered range (by default, 10^-5 -> 10 bar),
        the temperature is isothermal.
           
        Inputs:
            
        TBD

        Outputs:
           
        T => Array of temperature profiles for each sector and zone
       
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Initialise temperature arrays
    T = np.zeros(shape=(N_layers, N_sectors, N_zones))
    
    # Convert alpha and beta from degrees to radians
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    
    # Compute evening and morning temperatures in terminator plane
    T_Evening = T_bar_term + Delta_T_term/2.0
    T_Morning = T_bar_term - Delta_T_term/2.0
    
    # Compute 3D temperature field throughout atmosphere 
    for j in range(N_sectors):
        
        # Compute high temperature in terminator plane for given angle phi
        if (phi[j] <= -alpha_rad/2.0):
            T_term = T_Evening
        elif ((phi[j] > -alpha_rad/2.0) and (phi[j] < alpha_rad/2.0)):
            T_term = T_bar_term - (phi[j]/(alpha_rad/2.0)) * (Delta_T_term/2.0)
        elif (phi[j] >= -alpha_rad/2.0):
            T_term = T_Morning
            
        # Compute dayside and nightside temperatures for given angle phi
        T_Day   = T_term + Delta_T_DN/2.0
        T_Night = T_term - Delta_T_DN/2.0
        
        for k in range(N_zones):
            
            # Compute high temperature for given angles phi and theta
            if (theta[k] <= -beta_rad/2.0):
                T_high = T_Day
            elif ((theta[k] > -beta_rad/2.0) and (theta[k] < beta_rad/2.0)):
                T_high = T_term - (theta[k]/(beta_rad/2.0)) * (Delta_T_DN/2.0)
            elif (theta[k] >= -beta_rad/2.0):
                T_high = T_Night
                
            # Compute T gradient for layers between P_high and P_deep
            dT_dlogP = (T_deep - T_high) / (np.log10(P_deep/P_high))
            
            for i in range(N_layers):
                
                # Compute temperature profile for each atmospheric column
                if (P[i] <= P_high):
                    T[i,j,k] = T_high
                elif ((P[i] > P_high) and (P[i] < P_deep)):
                    T[i,j,k] = T_high + dT_dlogP * np.log10(P[i] / P_high)
                elif (P[i] >= P_deep):
                    T[i,j,k] = T_deep
        
    return T


@jit(nopython = True)
def compute_T_field_two_gradients(P, T_bar_term_high, T_bar_term_mid, 
                                  Delta_T_term_high, Delta_T_term_mid,
                                  Delta_T_DN_high, Delta_T_DN_mid, log_P_mid,
                                  T_deep, N_sectors, N_zones, alpha, beta,
                                  phi, theta, P_deep = 10.0, P_high = 1.0e-5):
    
    ''' Creates 3D temperature profile array storing T(P, phi, theta).
    
        For each atmospheric column, the temperature profile has two constant
        vertical gradients across the observable atmosphere. The transition 
        point between the two gradients is at log_P_mid. For pressures
        above and below the considered range (by default, 10^-5 -> 10 bar)
        the temperature is isothermal.
           
        Inputs:
            
        TBD

        Outputs:
           
        T => Array of temperature profiles for each sector and zone
       
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Initialise temperature arrays
    T = np.zeros(shape=(N_layers, N_sectors, N_zones))
    
    # Convert alpha and beta from degrees to radians
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    
    # Compute evening and morning temperatures in terminator plane
    T_Evening_high = T_bar_term_high + Delta_T_term_high/2.0
    T_Evening_mid  = T_bar_term_mid + Delta_T_term_mid/2.0
    T_Morning_high = T_bar_term_high - Delta_T_term_high/2.0
    T_Morning_mid  = T_bar_term_mid - Delta_T_term_mid/2.0

    P_mid = np.power(10.0, log_P_mid)

    # Compute 3D temperature field throughout atmosphere 
    for j in range(N_sectors):
        
        # Compute high and mid temperature in terminator plane for given angle phi
        if (phi[j] <= -alpha_rad/2.0):
            T_term_high = T_Evening_high
            T_term_mid  = T_Evening_mid
        elif ((phi[j] > -alpha_rad/2.0) and (phi[j] < alpha_rad/2.0)):
            T_term_high = T_bar_term_high - (phi[j]/(alpha_rad/2.0)) * (Delta_T_term_high/2.0)
            T_term_mid  = T_bar_term_mid - (phi[j]/(alpha_rad/2.0)) * (Delta_T_term_mid/2.0)
        elif (phi[j] >= -alpha_rad/2.0):
            T_term_high = T_Morning_high
            T_term_mid  = T_Morning_mid
            
        # Compute dayside and nightside temperatures for given angle phi
        T_Day_high   = T_term_high + Delta_T_DN_high/2.0
        T_Day_mid    = T_term_mid + Delta_T_DN_mid/2.0
        T_Night_high = T_term_high - Delta_T_DN_high/2.0
        T_Night_mid  = T_term_mid - Delta_T_DN_mid/2.0
        
        for k in range(N_zones):
            
            # Compute high and mid temperature for given angles phi and theta
            if (theta[k] <= -beta_rad/2.0):
                T_high = T_Day_high
                T_mid  = T_Day_mid
            elif ((theta[k] > -beta_rad/2.0) and (theta[k] < beta_rad/2.0)):
                T_high = T_term_high - (theta[k]/(beta_rad/2.0)) * (Delta_T_DN_high/2.0)
                T_mid  = T_term_mid - (theta[k]/(beta_rad/2.0)) * (Delta_T_DN_mid/2.0)
            elif (theta[k] >= -beta_rad/2.0):
                T_high = T_Night_high
                T_mid  = T_Night_mid
                
            # Compute T gradients
            dT_dlogP_1 = (T_mid - T_high) / (np.log10(P_mid/P_high))
            dT_dlogP_2 = (T_deep - T_mid) / (np.log10(P_deep/P_mid))
            
            for i in range(N_layers):
                
                # Compute temperature profile for each atmospheric column
                if (P[i] <= P_high):
                    T[i,j,k] = T_high
                elif ((P[i] > P_high) and (P[i] < P_mid)):
                    T[i,j,k] = T_high + dT_dlogP_1 * np.log10(P[i] / P_high)
                elif ((P[i] >= P_mid) and (P[i] < P_deep)):
                    T[i,j,k] = T_mid + dT_dlogP_2 * np.log10(P[i] / P_mid)
                elif (P[i] >= P_deep):
                    T[i,j,k] = T_deep
        
    return T


@jit(nopython = True)
def compute_X_field_gradient(P, log_X_state, N_sectors, N_zones, param_species, 
                             species_has_profile, alpha, beta, phi, theta, 
                             P_deep = 10.0, P_high = 1.0e-5):
    
    ''' Creates 4D abundance profile array storing X(species, P, phi, theta).
    
        For each atmospheric column, the abundance profile has either a constant 
        vertical gradient across the observable atmosphere or a uniform (isochem) 
        profile. The species with a gradient profile are specified by the user. 
        For gradient profiles, pressures above and below the considered range 
        (10^-5 -> 10 bar) have isochemical mixing ratios.
           
        Inputs:
            
        TBD

        Outputs:
           
        X => Array of abundance profiles for each species in each sector and zone
       
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Store lengths of species arrays for convenience
    N_param_species = len(param_species)
    
    # Initialise mixing ratio array
    X_profiles = np.zeros(shape=(N_param_species, N_layers, N_sectors, N_zones))

    # Convert alpha and beta from degrees to radians
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    
    # Loop over parametrised chemical species
    for q in range(N_param_species):
        
        # Unpack abundance field parameters for this species
        log_X_bar_term, Delta_log_X_term, \
        Delta_log_X_DN, log_X_deep = log_X_state[q,:]
        
        # Convert average terminator and deep abundances into linear space
        X_bar_term = np.power(10.0, log_X_bar_term)
        X_deep = np.power(10.0, log_X_deep)
    
        # Compute evening and morning abundances in terminator plane
        X_Evening = X_bar_term * np.power(10.0, (Delta_log_X_term/2.0))
        X_Morning = X_bar_term * np.power(10.0, (-Delta_log_X_term/2.0))
        
        # Compute 3D abundance field for species q throughout atmosphere 
        for j in range(N_sectors):
            
            # Compute high abundance in terminator plane for given angle phi
            if (phi[j] <= -alpha_rad/2.0):
                X_term = X_Evening
            elif ((phi[j] > -alpha_rad/2.0) and (phi[j] < alpha_rad/2.0)):
                X_term = X_bar_term * np.power(10.0, (-(phi[j]/(alpha_rad/2.0)) * (Delta_log_X_term/2.0)))
            elif (phi[j] >= -alpha_rad/2.0):
                X_term = X_Morning
                
            # Compute dayside and nightside abundances for given angle phi
            X_Day   = X_term * np.power(10.0, (Delta_log_X_DN/2.0))
            X_Night = X_term * np.power(10.0, (-Delta_log_X_DN/2.0))
            
            for k in range(N_zones):
                
                # Compute high abundance for given angles phi and theta
                if (theta[k] <= -beta_rad/2.0):
                    X_high = X_Day
                elif ((theta[k] > -beta_rad/2.0) and (theta[k] < beta_rad/2.0)):
                    X_high = X_term * np.power(10.0, (-(theta[k]/(beta_rad/2.0)) * (Delta_log_X_DN/2.0)))
                elif (theta[k] >= -beta_rad/2.0):
                    X_high = X_Night
                    
                # If the given species has a vertical profile with a gradient
                if (species_has_profile[q] == 1):  

                    # Compute abundance gradient for layers between P_high and P_deep
                    dlog_X_dlogP = np.log10(X_deep/X_high) / np.log10(P_deep/P_high)
                    
                    for i in range(N_layers):
                        
                        # Compute abundance profile for each atmospheric column
                        if (P[i] <= P_high):
                            X_profiles[q,i,j,k] = X_high
                        elif ((P[i] > P_high) and (P[i] < P_deep)):
                            X_profiles[q,i,j,k] = X_high * np.power((P[i] / P_high), dlog_X_dlogP)
                        elif (P[i] >= P_deep):
                            X_profiles[q,i,j,k] = X_deep
                  
                # Otherwise abundance is uniform with pressure
                else:
                    
                    X_profiles[q,:,j,k] = X_high
            
    return X_profiles


@jit(nopython = True)
def compute_X_field_two_gradients(P, log_X_state, N_sectors, N_zones, param_species, 
                                  species_has_profile, alpha, beta, phi, theta, 
                                  P_deep = 10.0, P_high = 1.0e-5):
    
    ''' Creates 4D abundance profile array storing X(species, P, phi, theta).
    
        For each atmospheric column, the abundance profile has either two 
        constant vertical gradients across the observable atmosphere or a 
        uniform (isochem) profile. The transition point between the two 
        gradients is at log_P_X_mid (species dependent). The species with 
        gradient profiles are specified by the user. For gradient profiles, 
        pressures above and below the considered range (10^-5 -> 10 bar) have 
        isochemical mixing ratios.
           
        Inputs:
            
        TBD

        Outputs:
           
        X => Array of abundance profiles for each species in each sector and zone
       
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Store lengths of species arrays for convenience
    N_param_species = len(param_species)
    
    # Initialise mixing ratio array
    X_profiles = np.zeros(shape=(N_param_species, N_layers, N_sectors, N_zones))

    # Convert alpha and beta from degrees to radians
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    
    # Loop over parametrised chemical species
    for q in range(N_param_species):
        
        # Unpack abundance field parameters for this species
        log_X_bar_term_high, log_X_bar_term_mid, \
        Delta_log_X_term_high, Delta_log_X_term_mid, \
        Delta_log_X_DN_high, Delta_log_X_DN_mid, \
        log_P_X_mid, log_X_deep = log_X_state[q,:]
        
        # Convert average terminator and deep abundances into linear space
        X_bar_term_high = np.power(10.0, log_X_bar_term_high)
        X_bar_term_mid = np.power(10.0, log_X_bar_term_mid)
        P_mid = np.power(10.0, log_P_X_mid)
        X_deep = np.power(10.0, log_X_deep)
 
        # Compute evening and morning temperatures in terminator plane
        X_Evening_high = X_bar_term_high * np.power(10.0, (Delta_log_X_term_high/2.0))
        X_Evening_mid  = X_bar_term_mid * np.power(10.0, (Delta_log_X_term_mid/2.0))
        X_Morning_high = X_bar_term_high * np.power(10.0, (-Delta_log_X_term_high/2.0))
        X_Morning_mid  = X_bar_term_mid * np.power(10.0, (-Delta_log_X_term_mid/2.0))

        # Compute 3D abundance field for species q throughout atmosphere 
        for j in range(N_sectors):
            
            # Compute high and mid abundance in terminator plane for given angle phi
            if (phi[j] <= -alpha_rad/2.0):
                X_term_high = X_Evening_high
                X_term_mid  = X_Evening_mid
            elif ((phi[j] > -alpha_rad/2.0) and (phi[j] < alpha_rad/2.0)):
                X_term_high = X_bar_term_high * np.power(10.0, (-(phi[j]/(alpha_rad/2.0)) * (Delta_log_X_term_high/2.0)))
                X_term_mid  = X_bar_term_mid * np.power(10.0, (-(phi[j]/(alpha_rad/2.0)) * (Delta_log_X_term_mid/2.0)))
            elif (phi[j] >= -alpha_rad/2.0):
                X_term_high = X_Morning_high
                X_term_mid  = X_Morning_mid
                
            # Compute dayside and nightside abundances for given angle phi
            X_Day_high   = X_term_high * np.power(10.0, (Delta_log_X_DN_high/2.0))
            X_Day_mid    = X_term_mid * np.power(10.0, (Delta_log_X_DN_mid/2.0))
            X_Night_high = X_term_high * np.power(10.0, (-Delta_log_X_DN_high/2.0))
            X_Night_mid  = X_term_mid * np.power(10.0, (-Delta_log_X_DN_mid/2.0))
  
            for k in range(N_zones):
                
                # Compute high abundance for given angles phi and theta
                if (theta[k] <= -beta_rad/2.0):
                    X_high = X_Day_high
                    X_mid  = X_Day_mid
                elif ((theta[k] > -beta_rad/2.0) and (theta[k] < beta_rad/2.0)):
                    X_high = X_term_high * np.power(10.0, (-(theta[k]/(beta_rad/2.0)) * (Delta_log_X_DN_high/2.0)))
                    X_mid  = X_term_mid * np.power(10.0, (-(theta[k]/(beta_rad/2.0)) * (Delta_log_X_DN_mid/2.0)))
                elif (theta[k] >= -beta_rad/2.0):
                    X_high = X_Night_high
                    X_mid  = X_Night_mid
                    
                # If the given species has a vertical profile with a gradient
                if (species_has_profile[q] == 1):  

                    # Compute abundance gradients
                    dlog_X_dlogP_1 = np.log10(X_mid/X_high) / np.log10(P_mid/P_high)
                    dlog_X_dlogP_2 = np.log10(X_deep/X_mid) / np.log10(P_deep/P_mid)
                    
                    for i in range(N_layers):
                        
                        # Compute abundance profile for each atmospheric column
                        if (P[i] <= P_high):
                            X_profiles[q,i,j,k] = X_high
                        elif ((P[i] > P_high) and (P[i] < P_mid)):
                            X_profiles[q,i,j,k] = X_high * np.power((P[i] / P_high), dlog_X_dlogP_1)
                        elif ((P[i] >= P_mid) and (P[i] < P_deep)):
                            X_profiles[q,i,j,k] = X_mid * np.power((P[i] / P_mid), dlog_X_dlogP_2)
                        elif (P[i] >= P_deep):
                            X_profiles[q,i,j,k] = X_deep
                  
                # Otherwise abundance is uniform with pressure
                else:
                    
                    X_profiles[q,:,j,k] = X_high
            
    return X_profiles


#@jit(nopython = True)
def add_bulk_component(P, X_param, N_species, N_sectors, N_zones, bulk_species,
                       He_fraction):
    
    ''' Add abundances of one or more bulk species to the parametrised mixing
        ratios to form full mixing ratio array.
    
    '''

    # Store number of layers for convenience
    N_layers = len(P)

    # Store lengths of species arrays for convenience
    N_bulk_species = len(bulk_species)
        
    # Initialise mixing ratio arrays
    X = np.zeros(shape=(N_species, N_layers, N_sectors, N_zones))
    
    # For H2+He bulk mixture
    if ('H2' and 'He' in bulk_species):
    
        # Compute H2 and He mixing ratios for a fixed H2/He fraction (defined in config.py)
        X_H2 = (1.0 - np.sum(X_param, axis=0))/(1.0 + He_fraction)   # H2 mixing ratio array
        X_He = He_fraction*X_H2                                      # He mixing ratio array
                
        # Add H2 and He mixing ratios to first two elements in X state vector for this region
        X[0,:,:,:] = X_H2  
        X[1,:,:,:] = X_He
        
    # For any other choice of bulk species, the first mixing ratio is the bulk species
    else: 

        if (len(bulk_species) > 1):
            raise Exception("Only a single species can be designated as bulk " +
                            "(besides models with H2 & He with a fixed He/H2 ratio).")
        
        # Calculate first mixing ratio in state vector
        X_0 = 1.0 - np.sum(X_param, axis=0)   

        # Add first mixing ratio to X state vector for this region
        X[0,:,:,:] = X_0   
        
    # Fill remainder of mixing ratio array with the trace species
    X[N_bulk_species:,:,:,:] = X_param
    
    return X


@jit(nopython = True)
def radial_profiles(P, T, g_0, R_p, P_ref, R_p_ref, mu, N_sectors, N_zones):
    
    ''' Solves the equation of hydrostatic equilibrium [ dP/dr = -G*M*rho/r^2 ] 
        to compute the radius in each atmospheric layer.
        
        Note: g is taken as an inverse square law with radius, by assuming the
              enclosed planet mass at a given radius is approx. M_p. This is
              valid as most mass is in the interior (atmosphere mass negligible).
       
        Inputs:
        
        P => pressure of each layer (bar)
        T => temperature of each layer (K)
        R_p_ref => radius at reference pressure
        mu => mean molecular mass of atmosphere (kg)
        N_sectors => 

        Outputs:
           
        n => number density of each layer (m^-3)
        r => radius at centre of each layer (m)
        r_up => radius of top edge of each layer (m)
        r_low => radius of top edge of each layer (m)
        dr => radial thickness of each layer (m)
       
    '''

    # Store number of layers for convenience
    N_layers = len(P)

    # Initialise 3D radial profile arrays    
    r = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_up = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_low = np.zeros(shape=(N_layers, N_sectors, N_zones))
    dr = np.zeros(shape=(N_layers, N_sectors, N_zones))
    n = np.zeros(shape=(N_layers, N_sectors, N_zones))
    
    # Compute radial extent in each sector and zone from the corresponding T(P)
    for j in range(N_sectors):
        
        for k in range(N_zones):
    
            # Compute number density in each atmospheric layer (ideal gas law)
            n[:,j,k] = (P*1.0e5)/((sc.k)*T[:,j,k])   # 1.0e5 to convert bar to Pa
        
            # Set reference pressure and reference radius (r(P_ref) = R_p_ref)
            P_0 = P_ref      # 10 bar default value
            r_0 = R_p_ref    # Radius at reference pressure
        
            # Find index of pressure closest to reference pressure (10 bar)
            i_ref = np.argmin(np.abs(P - P_0))
        
            # Set reference radius
            r[i_ref,j,k] = r_0
        
            # Compute integrand for hydrostatic calculation
            integrand = (sc.k * T[:,j,k])/(R_p**2 * g_0 * mu[:,j,k] * P*1.0e5)
        
            # Initialise stored values of integral for outwards and inwards sums
            integral_out = 0.0
            integral_in = 0.0
    
            # Working outwards from reference pressure
            for i in range(i_ref+1, N_layers, 1):
            
                integral_out += 0.5 * (integrand[i] + integrand[i-1]) * (P[i] - P[i-1])*1.0e5  # Trapezium rule integration
            
                r[i,j,k] = 1.0/((1.0/r_0) + integral_out)
            
            # Working inwards from reference pressure
            for i in range((i_ref-1), -1, -1):   
            
                integral_in += 0.5 * (integrand[i] + integrand[i+1]) * (P[i] - P[i+1])*1.0e5   # Trapezium rule integration
            
                r[i,j,k] = 1.0/((1.0/r_0) + integral_in)
    
            # Use radial profile to compute thickness and boundaries of each layer
            for i in range(1, N_layers-1): 
            
                r_up[i,j,k] = 0.5*(r[(i+1),j,k] + r[i,j,k])
                r_low[i,j,k] = 0.5*(r[i,j,k] + r[(i-1),j,k])
                dr[i,j,k] = 0.5 * (r[(i+1),j,k] - r[(i-1),j,k])
            
            # Edge cases for bottom layer and top layer    
            r_up[0,j,k] = 0.5*(r[1,j,k] + r[0,j,k])
            r_up[(N_layers-1),j,k] = r[(N_layers-1),j,k] + 0.5*(r[(N_layers-1),j,k] - r[(N_layers-2),j,k])
        
            r_low[0,j,k] = r[0,j,k] - 0.5*(r[1,j,k] - r[0,j,k])
            r_low[(N_layers-1),j,k] = 0.5*(r[(N_layers-1),j,k] + r[(N_layers-2),j,k])
        
            dr[0,j,k] = (r[1,j,k] - r[0,j,k])
            dr[(N_layers-1),j,k] = (r[(N_layers-1),j,k] - r[(N_layers-2),j,k])
            
    return n, r, r_up, r_low, dr


#@jit(nopython = True)
def mixing_ratio_categories(P, X, N_sectors, N_zones, included_species, 
                            active_species, cia_pairs, ff_pairs, bf_species):
    
    ''' Sort mixing ratios into those of active species, collision-induced
        absorption (CIA), free-free opacity, and bound-free opacity.
        
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Store number of species in different mixing ratio categories
    N_species_active = len(active_species)
    N_cia_pairs = len(cia_pairs)
    N_ff_pairs = len(ff_pairs)
    N_bf_species = len(bf_species)
    
    # Initialise mixing ratio category arrays
    X_active = np.zeros(shape=(N_species_active, N_layers, N_sectors, N_zones))
    X_cia = np.zeros(shape=(2, N_cia_pairs, N_layers, N_sectors, N_zones))
    X_ff = np.zeros(shape=(2, N_ff_pairs, N_layers, N_sectors, N_zones))
    X_bf = np.zeros(shape=(N_bf_species, N_layers, N_sectors, N_zones))
    
    # Find indices of chemical species that actively absorb light
    active_idx = np.isin(included_species, inactive_species)

    # Store mixing ratios of active species
    X_active = X[~active_idx,:,:,:]
               
    # Find mixing ratios contributing to CIA             
    for q in range(N_cia_pairs):
                
        pair = cia_pairs[q]
        
        # Find index of each species in the CIA pair
        pair_idx_1 = (included_species == pair.split('-')[0])
        pair_idx_2 = (included_species == pair.split('-')[1])
        
        # Store mixing ratios of the two CIA components
        X_cia[0,q,:,:,:] = X[pair_idx_1,:,:,:]  # E.g. 'H2' for 'H2-He'
        X_cia[1,q,:,:,:] = X[pair_idx_2,:,:,:]  # E.g. 'He' for 'H2-He'
                
    # Find mixing ratios contributing to free-free absorption
    for q in range(N_ff_pairs):
        
        pair = ff_pairs[q]
                    
        # Store mixing ratios of the two free-free components (only H- currently supported)
        if (pair == 'H-ff'): 
            X_ff[0,q,:,:,:] = X[included_species == 'H',:,:,:]
            X_ff[1,q,:,:,:] = X[included_species == 'e-',:,:,:]
                    
    # Find mixing ratios contributing to bound-free absorption
    for q in range(N_bf_species):
                
        species = bf_species[q]
        
        # Store mixing ratio of each bound-free species
        if (species == 'H-bf'): 
            X_bf[q,:,:,:] = X[included_species == 'H-',:,:,:]
                
    return X_active, X_cia, X_ff, X_bf


@jit(nopython = True)
def compute_mean_mol_mass(P, X, N_species, N_sectors, N_zones, masses_all):
    
    ''' Computes the mean molecular mass of the atmosphere.
    
        Inputs:
            
        X => volume mixing ratio array for model atmosphere
        N_species => 
        
        Outputs:
            
        mu => mean molecular mass of atmosphere (kg)
    
    '''
    
    # Store number of layers for convenience
    N_layers = len(P)

    # Initialise mean molecular mass array
    mu = np.zeros(shape=(N_layers, N_sectors, N_zones))
    
    for i in range(N_layers):
        for j in range(N_sectors):
            for k in range(N_zones):
                for q in range(N_species):
            
                    mu[i,j,k] += X[q,i,j,k] * masses_all[q]   # masses array in atomic mass units
            
    mu = mu * sc.u  # Convert from atomic mass units to kg
            
    return mu



#***** TBD: replace functions below with a general function for any elemental ratio *****#


def locate_X(X, species, included_species):
    
    ''' Finds the mixing ratio for a specified species. Returns zero if
        specified species is not included in this model.
    
        Inputs:
            
        X => volume mixing ratios of atmosphere
        species => string giving desired chemical species
        included_species => array of strings listing chemical species included in model
        
        Outputs:
            
        X_species => mixing ratio for this species
    
    '''   
    
    if (species in included_species):
        X_species = X[included_species == species]
    else:
        X_species = 0.0
        
    return X_species

#@jit(nopython = True)
def compute_metallicity(X, all_species):
    
    ''' Computes the metallicity [ (O/H)/(O/H)_solar ] of the atmosphere.
    
        Inputs:
            
        X => volume mixing ratios of atmosphere
        all_species => array of strings listing chemical species included in model
        
        Outputs:
            
        M => metallicity of atmosphere
    
    '''
    
    O_to_H_solar = np.power(10.0, (8.69-12.0))  # Asplund (2009) ~ 4.9e-4  (Present day photosphere value)
    
    X_H2 = locate_X(X, 'H2', all_species)
    X_H2O = locate_X(X, 'H2O', all_species)
    X_CH4 = locate_X(X, 'CH4', all_species)
    X_NH3 = locate_X(X, 'NH3', all_species)
    X_HCN = locate_X(X, 'HCN', all_species)
    X_CO = locate_X(X, 'CO', all_species)
    X_CO2 = locate_X(X, 'CO2', all_species)
    X_C2H2 = locate_X(X, 'C2H2', all_species)
    X_TiO = locate_X(X, 'TiO', all_species)
    X_VO = locate_X(X, 'VO', all_species)
    X_AlO = locate_X(X, 'AlO', all_species)
    X_CaO = locate_X(X, 'CaO', all_species)
    X_TiH = locate_X(X, 'TiH', all_species)
    X_CrH = locate_X(X, 'CrH', all_species)
    X_FeH = locate_X(X, 'FeH', all_species)
    X_ScH = locate_X(X, 'ScH', all_species)
    
    O_to_H = ((X_H2O + X_CO + 2*X_CO2 + X_TiO + X_VO + X_AlO + X_CaO)/
              (2*X_H2 + 2*X_H2O + 4*X_CH4 + 3*X_NH3 + X_HCN + +2*X_C2H2 + X_TiH + X_CrH + X_FeH + X_ScH))
        
    M = O_to_H / O_to_H_solar
    
    return M

#@jit(nopython = True)
def compute_C_to_O(X, all_species):
    
    ''' Computes the carbon-to-oxygen ratio of the atmosphere.
    
        Inputs:
            
        X => volume mixing ratios of atmosphere
        all_species => array of strings listing chemical species included in model
        
        Outputs:
            
        C_to_O => C/O ratio of atmosphere
    
    '''
    
    X_H2O = locate_X(X, 'H2O', all_species)
    X_CH4 = locate_X(X, 'CH4', all_species)
    X_HCN = locate_X(X, 'HCN', all_species)
    X_CO = locate_X(X, 'CO', all_species)
    X_CO2 = locate_X(X, 'CO2', all_species)
    X_PO = locate_X(X, 'PO', all_species)
    X_C2H2 = locate_X(X, 'C2H2', all_species)
    X_TiO = locate_X(X, 'TiO', all_species)
    X_VO = locate_X(X, 'VO', all_species)
    X_AlO = locate_X(X, 'AlO', all_species)
    X_CaO = locate_X(X, 'CaO', all_species)
    
    C_to_O = ((X_CH4 + X_HCN + X_CO + X_CO2 + 2*X_C2H2)/
              (X_H2O + X_CO + 2*X_CO2 + X_PO + X_TiO + X_VO + X_AlO + X_CaO))
    
    return C_to_O

def compute_O_to_H(X, all_species):
    
    ''' Computes the metallicity [ (O/H)/(O/H)_solar ] of the atmosphere.
    
        Inputs:
            
        X => volume mixing ratios of atmosphere
        all_species => array of strings listing chemical species included in model
        
        Outputs:
            
        M => metallicity of atmosphere
    
    '''
    
    X_H2O = locate_X(X, 'H2O', all_species)
    X_CH4 = locate_X(X, 'CH4', all_species)
    X_NH3 = locate_X(X, 'NH3', all_species)
    X_HCN = locate_X(X, 'HCN', all_species)
    X_CO = locate_X(X, 'CO', all_species)
    X_CO2 = locate_X(X, 'CO2', all_species)
    X_C2H2 = locate_X(X, 'C2H2', all_species)
    X_TiO = locate_X(X, 'TiO', all_species)
    X_VO = locate_X(X, 'VO', all_species)
    X_AlO = locate_X(X, 'AlO', all_species)
    X_CaO = locate_X(X, 'CaO', all_species)
    X_TiH = locate_X(X, 'TiH', all_species)
    X_CrH = locate_X(X, 'CrH', all_species)
    X_FeH = locate_X(X, 'FeH', all_species)
    X_ScH = locate_X(X, 'ScH', all_species)

    X_H2 = (1.0 - X_H2O)/(1.0 + 0.17) #locate_X(X, 'H2', all_species)
    
    O_to_H = ((X_H2O + X_CO + 2*X_CO2 + X_TiO + X_VO + X_AlO + X_CaO)/
              (2*X_H2 + 2*X_H2O + 4*X_CH4 + 3*X_NH3 + X_HCN + 2*X_C2H2 + X_TiH + X_CrH + X_FeH + X_ScH))
    
    return O_to_H

def compute_C_to_H(X, all_species):
    
    ''' Computes the carbon-to-hydrogen ratio of the atmosphere.
    
        Inputs:
            
        X => volume mixing ratios of atmosphere
        all_species => array of strings listing chemical species included in model
        
        Outputs:
            
        C_to_H => C/H ratio of atmosphere
    
    '''
    
    X_H2 = locate_X(X, 'H2', all_species)
    X_H2O = locate_X(X, 'H2O', all_species)
    X_CH4 = locate_X(X, 'CH4', all_species)
    X_NH3 = locate_X(X, 'NH3', all_species)
    X_HCN = locate_X(X, 'HCN', all_species)
    X_CO = locate_X(X, 'CO', all_species)
    X_CO2 = locate_X(X, 'CO2', all_species)
    X_C2H2 = locate_X(X, 'C2H2', all_species)
    X_H2S = locate_X(X, 'H2S', all_species)
    X_PH3 = locate_X(X, 'PH3', all_species)
    X_TiH = locate_X(X, 'TiH', all_species)
    X_CrH = locate_X(X, 'CrH', all_species)
    X_FeH = locate_X(X, 'FeH', all_species)
    X_ScH = locate_X(X, 'ScH', all_species)
    
    C_to_H = ((X_CH4 + X_HCN + X_CO + X_CO2 + 2*X_C2H2)/
              (2*X_H2 + 2*X_H2O + 4*X_CH4 + 3*X_NH3 + X_HCN + 2*X_C2H2 + 2*X_H2S + 
               3*X_PH3 + X_TiH + X_CrH + X_FeH + X_ScH))
    
    return C_to_H

def compute_N_to_H(X, all_species):
    
    ''' Computes the carbon-to-hydrogen ratio of the atmosphere.
    
        Inputs:
            
        X => volume mixing ratios of atmosphere
        all_species => array of strings listing chemical species included in model
        
        Outputs:
            
        C_to_H => C/H ratio of atmosphere
    
    '''
    
    X_H2 = locate_X(X, 'H2', all_species)
    X_H2O = locate_X(X, 'H2O', all_species)
    X_CH4 = locate_X(X, 'CH4', all_species)
    X_NH3 = locate_X(X, 'NH3', all_species)
    X_HCN = locate_X(X, 'HCN', all_species)
    X_C2H2 = locate_X(X, 'C2H2', all_species)
    X_H2S = locate_X(X, 'H2S', all_species)
    X_PH3 = locate_X(X, 'PH3', all_species)
    X_PN = locate_X(X, 'PN', all_species)
    X_TiH = locate_X(X, 'TiH', all_species)
    X_CrH = locate_X(X, 'CrH', all_species)
    X_FeH = locate_X(X, 'FeH', all_species)
    X_ScH = locate_X(X, 'ScH', all_species)
    
    N_to_H = ((X_NH3 + X_HCN + X_PN)/
              (2*X_H2 + 2*X_H2O + 4*X_CH4 + 3*X_NH3 + X_HCN + 2*X_C2H2 + 2*X_H2S + 
               3*X_PH3 + X_TiH + X_CrH + X_FeH + X_ScH))
    
    return N_to_H
    
#*****************************************************************************************

def profiles(P, R_p, g_0, PT_profile, X_profile, PT_state, P_ref, R_p_ref, 
             log_X_state, included_species, bulk_species, param_species, 
             active_species, cia_pairs, ff_pairs, bf_species, N_sectors, 
             N_zones, alpha, beta, phi, theta, species_vert_gradient, 
             He_fraction, T_input, X_input):
    
    ''' Main function to evaluate radial profiles of various quantities.
    
        Notes: all profiles are indexed to start from the base of the atmosphere.
               Profiles indexed first by layer, and second by atmospheric region.
       
        Inputs:
            
        PT_state => state vector containing P-T profile parameters
        R_p_ref => radius at reference pressure (in Jupiter radii)
        log_X_state => volume mixing ratios of atmosphere
        wl => model wavelength grid (m)
        eta_stored => refractive indices on model wl grid at standard conditions
        included_species =>
        bulk_species -> 
        param_species => array of strings with parametrised chemical species
        active_species =>
        ignore_species =>
        N_sectors => number of azimuthal sectors
        N_zones => number of zones along day-night path
        alpha => Morning-Evening terminator opening angle 
        beta => Day-Night terminator opening angle
        phi =>
        theta =>
        species_vert_gradient =>
       
        Outputs:
           
        P => pressure of each layer (bar)
        T => temperature of each layer (K)
        n => number density of each layer (m^-3)
        r => radius at centre of each layer (m)
        r_up => radius of top edge of each layer (m)
        r_low => radius of top edge of each layer (m)
        dr => radial thickness of each layer (m)
        mu => mean molecular mass of atmosphere (kg)
        eta => refractive index in each layer
        wl_eta => wavelengths for which eta is calculated (m)
        dlneta_dr => derivative of natural log of refractive index w.r.t height
                     for each wavelength in wl_eta (m^-1)
        is_physical => boolean specifying if P-T profile within allowed T range
       
    '''

    # For an isothermal profile
    if (PT_profile == 'isotherm'):
        
        # Unpack P-T profile parameters
        T_iso = PT_state[0]
    
        # Initialise temperature array
        T = T_iso * np.ones(shape=(len(P), 1, 1)) # 1D profile => N_sectors = N_zones = 1
        
        # Gaussian smooth P-T profile
        T_smooth = T   # No need to Gaussian smooth an isothermal profile

    # For the gradient profiles (1D, 2D, or 3D)
    elif (PT_profile == 'gradient'):
        
        # Unpack P-T profile parameters
        T_bar_term, Delta_T_term, Delta_T_DN, T_deep = PT_state
        
        # Reject if T_Morning > T_Evening or T_Night > T_day
        if ((Delta_T_term < 0.0) or (Delta_T_DN < 0.0)): 
            
            # Quit computations if model rejected
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        
        # If P-T parameters valid
        else:
            
            # Compute unsmoothed temperature field
            T_rough = compute_T_field_gradient(P, T_bar_term, Delta_T_term, 
                                               Delta_T_DN, T_deep, N_sectors, 
                                               N_zones, alpha, beta, phi, theta)

        # Gaussian smooth P-T profile
        T_smooth = gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')

    # For the two-gradients profiles (1D, 2D, or 3D)
    elif (PT_profile == 'two-gradients'):
        
        # Unpack P-T profile parameters
        T_bar_term_high, T_bar_term_mid, \
        Delta_T_term_high, Delta_T_term_mid, \
        Delta_T_DN_high, Delta_T_DN_mid, log_P_mid, T_deep = PT_state
        
        # Reject if T_Morning > T_Evening or T_Night > T_day
        if (((Delta_T_term_high < 0.0) or (Delta_T_DN_high < 0.0)) or
            ((Delta_T_term_mid < 0.0) or (Delta_T_DN_mid < 0.0))): 
            
            # Quit computations if model rejected
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        
        # If P-T parameters valid
        else:
            
            # Compute unsmoothed temperature field
            T_rough = compute_T_field_two_gradients(P, T_bar_term_high, T_bar_term_mid,
                                                    Delta_T_term_high, Delta_T_term_mid, 
                                                    Delta_T_DN_high, Delta_T_DN_mid,
                                                    log_P_mid, T_deep, N_sectors, 
                                                    N_zones, alpha, beta, phi, theta)

        # Gaussian smooth P-T profile
        T_smooth = gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')
        
    # For the Madhusudhan & Seager (2009) profile (1D only)
    elif (PT_profile == 'Madhu'):
        
        # Unpack P-T profile parameters
        a1, a2, log_P1, log_P2, log_P3, T_deep = PT_state
        
        # Profile requires P3 > P2 and P3 > P1, reject otherwise
        if ((log_P3 < log_P2) or (log_P3 < log_P1)):
            
            # Quit computations if model rejected
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        
        # If P-T parameters valid
        else:
            
            # Compute unsmoothed temperature profile
            T_rough = compute_T_Madhu(P, a1, a2, log_P1, log_P2, log_P3, T_deep)

        # Gaussian smooth P-T profile
        T_smooth = gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')

    # For the Piette & Madhusudhan (2020) profile (1D only)
    elif (PT_profile == 'slope'):
        
        # Unpack P-T profile parameters
        T_phot = PT_state[0]
        Delta_T_arr = np.array(PT_state[1:])
            
        # Compute unsmoothed temperature profile
        T_rough = compute_T_slope(P, T_phot, Delta_T_arr)

        # Find how many layers corresponds to 0.3 dex smoothing width
        smooth_width = round(0.3/(((np.log10(P[0]) - np.log10(P[-1]))/len(P))))

        # Gaussian smooth P-T profile
        T_smooth = gauss_conv(T_rough, sigma=smooth_width, axis=0, mode='nearest')

    # Read user provided P-T profile
    elif (PT_profile == 'file_read'):

        # Initialise temperature array
        T = T_input.reshape((len(P), 1, 1))   # 1D profile => N_sectors = N_zones = 1
        
        # Gaussian smooth P-T profile
        T_smooth = T   # No need to Gaussian smooth a user profile

    # Load number of distinct chemical species in model atmosphere
    N_species = len(bulk_species) + len(param_species)
    
    # Find which parametrised chemical species have a gradient profile
    species_has_profile = np.zeros(len(param_species)).astype(np.int64)
    
    if (X_profile in ['gradient', 'two-gradients']):
        species_has_profile[np.isin(param_species, species_vert_gradient)] = 1  
    
    # Read user provided mixing ratio profiles
    if (X_profile == 'file_read'):
        X = X_input.reshape((N_species, len(P), 1, 1))   

    else:   # Alternatively, compute 4D mixing ratio array

        # For isochemical or gradient profiles
        if (X_profile in ['isochem', 'gradient']):
            X_param = compute_X_field_gradient(P, log_X_state, N_sectors, N_zones, 
                                            param_species, species_has_profile, 
                                            alpha, beta, phi, theta)

        # For two-gradient profiles                            
        elif (X_profile == 'two-gradients'):
            X_param = compute_X_field_two_gradients(P, log_X_state, N_sectors, N_zones, 
                                                    param_species, species_has_profile, 
                                                    alpha, beta, phi, theta) 

        # Gaussian smooth any profiles with a vertical profile
        for q, species in enumerate(param_species):
            if (species_has_profile[q] == 1):
                X_param[q,:,:,:] = gauss_conv(X_param[q,:,:,:], sigma=3, axis=0, 
                                            mode='nearest')
        
        # Add bulk mixing ratios to form full mixing ratio array
        X = add_bulk_component(P, X_param, N_species, N_sectors, N_zones, 
                            bulk_species, He_fraction)
    
    # Check if any mixing ratios are negative (i.e. trace species sum to > 1, so bulk < 0)
    if (np.any(X[0,:,:,:] < 0.0)): 
        
        # Quit computations if model rejected
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False

    # Create mixing ratio category arrays for extinction calculations
    X_active, X_cia, \
    X_ff, X_bf = mixing_ratio_categories(P, X, N_sectors, N_zones, 
                                         included_species, active_species, 
                                         cia_pairs, ff_pairs, bf_species)
        
    # Store masses in an array to speed up mu calculation
    masses_all = np.zeros(N_species)
    
    # Load masses of each species from dictionary in species_data.py
    for q in range(N_species):
        species = included_species[q]
        masses_all[q] = masses[species]
    
    # Calculate mean molecular mass
    mu = compute_mean_mol_mass(P, X, N_species, N_sectors, N_zones, masses_all)
        
    # Calculate number density and radial profiles
    n, r, r_up, r_low, dr = radial_profiles(P, T_smooth, g_0, R_p, P_ref, 
                                            R_p_ref, mu, N_sectors, N_zones)
    
    return P, T_smooth, n, r, r_up, r_low, dr, X, X_active, X_cia, \
           X_ff, X_bf, mu, True
