''' 
Functions for calculating atmospheric temperature, mixing ratio, and other profiles.

'''

import numpy as np
import scipy.constants as sc
from scipy.ndimage import gaussian_filter1d as gauss_conv
from scipy.interpolate import pchip_interpolate
from numba.core.decorators import jit

from .supported_chemicals import inactive_species
from .species_data import masses
from .utility import prior_index
from .chemistry import interpolate_log_X_grid


@jit(nopython = True)
def compute_T_Madhu(P, a1, a2, log_P1, log_P2, log_P3, T_set, P_set):
    '''
    Computes the temperature profile for an atmosphere using a re-arranged
    form of the P-T profile parametrisation in Madhusudhan & Seager (2009).

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        a1 (float):
            Alpha_1 parameter (encodes slope in layer 1).
        a2 (float):
            Alpha_2 parameter encodes slope in layer 2).
        log_P1 (float):
            Pressure of layer 1-2 boundary.
        log_P2 (float):
            Pressure of inversion.
        log_P3 (float):
            Pressure of layer 2-3 boundary.
        T_set (float):
            Atmosphere temperature reference value at P = P_set (K).
        P_set (float):
            Pressure whether the temperature parameter T_set is defined (bar).
    
    Returns:
        T (3D np.array of float):
            Temperature of each layer as a function of pressure (K). 
            Only the first axis is used for this 1D profile.
    
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
        
        T3 = T_set  # T_deep is the isothermal deep temperature T3 here
        
        # Use the temperature parameter to compute boundary temperatures
        T2 = T3 - ((1.0/a2)*(log_P3 - log_P2))**2    
        T1 = T2 + ((1.0/a2)*(log_P1 - log_P2))**2    
        T0 = T1 - ((1.0/a1)*(log_P1 - log_P_min))**2   
        
    # If a different P_deep has been chosen, solve equations for layer 2...
    elif (log_P_set_i >= log_P1):   # Temperature parameter in layer 2
        
        # Use the temperature parameter to compute the boundary temperatures
        T2 = T_set - ((1.0/a2)*(log_P_set_i - log_P2))**2  
        T1 = T2 + ((1.0/a2)*(log_P1 - log_P2))**2   
        T3 = T2 + ((1.0/a2)*(log_P3 - log_P2))**2
        T0 = T1 - ((1.0/a1)*(log_P1 - log_P_min))**2   
        
    # ...or for layer 1
    elif (log_P_set_i < log_P1):  # Temperature parameter in layer 1
    
        # Use the temperature parameter to compute the boundary temperatures
        T0 = T_set - ((1.0/a1)*(log_P_set_i - log_P_min))**2
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


def compute_T_slope(P, T_phot, Delta_T_arr, log_P_phot = 0.5, 
                    log_P_arr = [-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0]):
    '''
    Computes the temperature profile for an atmosphere using the 'slope' P-T 
    profile parametrisation defined in Piette & Madhusudhan (2021).

    Note: The number of temperature difference parameters is the same as the
          number of pressure points (including the photosphere). For the default
          values, we have: [Delta_T (10-1mb), Delta_T (100-10mb), Delta_T (1-0.1b),
                            Delta_T (3.2-1b), Delta_T (10-3.2b), Delta_T (32-10b), 
                            Delta_T (100-32b)], where 'b' = bar.

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        T_phot (float):
            Temperature at the photosphere (located at log_P_phot) (K).
        Delta_T_arr (np.array of float):
            Temperature differences between each pressure point (K).
        log_P_phot (float):
            Photosphere pressure (default: 3.16 bar).
        log_P_arr (list):
            Pressures where the temperature difference parameters are defined (log bar).
    
    Returns:
        T (3D np.array of float):
            Temperature of each layer as a function of pressure (K).
            Only the first axis is used for this 1D profile.
    
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
    ''' 
    Creates the 3D temperature profile defined in MacDonald & Lewis (2022).
    
    Note: This profile assumes a linear in log-pressure temperature gradient 
          between P_deep and P_high (these anchor points are fixed such that 
          the photosphere generally lies within them). For pressures above and 
          below the considered range, the temperature is isothermal.
        
    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        T_bar_term (float):
            Average terminator plane temperature at P_high.
        Delta_T_term (float):
            Temperature difference between the evening and morning terminators at P_high(K).
        Delta_T_DN (float):
            Temperature difference between the dayside and nightside at P_high (K).
        T_deep (float):
            Global deep temperature at P_deep.
        N_sectors (int):
            Number of azimuthal sectors.
        N_zones (int):
            Number of zenith zones.
        alpha (float):
            Terminator opening angle (degrees).
        beta (float):
            Day-night opening angle (degrees).
        phi (np.array of float):
            Mid-sector angles (radians).
        theta (np.array of float):
            Mid-zone angles (radians).
        P_deep (float):
            Anchor point in deep atmosphere below which the atmosphere is homogenous.
        P_high (float):
            Anchor point high in the atmosphere above which all columns are isothermal.
    
    Returns:
        T (3D np.array of float):
            Temperature of each layer as a function of pressure, sector, and zone (K).
    
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
    ''' 
    Extension of 'compute_T_field_gradient' with a second gradient in the
    photosphere region. The two gradients connect at a new parameter: P_mid.
        
    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        T_bar_term_high (float):
            Average terminator plane temperature at P_high.
        T_bar_term_mid (float):
            Average terminator plane temperature at P_mid.
        Delta_T_term_high (float):
            Temperature difference between the evening and morning terminators at P_high (K).
        Delta_T_term_mid (float):
            Temperature difference between the evening and morning terminators at P_mid (K).
        Delta_T_DN_high (float):
            Temperature difference between the dayside and nightside at P_high (K).
        Delta_T_DN_mid (float):
            Temperature difference between the dayside and nightside at P_mid (K).
        log_P_mid (float):
            log10 of pressure where the two gradients switch (bar). 
        T_deep (float):
            Global deep temperature at P_deep.
        N_sectors (int):
            Number of azimuthal sectors.
        N_zones (int):
            Number of zenith zones.
        alpha (float):
            Terminator opening angle (degrees).
        beta (float):
            Day-night opening angle (degrees).
        phi (np.array of float):
            Mid-sector angles (radians).
        theta (np.array of float):
            Mid-zone angles (radians).
        P_deep (float):
            Anchor point in deep atmosphere below which the atmosphere is homogenous.
        P_high (float):
            Anchor point high in the atmosphere above which all columns are isothermal.
    
    Returns:
        T (3D np.array of float):
            Temperature of each layer as a function of pressure, sector, and zone (K).
    
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
    ''' 
    Creates the 4D abundance profile array storing X(species, layer, sector, zone).

    The functional dependence is the same as defined above in the function 
    'compute_T_field_gradient' (see also MacDonald & Lewis (2022)), with the 
    exception that not all chemical species have a vertical profile. Any 
    chemical species not in the array 'species_has_profile' have constant
    mixing ratios with altitude.
        
    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        log_X_state (2D np.array of float):
            Mixing ratio state array.
        N_sectors (int):
            Number of azimuthal sectors.
        N_zones (int):
            Number of zenith zones.
        param_species (np.array of str):
            Chemical species with parametrised mixing ratios.
        species_has_profile (np.array of int):
            Array with an integer '1' if a species in 'param_species' has a 
            gradient profile, or '0' for an constant mixing ratio with altitude.
        alpha (float):
            Terminator opening angle (degrees).
        beta (float):
            Day-night opening angle (degrees).
        phi (np.array of float):
            Mid-sector angles (radians).
        theta (np.array of float):
            Mid-zone angles (radians).
        P_deep (float):
            Anchor point in deep atmosphere below which all columns are isochemical.
        P_high (float):
            Anchor point high in the atmosphere above which all columns are isochemical.
    
    Returns:
        X_profiles (4D np.array of float):
            Mixing ratios in each layer as a function of pressure, sector, and zone.
    
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
    ''' 
    Extension of 'compute_X_field_gradient' with a second gradient in the
    photosphere region. The two gradients connect at P_mid.
        
    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        log_X_state (2D np.array of float):
            Mixing ratio state array.
        N_sectors (int):
            Number of azimuthal sectors.
        N_zones (int):
            Number of zenith zones.
        param_species (np.array of str):
            Chemical species with parametrised mixing ratios.
        species_has_profile (np.array of int):
            Array with an integer '1' if a species in 'param_species' has a 
            gradient profile, or '0' for an constant mixing ratio with altitude.
        alpha (float):
            Terminator opening angle (degrees).
        beta (float):
            Day-night opening angle (degrees).
        phi (np.array of float):
            Mid-sector angles (radians).
        theta (np.array of float):
            Mid-zone angles (radians).
        P_deep (float):
            Anchor point in deep atmosphere below which all columns are isochemical.
        P_high (float):
            Anchor point high in the atmosphere above which all columns are isochemical.
    
    Returns:
        X_profiles (4D np.array of float):
            Mixing ratios in each layer as a function of pressure, sector, and zone.
    
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


def compute_X_lever(P, log_X_state, species_has_profile, N_sectors, N_zones):
    '''
    The function takes in four parameters and returns an array of values called log_X  that represent the
    output of the function, which is an array of numbers that would be used to plot the profle of the chemical species.
    This is done by taking the difference in the logarithm of the pressures and the original array, log_xi
    and multiplying it by the slope of the isochemical line (angle between the isochemical line and the
    array log_xi). The volume mixing ratio log_xi is the number density / the total volume.

    The function takes in five parameters:

        log_xi: Logarithm of the mixing ratio at element i.
        log_pi: Logarithm of the pressure at element i.
        upsilon_i: The angle between the local vertical and the slope for element i.
        log_p: An array of logarithm of the pressures.

    Returns:
        log_x: the the mixing ratio  of the ith element as a function of pressure.
    '''

    log_p = np.log10(P)
    N_param_species = np.shape(log_X_state)[0]
    log_X = np.zeros(shape = (N_param_species, len(P), N_sectors, N_zones))
    
    # Loop over gases
    for q in range(N_param_species):

        # Unpack the abundance
        log_X_q, log_P_q, upsilon_q = log_X_state[q, :]

        # For angles between 0 and pi/2
        if np.abs(upsilon_q) <= np.pi/2:

            # Loop over layers
            for j in range (len(log_p)) :
                if log_p[j] <= log_P_q :
                    log_X[q,j,0,0] = log_X_q + np.tan(upsilon_q) * (log_p[j] - log_P_q)
                elif log_p[j] > log_P_q :
                    log_X[q,j,0,0] = log_X_q

        # For angles between pi/2  and pi
        elif (np.abs(upsilon_q) > np.pi/2) and (np.abs(upsilon_q) <= np.pi):
          for j in range (len(log_p)) :
              if log_p[j] > log_P_q :
                log_X[q,j,0,0] = log_X_q + np.tan(upsilon_q) * (log_p[j] - log_P_q)
              elif log_p[j] <= log_P_q :
                log_X[q,j,0,0] = log_X_q
        
    return np.power(10, log_X)

def add_bulk_component(P, X_param, N_species, N_sectors, N_zones, bulk_species,
                       He_fraction):
    ''' 
    Concatenates mixing ratios of the bulk species to the parametrised mixing
    ratios, forming the full mixing ratio array (i.e. sums to 1).

    Note: For H2 and He as bulk species, the output array has the mixing ratio
          profile of H2 as the first element and He second. For other bulk
          species (e.g. N2), that species occupies the first element.

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        X_param (4D np.array of float):
            Mixing ratios of the parametrised chemical species in each layer 
            as a function of pressure, sector, and zone.
        N_sectors (int):
            Number of azimuthal sectors.
        N_zones (int):
            Number of zenith zones.
        bulk_species (np.array of str):
            Bulk species dominating atmosphere (e.g. ['H2', 'He']).
        He_fraction (float):
            Assumed H2/He ratio (0.17 default corresponds to the solar ratio).
    
    Returns:
        X (4D np.array of float):
            Same as X_param, but with the bulk species appended.
    
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
def radial_profiles_test(P, T, g_0, R_p, P_ref, R_p_ref, mu, N_sectors, N_zones):
    ''' 
    Solves the equation of hydrostatic equilibrium [ dP/dr = -G*M*rho/r^2 ] 
    to compute the radius in each atmospheric layer.
        
    Note: g is taken as an inverse square law with radius by assuming the
          enclosed planet mass at a given radius is M_p. This assumes
          most mass is in the interior (negligible atmosphere mass).

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        T (3D np.array of float):
            Temperature profile (K).
        g_0 (float):
            Gravitational field strength at white light radius (m/s^2).
        R_p (float):
            Observed white light planet radius (m).
        P_ref (float):
            Reference pressure (bar).
        R_p_ref (float):
            Planet radius corresponding to reference pressure (m).
        mu (3D np.array of float):
            Mean molecular mass (kg).
        N_sectors (int):
            Number of azimuthal sectors comprising the background atmosphere.
        N_zones (int):
            Number of zenith zones comprising the background atmosphere.

    Returns:
        n (3D np.array of float):
            Number density profile (m^-3).
        r (3D np.array of float):
            Radial distance profile (m).
        r_up (3D np.array of float):
            Upper layer boundaries (m).
        r_low (3D np.array of float):
            Lower layer boundaries (m).    
        dr (3D np.array of float):
            Layer thicknesses (m).
    
    '''

    # Store number of layers for convenience
    N_layers = len(P)

    # Initialise 3D radial profile arrays    
    r = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_up = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_low = np.zeros(shape=(N_layers, N_sectors, N_zones))
    dr = np.zeros(shape=(N_layers, N_sectors, N_zones))
    n = np.zeros(shape=(N_layers, N_sectors, N_zones))

    log_P = np.log(P)

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

            # Iterative scheme
            tolerance = 0.1 #1e-6

            max_iterations = 100

            for i in range((i_ref+1), N_layers):

                r_prev = r[i-1,j,k]
                g_prev = g_0 * (R_p / r_prev)**2
                integrand_prev = (sc.k * T[i,j,k]) / (g_prev * mu[i,j,k])
                delta_log_P = (log_P[i] - log_P[i-1])
                converged = False

                count = 0

                r_proposed = r_prev
                integrand_proposed = integrand_prev
                
                while not converged and count < max_iterations:

                    count += 1

                    r_new = r_prev - 0.5*(integrand_prev + integrand_proposed) * delta_log_P
                    g_proposed = g_0 * (R_p / r_new)**2
                    integrand_proposed = (sc.k * T[i,j,k]) / (g_proposed * mu[i,j,k])
                #    r_new = r_prev - 0.5 * (integrand_prev + integrand_proposed) * delta_log_P

                    if np.abs(r_new - r_proposed) < tolerance:
                        converged = True
                    else:
                        r_proposed = r_new

                print(count)
                
                r[i] = r_new

            for i in range((i_ref-1), -1, -1):

                r_next = r[i+1,j,k]
                g_next = g_0 * (R_p / r_next)**2
                integrand_next = (sc.k * T[i,j,k]) / (g_next * mu[i,j,k])
                delta_log_P = log_P[i] - log_P[i+1]
                converged = False

                count = 0

                r_proposed = r_next
                integrand_proposed = integrand_next
                
                while not converged and count < max_iterations:

                    count +=1

                    r_new = r_next - 0.5 * (integrand_next + integrand_proposed) * delta_log_P
                    g_proposed = g_0 * (R_p / r_new)**2
                    integrand_proposed = (sc.k * T[i,j,k]) / (g_proposed * mu[i,j,k])
               #     r_new = r_next - 0.5 * (integrand_next + integrand_proposed) * delta_log_P

                    if np.abs(r_new - r_proposed) < tolerance:
                        converged = True
                    else:
                        r_proposed = r_new
                
                print(count)

                r[i] = r_new

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


@jit(nopython = True)
def radial_profiles(P, T, g_0, R_p, P_ref, R_p_ref, mu, N_sectors, N_zones):
    ''' 
    Solves the equation of hydrostatic equilibrium [ dP/dr = -G*M*rho/r^2 ] 
    to compute the radius in each atmospheric layer.
        
    Note: g is taken as an inverse square law with radius by assuming the
          enclosed planet mass at a given radius is M_p. This assumes
          most mass is in the interior (negligible atmosphere mass).

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        T (3D np.array of float):
            Temperature profile (K).
        g_0 (float):
            Gravitational field strength at white light radius (m/s^2).
        R_p (float):
            Observed white light planet radius (m).
        P_ref (float):
            Reference pressure (bar).
        R_p_ref (float):
            Planet radius corresponding to reference pressure (m).
        mu (3D np.array of float):
            Mean molecular mass (kg).
        N_sectors (int):
            Number of azimuthal sectors comprising the background atmosphere.
        N_zones (int):
            Number of zenith zones comprising the background atmosphere.

    Returns:
        n (3D np.array of float):
            Number density profile (m^-3).
        r (3D np.array of float):
            Radial distance profile (m).
        r_up (3D np.array of float):
            Upper layer boundaries (m).
        r_low (3D np.array of float):
            Lower layer boundaries (m).    
        dr (3D np.array of float):
            Layer thicknesses (m).
    
    '''

    # Store number of layers for convenience
    N_layers = len(P)

    # Initialise 3D radial profile arrays    
    r = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_up = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_low = np.zeros(shape=(N_layers, N_sectors, N_zones))
    dr = np.zeros(shape=(N_layers, N_sectors, N_zones))
    n = np.zeros(shape=(N_layers, N_sectors, N_zones))

    log_P = np.log(P)
    
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
            integrand = (sc.k * T[:,j,k])/(R_p**2 * g_0 * mu[:,j,k])

            # Initialise stored values of integral for outwards and inwards sums
            integral_out = 0.0
            integral_in = 0.0

            # Working outwards from reference pressure
            for i in range(i_ref+1, N_layers, 1):

                integral_out += 0.5 * (integrand[i] + integrand[i-1]) * (log_P[i] - log_P[i-1])  # Trapezium rule integration

                r[i,j,k] = 1.0/((1.0/r_0) + integral_out)

            # Working inwards from reference pressure
            for i in range((i_ref-1), -1, -1):   
            
                integral_in += 0.5 * (integrand[i] + integrand[i+1]) * (log_P[i] - log_P[i+1])   # Trapezium rule integration
            
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


@jit(nopython = True)
def radial_profiles_constant_g(P, T, g_0, P_ref, R_p_ref, mu, N_sectors, N_zones):
    ''' 
    Solves the equation of hydrostatic equilibrium [ dP/dr = -G*M*rho/r^2 ] 
    to compute the radius in each atmospheric layer.
        
    Note: This version of the solver assumes the gravitational field strength
          is constant with altitude (for testing purposes). The standard
          'radial_profiles' function should be used for any real calculation.

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        T (3D np.array of float):
            Temperature profile (K).
        g_0 (float):
            Gravitational field strength at white light radius (m/s^2).
        P_ref (float):
            Reference pressure (bar).
        R_p_ref (float):
            Planet radius corresponding to reference pressure (m).
        mu (3D np.array of float):
            Mean molecular mass (kg).
        N_sectors (int):
            Number of azimuthal sectors comprising the background atmosphere.
        N_zones (int):
            Number of zenith zones comprising the background atmosphere.

    Returns:
        n (3D np.array of float):
            Number density profile (m^-3).
        r (3D np.array of float):
            Radial distant profile (m).
        r_up (3D np.array of float):
            Upper layer boundaries (m).
        r_low (3D np.array of float):
            Lower layer boundaries (m).    
        dr (3D np.array of float):
            Layer thicknesses (m).
    
    '''

    # Store number of layers for convenience
    N_layers = len(P)

    # Initialise 3D radial profile arrays    
    r = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_up = np.zeros(shape=(N_layers, N_sectors, N_zones))
    r_low = np.zeros(shape=(N_layers, N_sectors, N_zones))
    dr = np.zeros(shape=(N_layers, N_sectors, N_zones))
    n = np.zeros(shape=(N_layers, N_sectors, N_zones))

    log_P = np.log(P)
    
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

            # Initialise stored values of integral for outwards and inwards sums
            integral_out = 0.0
            integral_in = 0.0

            # Compute integrand for hydrostatic calculation
            integrand = (sc.k * T[:,j,k])/(g_0 * mu[:,j,k])
        
            # Working outwards from reference pressure
            for i in range(i_ref+1, N_layers, 1):
            
                integral_out += 0.5 * (integrand[i] + integrand[i-1]) * (log_P[i] - log_P[i-1])  # Trapezium rule integration
            
                r[i,j,k] = r_0 - integral_out
            
            # Working inwards from reference pressure
            for i in range((i_ref-1), -1, -1):   
            
                integral_in += 0.5 * (integrand[i] + integrand[i+1]) * (log_P[i] - log_P[i+1])   # Trapezium rule integration
            
                r[i,j,k] = r_0 - integral_in

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


def mixing_ratio_categories(P, X, N_sectors, N_zones, included_species, 
                            active_species, CIA_pairs, ff_pairs, bf_species):
    ''' 
    Sort mixing ratios into those of active species, collision-induced
    absorption (CIA), free-free opacity, and bound-free opacity.

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        X (4D np.array of float):
            Mixing ratio profile
        N_sectors (int):
            Number of azimuthal sectors comprising the background atmosphere.
        N_zones (int):
            Number of zenith zones comprising the background atmosphere.
        included_species (np.array of str):
            List of chemical species included in the model (including bulk species).
        active_species (np.array of str):
            Spectroscopically active chemical species (see supported_opac.py).
        CIA_pairs (np.array of str):
            Collisionally-induced absorption (CIA) pairs.
        ff_pairs (np.array of str):
            Free-free absorption pairs.
        bf_species (np.array of str):
            Bound-free absorption species.

    Returns:
        X_active (4D np.array of float):
            Mixing ratios of active species.
        X_CIA (5D np.array of float):
            Mixing ratios of CIA pairs 
        X_ff (5D np.array of float):
            Mixing ratios of free-free pairs.
        X_bf (4D np.array of float):
            Mixing ratios of bound-free species.
    
    '''

    # Store number of layers for convenience
    N_layers = len(P)
    
    # Store number of species in different mixing ratio categories
    N_species_active = len(active_species)
    N_CIA_pairs = len(CIA_pairs)
    N_ff_pairs = len(ff_pairs)
    N_bf_species = len(bf_species)
    
    # Initialise mixing ratio category arrays
    X_active = np.zeros(shape=(N_species_active, N_layers, N_sectors, N_zones))
    X_CIA = np.zeros(shape=(2, N_CIA_pairs, N_layers, N_sectors, N_zones))
    X_ff = np.zeros(shape=(2, N_ff_pairs, N_layers, N_sectors, N_zones))
    X_bf = np.zeros(shape=(N_bf_species, N_layers, N_sectors, N_zones))
    
    # Find indices of chemical species that actively absorb light
    active_idx = np.isin(included_species, inactive_species)

    # Store mixing ratios of active species
    X_active = X[~active_idx,:,:,:]
               
    # Find mixing ratios contributing to CIA             
    for q in range(N_CIA_pairs):
                
        pair = CIA_pairs[q]
        
        # Find index of each species in the CIA pair
        pair_idx_1 = (included_species == pair.split('-')[0])
        pair_idx_2 = (included_species == pair.split('-')[1])
        
        # Store mixing ratios of the two CIA components
        X_CIA[0,q,:,:,:] = X[pair_idx_1,:,:,:]  # E.g. 'H2' for 'H2-He'
        X_CIA[1,q,:,:,:] = X[pair_idx_2,:,:,:]  # E.g. 'He' for 'H2-He'
                
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
                
    return X_active, X_CIA, X_ff, X_bf


@jit(nopython = True)
def compute_mean_mol_mass(P, X, N_species, N_sectors, N_zones, masses_all):
    ''' 
    Computes the mean molecular mass in each atmospheric column.

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        X (4D np.array of float):
            Mixing ratio profile.
        N_species (int):
            Number of chemical species.
        N_sectors (int):
            Number of azimuthal sectors comprising the background atmosphere.
        N_zones (int):
            Number of zenith zones comprising the background atmosphere.
        masses_all (np.array of float):
            Masses of each chemical species (atomic mass units, u)

    Returns:
        mu (3D np.array of float):
            Mean molecular mass (kg).
    
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


def count_atoms(molecule):
    '''
    Count how many atoms of each element are contained in a molecule.

    Args:
        molecule (str):
            Name of the molecule (e.g. 'H2O', 'CaOH', 'HC(CH3)3').

    Returns:
        counts (dict):
            Dictionary containing element counts (e.g. for H2O: {'H': 2, 'O': 1}).
    
    '''

    counts = {}   # Output dictionary
    i = 0         # Counter for character in molecule string

    while i < len(molecule):

        char = molecule[i]
        next_char = molecule[i+1] if i+1 < len(molecule) else None

        # If the current character is an uppercase letter or a lowercase letter, it represents the start of an element
        if char.isupper() or char.islower():
            element = char

            # If the next character is a lowercase letter, it represents the second letter of the element
            if next_char and next_char.islower():
                element += next_char
                i += 1

            counts[element] = 1   # Initialise count for this element to 1

            # If the next character is a number, it is the count for this element
            next_char = molecule[i+1] if i+1 < len(molecule) else None

            if next_char and next_char.isdigit():
                count = ''

                # Keep looping until we reach a character that is not a number
                while next_char and next_char.isdigit():
                    count += next_char
                    i += 1
                    next_char = molecule[i+1] if i+1 < len(molecule) else None

                # Update the count for this element in the dictionary
                counts[element] = int(count)

        # If the current character is an opening bracket, it represents the start of a submolecule
        elif char == '(':

            bracket_counter = 1  # Counter for the number of open and closed brackets
            submolecule = ''     # String to store the submolecule

            # Keep looping until we find the matching closing bracket
            while bracket_counter > 0:
                i += 1
                sub_char = molecule[i]

                # If we find an opening bracket, increment the counter
                if sub_char == '(':
                    bracket_counter += 1

                # If we find a closing bracket, decrement the counter
                elif sub_char == ')':
                    bracket_counter -= 1

                # Add the character to the submolecule string
                submolecule += sub_char

            sub_count = ''   # Count for submolecule

            # Keep looping until we reach a character that is not a number
            next_char = molecule[i + 1] if i + 1 < len(molecule) else None

            while next_char and next_char.isdigit():
                sub_count += next_char
                i += 1
                next_char = molecule[i + 1] if i + 1 < len(molecule) else None
            
            # If no count was specified, default to 1
            if not sub_count:
                sub_count = '1'
            
            # Recursively count the atoms in the submolecule and add them to the counts dictionary
            sub_counts = count_atoms(submolecule)

            for element, count in sub_counts.items():
                counts[element] = counts.get(element, 0) + int(sub_count) * count

        # Move to next character
        i += 1

    return counts


def elemental_ratio(included_species, X, element_1, element_2):
    '''
    Calculate the abundance ratio between any two elements in the atmosphere.

    Example: to compute the C/O ratio, use element_1 = 'C' and element_2 = 'O'.

    Args:
        included_species (np.array of str):
            List of all chemical species included in the model.
        X (4D np.array of float):
            Mixing ratio profiles.
        element_1 (str):
            First element in ratio.
        element_2 (str):
            First element in ratio.

    Returns:
        element_ratio (3D np.array of float):
            Abundance ratio in each layer, sector, and zone.
    
    '''

    # Store shape of mixing ratio array
    N_species, N_layers, N_sectors, N_zones = np.shape(X)

    # Initialise element ratio array for each layer, sector, and zone
    element_ratio = np.zeros(shape=(N_layers, N_sectors, N_zones))

    # Loop through atmosphere
    for i in range(N_layers):
        for j in range(N_sectors):
            for k in range(N_zones):

                element_1_abundance = 0.0   # First element in ratio
                element_2_abundance = 0.0   # Second element in ratio

                for q in range(N_species): 
                    
                    # Extract name and mixing ratio of molecule 'q'
                    molecule_q = included_species[q] 
                    X_q = X[q,i,j,k]

                    # Count how many atoms of each element are in this molecule
                    counts = count_atoms(molecule_q)

                    # Loop over elements
                    for element, count in counts.items():

                        # Add abundances of element 1 and 2 to the total
                        if (element == element_1):
                            element_1_abundance += count * X_q
                        elif (element == element_2):
                            element_2_abundance += count * X_q

                # Compute the element ratio in this layer, sector, and zone
                element_ratio[i,j,k] = element_1_abundance / element_2_abundance

    return element_ratio


def profiles(P, R_p, g_0, PT_profile, X_profile, PT_state, P_ref, R_p_ref, 
             log_X_state, included_species, bulk_species, param_species, 
             active_species, CIA_pairs, ff_pairs, bf_species, N_sectors, 
             N_zones, alpha, beta, phi, theta, species_vert_gradient, 
             He_fraction, T_input, X_input, P_param_set, log_P_slope_phot, 
             log_P_slope_arr, constant_gravity = False, chemistry_grid = None):
    '''
    Main function to calculate the vertical profiles in each atmospheric 
    column. The profiles cover the temperature, number density, mean molecular 
    mass, layer radial extents, and mixing ratio arrays.

    Notes: Most profiles are 3D arrays with format (N_layers, N_sectors, N_zones).
           The mixing ratio profiles are 4D (separate profiles for each species)
           or 5D (CIA or free-free pairs).
           The layer index starts from the base of the atmosphere.
           The sector index starts from the evening terminator.
           The zone index starts from the dayside.

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        R_p (float):
            Observed white light planet radius (m).
        g_0 (float):
            Gravitational field strength at white light radius (m/s^2).
        PT_profile (str):
            Chosen P-T profile parametrisation 
            (Options: isotherm / gradient / two-gradients / Madhu / slope / file_read).
        X_profile (str):
            Chosen mixing ratio profile parametrisation
            (Options: isochem / gradient / two-gradients / file_read).
        PT_state (np.array of float):
            P-T profile state array.
        P_ref (float):
            Reference pressure (bar).
        R_p_ref (float):
            Planet radius corresponding to reference pressure (m).
        log_X_state (2D np.array of float):
            Mixing ratio state array.
        included_species (np.array of str):
            List of chemical species included in the model (including bulk species).
        bulk_species (np.array of str):
            Bulk species dominating atmosphere (e.g. ['H2', 'He']).
        param_species (np.array of str):
            Chemical species with parametrised mixing ratios.
        active_species (np.array of str):
            Spectroscopically active chemical species (see supported_opac.py).
        CIA_pairs (np.array of str):
            Collisionally-induced absorption (CIA) pairs.
        ff_pairs (np.array of str):
            Free-free absorption pairs.
        bf_species (np.array of str):
            Bound-free absorption species.
        N_sectors (int):
            Number of azimuthal sectors comprising the background atmosphere.
        N_zones (int):
            Number of zenith zones comprising the background atmosphere.
        alpha (float):
            Terminator opening angle (degrees).
        beta (float):
            Day-night opening angle (degrees).
        phi (np.array of float):
            Mid-sector angles (radians).
        theta (np.array of float):
            Mid-zone angles (radians).
        species_vert_gradient (np.array of str):
            Chemical species with a vertical mixing ratio gradient.
        He_fraction (float):
            Assumed H2/He ratio (0.17 default corresponds to the solar ratio).
        T_input (np.array of float):
            Temperature profile (only if provided directly by the user).
        X_input (2D np.array of float):
            Mixing ratio profiles (only if provided directly by the user).
        P_param_set (float):
            Only used for the Madhusudhan & Seager (2009) P-T profile.
            Sets the pressure where the reference temperature parameter is 
            defined (P_param_set = 1.0e-6 corresponds to that paper's choice).
        log_P_phot_slope (float):
            Photosphere log pressure for the Piette & Madhusudhan (2020) P-T profile.
        log_P_slope_array (np.array of float):
            Log pressures where the Piette & Madhusudhan (2020) temperature difference 
            parameters are defined (log bar).
        constant_gravity (bool):
            If True, disable inverse square law gravity (only for testing).
        chemistry_grid (dict):
            For models with a pre-computed chemistry grid only, this dictionary
            is produced in chemistry.py.
    
    Returns:
        T (3D np.array of float):
            Temperature profile (K).
        n (3D np.array of float):
            Number density profile (m^-3).
        r (3D np.array of float):
            Radial distant profile (m).
        r_up (3D np.array of float):
            Upper layer boundaries (m).
        r_low (3D np.array of float):
            Lower layer boundaries (m).    
        dr (3D np.array of float):
            Layer thicknesses (m).
        mu (3D np.array of float):
            Mean molecular mass (kg).
        X (4D np.array of float):
            Mixing ratio profile.
        X_active (4D np.array of float):
            Mixing ratios of active species.
        X_CIA (5D np.array of float):
            Mixing ratios of CIA pairs. 
        X_ff (5D np.array of float):
            Mixing ratios of free-free pairs.
        X_bf (4D np.array of float):
            Mixing ratios of bound-free species.
        Bool:
            True if atmosphere physical, otherwise False.
    
    '''

    # For an isothermal profile
    if (PT_profile == 'isotherm'):
        
        # Unpack P-T profile parameters
        T_iso = PT_state[0]
    
        # Initialise temperature array
        T_rough = T_iso * np.ones(shape=(len(P), 1, 1)) # 1D profile => N_sectors = N_zones = 1
        
        # Gaussian smooth P-T profile
        T = T_rough   # No need to Gaussian smooth an isothermal profile

    # For the gradient profiles (1D, 2D, or 3D)
    elif (PT_profile == 'gradient'):
        
        # Unpack P-T profile parameters
        T_bar_term, Delta_T_term, Delta_T_DN, T_deep = PT_state
        
        # Reject if T_Morning > T_Evening or T_Night > T_day
        if ((Delta_T_term < 0.0) or (Delta_T_DN < 0.0)): 
            
            # Quit computations if model rejected
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        
        # If P-T parameters valid
        else:
            
            # Compute unsmoothed temperature field
            T_rough = compute_T_field_gradient(P, T_bar_term, Delta_T_term, 
                                               Delta_T_DN, T_deep, N_sectors, 
                                               N_zones, alpha, beta, phi, theta)

        # Gaussian smooth P-T profile
        T = gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')

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
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        
        # If P-T parameters valid
        else:
            
            # Compute unsmoothed temperature field
            T_rough = compute_T_field_two_gradients(P, T_bar_term_high, T_bar_term_mid,
                                                    Delta_T_term_high, Delta_T_term_mid, 
                                                    Delta_T_DN_high, Delta_T_DN_mid,
                                                    log_P_mid, T_deep, N_sectors, 
                                                    N_zones, alpha, beta, phi, theta)

        # Gaussian smooth P-T profile
        T = gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')
        
    # For the Madhusudhan & Seager (2009) profile (1D only)
    elif (PT_profile == 'Madhu'):
        
        # Unpack P-T profile parameters
        a1, a2, log_P1, log_P2, log_P3, T_set = PT_state
        
        # Profile requires P3 > P2 and P3 > P1, reject otherwise
        if ((log_P3 < log_P2) or (log_P3 < log_P1)):
            
            # Quit computations if model rejected
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
        
        # If P-T parameters valid
        else:
            
            # Compute unsmoothed temperature profile
            T_rough = compute_T_Madhu(P, a1, a2, log_P1, log_P2, log_P3, 
                                      T_set, P_param_set)

        # Gaussian smooth P-T profile
        T = gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')

    # For the Piette & Madhusudhan (2020) profile (1D only)
    elif (PT_profile == 'slope'):
        
        # Unpack P-T profile parameters
        T_phot = PT_state[0]
        Delta_T_arr = np.array(PT_state[1:])
            
        # Compute unsmoothed temperature profile
        T_rough = compute_T_slope(P, T_phot, Delta_T_arr, log_P_slope_phot,
                                  log_P_slope_arr)

        # Find how many layers corresponds to 0.3 dex smoothing width
        smooth_width = round(0.3/(((np.log10(P[0]) - np.log10(P[-1]))/len(P))))

        # Gaussian smooth P-T profile
        T = gauss_conv(T_rough, sigma=smooth_width, axis=0, mode='nearest')

    # Read user provided P-T profile
    elif (PT_profile == 'file_read'):

        # Initialise temperature array
        T_rough = T_input.reshape((len(P), 1, 1))   # 1D profile => N_sectors = N_zones = 1
        
        # Gaussian smooth P-T profile
        T = T_rough   # No need to Gaussian smooth a user profile

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
            
        elif (X_profile == 'lever'):
            X_param = compute_X_lever(P, log_X_state, species_has_profile, N_sectors, N_zones)

        # Read in equilibrium mixing ratio profiles 
        elif (X_profile == 'chem_eq'):

            if (chemistry_grid == None):
                raise Exception("Error: no chemistry grid loaded for an equilibrium model")

            # Unpack C/O and Metallicity 
            C_to_O = log_X_state[0]
            log_Met = log_X_state[1]

            log_X_input = interpolate_log_X_grid(chemistry_grid, np.log10(P), T, C_to_O, log_Met, 
                                                     param_species, return_dict = False)
            X_input = 10**log_X_input
            X_param = X_input

            '''
            if PT_profile == 'isotherm':

                log_X_input = interpolate_log_X_grid(chemistry_grid, np.log10(P), T, C_to_O, log_Met, 
                                                     param_species, return_dict = False)
                X_input = 10**log_X_input
                X_param = X_input

            elif PT_profile == 'gradient':

                log_X_input = interpolate_log_X_grid(chemistry_grid, np.log10(P), T, C_to_O, log_Met, 
                                                     param_species, return_dict = False)
                X_input = 10**log_X_input
                X_param = X_input
                
            else:
                raise Exception('Chemical Equilibrium only supports 1D Isothermal PT or Gradient PT (for now)')
            '''

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
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False

    # Create mixing ratio category arrays for extinction calculations
    X_active, X_CIA, \
    X_ff, X_bf = mixing_ratio_categories(P, X, N_sectors, N_zones, 
                                         included_species, active_species, 
                                         CIA_pairs, ff_pairs, bf_species)
        
    # Store masses in an array to speed up mu calculation
    masses_all = np.zeros(N_species)
    
    # Load masses of each species from dictionary in species_data.py
    for q in range(N_species):
        species = included_species[q]
        masses_all[q] = masses[species]
    
    # Calculate mean molecular mass
    mu = compute_mean_mol_mass(P, X, N_species, N_sectors, N_zones, masses_all)
        
    # Calculate number density and radial profiles
    if (constant_gravity == True):

        n, r, r_up, r_low, dr = radial_profiles_constant_g(P, T, g_0, P_ref, 
                                                           R_p_ref, mu, 
                                                           N_sectors, N_zones)
    else:

        n, r, r_up, r_low, dr = radial_profiles(P, T, g_0, R_p, P_ref, 
                                                R_p_ref, mu, N_sectors, N_zones)     

    
    return T, n, r, r_up, r_low, dr, mu, X, X_active, X_CIA, \
           X_ff, X_bf, True
