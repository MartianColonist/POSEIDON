# ***** Handles P-T and mixing ratio profiles *****

import numpy as np
import scipy.constants as sc
from scipy.ndimage import gaussian_filter1d as gauss_conv
from numba.core.decorators import jit

from .supported_opac import inactive_species
from .species_data import masses


@jit(nopython = True)
def compute_T_Madhu(P, a1, a2, log_P1, log_P2, log_P3, T_deep, P_set = 10.0):
    
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


@jit(nopython = True)
def compute_T_field_gradient(P, T_bar_term, Delta_T_term, Delta_T_DN, T_deep,
                             N_sectors, N_zones, alpha, beta, phi, theta,
                             P_deep, P_high):
    
    ''' Creates 3D temperature profile array storing T(P, phi, theta).
    
        For each atmospheric column, the temperature profile has a constant
        vertical gradient across the observable atmosphere. For pressures
        above and below the considered range (by default, 10^-5 -> 10 bar),
        are treated with an isotherm.
           
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
            
        # Compute dayside and nighside temperatures for given angle phi
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
def compute_X_field_gradient(P, log_X_state, N_sectors, N_zones, param_species, 
                             species_has_profile, alpha, beta, phi, theta, 
                             P_deep = 10.0, P_high = 1.0e-5):
    
    ''' Creates 4D abundance profile array storing X(species, P, phi, theta).
    
        For each atmospheric column, the abundance profile has either a 
        constant vertical gradient across the observable atmosphere or a 
        uniform (isochem) profile. The species with a gradient profile are
        specified by the user. For gradient profiles, pressures
        above and below the considered range (by default, 10^-5 -> 10 bar),
        are treated with an isochem.
           
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
    
    # Loop over parametrised chemical speciespasses
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
                
            # Compute dayside and nighside abundances for given angle phi
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

        if (len(bulk_species > 1)):
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
        absorption (CIA), free-free opacity, and bound-free opaciy.
        
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
def refractive_index_profile(P, X, n, dr, wl, eta_stored, N_species, 
                             N_sectors, N_zones):
    
    ''' NOT CURRENTLY USED (needs to be made at least 10x faster)
    
        Compute the wavelength-dependant refractive index of each layer in
        each region of the model atmosphere. 
        
        Note: 'eta' is used for refractive index, as 'n' is reserved for
               number density.
              
        The overall refractivity (eta-1) is given by nu_tot =  sum(X_q * nu_q)
        where nu_q is the refractivity of species q, scaled to the number 
        density in the given layer by nu_q = nu_q_ref * (n/n_ref).
        The wavelength-dependant refractive indices at standard conditions
        (eta_q_ref) are pre-computed in absorption.py.
        
        This function only outputs wavelength dependance over scales where
        the refractive index at reference conditions varies by > 1%.
       
        Inputs:
        
        X => atmosphere volume mixing ratios
        n => atmosphere number density grid
        dr => thickness of atmosphere layers
        wl => model wavelength grid (um)
        eta_stored => refractive indicies on model wl grid at standard conditions
        N_species => number of chemical species included in model
        N_sectors =>
       
        Outputs:
           
        eta_out => overall (weighted) refractive index at local layer and region
                   conditions for wavelengths where eta varies significantly
        wl_eta => wavelengths where eta_out is calculated
        dlneta_dr => derivative of natural log of refractive index w.r.t height
                     for each wavelength in wl_eta (used for refractive ray tracing)
       
    '''
    
    # Store number of layers for convenience
    N_layers = len(P)

    refractive_tol = 1.0e-2   # 1% tolerance for wavelength dependance of refractive index
    
    N_wl = len(wl)  # Number of wavelengths on model grid
    
    # Specify reference number density (m^-3) at standard conditions (1 atm and 0 C)
    n_ref = (101325.0/(sc.k * 273.15))   # Ideal gas law, P in Pa, T in K
    
    # Initialise refractive index reference array
    eta_ref = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))   # Refractive index at output wavelengths and reference density
    
    # Establish how many refractive index values are needed (due to wl variation)
    for i in range(N_layers):
            
        for j in range(N_sectors):
        
            for k in range(N_zones):
        
                # For first wavelength on grid, store refractive index for this wavelength
                eta_ref[i,j,k,0] = (1.0 + np.sum(X[:,i,j,k]*(eta_stored[:,0] - 1.0)))
                eta_tot_last = eta_ref[i,j,k,0]
                
                # For other wavelengths, only store refractive index if it changes by more than 1%
                for l in range(N_wl):
                    
                    nu_tot = 0.0
                    
                    # For each molecule / atom included in model
                    for q in range(N_species):
                        
                        # Compute overall refractivity at this wavelength (at reference number density) 
                        nu_tot += X[q,i,j,k]*(eta_stored[q,l] - 1.0)  
                        
                    eta_tot_new = (1.0 + nu_tot)   # Overall refractive index at this wavelength
                        
                    # Compute fractional error between refractive index at new wavelength and last stored
                    eta_err = np.abs((eta_tot_new - eta_tot_last)/eta_tot_last)
                        
                    # If error exceeds tolerance, then create new entry in wl-dependant refractive index array
                    if (eta_err >= refractive_tol):
                        eta_ref[i,j,k,l] = eta_tot_new
                        eta_tot_last = eta_tot_new
            
    #***** Now that the refractive index array has been reduced to consider only important *****#
    #      wavelength variations, can proceed to work out refractive index in each layer        #
    
    # Find wavelength indices (across all regions) where wavelength variations exceed threshold
    l_eta_change = np.unique(np.where(eta_ref != 0.0)[0])
    
    wl_eta = wl[l_eta_change]    # Wavelengths to output refractive index to
    N_wl_eta = len(wl_eta)       # Number of wavelengths to output refractive index
    
    # Initialise output refractive index and derivative arrays
    eta_out = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl_eta))
    dlneta_dr = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl_eta))
    
    # Compute refractive index profiles for each sector and zone
    for j in range(N_sectors):
        
        for k in range(N_zones):
        
            # For each wavelength where eta changes by >1%
            for l in range(N_wl_eta):
                
                # For each atmospheric layer
                for i in range(N_layers): 
                
                    # If no computed value in last loop (due to different wavelength dependance between terminator regions)
                    if (eta_ref[i,j,k,l_eta_change[l]] == 0):
                        
                        # Set value in this region equal to value at last wavelength (as refractive_tol not exceeded)
                        eta_ref[i,j,k,l_eta_change[l]] = eta_ref[i,j,k,l_eta_change[l-1]]
                
                    nu_ref = (eta_ref[i,j,k,l_eta_change[l]] - 1.0)   # Refractivity at reference number density
                
                    # Scale refractivity to number density in layer
                    eta_out[i,j,k,l] = (1.0 + (n[i,j,k]/n_ref)*nu_ref)   
             
                # For each atmospheric layer (except bottom and top)
                for i in range(1, N_layers-1):
            
                    # Store derivative of log refractive index w.r.t height
                    dlneta_dr[i,j,k,l] = (0.5 * np.log(eta_out[(i+1),j,k,l]/eta_out[(i-1),j,k,l]))/dr[i,j,k]
                
                # Edge cases for bottom and top layers
                dlneta_dr[0,j,k,l] = np.log(eta_out[1,j,k,l]/eta_out[0,j,k,l])/dr[0,j,k]
                dlneta_dr[(N_layers-1),j,k,l] = np.log(eta_out[(N_layers-1),j,k,l]/eta_out[(N_layers-2),j,k,l])/dr[(N_layers-1),j,k]
                
    return eta_out, wl_eta, dlneta_dr  


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
        X_species = X[included_species == species,:,:,:]
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

def profiles(planet, P, PT_profile, X_profile, PT_state, P_ref, R_p_ref, log_X_state, 
             included_species, bulk_species, param_species, active_species, 
             cia_pairs, ff_pairs, bf_species, N_sectors, N_zones, 
             alpha, beta, phi, theta, species_vert_gradient, He_fraction,
             P_deep, P_high):
    
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

    # Unpack planet properties
    g_0 = planet['planet_gravity']
    R_p = planet['planet_radius']
            
    # For isothermal or gradient profiles (1D, 2D, or 3D)
    if (PT_profile in ['isotherm', 'gradient']):
        
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
                                               N_zones, alpha, beta, phi, theta,
                                               P_deep, P_high)
        
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

    # Gaussian smooth P-T profiles to avoid gradient discontinuities
    T_smooth = gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')
    
    # Find min and max profile temperatures
  #  T_min = np.min(T_smooth)
  #  T_max = np.max(T_smooth)
        
    # Check if minimum or maximum temperatures are outside opacity range
  #  if ((T_max > T_fine_max) or (T_min < T_fine_min)): 
        
        # Quit computations if model rejected
  #      return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False
    
    # Load number of distinct chemical species in model atmosphere
    N_species = len(bulk_species) + len(param_species)
    
    # Find which parametrised chemical species have a gradient profile
    species_has_profile = np.zeros(len(param_species)).astype(np.int64)
    
    if (X_profile == 'gradient'):
        species_has_profile[np.isin(param_species, species_vert_gradient)] = 1  
    
    # Compute 4D mixing ratio array
    X_param = compute_X_field_gradient(P, log_X_state, N_sectors, N_zones, 
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
        
    # Calculate refractive index and its derivative w.r.t height
 #   eta, wl_eta, dlneta_dr = refractive_index_profile(P, X, n, dr, wl, eta_stored, 
 #                                                     N_species, N_sectors, N_zones)
    
    return P, T_smooth, n, r, r_up, r_low, dr, X, X_active, X_cia, \
           X_ff, X_bf, mu, True
        
