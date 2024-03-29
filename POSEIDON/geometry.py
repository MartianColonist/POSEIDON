''' 
Functions to compute geometric aspects of the model atmosphere.

'''

import numpy as np
import scipy.constants as sc
from scipy.special import erfinv
from numba.core.decorators import jit


def atmosphere_regions(Atmosphere_dimension, TwoD_type, N_slice_EM, N_slice_DN):
    '''
    Establish the number of azimuthal sectors (Evening-Morning) and zenith 
    zones (Day-Night) required to discretise the background atmosphere.

    Args:
        Atmosphere_dimension (int):
            The dimensionality of the model atmosphere
            (Options: 1 / 2 / 3).
        TwoD_type (str):
            For 2D models, specifies whether the model considers day-night
            gradients or evening-morning gradients
            (Options: D-N / E-M).
        N_slice_EM (even int):
            Number of azimuthal slices in the evening-morning transition region.
        N_slice_DN (even int):
            Number of zenith slices in the day-night transition region.
    
    Returns:
        N_sectors (int):
            Number of azimuthal sectors comprising the background atmosphere.
        N_zones (int):
            Number of zenith zones comprising the background atmosphere.
    
    '''
    
    # A 1D model has a single sector and zone spanning the entire atmosphere
    if (Atmosphere_dimension == 1):
        
        N_sectors = 1
        N_zones = 1
     
    # For 2D models, need to treat asymmetric Evening-Morning and Day-Night separately
    elif (Atmosphere_dimension == 2):
        
        # 2D model with Evening-Morning differences
        if (TwoD_type == 'E-M'):

            N_zones = 1

            if (N_slice_EM <= 0 or N_slice_EM % 2 != 0):
                raise Exception("Error: N_slice_EW must be an even integer.")
            else:
                N_sectors = 2 + N_slice_EM

        # 2D model with Day-Night differences
        elif (TwoD_type == 'D-N'):

            N_sectors = 1
            
            if (N_slice_DN < 0 or N_slice_DN % 2 != 0):
                raise Exception("Error: N_slice_DN must be an even integer.")
            else:
                N_zones = 2 + N_slice_DN
 
        else:
            raise Exception("Error: '" + TwoD_type + "' is not a valid 2D model type.")
            
    # General 3D case with Evening-Morning and Day-Night differences
    elif (Atmosphere_dimension == 3):

        if (N_slice_EM <= 0 or N_slice_EM % 2 != 0 or 
            N_slice_DN < 0 or N_slice_DN % 2 != 0):
                raise Exception("Error: N_slice_EW and N_slice_DN must be even integers.")
        else:
            N_sectors = 2 + N_slice_EM
            N_zones = 2 + N_slice_DN

    # Because I can ;)
    elif (Atmosphere_dimension == 4):
        raise Exception("Error: Planets can't be tesseracts!")
        
    else:
        raise Exception("Error: Invalid dimensionality for model atmosphere.")
    
    return N_sectors, N_zones


@jit(nopython=True)
def angular_grids(Atmosphere_dimension, TwoD_type, N_slice_EM, N_slice_DN, 
                  alpha, beta, sharp_DN_transition):
    '''
    Compute the grids of angles (sector / zone centres, edges, and differential 
    extents) for a given discretised atmosphere. 

    By convention, the angles are defined as:

    phi: angle in terminator plane measured clockwise from North pole.
    theta: angle from terminator plane measured towards nightside.
    
    Due to (assumed) North-South symmetry in all cases, we only need
    to consider theta > -pi/2 and/or phi < pi/2 (i.e. northern hemisphere).

    Args:
        Atmosphere_dimension (int):
            The dimensionality of the model atmosphere
            (Options: 1 / 2 / 3).
        TwoD_type (str):
            For 2D models, specifies whether the model considers day-night
            gradients or evening-morning gradients
            (Options: D-N / E-M).
        N_slice_EM (even int):
            Number of azimuthal slices in the evening-morning transition region.
        N_slice_DN (even int):
            Number of zenith slices in the day-night transition region.
        alpha (float):
            Terminator opening angle (degrees).
        beta (float):
            Day-night opening angle (degrees).
        sharp_DN_transition (bool):
            For 2D / 3D models, sets day-night transition width (beta) to 0.
    
    Returns:
        phi (np.array of float):
            Mid-sector angles (radians).
        theta (np.array of float):
            Mid-zone angles (radians).
        phi_edge (np.array of float):
            Boundary angles for each sector (radians).
        theta_edge (np.array of float):
            Boundary angles for each zone (radians).
        dphi (np.array of float):
            Angular width of each sector (radians).
        dtheta (np.array of float):
            Angular width of each zone (radians).
    
    '''

    # Convert alpha and beta from degrees to radians
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    
    # 1D atmosphere
    if (Atmosphere_dimension == 1):
        
        # Place boundaries in Evening-Morning / Day-Night equatorial plane
        phi_edge = np.array([-np.pi/2.0, np.pi/2.0])
        theta_edge = np.array([-np.pi/2.0, np.pi/2.0]) 
        
    # 2D atmosphere
    elif (Atmosphere_dimension == 2):
        
        if (TwoD_type == 'E-M'):
            
            # Day-Night equatorial plane defines edge of northern hemisphere
            theta_edge = np.array([-np.pi/2.0, np.pi/2.0])   
            
            
            phi_edge = np.array([-np.pi/2.0])
    
            # Add sector edges along terminator transition
            dphi_term = alpha_rad / N_slice_EM     # Angular width of each sector
            phi_edge = np.append(phi_edge, ((-1.0/2.0) * alpha_rad + 
                                                np.arange(N_slice_EM + 1) * dphi_term))
            
            # End with Morning equatorial plane
            phi_edge = np.append(phi_edge, np.array([np.pi/2.0])) 

        elif (TwoD_type == 'D-N'):
            
            # Evening-Morning equatorial plane defines edge of northern hemisphere
            phi_edge = np.array([-np.pi/2.0, np.pi/2.0])  
            
            # For sharp transitions, edges are in day-night equatorial plane and the terminator
            if (sharp_DN_transition == True):
                theta_edge = np.array([-np.pi/2.0, 0.0, np.pi/2.0])

            else:
    
                # Start from Day equatorial plane
                theta_edge = np.array([-np.pi/2.0])

                # Add zone edges along terminator transition
                dtheta_term = beta_rad / N_slice_DN     # Angular width of each zone
            
                # Add edges from each zone in the transition region
                theta_edge = np.append(theta_edge, ((-1.0/2.0) * beta_rad + 
                                                    np.arange(N_slice_DN + 1) * dtheta_term))
                
                # End with Night equatorial plane
                theta_edge = np.append(theta_edge, np.array([np.pi/2.0]))
                
    # 3D atmosphere
    elif (Atmosphere_dimension == 3):

        #***** Handle day-night edges first *****#

        # For sharp transitions, edges are in day-night equatorial plane and the terminator
        if (sharp_DN_transition == True):
            theta_edge = np.array([-np.pi/2.0, 0.0, np.pi/2.0])

        else:

            # Start from Day equatorial plane
            theta_edge = np.array([-np.pi/2.0])

            # Add zone edges along terminator transition
            dtheta_term = beta_rad / N_slice_DN     # Angular width of each zone
        
            # Add edges from each zone in the transition region
            theta_edge = np.append(theta_edge, ((-1.0/2.0) * beta_rad + 
                                                np.arange(N_slice_DN + 1) * dtheta_term))
            
            # End with Night equatorial plane
            theta_edge = np.append(theta_edge, np.array([np.pi/2.0]))

        #***** Evening-Morning edges next *****#
        
        # Start from West equatorial plane
        phi_edge = np.array([-np.pi/2.0])

        # Add sector and zone edges along terminator transition
        dphi_term = alpha_rad / N_slice_EM     # Angular width of each sector
        phi_edge = np.append(phi_edge, ((-1.0/2.0) * alpha_rad + 
                                            np.arange(N_slice_EM + 1) * dphi_term))

        # End with Morning / Night equatorial plane
        phi_edge = np.append(phi_edge, np.array([np.pi/2.0]))

            
    # Compute angular width of each sector and zone
    dphi = np.diff(phi_edge)
    dtheta = np.diff(theta_edge)
                
    # Compute array of angles defining center of each sector and zone
    phi = -np.pi/2.0 + np.cumsum(dphi) - (dphi/2.0)
    theta = -np.pi/2.0 + np.cumsum(dtheta) - (dtheta/2.0)    
    
    return phi, theta, phi_edge, theta_edge, dphi, dtheta
    

def opening_angle(R_p_ref, T, mu, g, log_sigma = -25.3, log_X = -3.3, 
                  tau_0 = 0.56, P_ref = 0.01, gamma = 0.01):
    '''
    Calculate the predicted terminator opening angle using the analytic equation
    from Wardenier+2022 (their Eq. 10).

    This analytic prescription is valid for a 1D isothermal atmosphere with
    constant gravity with opacity dominated by a single chemical species with
    constant cross section. Despite these caveats, the formula performs rather
    well compared to full radiative transfer models. Therefore, this offers
    a good first-order estimate for the angular width about the terminator
    plane for which transmission spectrum are sensitive.  

    Args:
        R_p_ref (float):
            Planetary radius at the reference pressure P_ref (typically the 
            white light transit radius).
        T (float):
            Representative planet temperature (typically T_eq) (K).
        mu (float):
            Mean molecular mass of the atmosphere (kg).
        g (float):
            Gravitational field strength of planet (m/s^2).
        log_sigma (float):
            Absorption cross section of dominant opacity source (log_10 (m^2)).
        log_X (float):
            Mixing ratio of chemical species dominating the opacity.
        tau_0 (float):
            Slant optical depth for impact parameter corresponding to the opening
            angle (0.56 default corresponds to Lecavelier des Etangs+2008).
        P_ref (float):
            Reference pressure corresponding to R_p_ref.
        gamma (float):
            'Tolerance factor' for the opening angle, such that the angle 
            encloses a fraction gamma -> (1 - gamma) of the total slant optical
            depth. The default corresponds to the 1-99 % optical depth region.
            Note: relabelled from 'beta' in Wardenier+2022, since POSEIDON uses
            beta for the opening angle itself.

    Returns:
        beta (float):
            Day-night opening angle (degrees).

    '''

    # Compute number density at reference pressure 
    n_0 = P_ref / (sc.k * T)    # Ideal gas law

    # Compute constant pressure scale height
    H = (sc.k * T) / (mu * g)

    # Convert cross section and mixing ratio to linear units
    sigma = np.power(10.0, log_sigma)   # Cross section (m^2)
    X = np.power(10.0, log_X)           # Mixing ratio

    # Store constant encoding assumed chemical composition
    chem_const = ((sigma * X) / tau_0)    # Default value: 10^(-28.3) m^2

    # Calculate terminator opening angle
    beta = 2.0 * np.arctan((np.sqrt(2 * R_p_ref * H) * erfinv (1.0 - gamma)) / 
                           (R_p_ref + H * np.log(np.sqrt(2.0 * np.pi * R_p_ref * H) * chem_const * n_0)))

    # Convert from radians to degrees
    beta *= (180.0 / np.pi)

    return beta

