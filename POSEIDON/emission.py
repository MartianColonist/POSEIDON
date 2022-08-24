''' 
Radiative transfer calculations for generating emission spectra.

'''

import numpy as np
from numba.core.decorators import jit
import scipy.constants as sc


@jit(nopython = True)
def planck_lambda_arr(T, wl):
    '''
    Compute the Planck function spectral radiance for a range of model
    wavelengths and atmospheric temperatures.

    Args:
        T (np.array of float):
            Array of temperatures in each atmospheric layer (K).
        wl (np.array of float): 
            Wavelength grid (μm).
    
    Returns:
        B_lambda (2D np.array of float):
            Planck function spectral radiance as a function of layer temperature
            and wavelength in SI units (W/m^2/sr/m).

    '''
    
    # Define Planck function array
    B_lambda = np.zeros(shape=(len(T),len(wl)))  # (Temperature, wavelength)

    # Convert wavelength array to m
    wl_m = wl * 1.0e-6
    
    # Second radiative constant
    c_2 = (sc.h * sc.c) / sc.k
    
    # For each wavelength
    for k in range(len(wl)):
        
        # Precompute Planck function coefficient prefactor
        coeff = (2.0 * sc.h * sc.c**2) / (wl_m[k]**5)
        
        # For each atmospheric layer
        for i in range(len(T)):
            
            # Evaluate Planck function spectral radiance
            B_lambda[i,k] = coeff * (1.0 / (np.exp(c_2 / (wl_m[k] * T[i])) - 1.0))
            
    return B_lambda


@jit(nopython = True)
def emission_rad_transfer(T, dz, wl, kappa, Gauss_quad = 2):
    '''
    Compute the emergent top-of-atmosphere flux from a planet or brown dwarf.

    This function  considers only pure thermal emission (i.e. no scattering).

    Args:
        T (np.array of float):
            Temperatures in each atmospheric layer (K).
        dz (np.array of float):
            Vertical extent of each atmospheric layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        kappa (2D np.array of float):
            Extinction coefficient in each layer as a function of wavelength (m^-1).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            (Options: 2 / 3).
    
    Returns:
        F (np.array of float):
            Spectral surface flux in SI units (W/m^2/sr/m).

    '''
    
    # Load weights and cos(theta) values for desired Gaussian quadrature scheme
    if (Gauss_quad == 2):
        W = np.array([0.5, 0.5])
        mu = np.array([0.5 - 0.5*np.sqrt(1.0/3.0), 0.5 + 0.5*np.sqrt(1.0/3.0)])
    elif (Gauss_quad == 3):
        W = np.array([5.0/18.0, 4.0/9.0, 5.0/18.0])
        mu = np.array([0.5 - 0.5*np.sqrt(3.0/5.0), 0.5, 0.5 + 0.5*np.sqrt(3.0/5.0)])
    
    # Calculate Planck function in each layer and each wavelength
    B = planck_lambda_arr(T, wl)
    
    # Initial intensity at the base of the atmosphere is a Planck function 
    I = np.ones(shape=(len(mu),len(wl))) * B[0,:]
    
    # Initialise surface flux array
    F = np.zeros(len(wl))
    
    # For each wavelength
    for k in range(len(wl)):
    
        # For each ray travelling at mu = cos(theta)
        for j in range(len(mu)):

            # For each atmospheric layer
            for i in range(len(T)):
    
                # Compute vertical optical depth across the layer
                tau_vert = kappa[i,k] * dz[i]
                
                # Compute transmissivity of the layer
                Trans = np.exp((-1.0 * tau_vert)/mu[j])
                
                # Solve for emergent intensity from the layer top
                I[j,k] = Trans * I[j,k] + (1.0 - Trans) * B[i,k]
                
            # Add contribution of this ray/angle to the surface flux
            F[k] += 2.0 * np.pi * mu[j] * I[j,k] * W[j]
    
    return F

