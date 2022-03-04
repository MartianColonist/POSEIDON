# Radiative transfer calculations for generating emission spectra

import numpy as np
from numba.core.decorators import jit
import scipy.constants as sc


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
def emission_rad_transfer(T, dr, wl, kappa_clear, order = 1, Gauss_quad = 3):
    '''
    ADD DOCSTRING
    '''
    
    if (Gauss_quad == 2):
        
        W = np.array([0.5, 0.5])
        mu = np.array([0.5 - 0.5*np.sqrt(1.0/3.0), 0.5 + 0.5*np.sqrt(1.0/3.0)])
        
    elif (Gauss_quad == 3):
        
        W = np.array([5.0/18.0, 4.0/9.0, 5.0/18.0])
        mu = np.array([0.5 - 0.5*np.sqrt(3.0/5.0), 0.5, 0.5 + 0.5*np.sqrt(3.0/5.0)])
    
    # The total extinction coefficient (without scattering) comes only from chemical absorption    
    kappa_tot = kappa_clear[:,0,0,:]  # 0 index to only consider one region for 1D models
    
    # Remove region dependance from layer thickness array
    dz = dr[:,0,0]   # 0 index to only consider one region for 1D models
    T = T[:,0,0]     # 0 index to only consider one region for 1D models
    
    # Calculate Planck function in each layer and each wavelength
    B = planck_lambda(T, wl)
    
    # Initialise intensity at base of atmosphere (for each wavelength and mu) as Planck function 
    I = np.ones(shape=(len(mu),len(wl))) * B[0,:]
    
    # Initialise surface flux array
    F = np.zeros(len(wl))

    # First order integration scheme
    if (order == 1):
    
        # For each wavelength
        for k in range(len(wl)):
        
            # For each ray travelling at mu = cos(theta)
            for j in range(len(mu)):

                # For each atmospheric layer
                for i in range(len(T)):
        
                    # Compute vertical optical depth in the layer
                    tau_vert = kappa_tot[i,k] * dz[i]
                    
                    # Compute transmissivity of the layer
                    Trans = np.exp((-1.0 * tau_vert)/mu[j])
                    
                    # Solve for emergent intensity from the layer top
                    I[j,k] = Trans * I[j,k] + (1.0 - Trans) * B[i,k]
                    
                # Add contribution of this angle to surface flux
                F[k] += 2.0 * np.pi * mu[j] * I[j,k] * W[j]
    
    return F

