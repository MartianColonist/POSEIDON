''' 
Radiative transfer calculations for generating emission spectra.

'''

import numpy as np
import scipy.constants as sc
from numba import jit, cuda
import math
import os

from .utility import mock_missing, interp_GPU

try:
    import cupy as cp
except ImportError:
    cp = mock_missing('cupy')

block = int(os.environ['block'])
thread = int(os.environ['thread'])


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


@cuda.jit
def planck_lambda_arr_GPU(T, wl, B_lambda):
    '''
    GPU variant of the 'planck_lambda_arr' function.

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
    
    thread = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    # Second radiative constant
    c_2 = (sc.h * sc.c) / sc.k
    
    # For each wavelength
    for k in range(thread, len(wl), stride):
        
        # Precompute Planck function coefficient prefactor
        coeff = (2.0 * sc.h * sc.c**2) / (wl[k]**5)
        
        # For each atmospheric layer
        for i in range(len(T)):
            
            # Evaluate Planck function spectral radiance
            B_lambda[i,k] = coeff * (1.0 / (math.exp(c_2 / (wl[k] * T[i])) - 1.0))


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

    # Initialise differential optical depth array
    dtau = np.zeros(shape=(len(T), len(wl)))
    
    # For each wavelength
    for k in range(len(wl)):
    
        # For each ray travelling at mu = cos(theta)
        for j in range(len(mu)):

            # For each atmospheric layer
            for i in range(len(T)):
    
                # Compute vertical optical depth across the layer
                dtau_vert = kappa[i,k] * dz[i]
                dtau[i,k] = dtau_vert
                
                # Compute transmissivity of the layer
                Trans = np.exp((-1.0 * dtau_vert)/mu[j])
                
                # Solve for emergent intensity from the layer top
                I[j,k] = Trans * I[j,k] + (1.0 - Trans) * B[i,k]
                
            # Add contribution of this ray/angle to the surface flux
            F[k] += 2.0 * np.pi * mu[j] * I[j,k] * W[j]
    
    return F, dtau


def emission_rad_transfer_GPU(T, dz, wl, kappa, Gauss_quad = 2):
    '''
    GPU variant of the 'emission_rad_transfer' function.

    Compute the emergent top-of-atmosphere flux from a planet or brown dwarf.

    This function considers only pure thermal emission (i.e. no scattering).

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
        W = cp.array([0.5, 0.5])
        mu = cp.array([0.5 - 0.5*cp.sqrt(1.0/3.0), 0.5 + 0.5*cp.sqrt(1.0/3.0)])
    elif (Gauss_quad == 3):
        W = cp.array([5.0/18.0, 4.0/9.0, 5.0/18.0])
        mu = cp.array([0.5 - 0.5*cp.sqrt(3.0/5.0), 0.5, 0.5 + 0.5*cp.sqrt(3.0/5.0)])
    
    # Define temperature, wavelength, and Planck function arrays on GPU
    T = cp.asarray(T)
    wl_m = cp.asarray(wl) * 1.0e-6
    B = cp.zeros((len(T), len(wl_m)))

    # Calculate Planck function in each layer and each wavelength    
    planck_lambda_arr_GPU[block, thread](T, wl_m, B)
    
    # Initial intensity at the base of the atmosphere is a Planck function 
    I = cp.ones(shape=(len(mu),len(wl))) * B[0,:]
    F_p = cp.zeros(len(wl))
    
    # Initialise differential optical depth array
    dtau = cp.zeros(shape=(len(T), len(wl)))
    
    @cuda.jit
    def helper_kernel(T, dz, wl, kappa, W, mu, B, I, F, dtau):
        thread = cuda.grid(1)
        stride = cuda.gridsize(1)

        # For each wavelength
        for k in range(thread, len(wl), stride):

            # For each ray travelling at mu = cos(theta)
            for j in range(len(mu)):

                # For each atmospheric layer
                for i in range(len(T)):
        
                    # Compute vertical optical depth across the layer
                    dtau_vert = kappa[i,k] * dz[i]
                    dtau[i,k] = dtau_vert
                    
                    # Compute transmissivity of the layer
                    Trans = math.exp((-1.0 * dtau_vert)/mu[j])
                    
                    # Solve for emergent intensity from the layer top
                    I[j,k] = Trans * I[j,k] + (1.0 - Trans) * B[i,k]
                    
                # Add contribution of this ray/angle to the surface flux
                cuda.atomic.add(F, k, 2.0 * math.pi * mu[j] * I[j,k] * W[j])

    helper_kernel[block, thread](T, cp.asarray(dz), cp.asarray(wl), kappa, W, mu, B, I, F_p, dtau)

    return cp.asnumpy(F_p), dtau


@jit(nopython = True)
def determine_photosphere_radii(dtau, r_low, wl, photosphere_tau = 2/3):
    '''
    Interpolate optical depth to find the radius corresponding to the
    photosphere (by default at tau = 2/3).

    Args:
        dtau (2D np.array of float):
            Vertical optical depth across each layer (starting from the top of
            the atmosphere) as a function of layer and wavelength.
        r_low (np.array of float):
            Radius at the lower boundary of each layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        photosphere_tau (float):
            Optical depth to determine photosphere radius.
    
    Returns:
        R_p_eff (np.array of float):
            Photosphere radius as a function of wavelength (m).

    '''

    # Initialise photosphere radius array
    R_p_eff = np.zeros(len(wl))

    # Calculate photosphere radius at tau = 2/3 for each wavelength
    for k in range(len(wl)):

        # Find cumulative optical depth from top of atmosphere down at each wavelength
        tau_lambda = np.cumsum(dtau[:,k])

        # Interpolate layer boundary radii to find radius where tau = 2/3 (lower because we are integrating down) 
        R_p_eff[k] = np.interp(photosphere_tau, tau_lambda, r_low)

    return R_p_eff


@cuda.jit
def determine_photosphere_radii_GPU(tau_lambda, r_low, wl, R_p_eff, photosphere_tau = 2/3):
    '''
    GPU variant of the 'determine_photosphere_radii' function.

    Interpolate optical depth to find the radius corresponding to the
    photosphere (by default at tau = 2/3).

    Args:
        dtau (2D np.array of float):
            Vertical optical depth across each layer (starting from the top of
            the atmosphere) as a function of layer and wavelength.
        r_low (np.array of float):
            Radius at the lower boundary of each layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        photosphere_tau (float):
            Optical depth to determine photosphere radius.
    
    Returns:
        R_p_eff (np.array of float):
            Photosphere radius as a function of wavelength (m).
    '''
    
    thread = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Calculate photosphere radius at tau = 2/3 for each wavelength
    for k in range(thread, len(wl), stride):

        # Interpolate layer boundary radii to find radius where tau = 2/3 (lower because we are integrating down) 
        R_p_eff[k] = interp_GPU(photosphere_tau, tau_lambda[:,k], r_low)