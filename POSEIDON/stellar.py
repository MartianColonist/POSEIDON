# Stellar spectra and starspot contamination calculations

import numpy as np
from numba.core.decorators import jit
from spectres import spectres
import scipy.constants as sc
import pysynphot as psyn

from .utility import closest_index


@jit(nopython = True)
def planck_lambda(T, wl):
    
    ''' Compute the Planck spectral radiance at temperature T as a 
        function of wavelength.
        
        Inputs: 
            
        T => effective temperature of star (K)
        wl => array of model wavelengths (μm)
        
        Outputs:
            
        B_lambda => Planck function array [wl] W m^-2 sr^-1 m^-1)
    
    '''
    
    # Define Planck function array
    B_lambda = np.zeros(shape=(len(wl))) 

    # Convert wavelength array to m
    wl_m = wl * 1.0e-6
    
    # Second radiative constant
    c_2 = (sc.h * sc.c) / sc.k
    
    # For each wavelength
    for k in range(len(wl)):
        
        # Precompute Planck function coefficient prefactor
        coeff = (2.0 * sc.h * sc.c**2) / (wl_m[k]**5)
            
        # Evaluate Planck function spectral radiance
        B_lambda[k] = coeff * (1.0 / (np.exp(c_2 / (wl_m[k] * T)) - 1.0))
            
    return B_lambda


# TBD: update functions below 

def load_stellar_pysynphot(T, M, logg):
    
    ''' Load a ck04 stellar spectrum using pysynphot. Pynshot's ICAT function
        automatically handles interpolation within the model grid to the
        specified stellar parameters.
        
        Inputs:
            
        T => stellar effective temperature (K)
        
        Fixed inputs (from config.py):
            
        M_s => stellar metallicity (dex, solar units)
        log_g_s => stellar log10(surface gravity) (cgs)
        
        Outputs:
            
        wl_grid => wavelengths in model stellar spectrum (um)
        Fs_grid => stellar surface flux (W/m^2/sr/m)
            
    '''
    
    # Load Phoenix model interpolated to stellar parameters
    sp = psyn.Icat('ck04models', T, M, logg)
    sp.convert("um")                          # Convert wavelengths to micron
    sp.convert('flam')                        # Convert to flux units (erg/s/cm^2/A)
    wl_grid = sp.wave                         # Phoenix model wavelength array (m)
    Fs_grid = (sp.flux*1e-7*1e4*1e10)/np.pi   # Convert to W/m^2/sr/m
    
    return wl_grid, Fs_grid

###### Function to pre-compute interpolated stellar grid here #######
    
def precompute_stellar_spectra(wl_out, component):
    
    ''' Precompute a grid of stellar spectra across a range of T_eff.
        This function uses the ck04 low-resolution models from pysynphot.
        
        Inputs:
            
        wl_out => desired output wavelength grid (um)
        component => stellar region to compute grid for ['photosphere' / 'unocculted']
        
        Fixed inputs (from config.py):
            
        Met_s => stellar metallicity (dex, solar units)
        log_g_s => stellar log10(surface gravity) (cgs)
        
        Outputs:
            
        T_grid => array of effective temperatures spanned by grid (K)
        I_out => stellar intensity array on output wavelength grid [T, wl] (W/m^2/sr/m)
    
    '''
    
    if (component == 'photosphere'):
        T_min, T_max, T_step = T_phot_min, T_phot_max, T_phot_step
    elif (component == 'unocculted'):
        T_min, T_max, T_step = T_het_min, T_het_max, T_het_step
    
    # Find number of spectra needed to span desired range (rounding up)
    N_spec = np.ceil((T_max - T_min)/T_step).astype(np.int64) + 1
    
    # Specify starting and ending grid temperatures
    T_start = T_min                      
    T_end = T_min + (N_spec * T_step)   # Slightly > T_max if T_step doesn't divide exactly 
    
    # Effective temperatures of model grid
    T_grid = np.linspace(T_start, T_end, N_spec)
    
    # Initialise output (interpolated) intensity array
    I_out = np.zeros(shape=(N_spec, len(wl_out)))
    
    # Interpolate and store stellar intensities
    for i in range(N_spec):
        
        # Load interpolated stellar spectrum from model grid
        wl_grid, I_grid = load_stellar_pysynphot(T_grid[i], Met_s, log_g_s)
        
        # Bin / interpolate stellar spectrum to output wavelength grid
        I_out[i,:] = spectres(wl_out, wl_grid, I_grid)

    return T_grid, I_out


@jit(nopython = True)
def stellar_contamination_single_spot(f, I_het, I_phot):
    
    ''' Computes the multiplicative stellar contamination factor for a
        transmission spectrum due to an unocculted starspot or facular region.
        Prescription is as in Rackham+(2017,2018).
        
        Note: this function assumes only ONE contaminating region.
        
        Inputs:
            
        f => spot / facular covering fraction
        I_het => specific intensity of unocculted spot / faculae (W/m^2/sr/m)
        I_phot => specific intensity of occulted photosphere (W/m^2/sr/m)
        
        Outputs:
            
        epsilon => stellar contamination factor (dimensionless)
        
    '''
    
    # Compute (wavelength-dependent) stellar heterogeneity correction
    epsilon = 1.0/(1.0 - f*(1.0 - I_het/I_phot))
    
    return epsilon

