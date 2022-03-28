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


def load_stellar_pysynphot(T_eff, Met, log_g, grid = 'cbk04'):
    
    ''' Load a stellar model using pysynphot. Pynshot's ICAT function
        automatically handles interpolation within the model grid to the
        specified stellar parameters.
        
        Inputs:
            
        T => stellar effective temperature (K)
        
        Fixed inputs (from config.py):
            
        Met => stellar metallicity (dex, solar units)
        log_g => stellar log10(surface gravity) (cgs)
        
        Outputs:
            
        wl_grid => wavelengths in model stellar spectrum (um)
        Fs_grid => stellar surface flux (W/m^2/sr/m)
            
    '''
    
    # Load Phoenix model interpolated to stellar parameters
    if (grid == 'cbk04'):
        sp = psyn.Icat('ck04models', T_eff, Met, log_g)
    elif (grid == 'phoenix'):
        sp = psyn.Icat('phoenix', T_eff, 0.0, log_g)   # Some Phoenix models with Met =/= 0 have issues...
    else:
        raise Exception("Unsupported stellar grid")

    sp.convert("um")                # Convert wavelengths to micron
    sp.convert('flam')              # Convert to flux units (erg/s/cm^2/A)

    wl_grid = sp.wave                         # Stellar wavelength array (m)
    Fs_grid = (sp.flux*1e-7*1e4*1e10)/np.pi   # Convert to W/m^2/sr/m
    
    return wl_grid, Fs_grid

    
def precompute_stellar_spectra(wl_out, star, prior_types, prior_ranges,
                               T_step_interp = 10):
    
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

    # Unpack stellar properties
    Met_s = star['stellar_metallicity']
    log_g_s = star['stellar_log_g']

    # Set range for T_phot according to whether prior is uniform or Gaussian
    if (prior_types['T_phot'] == 'uniform'):

        T_phot_min = prior_ranges['T_phot'][0]
        T_phot_max = prior_ranges['T_phot'][1]
        T_phot_step = T_step_interp        # Default interpolation step of 10 K

    elif (prior_types['T_phot'] == 'gaussian'):

        # Unpack Gaussian prior mean and std
        T_s = prior_ranges['T_phot'][0]
        err_T_s = prior_ranges['T_phot'][1]

        T_phot_min = T_s - (10.0 * err_T_s)  # 10 sigma below mean
        T_phot_max = T_s + (10.0 * err_T_s)  # 10 sigma above mean
        T_phot_step = err_T_s/10.0      # Interpolation step of 0.1 sigma

    # Set range for T_het according to whether prior is uniform or Gaussian
    if (prior_types['T_het'] == 'uniform'):

        T_het_min = prior_ranges['T_het'][0]
        T_het_max = prior_ranges['T_het'][1]
        T_het_step = T_step_interp        # Default interpolation step of 10 K

    elif (prior_types['T_het'] == 'gaussian'):

        # Unpack Gaussian prior mean and std
        T_het_mean = prior_ranges['T_het'][0]
        err_T_het = prior_ranges['T_het'][1]

        T_het_min = T_het_mean - (10.0 * err_T_het)  # 10 sigma below mean
        T_het_max = T_het_mean + (10.0 * err_T_het)  # 10 sigma above mean
        T_het_step = err_T_het/10.0      # Interpolation step of 0.1 sigma   

    #***** Interpolate photosphere spectra *****#
    
    # Find number of spectra needed to span desired range (rounding up)
    N_spec_phot = np.ceil((T_phot_max - T_phot_min)/T_phot_step).astype(np.int64) + 1

    # Specify starting and ending grid temperatures
    T_start = T_phot_min                      
    T_end = T_phot_min + (N_spec_phot * T_phot_step)   # Slightly > T_max if T_step doesn't divide exactly 
    
    # Initialise photosphere temperature array
    T_phot_grid = np.linspace(T_start, T_end, N_spec_phot)

    # Initialise output photosphere spectra array
    I_phot_out = np.zeros(shape=(N_spec_phot, len(wl_out)))

    # Interpolate and store stellar intensities
    for j in range(N_spec_phot):
        
        # Load interpolated stellar spectrum from model grid
        wl_grid, I_phot_grid = load_stellar_pysynphot(T_phot_grid[j], Met_s, log_g_s)
        
        # Bin / interpolate stellar spectrum to output wavelength grid
        I_phot_out[j,:] = spectres(wl_out, wl_grid, I_phot_grid)

    #***** Interpolate heterogeneity spectra *****#
    
    # Find number of spectra needed to span desired range (rounding up)
    N_spec_het = np.ceil((T_het_max - T_het_min)/T_het_step).astype(np.int64) + 1

    # Specify starting and ending grid temperatures
    T_start = T_het_min                      
    T_end = T_het_min + (N_spec_het * T_het_step)   # Slightly > T_max if T_step doesn't divide exactly 
    
    # Initialise heterogeneity temperature array
    T_het_grid = np.linspace(T_start, T_end, N_spec_het)

    # Initialise output heterogeneity spectra array
    I_het_out = np.zeros(shape=(N_spec_het, len(wl_out)))

    # Interpolate and store stellar intensities
    for j in range(N_spec_het):
        
        # Load interpolated stellar spectrum from model grid
        wl_grid, I_het_grid = load_stellar_pysynphot(T_het_grid[j], Met_s, log_g_s)
        
        # Bin / interpolate stellar spectrum to output wavelength grid
        I_het_out[j,:] = spectres(wl_out, wl_grid, I_het_grid)

    return T_phot_grid, T_het_grid, I_phot_out, I_het_out


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


@jit(nopython = True)
def stellar_contamination_general(f, I_het, I_phot):
    
    ''' Computes the multiplicative stellar contamination factor for a
        transmission spectrum due to an unocculted starspot or facular region.
        Prescription is as in Rackham+(2017,2018).
        
        Inputs:
            
        f => spot / facular covering fractions
        I_het => specific intensities of unocculted spot / faculae (W/m^2/sr/m)
        I_phot => specific intensity of occulted photosphere (W/m^2/sr/m)
        
        Outputs:
            
        epsilon => stellar contamination factor (dimensionless)
        
    '''
    
    N_wl = np.shape(I_phot)[0]

    sum = np.zeros(N_wl)   

    # Add contributions from each heterogeneity to contamination factor
    for i in range(len(f)):

        I_ratio = I_het[i,:]/I_phot

        sum += (f[i] * (1.0 - I_ratio))

    # Compute (wavelength-dependent) stellar heterogeneity correction
    epsilon = 1.0/(1.0 - sum)
    
    return epsilon


def stellar_contamination(star, wl_out):
    '''
    ADD DOCSTRING
    '''

    # Unpack stellar properties
    wl_s = star['wl_star']
    f_het = star['f_het']
    I_phot = star['I_phot']
    I_het = star['I_het']

    # Initialise stellar heterogeneity intensity array
    I_het_interp = np.zeros(len(wl_out))

    # Interpolate and store stellar intensities
    I_phot_interp = spectres(wl_out, wl_s, I_phot)

    # Obtain heterogeneity spectra by interpolation
    I_het_interp = spectres(wl_out, wl_s, I_het)

    epsilon = stellar_contamination_single_spot(f_het, I_het_interp, I_phot_interp)

    return epsilon