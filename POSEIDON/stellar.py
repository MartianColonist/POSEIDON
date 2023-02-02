''' 
Stellar spectra and star spot/faculae contamination calculations.

'''

import numpy as np
from numba.core.decorators import jit
from spectres import spectres
import scipy.constants as sc
import pysynphot as psyn


@jit(nopython = True)
def planck_lambda(T, wl):
    '''
    Compute the Planck function spectral radiance.

    Args:
        T (float):
            Effective temperature of star (K).
        wl (np.array of float): 
            Wavelength grid (μm).
    
    Returns:
        B_lambda (np.array of float):
            Planck function spectral radiance in SI units (W/m^2/sr/m).

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


def load_stellar_pysynphot(T_eff, Met, log_g, stellar_grid = 'cbk04'):
    '''
    Load a stellar model using pysynphot. Pynshot's ICAT function handles
    interpolation within the model grid to the specified stellar parameters.

    Note: Pysynphot's PHOENIX model grids have some bugs for non-solar 
          metallicity. So we default to the Castelli-Kurucz 2004 grids.

    Args:
        T_eff (float):
            Effective temperature of star (K).
        Met (float):
            Stellar metallicity [log10(Fe/H_star / Fe/H_solar)].
        log_g (float):
            Stellar log surface gravity (log10(cm/s^2) by convention).
        stellar_grid (str):
            Desired stellar model grid
            (Options: cbk04 / phoenix).
    
    Returns:
        wl_grid (np.array of float):
            Wavelength grid for model stellar spectrum (μm).
        I_grid (np.array of float):
            Stellar specific intensity spectrum in SI units (W/m^2/sr/m).

    '''
    
    # Load Phoenix model interpolated to stellar parameters
    if (stellar_grid == 'cbk04'):
        sp = psyn.Icat('ck04models', T_eff, Met, log_g)
    elif (stellar_grid == 'phoenix'):
        sp = psyn.Icat('phoenix', T_eff, 0.0, log_g)   # Some Phoenix models with Met =/= 0 have issues...
    else:
        raise Exception("Unsupported stellar grid")

    sp.convert("um")                # Convert wavelengths to micron
    sp.convert('flam')              # Convert to flux units (erg/s/cm^2/A)

    wl_grid = sp.wave                         # Stellar wavelength array (m)
    I_grid = (sp.flux*1e-7*1e4*1e10)/np.pi    # Convert to W/m^2/sr/m
    
    return wl_grid, I_grid

    
def precompute_stellar_spectra(wl_out, star, prior_types, prior_ranges,
                               stellar_grid = 'cbk04', T_step_interp = 10):
    '''
    Precompute a grid of stellar spectra across a range of T_eff.

    Note: Pysynphot's PHOENIX model grids have some bugs for non-solar 
          metallicity. So we default to the Castelli-Kurucz 2004 grids.

    Args:
        wl_out (np.array of float):
            Wavelength grid on which to output the stellar spectra (μm).
        star (dict):
            Collection of stellar properties used by POSEIDON.
        prior_types (dict):
            User-provided dictionary containing the prior type for each 
            free parameter in the retrieval model.
        prior_ranges (dict):
            User-provided dictionary containing numbers defining the prior range
            for each free parameter in the retrieval model.
        stellar_grid (str):
            Desired stellar model grid
            (Options: cbk04 / phoenix).
        T_step_interp (float):
            Temperature step for stellar grid interpolation.
    
    Returns:
        T_phot_grid (np.array of float):
            Photosphere temperatures corresponding to computed stellar spectra (K).
        T_het_grid (np.array of float):
            Heterogeneity temperatures corresponding to computed stellar spectra (K).
        I_phot_out (2D np.array of float):
            Stellar photosphere intensity as a function of T_phot and wl (W/m^2/sr/m).
        I_het_out (2D np.array of float):
            Stellar heterogeneity intensity as a function of T_het and wl (W/m^2/sr/m).

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
        T_phot_step = err_T_s/10.0           # Interpolation step of 0.1 sigma

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
        wl_grid, I_phot_grid = load_stellar_pysynphot(T_phot_grid[j], Met_s, 
                                                      log_g_s, stellar_grid)
        
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
    '''
    Computes the multiplicative stellar contamination factor for a transmission
    spectrum due to an unocculted starspot or facular region. The prescription
    used is as in Rackham+(2017,2018).

    Args:
        f (float):
            Fraction of the stellar surface covered by the heterogeneity.
        I_het (np.array of float):
            Stellar heterogeneity intensity as a function of wl (W/m^2/sr/m).
        I_phot (np.array of float):
            Stellar photosphere intensity as a function of wl (W/m^2/sr/m).
    
    Returns:
        epsilon (np.array of float):
            Stellar contamination factor as a function of wl.

    '''
    
    # Compute (wavelength-dependent) stellar heterogeneity correction
    epsilon = 1.0/(1.0 - f*(1.0 - I_het/I_phot))
    
    return epsilon


@jit(nopython = True)
def stellar_contamination_general(f, I_het, I_phot):
    '''
    Computes the multiplicative stellar contamination factor for a transmission
    spectrum due to a collection of unocculted starspot or facular regions, each
    of which may have their own coverage fraction and specific intensity.

    Args:
        f (np.array of float):
            Fraction of the stellar surface covered each heterogeneity.
        I_het (2D np.array of float):
            Stellar heterogeneity intensity as a function of region and wl (W/m^2/sr/m).
        I_phot (np.array of float):
            Stellar photosphere intensity as a function of wl (W/m^2/sr/m).
    
    Returns:
        epsilon (np.array of float):
            Stellar contamination factor as a function of wl.

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
    Computes the multiplicative stellar contamination factor for a transmission
    spectrum due to an unocculted starspot or facular region. The prescription
    used is as in Rackham+(2017,2018).

    This function unpacks the required stellar properties from POSEIDON's
    'star' object, interpolates the stellar intensities to the output 
    wavelength grid, then computes the stellar contamination for a single 
    heterogeneous region.

    Note: this function is only used for forward models, since for retrievals 
          the wavelength interpolation is handled by another function called 
          'precompute_stellar_spectra'.

    Args:
        star (dict):
            Collection of stellar properties used by POSEIDON.
        wl_out (np.array of float):
            Wavelength grid on which to output the stellar spectra (μm).
    
    Returns:
        epsilon (np.array of float):
            Stellar contamination factor as a function of wl.

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

    # Compute (wavelength-dependent) stellar heterogeneity correction
    epsilon = stellar_contamination_single_spot(f_het, I_het_interp, I_phot_interp)

    return epsilon