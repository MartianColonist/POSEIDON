# Stellar spectra and starspot contamination calculations

import numpy as np
from numba.core.decorators import jit
from utility import prior_index, prior_index_V2, closest_index
#from config import N_D, R_p, R_s
from utility import bin_spectrum_fast
from spectres import spectres
import scipy.constants as sc
import pysynphot as psyn
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter, NullFormatter

plt.style.use('classic')
plt.rc('font', family='serif')
#matplotlib.rcParams['svg.fonttype'] = 'none'

from config import Met_s, log_g_s, T_phot_min, T_phot_max, T_phot_step, T_het_min, T_het_max, T_het_step


@jit(nopython = True)
def planck_lambda(T, wl):
    
    ''' Compute the Planck spectral radiance at temperature T as a 
        function of wavelength.
        
        Inputs: 
            
        wl => array of model wavelengths (m)
        T => effective temperature of star (K)
        
        Outputs:
            
        B_lambda => Planck function array [wl] W m^-2 sr^-1 m^-1)
    
    '''
    
    # Define Planck function array
    B_lambda = np.zeros(shape=(len(wl)))  # (Temperature, wavelength)
    
    # Second radiative constant
    c_2 = (sc.h * sc.c) / sc.k
    
    # For each wavelength
    for k in range(len(wl)):
        
        # Precompute Planck function coefficient prefactor
        coeff = (2.0 * sc.h * sc.c**2) / (wl[k]**5)
            
        # Evalaute Planck function spectral radiance
        B_lambda[k] = coeff * (1.0 / (np.exp(c_2 / (wl[k] * T)) - 1.0))
            
    return B_lambda

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


if (1 == 2):
    
    wl_min = 0.2
    wl_max = 5.0
    R = 1000
    
    delta_log_wl = 1.0/R
    N_wl = (np.log(wl_max) - np.log(wl_min)) / delta_log_wl
    N_wl = np.around(N_wl).astype(np.int64)
    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), N_wl)    
    
    wl = np.exp(log_wl)        
    
    print("Wavelength grid is: R = " + str(R))
    
    T = 5980.0
    M = 0.130
    logg = 4.359
    
    f = 0.10
    
    # Compute Planck function for reference
    B = planck_lambda(T, wl*1.0e-6)
    
    # Load stellar models interpolated to surface parameters    
    wl_s, Fs_phot = load_stellar_pysynphot(T, Met_s, log_g_s)
    wl_s, Fs_spot = load_stellar_pysynphot((T-300), Met_s, log_g_s)
    wl_s, Fs_fac  = load_stellar_pysynphot((T+300), Met_s, log_g_s)
    
    # TEST: new stellar interpolation function
    T_s_grid, I_grid = precompute_stellar_spectra(wl, 'unocculted')
    
    Is_phot = I_grid[closest_index(T, T_s_grid[0], T_s_grid[-1], len(T_s_grid)),:]
    Is_spot = I_grid[closest_index((T-300), T_s_grid[0], T_s_grid[-1], len(T_s_grid)),:]
    Is_fac  = I_grid[closest_index((T+300), T_s_grid[0], T_s_grid[-1], len(T_s_grid)),:]
    
    # Compute starspot and faculae correction factors
    epsilon_spot = stellar_contamination_single_spot(f, Is_phot, Is_spot)
    epsilon_fac = stellar_contamination_single_spot(f, Is_phot, Is_fac)
    
    #wl_binned, Fs_phot = bin_spectrum_fast(wl_phoenix, Fs_phot_0, 0.2, 5.0, 100)
    #wl_binned, Fs_spot = bin_spectrum_fast(wl_phoenix, Fs_spot_0, 0.2, 5.0, 100)
    #wl_binned, Fs_facula = bin_spectrum_fast(wl_phoenix, Fs_facula_0, 0.2, 5.0, 100)
    
    # Plot stellar, starspot, and facular correction
    fig = plt.figure()  
        
    ax1 = plt.gca()
        
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()
    
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    
    ax1.plot(wl, B, alpha=0.4, color = 'black', label=r'Black body')
    ax1.plot(wl_s, Fs_phot, alpha=0.4, color = 'red', label=r'Photosphere (phoenix)')
 #   ax1.plot(wl, Is_phot, alpha=0.4, color = 'darkred', label=r'Photosphere ($R=1000$)')
    ax1.plot(wl_s, Fs_spot, alpha=0.4, color = 'green', label=r'Spot (phoenix)')
 #   ax1.plot(wl, Is_spot, alpha=0.4, color = 'darkgreen', label=r'Spot ($R=1000$)')
    ax1.plot(wl_s, Fs_fac, alpha=0.4, color = 'blue', label=r'Facula (phoenix)')
 #   ax1.plot(wl, Is_fac, alpha=0.4, color = 'darkblue', label=r'Facula ($R=1000$)')
            
    ax1.set_xlabel(r'Wavelength ' + r'($\mu$m)', fontsize = 16)
    ax1.set_ylabel(r'Flux ' + r'($W \, m^{-2} \, sr^{-1} \, m^{-1}$)', fontsize = 16)
    
    ax1.set_xlim([wl[0], wl[-1]])
    ax1.set_ylim([1.0e10, 1.0e14])
    ax1.set_xticks([0.2, 0.4, 0.7, 1.0, 2.0, 3.0, 5.0])
    
    ax1.legend(loc='upper right', shadow=True, prop={'size':10}, ncol=1, frameon=False)
    
    plt.savefig('../../output/plots/stellar_spectra_phoenix.pdf', bbox_inches='tight', fmt='pdf', dpi=300)
    
    # Plot stellar contamination spectrum
    fig = plt.figure()  
        
    ax1 = plt.gca()
        
    ax1.set_xscale("log")
  #  ax1.set_yscale("log")
    
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()
    
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    
    ax1.plot(wl, epsilon_spot, alpha=0.4, color = 'green', label=r'Spot (ck04)')
    ax1.plot(wl, epsilon_fac, alpha=0.4, color = 'blue', label=r'Facula (ck04)')
            
    ax1.set_xlabel(r'Wavelength ' + r'($\mu$m)', fontsize = 16)
    ax1.set_ylabel(r'Contamination Spectrum', fontsize = 16)
    
    ax1.set_xlim([wl[0], wl[-1]])
    ax1.set_ylim([0.75, 1.10])
    ax1.set_xticks([0.2, 0.4, 0.7, 1.0, 2.0, 3.0, 5.0])
    
    ax1.legend(loc='lower right', shadow=True, prop={'size':10}, ncol=1, frameon=False)
    
 #   plt.savefig('../../output/plots/contamination_spectra_test.pdf', bbox_inches='tight', fmt='pdf', dpi=300)


