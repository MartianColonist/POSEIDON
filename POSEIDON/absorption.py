# ***** Absorption.py - cross section and extinction computations *****
# V 8.0: Support for H- bound-free and free-free absorption added.

import os
import numpy as np
import h5py
from mpi4py import MPI
import scipy.constants as sc
from numba import cuda, jit
import math

from .utility import mock_missing

try:
    import cupy as cp
except ImportError:
    cp = mock_missing('cupy')

                   
from .species_data import polarisabilities
from .utility import prior_index, prior_index_V2, closest_index, closest_index_GPU
           

@jit(nopython = True)
def P_interpolate_wl_initialise_sigma(N_P_fine, N_T, N_P, N_wl, log_sigma,
                                      x, nu_l, nu_model, nu_r, b1, b2, 
                                      nu_opac, N_nu):
    '''
    Interpolates raw cross section onto the desired P and wl grids.
       
    Input sigma has format log10(cross_sec)[log(P)_grid, T_grid, nu_grid], 
    whilst output has format cross_sec[log(P)_pre, T_grid, wl_model].
            
    The input is in wavenumber to take advantage of fast prior index 
    location on a uniform grid, which wouldn't work for the (non-uniform) 
    wavelength grid. Array reversal to output in increasing wavelength is 
    handled by indexing by a factor of (N_wl-1)-k throughout .
            
    If N_P_fine = N_layer, then pressure interpolation is onto the model P grid.
    However, the low sensitivity to small pressure variations means that
    empirical tests have shown that pre-interpolation to N_P_fine >= 0.28*N_layer 
    and simply choosing the nearest element in log(P)_pre results in minimal
    errors. This has the advantage of lowering memory usage.
    
    Wavelength initialisation is handled via opacity sampling (i.e. setting the 
    cross section to the nearest pre-computed wavelength point).
       
    '''
    
    sigma_pre_inp = np.zeros(shape=(N_P_fine, N_T, N_wl))
    
    N_nu_opac = len(nu_opac)   # Number of wavenumber points in CIA array
    
    for k in range(N_nu): # Note that the k here is looping over wavenumber
        
        # Indices in pre-computed wavenumber array of LHS, centre, and RHS of desired wavenumber grid
        z_l = closest_index(nu_l[k], nu_opac[0], nu_opac[-1], N_nu_opac)
        z = closest_index(nu_model[k], nu_opac[0], nu_opac[-1], N_nu_opac)
        z_r = closest_index(nu_r[k], nu_opac[0], nu_opac[-1], N_nu_opac)
    
        for i in range(N_P_fine):
            for j in range(N_T):
            
                # If nu (wl) point out of range of opacity grid, set opacity to zero
                if ((z == 0) or (z == (N_nu_opac-1))):
                    sigma_pre_inp[i, j, ((N_wl-1)-k)] = 0.0
                
                else:
                            
                    # Opacity sampling
                    
                    # If pressure below minimum, set to value at min pressure
                    if (x[i] == -1):
                        sigma_pre_inp[i, j, ((N_wl-1)-k)] = 10 ** (log_sigma[0, j, z])
                        
                    # If pressure above maximum, set to value at max pressure
                    elif (x[i] == -2):
                        sigma_pre_inp[i, j, ((N_wl-1)-k)] = 10 ** (log_sigma[(N_P-1), j, z])
        
                    # Interpolate sigma in logsace, then power to get interp array
                    else:
                        reduced_sigma = log_sigma[x[i]:x[i]+2, j, z]
                        
                        sigma_pre_inp[i, j, ((N_wl-1)-k)] =  10 ** (b1[i]*(reduced_sigma[0]) +
                                                                    b2[i]*(reduced_sigma[1]))
                                                        
    return sigma_pre_inp
                    
                
@jit(nopython = True)
def wl_initialise_cia(N_T_cia, N_wl, log_cia, nu_l, nu_model, nu_r, 
                      nu_cia, N_nu):
    '''
    Interpolates raw collisionally-induced absorption (CIA) binary cross 
    section onto the desired model wl grid.
    
    Input cia has format log10(alpha)[T_grid, nu_grid], 
    whilst output has format alpha[T_grid, wl_model].
            
    The input is in wavenumber to take advantage of fast prior index 
    location on a uniform grid, which wouldn't work for the (non-uniform) 
    wavelength grid. Array reversal to output in increasing wavelength is 
    handled by indexing by a factor of (N_wl-1)-k throughout .
            
    Wavelength initialisation is handled via either opacity sampling
    (choosing nearest pre-computed wavelength point) or via averaging
    the logarithm of the cross section over the wavelength bin range 
    surrounding each wavelength on the output model wavelength grid.
       
    '''
    
    cia_pre_inp = np.zeros(shape=(N_T_cia, N_wl))
    
    N_nu_cia = len(nu_cia)   # Number of wavenumber points in CIA array

    for i in range(N_T_cia):  
        for k in range(N_nu):
            
            z_l = closest_index(nu_l[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            z = closest_index(nu_model[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            z_r = closest_index(nu_r[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            
            # If wl out of range of opacity, set opacity to zero
            if ((z_l == 0) or (z_r == (N_nu_cia-1))):
                cia_pre_inp[i, ((N_wl-1)-k)] = 0.0
                
            else:
                
                # Opacity sampling    
                cia_pre_inp[i, ((N_wl-1)-k)] = 10 ** (log_cia[i, z])
               
    return cia_pre_inp


@jit(nopython = True)
def T_interpolation_init(N_T_fine, T_grid, T_fine, y):
    ''' 
    Precomputes the T interpolation weight factors, so this does not
    need to be done multiple times across all species.
        
    '''
    
    w_T = np.zeros(N_T_fine)
    
    # Find T index in cross section arrays prior to fine temperature value
    for j in range(N_T_fine):
        
        if (T_fine[j] < T_grid[0]):   # If fine temperature point falls off LHS of temperature grid
            y[j] = -1                 # Special value (-1) stored, interpreted in interpellator
            w_T[j] = 0.0              # Weight not used in this case
            
        elif (T_fine[j] >= T_grid[-1]):   # If fine temperature point falls off RHS of temperature grid
            y[j] = -2                     # Special value (-2) stored, interpreted in interpellator
            w_T[j] = 0.0                  # Weight not used in this case
        
        else:
            
            # Have to use prior_index (V1) here as T_grid is not uniformly spaced
            y[j] = prior_index(T_fine[j], T_grid, 0)       # For cross section T interpolation
            
            # Pre-computed temperatures to left and right of fine temperature value
            T1 = T_grid[y[j]]
            T2 = T_grid[y[j]+1]
            
            # Precompute temperature weight factor
            w_T[j] = (1.0/((1.0/T2) - (1.0/T1)))
  
    return w_T


@jit(nopython = True)
def T_interpolate_sigma(N_P_fine, N_T_fine, N_T, N_wl, sigma_pre_inp, T_grid, 
                        T_fine, y, w_T):
    ''' 
    Interpolates pre-processed cross section onto the fine T grid.
       
    Note: input sigma has format cross_sec[log(P)_pre, T_grid, wl_model], 
          whilst output has format cross_sec[log(P)_pre, T_fine, wl_model].
             
    Output is the interpolated cross section as a 3D array.
        
    '''
    
    sigma_inp = np.zeros(shape=(N_P_fine, N_T_fine, N_wl))

    for i in range(N_P_fine):       # Loop over pressures
        for j in range(N_T_fine):   # Loop over temperatures
            
            T = T_fine[j]           # Temperature we wish to interpolate to
            T1 = T_grid[y[j]]
            T2 = T_grid[y[j]+1]
            
            for k in range(N_wl):   # Loop over wavelengths
                
                # If T_fine below min value (100 K), set sigma to value at min T
                if (y[j] == -1):
                    sigma_inp[i, j, k] = sigma_pre_inp[i, 0, k]
                    
                # If T_fine above max value (3500 K), set sigma to value at max T
                elif (y[j] == -2):
                    sigma_inp[i, j, k] = sigma_pre_inp[i, (N_T-1), k]
            
                # Interpolate sigma to fine temperature grid value
                else: 
                    sig_reduced = sigma_pre_inp[i, y[j]:y[j]+2, k]
                    sig_1, sig_2 = sig_reduced[0], sig_reduced[1]    # sigma(T1)[i,k], sigma(T2)[i,k]
                    
                    sigma_inp[i, j, k] =  (np.power(sig_1, (w_T[j]*((1.0/T2) - (1.0/T)))) *
                                           np.power(sig_2, (w_T[j]*((1.0/T) - (1.0/T1)))))
            
    return sigma_inp


@jit(nopython = True)
def T_interpolate_cia(N_T_fine, N_T_cia, N_wl, cia_pre_inp, T_grid_cia, 
                      T_fine, y, w_T):
    ''' 
    Interpolates pre-processed collisionally-induced absorption (CIA) 
    binary cross section onto the fine T grid.
       
    Note: input sigma has format alpha[T_grid, wl_model], 
          whilst output has format alpha[T_fine, wl_model].
            
    Output is the interpolated cia cross section as a 2D array.
        
    '''
    
    cia_inp = np.zeros(shape=(N_T_fine, N_wl))

    for j in range(N_T_fine):    # Loop over temperatures
            
        T = T_fine[j]            # Temperature we wish to interpolate to
        T1 = T_grid_cia[y[j]]
        T2 = T_grid_cia[y[j]+1]
        
        for k in range(N_wl):    # Loop over wavelengths
                
            # If T_fine below min value (200 K), set cia to value at min T
            if (y[j] == -1):
                cia_inp[j, k] = cia_pre_inp[0, k]
                
            # If T_fine above max value (3500 K), set cia to value at max T
            elif (y[j] == -2):
                cia_inp[j, k] = cia_pre_inp[(N_T_cia-1), k]
            
            # Interpolate sigma to fine temperature grid value
            else: 
                cia_reduced = cia_pre_inp[y[j]:y[j]+2, k]
                cia_1, cia_2 = cia_reduced[0], cia_reduced[1]    # sigma(T1)[i,k], sigma(T2)[i,k]
                    
                cia_inp[j, k] =  (np.power(cia_1, (w_T[j]*((1.0/T2) - (1.0/T)))) *
                                  np.power(cia_2, (w_T[j]*((1.0/T) - (1.0/T1)))))
            
    return cia_inp


def refractive_index(wl, n_ref, species):
    ''' 
    Computes the refractive index of a molecule / atom at a set of 
    wavelengths for standard conditions (T = 273.15K / P = 1 atm).
    
    'eta' is refractive index, as 'n' is reserved for number density.

    Inputs:
        
    wl => array of wavelength values (um)
    n_ref => number density at standard reference conditions (cm^-3)
    species => string specifying chosen chemical species
    
    Outputs:
        
    eta => refractive index array as a function of wavelength 
        
    '''
    
    nu = 1.0e4/wl  # Wavenumber in cm^-1
    
    if (species in ['H2', 'O2', 'N2O', 'CO2', 'NH3']):
        
        if (species == 'H2'):
            
            f_par = 1.62632           # Constants for Hohm, 1993 fit (see bottom of this function)
            w_par_sq = 0.23940245 
            f_perp = 1.40105 
            w_perp_sq = 0.29486069 
            
        elif (species == 'O2'):
                
            f_par = 2.74876
            w_par_sq = 0.18095751 
            f_perp = 4.86007 
            w_perp_sq = 0.58545449 
            
        elif (species == 'CO2'):
            
            f_par = 6.00332
            w_par_sq = 0.22525399 
            f_perp = 8.54433 
            w_perp_sq = 0.66083749
            
        elif (species == 'NH3'):
            
            f_par = 1.28964
            w_par_sq = 0.08454599 
            f_perp = 10.84943 
            w_perp_sq = 0.76338846 
        
        elif (species == 'N2O'):
            
            f_par = 5.65126
            w_par_sq = 0.17424213
            f_perp = 9.72095
            w_perp_sq = 0.72904985 
            
        # Now calculate polarisability using formula from Hohm, 1993
        alpha = ((1.0/3.0)*((f_par/(w_par_sq - (nu/219474.6305)**2)) +                  # Polarisability - Hohm, 1993
                       2.0*(f_perp/(w_perp_sq - (nu/219474.6305)**2))))*0.148184e-24    # Convert to cm^3

        eta = np.sqrt((1.0 + (8.0*np.pi*n_ref*alpha/3.0))/(1.0 - (4.0*np.pi*n_ref*alpha/3.0)))  # Lorentz-Lorenz relation 
        
    # Many of these below are from: http://refractiveindex.info (Polyanskiy, 2016)
    elif (species == 'He'):
        
        eta = 1.0 + ((0.014755297/(426.29740 - 1.0/(wl**2)))*1.0018141444038913)    # Cuthbertson & Cuthbertson, 1936 (multiplicative factor for continuity)
        eta[wl < 0.2753] = 1.00003578
        eta[wl > 0.4801] = 1.0 + (0.01470091/(423.98 - 1.0/(wl[wl > 0.4801]**2)))   # Mansfield & Peck, 1969
        eta[wl > 2.0586] = 1.00003469
    
    elif (species == 'N2'):
    
        eta = 1.0 + ((5677.465e-8 + (318.81874e4/(14.4e9 - nu**2)))*1.0001468057477378)   # Sneep & Ubachs, 2005 (sec 4.2) / Bates, 1984  (fmultiplicative actor for continuity)
        eta[wl < 0.2540] = 1.00030493
        eta[wl > 0.46816] = 1.0 + (6498.2e-8 + (307.43305e4/(14.4e9 - nu[wl > 0.46816]**2)))   # Sneep & Ubachs, 2005 (sec 4.2) / Peck & Khanna, 1966
        eta[wl > 2.0576] = 1.00027883
        
    elif (species == 'CH4'):
        
        eta = 1.0 + (46662.0e-8 + (4.02e-14 * nu**2))   # Sneep & Ubachs, 2005 (sec 5.2) / Hohm, 1993
        eta[wl < 0.325] = 1.000504679
        eta[wl > 0.633] = 1.000476653
        
    elif (species == 'H2O'):
        
        eta = 1.0 + ((3.011e-2/(124.40 - 1.0/(wl**2))) +   # Hill & Lawrence, 1986
                     (7.46e-3 * (0.203 - 1.0/wl))/(1.03 - 1.98e3/(wl**2) + 8.1e4/(wl**4) - 1.7e8/(wl**8)))   
        eta[wl < 0.360] = 1.000258047
        eta[wl > 17.60] = 1.000000000   # Technically formula goes to 19um, but can't have n<1.0  
                
    else:   # For all other species, use the static polarisability to derive a representative refractive index
        
        alpha = polarisabilities[species] * nu**0   # nu^0 is to make this an array with same size as nu
        eta = np.sqrt((1.0 + (8.0*np.pi*n_ref*alpha/3.0))/(1.0 - (4.0*np.pi*n_ref*alpha/3.0)))  # Lorentz-Lorenz relation
    
    # Finally, scale to 0 C and 1 atm (1.01325 bar), for refractive indices defined at 15 C and 1013 hPa - Sneep & Ubachs, 2005         
    if (species in ['N2', 'CH4']):
        eta = ((eta-1.0) * (288.15/273.15)) + 1.0   # -1 converts to refractivity to apply T scaling
            
    return eta  


def King_correction(wl, species):
    ''' 
    Computes the King correction factor of a molecule / atom at a set of 
    wavelengths. This accounts for depolarisation effects in Rayleigh
    scattering due to the non-spherical nature of atoms / molecules.

    Inputs:
        
    wl => array of wavelength values (um)
    species => string specifying chosen chemical species
    
    Outputs:
        
    F => King correction factor array as a function of wavelength 
        
    '''
    
    nu = 1.0e4/wl  # Wavenumber in cm^-1
    
    if (species in ['H2', 'O2', 'N2O', 'CO2', 'NH3']):
        
        if (species == 'H2'):
            
            f_par = 1.62632             # Constants for Hohm, 1993 fit (see below)
            w_par_sq = 0.23940245 
            f_perp = 1.40105 
            w_perp_sq = 0.29486069 
            
        elif (species == 'O2'):
            
            f_par = 2.74876
            w_par_sq = 0.18095751 
            f_perp = 4.86007 
            w_perp_sq = 0.58545449 
        
        elif (species == 'CO2'):
            
            f_par = 6.00332
            w_par_sq = 0.22525399 
            f_perp = 8.54433 
            w_perp_sq = 0.66083749
            
        elif (species == 'NH3'):
            
            f_par = 1.28964
            w_par_sq = 0.08454599 
            f_perp = 10.84943 
            w_perp_sq = 0.76338846 
            
        elif (species == 'N2O'):
            
            f_par = 5.65126
            w_par_sq = 0.17424213
            f_perp = 9.72095
            w_perp_sq = 0.72904985 
            
        # Calculate polarisability (alpha) and polarisability anisotropy (gamma) using formulae of Hohm, 1993
        alpha = ((1.0/3.0)*((f_par/(w_par_sq - (nu/219474.6305)**2)) +                
                       2.0*(f_perp/(w_perp_sq - (nu/219474.6305)**2))))         
        gamma = ((f_par/(w_par_sq - (nu/219474.6305)**2)) - 
                 (f_perp/(w_perp_sq - (nu/219474.6305)**2)))                    
        
        # Evaluate King correction factor
        F = 1.0 + 2.0 * (gamma/(3.0*alpha))**2  
        
    elif (species == 'N2'):    F = 1.034 + 3.17e-12 * nu**2   # Sneep & Ubachs, 2005 (sec 4.2) / Bates, 1984
    elif (species == 'He'):    F = 1.000000 * nu**0    # Spherical atom, so King correction factor = 1
    elif (species == 'H2O'):   F = 1.001005 * nu**0    # Derived from values in Hinchliffe , 2007
    elif (species == 'CH4'):   F = 1.000000 * nu**0    # Negligible difference from unity - Sneep & Ubachs, (sec 5.2)
    elif (species == 'O3'):    F = 1.060000 * nu**0    # Brasseur & De Rudder, 1986
    elif (species == 'CO'):    F = 1.016995 * nu**0    # Mean value from Bogaard+, 1978   
    elif (species == 'C2H2'):  F = 1.064385 * nu**0    # Mean value from Bogaard+, 1978   
    elif (species == 'C2H4'):  F = 1.042043 * nu**0    # Mean value from Bogaard+, 1978
    elif (species == 'C2H6'):  F = 1.006063 * nu**0    # Mean value from Bogaard+, 1978
    elif (species == 'OCS'):   F = 1.138786 * nu**0    # Mean value from Bogaard+, 1978
    elif (species == 'CH3Cl'): F = 1.026042 * nu**0    # Mean value from Bogaard+, 1978
    elif (species == 'H2S'):   F = 1.001880 * nu**0    # Mean value from Bogaard+, 1978
    elif (species == 'SO2'):   F = 1.062638 * nu**0    # Mean value from Bogaard+, 1978

    else: F = 1.0 * nu**0   # If no data available, set King correction factor to 1.0
        
    return F


def Rayleigh_cross_section(wl, species):
    '''
    Compute Rayleigh scattering cross section of a molecule / atom at a set
    of wavelengths, accounting for depolarisation effects where available. 
    
    Note: 'eta' is used for refractive index, as 'n' is reserved for
            number density.

    Inputs:
        
    wl => array of wavelength values in spectral model (um)
    species => string specifying chosen chemical species
    
    Outputs:
        
    sigma_Rayleigh => refractive index array as function of wavelength 
    eta => refractive index array as function of wavelength 
        
    '''
    
    nu = 1.0e4 / wl   # Wavenumbers in cm^-1
    
    # Compute number density at reference conditions of refractive index measurements
    n_ref = (101325.0/(sc.k * 273.15)) * 1.0e-6    # Number density (cm^-3) at 0 C and 1 atm (1.01325 bar) - http://refractiveindex.info
        
    eta = refractive_index(wl, n_ref, species)  # Refractive index for given species (nu dependant)
    F = King_correction(wl, species)            # King correction factor for depolarisation (nu dependant)
        
    # Compute Rayleigh scattering cross section (cm^2)
    sigma_Rayleigh = (((24.0 * np.pi**3 * nu**4)/(n_ref**2)) * (((eta**2 - 1.0)/(eta**2 + 2.0))**2) * F)
    
    sigma_Rayleigh = sigma_Rayleigh * 1.0e-4  # Convert to m^2
            
    return sigma_Rayleigh, eta


@jit(nopython=True)
def H_minus_bound_free(wl_um):
    ''' 
    Computes the bound-free cross section (alpha_bf) of the H- ion as a 
    function of wavelength. The fitting function is taken from "Continuous
    absorption by the negative hydrogen ion reconsidered" (John, 1988).
    
    The extinction coefficient (in m^-1) can then be calculated via:
    alpha_bf * n_(H-) [i.e. multiply by the H- number density (in m^-3) ].

    Inputs:
        
    wl_um => array of wavelength values (um)
    
    Outputs:
        
    alpha_bf => bound-free H- cross section (m^2 / n_(H-) ) at each input
                wavelength
        
    '''
    
    # Initialise array to store absorption coefficients
    alpha_bf = np.zeros(shape=(len(wl_um)))
    
    # Fitting function constants (John, 1988, p.191)
    C = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])
    
    for k in range(len(wl_um)):
        
        f = 0.0
        
        # Photodissociation only possible for photons with wl < 1.6421 micron 
        if (wl_um[k] <= 1.6421):
            
            for n in range(1,7):

                # Compute prefactor in H- bound-free cross section fit
                f += C[n-1] * np.power(((1.0/wl_um[k]) - (1.0/1.6421)), ((n-1.0)/2.0))
                
            # Compute bound-free absorption coefficient at this wavelength
            alpha_bf[k] = 1.0e-18 * wl_um[k]**3 * np.power(((1.0/wl_um[k]) - (1.0/1.6421)), (3.0/2.0)) * f
            alpha_bf[k] = alpha_bf[k] * 1.0e-4  # Convert from cm^2 / H- ion to m^2 / H- ion
        
        else:
            
            alpha_bf[k] = 1.0e-250   # Small value (proxy for zero, but avoids log(0) issues)
    
    return alpha_bf


@jit(nopython=True)
def H_minus_free_free(wl_um, T_arr):
    ''' 
    Computes the free-free cross section (alpha_ff) of the H- ion as a 
    function of wavelength. The fitting function is taken from "Continuous
    absorption by the negative hydrogen ion reconsidered" (John, 1988).
    
    The extinction coefficient (in m^-1) can then be calculated via:
    alpha_ff * n_H * n_(e-) [i.e. multiply by the H and e- number densities
    (both in in m^-3) ].

    Inputs:
        
    wl_um => array of wavelength values (um)
    T_arr => array of temperatures (K)
    
    Outputs:
        
    alpha_ff => free-free H- cross section (m^5 / n_H / n_e-) for each 
                input wavelength and temperature
        
    '''
    
    # Initialise array to store absorption coefficients
    alpha_ff = np.zeros(shape=(len(T_arr), len(wl_um)))
    
    wl = wl_um
    wl_2 = wl * wl
    wl_3 = wl_2 * wl
    wl_4 = wl_2 * wl_2
    
    # Short wavelength fit
    A_s = np.array([518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0])
    B_s = np.array([-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0])
    C_s = np.array([1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
    D_s = np.array([-479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0])
    E_s = np.array([93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0])
    F_s = np.array([-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0])
    
    # Long wavelength fit
    A_l = np.array([0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830])
    B_l = np.array([0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170])
    C_l = np.array([0.0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8640])
    D_l = np.array([0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880])
    E_l = np.array([0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880])
    F_l = np.array([0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850])
        
    for i in range(len(T_arr)):
            
        theta = 5040.0 / T_arr[i]      # Reciprocal temperature commonly used in these fits
        kT = 1.38066e-16 * T_arr[i]  # Boltzmann constant * temperature (erg)
    
        for k in range(len(wl_um)):
                
            # Range of validity of fit (wl > 0.182 um)
            if (wl[k] < 0.182):
                    
                alpha_ff[i, k] = 1.0e-250   # Small value (proxy for zero, but avoids log(0) issues)
            
            # Short wavelength fit
            elif ((wl[k] >= 0.182) and (wl[k] < 0.3645)):
                    
                # For each set of fit coefficients
                for n in range(1,7):
                
                    # Compute free-free absorption coefficient at this wavelength and temperature
                    alpha_ff[i, k] += 1.0e-29 * (np.power(theta, ((n + 1.0)/2.0)) * 
                                                (A_s[n-1]*(wl_2[k]) + B_s[n-1] + C_s[n-1]/wl[k] + 
                                                 D_s[n-1]/(wl_2[k]) + E_s[n-1]/(wl_3[k]) + 
                                                 F_s[n-1]/(wl_4[k]))) * kT
                    
                alpha_ff[i, k] *= 1.0e-10  # Convert from cm^5 / H- ion to m^5 / H- ion
                    
            # Long wavelength fit
            elif (wl[k] >= 0.3645):
                     
                # For each set of fit coefficients
                for n in range(1,7):
                
                    # Compute free-free absorption coefficient at this wavelength and temperature
                    alpha_ff[i, k] += 1.0e-29 * (np.power(theta, ((n + 1.0)/2.0)) * 
                                                (A_l[n-1]*(wl_2[k]) + B_l[n-1] + C_l[n-1]/wl[k] + 
                                                 D_l[n-1]/(wl_2[k]) + E_l[n-1]/(wl_3[k]) + 
                                                 F_l[n-1]/(wl_4[k]))) * kT  # cm^5 / H / e-
                alpha_ff[i, k] *= 1.0e-10  # Convert from cm^5 / H- ion to m^5 / H- ion
    
    return alpha_ff


def get_id_within_node(comm, rank):

    nodename =  MPI.Get_processor_name()
    nodelist = comm.allgather(nodename)

    process_id = len([i for i in nodelist[:rank] if i==nodename]) 

    return process_id


def shared_memory_array(rank, comm, shape):
    ''' 
    Creates a numpy array shared in memory across multiple cores.
    
    Adapted from :
    https://stackoverflow.com/questions/32485122/shared-memory-in-mpi4py
    
    '''
    
    # Create a shared array of size given by product of each dimension
    size = np.prod(shape)
    itemsize = MPI.DOUBLE.Get_size() 

    if (rank == 0): 
        nbytes = size * itemsize   # Array memory allocated for first process
    else:  
        nbytes = 0   # No memory storage on other processes
        
    # On rank 0, create the shared block
    # On other ranks, get a handle to it (known as a window in MPI speak)
    new_comm = MPI.Comm.Split(comm)
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=new_comm) 
 
    # Create a numpy array whose data points to the shared memory
    buf, itemsize = win.Shared_query(0) 
    assert itemsize == MPI.DOUBLE.Get_size() 
    array = np.ndarray(buffer=buf, dtype='d', shape=shape) 
    
    return array


def opacity_tables(rank, comm, wl_model, chemical_species, active_species, 
                   cia_pairs, ff_pairs, bf_species, T_fine, log_P_fine,
                   opacity_database = 'High-T', testing = False):
    ''' 
    Initialisation function to read in and pre-interpolate all opacities.
        
    Inputs:
        
    wl_model => array of wavelength values in spectral model (um)
    Note: many fixed input variables specified in config.py - see import
    
    Outputs:
        
    sigma_stored => molecular and atomic cross sections interpolated to 
                    'pre' P grid, fine T grid, and model wl grid 
    cia_stored => collisionally-induced absorption (CIA) binary cross sections
                    interpolated to fine T grid and model wl grid 
    Rayleigh_stored => Rayleigh scattering cross sections on model wl grid 
    eta_stored => refractive indices on model wl grid at standard conditions
    
    '''
            
    #***** First, initialise the various quantities needed for pre-interpolation *****#
        
    N_species = len(chemical_species)        # Number of chemical species
    N_species_active = len(active_species)   # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)             # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
    N_bf_species = len(bf_species)           # Number of bound-free species included
    N_T_fine = len(T_fine)           # No. of temperatures on fine temperature grid
    N_P_fine = len(log_P_fine)       # No. of pressures on fine pressure grid
    
    # Convert model wavelength grid to wavenumber grid
    nu_model = 1.0e4/wl_model    # Model wavenumber grid (cm^-1)
    nu_model = nu_model[::-1]    # Reverse direction, such that increases with wavenumber
        
    N_nu = len(nu_model)    # Number of wavenumbers on model grid
    N_wl = len(wl_model)    # Number of wavelengths on model grid
        
    # Identify the 'local rank' within each node
   # N_nodes, process_id = get_id_within_node(comm, rank)

    # Split communicator into separate communicators for each node
  #  color = int(rank / N_nodes)

  #  node_comm = comm.Split(color=color)
  #  node_rank = node_comm.Get_rank()

    node_rank = rank
    node_comm = comm

    # Initialise output opacity arrays
    cia_stored = shared_memory_array(node_rank, node_comm, (N_cia_pairs, N_T_fine, N_wl))                      # Collision-induced absorption
    ff_stored = shared_memory_array(node_rank, node_comm, (N_ff_pairs, N_T_fine, N_wl))                        # Free-free
    bf_stored = shared_memory_array(node_rank, node_comm, (N_bf_species, N_wl))                                # Bound-free
    sigma_stored = shared_memory_array(node_rank, node_comm, (N_species_active, N_P_fine, N_T_fine, N_wl))     # Molecular and atomic opacities
    Rayleigh_stored = shared_memory_array(node_rank, node_comm, (N_species, N_wl))                             # Rayleigh scattering
    eta_stored = shared_memory_array(node_rank, node_comm, (N_species, N_wl))                                  # Refractive indices
    
    # When using multiple cores, only the first core needs to handle interpolation
    if (node_rank == 0):
        
        if (rank == 0):
        
            print("Reading in cross sections in opacity sampling mode...")

        # Find the directory where the user downloaded the POSEIDON opacity data
        opacity_path = os.environ.get("POSEIDON_input_data")

        if opacity_path == None:
            raise Exception("POSEIDON cannot locate the opacity input data.\n"
                            "Please set the 'POSEIDON_input_data' variable in " +
                            "your .bashrc or .bash_profile to point to the " +
                            "directory containing the POSEIDON opacity database.")
        
        # If running the automated GitHub tests, don't read the opacity database
        if (testing == True):
            opac_file = ''
            log_P_grid = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2])
            N_P = len(log_P_grid)

        # For all other applications
        else:

            # Open HDF5 files containing molecular + atomic opacities
            if (opacity_database == 'High-T'):        # High T database
                opac_file = h5py.File(opacity_path + 'Opacity_database_0.01cm-1.hdf5', 'r')  
            elif (opacity_database == 'Temperate'):   # Low T database
                opac_file = h5py.File(opacity_path + 'Opacity_database_0.01cm-1_Temperate.hdf5', 'r')

            # Read P grid used in opacity files
            log_P_grid = np.array(opac_file['H2O/log(P)'])   # Units: log10(P/bar) - H2O choice arbitrary, all P grids are the same
            N_P = len(log_P_grid)                            # No. of pressures in opacity files
        
        # Open HDF5 files containing collision-induced absorption (CIA)
        cia_file = h5py.File(opacity_path + 'Opacity_database_cia.hdf5', 'r')
        
        # Initialise array of indices on pre-calculated pressure opacity grid prior to defined atmosphere layer pressures
        x = np.zeros(N_P_fine, dtype=np.int64)
        
        # Weights
        w_P = np.zeros(N_P_fine)
        
        # Useful functions of weights for interpolation
        b1 = np.zeros(shape=(N_P_fine))
        b2 = np.zeros(shape=(N_P_fine))
            
        # Now find closest P indices in opacity grid corresponding to model pressures
        for i in range(N_P_fine):
            
            # If pressure below minimum, do not interpolate
            if (log_P_fine[i] < log_P_grid[0]):
                x[i] = -1      # Special value (1) used in opacity initialiser
                w_P[i] = 0.0
            
            # If pressure above maximum, do not interpolate
            elif (log_P_fine[i] >= log_P_grid[-1]):
                x[i] = -2      # Special value (2) used in opacity initialiser
                w_P[i] = 0.0
            
            else:
                x[i] = prior_index_V2(log_P_fine[i], log_P_grid[0], log_P_grid[-1], N_P)
            
                # Weights - fractional distance along pressure axis of sigma array
                w_P[i] = (log_P_fine[i]-log_P_grid[x[i]])/(log_P_grid[x[i]+1]-log_P_grid[x[i]])     
             
            # Precalculate interpolation pre-factors to reduce computation overhead
            b1[i] = (1.0-w_P[i])
            b2[i] = w_P[i]  
            
        # Find wavenumber indices in arrays of model grid        
        nu_edges = np.append(nu_model[0] - (nu_model[1] - nu_model[0]), nu_model)
        nu_edges = np.append(nu_edges, nu_model[-1] + (nu_model[-1] - nu_model[-2]))
        nu_l = 0.5 * (nu_edges[:-2] + nu_edges[1:-1])
        nu_r = 0.5 * (nu_edges[1:-1] +  nu_edges[2:])
        
        #***** Process collision-induced absorption (CIA) *****#
        
        for q in range(N_cia_pairs):
            
            cia_pair_q = cia_pairs[q]    # Cia pair name
            
            #***** Read in T grid used in this cia file*****#
            T_grid_cia_q = np.array(cia_file[cia_pair_q + '/T'])
            N_T_cia_q = len(T_grid_cia_q)  # Number of temperatures in this grid
            
            # Read in wavenumber array used in this CIA file
            nu_cia_q = np.array(cia_file[cia_pair_q + '/nu'])
        
            # Evaluate temperature interpolation weighting factor
            y_cia_q = np.zeros(N_T_fine, dtype=np.int64)   # Index of T in CIA arrays prior to fine temperature value
            w_T_cia_q = T_interpolation_init(N_T_fine, T_grid_cia_q, T_fine, y_cia_q)   # Weighting factor
            
            # Read in log10(binary cross section) for specified CIA pair
            log_cia_q = np.array(cia_file[cia_pair_q + '/log(cia)']).astype(np.float32) 
            
            # Sample / interpolate / log-average CIA to model P and wl grid
            cia_pre_inp_q = wl_initialise_cia(N_T_cia_q, N_wl, log_cia_q, nu_l, nu_model, 
                                              nu_r, nu_cia_q, N_nu)

            del log_cia_q 
            
            # Interpolate CIA to model fine temperature grid
            cia_stored[q,:,:] = T_interpolate_cia(N_T_fine, N_T_cia_q, N_wl, 
                                                  cia_pre_inp_q, T_grid_cia_q, 
                                                  T_fine, y_cia_q, w_T_cia_q)
    
            del cia_pre_inp_q, nu_cia_q, w_T_cia_q, y_cia_q
            
            if (rank == 0):
                print(cia_pair_q + " done")
            
        #***** Process free-free absorption *****#
        
        # For free-free absorption, can just use fitting functions instead of pre-tabulated opacities
        for q in range(N_ff_pairs):
            
            ff_pair_q = ff_pairs[q]     # ff pair name
            
            # Calculate free-free opacities
            if (ff_pair_q == 'H-ff'):
                ff_stored[q,:,:] = H_minus_free_free(wl_model, T_fine)
            else:
                raise Exception("Unsupported free-free opacity.")
            
            if (rank == 0):
                print(ff_pair_q + " done")
            
        #***** Process bound-free absorption *****#
        
        # For bound-free absorption, can just use fitting functions instead of pre-tabulated opacities
        for q in range(N_bf_species):
            
            bf_species_q = bf_species[q]     # bf species name
            
            # Calculate bound-free opacities
            if (bf_species_q == 'H-bf'):
                bf_stored[q,:] = H_minus_bound_free(wl_model)
            else:
                raise Exception("Unsupported bound-free opacity.")
            
            if (rank == 0):
                print(bf_species_q + " done")
            
        #***** Process molecular and atomic opacities *****#

        if (testing == False):  # The automated tests don't download the full cross sections

            # Load molecular and atomic absorption cross sections
            for q in range(N_species_active):
                        
                species_q = active_species[q]     # Chemical species name
                    
                #***** Read in grids used in this opacity file*****#
                T_grid_q = np.array(opac_file[species_q + '/T'])   
                log_P_grid_q = np.array(opac_file[species_q + '/log(P)'])   # Units: log10(P/bar)
                N_T_q = len(T_grid_q)      # Number of temperatures in this grid
                N_P_q = len(log_P_grid_q)  # Number of pressures in this grid
                    
                # Read in wavenumber array used in this opacity file
                nu_q = np.array(opac_file[species_q + '/nu'])
                    
                # Evaluate temperature interpolation weighting factor
                y_q = np.zeros(N_T_fine, dtype=np.int)  # Index of T in cross section arrays prior to fine temperature value
                w_T_q = T_interpolation_init(N_T_fine, T_grid_q, T_fine, y_q)   # Weighting factor
        
                # Read in log10(cross section) of specified molecule (only need float 32 accuracy for exponents)
                log_sigma_q = np.array(opac_file[species_q + '/log(sigma)']).astype(np.float32)
                    
                # Pre-interpolate cross section to model (P, wl) grid 
                sigma_pre_inp_q = P_interpolate_wl_initialise_sigma(N_P_fine, N_T_q, N_P_q, 
                                                                    N_wl, log_sigma_q, x, nu_l,
                                                                    nu_model, nu_r, b1, b2, nu_q, 
                                                                    N_nu)

                del log_sigma_q   # Clear raw cross section to free memory
                
                # Interpolate cross section to fine temperature grid
                sigma_stored[q,:,:,:] = T_interpolate_sigma(N_P_fine, N_T_fine, N_T_q, 
                                                            N_wl, sigma_pre_inp_q, T_grid_q, 
                                                            T_fine, y_q, w_T_q)
                        
                del sigma_pre_inp_q, nu_q, w_T_q, y_q  
                
                if (rank == 0):
                    print(species_q + " done")
            
        #***** Process Rayleigh scattering cross sections *****#
        
        for q in range(N_species):
            
            species_q = chemical_species[q]     # Chemical species name
            
            # Compute the Rayleigh scattering cross section and refractive index
            Rayleigh_stored[q,:], eta_stored[q,:] = Rayleigh_cross_section(wl_model, species_q)
        
        # Clear up storage
        del nu_l, nu_r, nu_model
        
        cia_file.close()

        if (testing == False):
            opac_file.close()
        
    # Force secondary processors to wait for the primary to finish interpolating cross sections
    node_comm.Barrier()

    if (rank == 0): 
        print("Opacity pre-interpolation complete.")
            
    return sigma_stored, cia_stored, Rayleigh_stored, eta_stored, ff_stored, bf_stored


@jit(nopython = True)
def extinction(chemical_species, active_species, cia_pairs, ff_pairs, bf_species,
               n, T, P, wl, X, X_active, X_cia, X_ff, X_bf, a, gamma, P_cloud, 
               kappa_cloud_0, sigma_stored, cia_stored, Rayleigh_stored, ff_stored, 
               bf_stored, enable_haze, enable_deck, enable_surface, N_sectors, 
               N_zones, T_fine, log_P_fine, P_surf, P_deep = 1000.0):                          # DOES P_DEEP SOLVE BD PROBLEM?!
    
    ''' Main function to evaluate extinction coefficients for molecules / atoms,
        Rayleigh scattering, hazes, and clouds for parameter combination
        chosen in retrieval step.
        
        Takes in cross sections pre-interpolated to 'fine' P and T grids
        before retrieval run (so no interpolation is required at each step).
        Instead, for each atmospheric layer the extinction coefficient
        is simply kappa = n * sigma[log_P_nearest, T_nearest, wl], where the
        'nearest' values are the closest P_fine, T_fine points to the
        actual P, T values in each layer. This results in a large speed gain.
        
        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), terminator
        sector, and wavelength.
    
    '''
    
    # Store length variables for mixing ratio arrays 
    N_species = len(chemical_species)        # Number of chemical species
    N_species_active = len(active_species)   # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)             # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
    N_bf_species = len(bf_species)           # Number of bound-free species included
    
    N_wl = len(wl)     # Number of wavelengths on model grid
    N_layers = len(P)  # Number of layers
    
    # Define extinction coefficient arrays
    kappa_clear = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    
    # Fine temperature grid (for pre-interpolating opacities)    
    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)
    
    # Find index of deep pressure below which atmosphere is opaque
    i_bot = np.argmin(np.abs(P - P_deep))
    
    # If haze is enabled in this model
    if (enable_haze == 1):
        
        # Generalised scattering slope for haze
        slope = np.power((wl/0.35), gamma)    # Reference wavelength at 0.35 um
        
    # For each terminator sector (terminator plane)
    for j in range(N_sectors):
            
        # For each terminator zone (along day-night transition)
        for k in range(N_zones):
            
            # For each layer, find closest pre-computed cross section to P_fine, T_fine
            for i in range(i_bot,N_layers):
                
                n_level = n[i,j,k]
                
                # Find closest index in fine temperature array to given layer temperature
                idx_T_fine = closest_index(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
                idx_P_fine = closest_index(np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
                
                # For each collisionally-induced absorption (CIA) pair
                for q in range(N_cia_pairs): 
                    
                    n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                    n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                    n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_n_cia * cia_stored[q, idx_T_fine, l]
                        
                # For each free-free absorption pair
                for q in range(N_ff_pairs): 
                    
                    n_ff_1 = n_level*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
                    n_ff_2 = n_level*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
                    n_n_ff = n_ff_1*n_ff_2             # Product of number densities of ff pair
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_n_ff * ff_stored[q, idx_T_fine, l]
                        
                # For each source of bound-free absorption (photodissociation)
                for q in range(N_bf_species): 
                    
                    n_q = n_level*X_bf[q,i,j,k]   # Number density of dissociating species
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * bf_stored[q,l]
                
                # For each molecular / atomic species with active absorption features
                for q in range(N_species_active): 
                    
                    n_q = n_level*X_active[q,i,j,k]   # Number density of this active species
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]
                    
                # For each molecular / atomic species
                for q in range(N_species):  
                    
                    n_q = n_level*X[q,i,j,k]   # Number density of given species
                    
                    # For each wavelength
                    for l in range(N_wl):
                                
                        # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * Rayleigh_stored[q,l]
        
            # If haze is enabled in this model  
            if (enable_haze == 1):
                
                # For each layer
                for i in range(i_bot,N_layers):
                    
                    haze_amp = (n[i,j,k] * a * 5.31e-31)   # Final factor is H2 Rayleigh scattering cross section at 350 nm
                    
                    # For each wavelength
                    for l in range(N_wl):
                    
                        # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                        kappa_cloud[i,j,k,l] += haze_amp * slope[l]
                        
            # If a cloud deck is enabled in this model
            if (enable_deck == 1):
                
                # Set extinction inside cloud deck
                kappa_cloud[(P > P_cloud),j,k,:] += kappa_cloud_0

            # If a surface is enabled in this model
            if (enable_surface == 1):

                # Set extinction to infinity below surface
                kappa_clear[(P > P_surf),j,k,:] = 1.0e250
            
    return kappa_clear, kappa_cloud


@cuda.jit
def extinction_GPU(kappa_clear, kappa_cloud, i_bot, N_species, N_species_active, 
                   N_cia_pairs, N_ff_pairs, N_bf_species, n, T, P, wl, X, X_active, 
                   X_cia, X_ff, X_bf, a, gamma, P_cloud, kappa_cloud_0, sigma_stored, 
                   cia_stored, Rayleigh_stored, ff_stored, bf_stored, enable_haze,
                   enable_deck, enable_surface, N_sectors, N_zones, T_fine, 
                   log_P_fine, P_surf, P_deep = 1000.0):               
    
    ''' Main function to evaluate extinction coefficients for molecules / atoms,
        Rayleigh scattering, hazes, and clouds for parameter combination
        chosen in retrieval step.
        
        Takes in cross sections pre-interpolated to 'fine' P and T grids
        before retrieval run (so no interpolation is required at each step).
        Instead, for each atmospheric layer the extinction coefficient
        is simply kappa = n * sigma[log_P_nearest, T_nearest, wl], where the
        'nearest' values are the closest P_fine, T_fine points to the
        actual P, T values in each layer. This results in a large speed gain.
        
        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), terminator
        sector, and wavelength.
    
    '''
    
    thread = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    N_wl = len(wl)     # Number of wavelengths on model grid
    N_layers = len(P)  # Number of layers
    
    # Fine temperature grid (for pre-interpolating opacities)    
    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)
            
    # For each terminator sector (terminator plane)
    for j in range(N_sectors):
            
        # For each terminator zone (along day-night transition)
        for k in range(N_zones):
            
            # For each layer, find closest pre-computed cross section to P_fine, T_fine
            for i in range(i_bot, N_layers):
                
                n_level = n[i,j,k]
                
                # Find closest index in fine temperature array to given layer temperature
                idx_T_fine = closest_index_GPU(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
                idx_P_fine = closest_index_GPU(math.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
                
                # For each collisionally-induced absorption (CIA) pair
                for q in range(N_cia_pairs): 
                    
                    n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                    n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                    n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair
                    
                    # For each wavelength
                    for l in range(thread, N_wl, stride):
                        
                        # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_n_cia * cia_stored[q, idx_T_fine, l]
                        
                # For each free-free absorption pair
                for q in range(N_ff_pairs): 
                    
                    n_ff_1 = n_level*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
                    n_ff_2 = n_level*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
                    n_n_ff = n_ff_1*n_ff_2             # Product of number densities of ff pair
                    
                    # For each wavelength
                    for l in range(thread, N_wl, stride):
                        
                        # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_n_ff * ff_stored[q, idx_T_fine, l]
                        
                # For each source of bound-free absorption (photodissociation)
                for q in range(N_bf_species): 
                    
                    n_q = n_level*X_bf[q,i,j,k]   # Number density of dissociating species
                    
                    # For each wavelength
                    for l in range(thread, N_wl, stride):
                        
                        # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * bf_stored[q,l]
                
                # For each molecular / atomic species with active absorption features
                for q in range(N_species_active): 
                    
                    n_q = n_level*X_active[q,i,j,k]   # Number density of this active species
                    
                    # For each wavelength
                    for l in range(thread, N_wl, stride):
                        
                        # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]
                    
                # For each molecular / atomic species
                for q in range(N_species):  
                    
                    n_q = n_level*X[q,i,j,k]   # Number density of given species
                    
                    # For each wavelength
                    for l in range(thread, N_wl, stride):
                                
                        # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * Rayleigh_stored[q,l]
        
            # If haze is enabled in this model  
            if (enable_haze == 1):
                
                # For each layer
                for i in range(i_bot, N_layers):
                    
                    haze_amp = (n[i,j,k] * a * 5.31e-31)   # Final factor is H2 Rayleigh scattering cross section at 350 nm
                    
                    # For each wavelength
                    for l in range(thread, N_wl, stride):

                        # Generalised scattering slope for haze
                        slope = math.pow(wl[l]/0.35, gamma)

                        # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                        kappa_cloud[i,j,k,l] += haze_amp * slope
                        
            # If a cloud deck is enabled in this model
            if (enable_deck == 1):
                
                # For each wavelength
                for l in range(thread, N_wl, stride):

                    # For each layer
                    for i in range(i_bot, N_layers):

                        # Set extinction inside cloud deck
                        if P[i] > P_cloud:
                            kappa_cloud[i,j,k,l] += kappa_cloud_0

            # If a surface is enabled in this model
            if (enable_surface == 1):

                # For each wavelength
                for l in range(thread, N_wl, stride):

                    # For each layer
                    for i in range(i_bot, N_layers):

                        # Set extinction to infinity below surface
                        if P[i] > P_surf:
                            kappa_clear[i,j,k,l] = 1.0e250


#***** Special optimised functions for line-by-line case *****#

@jit(nopython=True)
def interpolate_cia_LBL(P, log_cia, nu_model, nu_cia, T, T_grid_cia, N_T_cia, 
                        N_wl, N_nu, y, w_T):
    
    ''' Interpolates a collisionally-induced absorption (CIA) binary cross 
        section onto the T value in each layer of the model atmosphere.
        Special function optimised for line-by-line case.
        
    '''
   
    N_layers = len(P)

    cia_inp = np.zeros(shape=(N_layers, N_wl))   # Initialise output CIA cross section
    
    N_nu_cia = len(nu_cia)   # Number of wavenumber points in CIA array

    for i in range(N_layers):    # Loop over layer temperatures
    
        T_i = T[i]               # Temperature to interpolate to
        T1 = T_grid_cia[y[i]]    # Closest lower temperature on CIA opacity grid 
        T2 = T_grid_cia[y[i]+1]  # Closest higher temperature on CIA opacity grid 
        
        for k in range(N_nu):  # Loop over wavenumbers
            
            # Find closest index on CIA wavenumber grid to each model wavenumber
            z = closest_index(nu_model[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            
            # If this wavenumber falls out of range of opacity grid, set opacity to zero
            if ((z == 0) or (z == (N_nu_cia-1))):
                cia_inp[i, ((N_wl-1)-k)] = 0.0
                
            else:  # Otherwise, proceed with interpolation
                
                # If layer T below min value on CIA grid (200 K), set CIA to value at min T on opac grid
                if (y[i] == -1):
                    cia_inp[i, ((N_wl-1)-k)] = 10 ** (log_cia[0, z])
                    
                # If layer T above max value (3500 K), set CIA to value at max T on opac grid
                elif (y[i] == -2):
                    cia_inp[i, ((N_wl-1)-k)] = 10 ** (log_cia[(N_T_cia-1), z])
                
                # Interpolate CIA to temperature in this layer
                else: 
                    cia_reduced = 10 ** (log_cia[y[i]:y[i]+2, z])
                    cia_1, cia_2 = cia_reduced[0], cia_reduced[1]    # CIA(T1)[j,z], CIA(T2)[j,z]
                        
                    cia_inp[i, ((N_wl-1)-k)] =  (np.power(cia_1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                 np.power(cia_2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
            
    return cia_inp


@jit(nopython=True)
def interpolate_sigma_LBL(log_sigma, nu_model, nu_opac, P, T, log_P_grid, T_grid,
                          N_T, N_P, N_wl, N_nu, y, w_T):
    
    ''' Interpolates a cross section onto the (P,T) values in each layer of
        the model atmosphere. Special function optimised for line-by-line case.
        
    '''

    N_layers = len(P)
   
    sigma_inp = np.zeros(shape=(N_layers, N_wl))   # Initialise output cross section
    
    #***** Firstly, find the (exact) indices in opacity wavenumber grid corresponding to edges of model wavenumber grid *****#

    nu_opac_min = nu_opac[0]
    nu_opac_max = nu_opac[-1]
    N_nu_opac = len(nu_opac)   # Number of wavenumber points in opacity array
    
    nu_model_min = nu_model[0]   # Minimum model wavenumber
    nu_model_max = nu_model[-1]  # Maximum model wavenumber
    
    z_grid_min = closest_index(nu_model_min, nu_opac[0], nu_opac[-1], N_nu_opac)  # Minimum opacity grid index to consider
    z_grid_max = closest_index(nu_model_max, nu_opac[0], nu_opac[-1], N_nu_opac)  # Maximum opacity grid index to consider
    
    # Restrict opacity array to wavenumbers within model grid range
    reduced_log_sigma = log_sigma[:,:,z_grid_min:z_grid_max+1]
    
    #***** Secondly, find pressure interpolation weighting factors *****#
    log_P = np.log10(P)  # Log of model pressure grid
    
    # Array of indices on opacity pressure opacity grid prior to model atmosphere layer pressures
    x = np.zeros(N_layers).astype(np.int64) 
    
    w_P = np.zeros(N_layers)  # Pressure weights
    
    # Useful functions of weights for interpolation
    b1 = np.zeros(shape=(N_layers))
    b2 = np.zeros(shape=(N_layers))
            
    # Find closest P indices in opacity grid corresponding to model layer pressures
    for i in range(N_layers):
        
        # If pressure below minimum, do not interpolate
        if (log_P[i] < log_P_grid[0]):
            x[i] = -1      # Special value (1) used in opacity initialiser
            w_P[i] = 0.0
        
        # If pressure above maximum, do not interpolate
        elif (log_P[i] >= log_P_grid[-1]):
            x[i] = -2      # Special value (2) used in opacity initialiser
            w_P[i] = 0.0
        
        else:
            x[i] = prior_index_V2(log_P[i], log_P_grid[0], log_P_grid[-1], N_P)
        
            # Weights - fractional distance along pressure axis of sigma array
            w_P[i] = (log_P[i]-log_P_grid[x[i]])/(log_P_grid[x[i]+1]-log_P_grid[x[i]])     
         
        # Precalculate interpolation pre-factors to reduce computation overhead
        b1[i] = (1.0-w_P[i])
        b2[i] = w_P[i] 
        
    # Note: temperature interpolation indices and weights passed through function arguments

    # Begin interpolation procedure
    for i in range(N_layers):   # Loop over model layers
        
        T_i = T[i]           # Layer temperature to interpolate to
        T1 = T_grid[y[i]]    # Closest lower temperature on opacity grid 
        T2 = T_grid[y[i]+1]  # Closest higher temperature on opacity grid 
        
        for k in range(N_nu):   # Loop over model wavenumbers
            
            # Extract wavenumber of this model point
            nu_model_k = nu_model[k]
            
            # If this wavenumber falls out of range of opacity grid, set opacity to zero
            if ((nu_model_k < nu_opac_min) or (nu_model_k > nu_opac_max)):
                sigma_inp[i, ((N_wl-1)-k)] = 0.0
                
            else:  # Otherwise, proceed with interpolation

                # Find rectangle of stored opacity points located at [log_P1, log_P2, T1, T2, ]
                log_sigma_PT_rectangle = reduced_log_sigma[x[i]:x[i]+2, y[i]:y[i]+2, k]

                # Pressure interpolation is handled first, followed by temperature interpolation
                # First, check for off-grid special cases
                
                # If layer P below minimum on opacity grid (1.0e-6 bar), set value to edge case
                if (x[i] == -1):      
                    
                    # If layer T also below minimum on opacity grid (100 K), set value to edge case
                    if (y[i] == -1):
                        sigma_inp[i, ((N_wl-1)-k)] = 10 ** (reduced_log_sigma[0, 0, k])  # No interpolation needed
                     
                    # If layer T above maximum on opacity grid (3500 K), set value to edge case
                    elif (y[i] == -2):
                        sigma_inp[i, ((N_wl-1)-k)] = 10 ** (reduced_log_sigma[0, (N_T-1), k])  # No interpolation needed
                        
                    # If desired temperature is on opacity grid, set T1 and T2 values to those at min P on grid
                    else:
                        sig_T1 = 10 ** (reduced_log_sigma[0, y[i], k])     
                        sig_T2 = 10 ** (reduced_log_sigma[0, y[i]+1, k])
                        
                        # Only need to interpolate over temperature interpolate cross section to layer temperature                    
                        sigma_inp[i, ((N_wl-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                       np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                    
                # If layer P above maximum on opacity grid (100 bar), set value to edge case
                elif (x[i] == -2):
                    
                    # If layer T below minimum on opacity grid (100 K), set value to edge case
                    if (y[i] == -1):
                        sigma_inp[i, ((N_wl-1)-k)] = 10 ** (reduced_log_sigma[(N_P-1), 0, k])  # No interpolation needed
                     
                    # If layer T also above maximum on opacity grid (3500 K), set value to edge case
                    elif (y[i] == -2):
                        sigma_inp[i, ((N_wl-1)-k)] = 10 ** (reduced_log_sigma[(N_P-1), (N_T-1), k])  # No interpolation needed
                        
                    # If desired temperature is on opacity grid, set T1 and T2 values to those at maximum P on grid             
                    else:
                        sig_T1 = 10 ** (reduced_log_sigma[(N_P-1), y[i], k])
                        sig_T2 = 10 ** (reduced_log_sigma[(N_P-1), y[i]+1, k])
                        
                        # Now interpolate cross section to layer temperature                    
                        sigma_inp[i, ((N_wl-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                       np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
            
                # If both desired P and T are on opacity grid (should be true in most cases!)
                else:
                    
                    # Interpolate log(cross section) in log(P), then power to get interpolated values at T1 and T2
                    sig_T1 =  10 ** (b1[i]*(log_sigma_PT_rectangle[0,0]) +       # Cross section at T1
                                     b2[i]*(log_sigma_PT_rectangle[1,0]))
                    sig_T2 =  10 ** (b1[i]*(log_sigma_PT_rectangle[0,1]) +       # Cross section at T2
                                     b2[i]*(log_sigma_PT_rectangle[1,1]))
        
                    # Now interpolate cross section to layer temperature                    
                    sigma_inp[i, ((N_wl-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                   np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
            
    
    return sigma_inp


@jit
def store_Rayleigh_eta_LBL(wl_model, chemical_species):
    
    ''' In line-by-line case, output refractive index array (eta) and
        Rayleigh scattering separately from main extinction calculation.
        
        This is simply to pass eta to profiles.py, where it is needed.
    
    '''
    
    N_wl = len(wl_model)                 # Number of wavelengths on model grid
    N_species = len(chemical_species)    # Number of chemical species
        
    #***** Process Rayleigh scattering cross sections *****#
    
    # Initialise pre-computed Rayleigh cross section array for all species
    Rayleigh_stored = np.zeros(shape=(N_species, N_wl))
    
    # Initialise pre-computed refractive index array for all species
    eta_stored = np.zeros(shape=(N_species, N_wl))    
    
    for q in range(N_species):
        
        species_q = chemical_species[q]     # Given chemical species name
        
        # Compute Rayleigh scattering cross section and refractive index on model wavelength array
        Rayleigh_stored[q,:], eta_stored[q,:] = Rayleigh_cross_section(wl_model, species_q)
    
    return Rayleigh_stored, eta_stored


@jit(nopython=True)
def compute_kappa_LBL(j, k, wl_model, X, X_active, X_cia, X_ff, X_bf, n, P,
                      a, gamma, P_cloud, kappa_cloud_0, N_species, N_species_active,
                      N_cia_pairs, N_ff_pairs, N_bf_species, sigma_interp,
                      cia_interp, Rayleigh_stored, ff_stored, bf_stored, 
                      enable_haze, enable_deck, enable_surface, kappa_clear, 
                      kappa_cloud, P_surf, disable_continuum):
    
    ''' Computes extinction coefficients for given sector and zone. 
        Special function optimised for line-by-line case.
    
    '''
    
    N_wl = len(wl_model)  # Number of wavelengths on model grid
    N_layers = len(P)     # Number of layers
    
    # If haze is enabled in this model
    if (enable_haze == 1):
        
        # Generalised scattering slope for haze
        slope = np.power((wl_model/0.35), gamma)      # Reference wavelength at 0.35 um
            
    # For each layer
    for i in range(N_layers):
            
        if (disable_continuum == False):

            # For each collisionally-induced absorption (CIA) pair
            for q in range(N_cia_pairs): 
                    
                n_cia_1 = n[i,j,k]*X_cia[0,q,i,j,k]   # Number density of first CIA species in pair
                n_cia_2 = n[i,j,k]*X_cia[1,q,i,j,k]   # Number density of second CIA species in pair
                n_n_cia = n_cia_1*n_cia_2             # Product of number densities of CIA pair
                    
                # For each wavelength
                for l in range(N_wl):
                        
                    # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                    kappa_clear[i,j,k,l] += n_n_cia * cia_interp[q,i,l]

            # For each molecular / atomic species
            for q in range(N_species):  
                    
                n_q = n[i,j,k]*X[q,i,j,k]       # Number density of given species
                    
                # For each wavelength
                for l in range(N_wl):
                                
                    # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                    kappa_clear[i,j,k,l] += n_q * Rayleigh_stored[q,l]
                
        # For each free-free absorption pair
        for q in range(N_ff_pairs): 
                
            n_ff_1 = n[i,j,k]*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
            n_ff_2 = n[i,j,k]*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
            n_n_ff = n_ff_1*n_ff_2            # Product of number densities of ff pair
                
            # For each wavelength
            for l in range(N_wl):
                    
                # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                kappa_clear[i,j,k,l] += n_n_ff * ff_stored[q,i,l]
                    
        # For each source of bound-free absorption (photodissociation)
        for q in range(N_bf_species): 
                
            n_q = n[i,j,k]*X_bf[q,i,j,k]   # Number density of dissociating species
                
            # For each wavelength
            for l in range(N_wl):
                    
                # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                kappa_clear[i,j,k,l] += n_q * bf_stored[q,l]
            
        # For each molecular / atomic species with active absorption features
        for q in range(N_species_active): 
                
            n_q = n[i,j,k]*X_active[q,i,j,k]   # Number density of this active species
                
            # For each wavelength
            for l in range(N_wl):
                    
                # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                kappa_clear[i,j,k,l] += n_q * sigma_interp[q,i,l]
    
    # If haze is enabled in this model  
    if (enable_haze == 1):
            
        # For each layer
        for i in range(N_layers):
                
            haze_amp = (n[i,j,k] * a * 5.31e-31)   # Final factor is Rayleigh scattering cross section at 350 nm
                
            # For each wavelength
            for l in range(N_wl):
                
                # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                kappa_cloud[i,j,k,l] += haze_amp * slope[l]

        # If a cloud deck is enabled in this model
        if (enable_deck == 1):
            
            # Set extinction inside cloud deck
            kappa_cloud[(P > P_cloud),j,k,:] += kappa_cloud_0

        # If a surface is enabled in this model
        if (enable_surface == 1):

            # Set extinction to infinity below surface
            kappa_clear[(P > P_surf),j,k,:] = 1.0e250
        
    
def extinction_LBL(chemical_species, active_species, cia_pairs, ff_pairs, 
                   bf_species, n, T, P, wl_model, X, X_active, X_cia, X_ff, X_bf, 
                   a, gamma, P_cloud, kappa_cloud_0, Rayleigh_stored, enable_haze, 
                   enable_deck, enable_surface, N_sectors, N_zones, P_surf,
                   opacity_database = 'High-T', disable_continuum = False,
                   suppress_print = False):
    
    ''' Evaluate extinction coefficients for molecules / atoms, Rayleigh 
        scattering, hazes, and clouds. Special function optimised for 
        line-by-line case.
        
        Here, kappa = n[layer] * sigma[P_layer, T_layer, wl], where
        the cross sections are all evaluated on their native (line-by-line) 
        wavenumber resolution with interpolation to the T and P in each layer.
        
        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), atmospheric
        sector, atmospheric zone, and wavelength.
    
    '''
    
    if (suppress_print == False):
        print("Reading in cross sections in line-by-line mode...")
    
    #***** First, initialise the various quantities needed *****#

    N_species = len(chemical_species)        # Number of chemical species
    N_species_active = len(active_species)   # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)             # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
    N_bf_species = len(bf_species)           # Number of bound-free species included
    N_layers = len(P)                        # Number of layers
    
    # Convert model wavelength grid to wavenumber grid
    nu_model = 1.0e4/wl_model    # Model wavenumber grid (cm^-1)
    nu_model = nu_model[::-1]     # Reverse direction to increase with wavenumber
    
    N_nu = len(nu_model)    # Number of wavenumbers on model grid
    N_wl = len(wl_model)    # Number of wavelengths on model grid
    
    # Define extinction coefficient arrays
    kappa_clear = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))

    # Find the directory where the user downloaded the POSEIDON opacity data
    opacity_path = os.environ.get("POSEIDON_input_data")

    if opacity_path == None:
        raise Exception("POSEIDON cannot locate the opacity input data.\n"
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "directory containing the POSEIDON opacity database.")
    
    # Open HDF5 files containing molecular + atomic opacities
    if (opacity_database == 'High-T'):        # High T database
        opac_file = h5py.File(opacity_path + 'Opacity_database_0.01cm-1.hdf5', 'r')  
    elif (opacity_database == 'Temperate'):   # Low T database
        opac_file = h5py.File(opacity_path + 'Opacity_database_0.01cm-1_Temperate.hdf5', 'r')
    
    # Open HDF5 files containing collision-induced absorption (CIA)
    cia_file = h5py.File(opacity_path + 'Opacity_database_cia.hdf5', 'r')

    #***** Process collisionally Induced Absorption (CIA) *****#
     
    # For each terminator sector
    for j in range(N_sectors):
        
        for k in range(N_zones):
            
            # Initialise cia opacity array interpolated to temperature in each model layer
            cia_interp = np.zeros(shape=(N_cia_pairs, N_layers, N_wl))
            
            for q in range(N_cia_pairs):
                
                cia_pair_q = cia_pairs[q]     # Cia pair name
                
                #***** Read in T grid used in this CIA file*****#
                T_grid_cia_q = np.array(cia_file[cia_pair_q + '/T'])
                
                # Read in wavenumber array used in this CIA file
                nu_cia_q = np.array(cia_file[cia_pair_q + '/nu'])
                
                N_T_cia_q = len(T_grid_cia_q)  # Number of temperatures in this grid
            
                # Evaluate temperature interpolation weighting factor
                y_cia_q = np.zeros(N_layers, dtype=np.int)   # Layer idex in cia arrays prior to layer temperature value
                w_T_cia_q = T_interpolation_init(N_layers, T_grid_cia_q, T[:,j,k], y_cia_q)   # Weighting factor
                
                # Read in log10(binary cross section) for specified CIA pair
                log_cia_q = np.array(cia_file[cia_pair_q + '/log(cia)']).astype(np.float32)   
                
                # Interpolate CIA to temperature in each atmospheric layer
                cia_interp[q,:,:] = interpolate_cia_LBL(P, log_cia_q, nu_model, nu_cia_q, 
                                                        T[:,j,k], T_grid_cia_q, N_T_cia_q, 
                                                        N_wl, N_nu, y_cia_q, w_T_cia_q)
                
                del log_cia_q, nu_cia_q, w_T_cia_q, y_cia_q  # Clear raw cross section to free up memory
                
                if (suppress_print == False):
                    print(cia_pair_q + " done")
                
            cia_file.close()
            
            #***** Process free-free absorption *****#
             
            # Initialise ff opacity array interpolated to model wavelengths + fine temperature grid
            ff_stored = np.zeros(shape=(N_ff_pairs, N_layers, N_wl))
            
            # For free-free absorption, can just use fitting functions instead of pre-tabulated opacities
            for q in range(N_ff_pairs):
                
                ff_pair_q = ff_pairs[q]     # ff pair name
                
                # Calculate free-free opacities
                if (ff_pair_q == 'H-ff'):
                    ff_stored[q,:,:] = H_minus_free_free(wl_model, T[:,j,k])
                else:
                    raise Exception("Unsupported free-free opacity.")
                
                if (suppress_print == False):
                    print(ff_pair_q + " done")
                
            #***** Process bound-free absorption *****#
             
            # Initialise bf opacity array interpolated to model wavelengths
            bf_stored = np.zeros(shape=(N_bf_species, N_wl))
            
            # For bound-free absorption, can just use fitting functions instead of pre-tabulated opacities
            for q in range(N_bf_species):
                
                bf_species_q = bf_species[q]     # bf species name
                
                # Calculate bound-free opacities
                if (bf_species_q == 'H-bf'):
                    bf_stored[q,:] = H_minus_bound_free(wl_model)
                else:
                    raise Exception("Unsupported bound-free opacity.")
                
                if (suppress_print == False):
                    print(bf_species_q + " done")
                
            #***** Process molecular and atomic opacities *****#
            
            # Initialise molecular and atomic opacity array, interpolated to temperatures and pressures in each model layer
            sigma_interp = np.zeros(shape=(N_species_active, N_layers, N_wl))
            
            # Load molecular and atomic absorption cross sections
            for q in range(N_species_active):
                    
                species_q = active_species[q]     # Molecule name (defined in config.py)
                
                #***** Read in grids used in this opacity file*****#
                T_grid_q = np.array(opac_file[species_q + '/T'])   
                log_P_grid_q = np.array(opac_file[species_q + '/log(P)'])   # Units: log10(P/bar)!
                
                # Read in wavenumber array used in this opacity file
                nu_q = np.array(opac_file[species_q + '/nu'])
                
                N_T_q = len(T_grid_q)      # Number of temperatures in this grid
                N_P_q = len(log_P_grid_q)  # Number of pressures in this grid
                
                # Evaluate temperature interpolation weighting factor
                y_q = np.zeros(N_layers, dtype=np.int)  # Layer index in cross section arrays prior to layer temperature value
                w_T_q = T_interpolation_init(N_layers, T_grid_q, T[:,j,k], y_q)   # Weighting factor
                
                # Read in log10(cross section) of specified molecule (only need float 32 accuracy for exponents)
                log_sigma_q = np.array(opac_file[species_q + '/log(sigma)']).astype(np.float32) 
                    
                # Interpolate cross section to (P,T) in each atmospheric layer
                sigma_interp[q,:,:] = interpolate_sigma_LBL(log_sigma_q, nu_model, nu_q, P, 
                                                            T[:,j,k], log_P_grid_q, T_grid_q, 
                                                            N_T_q, N_P_q, N_wl, N_nu, y_q, w_T_q)
                
                del log_sigma_q, nu_q, w_T_q, y_q   # Clear raw cross section to free up memory
                
                if (suppress_print == False):
                    print(species_q + " done")
                
            opac_file.close()
            
            #***** Now compute extinction coefficients *****#
            
            # Populate extinction coefficients for this sector (using optimised compiled function)
            compute_kappa_LBL(j, k, wl_model, X, X_active, X_cia, X_ff, X_bf, n, 
                              P, a, gamma, P_cloud, kappa_cloud_0, N_species, 
                              N_species_active, N_cia_pairs, N_ff_pairs, N_bf_species, 
                              sigma_interp, cia_interp, Rayleigh_stored, ff_stored, 
                              bf_stored, enable_haze, enable_deck, enable_surface,
                              kappa_clear, kappa_cloud, P_surf, disable_continuum)

    if (suppress_print == False):
        print("Finished producing extinction coefficients")
            
    return kappa_clear, kappa_cloud

# Elijah New Functions 
    
@jit(nopython = True)
def extinction_spectrum_contribution(chemical_species, active_species, cia_pairs, ff_pairs, bf_species,
                            n, T, P, wl, X, X_active, X_cia, X_ff, X_bf, a, gamma, P_cloud, 
                            kappa_cloud_0, sigma_stored, cia_stored, Rayleigh_stored, ff_stored, 
                            bf_stored, enable_haze, enable_deck, enable_surface, N_sectors, 
                            N_zones, T_fine, log_P_fine, P_surf, P_deep = 1000.0,
                            contribution_molecule_list = [],
                            bulk = False):                          # DOES P_DEEP SOLVE BD PROBLEM?!
    
    ''' Main function to evaluate extinction coefficients for molecules / atoms,
        Rayleigh scattering, hazes, and clouds for parameter combination
        chosen in retrieval step.
        
        Takes in cross sections pre-interpolated to 'fine' P and T grids
        before retrieval run (so no interpolation is required at each step).
        Instead, for each atmospheric layer the extinction coefficient
        is simply kappa = n * sigma[log_P_nearest, T_nearest, wl], where the
        'nearest' values are the closest P_fine, T_fine points to the
        actual P, T values in each layer. This results in a large speed gain.
        
        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), terminator
        sector, and wavelength.

        This is to turn off every opacity except one molecule.
        For now, I am ignoring CIA 
    
    '''
    
    # Store length variables for mixing ratio arrays 
    N_species = len(chemical_species)        # Number of chemical species
    N_species_active = len(active_species)   # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)             # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
    N_bf_species = len(bf_species)           # Number of bound-free species included
    
    # Set up all the indices for contribution functions (keeps bulk species on)
    N_bulk_species = N_species - N_species_active
    bulk_species_indices = range(N_bulk_species)

    # Find the name of the bulk species to check and see if they are in the cia list 
    bulk_species_names = chemical_species[:N_bulk_species]

    # Find the bulk species indices for cia 
    # For this to occur, both need to be a bulk species
    bulk_cia_indices = []
    for i in range(len(cia_pairs)):
        pair_1, pair_2 = cia_pairs[i].split('-')
        pair_1_bool = False
        pair_2_bool = False 
        for j in bulk_species_names:
            if pair_1 == j:
                pair_1_bool = True
            if pair_2 == j:
                pair_2_bool = True
        
        if pair_1_bool == True and pair_2_bool == True:
            bulk_cia_indices.append(i)

    if bulk == False:
        for i in range(len(chemical_species)):
            if contribution_molecule_list[0] == chemical_species[i]:
                contribution_molecule_species_index = i

        for i in range(len(active_species)):
            if contribution_molecule_list[0] == active_species[i]:
                contribution_molecule_active_index = i

        # Now I need to find the cia_pair indices 
        cia_indices = []
        for i in range(len(cia_pairs)):
            pair_1, pair_2 = cia_pairs[i].split('-')
            if contribution_molecule_list[0] == pair_1 or contribution_molecule_list[0] == pair_2:
                cia_indices.append(i)

    
    # Layers and wavelengths 
    N_wl = len(wl)     # Number of wavelengths on model grid
    N_layers = len(P)  # Number of layers
    
    # Define extinction coefficient arrays
    kappa_clear = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    
    # Fine temperature grid (for pre-interpolating opacities)    
    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)
    
    # Find index of deep pressure below which atmosphere is opaque
    i_bot = np.argmin(np.abs(P - P_deep))
    
    # If haze is enabled in this model
    if (enable_haze == 1):
        
        # Generalised scattering slope for haze
        slope = np.power((wl/0.35), gamma)    # Reference wavelength at 0.35 um
        
    # For each terminator sector (terminator plane)
    for j in range(N_sectors):
            
        # For each terminator zone (along day-night transition)
        for k in range(N_zones):
            
            # For each layer, find closest pre-computed cross section to P_fine, T_fine
            for i in range(i_bot,N_layers):
                
                n_level = n[i,j,k]
                
                # Find closest index in fine temperature array to given layer temperature
                idx_T_fine = closest_index(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
                idx_P_fine = closest_index(np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
                
                # For each collisionally-induced absorption (CIA) pair
                # Need to fix this for non bulk species absorption 
                for q in range(N_cia_pairs): 
                    
                    if bulk == False:
                        if q in bulk_cia_indices or q in cia_indices:
                            n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                            n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                            n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair

                    if bulk == True:
                        if q in bulk_cia_indices:
                            n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                            n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                            n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair
                        
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_n_cia * cia_stored[q, idx_T_fine, l]
                        
                # For each free-free absorption pair
                for q in range(N_ff_pairs): 
                    
                    n_ff_1 = n_level*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
                    n_ff_2 = n_level*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
                    n_n_ff = n_ff_1*n_ff_2             # Product of number densities of ff pair
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_n_ff * ff_stored[q, idx_T_fine, l]
                        
                # For each source of bound-free absorption (photodissociation)
                for q in range(N_bf_species): 
                    
                    n_q = n_level*X_bf[q,i,j,k]   # Number density of dissociating species
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * bf_stored[q,l]
                
                # For each molecular / atomic species with active absorption features

                for q in range(N_species_active): 

                    if bulk == False:
                        if q == contribution_molecule_active_index:
                            n_q = n_level*X_active[q,i,j,k]   # Number density of this active species
                
                        else:
                            n_q = 0
                    
                    else:
                        # If bulk is true, then everything in active is turned off 
                        n_q = 0
                    
                    # For each wavelength
                    for l in range(N_wl):
                        
                        # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]
                    
                # For each molecular / atomic species
                for q in range(N_species):  
                    
                    if bulk == False:
                        if q == contribution_molecule_species_index:
                            n_q = n_level*X[q,i,j,k]   # Number density of given species
                        elif q in bulk_species_indices:
                            n_q = n_level*X[q,i,j,k]   # Number density of given species
                        else:
                            n_q = 0

                    else:
                        # If bulk is true, only keep the bulk species on 
                        if q in bulk_species_indices:
                            n_q = n_level*X[q,i,j,k]   # Number density of given species
                        else:
                            n_q = 0
                    
                    # For each wavelength
                    for l in range(N_wl):
                                
                        # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_clear[i,j,k,l] += n_q * Rayleigh_stored[q,l]
        
            # If haze is enabled in this model  
            if (enable_haze == 1):
                
                # For each layer
                for i in range(i_bot,N_layers):
                    
                    haze_amp = (n[i,j,k] * a * 5.31e-31)   # Final factor is H2 Rayleigh scattering cross section at 350 nm
                    
                    # For each wavelength
                    for l in range(N_wl):
                    
                        # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                        kappa_cloud[i,j,k,l] += haze_amp * slope[l]
                        
            # If a cloud deck is enabled in this model
            if (enable_deck == 1):
                
                # Set extinction inside cloud deck
                kappa_cloud[(P > P_cloud),j,k,:] += kappa_cloud_0

            # If a surface is enabled in this model
            if (enable_surface == 1):

                # Set extinction to infinity below surface
                kappa_clear[(P > P_surf),j,k,:] = 1.0e250
            
    return kappa_clear, kappa_cloud


@jit(nopython = True)
def extinction_spectrum_pressure_contribution(chemical_species, active_species, cia_pairs, ff_pairs, bf_species,
                            n, T, P, wl, X, X_active, X_cia, X_ff, X_bf, a, gamma, P_cloud, 
                            kappa_cloud_0, sigma_stored, cia_stored, Rayleigh_stored, ff_stored, 
                            bf_stored, enable_haze, enable_deck, enable_surface, N_sectors, 
                            N_zones, T_fine, log_P_fine, P_surf, P_deep = 1000.0,
                            contribution_molecule = '',layer_to_ignore = 0, total = False):                          # DOES P_DEEP SOLVE BD PROBLEM?!
    
    ''' Main function to evaluate extinction coefficients for molecules / atoms,
        Rayleigh scattering, hazes, and clouds for parameter combination
        chosen in retrieval step.
        
        Takes in cross sections pre-interpolated to 'fine' P and T grids
        before retrieval run (so no interpolation is required at each step).
        Instead, for each atmospheric layer the extinction coefficient
        is simply kappa = n * sigma[log_P_nearest, T_nearest, wl], where the
        'nearest' values are the closest P_fine, T_fine points to the
        actual P, T values in each layer. This results in a large speed gain.
        
        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), terminator
        sector, and wavelength.

        This is to turn off every opacity except one molecule.
        For now, I am ignoring CIA 
    
    '''
    
    # Store length variables for mixing ratio arrays 
    N_species = len(chemical_species)        # Number of chemical species
    N_species_active = len(active_species)   # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)             # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)               # Number of free-free pairs included
    N_bf_species = len(bf_species)           # Number of bound-free species included

    # Set up all the indices for contribution functions
    N_bulk_species = N_species - N_species_active
    bulk_species_indices = range(N_bulk_species)
    
    # If total = false then it was passed a chemical species 
    if total == False:
        for i in range(len(chemical_species)):
            if contribution_molecule == chemical_species[i]:
                contribution_molecule_species_index = i

        for i in range(len(active_species)):
            if contribution_molecule == active_species[i]:
                contribution_molecule_active_index = i
    
    # Layers and wavelengths 
    N_wl = len(wl)     # Number of wavelengths on model grid
    N_layers = len(P)  # Number of layers
    
    # Define extinction coefficient arrays
    kappa_clear = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    
    # Fine temperature grid (for pre-interpolating opacities)    
    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)
    
    # Find index of deep pressure below which atmosphere is opaque
    i_bot = np.argmin(np.abs(P - P_deep))
    
    # If haze is enabled in this model
    if (enable_haze == 1):
        
        # Generalised scattering slope for haze
        slope = np.power((wl/0.35), gamma)    # Reference wavelength at 0.35 um

    if total == True:
        # For each terminator sector (terminator plane)
        for j in range(N_sectors):
                
            # For each terminator zone (along day-night transition)
            for k in range(N_zones):
                
                # For each layer, find closest pre-computed cross section to P_fine, T_fine
                for i in range(i_bot,N_layers):
                    
                    n_level = n[i,j,k]
                    
                    # Find closest index in fine temperature array to given layer temperature
                    idx_T_fine = closest_index(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
                    idx_P_fine = closest_index(np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
                    
                    # For each collisionally-induced absorption (CIA) pair
                    for q in range(N_cia_pairs): 
                        
                        n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                        n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                        n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair
                        
                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_n_cia * cia_stored[q, idx_T_fine, l]
                            
                    # For each free-free absorption pair
                    for q in range(N_ff_pairs): 
                        
                        n_ff_1 = n_level*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
                        n_ff_2 = n_level*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
                        n_n_ff = n_ff_1*n_ff_2             # Product of number densities of ff pair
                        
                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_n_ff * ff_stored[q, idx_T_fine, l]
                            
                    # For each source of bound-free absorption (photodissociation)
                    for q in range(N_bf_species): 
                        
                        n_q = n_level*X_bf[q,i,j,k]   # Number density of dissociating species
                        
                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_q * bf_stored[q,l]
                    
                    # For each molecular / atomic species with active absorption features

                    for q in range(N_species_active): 

                        # If its the molecule and the layer we are ignoring it in, set to 0 
                        if i == layer_to_ignore:
                            n_q = 0
                        
                        else:
                            n_q = n_level*X_active[q,i,j,k]   # Number density of this active species

                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]
                        
                    # For each molecular / atomic species
                    for q in range(N_species):  
                        
                        # If total then just turn off 
                        if i == layer_to_ignore and q not in bulk_species_indices:
                            n_q = 0
                        else:
                            n_q = n_level*X[q,i,j,k]
                        
                        # For each wavelength
                        for l in range(N_wl):
                                    
                            # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_q * Rayleigh_stored[q,l]
            
                # If haze is enabled in this model  
                if (enable_haze == 1):
                    
                    # For each layer
                    for i in range(i_bot,N_layers):
                        
                        haze_amp = (n[i,j,k] * a * 5.31e-31)   # Final factor is H2 Rayleigh scattering cross section at 350 nm
                        
                        # For each wavelength
                        for l in range(N_wl):
                        
                            # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                            kappa_cloud[i,j,k,l] += haze_amp * slope[l]
                            
                # If a cloud deck is enabled in this model
                if (enable_deck == 1):
                    
                    # Set extinction inside cloud deck
                    kappa_cloud[(P > P_cloud),j,k,:] += kappa_cloud_0

                # If a surface is enabled in this model
                if (enable_surface == 1):

                    # Set extinction to infinity below surface
                    kappa_clear[(P > P_surf),j,k,:] = 1.0e250

    else:
        # For each terminator sector (terminator plane)
        for j in range(N_sectors):
                
            # For each terminator zone (along day-night transition)
            for k in range(N_zones):
                
                # For each layer, find closest pre-computed cross section to P_fine, T_fine
                for i in range(i_bot,N_layers):
                    
                    n_level = n[i,j,k]
                    
                    # Find closest index in fine temperature array to given layer temperature
                    idx_T_fine = closest_index(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
                    idx_P_fine = closest_index(np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
                    
                    # For each collisionally-induced absorption (CIA) pair
                    for q in range(N_cia_pairs): 
                        
                        n_cia_1 = n_level*X_cia[0,q,i,j,k]   # Number density of first cia species in pair
                        n_cia_2 = n_level*X_cia[1,q,i,j,k]   # Number density of second cia species in pair
                        n_n_cia = n_cia_1*n_cia_2            # Product of number densities of cia pair
                        
                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_n_cia * cia_stored[q, idx_T_fine, l]
                            
                    # For each free-free absorption pair
                    for q in range(N_ff_pairs): 
                        
                        n_ff_1 = n_level*X_ff[0,q,i,j,k]   # Number density of first species in ff pair
                        n_ff_2 = n_level*X_ff[1,q,i,j,k]   # Number density of second species in ff pair
                        n_n_ff = n_ff_1*n_ff_2             # Product of number densities of ff pair
                        
                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_n_ff * ff_stored[q, idx_T_fine, l]
                            
                    # For each source of bound-free absorption (photodissociation)
                    for q in range(N_bf_species): 
                        
                        n_q = n_level*X_bf[q,i,j,k]   # Number density of dissociating species
                        
                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_q * bf_stored[q,l]
                    
                    # For each molecular / atomic species with active absorption features

                    for q in range(N_species_active): 

                        # If its the molecule and the layer we are ignoring it in, set to 0 
                        if q == contribution_molecule_active_index and i == layer_to_ignore:
                            n_q = 0
                        
                        else:
                            n_q = n_level*X_active[q,i,j,k]   # Number density of this active species

                        # For each wavelength
                        for l in range(N_wl):
                            
                            # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]
                        
                    # For each molecular / atomic species
                    for q in range(N_species):  
                        
                        if q == contribution_molecule_species_index and i == layer_to_ignore:
                            n_q = 0                    # Number density of given species
                        else:
                            n_q = n_level*X[q,i,j,k]
                        
                        # For each wavelength
                        for l in range(N_wl):
                                    
                            # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                            kappa_clear[i,j,k,l] += n_q * Rayleigh_stored[q,l]
            
                # If haze is enabled in this model  
                if (enable_haze == 1):
                    
                    # For each layer
                    for i in range(i_bot,N_layers):
                        
                        haze_amp = (n[i,j,k] * a * 5.31e-31)   # Final factor is H2 Rayleigh scattering cross section at 350 nm
                        
                        # For each wavelength
                        for l in range(N_wl):
                        
                            # Add haze scattering to total extinction in layer i, sector j, for each wavelength
                            kappa_cloud[i,j,k,l] += haze_amp * slope[l]
                            
                # If a cloud deck is enabled in this model
                if (enable_deck == 1):
                    
                    # Set extinction inside cloud deck
                    kappa_cloud[(P > P_cloud),j,k,:] += kappa_cloud_0

                # If a surface is enabled in this model
                if (enable_surface == 1):

                    # Set extinction to infinity below surface
                    kappa_clear[(P > P_surf),j,k,:] = 1.0e250
            
    return kappa_clear, kappa_cloud