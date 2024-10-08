''' 
Stellar spectra and star spot/faculae contamination calculations.

'''

import os
import numpy as np
from numba.core.decorators import jit
from spectres import spectres
import scipy.constants as sc
import pysynphot as psyn
from mpi4py import MPI

from .utility import mock_missing, shared_memory_array

try:
    import pymsg as pymsg
except ImportError:
    pymsg = mock_missing('pymsg')


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


def load_stellar_pysynphot(wl_out, T_eff, Met, log_g, stellar_grid = 'cbk04'):
    '''
    Load a stellar model using pysynphot. Pynshot's ICAT function handles
    interpolation within the model grid to the specified stellar parameters.

    Note: Pysynphot's PHOENIX model grids have some bugs for non-solar 
          metallicity. So we default to the Castelli-Kurucz 2004 grids.

    Args:
        wl_out (np.array of float):
            Wavelength grid on which to output the stellar spectra (μm).
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
        I_out (np.array of float):
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

    wl_grid = sp.wave                         # Stellar wavelength array (um)
    I_grid = (sp.flux*1e-7*1e4*1e10)/np.pi    # Convert to W/m^2/sr/m

    # Bin / interpolate stellar spectrum to output wavelength grid
    I_out = spectres(wl_out, wl_grid, I_grid)
    
    return I_out


def open_pymsg_grid(stellar_grid):
    '''
    Check if pymsg is installed, then opens the HDF5 file for the  grid.

    Args:
        stellar_grid (string):
            Stellar model grid to use
            (Options: PHOENIX (or Goettingen-HiRes) / SPHINX).

    Returns:
        specgrid (pymsg object):
            Stellar grid object in pymsg format.
    
    '''

    # Check if pymsg is installed (required for this optional functionality)
    try:
        import pymsg as pymsg
    except ImportError:
        raise Exception("PyMSG is not installed on this machine. PyMSG " +
                        "is an optional add-on to POSEIDON, so please " +
                        "either install it or fall back on the default " +
                        "interpolation scheme interp_backend = 'pysynphot'.")
    
    # Allow alias 'phoenix' for pymsg stellar grid
    if (stellar_grid.lower() == 'phoenix'):
        stellar_grid = 'Goettingen-HiRes'   # Alias
    elif (stellar_grid.lower() == 'sphinx'):
        stellar_grid = 'SPHINX'
    else:
        raise Exception("Unsupported stellar grid for PyMSG")

    # Find user's MSG stellar grid directory
    MSG_DIR = os.environ['MSG_DIR']
    GRID_DIR = os.path.join(MSG_DIR, 'data', 'grids')

    # Open stellar grid HDF5 file
    specgrid_file_name = os.path.join(GRID_DIR, 'sg-' + stellar_grid + '.h5')
    specgrid = pymsg.SpecGrid(specgrid_file_name)

    return specgrid


def load_stellar_pymsg(wl_out, specgrid, T_eff, Met, log_g, stellar_grid):
    '''
    Load a stellar model using PyMSG. The MSG package 
    (https://msg.readthedocs.io/en/stable/index.html) handles
    interpolation within the model grid to the specified stellar parameters.

    Args:
        wl_out (np.array of float):
            Wavelength grid on which to output the stellar spectra (μm).
        specgrid (pymsg object):
            Stellar grid object in pymsg format.
        T_eff (float):
            Effective temperature of star (K).
        Met (float):
            Stellar metallicity [log10(Fe/H_star / Fe/H_solar)].
        log_g (float):
            Stellar log surface gravity (log10(cm/s^2) by convention).
        stellar_grid (string):
            Stellar model grid to use
            (Options: phoenix / sphinx).
    
    Returns:
        I_grid (np.array of float):
            Stellar specific intensity spectrum in SI units (W/m^2/sr/m).

    '''

    # Load minimum and maximum grid wavelengths
    wl_s_min = specgrid.lam_min / 10000   # Convert from A to μm
    wl_s_max = specgrid.lam_max / 10000   # Convert from A to μm

    # Compute array of wavelength grid edges
    wl_edges = np.zeros(len(wl_out)+1)
    wl_edges[0] = wl_out[0] - 0.5*(wl_out[1] - wl_out[0])
    wl_edges[-1] = wl_out[-1] + 0.5*(wl_out[-1] - wl_out[-2])
    wl_edges[1:-1] = 0.5*(wl_out[1:] + wl_out[:-1])

    # Check for user's wavelength grid lying outside PyMSG's grid wavelength range
    if (wl_edges[0] < wl_s_min):
        raise Exception("Wavelength grid extends lower than PyMSG's minimum " + 
                        "wavelength for this grid (" + str(wl_s_min) + "μm)")
    if (wl_edges[-1] > wl_s_max):
        raise Exception("Wavelength grid extends higher than PyMSG's maximum " + 
                        "wavelength for this grid (" + str(wl_s_max) + "μm)")

    # Package stellar parameters into PyMSG's expected input dictionary format
    if (stellar_grid.lower() == 'phoenix'):
        x = {'Teff': T_eff, 'log(g)': log_g, '[Fe/H]': Met, '[alpha/Fe]': 0.0}
    elif (stellar_grid.lower() == 'sphinx'):
        x = {'Teff': T_eff, 'log(g)': log_g, '[Fe/H]': Met, 'C/O': 0.55}   # FIXING stellar C/O to solar (for now)

    # Interpolate stellar grid to obtain stellar flux (also handles wl interpolation)
    F_s = specgrid.flux(x, wl_edges*10000)   # PyMSG expects Angstroms
    F_s = np.array(F_s) * 1e7   # Convert flux from erg/s/cm^2/A to W/m^2/m

    # Calculate average specific intensity
    I_grid = F_s / np.pi    # W/m^2/sr/m

    return I_grid


def precompute_stellar_spectra(comm, wl_out, star, prior_types, prior_ranges,
                               stellar_contam, T_step_interp = 20,
                               log_g_step_interp = 0.10, 
                               interp_backend = 'pysynphot',
                               ):
    '''
    Precompute a grid of stellar spectra across a range of T_eff and log g.

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
        stellar_contam (str):
            Chosen prescription for modelling unocculted stellar contamination
            (Options: one_spot / one_spot_free_log_g / two_spots).
        stellar_grid (str):
            Desired stellar model grid
            (Options: cbk04 / phoenix / sphinx).
        T_step_interp (float):
            Temperature step for stellar grid interpolation (uniform priors only).
        log_g_step_interp (float):
            log g step for stellar grid interpolation (uniform priors only).
        interp_backend (str):
            Stellar grid interpolation package for POSEIDON to use.
            (Options: pysynphot / pymsg).
        comm (MPI communicator):
            Communicator used to allocate shared memory on multiple cores.
    
    Returns:
        T_phot_grid (np.array of float):
            Photosphere temperatures corresponding to computed stellar spectra (K).
        T_het_grid (np.array of float):
            Heterogeneity temperatures corresponding to computed stellar spectra (K).
        log_g_phot_grid (np.array of float):
            Photosphere log g corresponding to computed stellar spectra (log10(cm/s^2)).
        log_g_het_grid (np.array of float):
            Heterogeneity log g corresponding to computed stellar spectra (log10(cm/s^2)).
        I_phot_out (3D np.array of float):
            Stellar photosphere intensity as a function of T_phot, log_g_phot, 
            and wl (W/m^2/sr/m).
        I_het_out (3D np.array of float):
            Stellar heterogeneity intensity as a function of T_het, log_g_het, 
            and wl (W/m^2/sr/m).

    '''

    # Find MPI rank and size
    rank = comm.Get_rank()    # Core number
    size = comm.Get_size()    # Number of cores

    # Unpack stellar properties
    Met_phot = star['Met']
    log_g_phot = star['log_g']
    stellar_grid = star['stellar_grid']

    if (interp_backend not in ['pysynphot', 'pymsg']):
        raise Exception("Error: supported stellar grid interpolater backends are " +
                        "'pysynphot' or 'pymsg'.")
    
    # If using PyMSG, load spectral grid
    if (interp_backend == 'pymsg'):
        specgrid = open_pymsg_grid(stellar_grid)
    
    #***** Find photosphere grid ranges *****#

    # Set range for T_phot according to whether prior is uniform or Gaussian
    if (prior_types['T_phot'] == 'uniform'):

        T_phot_min = prior_ranges['T_phot'][0]
        T_phot_max = prior_ranges['T_phot'][1]
        T_phot_step = T_step_interp                # Default interpolation step of 20 K

    elif (prior_types['T_phot'] == 'gaussian'):

        # Unpack Gaussian prior mean and std
        T_phot_mean = prior_ranges['T_phot'][0]  # Mean
        err_T_phot = prior_ranges['T_phot'][1]   # Standard deviation

        T_phot_min = T_phot_mean - (5.0 * err_T_phot)  # 5 sigma below mean
        T_phot_max = T_phot_mean + (5.0 * err_T_phot)  # 5 sigma above mean
        T_phot_step = err_T_phot/5.0                   # Interpolation step of 0.2 sigma

    # Find number of spectra needed to span desired range (rounding up)
    N_spec_T_phot = np.ceil((T_phot_max - T_phot_min)/T_phot_step).astype(np.int64) + 1

    # Specify starting and ending grid points
    T_start = T_phot_min                      
    T_end = T_phot_min + ((N_spec_T_phot-1) * T_phot_step)   # Slightly > T_max if T_step doesn't divide exactly

    # Initialise photosphere temperature array
    T_phot_grid = np.linspace(T_start, T_end, N_spec_T_phot)

    # For free log g, we also need to interpolate over a range of stellar log g
    if ('free_log_g' in stellar_contam):

        # Set range for log_g_phot according to whether prior is uniform or Gaussian
        if (prior_types['log_g_phot'] == 'uniform'):

            log_g_phot_min = prior_ranges['log_g_phot'][0]
            log_g_phot_max = prior_ranges['log_g_phot'][1]
            log_g_phot_step = log_g_step_interp               # Default interpolation step of 0.1

        elif (prior_types['log_g_phot'] == 'gaussian'):

            # Unpack Gaussian prior mean and std
            log_g_phot_mean = prior_ranges['log_g_phot'][0]   # Mean
            err_log_g_phot = prior_ranges['log_g_phot'][1]    # Standard deviation

            log_g_phot_min = log_g_phot_mean - (5.0 * err_log_g_phot)  # 5 sigma below mean
            log_g_phot_max = log_g_phot_mean + (5.0 * err_log_g_phot)  # 5 sigma above mean
            log_g_phot_step = err_log_g_phot/2.0                       # Interpolation step of 0.5 sigma

        # Find number of spectra needed to span desired range (rounding up)
        N_spec_log_g_phot = np.ceil((log_g_phot_max - log_g_phot_min)/log_g_phot_step).astype(np.int64) + 1

        # Specify starting and ending grid points
        log_g_start = log_g_phot_min                      
        log_g_end = log_g_phot_min + ((N_spec_log_g_phot-1) * log_g_phot_step)   # Slightly > log_g_max if log_g_step doesn't divide exactly 
        
        # Initialise photosphere log_g array
        log_g_phot_grid = np.linspace(log_g_start, log_g_end, N_spec_log_g_phot)

    # If log g fixed, we only need a single log g
    else:
        log_g_phot_grid = np.array([log_g_phot])
        N_spec_log_g_phot = 1

    #***** Interpolate photosphere spectra *****#
    
    # Initialise output photosphere spectra array
    I_phot_out, win_phot = shared_memory_array(rank, comm, (N_spec_T_phot, N_spec_log_g_phot, len(wl_out)))

    # Calculate chunk size and offsets for each process
   # chunk_size = N_spec_T_phot // size     # Parallelise over T_phot
   # offset = rank * chunk_size

    # Handle remainder for the final core
   # if (rank == (size - 1)):
   #     chunk_size += N_spec_T_phot % size  

   # for i in range(offset, offset + chunk_size):

    # Only first core interpolates stellar grids into shared memory
    if (rank == 0):

        # Interpolate and store stellar intensities
        for i in range(N_spec_T_phot):
            for j in range(N_spec_log_g_phot):
            
                # Load interpolated stellar spectrum from model grid
                if (interp_backend == 'pysynphot'):
                    I_phot_local = load_stellar_pysynphot(wl_out, T_phot_grid[i], 
                                                          Met_phot, log_g_phot_grid[j], 
                                                          stellar_grid)
                elif (interp_backend == 'pymsg'):
                    I_phot_local_1 = load_stellar_pymsg(wl_out[wl_out < 5.499], specgrid, T_phot_grid[i], 
                                                        Met_phot, log_g_phot_grid[j], stellar_grid)
                    I_phot_local_2 = planck_lambda(T_phot_grid[i], wl_out[wl_out >= 5.499])
                    I_phot_local = np.concatenate([I_phot_local_1, I_phot_local_2])
                    
                # Lock the shared memory window before copying results
               # win_phot.Lock(rank)
                
                # Copy local results to shared memory
                I_phot_out[i,j,:] = I_phot_local
                
                # Unlock the shared memory window
              #  win_phot.Unlock(rank)

    #***** Find heterogeneity grid ranges *****#

    if ('one_spot' in stellar_contam):

        T_het_min = prior_ranges['T_het'][0]
        T_het_max = prior_ranges['T_het'][1]

    elif ('two_spots' in stellar_contam):

        T_spot_min = prior_ranges['T_spot'][0]
        T_spot_max = prior_ranges['T_spot'][1]
        T_fac_min = prior_ranges['T_fac'][0]
        T_fac_max = prior_ranges['T_fac'][1]

        T_het_min = min(T_spot_min, T_fac_min)
        T_het_max = max(T_spot_max, T_fac_max)
        
    T_het_step = T_step_interp    # Default interpolation step of 20 K

    # Find number of spectra needed to span desired range (rounding up)
    N_spec_T_het = np.ceil((T_het_max - T_het_min)/T_het_step).astype(np.int64) + 1

    # Specify starting and ending grid points
    T_start = T_het_min                      
    T_end = T_het_min + ((N_spec_T_het-1) * T_het_step)   # Slightly > T_max if T_step doesn't divide exactly

    # Initialise heterogeneity temperature array
    T_het_grid = np.linspace(T_start, T_end, N_spec_T_het)

    # For free log g, we also need to interpolate over a range of stellar log g
    if ('free_log_g' in stellar_contam):

        if ('one_spot' in stellar_contam):

            log_g_het_min = prior_ranges['log_g_het'][0]
            log_g_het_max = prior_ranges['log_g_het'][1]

        elif ('two_spots' in stellar_contam):

            log_g_spot_min = prior_ranges['log_g_spot'][0]
            log_g_spot_max = prior_ranges['log_g_spot'][1]
            log_g_fac_min = prior_ranges['log_g_fac'][0]
            log_g_fac_max = prior_ranges['log_g_fac'][1]

            log_g_het_min = min(log_g_spot_min, log_g_fac_min)
            log_g_het_max = max(log_g_spot_max, log_g_fac_max)
            
        log_g_het_step = log_g_step_interp    # Default interpolation step of 0.1

        # Find number of spectra needed to span desired range (rounding up)
        N_spec_log_g_het = np.ceil((log_g_het_max - log_g_het_min)/log_g_het_step).astype(np.int64) + 1

        # Specify starting and ending grid points
        log_g_start = log_g_het_min                      
        log_g_end = log_g_het_min + ((N_spec_log_g_het-1) * log_g_het_step)   # Slightly > log_g_max if log_g_step doesn't divide exactly 
        
        # Initialise heterogeneity log_g array
        log_g_het_grid = np.linspace(log_g_start, log_g_end, N_spec_log_g_het)

    # If log g fixed, we only need a single log g
    else:
        log_g_het_grid = np.array([log_g_phot])
        N_spec_log_g_het = 1

    #***** Interpolate heterogeneity spectra *****#
    
    # Initialise output heterogeneity spectra array
    I_het_out, win_het = shared_memory_array(rank, comm, (N_spec_T_het, N_spec_log_g_het, len(wl_out)))

    # Calculate chunk size and offsets for each process
  #  chunk_size = N_spec_T_het // size     # Parallelise over T_het
  #  offset = rank * chunk_size

    # Handle remainder for the final core
  #  if (rank == (size - 1)):
  #      chunk_size += N_spec_T_het % size  

    # Only first core interpolates stellar grids into shared memory
    if (rank == 0):

        # Interpolate and store stellar intensities
        for i in range(N_spec_T_het):
            for j in range(N_spec_log_g_het):
            
                # Load interpolated stellar spectrum from model grid
                if (interp_backend == 'pysynphot'):
                    I_het_local = load_stellar_pysynphot(wl_out, T_het_grid[i], 
                                                         Met_phot, log_g_het_grid[j], 
                                                         stellar_grid)
                elif (interp_backend == 'pymsg'):
                    I_het_local_1 = load_stellar_pymsg(wl_out[wl_out < 5.499], specgrid, T_het_grid[i], 
                                                       Met_phot, log_g_het_grid[j], stellar_grid)
                    I_het_local_2 = planck_lambda(T_het_grid[i], wl_out[wl_out >= 5.499])
                    I_het_local = np.concatenate([I_het_local_1, I_het_local_2])
                    
                # Lock the shared memory window before copying results
             #   win_het.Lock(rank)
                
                # Copy local results to shared memory
                I_het_out[i,j,:] = I_het_local
                
                # Unlock the shared memory window
              #  win_het.Unlock(rank)
                
    # Force cores to wait for all to finish loops 
    comm.Barrier()

    return T_phot_grid, T_het_grid, log_g_phot_grid, log_g_het_grid, \
           I_phot_out, I_het_out


def precompute_stellar_spectra_OLD(wl_out, star, prior_types, prior_ranges,
                               stellar_contam, T_step_interp = 20,
                               log_g_step_interp = 0.10, 
                               interp_backend = 'pysynphot'):
    '''
    Precompute a grid of stellar spectra across a range of T_eff and log g.

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
        stellar_contam (str):
            Chosen prescription for modelling unocculted stellar contamination
            (Options: one_spot / one_spot_free_log_g / two_spots).
        stellar_grid (str):
            Desired stellar model grid
            (Options: cbk04 / phoenix).
        T_step_interp (float):
            Temperature step for stellar grid interpolation (uniform priors only).
        log_g_step_interp (float):
            log g step for stellar grid interpolation (uniform priors only).
        interp_backend (str):
            Stellar grid interpolation package for POSEIDON to use.
            (Options: pysynphot / pymsg).
    
    Returns:
        T_phot_grid (np.array of float):
            Photosphere temperatures corresponding to computed stellar spectra (K).
        T_het_grid (np.array of float):
            Heterogeneity temperatures corresponding to computed stellar spectra (K).
        log_g_phot_grid (np.array of float):
            Photosphere log g corresponding to computed stellar spectra (log10(cm/s^2)).
        log_g_het_grid (np.array of float):
            Heterogeneity log g corresponding to computed stellar spectra (log10(cm/s^2)).
        I_phot_out (3D np.array of float):
            Stellar photosphere intensity as a function of T_phot, log_g_phot, 
            and wl (W/m^2/sr/m).
        I_het_out (3D np.array of float):
            Stellar heterogeneity intensity as a function of T_het, log_g_het, 
            and wl (W/m^2/sr/m).

    '''

    # Unpack stellar properties
    Met_phot = star['Met']
    log_g_phot = star['log_g']
    stellar_grid = star['stellar_grid']

    if (interp_backend not in ['pysynphot', 'pymsg']):
        raise Exception("Error: supported stellar grid interpolater backends are " +
                        "'pysynphot' or 'pymsg'.")
    
    # If using PyMSG, load spectral grid
    if (interp_backend == 'pymsg'):
        specgrid = open_pymsg_grid(stellar_grid)
    
    #***** Find photosphere grid ranges *****#

    # Set range for T_phot according to whether prior is uniform or Gaussian
    if (prior_types['T_phot'] == 'uniform'):

        T_phot_min = prior_ranges['T_phot'][0]
        T_phot_max = prior_ranges['T_phot'][1]
        T_phot_step = T_step_interp                # Default interpolation step of 20 K

    elif (prior_types['T_phot'] == 'gaussian'):

        # Unpack Gaussian prior mean and std
        T_phot_mean = prior_ranges['T_phot'][0]  # Mean
        err_T_phot = prior_ranges['T_phot'][1]   # Standard deviation

        T_phot_min = T_phot_mean - (5.0 * err_T_phot)  # 10 sigma below mean
        T_phot_max = T_phot_mean + (5.0 * err_T_phot)  # 10 sigma above mean
        T_phot_step = err_T_phot/5.0                   # Interpolation step of 0.2 sigma

    # Find number of spectra needed to span desired range (rounding up)
    N_spec_T_phot = np.ceil((T_phot_max - T_phot_min)/T_phot_step).astype(np.int64) + 1

    # Specify starting and ending grid points
    T_start = T_phot_min                      
    T_end = T_phot_min + ((N_spec_T_phot-1) * T_phot_step)   # Slightly > T_max if T_step doesn't divide exactly

    # Initialise photosphere temperature array
    T_phot_grid = np.linspace(T_start, T_end, N_spec_T_phot)

    # For free log g, we also need to interpolate over a range of stellar log g
    if ('free_log_g' in stellar_contam):

        # Set range for log_g_phot according to whether prior is uniform or Gaussian
        if (prior_types['log_g_phot'] == 'uniform'):

            log_g_phot_min = prior_ranges['log_g_phot'][0]
            log_g_phot_max = prior_ranges['log_g_phot'][1]
            log_g_phot_step = log_g_step_interp               # Default interpolation step of 0.1

        elif (prior_types['log_g_phot'] == 'gaussian'):

            # Unpack Gaussian prior mean and std
            log_g_phot_mean = prior_ranges['log_g_phot'][0]   # Mean
            err_log_g_phot = prior_ranges['log_g_phot'][1]    # Standard deviation

            log_g_phot_min = log_g_phot_mean - (5.0 * err_log_g_phot)  # 5 sigma below mean
            log_g_phot_max = log_g_phot_mean + (5.0 * err_log_g_phot)  # 5 sigma above mean
            log_g_phot_step = err_log_g_phot/2.0                       # Interpolation step of 0.5 sigma

        # Find number of spectra needed to span desired range (rounding up)
        N_spec_log_g_phot = np.ceil((log_g_phot_max - log_g_phot_min)/log_g_phot_step).astype(np.int64) + 1

        # Specify starting and ending grid points
        log_g_start = log_g_phot_min                      
        log_g_end = log_g_phot_min + ((N_spec_log_g_phot-1) * log_g_phot_step)   # Slightly > log_g_max if log_g_step doesn't divide exactly 
        
        # Initialise photosphere log_g array
        log_g_phot_grid = np.linspace(log_g_start, log_g_end, N_spec_log_g_phot)

    # If log g fixed, we only need a single log g
    else:
        log_g_phot_grid = np.array([log_g_phot])
        N_spec_log_g_phot = 1

    #***** Interpolate photosphere spectra *****#
    
    # Initialise output photosphere spectra array
    I_phot_out = np.zeros(shape=(N_spec_T_phot, N_spec_log_g_phot, len(wl_out)))

    # Interpolate and store stellar intensities
    for i in range(N_spec_T_phot):
        for j in range(N_spec_log_g_phot):
        
            # Load interpolated stellar spectrum from model grid
            if (interp_backend == 'pysynphot'):
                I_phot_out[i,j,:] = load_stellar_pysynphot(wl_out, T_phot_grid[i], 
                                                           Met_phot, log_g_phot_grid[j], 
                                                           stellar_grid)
            elif (interp_backend == 'pymsg'):
                I_phot_out[i,j,:] = load_stellar_pymsg(wl_out, specgrid, T_phot_grid[i], 
                                                       Met_phot, log_g_phot_grid[j], stellar_grid)

    #***** Find heterogeneity grid ranges *****#

    if ('one_spot' in stellar_contam):

        T_het_min = prior_ranges['T_het'][0]
        T_het_max = prior_ranges['T_het'][1]

    elif ('two_spots' in stellar_contam):

        T_spot_min = prior_ranges['T_spot'][0]
        T_spot_max = prior_ranges['T_spot'][1]
        T_fac_min = prior_ranges['T_fac'][0]
        T_fac_max = prior_ranges['T_fac'][1]

        T_het_min = min(T_spot_min, T_fac_min)
        T_het_max = max(T_spot_max, T_fac_max)
        
    T_het_step = T_step_interp    # Default interpolation step of 20 K

    # Find number of spectra needed to span desired range (rounding up)
    N_spec_T_het = np.ceil((T_het_max - T_het_min)/T_het_step).astype(np.int64) + 1

    # Specify starting and ending grid points
    T_start = T_het_min                      
    T_end = T_het_min + ((N_spec_T_het-1) * T_het_step)   # Slightly > T_max if T_step doesn't divide exactly

    # Initialise heterogeneity temperature array
    T_het_grid = np.linspace(T_start, T_end, N_spec_T_het)

    # For free log g, we also need to interpolate over a range of stellar log g
    if ('free_log_g' in stellar_contam):

        if ('one_spot' in stellar_contam):

            log_g_het_min = prior_ranges['log_g_het'][0]
            log_g_het_max = prior_ranges['log_g_het'][1]

        elif ('two_spots' in stellar_contam):

            log_g_spot_min = prior_ranges['log_g_spot'][0]
            log_g_spot_max = prior_ranges['log_g_spot'][1]
            log_g_fac_min = prior_ranges['log_g_fac'][0]
            log_g_fac_max = prior_ranges['log_g_fac'][1]

            log_g_het_min = min(log_g_spot_min, log_g_fac_min)
            log_g_het_max = max(log_g_spot_max, log_g_fac_max)
            
        log_g_het_step = log_g_step_interp    # Default interpolation step of 0.1

        # Find number of spectra needed to span desired range (rounding up)
        N_spec_log_g_het = np.ceil((log_g_het_max - log_g_het_min)/log_g_het_step).astype(np.int64) + 1

        # Specify starting and ending grid points
        log_g_start = log_g_het_min                      
        log_g_end = log_g_het_min + ((N_spec_log_g_het-1) * log_g_het_step)   # Slightly > log_g_max if log_g_step doesn't divide exactly 
        
        # Initialise heterogeneity log_g array
        log_g_het_grid = np.linspace(log_g_start, log_g_end, N_spec_log_g_het)

    # If log g fixed, we only need a single log g
    else:
        log_g_het_grid = np.array([log_g_phot])
        N_spec_log_g_het = 1

    #***** Interpolate heterogeneity spectra *****#
    
    # Initialise output heterogeneity spectra array
    I_het_out = np.zeros(shape=(N_spec_T_het, N_spec_log_g_het, len(wl_out)))

    # Interpolate and store stellar intensities
    for i in range(N_spec_T_het):
        for j in range(N_spec_log_g_het):
        
            # Load interpolated stellar spectrum from model grid
            if (interp_backend == 'pysynphot'):
                I_het_out[i,j,:] = load_stellar_pysynphot(wl_out, T_het_grid[i], 
                                                          Met_phot, log_g_het_grid[j], 
                                                          stellar_grid)
            elif (interp_backend == 'pymsg'):
                I_het_out[i,j,:] = load_stellar_pymsg(wl_out, specgrid, T_het_grid[i], 
                                                      Met_phot, log_g_het_grid[j], stellar_grid)

    return T_phot_grid, T_het_grid, log_g_phot_grid, log_g_het_grid, \
           I_phot_out, I_het_out


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
def stellar_contamination_general(f_het, I_het, I_phot):
    '''
    Computes the multiplicative stellar contamination factor for a transmission
    spectrum due to a collection of unocculted starspot or facular regions, each
    of which may have their own coverage fraction and specific intensity.

    Args:
        f_het (np.array of float):
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
    for i in range(len(f_het)):

        I_ratio = I_het[i,:]/I_phot

        sum += (f_het[i] * (1.0 - I_ratio))

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
    I_phot = star['I_phot']
    stellar_contam = star['stellar_contam']

    # Interpolate stellar photosphere intensity to model wavelength grid
    I_phot_interp = spectres(wl_out, wl_s, I_phot)

    # For a single heterogeneity
    if ('one_spot' in stellar_contam):

        # Unpack relevant stellar properties
        f_het = star['f_het']
        I_het = star['I_het']

        # Interpolate heterogeneity intensity to model wavelength grid
        I_het_interp = spectres(wl_out, wl_s, I_het)

        # Compute (wavelength-dependent) stellar heterogeneity correction
        epsilon = stellar_contamination_single_spot(f_het, I_het_interp, I_phot_interp)

    # For two heterogeneities
    elif ('two_spots' in stellar_contam):

        # Unpack relevant stellar properties
        f_spot = star['f_spot']
        f_fac = star['f_fac']
        I_spot = star['I_spot']
        I_fac = star['I_fac']

        # Interpolate heterogeneity intensities to model wavelength grid
        I_spot_interp = spectres(wl_out, wl_s, I_spot)
        I_fac_interp = spectres(wl_out, wl_s, I_fac)

        # Stack spot and faculae spectra and fractions (for general formula)
        I_het_interp = np.vstack((I_spot_interp, I_fac_interp))
        f_het = np.array([f_spot, f_fac])
    
        # Compute (wavelength-dependent) stellar heterogeneity correction
        epsilon = stellar_contamination_general(f_het, I_het_interp, I_phot_interp)

    return epsilon