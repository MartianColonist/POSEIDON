''' 
Functions for calculating clouds from database aerosols

'''
import numpy as np
import scipy
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
import os 

from .utility import shared_memory_array
from .supported_opac import aerosol_supported_species

############################################################################################
# Utility Functions
############################################################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# I have to copy this into the py file because otherwise it causes a circular import
def wl_grid_constant_R(wl_min, wl_max, R):
    '''
    Create a wavelength array with constant spectral resolution (R = wl/dwl).

    Args:
        wl_min (float):
            Minimum wavelength of grid (μm).
        wl_max (float): 
            Maximum wavelength of grid (μm).
        R (int or float):
            Spectral resolution of desired wavelength grid.
    
    Returns:
        wl (np.array of float):
            Model wavelength grid (μm).

    '''

    # Constant R -> uniform in log(wl)
    delta_log_wl = 1.0/R
    N_wl = (np.log(wl_max) - np.log(wl_min)) / delta_log_wl
    N_wl = np.around(N_wl).astype(np.int64)
    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), N_wl)    

    wl = np.exp(log_wl)
    
    return wl

# Plot the cross section for a specific aersol in the database (for testing)
def plot_effective_cross_section_aerosol(aerosol, wl, r_m):

    # Load in the aerosol database
    input_file_path = os.environ.get("POSEIDON_input_data")

    if (input_file_path == None):
        raise Exception("POSEIDON cannot locate the input folder.\n" +
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "POSEIDON input folder.")

    try :
        database = h5py.File(input_file_path + '/aerosol_database.hdf5', 'r')
    except :
        raise Exception('Please put aerosol_database.hdf5 in the input folder')
    
    # Create an interpolate object 
    sigma_Mie_full = np.array(database[aerosol+'/sigma_Mie'])
    wavelengths = wl_grid_constant_R(0.2, 30, 1000)
    r_m_array = 10**np.linspace(-3,1,1000)
    interp = RegularGridInterpolator((r_m_array, wavelengths), sigma_Mie_full, bounds_error=False, fill_value=None)

    # Plot the interpolated effective cross sections
    sigma_Mie = interp((r_m,wl))
    label = 'r_m (um) : ' + str(r_m)
    title = aerosol + ' Effective Cross Section'
    plt.figure(figsize=(10,6))
    plt.plot(wl,sigma_Mie, label = label)
    plt.legend()
    plt.title(title)
    plt.xlabel('Wavelengths (um)')
    plt.ylabel('Effective Cross Section')
    plt.show()

    database.close()

# Plot the number density above the fuzzy deck (for testing)
def plot_aerosol_number_denstiy_fuzzy_deck(atmosphere,log_P_cloud,log_n_max,fractional_scale_height):

    r = atmosphere['r']
    H = atmosphere['H']
    P = atmosphere['P']

    P_cloud = 10**log_P_cloud

    # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
    n_aerosol = np.empty_like(r)
    P_cloud_index = find_nearest(P,P_cloud)
    # Find the radius corresponding to the cloud top pressure 
    cloud_top_height = r[P_cloud_index]
    # Height above cloud 
    h = r[P_cloud_index:] - cloud_top_height
    # Find number density below and above P_cloud
    n_aerosol[:P_cloud_index] = 1.0e250
    n_aerosol[P_cloud_index:] = (10**log_n_max) * np.exp(-h/(fractional_scale_height*H[P_cloud_index:]))


    title = ('Number Density of Aerosol above Cloud Deck\n log_P_cloud: ' + str(log_P_cloud) + 
             ' log_n_max: ' + str(log_n_max) + ' f: ' + str(fractional_scale_height))
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    ax.plot(np.log10(n_aerosol.T[0][0])[P_cloud_index:],np.log10(P)[P_cloud_index:])
    ax.axhspan(log_P_cloud, np.log10(np.max(P)), alpha=0.5, color='gray', label = 'Opaque Cloud')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('log(Number Density)')
    ax.set_ylabel('log(P)')
    ax.legend()
    plt.show()

############################################################################################
# Loading Saved Array 
############################################################################################

def load_aerosol_grid(aerosol_species, grid = 'aerosol', 
                        comm = MPI.COMM_WORLD, rank = 0):
    '''
    Load a aerosol cross section grid.
    Args:
        aerosol_species (list or np.array of str):
            List of aerosol species to load mixing ratios from grid.
        grid (str):
            Name of the pre-computed aerosol cross section grid. The file should be
            located in the POSEIDON input directory (specified in your .bashrc
            file) with a name format like 'GRID_database.hdf5' 
            (e.g. 'aerosol_database.hdf5'). By default, POSEIDON ships with
            an aerosol grid computed from the LX-MIE algorith:
            (Options: aerosol).
        comm (MPI communicator):
            Communicator used to allocate shared memory on multiple cores.
        rank (MPI rank):
            Rank used to allocate shared memory on multiple cores.
    Returns:
        chemistry_grid (dict):
            Dictionary containing the chemical abundance database.
    
    '''

    if (rank == 0):
        print("Reading in database for aerosol cross sections...")

    # Check that the selected chemistry grid is supported
    if (grid not in ['aerosol']):
        raise Exception("Error: unsupported aerosol grid")

    # Find the directory where the user downloaded the input grid
    input_file_path = os.environ.get("POSEIDON_input_data")

    if input_file_path == None:
        raise Exception("POSEIDON cannot locate the input folder.\n" +
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "POSEIDON input folder.")
    
    aerosol_species = np.array(aerosol_species)
    
    # Open chemistry grid HDF5 file
    database = h5py.File(input_file_path  + grid + '_database.hdf5', 'r')

    # Load the dimensions of the grid
    wl_grid = np.array(database['Info/Wavelength grid'])
    r_m_grid = np.array(database['Info/Particle Size grid'])

    # Find sizes of each dimension
    wl_num, r_m_num = len(wl_grid), len(r_m_grid)

    # Store number of chemical species
    N_species = len(aerosol_species)

    # Create array to store the log mixing ratios from the grid 
    sigma_Mie_grid = shared_memory_array(rank, comm, (N_species, r_m_num, wl_num))
    
    # Only first core needs to load the aerosols into shared memory
    if (rank == 0):

        # Add each aerosol species to mixing ratio array
        for q, species in enumerate(aerosol_species):

            # Load grid for species q, then reshape into a 2D numpy array
            array = np.array(database[species+'/sigma_Mie'])
            array = array.reshape(r_m_num, wl_num)

            # Package grid for species q into combined array
            sigma_Mie_grid[q,:,:] = array

    # Close HDF5 file
    database.close()
        
    # Force secondary processors to wait for the primary to finish
    comm.Barrier()

    # Package atmosphere properties
    aerosol_grid = {'grid': grid, 'sigma_Mie_grid': sigma_Mie_grid, 'wl_grid': wl_grid, 'r_m_grid' : r_m_grid}

    return aerosol_grid


def interpolate_sigma_Mie_grid(aerosol_grid, wl, r_m_array, 
                               aerosol_species, return_dict = True):
    '''
    Interpolate a pre-computed grid of aerosol cross sections
    onto a model wl range, and mean particle size.
    Args:
        aerosol_grid (dict):
            Dictionary containing the aerosol cross section database.
        wl (np.array of float):
            Model wavelength grid (μm).
        r_m_array   (float) : 
            Mean particle size (in um) (for each aerosol_species)
        chemical_species (str or np.array of str):
            List of chemical species to interpolate mixing ratios for.
        return_dict (bool):
            If False, return an array of shape (len(species), len(P_array)).
    Returns:
        sigma_Mie_interp_dict (dict) ---> if return_dict = True:
            A dictionary of effective cross sections with keys being the same names as 
            specified in aerosol_species.
        sigma_Mie_interp_array (np.array of float) ---> if return_dict=False:
            An array containing the effective cross sections for the species specified
            in aerosol_species.
    
    '''

    # Unpack chemistry grid properties
    grid = aerosol_grid['grid']
    sigma_Mie_grid = aerosol_grid['sigma_Mie_grid']
    r_m_grid = aerosol_grid['r_m_grid']
    wl_grid = aerosol_grid['wl_grid']
    aerosol_species = np.array(aerosol_species)
    

    # Store lengths of input P, T, C/O and metallicity arrays
    len_r_m, len_wl = np.array(r_m_array).size, np.array(wl).size
    max_len = max(len_r_m, len_wl)

    np.seterr(divide = 'ignore')

    # Check that the chemical species we want to interpolate are supported
    if (grid == 'aerosol'):
        supported_species = aerosol_supported_species
    else:
        raise Exception("Error: unsupported aerosol grid")
    if isinstance(aerosol_species, str):
        if aerosol_species not in supported_species: 
            raise Exception(aerosol_species + " is not supported by the aerosol grid. Check supported_opac.py")

    # Check that the desired wl and r_m
    def not_valid(params, grid):
        return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

    if not_valid(wl, wl_grid):
        raise Exception("Requested wavelength range is out of the grid bounds (0.2 to 30 um).")
    if not_valid(r_m_array, r_m_grid):
        raise Exception("Requested mean particle size is out of the grid bounds. (0.001 to 10 um)")

    # Interpolate cross sections onto the r_m and wl 
    def interpolate(species):

        # Find index of the species
        q = np.where(aerosol_species == species)[0][0]

        # Create interpolator object
        grid_interp = RegularGridInterpolator((r_m_grid, wl_grid), sigma_Mie_grid[q,:,:])
        
        return grid_interp((r_m_array[q],wl))
    
    # Returning an array (default) 
    if not return_dict:
        if isinstance(aerosol_species, str):
            return interpolate(aerosol_species)
        sigma_Mie_list = []
        for _, species in enumerate(aerosol_species):
            sigma_Mie_list.append(interpolate(species))
        sigma_Mie_interp_array = np.array(sigma_Mie_list)
        return sigma_Mie_interp_array
    
    # Returning a dictionary
    else:
        sigma_Mie_interp_dict = {}
        if isinstance(aerosol_species, str):
            sigma_Mie_interp_dict[aerosol_species] = interpolate(aerosol_species)
            return sigma_Mie_interp_dict
        for _, species in enumerate(aerosol_species):
            sigma_Mie_interp_dict[species] = interpolate(species)
        return sigma_Mie_interp_dict

############################################################################################
# Main Cloud Functions
############################################################################################

def Mie_cloud(P,wl,r, H, n,
              r_m, 
              aerosol_species,
              cloud_type,
              aerosol_grid = None,
              P_cloud = 0,
              log_n_max = 0, 
              fractional_scale_height = 0,
              log_X_Mie = 0,):


    '''
    Calculates the number density n(P) and cross section sigma(wavelength) for a aerosol cloud.
    aerosol clouds are defined as being opaque below P_cloud. 
    Returns the absorption coefficient kappa = n * cross section

    Args:

        P (np.array of float):
            Model pressure grid (bar). (From atmosphere['P'])

        wl (np.array of float):
            Model wavelength grid (μm).
        
        r (3D np.array of float):
            Radial distant profile (m). (From atmosphere['P'])

        H (np.array of float) : 
            gas scale height

        n (np.array of float) :
            total number density array 

        r_m  (np.array of float) : 
            Mean particle sizes (in um)

        aerosol_species (np.array of string) : 
            Array with aerosol species in it 

        cloud_type (string):
            uniform_X or fuzzy_deck

        aerosol_grid (dict) : 
            Precomputed aerosol cross section dictionary 
            If = None, loads it in 

        -------- Semi- Optional Arguments -------

        Fuzzy Deck Arguments

        P_cloud (float) : 
            Cloud Top Pressure (everything below P_cloud is opaque). 
            If cloud coverage is complete, P_cloud is located at R_p

        log_n_max (array of float) : 
            Logorithm of maximum number density (at the cloud top)

        fractional_scale_height (array of float) :
            fractional scale height of aerosol 

        Uniform X Arguments

        log_X_Mie (array of float) : 
            Mixing ratio for a mie aerosol (either specified or free, only for uniform haze models)

        -------- Optional Arguments -------

        r_m_std_dev (float) :
            Geometric standard deviation for particle size 

        z_max (float) : 
            Maximum z that you want the effective cross section integral carried out over
            z = [ln(r) - ln(r_m)] / [r_m_std_dev^2], where r is the particle size 
            Integral carried out from -z to z with more density around 0 (size ~ mean size)

        num_integral_points (int) : 
            Number of points in the z array 

        R_Mie (int) : 
            Optional wavelength resolution used to calculate ETA 

    
    Returns: n_aerosol, sigma_Mie
          
    '''

    #########################
    # Initialize number density array
    #########################

    n_aerosol_array = []

    #########################
    # Loop through each aerosol
    #########################

    for q in range(len(r_m)):
    
        #########################
        # Calculate the number density above the cloud top or apply a uniform haze
        #########################
        
        # Fuzzy Deck Model 
        if (cloud_type == 'fuzzy_deck'):
            # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
            n_aerosol = np.empty_like(r)
            P_cloud_index = find_nearest(P,P_cloud)
            # Find the radius corresponding to the cloud top pressure 
            cloud_top_height = r[P_cloud_index]
            # Height above cloud 
            h = r[P_cloud_index:] - cloud_top_height
            # Find number density below and above P_cloud
            n_aerosol[:P_cloud_index] = 1.0e250
            n_aerosol[P_cloud_index:] = (10**log_n_max[q]) * np.exp(-h/(fractional_scale_height[q]*H[P_cloud_index:]))
            n_aerosol_array.append(n_aerosol)
        # Uniform X Model 
        else:
            n_aerosol = np.empty_like(r)
            n_aerosol = (n)*np.float_power(10,log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

    #########################
    # Load in effective cross section (as function of wavelength)
    #########################

    # If the aerosol grid wasn't read in already 
    if (aerosol_grid == None):
        aerosol_grid = load_aerosol_grid(aerosol_species, grid = 'aerosol', 
                        comm = MPI.COMM_WORLD, rank = 0)

    sigma_Mie_array = interpolate_sigma_Mie_grid(aerosol_grid, wl, r_m, 
                               aerosol_species, return_dict = False)


    return n_aerosol_array, sigma_Mie_array
