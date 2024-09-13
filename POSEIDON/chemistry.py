''' 
Functions to interpolate chemical composition grids.

'''

import os
import h5py
import numpy as np
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator

from .utility import shared_memory_array
from .supported_chemicals import supported_species, fastchem_supported_species


def load_chemistry_grid(chemical_species, grid = 'fastchem', 
                        comm = MPI.COMM_WORLD, rank = 0):
    '''
    Load a chemical abundance grid.

    Args:
        chemical_species (list or np.array of str):
            List of chemical species to load mixing ratios from grid.
        grid (str):
            Name of the pre-computed chemical abundance grid. The file should be
            located in the POSEIDON input directory (specified in your .bashrc
            file) with a name format like 'GRID_database.hdf5' 
            (e.g. 'fastchem_database.hdf5'). By default, POSEIDON ships with
            an equilibrium chemistry grid computed from the fastchem code:
            https://github.com/exoclime/FastChem
            (Options: fastchem).
        comm (MPI communicator):
            Communicator used to allocate shared memory on multiple cores.
        rank (MPI rank):
            Rank used to allocate shared memory on multiple cores.

    Returns:
        chemistry_grid (dict):
            Dictionary containing the chemical abundance database.
    
    '''

    if (rank == 0):
        print("Reading in database for equilibrium chemistry model...")

    # Check that the selected chemistry grid is supported
    if (grid not in ['fastchem']):
        raise Exception("Error: unsupported chemistry grid")

    # Find the directory where the user downloaded the input grid
    input_file_path = os.environ.get("POSEIDON_input_data")

    if input_file_path == None:
        raise Exception("POSEIDON cannot locate the input folder.\n" +
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "POSEIDON input folder.")

    # Load list of chemical species supported by both the fastchem grid and POSEIDON
    supported_chem_eq_species = np.intersect1d(supported_species, 
                                                fastchem_supported_species)
        
    # If chemical_species = ['all'] then default to all species
    if ('all' in chemical_species):
        chemical_species = supported_chem_eq_species

    # Check all user-specified species are compatible with the fastchem grid
    else:
        if (np.any(~np.isin(chemical_species, supported_chem_eq_species)) == True):
            raise Exception("A chemical species you selected is not supported " +
                            "for equilibrium chemistry models.\n")
            
    # Open chemistry grid HDF5 file
    database = h5py.File(input_file_path + '/chemistry_grids/' + grid + '_database.hdf5', 'r')

    # Load the dimensions of the grid
    T_grid = np.array(database['Info/T grid'])
    P_grid = np.array(database['Info/P grid'])
    Met_grid = np.array(database['Info/M/H grid'])
    C_to_O_grid = np.array(database['Info/C/O grid'])

    # Find sizes of each dimension
    T_num, P_num, \
    Met_num, C_O_num = len(T_grid), len(P_grid), len(Met_grid), len(C_to_O_grid)

    # Store number of chemical species
    N_species = len(chemical_species)

    # Create array to store the log mixing ratios from the grid 
    log_X_grid, _ = shared_memory_array(rank, comm, (N_species, Met_num, C_O_num, T_num, P_num))
    
    # Only first core needs to load the mixing ratios into shared memory
    if (rank == 0):

        # Add each chemical species to mixing ratio array
        for q, species in enumerate(chemical_species):

            # Load grid for species q, then reshape into a 4D numpy array
            array = np.array(database[species+'/log(X)'])
            array = array.reshape(Met_num, C_O_num, T_num, P_num)

            # Package grid for species q into combined array
            log_X_grid[q,:,:,:,:] = array

    # Close HDF5 file
    database.close()
        
    # Force secondary processors to wait for the primary to finish
    comm.Barrier()

    # Package atmosphere properties
    chemistry_grid = {'grid': grid, 'log_X_grid': log_X_grid, 'T_grid': T_grid, 
                      'P_grid': P_grid, 'Met_grid': Met_grid, 'C_to_O_grid': C_to_O_grid,
                     }

    return chemistry_grid


def interpolate_log_X_grid(chemistry_grid, log_P, T, C_to_O, log_Met, 
                           chemical_species, return_dict = True):
    '''
    Interpolate a pre-computed grid of chemical abundances (e.g. an equilibrium
    chemistry grid) onto a model P-T profile, metallicity, and C/O ratio.

    Args:
        chemistry_grid (dict):
            Dictionary containing the chemical abundance database.
        log_P (float or np.array of float): 
            Pressure profile provided by the user (in log scale and in bar).
            A single value will be expanded into an array np.full(length, P), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            10^{-7} to 10^{2} bar are supported.
        T (float or np.array of float):
            Temperature profile provided by the user (K).
            A single value will be expanded into an array np.full(length, T), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            300 to 4000 K are supported.
        C_to_O (float or np.array of float):
            Carbon to Oxygen (C/O) ratio provided by the user.
            A single value will be expanded into an array np.full(length, C_O), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            0.2 to 2 are supported.
        log_Met (float or np.array of float):
            Planetary metallicity (in log scale. 0 represents 1x solar).
            A single value will be expanded into an array np.full(length, Met), 
            where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            -1 to 4 are supported.
        chemical_species (str or np.array of str):
            List of chemical species to interpolate mixing ratios for.
        return_dict (bool):
            If False, return an array of shape (len(species), len(P_array)).

    Returns:
        log_X_interp_dict (dict) ---> if return_dict = True:
            A dictionary of log mixing ratios with keys being the same names as 
            specified in chemical_species.

        log_X_interp_array (np.array of float) ---> if return_dict=False:
            An array containing the log mixing ratios for the species specified
            in chemical_species.
    
    '''

    # Unpack chemistry grid properties
    grid = chemistry_grid['grid']
    log_X_grid = chemistry_grid['log_X_grid']
    T_grid = chemistry_grid['T_grid']
    P_grid = chemistry_grid['P_grid']
    Met_grid = chemistry_grid['Met_grid']
    C_to_O_grid = chemistry_grid['C_to_O_grid']

    # Store lengths of input P, T, C/O and metallicity arrays
    len_P, len_T, \
    len_C_to_O, len_Met = np.array(log_P).size, np.array(T).size, \
                          np.array(C_to_O).size, np.array(log_Met).size
    max_len = max(len_P, len_T, len_C_to_O, len_Met)

    np.seterr(divide = 'ignore')

    # Check that the chemical species we want to interpolate are supported
    if (grid == 'fastchem'):
        supported_species = fastchem_supported_species
    else:
        raise Exception("Error: unsupported chemistry grid")
    if isinstance(chemical_species, str):
        if chemical_species not in supported_species: 
            raise Exception(chemical_species + " is not supported by the equilibrium grid.")
    else:
        for species in chemical_species:
            if species not in supported_species: 
                raise Exception(species + " is not supported by the equilibrium grid.")

    # Check that the desired pressures, temperatures, C/O and metallicity fall within the grid
    def not_valid(params, grid, is_log):
        if is_log:
            return (10**np.max(params) < grid[0]) or (10**np.min(params) > grid[-1])
        else:
            return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

    if not_valid(log_P, P_grid, True):
        raise Exception("Requested pressure is out of the grid bounds.")
    if not_valid(T, T_grid, False):
        raise Exception("Requested temperature is out of the grid bounds.")
    if not_valid(C_to_O, C_to_O_grid, False):
        raise Exception("Requested C/O is out of the grid bounds.")
    if not_valid(log_Met, Met_grid, True):
        raise Exception("Requested M/H is out of the grid bounds.")
    
    # For POSEIDON's standard 3D temperature field
    if (len(T.shape) == 3):

        # Check validity of input array shapes
        T_shape = np.array(T).shape
        assert len_C_to_O == 1                # C_O should be a single value
        assert len_Met == 1                   # log_Met should be a single value
        assert len(log_P.shape) == 1          # log_P should be a 1D array
        assert log_P.shape[0] == T_shape[0]   # Size of log_P should match first dimension of T
        
        reps = np.array(T_shape[1:])
        reps = np.insert(reps, 0, 1)
        log_P = log_P.reshape(-1, 1, 1)
        log_P = np.tile(log_P, reps) # 1+T_shape[1:] is supposed to be (1, a, b) where T_shape[1:] = (a,b) is the second and third dimension of T. log_P should have the same dimension as T: (len(P), a, b)
        C_to_O = np.full(T_shape, C_to_O)
        log_Met = np.full(T_shape, log_Met)

    # For either a single (P, T, Met, C_to_O) or arrays
    else:
        if not (len_P in (1, max_len) and len_T in (1, max_len) and len_C_to_O in (1, max_len) and len_Met in (1, max_len)):
            raise Exception("Input shape not accepted. The lengths must either be the same or 1.")

        if len_P == 1:
            log_P = np.full(max_len, log_P)
        if len_T == 1:
            T = np.full(max_len, T)
        if len_C_to_O == 1:
            C_to_O = np.full(max_len, C_to_O)
        if len_Met == 1:
            log_Met = np.full(max_len, log_Met)

    # Interpolate mixing ratios from grid onto P-T profile, metallicity, and C/O of the atmosphere
    def interpolate(species):

        # Find index of the species
        q = np.where(chemical_species == species)[0][0]

        # Create interpolator object
        grid_interp = RegularGridInterpolator((np.log10(Met_grid), C_to_O_grid, T_grid, 
                                              np.log10(P_grid)), log_X_grid[q,:,:,:,:])
        
        return grid_interp(np.vstack((np.expand_dims(log_Met, 0), np.expand_dims(C_to_O, 0), 
                                      np.expand_dims(T, 0), np.expand_dims(log_P, 0))).T).T
    
    # Returning an array (default) 
    if not return_dict:
        if isinstance(chemical_species, str):
            return interpolate(chemical_species)
        log_X_list = []
        for _, species in enumerate(chemical_species):
            log_X_list.append(interpolate(species))
        log_X_interp_array = np.array(log_X_list)
        return log_X_interp_array
    
    # Returning a dictionary
    else:
        log_X_interp_dict = {}
        if isinstance(chemical_species, str):
            log_X_interp_dict[chemical_species] = interpolate(chemical_species)
            return log_X_interp_dict
        for _, species in enumerate(chemical_species):
            log_X_interp_dict[species] = interpolate(species)
        return log_X_interp_dict
