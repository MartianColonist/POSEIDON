import numpy as np
import h5py
import os
from scipy.interpolate import RegularGridInterpolator

print("Reading in database for equillrium chemistry model...")

# Find the directory where the user downloaded the POSEIDON opacity data
database_path = os.environ.get("POSEIDON_input_data")
if database_path == None:
    raise Exception("POSEIDON cannot locate the database for equillibrium chemistry.\n"
                    "Please set the 'POSEIDON_input_data' variable in " +
                    "your .bashrc or .bash_profile to point to the " +
                    "directory containing the POSEIDON opacity database.")
database = h5py.File(database_path+'eq_database.hdf5', 'r')
temperature_gird = np.array(database['Info'+'/T grid'])
pressure_grid = np.array(database['Info'+'/P grid'])
metallicity_grid = np.array(database['Info'+'/M/H grid'])
c_o_grid = np.array(database['Info'+'/C/O grid'])

def print_grid_info():
    print("Temperature grid: "+str(temperature_gird))
    print("_________________________________________________________________________")
    print("Pressure grid: "+str(pressure_grid))
    print("_________________________________________________________________________")
    print("Metallicity grid: "+str(metallicity_grid))
    print("_________________________________________________________________________")
    print("C/O grid: "+str(c_o_grid))

def get_grid(parameter):
    if parameter in ["T", "t", "Temperature", "Temp", "temperature", "temp"]:
        return temperature_gird
    if parameter in ["P", "p", "Pressure", "pressure"]:
        return pressure_grid
    if parameter in ["C/O", "c/o", "C-O", "c-o", "C_O", "c_o"]:
        return c_o_grid
    if parameter in ["M/H", "Met", "met", "Metallicity", "metallicity"]:
        return metallicity_grid
    raise Exception("Your input is not associated with any parameters.\n"
                    "Please check specification for a list of accepted input.")

def get_supported_species():
    return ['H2O', 'CO2', 'OH', 'SO', 'C2H2', 
            'C2H4', 'H2S', 'O2', 'O3', 'HCN',
            'NH3', 'SiO', 'CH4', 'CO', 'C2', 
            'CaH', 'CrH', 'FeH', 'HCl', 'K',
            'MgH', 'N2', 'Na', 'NO', 'NO2',
            'OCS', 'PH3', 'SH', 'SiH', 'SO2',
            'TiH', 'TiO', 'VO'] # add H-

### P, Met are in logarithmic scale; T, C_O are in linear scale
def read_logX(log_P, T, C_O, log_Met, species, return_dict=True):
    '''
    Inquire the traces of a list of chemical species at a given combination of C/O ratio, 
    metallicity, and pressure-temperature profile, assuming equillibrium chemistry.

    Args:
        log_P (float or array of float): 
            Pressure profile provided by the user (in log scale and in bar).
            A single value will be expanded into an array np.full(length, P), where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            10^{-7} to 10^{2} bar are supported.

        T (float or array of float):
            Temperature profile provided by the user (in Kelvin).
             A single value will be expanded into an array np.full(length, T), where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            300 to 4000 K are supported.

        C_O (float or array of float):
            Carbon to Oxygen (C/O) ratio provided by the user.
            A single value will be expanded into an array np.full(length, C_O), where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            0.2 to 2 are supported.

        log_Met (float or array of float):
            Stellar metallicity (in log scale. 0 represents a metallicity of 10^0=1)
            A single value will be expanded into an array np.full(length, Met), where length == max(len(P_array), len(T_array), len(C_O), len(Met)).
            -1 to 4 are supported.

        species (string or list of string):
            A list of chemical species to calculate mixing ratios for.
            Supported species are ['H2O', 'CO2', 'OH', 'SO', 'C2H2', 'C2H4', 'H2S',
                                   'O2', 'O3', 'HCN', 'NH3', 'SiO', 'CH4', 'CO', 'C2', 
                                   'CaH', 'CrH', 'FeH', 'HCl', 'K', 'MgH', 'N2', 
                                   'Na', 'NO', 'NO2', 'OCS', 'PH3', 'SH', 'SiH',
                                   'SO2', 'TiH', 'TiO', 'VO']

        return_dict (boolean):
            If False, return an array of shape (len(species), len(P_array)). The order is the same as species.

    Returns:
        MR_dict (dict) (if return_dict=True):
            A dictionary with keys being the names for species inquired and values being mixing ratios (in log scale).

        MR_array (array of float) (if return_dict=False):
            An array containing the mixing ratios of the inquired species (in the same order as the order in species)

    Prerequisites:
        len(P_array) = len(T_array)
    '''
    
    supported_species = get_supported_species()
    if isinstance(species, str):
        if species not in supported_species: 
            raise Exception("Your species is not supported. Use get_supported_species to check the list of supported species.")
    else:
        for molecule in species:
            if molecule not in supported_species: 
                raise Exception("At least one of your species is not supported. Use get_supported_species to check the list of supported species.")

    np.seterr(divide = 'ignore')
    len_P, len_T, len_C_O, len_Met = np.array(log_P).size, np.array(T).size, np.array(C_O).size, np.array(log_Met).size
    max_len = max(len_P, len_T, len_C_O, len_Met)
    if not (len_P in (1, max_len) and len_T in (1, max_len) and len_C_O in (1, max_len) and len_Met in (1, max_len)):
        raise Exception("Input shape not accepted. The lengths must either be the same or 1 (to be extended).")

    C_O_num, Met_num, T_num, P_num = len(c_o_grid), len(metallicity_grid), len(temperature_gird), len(pressure_grid)
    if len_P == 1:
        log_P = np.full(max_len, log_P)
    if len_T == 1:
        T = np.full(max_len, T)
    if len_C_O == 1:
        C_O = np.full(max_len, C_O)
    if len_Met == 1:
        log_Met = np.full(max_len, log_Met)

    def not_valid(params, grid, is_log):
        if is_log:
            return (10**params[0] < grid[0]) or (10**params[-1] > grid[-1])
        else:
            return (params[0] < grid[0]) or (params[-1] > grid[-1])

    if not_valid(log_P, pressure_grid, True):
        raise Exception("Requested pressure is out of the grid. Use get_grid_info() to check the information about the grid.")
    if not_valid(T, temperature_gird, False):
        raise Exception("Requested temperature is out of the grid. Use get_grid_info() to check the information about the grid.")
    if not_valid(C_O, c_o_grid, False):
        raise Exception("Requested C/O is out of the grid. Use get_grid_info() to check the information about the grid.")
    if not_valid(log_Met, metallicity_grid, True):
        raise Exception("Requested M/H is out of the grid. Use get_grid_info() to check the information about the grid.")

    def interpolate(species):
        array = np.array(database[species+'/log(X)'])
        array = array.reshape(Met_num, C_O_num, T_num, P_num)
        grid = RegularGridInterpolator((np.log10(metallicity_grid), c_o_grid, temperature_gird, np.log10(pressure_grid)), array) # since log(X) is corrected to log space, we should change np.log10(array) into just array.
        return grid(np.vstack((log_Met, C_O, T, log_P)).T)
    if not return_dict:
        if isinstance(species, str):
            return interpolate(species)
        MR_list = []
        for _, molecule in enumerate(species):
            MR_list.append(interpolate(molecule))
        MR_array = np.array(MR_list)
        return MR_array
    else:
        MR_dict = {}
        if isinstance(species, str):
            MR_dict[species] = interpolate(species)
            return MR_dict
        for _, molecule in enumerate(species):
            MR_dict[molecule] = interpolate(molecule)
        return MR_dict
