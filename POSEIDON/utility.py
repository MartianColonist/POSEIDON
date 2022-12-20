''' 
Various miscellaneous functions.

'''

import os
import numpy as np
import pandas as pd
import pymultinest
from numba import jit, cuda
from spectres import spectres
from scipy.interpolate import interp1d as Interp


def create_directories(base_dir, planet_name):
    ''' 
    Create various directories used by POSEIDON, such as output folders
    to store retrieval results.

    Args:
        base_dir (str):
            Directory where the output directory will be created.
        planet_name (str):
            Identifier for planet object (e.g. HD209458b).

    Returns:
        None

    '''
        
    # Specify desired paths of output directories to be created
    output_dir = base_dir + '/POSEIDON_output'
    planet_dir = output_dir + '/' + planet_name
    output_directories = [planet_dir + '/spectra', planet_dir + '/plots',
                          planet_dir + '/retrievals']
            
    # Create main output directory
    if (os.path.exists(output_dir) == False):
        os.mkdir(output_dir)

    # Create directory for this specific planet
    if (os.path.exists(planet_dir) == False):
        os.mkdir(planet_dir)
        
    # Create output subdirectories
    for output_sub_dir in output_directories:
        if (os.path.exists(output_sub_dir) == False):
            os.mkdir(output_sub_dir)
             
    # Create retrieval output directories
    retrieval_dir = planet_dir + '/retrievals'
    if (os.path.exists(retrieval_dir) == False):
        os.mkdir(retrieval_dir)
    if (os.path.exists(retrieval_dir + '/results') == False):
        os.mkdir(retrieval_dir + '/results')
    if (os.path.exists(retrieval_dir + '/MultiNest_raw') == False):
        os.mkdir(retrieval_dir + '/MultiNest_raw')
    if (os.path.exists(retrieval_dir + '/samples') == False):
        os.mkdir(retrieval_dir + '/samples')
        

@jit(nopython = True)
def prior_index(value, grid, start = 0):
    ''' 
    Search a grid to find the previous index closest to a specified value (i.e. 
    the index of the grid where the grid value is last less than the value). 
    This function assumes the input grid monotonically increases.

    Args:
        value (float):
            Value for which the prior grid index is desired.
        grid (np.array of float):
            Input grid.
        start (int):
            Optional start index when existing knowledge is available.

    Returns:
        index (int):
            Prior index of the grid corresponding to the value.

    '''
        
    if (value > grid[-1]):
        return (len(grid) - 1)
    
    # Check if value out of bounds, if so set to edge value
    if (value < grid[0]): value = grid[0]
    if (value > grid[-2]): value = grid[-2]
    
    index = start
    
    for i in range(len(grid)-start):
        if (grid[i+start] > value): 
            index = (i+start) - 1
            break
            
    return index


@cuda.jit(device=True)
def prior_index_GPU(value, grid):
    ''' 
    GPU variant of the 'prior_index' function.

    Search a grid to find the previous index closest to a specified value (i.e. 
    the index of the grid where the grid value is last less than the value). 
    This function assumes the input grid monotonically increases.

    Args:
        value (float):
            Value for which the prior grid index is desired.
        grid (np.array of float):
            Input grid.

    Returns:
        index (int):
            Prior index of the grid corresponding to the value.

    '''
        
    if (value > grid[-1]):
        return (len(grid) - 1)
    
    # Check if value out of bounds, if so set to edge value
    if (value < grid[0]): value = grid[0]
    if (value > grid[-2]): value = grid[-2]
    
    index = 0
    
    for i in range(len(grid)):
        if (grid[i] > value): 
            index = (i) - 1
            break
            
    return index


@cuda.jit(device=True)
def interp_GPU(x_value, x, y):
    '''
    Linear interpolation using a GPU.

    Args:
        x_value (float):
            x value for which y is desired.
        x (np.array of float):
            Input x grid.
        y (np.array of float):
            Input y grid.

    Returns:
        y_interp (float):
            Linearly interpolated value of y evaluated at x_value.

    '''

    prior_index = prior_index_GPU(x_value, x)

    y_interp = y[prior_index] + (((y[prior_index+1]-y[prior_index])/(x[prior_index+1]-x[prior_index])) *
                                 (x_value-x[prior_index]))
                                
    return y_interp


@jit(nopython=True)
def prior_index_V2(value, grid_start, grid_end, N_grid):
    ''' 
    Find the previous index of a *uniformly spaced* grid closest to a specified 
    value. When a uniform grid can be assumed, this function is much faster 
    than 'prior_index' due to there being no need for a loop. However, 
    for non-uniform grids one should still default to 'prior_index'.
    This function assumes the input grid monotonically increases.

    Args:
        value (float):
            The value for which the prior grid index is desired.
        grid_start (float):
            The value at the left edge of the uniform grid (array[0]).
        grid_start (float):
            The value at the right edge of the uniform grid (array[-1]).
        N_grid (int):
            The number of points on the uniform grid.

    Returns:
        (int):
            Prior index of the grid corresponding to the value.

    '''
    
    # Set to lower boundary
    if (value < grid_start):
        return 0
    
    # Set to upper boundary
    elif (value > grid_end):
        return N_grid-1
    
    # Use the equation of a straight line, then round down to integer.
    else:
        i = (N_grid-1) * ((value - grid_start) / (grid_end - grid_start))
        return int(i)


@jit(nopython=True)
def closest_index(value, grid_start, grid_end, N_grid):
    '''
    Same as 'prior_index_V2', but for the closest index (i.e. can also round up).

    Args:
        val (float): 
            The value for which closest index is desired.
        grid_start (float):
            The value at the left edge of the uniform grid (array[0]).
        grid_start (float):
            The value at the right edge of the uniform grid (array[-1]).
        N_grid (int):
            The number of points on the uniform grid.

    Returns:
        (int):
            The index of the uniform grid closest to 'value'.

    '''

    # Set to lower boundary
    if (value < grid_start): 
        return 0
    
    # Set to upper boundary
    elif (value > grid_end):
        return N_grid-1
    
    # Use the equation of a straight line, then round to nearest integer.
    else:
        i = (N_grid-1) * ((value - grid_start) / (grid_end - grid_start))
        if ((i%1) <= 0.5):
            return int(i)     # Round down
        else:
            return int(i)+1   # Round up
        

@cuda.jit(device=True)
def closest_index_GPU(value, grid_start, grid_end, N_grid):
    '''
    GPU variant of the 'closest_index' function.

    Same as 'prior_index_V2', but for the closest index (i.e. can also round up).

    Args:
        val (float): 
            The value for which closest index is desired.
        grid_start (float):
            The value at the left edge of the uniform grid (array[0]).
        grid_start (float):
            The value at the right edge of the uniform grid (array[-1]).
        N_grid (int):
            The number of points on the uniform grid.
    Returns:
        (int):
            The index of the uniform grid closest to 'value'.
    '''

    # Set to lower boundary
    if (value < grid_start): 
        return 0
    
    # Set to upper boundary
    elif (value > grid_end):
        return N_grid-1
    
    # Use the equation of a straight line, then round to nearest integer.
    else:
        i = (N_grid-1) * ((value - grid_start) / (grid_end - grid_start))
        if ((i%1) <= 0.5):
            return int(i)     # Round down
        else:
            return int(i)+1   # Round up


def size_profile(arr):
    '''
    Profile the disk storage size of a numpy array. The resultant size in
    Megabytes is printed to the terminal.

    Args:
        arr (np.array): 
            Any numpy array.

    Returns:
        None

    '''
    
    print("%d Mb" % ((arr.size * arr.itemsize)/1048576.0))


def read_data(data_dir, fname, wl_unit = 'micron', bin_width = 'half', 
              spectrum_unit = '(Rp/Rs)^2', skiprows = None):
    '''
    Read an external dataset file. The expected file format is:

    wavelength | bin half width | spectrum | error on spectrum

    Args:
        data_dir (str):
            Path to the directory containing the data file.
        fname (str):
            File name of data file.
        wl_unit (str):
            Unit of wavelength column (first column in file)
            (Options: micron (or equivalent) / nm / A / m)
        bin_width (str):
            Whether bin width (second column) is half or full width
            (Options: half / full).
        spectrum_unit (str):
            Unit of spectrum (third column) and spectrum errors (fourth column)
            (Options: (Rp/Rs)^2 / Rp/Rs / Fp/Fs / Fp (or equivalent units)).
        skiprows (int):
            The number of rows to skip (e.g. use 1 if file has a header line).

    Returns:
        wl_data (np.array of float): 
            Bin centre wavelengths of data points (μm).
        half_bin (np.array of float): 
            Bin half widths of data points (μm).
        spectrum (np.array of float):
            Transmission or emission spectrum dataset.
        err (np.array of float):
            1σ error bar on spectral data.

    '''
    
    # Load data file
    data = pd.read_csv(data_dir + '/' + fname, sep = '[\s]{1,20}', 
                       header = None, skiprows = skiprows, engine = 'python')

    # Load wavelength and half bin width, then convert both to μm
    if (wl_unit in ['micron', 'um', 'μm']):
        wl_data = np.array(data[0])
        wl_bin = np.array(data[1])
    elif (wl_unit == 'nm'):
        wl_data = np.array(data[0]) * 1e-3
        wl_bin = np.array(data[1]) * 1e-3
    elif (wl_unit == 'A'):
        wl_data = np.array(data[0]) * 1e-4
        wl_bin = np.array(data[1]) * 1e-4
    elif (wl_unit == 'm'):
        wl_data = np.array(data[0]) * 1e6
        wl_bin = np.array(data[1]) * 1e6
    else:
        raise Exception("Error: unrecognised wavelength unit when reading data.")

    # Divide bin widths by 2 if the file contains full bin widths
    if (bin_width == 'half'):
        half_bin = wl_bin
    elif (bin_width == 'full'):
        half_bin = wl_bin/2.0
    else:
        raise Exception("Error: unrecognised bin width unit when reading data.")

    # Load spectrum and errors (converting to transit depth if Rp/Rs provided).
    if (spectrum_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', 
                          'transit_depth', 'eclipse_depth', 
                          'Fp/Fs', 'Fp/F*', 'Fp']):
        spectrum = np.array(data[2])
        err = np.array(data[3])
    elif (spectrum_unit == 'ppm'):
        spectrum = np.array(data[2]) * 1e-6
        err = np.array(data[3]) * 1e-6
    elif (spectrum_unit in ['%', 'percent']):
        spectrum = np.array(data[2]) * 1e-2
        err = np.array(data[3]) * 1e-2
    elif (spectrum_unit in ['Rp/Rs', 'Rp/R*']):
        spectrum = (np.array(data[2]))**2
        err = (2.0 * (np.array(data[3]) / np.array(data[2]))) * spectrum  # Error propagation for Rp/Rs -> (Rp/Rs)^2
    else:
        raise Exception("Error: unrecognised spectrum unit when reading file.")
    
    return wl_data, half_bin, spectrum, err

    
def read_spectrum(planet_name, fname, wl_unit = 'micron'):
    '''
    Read a previously computed spectrum from the POSEIDON output folder
    (POSEIDON_output/planet_name/spectra).

    Args:
        planet_name (str):
            Identifier for planet object (e.g. HD209458b).
        fname (str):
            Name of spectrum file.
        wl_unit (str):
            Unit of wavelength column (first column in file)
            (Options: micron (or equivalent) / nm / A / m)

    Returns:
        wavelength (np.array of float): 
            Model wavelength grid (μm).
        spectrum (np.array of float):
            Transmission or emission spectrum.

    '''

    # Load POSEIDON directory location where the spectrum is stored
    input_dir = './POSEIDON_output/' + planet_name + '/spectra/'
    
    # Open file
    data = pd.read_csv(input_dir + fname, sep = '[\s]{1,20}', 
                       engine = 'python', header=None)

    # Load wavelength then convert to μm
    if (wl_unit in ['micron', 'um', 'μm']):
        wavelength = np.array(data[0])
    elif (wl_unit == 'nm'):
        wavelength = np.array(data[0]) * 1e-3
    elif (wl_unit == 'A'):
        wavelength = np.array(data[0]) * 1e-4
    elif (wl_unit == 'm'):
        wavelength = np.array(data[0]) * 1e6
    else:
        raise Exception("Error: unrecognised wavelength unit when reading file.")

    # Load spectrum (transit or eclipse depth)
    spectrum = np.array(data[1])
    
    return wavelength, spectrum


def read_PT_file(PT_file_dir, PT_file_name, P_grid, P_unit = 'bar',
                 P_column = None, T_column = None, skiprows = None):
    '''
    Read an external file containing the temperature as a function of pressure.

    Args:
        PT_file_dir (str):
            Directory containing the pressure-temperature file.
        PT_file_name (str):
            Name of pressure-temperature profile file.
        P_grid (np.array of float):
            POSEIDON model pressure grid (to interpolate external profile onto).
        P_unit (str):
            Pressure unit in external file
            (Options: bar / Pa / atm).
        P_column (int):
            File column containing the pressure.
        T_column (int):
            File column containing the temperature.
        skiprows (int):
            The number of rows to skip (e.g. use 1 if file has a header line).

    Returns:
        T_interp (np.array of float): 
            Temperature profile from external file interpolated onto the
            POSEIDON model's pressure grid (K).

    '''

    # If the user is running the tutorial, point to the reference data folder
    if (PT_file_dir == 'Tutorial/TRAPPIST-1e'):
        PT_file_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
                                      '.', 'reference_data/models/TRAPPIST-1e/'))
    
    PT_file = pd.read_csv(PT_file_dir + '/' + PT_file_name, sep = '[\s]{1,20}', 
                          header = None, skiprows = skiprows, engine = 'python')
    
    # Read pressure
    if (P_column != None):
        P_raw = np.array(PT_file[int(P_column)-1])
    else:
        P_raw = np.array(PT_file[0])


    # Convert pressure grid from file into bar
    if (P_unit == 'bar'):
        P_raw = P_raw
    elif (P_unit == 'Pa'):
        P_raw = P_raw / 1e5
    elif (P_unit == 'atm'):
        P_raw = P_raw * 1.01325
    else:
        raise Exception("Error: unrecognised pressure unit when reading file.")

    # Read temperature
    if (T_column != None):
        T_raw = np.array(PT_file[int(T_column)-1])
    else:
        T_raw = np.array(PT_file[1])
    
    # Flip arrays if necessary so that arrays begin at bottom of the atmosphere
    if (P_raw[0] < P_raw[-1]):
        P_raw = P_raw[::-1]
        T_raw = T_raw[::-1]

    # Interpolate the file temperature profile onto the POSEIDON model P grid
    PT_interp = Interp(np.log10(P_raw), T_raw, kind='linear', bounds_error=False, 
                       fill_value=(T_raw[-1], T_raw[0]))
    T_interp = PT_interp(np.log10(P_grid))                  
    
    return T_interp


def read_chem_file(chem_file_dir, chem_file_name, P_grid, chem_species_in_file, 
                   chem_species_in_model, P_unit = 'bar', skiprows = None):
    '''
    Read an external file containing mixing ratios as a function of pressure.

    Args:
        chem_file_dir (str):
            Directory containing the mixing ratio file.
        chem_file_name (str):
            Name of mixing ratio file.
        P_grid (np.array of float):
            POSEIDON model pressure grid (to interpolate mixing ratios onto).
        chem_species_in_file (list of str):
            The chemical species included in the external file.
        chem_species_in_model (list of str):
            The chemical species included in the POSEIDON model.
        P_unit (str):
            Pressure unit in external file
            (Options: bar / Pa / atm).
        skiprows (int):
            The number of rows to skip (e.g. use 1 if file has a header line).

    Returns:
        X_interp (2D np.array of float): 
            Mixing ratio profiles from external file interpolated onto the
            POSEIDON model's pressure grid. Only includes the chemical species
            specified in the POSEIDON model (i.e. chem_species_in_model).

    '''
    
    # If the user is running the tutorial, point to the reference data folder
    if (chem_file_dir == 'Tutorial/TRAPPIST-1e'):
        chem_file_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
                                        '.', 'reference_data/models/TRAPPIST-1e/'))

    chem_file_input = pd.read_csv(chem_file_dir + '/' + chem_file_name, sep = '[\s]{1,20}', 
                                  header = None, skiprows = skiprows, engine = 'python')
    
    # Read pressure and mixing ratios
    P_raw = np.array(chem_file_input)[:,0]
    X_raw = np.array(chem_file_input)[:,1:]

    # Flip arrays if necessary so that arrays begin at bottom of the atmosphere
    if (P_raw[0] < P_raw[-1]):
        P_raw = P_raw[::-1]
        X_raw = X_raw[::-1,:]
    
    # Convert pressure grid from file into bar
    if (P_unit == 'bar'):
        P_raw = P_raw
    elif (P_unit == 'Pa'):
        P_raw = P_raw / 1e5
    elif (P_unit == 'atm'):
        P_raw = P_raw * 1.01325
    else:
        raise Exception("Error: unrecognised pressure unit when reading file.")

    # Initialise interpolated mixing ratio array
    X_interp = np.zeros(shape=(len(chem_species_in_model), len(P_grid)))  

    # Loop over chemical species, interpolating each onto the POSEIDON model P grid        
    for q in range(len(chem_species_in_model)):
    
        species = chem_species_in_model[q]

        # Chemicals not included in the external file will have zero mixing ratio
        if (species in chem_species_in_file):

            # Find column index in file containing the model chemical species
            idx = chem_species_in_file.index(species)

            # Interpolate from chemistry pressure grid onto model pressure grid
            chem_interp = Interp(np.log10(P_raw), X_raw[:,idx],
                                 kind = 'linear', bounds_error = False, 
                                 fill_value = (X_raw[-1,idx], 
                                               X_raw[0,idx]))
            X_interp[q,:] = chem_interp(np.log10(P_grid))                  
    
    return X_interp


def bin_spectrum(wl_native, spectrum_native, R_bin, err_data = []):
    '''
    Bin a model spectrum down to a specific spectral resolution. 
    
    This is a wrapper around the Python package SpectRes (for details on the 
    resampling algorithm, see https://arxiv.org/abs/1705.05165).

    Args:
        wl_native (np.array of float): 
            Input wavelength grid (μm).
        spectrum_native (np.array of float): 
            Input spectrum.
        R_bin (float or int):
            Spectral resolution (R = wl/dwl) to re-bin the spectrum onto.
        err_data (np.array of float):
            1σ errors on the spectral data.

    Returns:
        wl_binned (np.array of float): 
            New wavelength grid spaced at R = R_bin (μm).
        spectrum_binned (np.array of float):
            Re-binned spectrum at resolution R = R_bin.
        err_binned (np.array of float):
            Re-binned errors at resolution R = R_bin.

    '''
        
    # Create binned wavelength grid at resolution R_bin
    delta_log_wl_bins = 1.0/R_bin
    N_wl_bins = (np.log(wl_native[-1]) - np.log(wl_native[0])) / delta_log_wl_bins
    N_wl_bins = np.around(N_wl_bins).astype(np.int64)
    log_wl_binned = np.linspace(np.log(wl_native[0]), np.log(wl_native[-1]), N_wl_bins)    
    wl_binned = np.exp(log_wl_binned)
    
    # Call Spectres routine
    if (err_data != []):
        spectrum_binned, err_binned = spectres(wl_binned, wl_native, spectrum_native,
                                               spec_errs = err_data, verbose = False)

        # Cut out first and last values to avoid SpectRes boundary NaNs
        wl_binned = wl_binned[1:-1]
        spectrum_binned = spectrum_binned[1:-1]

        # Replace Spectres boundary NaNs with second and penultimate values
        err_binned[0] = err_binned[1]
        err_binned[-1] = err_binned[-2]

    # Call Spectres routine
    else:
        spectrum_binned = spectres(wl_binned, wl_native, spectrum_native,
                                   verbose = False)

        # Cut out first and last values to avoid SpectRes boundary NaNs
        wl_binned = wl_binned[1:-1]
        spectrum_binned = spectrum_binned[1:-1]
        err_binned = None

    return wl_binned, spectrum_binned, err_binned
                

def write_spectrum(planet_name, model_name, spectrum, wl):
    ''' 
    Writes a given model spectrum.
    
    '''

    # Identify output directory location where the spectrum will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/spectra/'

    # Write spectrum
    f = open(output_dir + planet_name + '_' + model_name + '_spectrum.txt', 'w')
    
    for i in range(len(wl)):
        f.write('%.8e %.8e \n' %(wl[i], spectrum[i]))   # wl (um) | Rp/Rs^2
        
    f.close()


def write_retrieved_spectrum(retrieval_name, wl, spec_low2, 
                             spec_low1, spec_median, spec_high1, spec_high2):
    '''
    ADD DOCSTRING
    '''

    # Identify output directory location where the retrieved spectrum will be saved
    output_dir = '../samples/'
    
    # Write retrieved spectrum
    f = open(output_dir + retrieval_name + '_spectrum_retrieved.txt', 'w')
    
    # Write top line
    f.write('wl (μm) | spectrum: -2σ | spectrum: -1σ | spectrum: median ' + 
            '| spectrum: +1σ | spectrum: +2σ \n')
    
    for i in range(len(wl)):
        f.write('%.8e %.8e %.8e %.8e %.8e %.8e \n' %(wl[i], spec_low2[i], spec_low1[i], 
                                                     spec_median[i], spec_high1[i], spec_high2[i]))
        
    f.close()


def write_retrieved_PT(retrieval_name, P, T_low2, T_low1, 
                       T_median, T_high1, T_high2):
    '''
    ADD DOCSTRING
    '''

    # Identify output directory location where the retrieved P-T profile will be saved
    output_dir = '../samples/'
    
    # Write retrieved spectrum
    f = open(output_dir + retrieval_name + '_PT_retrieved.txt', 'w')
    
    # Write top line
    f.write('P (bar) | T: -2σ | T: -1σ  | T: median | T: +1σ | T: +2σ \n')
    
    for i in range(len(P)):
        f.write('%.8e %.8e %.8e %.8e %.8e %.8e \n' %(P[i], T_low2[i], T_low1[i], 
                                                     T_median[i], T_high1[i], T_high2[i]))
        
    f.close()


def write_retrieved_log_X(retrieval_name, chemical_species, P, log_X_low2, 
                          log_X_low1, log_X_median, log_X_high1, log_X_high2):
    '''
    ADD DOCSTRING
    '''

    # Identify output directory location where the retrieved mixing ratio profiles will be saved
    output_dir = '../samples/'
    
    # Write retrieved spectrum
    f = open(output_dir + retrieval_name + '_log_X_retrieved.txt', 'w')

    # First line of file lists the chemical species included in this model
    chem_species_string = 'Chemical species: '

    # Add each chemical species to top line string
    for q in range(len(chemical_species)):
        chem_species_string += chemical_species[q]
        if (q < len(chemical_species)-1):
            chem_species_string += ' '   # Don't add a space for final chemical

    # Write top line
    f.write(chem_species_string + '\n')

    # For each chemical species, write a block with the retrieved mixing ratio profile
    for q in range(len(chemical_species)):

        log_species = 'log_' + chemical_species[q]
    
        f.write('P (bar) | ' + log_species + ': -2σ | ' + log_species + ': -1σ | ' +
                log_species + ': median | ' + log_species + ': +1σ | ' + 
                log_species + ': +2σ \n')
    
        for i in range(len(P)):
            f.write('%.8e %.8f %.8f %.8f %.8f %.8f \n' %(P[i], log_X_low2[q,i],
                                                         log_X_low1[q,i], log_X_median[q,i], 
                                                         log_X_high1[q,i], log_X_high2[q,i]))
        
    f.close()
    

def read_retrieved_spectrum(planet_name, model_name, retrieval_name = None):
    '''
    ADD DOCSTRING
    '''

    if (retrieval_name is None):
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location where the retrieved spectrum is located
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/samples/'

    # Find retrieved spectrum file
    fname = output_dir + retrieval_name + '_spectrum_retrieved.txt'

    # Read retrieved spectrum confidence intervals
    spec_file = pd.read_csv(fname, sep = '[\s]{1,20}', engine = 'python', 
                            header = None, skiprows = 1)

    wl = np.array(spec_file[0])           # Wavelengths (um)
    spec_low2 = np.array(spec_file[1])    # -2σ
    spec_low1 = np.array(spec_file[2])    # -1σ
    spec_median = np.array(spec_file[3])  # Median
    spec_high1 = np.array(spec_file[4])   # +1σ
    spec_high2 = np.array(spec_file[5])   # +2σ
    
    return wl, spec_low2, spec_low1, spec_median, spec_high1, spec_high2


def read_retrieved_PT(planet_name, model_name, retrieval_name = None):
    '''
    ADD DOCSTRING
    '''

    if (retrieval_name is None):
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location where the retrieved P-T profile is located
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/samples/'

    # Find retrieved P-T profile file
    fname = output_dir + retrieval_name + '_PT_retrieved.txt'

    # Read retrieved temperature confidence intervals
    PT_file = pd.read_csv(fname, sep = '[\s]{1,20}', engine = 'python', 
                          header = None, skiprows = 1)

    P = np.array(PT_file[0])         # Pressure (bar)
    T_low2 = np.array(PT_file[1])    # -2σ
    T_low1 = np.array(PT_file[2])    # -1σ
    T_median = np.array(PT_file[3])  # Median
    T_high1 = np.array(PT_file[4])   # +1σ
    T_high2 = np.array(PT_file[5])   # +2σ
    
    return P, T_low2, T_low1, T_median, T_high1, T_high2


def read_retrieved_log_X(planet_name, model_name, retrieval_name = None):
    '''
    ADD DOCSTRING
    '''

    if (retrieval_name is None):
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location where the retrieved P-T profile is located
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/samples/'

    # Find retrieved P-T profile file
    fname = output_dir + retrieval_name + '_log_X_retrieved.txt'

    # Read file to figure out number of layers and chemical species
    file = open(fname, 'r')
    lines = file.readlines()
  
    line_number = 0
    block_line_numbers = []

    # Looping through the file to find the chemical species and number of layers
    for line in lines:

        line_number += 1

        # Read in list of chemical species from top line
        if ('Chemical species:' in line):
            chemical_species = np.array(line[18:].strip().split(' '))

        # Note the line numbers where each chemical species block begins
        if ('P (bar)' in line):
            block_line_numbers.append(line_number)

    N_species = len(chemical_species)
    N_D = (block_line_numbers[1] - block_line_numbers[0]) - 1

    file.close()

    # Initialise the retrieved mixing ratio profile arrays
    log_X_low2 = np.zeros(shape=(N_species, N_D))
    log_X_low1 = np.zeros(shape=(N_species, N_D))
    log_X_median = np.zeros(shape=(N_species, N_D))
    log_X_high1 = np.zeros(shape=(N_species, N_D))
    log_X_high2 = np.zeros(shape=(N_species, N_D))

    # For each chemical species, read in the retrieved mixing ratio block
    for q in range(N_species):

        # Read retrieved mixing ratio confidence intervals
        X_file = pd.read_csv(fname, sep = '[\s]{1,20}', header = None, 
                             skiprows = block_line_numbers[q], nrows = N_D,
                             engine = 'python')

        P = np.array(X_file[0])                  # Pressure (bar)
        log_X_low2[q,:] = np.array(X_file[1])    # -2σ
        log_X_low1[q,:] = np.array(X_file[2])    # -1σ
        log_X_median[q,:] = np.array(X_file[3])  # Median
        log_X_high1[q,:] = np.array(X_file[4])   # +1σ
        log_X_high2[q,:] = np.array(X_file[5])   # +2σ
    
    return P, chemical_species, log_X_low2, log_X_low1, log_X_median, \
           log_X_high1, log_X_high2
 

def plot_collection(new_y, new_x, collection = []):
    
    ''' Convenient function to combine distinct spectra and wavelength
        grids into a single object for plotting purposes.
    
    '''
        
    collection.append((new_y, new_x))
    
    return collection
    
    
def round_sig_figs(value, sig_figs):
    
    ''' Round a quantity to a specified number of significant figures.
    
    '''
    
    return round(value, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)
    

def confidence_intervals(sample_draws, array, length, integer=False):
    
    ''' Order posterior samples to create 1 & 2 sigma contours + median values.
    
    '''
        
    prob = np.empty(sample_draws)
    prob.fill(1.0/sample_draws)
        
    sig_1 = 0.5 + 0.6826/2.0
    sig_2 = 0.5 + 0.954/2.0
    sig_3 = 0.5 + 0.997/2.0
        
    if (length > 0):
            
        arr_low3 = np.zeros(shape=(length))
        arr_low2 = np.zeros(shape=(length))
        arr_low1 = np.zeros(shape=(length))
        arr_median = np.zeros(shape=(length))
        arr_high1 = np.zeros(shape=(length))
        arr_high2 = np.zeros(shape=(length))
        arr_high3 = np.zeros(shape=(length))
            
        for i in range(length):
        
            arr_ordered = list(zip(prob[:], array[:, i]))
            arr_ordered.sort(key=lambda x: x[1])
            arr_ordered = np.array(arr_ordered)
    
            arr_ordered[:,0] = arr_ordered[:,0].cumsum()
    
            arr_ordered_interp = lambda x: np.interp(x, arr_ordered[:,0], arr_ordered[:,1],
                                                     left=arr_ordered[0,1], right=arr_ordered[-1,1])
    
            arr_low3[i] = arr_ordered_interp(1-sig_3)
            arr_low2[i] = arr_ordered_interp(1-sig_2)
            arr_low1[i] = arr_ordered_interp(1-sig_1)
            arr_median[i] = arr_ordered_interp(0.5)
            arr_high1[i] = arr_ordered_interp(sig_1)
            arr_high2[i] = arr_ordered_interp(sig_2) 
            arr_high3[i] = arr_ordered_interp(sig_3) 
            
        return arr_low3, arr_low2, arr_low1, arr_median, arr_high1, arr_high2, arr_high3
            
    if (length == 0):
            
        arr_ordered = list(zip(prob[:], array[:]))
        arr_ordered.sort(key=lambda x: x[1])
        arr_ordered = np.array(arr_ordered)
    
        arr_ordered[:,0] = arr_ordered[:,0].cumsum()
    
        arr_ordered_interp = lambda x: np.interp(x, arr_ordered[:,0], arr_ordered[:,1],
                                                 left=arr_ordered[0,1], right=arr_ordered[-1,1])
            
        if (integer == True):
                
            arr_low3 = np.around(arr_ordered_interp(1-sig_3)).astype(np.int64)
            arr_low2 = np.around(arr_ordered_interp(1-sig_2)).astype(np.int64)
            arr_low1 = np.around(arr_ordered_interp(1-sig_1)).astype(np.int64)
            arr_median = np.around(arr_ordered_interp(0.5)).astype(np.int64)
            arr_high1 = np.around(arr_ordered_interp(sig_1)).astype(np.int64)
            arr_high2 = np.around(arr_ordered_interp(sig_2)).astype(np.int64)
            arr_high3 = np.around(arr_ordered_interp(sig_3)).astype(np.int64)
                
        elif (integer == False):
                
            arr_low3 = arr_ordered_interp(1-sig_3)
            arr_low2 = arr_ordered_interp(1-sig_2)
            arr_low1 = arr_ordered_interp(1-sig_1)
            arr_median = arr_ordered_interp(0.5)
            arr_high1 = arr_ordered_interp(sig_1)
            arr_high2 = arr_ordered_interp(sig_2) 
            arr_high3 = arr_ordered_interp(sig_3) 
                
        return arr_low3, arr_low2, arr_low1, arr_median, arr_high1, arr_high2, arr_high3
          
           
def write_params_file(param_names, results_prefix):
    
    ''' Write file containing a single column listing the free parameters
        used in a retrieval. This file can be read in later when generating 
        corner plots at a future time.
        
    '''

    param_file = open(results_prefix + '_param_names.txt','w')

    for param in param_names:
        param_file.write(param + '\n')
    
    
def write_samples_file(samples, param_names, n_params, results_prefix):
    
    ''' Write file containing the equally weighted posterior samples for 
        each free parameter in a retrieval.
    
    '''
    
    header = ''    # String to store parameter names as a file header
    
    for i in range(n_params):
        header += param_names[i]
        if (i != n_params-1):
            header += ' | '    # Add a '|' between each parameter name
            
    # Save samples to file
    np.savetxt(results_prefix + '_samples.txt', samples, 
               comments = '', header = header)
    
        
def find_str(string, substring):
    
    ''' Find positional index within a string where a substring starts.
    
    '''
    
    index = 0

    if substring in string:
        s = substring[0]
        for char in string:
            if char == s:
                if string[index:index+len(substring)] == substring:
                    return index

            index += 1

    return -1

    
def generate_latex_param_names(param_names):
    
    ''' Generate LaTeX code for an array of parameters for use in plots.
    
    '''
    
    # Define lower and upper Greek alphabets for later use
    greek_letters_low = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
                         'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu',
                         'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
                         'phi', 'chi', 'psi', 'omega']
    greek_letters_up =  ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta',
                         'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu',
                         'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon',
                         'Phi', 'Chi', 'Psi', 'Omega']
    
    # Define key parameters used in subscripts of parameter names
    phrases = ['high', 'mid', 'deep', 'ref', 'DN', 'term', 'Morn', 'Even', 'Day', 'Night', 
               'cloud', 'rel', '0', 'het', 'phot', 'p']
    
    # Initialise output array
    latex_names = []
    
    # Loop over each free parameter
    for param in param_names:
        
        components = []    # Array of 'special' components for this parameter (e.g. 'log', 'bar')
        idxs = []          # Indices where each component starts
        lens= []           # Number of characters in each component
        
        captured_characters = np.zeros(len(param)).astype(np.int32)  # Stays zero for entries with solo letters (e.g. 'H' in 'H2O')
                
        # Find which components are in this parameter's name, and where they occur
        if ('log' in param):
            idx = find_str(param, 'log')  # Find index where 'log' starts
            idxs += [idx]   
            components += ['log']
            lens += [3]
            captured_characters[idx:idx+3] = 1
            
        if ('bar' in param):
            idx = find_str(param, 'bar')   # Find index where 'bar' starts
            idxs += [idx]   
            components += ['bar']
            lens += [3]
            captured_characters[idx:idx+3] = 1
            
        for letter in greek_letters_low:
            if (letter == 'eta'):   # Special check for 'eta', since  contained in 'beta' and 'theta'
                if ((letter in param) and ('theta' not in param) and ('beta' not in param)):
                    idx = find_str(param, letter)   # Find index where Greek letter starts
                    idxs += [idx]
                    components += ['greek_low']
                    lens += [len(letter)]
                    captured_characters[idx:idx+len(letter)] = 1
            else:
                if (letter in param):
                    idx = find_str(param, letter)   # Find index where Greek letter starts
                    idxs += [idx]
                    components += ['greek_low']
                    lens += [len(letter)]
                    captured_characters[idx:idx+len(letter)] = 1
            
        for letter in greek_letters_up:
            if (letter in param):
                idx = find_str(param, letter)   # Find index where Greek letter starts
                idxs += [idx]
                components += ['greek_up']
                lens += [len(letter)]
                captured_characters[idx:idx+len(letter)] = 1
                
        for phrase in phrases:
            if (phrase == 'p'):      # Special check for 'p' to avoid double counting phrases containing 'p'
                if ('_p_' in param):
                    idx = find_str(param, phrase)   # Find index where phrase starts
                    idxs += [idx]
                    components += ['phrase']
                    lens += [len(phrase)]
                    captured_characters[idx:idx+len(phrase)] = 1
            elif (phrase == 'het'):      # Special check for 'het' to avoid double counting phrases containing 'het'
                if ('_het' in param):
                    idx = find_str(param, phrase)   # Find index where phrase starts
                    idxs += [idx]
                    components += ['phrase']
                    lens += [len(phrase)]
                    captured_characters[idx:idx+len(phrase)] = 1
            else:
                if (phrase in param):
                    idx = find_str(param, phrase)   # Find index where phrase starts
                    idxs += [idx]
                    components += ['phrase']
                    lens += [len(phrase)]
                    captured_characters[idx:idx+len(phrase)] = 1
        
        for idx, char in enumerate(param):
            if (char == '_'):
                idxs += [idx]
                components += ['_']
                lens += [1]
                captured_characters[idx:idx+1] = 1
            if (char in ['-', '+']):
                idxs += [idx]
                components += ['charge']
                lens += [1]
                captured_characters[idx:idx+1] = 1
            if ((char.isdigit()) and (param[idx-1] != '_')):
                idxs += [idx]
                components += ['digit']
                lens += [1]
                captured_characters[idx:idx+1] = 1
            else:
                if (captured_characters[idx] == 0):
                    idxs += [idx]
                    components += ['letter']
                    lens += [1]
                    
        # Convert from lists to numpy arrays
        components = np.array(components)
        idxs = np.array(idxs)
        lens = np.array(lens)
        
        # Sort arrays
        components_sort = components[np.argsort(idxs)]
        idxs_sort = idxs[np.argsort(idxs)]
        lens_sort = lens[np.argsort(idxs)]
        
        # Count number of letters, digits, and phrases appearing in this parameter name
        N_letter = np.count_nonzero(components_sort == 'letter')
        N_digit = np.count_nonzero(components_sort == 'digit')
        N_charge = np.count_nonzero(components_sort == 'charge')
        N_phrase = np.count_nonzero(components_sort == 'phrase')
        N_letter_digit = N_letter + N_digit + N_charge
        
        # Start LaTeX string for this parameter
        latex_name = '$'
        
        # Define variable to track whether we have an open bracket {}
        bar_bracket_open = 0
        letter_digit_bracket_open = 0
        phrase_bracket_open = 0

        # If first set of characters have no special component
        if (idxs_sort[0] != 0):
            latex_name += param[:idxs_sort[0]]
        
        # Overline (for average parameters) takes priority in string order
        if ('bar' in components_sort):
            
            # Begin LaTeX string with open overline bracket
            latex_name += '\overline{'
            
            bar_bracket_open = 1
            
            # Remove overline component (and preceding backspace) now it has been dealt with
            j = np.where(np.copy(components_sort) == 'bar')[0][0]
            components_sort = np.delete(components_sort, j)
            components_sort = np.delete(components_sort, j-1)
            idxs_sort = np.delete(idxs_sort, j)
            idxs_sort = np.delete(idxs_sort, j-1)
            lens_sort = np.delete(lens_sort, j)
            lens_sort = np.delete(lens_sort, j-1)
            
            # Remove 'bar' from parameter name
            param = param.replace('_bar', '')
            idxs_sort[j-1:] -= 4   # Any indices after 'bar' move three to the left
            
        # Work through each component, adding LaTeX code in the appropriate places
        for i in range(len(components_sort)):
            
            if (components_sort[i] == 'log'):
                latex_name += ('\\' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + ' \, ')
            elif (components_sort[i] in ['greek_low', 'greek_up']):
                latex_name += ('\\' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]])
            elif (components_sort[i] in ['digit', 'letter', 'charge']):
                if (letter_digit_bracket_open == 0):
                    latex_name += '\mathrm{'
                    letter_digit_bracket_open = 1
                if (letter_digit_bracket_open == 1):
                    if (N_letter_digit >= 2):
                        if (components_sort[i] == 'digit'):
                            latex_name += ('_' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + ' ')  
                        elif (components_sort[i] == 'letter'):
                            latex_name += param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]]
                        elif (components_sort[i] == 'charge'):
                            latex_name += ('^' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + ' ')
                    elif (N_letter_digit == 1):
                        if (components_sort[i] == 'digit'):
                            latex_name += ('_' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + '}')  
                        elif (components_sort[i] == 'letter'):
                            latex_name += (param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + '}')
                        elif (components_sort[i] == 'charge'):
                            latex_name += ('^' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + '}')
                        letter_digit_bracket_open = 0
                N_letter_digit -= 1
            elif (components_sort[i] == 'phrase'):
                if (bar_bracket_open == 1):
                    latex_name += '}'
                    bar_bracket_open = 0
                if (phrase_bracket_open == 0):
                    latex_name += '_{\mathrm{'
                    phrase_bracket_open = 1
                if (phrase_bracket_open == 1):
                    if (N_phrase >= 2):
                        latex_name += (param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + ', \, ')
                    elif (N_phrase == 1):
                        latex_name += (param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + '}}')
                        phrase_bracket_open = 0
                N_phrase -= 1
        if (bar_bracket_open == 1):
            latex_name += '}'
            
        latex_name += '$'
        
        latex_names += [latex_name]
    
    return latex_names


def return_quantiles(stats, param, i, quantile = '1 sigma'):
    
    ''' Extract the median, +/- N sigma (specified by 'quantile'), string 
        formatter and units for a given free parameter.
        
        Note: 'quantile' supports 1, 2, 3, or 5 sigma.
    
    '''
    
    quantile = quantile.replace(' ', '')   # Remove space to match PyMultiNest key

    # Load PyMultiNest output to extract median and specified sigma quantiles
    m = stats['marginals'][i]
        
    centre = m['median']
    low, high = m[quantile]
        
    sig_p = high - centre
    sig_m = centre - low
    
    # Specify number of decimal places for rounding each quantity
    if ((param == 'T') or ('T_' in param)):
        formatter = '{:.1f}'
        unit = 'K'
    elif ('R_p' in param):
        decimal_count = 1   # Find minimum number of decimal places for R_p
        while ((round(sig_p, decimal_count) == 0.0) or 
               (round(sig_m, decimal_count) == 0.0)):
            decimal_count += 1
        formatter = '{:.' + str(decimal_count) + 'f}'
        unit = 'RJ'
    elif ('log' in param):
        formatter = '{:.2f}'
        unit = ''
    else:
        formatter = '{:.2f}'
        unit = ''
        
    return sig_m, centre, sig_p, formatter, unit

    
def write_summary_file(results_prefix, planet_name, retrieval_name, 
                       sampling_algorithm, n_params, N_live, ev_tol, param_names, 
                       stats, ln_Z, ln_Z_err, reduced_chi_square, chi_square,
                       dof, best_fit_params, wl, R, instruments, datasets):
    ''' 
    Write a file summarising the main results from a POSEIDON retrieval.
        
    Contains the key model stats (Bayesian evidence and best-fitting chi^2)
    and the +/- 1, 2, 3, and 5 sigma constraints, alongside other helpful
    information for future reference.
    
    '''
    
    # Specify location where results file will be saved
    summary_file = open(results_prefix + '_results.txt','w')

    # Define fixed properties for file
    stats_formatter = '{:.2f}'                         # String formatter for evidence and chi-square
    max_param_len = len(max(param_names, key = len))   # Used to align '=' signs for parameters

    if (R is not None):
        wl_grid_description = ('R = ' + str(R))
    else:
        wl_grid_description = (str(len(wl)) + ' wavelength points')

    # Start populating lines of results file
    lines = ['Ψ*************************************Ψ\n',
             'Ψ#####################################Ψ\n',
             'Ψ#####                           #####Ψ\n',
             'Ψ##### POSEIDON Retrieval Output #####Ψ\n',
             'Ψ#####                           #####Ψ\n',
             'Ψ#####################################Ψ\n',
             'Ψ*************************************Ψ\n',
             '\n',
             '#################################\n',
             '\n',
             'PLANET: ' + planet_name + '\n',
             '\n',
             'Model: ' + retrieval_name + '\n',
             '\n',
             '#################################\n',
             '\n',
             'Model wavelength range:\n',
             '\n',
             '-> ' + str(wl[0]) + ' - ' + str(wl[-1]) + ' um @ ' + wl_grid_description + '\n',
             '\n',
             '#################################\n',
             '\n',
             'Datasets:\n',
             '\n']
    
    # Write datasets and instruments used in retrieval
    for i in range(len(datasets)):
        
        lines += ['-> ' + instruments[i] + ' (' + datasets[i] + ')\n']
           
    # Add model stats
    lines += ['\n',
              '#################################\n',
              '\n',
              'Algorithm = ' + sampling_algorithm + '\n',
              'N_params = ' + str(n_params) + '\n',
              'N_live = ' + str(N_live) + '\n',
              'evidence_tol = ' + str(ev_tol) + '\n',
              '\n',
              '#################################\n',
              '\n',
              'Model Bayesian evidence:\n',
              '\n',
              '-> ln Z = ' + stats_formatter.format(ln_Z) + ' +/- ' + stats_formatter.format(ln_Z_err) + '\n',
              '\n',
              '#################################\n']

    # Add chi^2 statistics
    if (np.isnan(reduced_chi_square) == False):
        lines += ['\n',
                'Best reduced chi-square:\n',
                '\n',
                '-> chi^2_red = ' + stats_formatter.format(reduced_chi_square) + '\n',
                '\n',
                '-> degrees of freedom = ' + str(dof) + '\n',
                '\n',
                '-> chi^2 = ' + stats_formatter.format(chi_square) + '\n',
                '\n',
                '#################################\n']
    else:
        lines += ['\n',
                'Reduced chi-square undefined because N_params >= N_data!\n',
                '\n',
                '-> chi^2_red = Undefined\n',
                '\n',
                '-> degrees of freedom = Undefined\n',
                '\n',
                '-> chi^2 = ' + stats_formatter.format(chi_square) + '\n',
                '\n',
                '#################################\n']
    
    # Add retrieved parameter constraints
    lines += ['\n',
              '******************************************\n',
              '1 σ constraints\n',
              '******************************************\n',
              '\n']
    
    # Write 1 sigma parameter constraints
    for i, param in enumerate(param_names):
        
        sig_m, centre, \
        sig_p, formatter, unit = return_quantiles(stats, param, i, quantile = '1 sigma')

        lines += [param + ' '*(max_param_len + 1 - len(param)) + '=   ' +        # Handles number of spaces before equal sign
                  formatter.format(centre) + ' (+' + formatter.format(sig_p) + 
                  ') (-' + formatter.format(sig_m) + ') ' + unit + '\n']

    lines += ['\n',
              '******************************************\n',
              '2 σ constraints\n',
              '******************************************\n',
              '\n']

    # Write 2 sigma parameter constraints
    for i, param in enumerate(param_names):
        
        sig_m, centre, \
        sig_p, formatter, unit = return_quantiles(stats, param, i, quantile = '2 sigma')

        lines += [param + ' '*(max_param_len + 1 - len(param)) + '=   ' +        # Handles number of spaces before equal sign
                  formatter.format(centre) + ' (+' + formatter.format(sig_p) + 
                  ') (-' + formatter.format(sig_m) + ') ' + unit + '\n']
 
    lines += ['\n',
              '******************************************\n',
              '3 σ constraints\n',
              '******************************************\n',
              '\n']

    # Write 3 sigma parameter constraints
    for i, param in enumerate(param_names):
        
        sig_m, centre, \
        sig_p, formatter, unit = return_quantiles(stats, param, i, quantile = '3 sigma')

        lines += [param + ' '*(max_param_len + 1 - len(param)) + '=   ' +        # Handles number of spaces before equal sign
                  formatter.format(centre) + ' (+' + formatter.format(sig_p) + 
                  ') (-' + formatter.format(sig_m) + ') ' + unit + '\n']
        
    lines += ['\n',
              '******************************************\n',
              '5 σ constraints\n',
              '******************************************\n',
              '\n']

    # Write 5 sigma parameter constraints
    for i, param in enumerate(param_names):
        
        sig_m, centre, \
        sig_p, formatter, unit = return_quantiles(stats, param, i, quantile = '5 sigma')

        lines += [param + ' '*(max_param_len + 1 - len(param)) + '=   ' +        # Handles number of spaces before equal sign
                  formatter.format(centre) + ' (+' + formatter.format(sig_p) + 
                  ') (-' + formatter.format(sig_m) + ') ' + unit + '\n']
    
    # Add best-fitting model parameters
    lines += ['\n',
              '******************************************\n',
              'Best-fitting parameters\n',
              '******************************************\n',
              '\n']
    
    # Write best-fitting parameter values
    for i, param in enumerate(param_names):
        
        _, _, _, \
        formatter, unit = return_quantiles(stats, param, i)     # We only need the formatter and unit for the best fit parameters

        lines += [param + ' '*(max_param_len + 1 - len(param)) + '=   ' +        # Handles number of spaces before equal sign
                  formatter.format(best_fit_params[i]) + ' ' + unit + '\n']

    # Commit the lines array to file
    summary_file.writelines(lines)
    
    
def write_MultiNest_results(planet, model, data, retrieval_name,
                            N_live, ev_tol, sampling_algorithm, wl, R):
    ''' 
    Process raw retrieval output into human readable output files.
    
    '''
    
    # Unpack planet name
    planet_name = planet['planet_name']

    # Unpack number of free parameters
    param_names = model['param_names']
    n_params = len(param_names)

    # Unpack data properties
    err_data = data['err_data']
    ydata = data['ydata']
    instruments = data['instruments']
    datasets = data['datasets']
    
    # Load relevant output directory
    output_prefix = retrieval_name + '-'
    
    # Run PyMultiNest analyser to extract posterior samples and model evidence
    analyzer = pymultinest.Analyzer(n_params, outputfiles_basename = output_prefix,
                                    verbose = False)
    stats = analyzer.get_stats()
    best_fit = analyzer.get_best_fit()
    samples = analyzer.get_equal_weighted_posterior()[:,:-1]
    
    # Store model evidence
    ln_Z = stats['nested sampling global log-evidence']
    ln_Z_err = stats['nested sampling global log-evidence error']
    
    # Store best-fitting reduced chi-squared
    max_likelihood = best_fit['log_likelihood']
    best_fit_params = best_fit['parameters']
    norm_log = (-0.5*np.log(2.0*np.pi*err_data*err_data)).sum()
    best_chi_square = -2.0 * (max_likelihood - norm_log)

    # Check for N_params >= N_data, for which chi^2_r is undefined
    if ((len(ydata) - n_params) > 0):
        dof = (len(ydata) - n_params)  
        reduced_chi_square = best_chi_square/dof
    else:
        dof = np.nan
        reduced_chi_square = np.nan

    # Load relevant results directories
    samples_prefix = '../samples/' + retrieval_name
    results_prefix = '../results/' + retrieval_name
    
    # Write samples to file
    write_samples_file(samples, param_names, n_params, samples_prefix)
            
    # Write POSEIDON retrieval summary file
    write_summary_file(results_prefix, planet_name, retrieval_name, 
                       sampling_algorithm, n_params, N_live, ev_tol, param_names, 
                       stats, ln_Z, ln_Z_err, reduced_chi_square, best_chi_square,
                       dof, best_fit_params, wl, R, instruments, datasets)
    

def mock_missing(name):
    def init(self, *args, **kwargs):
        raise ImportError(
            f'The module {name} you tried to call is not importable; '
            f'this is likely due to it not being installed.')
    return type(name, (), {'__init__': init})

