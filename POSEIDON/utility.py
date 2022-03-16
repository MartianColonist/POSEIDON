# Various miscellaneous functions

import os
import numpy as np
import pandas as pd
import pymultinest
from numba.core.decorators import jit
from spectres import spectres


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
                          planet_dir + '/geometry', planet_dir + '/retrievals']
            
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
def prior_index(val, vec, start):
    
    ''' Finds the index of a grid closest to a specified value.
    
    '''
    
    value_tmp = val
    
    if (value_tmp > vec[-1]):
        return (len(vec) - 1)
    
    # Check if value out of bounds, if so set to edge value
    if (value_tmp < vec[0]): value_tmp = vec[0]
    if (value_tmp > vec[-2]): value_tmp = vec[-2]
    
    index = start
    
    for i in range(len(vec)-start):
        if (vec[i+start] > value_tmp): 
            index = (i+start) - 1
            break
            
    return index



@jit(nopython=True)
def prior_index_V2(val, grid_start, grid_end, N_grid):
    
    ''' Finds the previous index of a UNIFORM grid closest to a specified value.
        A uniform grid dramatically speeds calculation over a non-uniform grid.
       
    '''
    
    if (val < grid_start): 
        return 0
    
    elif (val > grid_end):
        return N_grid-1
    
    else:
        i = (N_grid-1) * ((val - grid_start) / (grid_end - grid_start))
        return int(i)


@jit(nopython=True)
def closest_index(val, grid_start, grid_end, N_grid):
    '''
    Finds the index of a uniform grid closest to a specified value.

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
            The index of the uniform grid closest to 'val'.

    '''

    if (val < grid_start): 
        return 0
    
    elif (val > grid_end):
        return N_grid-1
    
    else:
        i = (N_grid-1) * ((val - grid_start) / (grid_end - grid_start))
        if ((i%1)<=0.5):
            return int(i)
        else:
            return int(i)+1
        

def size_profile(arr):
    
    ''' Profiles the size of a numpy array (returns size in Megabytes).
    
    '''
    
    print("%d Mb" % ((arr.size * arr.itemsize)/1048576.0))


def file_name_check(file_path):

    filename, extension = os.path.splitext(file_path)
    counter = 1

    while os.path.exists(file_path):
        file_path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return file_path
        

def read_data(data_dir, fname):
    ''' 
    Read a transmission spectrum dataset.
    
    '''
    
   # data_dir = '../../observations/' + planet_name + '/'

    data = pd.read_csv(data_dir + '/' + fname, sep = ' ', header=None)
    wavelength = np.array(data[0])  # Wavelengths (um)
    bin_size = np.array(data[1])    # Spectral bin size (um)
    spectrum = np.array(data[2])    # Transit depth
    err = np.array(data[3])         # Error on transit depth
    
    return wavelength, bin_size, spectrum, err

    
def read_spectrum(fname):
    
    ''' Read in a pre-computed transmission spectrum.
    
    '''
    
    data = pd.read_csv(fname, sep = ' ', header=None)
    wavelength = np.array(data[0])  # Wavelengths (um)
    spectrum = np.array(data[1])    # Transit depth
    
    return wavelength, spectrum


def bin_spectrum_fast(wl_native, spectrum_native, R_bin):
        
    # Create binned wavelength grid at resolution R_bin
    delta_log_wl_bins = 1.0/R_bin
    N_wl_bins = (np.log(wl_native[-1]) - np.log(wl_native[0])) / delta_log_wl_bins
    N_wl_bins = np.around(N_wl_bins).astype(np.int64)
    log_wl_binned = np.linspace(np.log(wl_native[0]), np.log(wl_native[-1]), N_wl_bins)    
    wl_binned = np.exp(log_wl_binned)
    
    spectrum_binned = spectres(wl_binned, wl_native, spectrum_native,
                               verbose = False)
    
    return wl_binned[1:-1], spectrum_binned[1:-1]   # Cut out first and last values to avoid nans
                

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
    output_dir = '../results/'
    
    # Write retrieved spectrum
    f = open(output_dir + retrieval_name + '_spectrum_retrieved.txt', 'w')
    
    f.write('wl (μm) | spectrum: -2σ | spectrum: -1σ | spectrum: median ' + 
            '| spectrum: +1σ | spectrum: +2σ \n')
    
    for i in range(len(wl)):
        f.write('%.8e %.8e %.8e %.8e %.8e %.8e \n' %(wl[i], spec_low2[i], spec_low1[i], 
                                                     spec_median[i], spec_high1[i], spec_high2[i]))
        
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
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'

    # Find retrieved spectrum file
    fname = output_dir + retrieval_name + '_spectrum_retrieved.txt'

    # Read retrieved spectrum confidence intervals
    spec = pd.read_csv(fname, sep = ' ', header = None, skiprows = 1)

    wl = np.array(spec[0])           # Wavelengths (um)
    spec_low2 = np.array(spec[1])    # -2σ
    spec_low1 = np.array(spec[2])    # -1σ
    spec_median = np.array(spec[3])  # Median
    spec_high1 = np.array(spec[4])   # +1σ
    spec_high2 = np.array(spec[5])   # +2σ
    
    return wl, spec_low2, spec_low1, spec_median, spec_high1, spec_high2


def write_retrieved_PT(retrieval_name, P, T_low2, T_low1, 
                       T_median, T_high1, T_high2):
    '''
    ADD DOCSTRING
    '''

    # Identify output directory location where the retrieved spectrum will be saved
    output_dir = '../results/'
    
    # Write retrieved spectrum
    f = open(output_dir + retrieval_name + '_PT_retrieved.txt', 'w')
    
    f.write('P (bar) | T: -2σ | T: -1σ  | T: median | T: +1σ | T: +2σ \n')
    
    for i in range(len(P)):
        f.write('%.8e %.8e %.8e %.8e %.8e %.8e \n' %(P[i], T_low2[i], T_low1[i], 
                                                     T_median[i], T_high1[i], T_high2[i]))
        
    f.close()
    
    
def read_retrieved_PT(planet_name, model_name, retrieval_name = None):
    '''
    ADD DOCSTRING
    '''

    if (retrieval_name is None):
        retrieval_name = model_name
    else:
        retrieval_name = model_name + '_' + retrieval_name

    # Identify output directory location where the retrieved spectrum is located
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'

    # Find retrieved spectrum file
    fname = output_dir + retrieval_name + '_PT_retrieved.txt'

    # Read retrieved spectrum confidence intervals
    spec = pd.read_csv(fname, sep = ' ', header = None, skiprows = 1)

    P = np.array(spec[0])           # Wavelengths (um)
    T_low2 = np.array(spec[1])    # -2σ
    T_low1 = np.array(spec[2])    # -1σ
    T_median = np.array(spec[3])  # Median
    T_high1 = np.array(spec[4])   # +1σ
    T_high2 = np.array(spec[5])   # +2σ
    
    return P, T_low2, T_low1, T_median, T_high1, T_high2


def write_output_binned(planet_name, wl, spectrum_binned, description):
    
    ''' Writes out a given model (binned) spectrum.
    
    '''
    
    # Write spectrum
    f = open('../../output/spectra/' + planet_name + '/' + planet_name + '_' + 
             description + '_best_binned.dat','w')
    
    for i in range(len(wl)):
        f.write('%.8e %.8e \n' %(wl[i], spectrum_binned[i]))
        
    f.close()
    
    
def write_geometry(planet_name, r, T, theta_edge, phi_edge, description):
    
    file = '../../output/geometry/' + planet_name + '/' + planet_name + '_' + description + '.npy'
    
    with open(file, 'wb') as f:
        np.save(f, r)
        np.save(f, T)
        np.save(f, theta_edge)
        np.save(f, phi_edge)
    
    
def write_data(planet_name, wl_data, bin_size, spectrum, err_data, instrument):
    
    # Write spectrum
    f = open('../../observations/' + planet_name + '/' + planet_name + '_SYNTHETIC_' + instrument + '.dat','w')
    
    for j in range(len(wl_data)):
        f.write('%.8f %.8f %.8e %.8e \n' %(wl_data[j], bin_size[j], spectrum[j], err_data[j]))
        
    f.close()
    

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
    phrases = ['high', 'deep', 'ref', 'DN', 'term', 'Morn', 'Even', 'Day', 'Night', 
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
        N_phrase = np.count_nonzero(components_sort == 'phrase')
        N_letter_digit = N_letter + N_digit
        
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
            elif (components_sort[i] in ['digit', 'letter']):
                if (letter_digit_bracket_open == 0):
                    latex_name += '\mathrm{'
                    letter_digit_bracket_open = 1
                if (letter_digit_bracket_open == 1):
                    if (N_letter_digit >= 2):
                        if (components_sort[i] == 'digit'):
                            latex_name += ('_' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + ' ')  
                        elif (components_sort[i] == 'letter'):
                            latex_name += param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]]
                    elif (N_letter_digit == 1):
                        if (components_sort[i] == 'digit'):
                            latex_name += ('_' + param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + '}')  
                        elif (components_sort[i] == 'letter'):
                            latex_name += (param[idxs_sort[i]:idxs_sort[i]+lens_sort[i]] + '}')
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
                       stats, ln_Z, ln_Z_err, reduced_chi_square, best_fit_params,
                       wl, R, instruments, datasets):
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
              '#################################\n',
              '\n',
              'Best reduced chi-square:\n',
              '\n',
              '-> chi^2_min = ' + stats_formatter.format(reduced_chi_square) + '\n',
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
    reduced_chi_square = best_chi_square/(len(ydata) - n_params)  

    # Load relevant results directories
    samples_prefix = '../samples/' + retrieval_name
    results_prefix = '../results/' + retrieval_name
    
    # Write samples to file
    write_samples_file(samples, param_names, n_params, samples_prefix)
            
    # Write POSEIDON retrieval summary file
    write_summary_file(results_prefix, planet_name, retrieval_name, 
                       sampling_algorithm, n_params, N_live, ev_tol, param_names, 
                       stats, ln_Z, ln_Z_err, reduced_chi_square, best_fit_params,
                       wl, R, instruments, datasets)
    



