''' 
Functions for calculating the effective cross sections for the database (from lab data).

'''
import numpy as np
import scipy
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
import os 
import h5py
import glob

############################################################################################
# Empty Arrays for Qext calculations
############################################################################################

# All refractive indices
all_etas = []

# Inputs to Q_ext (2 pi r / lambda )
all_xs = []

# All Q_ext values already computed 
all_Qexts = []

# Wavelength Array for Mie Calculations, default resolution = 1000
wl_Mie = np.array([])

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


def load_refractive_indices_from_file(wl,file_name):

    '''
    Loads in the refractive indices from a text file (2 rows skipped, columns : wl n k)

    Args:
        wl (np.array of float):
            Model wavelength grid (μm).

        file_name (string):
            File name (with directory included)
    
    Returns:
        r_i_real (np.array of float)
            Array with the loaded in real indices interpolated onto wl_Mie

        r_i_complex (np.array of float)
            Array with the loaded in imaginary indices interpolated onto wl_Mie

    '''

    wl_min = np.min(wl)
    wl_max = np.max(wl)
    wl_Mie = wl_grid_constant_R(wl_min, wl_max, 1000)

    file_as_numpy = np.loadtxt(file_name,skiprows=2).T
    wavelengths = file_as_numpy[0]
    real_indices = file_as_numpy[1]
    imaginary_indices = file_as_numpy[2]

    interp_reals = interp1d(wavelengths, file_as_numpy[1])
    interp_complexes = interp1d(wavelengths, file_as_numpy[2])

    return interp_reals(wl_Mie), interp_complexes(wl_Mie)


def plot_refractive_indices_from_file(wl, file_name):

    '''
    Plots the refractive indices from a txt file (2 rows skipped, columns : wl n k)

    Args:
        wl (np.array of float):
            Model wavelength grid (μm).

        file_name (string):
            File name (with directory included)

    '''

    real_indices, imaginary_indices = load_refractive_indices_from_file(wl,file_name)
    wl_min = np.min(wl)
    wl_max = np.max(wl)
    wl_Mie = wl_grid_constant_R(wl_min, wl_max, 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 6)
    suptitle = 'Refractive Indices for ' + file_name.split('/')[-1][:-4]
    fig.suptitle(suptitle)
    ax1.plot(wl_Mie, real_indices)
    ax2.plot(wl_Mie, imaginary_indices)

    ax1.set_xlabel('Wavelength (um)')
    ax2.set_xlabel('Wavelength (um)')
    ax1.set_ylabel('Real Indices')
    ax2.set_ylabel('Imaginary Indices')

    plt.show()


def plot_effective_cross_section_from_file(wl, r_m, file_name):

    '''
    Plots the effective cross sections from a txt file (2 rows skipped, columns : wl n k)

    Args:
        wl (np.array of float):
            Model wavelength grid (μm).

        r_m (float):
            Mean particle size (um)

        file_name (string):
            File name (with directory included)

    '''

    r_i_real, r_i_complex = load_refractive_indices_from_file(wl,file_name)
    wl_min = np.min(wl)
    wl_max = np.max(wl)
    wl_Mie = wl_grid_constant_R(wl_min, wl_max, 1000)
    eff_cross_section = precompute_cross_sections_from_indices(wl_Mie,r_i_real,r_i_complex, r_m)

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_cross_section, label = label)
    title = 'Effective Cross Sections ' + file_name.split('/')[1][:-4]
    plt.title(title)
    plt.ylabel('Effective Cross Section')
    plt.xlabel('Wavelength (um)')
    plt.show()


############################################################################################
# LX MIE Algorithm - See https://arxiv.org/abs/1710.04946
############################################################################################

def get_iterations_required(xs, c=4.3):
    # c=4.3 corresponds to epsilon=1e-8, according to Cachorro & Salcedo 2001
    # (https://arxiv.org/abs/physics/0103052)
    num_iters = xs + c * xs**(1.0/3)
    num_iters = num_iters.astype(int) + 2
    return num_iters


def get_An(zs, n):
    # Evaluate A_n(z) for an array of z's using the continued fraction method.
    # See eq 12 in https://arxiv.org/abs/1710.04946 
    # An is the logarithmic derivative of Riccati-Bessel functions
    # This is necessary for downward recursion of A_n(z) for lower n's.
    # The algorithm is from http://adsabs.harvard.edu/abs/1976ApOpt..15..668L
    nu = n + 0.5
    ratio = 1

    numerator = None
    denominator = None

    i = 1
    
    while True:
        an = (-1)**(i+1) * 2 * (nu + i - 1)/zs
        if i == 1:
            numerator = an
        elif i > 1:
            numerator = an + 1.0/numerator

        ratio *= numerator
        
        if i == 2:
            denominator = an
        elif i > 2:
            denominator = an + 1.0/denominator

        if denominator is not None:
            ratio /= denominator
            if np.allclose(numerator, denominator):
                break
        i += 1

    A_n =  -n/zs + ratio
    return A_n
            

def get_As(max_n, zs):
    # Returns An(zs) from n=0 to n = max_n-1 using downward recursion.
    # zs should be an array of real or complex numbers.
    An = np.zeros((max_n, len(zs)), dtype=complex)
    An[max_n - 1] = get_An(zs, max_n-1)
    
    for i in range(max_n - 2, -1, -1):
        An[i] = (i + 1)/zs - 1.0/((i + 1)/zs + An[i+1])
    return An
    

def get_Qext(m, xs):
    # Uses algorithm from Kitzmann & Heng 2017 to compute Qext(x) for an array
    # of x's and refractive index m.  This algorithm is stable and does not
    # lead to numerical overflows.  Paper: https://arxiv.org/abs/1710.04946

    xs = np.array(xs)
    num_iterations = get_iterations_required(xs) 
    max_iter = max(max(num_iterations) , 1)
    
    A_mx = get_As(max_iter, m * xs)
    A_x = get_As(max_iter, xs)
    Qext = np.zeros(len(xs))

    curr_B = 1.0/(1 + 1j * (np.cos(xs) + xs*np.sin(xs))/(np.sin(xs) - xs*np.cos(xs)))
    curr_C = -1.0/xs + 1.0/(1.0/xs  + 1.0j)
    
    for i in range(1, max_iter):
        cond = num_iterations > i
        if i > 1:
            curr_C[cond] = -i/xs[cond] + 1.0/(i/xs[cond] - curr_C[cond])
            curr_B[cond] = curr_B[cond] * (curr_C[cond] + i/xs[cond])/(A_x[i][cond] + i/xs[cond])
            
        an = curr_B[cond] * (A_mx[i][cond]/m - A_x[i][cond])/(A_mx[i][cond]/m - curr_C[cond])
        bn = curr_B[cond] * (A_mx[i][cond]*m - A_x[i][cond])/(A_mx[i][cond]*m - curr_C[cond])
        
        Qext[cond] += (2*i + 1) * (an + bn).real
        
    Qext *= 2/xs**2
    return Qext

############################################################################################
# Auxiliary Functions (Retrieving and updating cached arrays)
############################################################################################

# Function that tries to find an existing Qext (within 5% marigin error) via interpolation 
# Or returns a miss 
# INPUTS : Refractive Indices array, 2 pi r / lambda array

# max_frac_error = 0.05

def get_from_cache(eta, xs, max_frac_error = 0.05):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    # Create an array of nans the same length as xs 
    result = np.full(len(xs),np.nan)

    # ADD THIS LATER 
    # All_xs will now be sorted by eta
    # Find the all_xs_eta that matches the eta you are on 
    # If eta exists in the eta array already, you can proceed using the following all_xs_eta
    # If not, create a new all_xs_eta array that is returned as the result 
    # The rest just follows naturally 

    if eta in all_etas:
        #x_s_index = np.argwhere(all_etas == eta)
        x_s_index = all_etas.index(eta)
        all_xs_eta = all_xs[x_s_index]
        all_Qexts_eta = all_Qexts[x_s_index]
    else:
        return result

    # If its the first iteration, Qext is empty 
    if len(all_xs_eta) == 0:
        return result
    
    # This just returns an array of True
    in_cache = np.ones(len(xs), dtype=bool)

    # Find the indices into a sorted array all_xs such that, 
    # if the corresponding elements in xs were inserted before the indices, 
    # the order of a would be preserved.
    # If == len(all_xs), then its above the maximum value in all_xs
    # Note that this will count the left most point as a miss (if they are exactly equal, its set to index 0 which is a miss)

    # Ok the closest matches doesn't work with a sorted array by xs
    closest_matches = np.searchsorted(all_xs_eta, xs)

    # np.logical_or computes the truth value of x1 OR x2 element wise 
    # if closest_matches == 0 (Closest index is the lower bound) or the highest bound 
    # set that value automatically to false
    in_cache[np.logical_or(closest_matches == 0, closest_matches == len(all_xs_eta))] = False

    
    # Makes all the indices that were above the maximum one less (so that it works with all_etas array)
    # all_etas is ordered the same way as all_xs 
    # Pretty much, its a possibility that you can have a particle a specific size but not
    # Have the same refractive index. These statements make those a miss 

    closest_matches[closest_matches == len(all_xs_eta)] -= 1
    #in_cache[all_etas[closest_matches] != eta] = False

    # Computes the fractional error of surviving matches 
    # If the fractional error is greater than some amount (5%), its set to false 
    frac_errors = np.abs(all_xs_eta[closest_matches] - xs)/xs
    in_cache[frac_errors > max_frac_error] = False

    # If the all_etas array is empty, return an empty result array as well 
    # This can happen occasionally after the first initialization I think (we have the x, but not the eta)
    #if np.sum(all_etas == eta) == 0: 
    #    return result
    
    # Interpolates the results for hits
    result[in_cache] = np.interp(
        xs[in_cache],
        all_xs_eta,
        all_Qexts_eta)

    return result

# Function that adds the new Qext if there was a miss 
# INPUTS : eta, xs, and new Qexts
def add(eta, xs, Qexts, size_limit=1000000):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    # ??? I think this is to prevent some sort of bug, I can't think of one though 
    if len(xs) == 0:
        return
    
    # Add to existing arrays in the cache 
    if eta in all_etas:
        x_s_index = all_etas.index(eta)
        all_xs[x_s_index] = np.append(all_xs[x_s_index], xs)
        all_Qexts[x_s_index] = np.append(all_Qexts[x_s_index], Qexts)

        # In order to save memory, if there are more than the size limit it deletes some random ones 
        # Only matters if an eta already exists
        if len(all_xs[x_s_index]) > size_limit:
            to_remove = np.random.choice(
                range(len(all_xs[x_s_index])), len(all_xs[x_s_index]) - size_limit + 1,
                replace=False)
            
            all_xs[x_s_index] = np.delete(all_xs[x_s_index], to_remove)
            all_Qexts[x_s_index] = np.delete(all_Qexts[x_s_index], to_remove)

        # Sort all of the cahced arrays 
        p = np.argsort(all_xs[x_s_index])
        all_xs[x_s_index] = all_xs[x_s_index][p]
        all_Qexts[x_s_index] = all_Qexts[x_s_index][p]

    # If not, append a new array for a new eta 
    else:

        all_xs.append(np.array(xs))
        all_Qexts.append(np.array(Qexts))
        all_etas.append(eta)

        # Sort the cached arrays 
        p = np.argsort(all_xs[-1])
        all_xs[-1] = all_xs[-1][p]
        all_Qexts[-1] = all_Qexts[-1][p]

# Function that either 1) Updates Qext or 2) Returns it if value already exist
# INPUTS : Refractive Indices array, 2 pi r / lambda array
def get_and_update(eta,xs):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    # This array will be full of nans and Qext (corresponding to misses and hits in the cache)
    Qexts_eta = get_from_cache(eta, xs)

    # If there are ANY misses, we run the LX_MIE algorithm to find the Qext
    # And then we add them to the cached arrays 
    cache_misses = np.isnan(Qexts_eta)
        
    # Not a new eta, but with misses
    if np.sum(cache_misses) > 0:
        # LX MIE Algorithm
        Qexts_eta[cache_misses] = get_Qext(eta, xs[cache_misses])
        # Adds to cahced arrays
        add(eta, xs[cache_misses], Qexts_eta[cache_misses])

    return Qexts_eta

############################################################################################
# Main DataBase Functions
############################################################################################

# This will can be used to make new cross sections from lab data to add to database
def precompute_cross_sections_one_aerosol(aerosol):

    '''
    Calculates the .npy file from a refractive index txt file (lab data)
    Takes ~ a day to generate the npy file that then can be easily added to the aerosol database  
    Please ensure that the first two rows are skippable, and that the columns are 
    Wavelength (microns) | Real Index | Imaginary Index
    Please also ensure that the wavelengths are at LEAST from 0.2 to 30 um. If not, add 0s into your txt files 

    INPUTS 

    aerosol (string)
        string that goes in front of the _complex.txt
    '''

    global all_etas, all_xs, all_Qexts

    # Constants that for the Qext Claculation
    r_m_std_dev = 0.5
    z_max = 5
    num_integral_points = 100
    R_Mie = 1000

    # Saved Arrays 
    effective_cross_section_array = []
    wl_Mie = []

    # Initialize the wl and rm arrays 
    wl_min = 0.2
    wl_max = 30
    wl_Mie = np.append(wl_Mie,wl_grid_constant_R(wl_min, wl_max, R_Mie))

    r_m_array = 10**np.linspace(-3,1,1000)

    # Load in the input file path
    input_file_path = os.environ.get("POSEIDON_input_data")

    if input_file_path == None:
        raise Exception("POSEIDON cannot locate the input folder.\n" +
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "POSEIDON input folder.")


    #########################
    # Load in refractive indices (as function of wavelength)
    #########################
    try :
        file_name = input_file_path + 'refractive_indices/' + aerosol + '_complex.txt'
        print('Loading in : ', file_name)
        file_as_numpy = np.loadtxt(file_name,skiprows=2).T
    except :
        raise Exception('Please put the txt file of the lab data in input/refractive_indices with the naming convention aerosol + _complex.txt')

    wavelengths = file_as_numpy[0]

    if np.max(wavelengths) < 30 or np.min(wavelengths) > 0.2:
        raise Exception('Please ensure that the wavelength column spans 0.2 to 30 um')
    
    interp_reals = interp1d(wavelengths, file_as_numpy[1])
    interp_complexes = interp1d(wavelengths, file_as_numpy[2])
    eta_array = interp_reals(wl_Mie) + -1j *interp_complexes(wl_Mie)

    counter = 0

    # Loop over all r_ms
    print('Generating .npy file of cross sections for ' + aerosol)
    print('This will take approximately 1 day. Every 10 iterations, the current r_m is printed.')

    for r_m in r_m_array:

        #########################
        # Caculate the effective cross section of the particles (as a function of wavelength)
        #########################

        if counter % 10 == 0:
            print(r_m)

        # Clear the arrays every 250 iterations (to save memory)
        if counter % 250 == 0:
            all_etas = []
            all_xs = []
            all_Qexts = []

        eff_cross_sections = np.zeros(len(wl_Mie))

        z = -np.logspace(np.log10(0.1), np.log10(z_max), int(num_integral_points/2)) 
        z = np.append(z[::-1], -z)

        probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
        radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
        geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric
 

        Qext_intpl_array = []

        # Loop through each wavelength 
        for m in range(len(wl_Mie)):
            
            # Take the dense xs, but keep wavelength constant this time around
            dense_xs = 2*np.pi*radii / wl_Mie[m]
            dense_xs = dense_xs.flatten()

            # Make xs more coarse
            x_hist = np.histogram(dense_xs, bins='auto')[1]

            # Pull the refractive index for the wavelength we are on 
            eta = eta_array[m]

            # Get the coarse Qext with the constant eta 
            Qext_hist = get_and_update(eta, x_hist) 

            # Revert from coarse Qext back to dense Qext 
            spl = scipy.interpolate.splrep(x_hist, Qext_hist)
            Qext_intpl = scipy.interpolate.splev(dense_xs, spl)

            # Append it to the array that will have all the Qext
            Qext_intpl_array.append(Qext_intpl)

        # Reshape the mega array so that the first index is wavelngth, second is radius 
        Qext_intpl = np.reshape(Qext_intpl_array, (len(wl_Mie), len(radii)))

        # Effective Cross section is a trapezoidal integral
        eff_cross_section = np.trapz(probs*geometric_cross_sections*Qext_intpl, z)

        effective_cross_section_array.append(eff_cross_section)

        counter += 1

    title = input_file_path + '/refractive_indices/sigma_Mie_' + aerosol
    np.save(title,effective_cross_section_array,allow_pickle = True)
    all_etas = []
    all_xs = []
    all_Qexts = []
    print('Remember to update aerosol_supported_species in supported_opac.py!')

# Allows the user to make one set of cross sections from an input array of imaginary and real indices 
def precompute_cross_sections_from_indices(wl,real_indices_array,imaginary_indices_array, r_m):

    '''
    Calculates and returns the effective cross section from an input wl grid, real and imaginary indices array 
    And the particle size in um 

    Allows the user to directly quirey the LX_MIE algorithm with their refractive index data 

    INPUTS 

    wl (np.array of float):
        Model wavelength grid (μm).

    real_indices_array (np.array of float):
        Real indices 
    
    imaginary_indices_array (np.array of float):
        Imaginary indices 

    r_m
    '''
        
    global all_etas, all_xs, all_Qexts

    # Constants that for the Qext Claculation
    r_m_std_dev = 0.5
    z_max = 5
    num_integral_points = 100

    # Initialize the wl 

    wavelengths = wl
    
    eta_array = real_indices_array + -1j * imaginary_indices_array


    #########################
    # Caculate the effective cross section of the particles (as a function of wavelength)
    #########################

    eff_cross_sections = np.zeros(len(wavelengths))

    z = -np.logspace(np.log10(0.1), np.log10(z_max), int(num_integral_points/2)) 
    z = np.append(z[::-1], -z)

    probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
    radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
    geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric

    Qext_intpl_array = []

    # Loop through each wavelength 
    for m in range(len(wavelengths)):
        
        # Take the dense xs, but keep wavelength constant this time around
        dense_xs = 2*np.pi*radii / wavelengths[m]
        dense_xs = dense_xs.flatten()

        # Make xs more coarse
        x_hist = np.histogram(dense_xs, bins='auto')[1]

        # Pull the refractive index for the wavelength we are on 
        eta = eta_array[m]

        # Get the coarse Qext with the constant eta 
        Qext_hist = get_and_update(eta, x_hist) 

        # Revert from coarse Qext back to dense Qext 
        spl = scipy.interpolate.splrep(x_hist, Qext_hist)
        Qext_intpl = scipy.interpolate.splev(dense_xs, spl)

        # Append it to the array that will have all the Qext
        Qext_intpl_array.append(Qext_intpl)

    # Reshape the mega array so that the first index is wavelngth, second is radius 
    Qext_intpl = np.reshape(Qext_intpl_array, (len(wavelengths), len(radii)))

    # Effective Cross section is a trapezoidal integral
    eff_cross_section = np.trapz(probs*geometric_cross_sections*Qext_intpl, z)

    all_etas = []
    all_xs = []
    all_Qexts = []

    return eff_cross_section

# Reformulate database
def make_aerosol_database():

    '''
    Regenerates the aerosol_database from all npy files in input/refractive_indices
    This functionality allows for users to create new eff cross section arrays using 
    precompute_cross_Sections_one_aerosol()
    With their own lab data 
    '''
    input_file_path = os.environ.get("POSEIDON_input_data")

    if input_file_path == None:
        raise Exception("POSEIDON cannot locate the input folder.\n" +
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "POSEIDON input folder.")

    # Load in the aerosol list
    mydir = input_file_path + "refractive_indices/"
    file_list = glob.glob(mydir + "*.npy")

    print('---------------------')
    print('Loading in .npy files from')
    print(mydir)
    print('---------------------')

    aerosol_list = []

    # Getting a string for each aerosol in the folder 
    for file in file_list:
        file_split = file.split('/')
        file = file_split[-1]
        file = file[10:]
        file = file[:-4]
        
        aerosol_list.append(file)

    print('---------------------')
    print('Generating database from the following aerosols')
    print('---------------------')

    # Wavelength and r_m array used to generate the npy files
    R_Mie = 1000
    wavelengths = wl_grid_constant_R(0.2, 30, R_Mie)
    r_m_array = 10**np.linspace(-3,1,1000)

    # Create the dictionary for aerosols and load in all npy files
    aerosols_dict = {}
    for i in range(len(aerosol_list)):
        title = file_list[i]
        sigma_Mie = np.load(title,allow_pickle = True)
        aerosols_dict[aerosol_list[i]] = sigma_Mie

    # Initialize and generate new data_base 
    database = h5py.File(input_file_path + '/aerosol_database.hdf5', 'w')

    h = database.create_group('Info')
    h1 = h.create_dataset('Wavelength grid', data=wavelengths, compression='gzip', dtype='float64', shuffle=True)
    h2 = h.create_dataset('Particle Size grid', data=r_m_array, compression='gzip', dtype='float64', shuffle=True)

    h1.attrs["Variable"] = "wl"
    h2.attrs["Variable"] = "r_m"

    h1.attrs["Units"] = "um"
    h2.attrs["Units"] = "um"

    for i in range(len(aerosol_list)):
        print(aerosol_list[i])
        g = database.create_group(aerosol_list[i])
        g1 = g.create_dataset('sigma_Mie', data=aerosols_dict[aerosol_list[i]], compression='gzip', dtype='float32', shuffle=True)
        g1.attrs["Varaible"] = "Effective Cross Section"
        g1.attrs["Units"] = "um^2"

    print('---------------------')
    print('Saving new aerosol database as')
    print(input_file_path + 'aerosol_database.hdf5')
    print('---------------------')

    database.close()
