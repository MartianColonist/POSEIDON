''' 
Functions for calculating clouds.

'''
import numpy as np
import scipy
from scipy.interpolate import interp1d, RegularGridInterpolator

directory = '/Users/elijahmullens/Desktop/Poseidon-temp/input/opacity/refractive_indices/'

############################################################################################
# Empty Arrays for Qext (later to be changed to a shared memory array, see chemistry.py)
############################################################################################

# All refractive indices
all_etas = []

# Inputs to Q_ext (2 pi r / lambda )
all_xs = []

# All Q_ext values already computed 
all_Qexts = []

# Eta array for pre-loaded indices
eta_supported_aerosol_array = np.array([])
eta_supported_aerosol_array_aerosol = ''

# Wavelength Array for Mie Calculations, default resolution = 1000
wl_Mie = np.array([])

# Interpolated data_base 
interp_aerosols = []
interp_saved = []


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
# Main Cloud Functions
############################################################################################

def get_supported_aerosols():
    return ['SiO2', 'Al2O3', 'CaTiO3', 'CH4', 'Fe2O3', 'Fe2SiO4',
                          'H2O','Hexene','Hibonite','KCl','Mg2SiO4',
                          'Mg2SiO4poor','MgAl2O4','MgSiO3','MnS',
                          'Na2S','NaCl','SiO2','Tholin','TiO2','ZnS',
                          'SiO2_amorph','C','Cr','Fe', 'FeS', 'Mg2SiO4_amorph_sol_gel',
                          'Mg04Fe06SiO3_amorph_glass','Mg05Fe05SiO3_amorph_glass',
                          'Mg08Fe02SiO3_amorph_glass','Mg08Fe12SiO4_amorph_glass',
                          'MgFeSiO4_amorph_glass','MgO','MgSiO3_amorph_glass',
                          'MgSiO3_amorph_sol_gel_complex','SiC','SiO','TiC','TiO2_anatase']

def get_supported_aerosol_wl_range(aerosol):
    file_name = directory + aerosol + '_complex.txt'
    file_as_numpy = np.loadtxt(file_name,skiprows=2).T
    wavelengths = file_as_numpy[0]
    print('---', aerosol,'---')
    print('Minumum wavelength:', min(wavelengths), 'um')
    print('Maximum wavelength:', max(wavelengths), 'um')

def Mie_cloud(P,wl,r,
              P_cloud, r_m, log_n_max, fractional_scale_height, H, n,
              aerosol = 'free',
              r_i_real = 0,
              r_i_complex = 0,
              log_X_Mie = 100,
              r_m_std_dev = 0.5,
              z_max = 5,
              num_integral_points = 100,
              R_Mie = 1000):


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

        P_cloud (float) : 
            Cloud Top Pressure (everything below P_cloud is opaque). 
            If cloud coverage is complete, P_cloud is located at R_p

        r_m   (float) : 
            Mean particle size (in um)

        log_n_max (float) : 
            Logorithm of maximum number density (at the cloud top)

        fractional_scale_height (float) :
            fractional scale height of aerosol

        H (np.array of float) : 
            gas scale height

        n (np.array of float) :
            total number density array 

        -------- Semi- Optional Arguments -------

        aerosol (string) :
            Name of aerosol (e.g., 'SiO2')
            If this is left empty or = 'free', the refractive index must be entered / be a free variable

        r_i_real (float) :
            Real component of the complex refractive index
            If aerosol = string, the refractive index will be taken from a precomputed list 

        r_i_complex (float) : 
            Imaginary component of the complex refractive index
            If aerosol = string, the refractive index will be taken from a precomputed list 

        log_X_Mie (float) : 
            Mixing ratio for a mie aerosol (either specified or free, only for uniform haze models)

        MUST have either a specified aerosol OR a r_i_real + r_i_complex

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

    
    Returns:
          
    '''

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts, eta_supported_aerosol_array, eta_supported_aerosol_array_aerosol, wl_Mie, interp_saved, interp_aerosols

    # Up until ZnS is Wakeford, after that is Kitzman 
    supported_aerosols = ['SiO2', 'Al2O3', 'CaTiO3', 'CH4', 'Fe2O3', 'Fe2SiO4',
                          'H2O','Hexene','Hibonite','KCl','Mg2SiO4',
                          'Mg2SiO4poor','MgAl2O4','MgSiO3','MnS',
                          'Na2S','NaCl','Tholin','TiO2','ZnS',
                          'SiO2_amorph','C','Cr','Fe', 'FeS', 'Mg2SiO4_amorph_sol_gel',
                          'Mg04Fe06SiO3_amorph_glass','Mg05Fe05SiO3_amorph_glass',
                          'Mg08Fe02SiO3_amorph_glass','Mg08Fe12SiO4_amorph_glass',
                          'MgFeSiO4_amorph_glass','MgO','MgSiO3_amorph_glass',
                          'MgSiO3_amorph_sol_gel','SiC','SiO','TiC','TiO2_anatase']

    #########################
    # Error messages 
    #########################

    if aerosol not in supported_aerosols and aerosol != 'free':
        return Exception('Mie scattering does not support precalculated refractives indices of : ', aerosol)
    

    #########################
    # Set up wl_mie (a wl array with R = 1000)
    #########################

    wl_min = wl[0]
    wl_max = wl[-1]

    # Initialize wl_Mie
    if len(wl_Mie) == 0:
        wl_Mie = np.append(wl_Mie,wl_grid_constant_R(wl_min, wl_max, R_Mie))

    # If its a new wl array 
    if  wl[0] != wl_min or wl[-1] != wl_max:
        wl_min = wl[0]
        wl_max = wl[-1]
        wl_Mie = []
        wl_Mie = np.append(wl_Mie,wl_grid_constant_R(wl_min, wl_max, R_Mie))

    #########################
    # Calculate the number density above the cloud top or apply a uniform haze
    #########################
    
    # Fuzzy Deck Model 
    if fractional_scale_height != 0:
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

    # Uniform Haze
    else:
        n_aerosol = np.empty_like(r)
        n_aerosol = (n)*np.float_power(10,log_X_Mie)

    #########################
    # Load in refractive indices (as function of wavelength)
    #########################

    if aerosol in supported_aerosols:

        # First, check to see if a precomputed cross section database exists for the chosen aerosol. 
        # If so, load in and interpolate the database, and return the cross sections 
        # If not, run through the entire Qext cacheing code 

        try:
            # If an interp object has not been made for the aerosol being queried, make one and add to the saved list 
            # Else, just load in the prexisting one
            if aerosol not in interp_aerosols:
                title = '/Users/elijahmullens/Desktop/Poseidon-Temp/input/refractive_indices/sigma_Mie_'+aerosol+'.npy'
                sigma_Mie = np.load(title,allow_pickle = True)
                wavelengths = wl_grid_constant_R(0.2, 30, R_Mie)
                r_m_array = 10**np.linspace(-3,1,1000)
                interp = RegularGridInterpolator((r_m_array, wavelengths), sigma_Mie, bounds_error=False, fill_value=None)

                interp_aerosols.append(aerosol)
                interp_saved.append(interp)

            else:
                interp = interp_saved[interp_aerosols.index(aerosol)]
                
            sigma_Mie = interp((r_m,wl))

            return n_aerosol, sigma_Mie
            
        except:
            # Check to see if eta_supported-array array is empty (this is so it doesn't have to run twice in a retrieval)
            # It also might be the case that the aerosol in the model was changed  (i.e. the indices for SiO2 are saved but we changed aerosols)
            if len(eta_supported_aerosol_array) == 0 or eta_supported_aerosol_array_aerosol!=aerosol:
                ## Load in refractive index w/ wl. Then interpolate it
                # $$$ This can be made into a saved array as well, for retrievals (so it doesn't have to be loaded every time)
                # Complex must be negative
                file_name = directory + aerosol + '_complex.txt'
                file_as_numpy = np.loadtxt(file_name,skiprows=2).T
                wavelengths = file_as_numpy[0]
                interp_reals = interp1d(wavelengths, file_as_numpy[1])
                interp_complexes = interp1d(wavelengths, file_as_numpy[2])
                eta_array = interp_reals(wl_Mie) + -1j *interp_complexes(wl_Mie)

                # First time through 
                if eta_supported_aerosol_array_aerosol == '':
                    eta_supported_aerosol_array = np.append(eta_supported_aerosol_array,eta_array)
                    eta_supported_aerosol_array_aerosol = aerosol
                elif eta_supported_aerosol_array_aerosol != aerosol:
                    # Reset all the arrays if its a new aerosol
                    eta_supported_aerosol_array = np.array([])
                    eta_supported_aerosol_array = np.append(eta_supported_aerosol_array,eta_array)
                    eta_supported_aerosol_array_aerosol = aerosol
                    all_etas = []
                    all_xs = []
                    all_Qexts = []

            else:
                eta_array = eta_supported_aerosol_array
            
            
    else:
        # Apply constant eta to entire array 
        # complex index must be negative
        eta = complex(r_i_real,-r_i_complex)
        eta_array = np.full(len(wl_Mie),eta)

    #########################
    # Caculate the effective cross section of the particles (as a function of wavelength)
    #########################

    # There is an effective cross section for every wavelength 
    # Therefore, we need an array of resultant effective cross sections the same size as wl 

    eff_cross_sections = np.zeros(len(wl_Mie))

    # Eventually we will be numerically integrating over z, where z = ln(r) - ln(r_m) / r_m_std_dev^2
    # r is the particle size, given by a log-normal particle size distribution P(r)
    # PLATON integrates from z = -5 to 5 with more density near z = 0  (r = r_m)
    # Following code creates half of the logspace from 0.1 to z_max, and then flips it and appends them together

    z = -np.logspace(np.log10(0.1), np.log10(z_max), int(num_integral_points/2)) 
    z = np.append(z[::-1], -z)

    # For the effective cross section integral we need three components 
    # 1) Geometric cross section
    # 2) Probability distribution of particle size 
    # 3) Qext, which is given by the LX-MIE algorithm

    # ??? Still not sure about the constant here
    probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
    radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
    geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric

    # Now to create the array that is fed into Qext (2 pi r / lambda) that isn't refractive index

    # Ok, this part is also a bit of magic via Dr. Zhang
    # What np.newaxis adds dimensionality to the numpy arrays 
    # Instead of using a for loop to loop over all radii / all lambda, you can use 
    # Matrix division by adding dimensionality to both the radii and wl arrays 
    # You then flatten it out to just get all possible values 

    # If aersol = free, then the refractive index is not wavelength dependent 
    # We can just follow PLATON algorithm 
    if aerosol == 'free': 
        dense_xs = 2*np.pi*radii[np.newaxis,:] / wl_Mie[:,np.newaxis] # here the um crosses out 
        dense_xs = dense_xs.flatten()

        # Now we make the histogram 
        # np.histogram returns both [0] counts in the array and [1] the bin edges
        # I think this is so that the flatted dense_xs is a coarser array

        x_hist = np.histogram(dense_xs, bins='auto')[1]

        # Now to feed everything into Q_ext

        # Ok I am going to walk through this self statement as best as I can 
        # self in this case is atmosphere_solver, the main body of PLATON (like main.py)
        # In the preamble, we set self._mie_cache to MieCache()
        # This intializes MieCache and runs init, which creates the empty arrays for saving cs, Qexts, and etas
        # This next stage, get_and_update is a function that takes in eta and x_hist
        # It then tries to see if it can get the cross sections from the cache (see below) and if not
        # runs get_Qext (the LX_MIE algorithm) and adds it to the cache 
        # If there exists a close enough point, it just interpolates along the eta axis to get Q.

        ### From Platon 
        # Every time the value of Qext (m, x) is required, the cache is first consulted. 
        # If at least one value in the cache has the same m and an x within 5% of the requested x,
        # we perform linear interpolation on all cache values with the same m and return the interpolated value. 
        # If no cache value sat- isfies these criteria, we consider this a cache miss. 
        # Qext is then calculated for all cache misses and added to the cache.
        ###

        Qext_hist = get_and_update(eta, x_hist) 

        # This next part interpolated the Qext points that were made from the coarse x histogram 
        # And interpolates them back onto the dense x array 

        # Find the B-spline representation of a 1-D curve.
        spl = scipy.interpolate.splrep(x_hist, Qext_hist)
        # Evaluate a B-spline or its derivatives. (returns from the coarse x array to the dense one)
        Qext_intpl = scipy.interpolate.splev(dense_xs, spl)
        # Reshapes Qext in array with the first index - lambda and the second index - radius distribution 
        Qext_intpl = np.reshape(Qext_intpl, (len(wl_Mie), len(radii)))


    # If aersol != free, then the refractive index is wavelength dependent 
    # We have to change up the algorithm a bit by looping through wavelengths 
    if aerosol != 'free': 

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

    # Interpolate the eff_cross_section from wl_Mie back to native wl

    interp_eff_cross_section = interp1d(wl_Mie, eff_cross_section)
    eff_cross_section = interp_eff_cross_section(wl)

    #np.save('wl',wl,allow_pickle=True)
    #np.save('poseidon_cross_section',eff_cross_section,allow_pickle=True)

    # Uncomment to see cross sections 
    # import matplotlib.pyplot as plt 
    # plt.plot(wl,eff_cross_section)
    # plt.xlabel('wl_Mie')
    # plt.ylabel('effective cross section')
    # plt.title('Effective Cross Sections')
    # plt.show()

    # We redefine n and eff_cross_section to be more in line with Poseidon's exisiting language

    sigma_Mie = eff_cross_section


    return n_aerosol, sigma_Mie


###########################

def precompute_cross_sections_aerosol():

    global all_etas, all_xs, all_Qexts

    supported_aerosols_wl = ['SiO2', 'CH4', 'Fe2O3',
                          'H2O','KCl','MgSiO3','MnS',
                          'Na2S','NaCl','Tholin',
                          'SiO2_amorph','C','Cr','Fe', 'FeS', 'Mg2SiO4_amorph_sol-gel',
                          'Mg04Fe06SiO3_amorph_glass','Mg05Fe05SiO3_amorph_glass',
                          'Mg08Fe02SiO3_amorph_glass','Mg08Fe12SiO4_amorph_glass',
                          'MgFeSiO4_amorph_glass','MgO','MgSiO3_amorph_glass',
                          'MgSiO3_amorph_sol-gel','SiC','SiO','TiC','TiO2_anatase']

    sigma_Mie_all = []
    wl_Mie = []

    r_m_std_dev = 0.5
    z_max = 5
    num_integral_points = 100
    R_Mie = 1000

    for aerosol in supported_aerosols_wl:

        effective_cross_section_array = []

        #########################
        # Load in refractive indices (as function of wavelength)
        #########################
        file_name = directory + aerosol + '_complex.txt'
        file_as_numpy = np.loadtxt(file_name,skiprows=2).T

        wavelengths = file_as_numpy[0]
        wl_min = 0.2
        wl_max = 30
        wl_Mie = np.append(wl_Mie,wl_grid_constant_R(wl_min, wl_max, R_Mie))
        
        interp_reals = interp1d(wavelengths, file_as_numpy[1])
        interp_complexes = interp1d(wavelengths, file_as_numpy[2])
        eta_array = interp_reals(wl_Mie) + -1j *interp_complexes(wl_Mie)

        # Create the r_m array 

        r_m_array = 10**np.linspace(-3,1,1000)

        for r_m in r_m_array:

            #########################
            # Caculate the effective cross section of the particles (as a function of wavelength)
            #########################

            eff_cross_sections = np.zeros(len(wl_Mie))

            z = -np.logspace(np.log10(0.1), np.log10(z_max), int(num_integral_points/2)) 
            z = np.append(z[::-1], -z)

            probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
            radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
            geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric

            if aerosol != 'free': 

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

        title = '/Users/elijahmullens/Desktop/sigma_Mie_' + aerosol
        np.save(title,effective_cross_section_array,allow_pickle = True)
        sigma_Mie_all.append(effective_cross_section_array)
        all_etas = []
        all_xs = []
        all_Qexts = []

    title = '/Users/elijahmullens/Desktop/sigma_Mie_database'
    np.save('/Users/elijahmullens/Desktop/R_M_database', r_m_array, allow_pickle = True)
    np.save('/Users/elijahmullens/Desktop/WL_MIE_database', wl_Mie, allow_pickle = True)
    np.save(title,sigma_Mie_all,allow_pickle = True)

def precompute_cross_sections_one_aerosol(aerosol):

    global all_etas, all_xs, all_Qexts


    r_m_std_dev = 0.5
    z_max = 5
    num_integral_points = 100
    R_Mie = 1000

    sigma_Mie_all = []
    wl_Mie = []

    wl_min = 0.2
    wl_max = 30
    wl_Mie = np.append(wl_Mie,wl_grid_constant_R(wl_min, wl_max, R_Mie))

    r_m_array = 10**np.linspace(-3,1,1000)


    print(aerosol)

    effective_cross_section_array = []

    #########################
    # Load in refractive indices (as function of wavelength)
    #########################
    file_name = directory + aerosol + '_complex.txt'
    file_as_numpy = np.loadtxt(file_name,skiprows=2).T

    wavelengths = file_as_numpy[0]
    
    interp_reals = interp1d(wavelengths, file_as_numpy[1])
    interp_complexes = interp1d(wavelengths, file_as_numpy[2])
    eta_array = interp_reals(wl_Mie) + -1j *interp_complexes(wl_Mie)

    counter = 0

    for r_m in r_m_array:

        #########################
        # Caculate the effective cross section of the particles (as a function of wavelength)
        #########################

        if counter % 10 == 0:
            print(r_m)

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

        if aerosol != 'free': 

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

    title = './sigma_Mie_' + aerosol
    np.save(title,effective_cross_section_array,allow_pickle = True)
    sigma_Mie_all.append(effective_cross_section_array)
    all_etas = []
    all_xs = []
    all_Qexts = []