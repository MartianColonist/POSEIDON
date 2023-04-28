''' 
Functions for calculating clouds.

'''


import numpy as np
import scipy
from scipy.interpolate import interp1d

############################################################################################
# Where refractive index data is 
############################################################################################

directory = '/Users/elijahmullens/Desktop/Poseidon-temp/input/opacity/refractive_indices/'

############################################################################################
# Empty Arrays for Qext (later to be changed to a shared memory array, see chemistry.py)
############################################################################################

# All refractive indices
all_etas = np.array([], dtype=complex)

# Inputs to Q_ext (2 pi r / lambda )
all_xs = np.array([])

# All Q_ext values already computed 
all_Qexts = np.array([])

############################################################################################
# Utility Functions
############################################################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

############################################################################################
# LX MIE Algorithm
############################################################################################

def get_iterations_required(xs, c=4.3):
    # c=4.3 corresponds to epsilon=1e-8, according to Cachorro & Salcedo 2001
    # (https://arxiv.org/abs/physics/0103052)
    num_iters = xs + c * xs**(1.0/3)
    num_iters = num_iters.astype(int) + 2
    return num_iters

def get_An(zs, n):
    # Evaluate A_n(z) for an array of z's using the continued fraction method.
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
def get_from_cache(eta, xs, max_frac_error = 0.05):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    # Create an array of nans the same length as xs 
    result = np.full(len(xs),np.nan)

    # If its the first iteration, Qext is empty 
    if len(all_xs) == 0:
        return result
    
    # This just returns an array of True
    in_cache = np.ones(len(xs), dtype=bool)

    # Find the indices into a sorted array all_xs such that, 
    # if the corresponding elements in xs were inserted before the indices, 
    # the order of a would be preserved.
    # If == len(all_xs), then its above the maximum value in all_xs
    closest_matches = np.searchsorted(all_xs, xs)

    # np.logical_or computes the truth value of x1 OR x2 element wise 
    # if closest_matches == 0 (Closest index is the lower bound) or the highest bound 
    # set that value automatically to false
    in_cache[np.logical_or(closest_matches == 0, closest_matches == len(all_xs))] = False
    
    # Makes all the indices that were above the maximum one less (so that it works with all_etas array)
    # all_etas is ordered the same way as all_xs 
    # Pretty much, its a possibility that you can have a particle a specific size but not
    # Have the same refractive index. These statements make those a miss 
    closest_matches[closest_matches == len(all_xs)] -= 1
    in_cache[all_etas[closest_matches] != eta] = False

    # Computes the fractional error of surviving matches 
    # If the fractional error is greater than some amount (5%), its set to false 
    frac_errors = np.abs(all_xs[closest_matches] - xs)/xs
    in_cache[frac_errors > max_frac_error] = False

    # If the all_etas array is empty, return an empty result array as well 
    # This can happen occasionally after the first initialization I think (we have the x, but not the eta)
    if np.sum(all_etas == eta) == 0: 
        return result
    
    # Interpolates the results for hits
    result[in_cache] = np.interp(
        xs[in_cache],
        all_xs[all_etas == eta],
        all_Qexts[all_etas == eta])

    return result

# Function that adds the new Qext if there was a miss 
# INPUTS : eta, xs, and new Qexts
def add(eta, xs, Qexts, size_limit=1000000):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    # ??? I think this is to prevent some sort of bug, I can't think of one though 
    if len(xs) == 0:
        return
    
    # Adding the new xs, Qext, and refractive indices to the cached arrays 
    all_xs = np.append(all_xs, xs)
    all_Qexts = np.append(all_Qexts, Qexts)
    # ??? I am honestly not too sure why they do this? Multiply the refractive index y
    all_etas = np.append(all_etas, np.array([eta] * len(xs)))

    # In order to save memory, if there are more than the size limit it deletes some random ones 
    if len(all_xs) > size_limit:
        to_remove = np.random.choice(
            range(len(all_xs)), len(all_xs) - size_limit + 1,
            replace=False)
        
        all_xs = np.delete(all_xs, to_remove)
        all_Qexts = np.delete(all_Qexts, to_remove)
        all_etas = np.delete(all_etas, to_remove)

    # Sort all of the cahced arrays 
    p = np.argsort(all_xs)
    all_xs = all_xs[p]
    all_Qexts = all_Qexts[p]
    all_etas = all_etas[p]

# Function that either 1) Updates Qext or 2) Returns it if value already exist
# INPUTS : Refractive Indices array, 2 pi r / lambda array
def get_and_update(eta,xs):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    # This array will be full of nans and Qext (corresponding to misses and hits in the cache)
    Qexts = get_from_cache(eta, xs)

    # If there are ANY misses, we run the LX_MIE algorithm to find the Qext
    # And then we add them to the cached arrays 
    cache_misses = np.isnan(Qexts)

    if np.sum(cache_misses) > 0:
        # LX MIE Algorithm
        Qexts[cache_misses] = get_Qext(eta, xs[cache_misses])
        # Adds to cahced arrays
        add(eta, xs[cache_misses], Qexts[cache_misses])
    return Qexts

############################################################################################
# Main Function
############################################################################################

def mie_cloud(P,wl,r,
              P_cloud, r_m, n_max, fractional_scale_height, H,
              aerosol = 'free',
              r_i_real = 0,
              r_i_complex = 0,
              r_m_std_dev = 0.5,
              z_max = 5,
              num_integral_points = 100):


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
            Mean particle size (in m)

        n_max (float) : 
            Maximum number density (at the cloud top)

        fractional_scale_height (float) :
            fractional scale height of aerosol

        H (np.array of float) : 
            gas scale height

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

    
    Returns:
          
    '''

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    supported_aerosols = ['SiO2', 'Al2O3', 'CaTiO3', 'CH4', 'Fe2O3', 'Fe2SiO4',
                          'H2O','hexene','Hibonite','KCl','Mg2SiO4',
                          'Mg2SiO4poor','MgAl2O4','MgSiO3','MnS',
                          'Na2S','NaCl','SiO2','tholin','TiO2','ZnS']

    #########################
    # Error messages 
    #########################

    if aerosol not in supported_aerosols and aerosol != 'free':
        return Exception('Mie scattering does not support precalculated refractives indices of : ', aerosol)
    
    #########################
    # Load in refractive indices (as function of wavelength)
    #########################

    if aerosol in supported_aerosols:
        ## Load in refractive index w/ wl. Then interpolate it
        # This can be made into a saved array as well, for retrievals
        file_name = directory + aerosol + '_complex.txt'
        file_as_numpy = np.loadtxt(file_name,skiprows=2).T
        wavelengths = file_as_numpy[0]
        interp_reals = interp1d(wavelengths, file_as_numpy[1])
        interp_complexes = interp1d(wavelengths, file_as_numpy[2])
        eta_array = interp_reals(wl) + 1j *interp_complexes(wl)

    else:
        # Apply constant eta to entire array 
        eta = complex(r_i_real,r_i_complex)
        eta_array = np.full(len(wl),eta)


    #########################
    # Calculate the number density above the cloud top
    #########################
    
    # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
    n = np.empty_like(r)
    P_cloud_index = find_nearest(P,P_cloud)
    # Find the radius corresponding to the cloud top pressure 
    cloud_top_height = r[P_cloud_index]
    # Height above cloud 
    h = r[P_cloud_index:] - cloud_top_height
    # Find number density below and above P_cloud
    n[:P_cloud_index] = n_max
    n[P_cloud_index:] = n_max * np.exp(-h/(fractional_scale_height*H))

    #########################
    # Caculate the effective cross section of the particles (as a function of wavelength)
    #########################

    # There is an effective cross section for every wavelength 
    # Therefore, we need an array of resultant effective cross sections the same size as wl 

    eff_cross_sections = np.zeros(len(wl))

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
    geometric_cross_sections = np.pi * radii**2

    # Now to create the array that is fed into Qext (2 pi r / lambda) that isn't refractive index

    # Ok, this part is also a bit of magic via Dr. Zhang
    # What np.newaxis adds dimensionality to the numpy arrays 
    # Instead of using a for loop to loop over all radii / all lambda, you can use 
    # Matrix division by adding dimensionality to both the radii and wl arrays 
    # You then flatten it out to just get all possible values 

    dense_xs = 2*np.pi*radii[np.newaxis,:] / wl[:,np.newaxis]
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
    Qext_intpl = np.reshape(Qext_intpl, (len(wl), len(radii)))

    # Effective Cross section is a trapezoidal integral
    eff_cross_section = np.trapz(probs*geometric_cross_sections*Qext_intpl, z)

    # Kappa = n * eff_cross_section 
    # Again, I think the indices here are in order to make the matrix multiplication work
    # This will need to be reshaped probably 
    # kappa_mie = n[np.newaxis, :, np.newaxis] * eff_cross_section[np.newaxis, np.newaxis, :]

    n_cloud = n 
    sigma_mie = eff_cross_section 

    return n_cloud, sigma_mie






