''' 
Functions for calculating clouds (not using the database).

'''
import numpy as np
import scipy
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt

from .core import wl_grid_constant_R


############################################################################################
# Empty Arrays for Qext calculations
############################################################################################

# All refractive indices
all_etas = []

# Inputs to Q_ext (2 pi r / lambda )
all_xs = []

# All Q_ext values already computed 
all_Qexts = []
all_Qscats = []
all_Qbacks = []
all_gs = []

# Wavelength Array for Mie Calculations, default resolution = 1000
wl_Mie = np.array([])

# Free or file_read switch
# This is just a saved variable that acts like a kill switch if the model is 
# Switched between free and file_read in the same notebook 
free_or_file = ''


############################################################################################
# Utility Functions
############################################################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Plot the cross section for a specific aersol in the database (for testing)
def plot_effective_cross_section_free(wl, r_m, r_i_real, r_i_complex):

    # For documentation, see Mie_cloud_free

    r_m_std_dev = 0.5
    z_max = 5
    num_integral_points = 100
    R_Mie = 1000

    wl_min = wl[0]
    wl_max = wl[-1]
    wl_Mie = wl_grid_constant_R(wl_min, wl_max, R_Mie)
    
    eta = complex(r_i_real,-r_i_complex)
    eta_array = np.full(len(wl_Mie),eta)
    eff_cross_sections = np.zeros(len(wl_Mie))
    z = -np.logspace(np.log10(0.1), np.log10(z_max), int(num_integral_points/2)) 
    z = np.append(z[::-1], -z)
    probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
    radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
    geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric
    dense_xs = 2*np.pi*radii[np.newaxis,:] / wl_Mie[:,np.newaxis] # here the um crosses out 
    dense_xs = dense_xs.flatten()
    x_hist = np.histogram(dense_xs, bins='auto')[1]
    Qext_hist = get_and_update(eta, x_hist) 
    spl = scipy.interpolate.splrep(x_hist, Qext_hist)
    Qext_intpl = scipy.interpolate.splev(dense_xs, spl)
    Qext_intpl = np.reshape(Qext_intpl, (len(wl_Mie), len(radii)))
    eff_cross_section = np.trapz(probs*geometric_cross_sections*Qext_intpl, z)
    interp_eff_cross_section = interp1d(wl_Mie, eff_cross_section)
    eff_cross_section = interp_eff_cross_section(wl)

    # Plot the interpolated effective cross sections
    label = 'r_m (um) : ' + str(r_m)
    title = "Index = " + str(r_i_real) + " + " + str(r_i_complex) + "j Effective Cross Section"
    plt.figure(figsize=(10,6))
    plt.plot(wl,eff_cross_section, label = label)
    plt.legend()
    plt.xlabel('Wavelengths (um)')
    plt.ylabel('Effective Cross Section')
    plt.title(title)
    plt.show()

    all_etas = []
    all_xs = []
    all_Qexts = []


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

def get_extinctions(m, xs):
    # Builds upon the algorithm from Kitzmann & Heng 2017 to compute Qext(x) for an array
    # of x's and refractive index m.  This algorithm is stable and does not
    # lead to numerical overflows.  Paper: https://arxiv.org/abs/1710.04946
    
    # This new function includes Q_scat, Q_back, and asymmetry parameter g
    # From : https://pymiescatt.readthedocs.io/en/latest/forward.html

    xs = np.array(xs)
    num_iterations = get_iterations_required(xs) 
    max_iter = max(max(num_iterations) , 1)
    
    A_mx = get_As(max_iter, m * xs)
    A_x = get_As(max_iter, xs)
    A_mx_plus_1 = get_As(max_iter+1, m * xs)
    A_x_plus_1 = get_As(max_iter+1, xs)


    # Initialize arrays for extinction efficiencies and asymmetry parameter
    Qext = np.zeros(len(xs))
    Qscat = np.zeros(len(xs))
    Qback = np.zeros(len(xs), dtype = 'complex128')

    # Asymmetry factor is made up of two sums 
    an_array = []
    bn_array = []
    g_1 = np.zeros(len(xs))
    g_2 = np.zeros(len(xs))

    curr_B = 1.0/(1 + 1j * (np.cos(xs) + xs*np.sin(xs))/(np.sin(xs) - xs*np.cos(xs)))
    curr_C = -1.0/xs + 1.0/(1.0/xs  + 1.0j)
    
    for i in range(1, max_iter):
        # The condition sets the nmax for the sums in each wavelength bin
        cond = num_iterations > i
        if i > 1:
            curr_C[cond] = -i/xs[cond] + 1.0/(i/xs[cond] - curr_C[cond])
            curr_B[cond] = curr_B[cond] * (curr_C[cond] + i/xs[cond])/(A_x[i][cond] + i/xs[cond])
            
        an = curr_B[cond] * (A_mx[i][cond]/m - A_x[i][cond])/(A_mx[i][cond]/m - curr_C[cond])
        bn = curr_B[cond] * (A_mx[i][cond]*m - A_x[i][cond])/(A_mx[i][cond]*m - curr_C[cond])

        Qext[cond]  += (2*i + 1) * (an + bn).real
        Qscat[cond] += (2*i + 1) * (abs(an)**2 + abs(bn)**2)
        Qback[cond] += (2*i + 1) * ((-1)**i) * (an-bn)

        
    # Calculating complete an and bn array for the asymmetry parameter 
    # This is a quick fix 
    curr_B = 1.0/(1 + 1j * (np.cos(xs) + xs*np.sin(xs))/(np.sin(xs) - xs*np.cos(xs)))
    curr_C = -1.0/xs + 1.0/(1.0/xs  + 1.0j)
    
    for i in range(1, max_iter+1):

        if i > 1:
            curr_C = -i/xs + 1.0/(i/xs - curr_C)
            curr_B = curr_B * (curr_C + i/xs)/(A_x_plus_1[i] + i/xs)
            
        an = curr_B * (A_mx_plus_1[i]/m - A_x_plus_1[i])/(A_mx_plus_1[i]/m - curr_C)
        bn = curr_B * (A_mx_plus_1[i]*m - A_x_plus_1[i])/(A_mx_plus_1[i]*m - curr_C)

        an_array.append(an)
        bn_array.append(bn)
    
    for i in range(1, max_iter):

        cond = num_iterations > i
        
        an = an_array[i-1][cond]
        bn = bn_array[i-1][cond]

        an_plus_1 = an_array[i][cond]
        bn_plus_1 = bn_array[i][cond]

        g_1[cond] += (i*(i+2))/(i+1) * (((an)*(an_plus_1.conjugate())) + ((bn)*(bn_plus_1.conjugate()))).real
        g_2[cond] += (2*i + 1)/(i*(i+1)) * (an * bn.conjugate()).real


    
    # Add all the prefactors to the sums
    Qext *= 2/xs**2
    Qscat *= 2/xs**2
    Qback = np.absolute(Qback)**2
    Qback *= 1/xs**2

    g = g_1 + g_2
    g *= (4/Qscat)*(1/xs**2)

    return Qext, Qscat, Qback, g

############################################################################################
# Auxiliary Functions (Retrieving and updating cached arrays)
############################################################################################

# Function that tries to find an existing Qext (within 5% marigin error) via interpolation 
# Or returns a miss 
# INPUTS : Refractive Indices array, 2 pi r / lambda array

# max_frac_error = 0.05

def get_from_cache(eta, xs, max_frac_error = 0.05):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts, all_Qscats, all_Qbacks, all_gs

    # Create an array of nans the same length as xs 
    result_Qext = np.full(len(xs),np.nan)
    result_Qscat = np.full(len(xs),np.nan)
    result_Qback = np.full(len(xs),np.nan)
    result_g = np.full(len(xs),np.nan)

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
        all_Qscats_eta = all_Qscats[x_s_index]
        all_Qbacks_eta = all_Qbacks[x_s_index]
        all_gs_eta = all_gs[x_s_index]

    else:
        return result_Qext, result_Qscat, result_Qback, result_g

    # If its the first iteration, Qext is empty 
    if len(all_xs_eta) == 0:
        return result_Qext, result_Qscat, result_Qback, result_g
    
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
    result_Qext[in_cache] = np.interp(
        xs[in_cache],
        all_xs_eta,
        all_Qexts_eta,)
    
    result_Qscat[in_cache] = np.interp(
        xs[in_cache],
        all_xs_eta,
        all_Qscats_eta,)
    
    result_Qback[in_cache] = np.interp(
        xs[in_cache],
        all_xs_eta,
        all_Qbacks_eta,)

    result_g[in_cache] = np.interp(
        xs[in_cache],
        all_xs_eta,
        all_gs_eta,)

    return result_Qext, result_Qscat, result_Qback, result_g

# Function that adds the new Qext if there was a miss 
# INPUTS : eta, xs, and new Qexts
def add(eta, xs, Qexts, Qscats, Qbacks, g, size_limit=1000000):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts, all_Qscats, all_Qbacks, all_gs

    # ??? I think this is to prevent some sort of bug, I can't think of one though 
    if len(xs) == 0:
        return
    
    # Add to existing arrays in the cache 
    if eta in all_etas:
        x_s_index = all_etas.index(eta)
        all_xs[x_s_index] = np.append(all_xs[x_s_index], xs)
        all_Qexts[x_s_index] = np.append(all_Qexts[x_s_index], Qexts)
        all_Qscats[x_s_index] = np.append(all_Qscats[x_s_index], Qscats)
        all_Qbacks[x_s_index] = np.append(all_Qbacks[x_s_index], Qbacks)
        all_gs[x_s_index] = np.append(all_gs[x_s_index], g)

        # In order to save memory, if there are more than the size limit it deletes some random ones 
        # Only matters if an eta already exists
        if len(all_xs[x_s_index]) > size_limit:
            to_remove = np.random.choice(
                range(len(all_xs[x_s_index])), len(all_xs[x_s_index]) - size_limit + 1,
                replace=False)
            
            all_xs[x_s_index] = np.delete(all_xs[x_s_index], to_remove)
            all_Qexts[x_s_index] = np.delete(all_Qexts[x_s_index], to_remove)
            all_Qscats[x_s_index] = np.delete(all_Qscats[x_s_index], to_remove)
            all_Qbacks[x_s_index] = np.delete(all_Qbacks[x_s_index], to_remove)
            all_gs[x_s_index] = np.delete(all_gs[x_s_index], to_remove)

        # Sort all of the cahced arrays 
        p = np.argsort(all_xs[x_s_index])
        all_xs[x_s_index] = all_xs[x_s_index][p]
        all_Qexts[x_s_index] = all_Qexts[x_s_index][p]
        all_Qscats[x_s_index] = all_Qscats[x_s_index][p]
        all_Qbacks[x_s_index] = all_Qbacks[x_s_index][p]
        all_gs[x_s_index] = all_gs[x_s_index][p]

    # If not, append a new array for a new eta 
    else:

        all_xs.append(np.array(xs))
        all_Qexts.append(np.array(Qexts))
        all_Qscats.append(np.array(Qscats))
        all_Qbacks.append(np.array(Qscats))
        all_gs.append(np.array(g))
        all_etas.append(eta)

        # Sort the cached arrays 
        p = np.argsort(all_xs[-1])
        all_xs[-1] = all_xs[-1][p]
        all_Qexts[-1] = all_Qexts[-1][p]
        all_Qscats[-1] = all_Qscats[-1][p]
        all_Qbacks[-1] = all_Qbacks[-1][p]
        all_gs[-1] = all_gs[-1][p]

# Function that either 1) Updates Qext or 2) Returns it if value already exist
# INPUTS : Refractive Indices array, 2 pi r / lambda array
def get_and_update(eta,xs):

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts

    # This array will be full of nans and Qext (corresponding to misses and hits in the cache)
    Qexts_eta, Qscats_eta, Qbacks_eta, g_eta = get_from_cache(eta, xs)

    # If there are ANY misses, we run the LX_MIE algorithm to find the Qext
    # And then we add them to the cached arrays 
    cache_misses = np.isnan(Qexts_eta)
        
    # Not a new eta, but with misses
    if np.sum(cache_misses) > 0:
        # LX MIE Algorithm
        Qexts_eta[cache_misses], Qscats_eta[cache_misses], Qbacks_eta[cache_misses], g_eta[cache_misses] = get_extinctions(eta, xs[cache_misses])
        # Adds to cahced arrays
        add(eta, xs[cache_misses], Qexts_eta[cache_misses], Qscats_eta[cache_misses], Qbacks_eta[cache_misses],g_eta[cache_misses])

    return Qexts_eta, Qscats_eta, Qbacks_eta, g_eta


############################################################################################
# Main Cloud Functions
############################################################################################

def Mie_cloud_free(P,wl,r, H, n,
              r_m,
              r_i_real,
              r_i_complex,
              P_cloud = 0,
              log_n_max = 0, 
              fractional_scale_height = 0,
              log_X_Mie = 0,
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

        H (np.array of float) : 
            gas scale height

        n (np.array of float) :
            total number density array 

        r_m  (float) : 
            Mean particle sizes (in um)



        -------- Semi- Optional Arguments -------

        Fuzzy Deck Arguments

        P_cloud (float) : 
            Cloud Top Pressure (everything below P_cloud is opaque). 
            If cloud coverage is complete, P_cloud is located at R_p

        log_n_max (float) : 
            Logorithm of maximum number density (at the cloud top)

        fractional_scale_height (float) :
            fractional scale height of aerosol 

        Uniform X Arguments

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

    
    Returns: n_aerosol, sigma_Mie
          
    '''

    # DELETE THIS LATER 
    global all_etas, all_xs, all_Qexts, wl_Mie, free_or_file


    #########################
    # Set up wl_mie (a wl array with R = 1000). This is only for aerosol = 'free' or 'file'
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

    # At this point, the r_i_real, r_i_complex is either a float or an array (free vs file_read)

    # If its a scalar, this will work
    try:
        # Constant eta array 
        eta = complex(r_i_real,-r_i_complex)
        eta_array = np.full(len(wl_Mie),eta)
    # Else, its an array (file_read) and this will run
    except :
        eta = 0
        eta_array = r_i_real + -1j  * r_i_complex

    # Kill switch if the model was switched in the same kernel 
    # I.e. clear out the qext arrays 

    if eta != 0:
        # If the saved killswitch is empty, make it not empty
        if free_or_file == '':
            free_or_file = 'free'
        # Check to see if model has changed 
        elif free_or_file == 'file':
            all_etas = []
            all_xs = [] 
            all_Qexts = []
            free_or_file == 'free'
        
    else:
        # If the saved killswitch is empty, make it not empty
        if free_or_file == '':
            free_or_file = 'file'
        # Check to see if model has changed 
        elif free_or_file == 'free':
            all_etas = []
            all_xs = [] 
            all_Qexts = []
            free_or_file == 'file'

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

    # If aerosol != 'file_read'
    if eta != 0:

        # Now to create the array that is fed into Qext (2 pi r / lambda) that isn't refractive index

        # Ok, this part is also a bit of magic via Dr. Zhang
        # What np.newaxis adds dimensionality to the numpy arrays 
        # Instead of using a for loop to loop over all radii / all lambda, you can use 
        # Matrix division by adding dimensionality to both the radii and wl arrays 
        # You then flatten it out to just get all possible values 

        # If aersol = free, then the refractive index is not wavelength dependent 
        # We can just follow PLATON algorithm 
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

        Qext_hist, Qscat_hist, Qback_hist, g_hist = get_and_update(eta, x_hist) 

        # This next part interpolated the Qext points that were made from the coarse x histogram 
        # And interpolates them back onto the dense x array 

        # SSA and weighted g 
        w_hist = Qscat_hist/Qext_hist

        # Revert from coarse Qext back to dense Qext (/ coarse back to dense for everything)
        spl = scipy.interpolate.splrep(x_hist, Qext_hist)
        Qext_intpl = scipy.interpolate.splev(dense_xs, spl)

        spl = scipy.interpolate.splrep(x_hist, Qscat_hist)
        Qscat_intpl = scipy.interpolate.splev(dense_xs, spl)

        spl = scipy.interpolate.splrep(x_hist, w_hist)
        w_intpl = scipy.interpolate.splev(dense_xs, spl)

        spl = scipy.interpolate.splrep(x_hist, Qback_hist)
        Qback_intpl = scipy.interpolate.splev(dense_xs, spl)

        spl = scipy.interpolate.splrep(x_hist, g_hist)
        g_intpl = scipy.interpolate.splev(dense_xs, spl)

        # Reshape the mega array so that the first index is wavelngth, second is radius 
        Qext_intpl = np.reshape(Qext_intpl, (len(wl_Mie), len(radii)))
        Qscat_intpl = np.reshape(Qscat_intpl, (len(wl_Mie), len(radii)))
        Qback_intpl = np.reshape(Qback_intpl, (len(wl_Mie), len(radii)))
        w_intpl = np.reshape(w_intpl, (len(wl_Mie), len(radii)))
        g_intpl = np.reshape(g_intpl, (len(wl_Mie), len(radii)))


    # aerosol = 'file_read'
    # Have to loop through each wavelength
    else: 

        Qext_intpl_array = []
        Qscat_intpl_array = []
        Qback_intpl_array = []
        w_intpl_array = []
        g_intpl_array = []

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
            Qext_hist, Qscat_hist, Qback_hist, g_hist = get_and_update(eta, x_hist) 

            # Revert from coarse Qext back to dense Qext 
            spl = scipy.interpolate.splrep(x_hist, Qext_hist)
            Qext_intpl = scipy.interpolate.splev(dense_xs, spl)

            # SSA and weighted g 
            w_hist = Qscat_hist/Qext_hist

            # Revert from coarse Qext back to dense Qext (/ coarse back to dense for everything)
            spl = scipy.interpolate.splrep(x_hist, Qext_hist)
            Qext_intpl = scipy.interpolate.splev(dense_xs, spl)

            spl = scipy.interpolate.splrep(x_hist, Qscat_hist)
            Qscat_intpl = scipy.interpolate.splev(dense_xs, spl)

            spl = scipy.interpolate.splrep(x_hist, w_hist)
            w_intpl = scipy.interpolate.splev(dense_xs, spl)

            spl = scipy.interpolate.splrep(x_hist, Qback_hist)
            Qback_intpl = scipy.interpolate.splev(dense_xs, spl)

            spl = scipy.interpolate.splrep(x_hist, g_hist)
            g_intpl = scipy.interpolate.splev(dense_xs, spl)

            # Append it to the array that will have all the Qext
            Qext_intpl_array.append(Qext_intpl)
            Qscat_intpl_array.append(Qscat_intpl)
            Qback_intpl_array.append(Qback_intpl)
            w_intpl_array.append(w_intpl)
            g_intpl_array.append(g_intpl)

        # Reshape the mega array so that the first index is wavelngth, second is radius 
        Qext_intpl = np.reshape(Qext_intpl_array, (len(wl_Mie), len(radii)))
        Qscat_intpl = np.reshape(Qscat_intpl_array, (len(wl_Mie), len(radii)))
        Qback_intpl = np.reshape(Qback_intpl_array, (len(wl_Mie), len(radii)))
        w_intpl = np.reshape(w_intpl_array, (len(wl_Mie), len(radii)))
        g_intpl = np.reshape(g_intpl_array, (len(wl_Mie), len(radii)))


    # Effective Cross section is a trapezoidal integral
    eff_ext_cross_section = np.trapz(probs*geometric_cross_sections*Qext_intpl, z)

    # Scattering Cross section 
    eff_scat_cross_section = np.trapz(probs*geometric_cross_sections*Qscat_intpl, z)

    # Absorption Cross section
    eff_abs_cross_section = eff_ext_cross_section - eff_scat_cross_section

    # BackScatter Cross section 
    eff_back_cross_section = np.trapz(probs*geometric_cross_sections*Qback_intpl, z)

    # Effective w and g
    eff_w = np.median(w_intpl, axis=1)
    eff_g = np.median(g_intpl, axis=1)

    # Interpolate the eff_cross_section from wl_Mie back to native wl
    # This can probably be made faster 
    interp = interp1d(wl_Mie, eff_ext_cross_section)
    eff_ext = interp(wl)

    interp = interp1d(wl_Mie, eff_abs_cross_section)
    eff_abs = interp(wl)

    interp = interp1d(wl_Mie, eff_scat_cross_section)
    eff_scat = interp(wl)

    interp = interp1d(wl_Mie, eff_back_cross_section)
    eff_back = interp(wl)

    interp = interp1d(wl_Mie, eff_g)
    eff_g = interp(wl)

    interp = interp1d(wl_Mie, eff_w)
    eff_w = interp(wl)

    print('================================')
    print('This print statement is from clouds_LX_Mie_emission.py')
    print('This is to test and make sure that things are working.')
    
    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl, eff_ext, label = label)
    title = 'Effective Extinction (Scattering + Absorption) Cross Sections' 
    plt.title(title)
    plt.ylabel('Effective Cross Section')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_g, label = label)
    title = 'Asymmetry Parameter\n0 (completely back scattering) and +1 (total forward scattering) '
    plt.title(title)
    plt.ylabel('g')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_w, label = label)
    title = 'Single Scattering Albedo\n0 (black, completely absorbing) to 1 (white, completely scattering)'
    plt.title(title)
    plt.ylabel('w')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    print('Thanks for your patience! Make sure to set clouds_LX_Mie.py back in core.py for transmission')
    print('================================')

    # We redefine n and eff_cross_section to be more in line with Poseidon's exisiting language
    sigma_Mie = eff_ext

    # To work with Numba
    n_aerosol_array = []
    n_aerosol_array.append(n_aerosol)

    sigma_Mie_array = []
    sigma_Mie_array.append(sigma_Mie)


    return n_aerosol_array, sigma_Mie_array
