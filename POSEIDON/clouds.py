######################################################
######################################################
#  Functions that use the aerosol database 
######################################################
######################################################

import numpy as np
import scipy
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
import os 
import glob

from .utility import shared_memory_array
from .supported_chemicals import aerosol_supported_species
#from .core import wl_grid_constant_R

############################################################################################
# Utility Functions
############################################################################################

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

    # Fix for numerical rounding error
    wl[0] = wl_min
    wl[-1] = wl_max
    
    return wl

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
        database = h5py.File(input_file_path + 'opacity/aerosol_database.hdf5', 'r')
    except :
        raise Exception('Please put aerosol_database.hdf5 in the inputs/opacity folder')
    
    from .core import wl_grid_constant_R

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

# File read in utility functions 
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

    #########################
    # Load in refractive indices (as function of wavelength)
    #########################
    print('Loading in : ', file_name)
    try:
        file_as_numpy = np.loadtxt(file_name, comments = '#').T
    except:
        file_as_numpy = np.loadtxt(file_name, skiprows = 2).T

    # If its index, wavelength, n, k we need to do something different. 
    if len(file_as_numpy) == 4:
        wavelengths = file_as_numpy[1]
        real_indices = file_as_numpy[2]
        imaginary_indices = file_as_numpy[3]
        file_as_numpy = np.array([wavelengths,real_indices,imaginary_indices])

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

    molecule = file_name.split('/')[1][:-4]
    molecule = molecule.split('_')[0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 6)
    suptitle = 'Refractive Indices for ' + molecule
    #suptitle = 'Refractive Indices for H$_2$O'
    fig.suptitle(suptitle)
    ax1.plot(wl_Mie, real_indices)
    ax2.plot(wl_Mie, imaginary_indices)

    ax1.set_xlabel('Wavelength ($\mu$m)')
    ax2.set_xlabel('Wavelength ($\mu$m)')
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
    eff_ext_cross_section, eff_scat_cross_section, eff_abs_cross_section, eff_back_cross_section, eff_w, eff_g = precompute_cross_sections_from_indices(wl_Mie,r_i_real,r_i_complex, r_m)

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_ext_cross_section, label = label)
    title = 'Effective Extinction (Scattering + Absorption) Cross Sections ' + file_name.split('/')[1][:-4] + '\n'
    plt.title(title)
    plt.ylabel('Effective Cross Section')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_scat_cross_section, label = label)
    title = 'Effective Scattering Cross Sections ' + file_name.split('/')[1][:-4]+ '\n'
    plt.title(title)
    plt.ylabel('Effective Cross Section')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_abs_cross_section, label = label)
    title = 'Effective Absorption Cross Sections ' + file_name.split('/')[1][:-4]+ '\n'
    plt.title(title)
    plt.ylabel('Effective Cross Section')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_back_cross_section, label = label)
    title = 'Effective Back-Scattering Cross Sections ' + file_name.split('/')[1][:-4]+ '\n'
    plt.title(title)
    plt.ylabel('Effective Cross Section')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_w, label = label)
    title = 'Single Scattering Albedo ' + file_name.split('/')[1][:-4] + '\n0 (black, completely absorbing) to 1 (white, completely scattering)'+ '\n'
    plt.title(title)
    plt.ylabel('SSA')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    label = 'r_m ' + str(r_m) + ' (um)'
    plt.plot(wl_Mie, eff_g, label = label)
    title = 'Asymmetry Parameter ' + file_name.split('/')[1][:-4] + '\n0 (Rayleigh limit) and +1 (total foward scattering limit) '+ '\n'
    plt.title(title)
    plt.ylabel('g')
    plt.xlabel('Wavelength (um)')
    plt.legend()
    plt.show()

    molecule = file_name.split('/')[1][:-4]
    molecule = molecule.split('_')[0]
    title = molecule + ' : Normalized $\sigma_{ext}$ , Asymmetry Parameter, and Single Scattering Albedo' + '\n $r_m$ = ' + str(round(r_m, 3)) + ' ($\mu$m)' +  '\n $\omega$ : 0 (black, completely absorbing) to 1 (white, completely scattering)'+ '\n g : 0 (Rayleigh Limit) to 1 (Total Forward Scattering) '+ '\n'
    plt.figure(figsize=(10,6))
    plt.plot(wl_Mie, eff_ext_cross_section/np.max(eff_ext_cross_section), label = '$\sigma_{ext}$ = $\sigma_{abs}$ + $\sigma_{scat}$ (Normalized)')
    plt.plot(wl_Mie, eff_w, label = 'Single Scattering Albedo ($\omega$)')
    plt.plot(wl_Mie, eff_g, label = 'Asymmetry Parameter (g)')
    plt.title(title)
    plt.xlabel('Wavelength ($\mu$m)')
    plt.legend()
    plt.show()

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

    # For the effective cross section integral we need three components 
    # 1) Geometric cross section
    # 2) Probability distribution of particle size 
    # 3) Qext, which is given by the LX-MIE algorithm

    # ??? Still not sure about the constant here
    probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
    radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
    geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric

    dense_xs = 2*np.pi*radii[np.newaxis,:] / wl_Mie[:,np.newaxis] # here the um crosses out 
    dense_xs = dense_xs.flatten()

    x_hist = np.histogram(dense_xs, bins='auto')[1]

    Qext_hist, Qscat_hist, Qback_hist, g_hist = get_and_update(eta, x_hist) 
    w_hist = Qscat_hist/Qext_hist

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

    Qext_intpl = np.reshape(Qext_intpl, (len(wl_Mie), len(radii)))
    Qscat_intpl = np.reshape(Qscat_intpl, (len(wl_Mie), len(radii)))
    Qback_intpl = np.reshape(Qback_intpl, (len(wl_Mie), len(radii)))
    w_intpl = np.reshape(w_intpl, (len(wl_Mie), len(radii)))
    g_intpl = np.reshape(g_intpl, (len(wl_Mie), len(radii)))

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

    # Plot the interpolated effective cross sections
    label = 'r_m (um) : ' + str(r_m)
    title = "Index = " + str(r_i_real) + " + " + str(r_i_complex) + "j Effective Cross Section"
    plt.figure(figsize=(10,6))
    plt.plot(wl,eff_ext, label = label)
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

def contribution_clouds_transmission(model, planet, star, wl, opac, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params_og,
                        show_chemical_species = True, put_one_in = True):

    from POSEIDON.core import define_model
    from POSEIDON.core import make_atmosphere
    from POSEIDON.core import compute_spectrum
    from POSEIDON.visuals import plot_spectra
    from POSEIDON.utility import plot_collection

    spectra_array = []
    spectra_labels = []
    colour_list = ['dimgray', 'darkturquoise', 'springgreen', 'darkorchid', 'salmon', '#ff7f00', 'hotpink', 'red', 'orange', 'green', 'blue', 'purple']

    # Real spectrum 
    model_name = 'Full Spectrum'
    bulk_species = ['H2','He']
    species_list = model['param_species']
    param_species = species_list

    # Only if the model has aerosol species do we include them 
    # Compute the full spectrum that the contribution will build onto 
    if model['cloud_model'] != 'Mie':

        model_full = define_model(model_name,bulk_species,param_species,
                                PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                cloud_model = model['cloud_model'], cloud_type = model['cloud_type'])

        atmosphere_full = make_atmosphere(planet, model_full, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params_og)

        spectrum_full = compute_spectrum(planet, star, model_full, atmosphere_full, opac, wl,
                                        spectrum_type = 'transmission')

    else:
        aerosol_species = model['aerosol_species']

        model_full = define_model(model_name,bulk_species,param_species,
                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                        aerosol_species = aerosol_species)

        atmosphere_full = make_atmosphere(planet, model_full, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params_og)

        spectrum_full = compute_spectrum(planet, star, model_full, atmosphere_full, opac, wl,
                                        spectrum_type = 'transmission')
        

    # H + He atmosphere
    # If and only if you are preforming a 'put one in' model will this make sense 

    if put_one_in == True:

        model_name = 'Bulk'
        param_species = []

        # Model H-H2
        model_bulk = define_model(model_name,bulk_species,param_species,
                                PT_profile = model['PT_profile'], X_profile = model['X_profile'])

        log_X_params = np.array([])
        atmosphere_bulk = make_atmosphere(planet, model_bulk, P, P_ref, R_p_ref, PT_params_og)

        spectrum_bulk = compute_spectrum(planet, star, model_bulk, atmosphere_bulk, opac, wl,
                                        spectrum_type = 'transmission')
        
        spectra_array.append(spectrum_bulk)

        spectra_labels.append('Bulk Species')


    # Model Clouds 
    if model['cloud_model'] == 'MacMad17':

        if put_one_in == True:

            model_clouds = define_model(model_name,bulk_species,param_species,
                                    PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                    cloud_model = model['cloud_model'], cloud_type = model['cloud_type'])

            log_X_params = np.array([])
            atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params, cloud_params_og)

            spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                            spectrum_type = 'transmission')
            
            spectra_array.append(spectrum_clouds)
            spectra_labels.append(model['cloud_type'])

        else:

            model_no_clouds = define_model(model_name,bulk_species,param_species,
                        PT_profile = model['PT_profile'], X_profile = model['X_profile'])

            atmosphere_no_clouds = make_atmosphere(planet, model_no_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params_og)

            spectrum_no_clouds = compute_spectrum(planet, star, model_no_clouds, atmosphere_no_clouds, opac, wl,
                                            spectrum_type = 'transmission')
            
            spectra_array.append(spectrum_no_clouds)
            label = '- ' + model['cloud_type']
            spectra_labels.append(label)


    elif model['cloud_model'] == 'Mie':

        aerosol_species = model['aerosol_species']

        if put_one_in == True:

            model_clouds = define_model(model_name,bulk_species,param_species,
                            PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                            cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                            aerosol_species = aerosol_species)

            log_X_params = np.array([])
            atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params, cloud_params_og)

            spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                            spectrum_type = 'transmission')
            
            spectra_array.append(spectrum_clouds)

            if len(aerosol_species) > 1:
                label = 'Total Cloud'
            else:
                label = aerosol_species[0]

            spectra_labels.append(label)

            if len(aerosol_species) > 1:

                if model['cloud_type'] == 'slab' or model['cloud_type'] == 'opaque_deck_plus_slab':

                    log_X_indices = np.where(np.char.find(model['cloud_param_names'], 'log_X')!= -1)[0]

                    for n in log_X_indices:

                        cloud_params = np.copy(cloud_params_og)

                        for m in log_X_indices:
                            if m != n:
                                cloud_params[m] = -50

                        model_clouds = define_model(model_name,bulk_species,param_species,
                                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                        aerosol_species = aerosol_species)

                        log_X_params = np.array([])
                        atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params, cloud_params)

                        spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                                        spectrum_type = 'transmission')
                        
                        spectra_array.append(spectrum_clouds)

                        spectra_labels.append(model['cloud_param_names'][n].split('_')[2])
                
                
                elif model['cloud_type'] == 'fuzzy_deck_plus_slab':

                    f_index = np.where(np.char.find(model['cloud_param_names'], 'f')!= -1)[0]
                    log_X_indices = np.where(np.char.find(model['cloud_param_names'], 'log_X')!= -1)[0]

                    for n in log_X_indices:

                        cloud_params = np.copy(cloud_params_og)

                        # Sets the f parameter close to 0, keeping the opaque slab and getting rid of the fuzziness
                        cloud_params[f_index] = 0.000001

                        for m in log_X_indices:
                            if m != n:
                                cloud_params[m] = -50

                        model_clouds = define_model(model_name,bulk_species,param_species,
                                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                        aerosol_species = aerosol_species)

                        log_X_params = np.array([])
                        atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params, cloud_params)

                        spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                                        spectrum_type = 'transmission')
                        
                        spectra_array.append(spectrum_clouds)

                        spectra_labels.append(model['cloud_param_names'][n].split('_')[2])

                    # Turn off both except the fuzzy deck 
                    cloud_params = np.copy(cloud_params_og)
                    for n in log_X_indices:
                        cloud_params[n] = -50

                    model_clouds = define_model(model_name,bulk_species,param_species,
                                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                        aerosol_species = aerosol_species)
                    
                    atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params, cloud_params)

                    spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                                        spectrum_type = 'transmission')
                        
                    spectra_array.append(spectrum_clouds)
                    
                    spectra_labels.append(model['cloud_param_names'][f_index][0].split('_')[1])

        else:

            model_no_clouds = define_model(model_name,bulk_species,param_species,
                PT_profile = model['PT_profile'], X_profile = model['X_profile'])

            atmosphere_no_clouds = make_atmosphere(planet, model_no_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params_og)

            spectrum_no_clouds = compute_spectrum(planet, star, model_no_clouds, atmosphere_no_clouds, opac, wl,
                                            spectrum_type = 'transmission')
            
            spectra_array.append(spectrum_no_clouds)

            if len(aerosol_species) > 1:
                label = '- Total Cloud'
            else:
                label = '- ' + aerosol_species[0]

            spectra_labels.append(label)

            if len(aerosol_species) > 1:

                if model['cloud_type'] == 'slab' or model['cloud_type'] == 'opaque_deck_plus_slab':

                    log_X_indices = np.where(np.char.find(model['cloud_param_names'], 'log_X')!= -1)[0]

                    for n in log_X_indices:

                        cloud_params = np.copy(cloud_params_og)

                        for m in log_X_indices:
                            if m == n:
                                cloud_params[m] = -50

                        model_clouds = define_model(model_name,bulk_species,param_species,
                                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                        aerosol_species = aerosol_species)

                        atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params)

                        spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                                        spectrum_type = 'transmission')
                        
                        spectra_array.append(spectrum_clouds)

                        label = '- ' + model['cloud_param_names'][n].split('_')[2]
                        spectra_labels.append(label)
                
                
                elif model['cloud_type'] == 'fuzzy_deck_plus_slab':

                    f_index = np.where(np.char.find(model['cloud_param_names'], 'f')!= -1)[0]
                    log_X_indices = np.where(np.char.find(model['cloud_param_names'], 'log_X')!= -1)[0]

                    for n in log_X_indices:

                        cloud_params = np.copy(cloud_params_og)

                        for m in log_X_indices:
                            if m == n:
                                cloud_params[m] = -50

                        model_clouds = define_model(model_name,bulk_species,param_species,
                                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                        aerosol_species = aerosol_species)

                        atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params)

                        spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                                        spectrum_type = 'transmission')
                        
                        spectra_array.append(spectrum_clouds)

                        label = '- ' + model['cloud_param_names'][n].split('_')[2]
                        spectra_labels.append(label)

                    # Turn only the fuzzy deck 
                    cloud_params = np.copy(cloud_params_og)
                    cloud_params[f_index] = 0.000001

                    model_clouds = define_model(model_name,bulk_species,param_species,
                                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                        aerosol_species = aerosol_species)
                    
                    atmosphere_clouds = make_atmosphere(planet, model_clouds, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params)

                    spectrum_clouds = compute_spectrum(planet, star, model_clouds, atmosphere_clouds, opac, wl,
                                                        spectrum_type = 'transmission')
                        
                    spectra_array.append(spectrum_clouds)
                    
                    label = '- '+ model['cloud_param_names'][f_index][0].split('_')[1]
                    spectra_labels.append(label)

    # For each chemical species in the model
    if show_chemical_species == True:

        if put_one_in == True:

            for i in range(len(species_list)):

                model_name = 'One-Species'
                param_species = [species_list[i]]

                model_one_species = define_model(model_name,bulk_species,param_species,
                                                PT_profile = model['PT_profile'], X_profile = model['X_profile'])

                log_X_params = np.array([log_X_params_og[0][i]])

                atmosphere_one_species = make_atmosphere(planet, model_one_species, P, P_ref, R_p_ref, PT_params_og, log_X_params)

                spectrum_one_species = compute_spectrum(planet, star, model_one_species, atmosphere_one_species, opac, wl,
                                                spectrum_type = 'transmission')

                spectra_array.append(spectrum_one_species)
                spectra_labels.append(species_list[i])
        
        else:

            for n in range(len(species_list)):

                log_X_params = np.copy(log_X_params_og)

                for m in range(len(log_X_params[0])):
                    if m == n:
                        log_X_params[0][m] = -50

                model_name = 'Minus-One-Species'

                if model['cloud_model'] != 'Mie':

                    model_one_species = define_model(model_name,bulk_species,param_species,
                                                    PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                                    cloud_model = model['cloud_model'], cloud_type = model['cloud_type'])
                    
                else:
                    aerosol_species = model['aerosol_species']

                    model_one_species = define_model(model_name,bulk_species,param_species,
                                PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                aerosol_species = aerosol_species)


                atmosphere_one_species = make_atmosphere(planet, model_one_species, P, P_ref, R_p_ref, PT_params_og, log_X_params, cloud_params_og)

                spectrum_one_species = compute_spectrum(planet, star, model_one_species, atmosphere_one_species, opac, wl,
                                                spectrum_type = 'transmission')

                spectra_array.append(spectrum_one_species)
                label = '- ' + species_list[n]
                spectra_labels.append(label)

    # Plot
    # Initialize plot collection with the full spectrum

    # To make sure the full spectrum is plotted on top of everything else
    spectra_array.append(spectrum_full)
    spectra_labels.append('Full Spectrum')

    spectra = []

    # Loop through the contribution spectra 
    for s in spectra_array:
        spectra = plot_collection(s, wl, collection = spectra)

    colour_list = colour_list[:len(spectra_array)-1]
    colour_list.append('black')
    
    fig = plot_spectra(spectra, planet, R_to_bin = 100,
                   plt_label = 'Cloud Contribution Plot',
                   spectra_labels = spectra_labels,
                   plot_full_res = False, 
                   save_fig = False,
                   colour_list = colour_list[:len(spectra_array)])

def plot_clouds(planet,model,atmosphere, colour_list = []):

    # Unpack model 'strings'
    planet_name = planet['planet_name']
    model_name = model['model_name']
    aerosol_species = model['aerosol_species']
    Atmosphere_dimension = model['Atmosphere_dimension']
    TwoD_type = model['TwoD_type']
    cloud_type = model['cloud_type']

    # Global parameters
    P = atmosphere['P']
    log_P = np.log10(P)
    r = atmosphere['r']
    n = atmosphere['n']

    # Aerosol parameters
    P_cloud = atmosphere['P_cloud']
    H = atmosphere['H']
    r_m = atmosphere['r_m']
    log_n_max = atmosphere['log_n_max']
    fractional_scale_height = atmosphere['fractional_scale_height']
    aerosol_species = np.copy(atmosphere['aerosol_species'])
    log_X_Mie = atmosphere['log_X_Mie']
    P_cloud_bottom = atmosphere['P_cloud_bottom']

    # Turn everything into arrays (only matters for file read and free)
    if isinstance(P_cloud, np.ndarray) == False:
        P_cloud = np.array([P_cloud])
    if isinstance(log_n_max, np.ndarray) == False:
        log_n_max = np.array([log_n_max])
    if isinstance(fractional_scale_height, np.ndarray) == False:
        fractional_scale_height = np.array([fractional_scale_height])
    if isinstance(log_X_Mie, np.ndarray) == False:
        log_X_Mie = np.array([log_X_Mie])
    if isinstance(P_cloud_bottom, np.ndarray) == False:
        P_cloud_bottom = np.array([P_cloud_bottom])

    if aerosol_species[0] == 'free':
        r_i_real = atmosphere['r_i_real']
        r_i_complex = atmosphere['r_i_complex']
        free_string = str(r_i_real) + " + " + str(r_i_complex) + " j"

    # Initialize plot
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    # Define colours for mixing ratio profiles (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['orange','royalblue', 'darkgreen', 'magenta', 'crimson', 'darkgrey', 
                   'black', 'darkorange', 'navy']
    else:
        colours = colour_list
    

    #########################
    # Calculate the number density above the cloud top or apply a uniform haze
    #########################

    # Go through all the different cloud models 
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
        n_aerosol[P_cloud_index:] = (10**log_n_max[0]) * np.exp(-h/(fractional_scale_height[0]*H[P_cloud_index:]))
        
        # Convert to mixing ratio 
        mixing_ratio = np.log10(n_aerosol.flatten()/n.flatten())

        if aerosol_species[0] == 'free':
            label = free_string
        else:
            label = aerosol_species[0]
    
        ax.plot(mixing_ratio[P_cloud_index:], log_P[P_cloud_index:], label = label, color = colours[0])
        ax.axhspan(log_P[P_cloud_index], np.log10(np.max(P)), alpha=0.5, color='gray', label = 'Opaque Cloud')

    # Slab Model 
    elif (cloud_type == 'slab'):

        # Loop through the aerosols, since you can have more than one slab 
        for q in range(len(aerosol_species)):
            # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
            n_aerosol = np.empty_like(r)
            P_cloud_index_top = find_nearest(P,P_cloud[q])
            P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[q])

            log_X = log_X_Mie[q]

            if aerosol_species[0] == 'free':
                label = free_string
            else:
                label = aerosol_species[q]

            plt.vlines(x = log_X, ymin = log_P[P_cloud_index_bttm], ymax = log_P[P_cloud_index_top], color = colours[q], linewidth=5.0)
            ax.axvline(x = log_X, color = colours[q], linewidth=1.0, linestyle = '--')
            ax.axhspan(log_P[P_cloud_index_top], log_P[P_cloud_index_bttm], alpha=0.5, color= colours[q], label = label)

    # Combined Models 
    elif (cloud_type == 'fuzzy_deck_plus_slab'):

        for q in range(len(aerosol_species)):

            if aerosol_species[0] == 'free':
                label = free_string
            else:
                label = aerosol_species[q]

                # The first index will be the fuzzy deck 
                if q == 0:
                    # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
                    n_aerosol = np.empty_like(r)
                    P_cloud_index = find_nearest(P,P_cloud[q])
                    # Find the radius corresponding to the cloud top pressure 
                    cloud_top_height = r[P_cloud_index]
                    # Height above cloud 
                    h = r[P_cloud_index:] - cloud_top_height
                    # Find number density below and above P_cloud
                    n_aerosol[:P_cloud_index] = 1.0e250
                    n_aerosol[P_cloud_index:] = (10**log_n_max[0]) * np.exp(-h/(fractional_scale_height[0]*H[P_cloud_index:]))
                    
                    # Convert to mixing ratio 
                    mixing_ratio = np.log10(n_aerosol.flatten()/n.flatten())

                    ax.plot(mixing_ratio[P_cloud_index:], log_P[P_cloud_index:], label = label, color = colours[0], linewidth = 5.0)
                    ax.axhspan(log_P[P_cloud_index], np.log10(np.max(P)), alpha=0.5, color='gray', label = 'Opaque Cloud')

                # Others will be slabs 
                else:
                    # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
                    n_aerosol = np.empty_like(r)
                    P_cloud_index_top = find_nearest(P,P_cloud[q])
                    P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[q-1])

                    log_X = log_X_Mie[q-1]
                    plt.vlines(x = log_X, ymin = log_P[P_cloud_index_bttm], ymax = log_P[P_cloud_index_top], color = colours[q], linewidth=5.0)
                    ax.axvline(x = log_X, color = colours[q], linewidth=1.0, linestyle = '--')
                    ax.axhspan(log_P[P_cloud_index_top], log_P[P_cloud_index_bttm], alpha=0.5, color= colours[q], label = label)


    # For the opaque deck, the opaque deck will be added to the aerosol array even though its not really an aerosol species 
    elif (cloud_type == 'opaque_deck_plus_slab'):

        try:

            if aerosol_species[0] == 'free':

                label = free_string

                # Deck First
                P_cloud_index = find_nearest(P,P_cloud[0]) # The deck top pressure is the first element in the P_cloud
                ax.axhspan(log_P[P_cloud_index], np.log10(np.max(P)), alpha=0.5, color='gray', label = 'Opaque Cloud')
                
                # Slab Second
                n_aerosol = np.empty_like(r)
                P_cloud_index_top = find_nearest(P,P_cloud[1])
                P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[0])

                log_X = log_X_Mie[0]
                plt.vlines(x = log_X, ymin = log_P[P_cloud_index_bttm], ymax = log_P[P_cloud_index_top], color = colours[0], linewidth=5.0)
                ax.axvline(x = log_X, color = colours[0], linewidth=1.0, linestyle = '--')
                ax.axhspan(log_P[P_cloud_index_top], log_P[P_cloud_index_bttm], alpha=0.5, color= colours[0], label = label)
            
            else:

                for q in range(len(r_m)):

                        if q ==0:
                            P_cloud_index = find_nearest(P,P_cloud[0]) # The deck top pressure is the first element in the P_cloud
                            ax.axhspan(log_P[P_cloud_index], np.log10(np.max(P)), alpha=0.5, color='gray', label = 'Opaque Cloud')
                        
                        # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
                        n_aerosol = np.empty_like(r)
                        P_cloud_index_top = find_nearest(P,P_cloud[q+1])
                        P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[q])

                        log_X = log_X_Mie[q]
                        plt.vlines(x = log_X, ymin = log_P[P_cloud_index_bttm], ymax = log_P[P_cloud_index_top], color = colours[q], linewidth=5.0)
                        ax.axvline(x = log_X, color = colours[q], linewidth=1.0, linestyle = '--')
                        ax.axhspan(log_P[P_cloud_index_top], log_P[P_cloud_index_bttm], alpha=0.5, color= colours[q], label = aerosol_species[q])

        # If its file read or free
        except:

            # Deck First
            P_cloud_index = find_nearest(P,P_cloud[0]) # The deck top pressure is the first element in the P_cloud
            ax.axhspan(log_P[P_cloud_index], np.log10(np.max(P)), alpha=0.5, color='gray', label = 'Opaque Cloud')
            
            # Slab Second
            n_aerosol = np.empty_like(r)
            P_cloud_index_top = find_nearest(P,P_cloud[1])
            P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[0])

            log_X = log_X_Mie[0]
            plt.vlines(x = log_X, ymin = log_P[P_cloud_index_bttm], ymax = log_P[P_cloud_index_top], color = colours[0], linewidth=5.0)
            ax.axvline(x = log_X, color = colours[0], linewidth=1.0, linestyle = '--')
            ax.axhspan(log_P[P_cloud_index_top], log_P[P_cloud_index_bttm], alpha=0.5, color= colours[0], label = aerosol_species[0])


    # Uniform X Model 
    else:
        for q in range(len(aerosol_species)):

            if aerosol_species[0] == 'free':
                label = free_string
            else:
                label = aerosol_species[q]

            log_X = log_X_Mie[q]
            ax.axvline(x = log_X, color = colours[q], linewidth=1.0, label = label)


    ax.invert_yaxis()
    ax.set_ylim(log_P[0], log_P[-1])  
    ax.set_xlim(-30, -1)  
    ax.set_xlabel('Mixing Ratios (log $X_i$)')
    ax.set_ylabel('Pressure (log P) (bar)')
    ax.legend()
    plt.show()

def vary_one_parameter(model, planet, star, param_name, vary_list,
                       wl, opac, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params_og):

    from POSEIDON.core import define_model
    from POSEIDON.core import make_atmosphere
    from POSEIDON.core import compute_spectrum
    from POSEIDON.visuals import plot_spectra
    from POSEIDON.utility import plot_collection

    spectra_array = []
    spectra_labels = []

    # Real spectrum 
    model_name = 'Vary-One-Thing'
    bulk_species = ['H2','He']
    species_list = model['param_species']
    param_species = species_list

    if model['cloud_model'] != 'Mie':

        model = define_model(model_name,bulk_species,param_species,
                                PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                cloud_model = model['cloud_model'], cloud_type = model['cloud_type'])

    else:
        aerosol_species = model['aerosol_species']

        model = define_model(model_name,bulk_species,param_species,
                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                        aerosol_species = aerosol_species)


    # Try to find the variable they want to vary 

    if param_name in model['PT_param_names']:

        index = np.argwhere(model['PT_param_names'] == param_name)[0][0]

        for i in range(len(vary_list)):

            PT_params = np.copy(PT_params_og)
            PT_params[index] = vary_list[i]
            
            atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, log_X_params_og, cloud_params_og)

            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                      spectrum_type = 'transmission')
            
            spectra_array.append(spectrum)
            label = param_name + ' = ' + str(vary_list[i])
            spectra_labels.append(label)

    elif param_name in model['X_param_names']:

        index = np.argwhere(model['X_param_names'] == param_name)[0][0]

        for i in range(len(vary_list)):

            log_X_params = np.copy(log_X_params_og)
            log_X_params[index] = vary_list[i]
            
            atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params_og, log_X_params, cloud_params_og)

            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                      spectrum_type = 'transmission')
            
            spectra_array.append(spectrum)
            label = param_name + ' = ' + str(vary_list[i])
            spectra_labels.append(label)

    elif param_name in model['cloud_param_names']:

        index = np.argwhere(model['cloud_param_names'] == param_name)[0][0]

        for i in range(len(vary_list)):

            cloud_params = np.copy(cloud_params_og)
            cloud_params[index] = vary_list[i]
            
            atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params_og, log_X_params_og, cloud_params)

            spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                      spectrum_type = 'transmission')
            
            spectra_array.append(spectrum)
            label = param_name + ' = ' + str(vary_list[i])
            spectra_labels.append(label)

    else:
        raise(Exception(param_name, ' is not in the param list. Check model[\'param_names\']'))


    # Plot
    for s in range(len(spectra_array)):
        if s == 0:
            spectra = plot_collection(spectra_array[s], wl, collection = [])
        else:
            spectra = plot_collection(spectra_array[s], wl, collection = spectra)

    label = 'Varying ' + param_name
    fig = plot_spectra(spectra, planet, R_to_bin = 100,
                   plt_label = label,
                   spectra_labels = spectra_labels,
                   plot_full_res = False,
                   save_fig = False)

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
    database = h5py.File(input_file_path + 'opacity/'  + grid + '_database_emission.hdf5', 'r')

    # Load the dimensions of the grid
    wl_grid = np.array(database['Info/Wavelength grid'])
    r_m_grid = np.array(database['Info/Particle Size grid'])

    # Find sizes of each dimension
    wl_num, r_m_num = len(wl_grid), len(r_m_grid)

    # Store number of chemical species
    N_species = len(aerosol_species)

    # Create array to store the log mixing ratios from the grid 
    sigma_Mie_grid, win = shared_memory_array(rank, comm, (N_species, 6, r_m_num, wl_num))
    
    # Only first core needs to load the aerosols into shared memory
    if (rank == 0):

        # Add each aerosol species to mixing ratio array
        for q, species in enumerate(aerosol_species):

            # Load grid for species q, then reshape into a 2D numpy array
            ext_array = np.array(database[species]['eff_ext'])
            ext_array = ext_array.reshape(r_m_num, wl_num)

            abs_array = np.array(database[species]['eff_abs'])
            abs_array = abs_array.reshape(r_m_num, wl_num)

            scat_array = np.array(database[species]['eff_scat'])
            scat_array = scat_array.reshape(r_m_num, wl_num)
            
            back_array = np.array(database[species]['eff_back'])
            back_array = back_array.reshape(r_m_num, wl_num)

            g_array = np.array(database[species]['eff_g'])
            g_array = g_array.reshape(r_m_num, wl_num)

            w_array = np.array(database[species]['eff_w'])
            w_array = w_array.reshape(r_m_num, wl_num)

            # Package grid for species q into combined array
            sigma_Mie_grid[q,0,:,:] = ext_array
            sigma_Mie_grid[q,1,:,:] = abs_array
            sigma_Mie_grid[q,2,:,:] = scat_array
            sigma_Mie_grid[q,3,:,:] = back_array
            sigma_Mie_grid[q,4,:,:] = g_array
            sigma_Mie_grid[q,5,:,:] = w_array

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
        grid_interp = RegularGridInterpolator(([0,1,2,3,4,5],r_m_grid, wl_grid), sigma_Mie_grid[q,:,:,:])
        
        return [grid_interp((0,r_m_array[q],wl)), grid_interp((1,r_m_array[q],wl)), grid_interp((2,r_m_array[q],wl)), 
                grid_interp((3,r_m_array[q],wl)), grid_interp((4,r_m_array[q],wl)), grid_interp((5,r_m_array[q],wl))]
    
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
            sigma_Mie_interp_dict[species] = {}
            sigma_Mie_interp_dict[species]['eff_ext'] = interpolate(species)[0]
            sigma_Mie_interp_dict[species]['eff_abs'] = interpolate(species)[1]
            sigma_Mie_interp_dict[species]['eff_scat'] = interpolate(species)[2]
            sigma_Mie_interp_dict[species]['eff_back'] = interpolate(species)[3]
            sigma_Mie_interp_dict[species]['eff_g'] = interpolate(species)[4]
            sigma_Mie_interp_dict[species]['eff_w'] = interpolate(species)[5]
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
              log_X_Mie = 0,
              P_cloud_bottom = 0):


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
            uniform_X, fuzzy_deck, slab, opaque_deck_with_slab, fuzzy_deck_with_slab

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

        Slab Arguments 

        P_cloud (float) : 
            Cloud Top Pressure (everything between P_cloud and P_cloud_bottom is uniform X). 

        P_cloud_bottom (array of float) : 
            Pressure of the bottom of the slab 

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
        # Calculate the number density above the cloud top, in the slab, or applies a uniform haze
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

        # Slab Model 
        elif (cloud_type == 'slab'):
            # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
            n_aerosol = np.empty_like(r)
            P_cloud_index_top = find_nearest(P,P_cloud[q])
            P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[q])

            n_aerosol = np.empty_like(r)
            n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = (n[P_cloud_index_bttm:P_cloud_index_top])*np.float_power(10,log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

        # Combined Models 
        elif (cloud_type == 'fuzzy_deck_plus_slab'):
            # The first index will be the fuzzy deck 
            if q == 0:
                n_aerosol = np.empty_like(r)
                P_cloud_index = find_nearest(P,P_cloud[q])
                cloud_top_height = r[P_cloud_index]
                h = r[P_cloud_index:] - cloud_top_height
                n_aerosol[:P_cloud_index] = 1.0e250
                n_aerosol[P_cloud_index:] = (10**log_n_max[q]) * np.exp(-h/(fractional_scale_height[q]*H[P_cloud_index:]))
                n_aerosol_array.append(n_aerosol)

            else:
                # Others will be slabs 
                n_aerosol = np.empty_like(r)
                P_cloud_index_top = find_nearest(P,P_cloud[q])
                P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[q-1]) #Because this is one shorter than the P_cloud array, decks don't have P_bottom
                n_aerosol = np.empty_like(r)
                n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = (n[P_cloud_index_bttm:P_cloud_index_top])*np.float_power(10,log_X_Mie[q-1]) # same reason
                n_aerosol_array.append(n_aerosol)

        # For the opaque deck, the opaque deck will be added to the aerosol array even though its not really an aerosol species 
        elif (cloud_type == 'opaque_deck_plus_slab'):
            
            if q == 0:
                n_aerosol = np.empty_like(r)
                P_cloud_index = find_nearest(P,P_cloud[0]) # The deck top pressure is the first element in the P_cloud
                n_aerosol[:P_cloud_index] = 1.0e250
                n_aerosol_array.append(n_aerosol)
                
            n_aerosol = np.empty_like(r)
            P_cloud_index_top = find_nearest(P,P_cloud[q+1]) # The slab top pressure are next after the deck 
            P_cloud_index_bttm = find_nearest(P,P_cloud_bottom[q]) # Doesn't change
            n_aerosol = np.empty_like(r)
            n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = (n[P_cloud_index_bttm:P_cloud_index_top])*np.float_power(10,log_X_Mie[q])
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

    sigma_Mie_interp_dict = interpolate_sigma_Mie_grid(aerosol_grid, wl, r_m, 
                               aerosol_species, return_dict = True)
    
    # To work with Numba
    sigma_ext_cld_array = []
    g_cld_array = []
    w_cld_array = []

    for aerosol in aerosol_species:

        sigma_ext = sigma_Mie_interp_dict[aerosol]['eff_ext']
        sigma_ext_cld_array.append(sigma_ext)

        eff_g = sigma_Mie_interp_dict[aerosol]['eff_g']
        g_cld_array.append(eff_g)

        eff_w = sigma_Mie_interp_dict[aerosol]['eff_w']
        w_cld_array.append(eff_w)


    return n_aerosol_array, sigma_ext_cld_array, g_cld_array, w_cld_array


######################################################
######################################################
#  Functions that don't use the aerosol database 
######################################################
######################################################

############################################################################################
# Empty Arrays for Qext calculations
# RYAN : Is there a better way to do this? 
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
wl_Mie_empty = np.array([])

# Free or file_read switch
# This is just a saved variable that acts like a kill switch if the model is 
# Switched between free and file_read in the same notebook 
free_or_file = ''


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

def Mie_cloud_free(P, wl, wl_Mie_in, r, H, n, r_m, r_i_real, r_i_complex, cloud_type, 
                   P_cloud = 0, log_n_max = 0, fractional_scale_height = 0,
                   log_X_Mie = 0, P_cloud_bottom = -100, r_m_std_dev = 0.5, z_max = 5,
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

        H (np.array of float) : 
            gas scale height

        n (np.array of float) :
            total number density array 

        r_m  (float) : 
            Mean particle sizes (in um)

        cloud_type (string):
            uniform_X, fuzzy_deck, slab, opaque_deck_with_slab, fuzzy_deck_with_slab




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

        Slab Arguments 

        P_cloud (float) : 
            Cloud Top Pressure (everything between P_cloud and P_cloud_bottom is uniform X). 

        P_cloud_bottom (array of float) : 
            Pressure of the bottom of the slab 

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
    global all_etas, all_xs, all_Qexts, all_Qscats, all_Qbacks, all_gs, wl_Mie_empty, free_or_file


    #########################
    # Set up wl_mie (a wl array with R = 1000). This is only for aerosol = 'free' or 'file'
    #########################

    wl_min = wl[0]
    wl_max = wl[-1]

    # Initialize wl_Mie
    if len(wl_Mie_empty) == 0:
        wl_Mie = np.append(wl_Mie_empty, wl_Mie_in)

    # If its a new wl array 
    if  wl[0] != wl_min or wl[-1] != wl_max:
        wl_min = wl[0]
        wl_max = wl[-1]
        wl_Mie = []
        wl_Mie = np.append(wl_Mie, wl_Mie_in)

    #########################
    # Calculate the number density above the cloud top or apply a uniform haze
    #########################
    
    # Fuzzy Deck Model 
    if cloud_type == 'fuzzy_deck':
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

    # Slab Model 
    # If P_cloud is a float its just the slab, without a deck
    elif cloud_type == 'slab':
        # r is a 3d array that follows (N_layers, terminator plane sections, day-night sections)
        n_aerosol = np.empty_like(r)
        P_cloud_index_top = find_nearest(P,P_cloud)
        P_cloud_index_bttm = find_nearest(P,P_cloud_bottom)

        n_aerosol = np.zeros_like(r)
        n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = (n[P_cloud_index_bttm:P_cloud_index_top])*np.float_power(10,log_X_Mie)

    # Opaque Deck + Slabs Model 
    # In this model, the P_cloud has the deck pressure and the slab pressure. For the others, its a int 
    elif cloud_type == 'opaque_deck_plus_slab':
        # Deck First 
        n_aerosol = np.empty_like(r)
        P_cloud_index = find_nearest(P,P_cloud[0])
        n_aerosol[:P_cloud_index] = 1.0e250

        # Slab Next
        P_cloud_index_top = find_nearest(P,P_cloud[1])
        P_cloud_index_bttm = find_nearest(P,P_cloud_bottom)
        n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = (n[P_cloud_index_bttm:P_cloud_index_top])*np.float_power(10,log_X_Mie)


    # Uniform X
    elif cloud_type == 'uniform_X':
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
            all_etas = []
            all_xs = [] 
            all_Qexts = []
            all_Qscats = []
            all_Qbacks = []
            all_gs = []
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
            all_Qscats = []
            all_Qbacks = []
            all_gs = []
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

    # We redefine n and eff_cross_section to be more in line with Poseidon's exisiting language
    sigma_ext = eff_ext

    # To work with Numba
    n_aerosol_array = []
    n_aerosol_array.append(n_aerosol)

    sigma_ext_cld_array = []
    sigma_ext_cld_array.append(sigma_ext)

    g_cld_array = []
    g_cld_array.append(eff_g)

    w_cld_array = []
    w_cld_array.append(eff_w)


    return n_aerosol_array, sigma_ext_cld_array, g_cld_array, w_cld_array


######################################################
######################################################
#  Functions that can add molecules to the aerosol database
######################################################
######################################################

############################################################################################
# Main DataBase Functions
############################################################################################

# This will can be used to make new cross sections from lab data to add to database
def precompute_cross_sections_one_aerosol(file_name, aerosol_name):

    '''
    Calculates the .npy file from a refractive index txt file (lab data)
    Takes ~ a day to generate the npy file that then can be easily added to the aerosol database  
    Please ensure that the first two rows are skippable, and that the columns are 
    Wavelength (microns) | Real Index | Imaginary Index
    Please also ensure that the wavelengths are at LEAST from 0.2 to 30 um. If not, add 0s into your txt files 

    INPUTS 

    file_name (txt):
        file name of the txt file with the directory included

    aerosoL_name (txt):
        name that you want the npy file saved with
    '''

    global all_etas, all_xs, all_Qexts, all_Qscats, all_Qbacks, all_gs

    # Constants for the Qext Calculation
    r_m_std_dev = 0.5
    z_max = 5
    num_integral_points = 100
    R_Mie = 1000

    # Saved Arrays 
    sigma_Mie_all = []
    wl_Mie = []
    ext_array = []
    scat_array = []
    abs_array = []
    back_array = []
    w_array = []
    g_array = []
    jumbo_array = []

    # Reset the saved arrays 
    all_etas = []
    all_xs = []
    all_Qexts = []
    all_Qscats = []
    all_Qbacks = []
    all_gs = []

    wl_min = 0.2
    wl_max = 30
    wl_Mie = np.append(wl_Mie,wl_grid_constant_R(wl_min, wl_max, R_Mie))

    # Default indices for the cross section, g, and w arrays
    # This only changes if the lab data doesn't span the entire wl range 
    idx_start = 0
    idx_end = 5011

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
        file_name = file_name
        print('Loading in : ', file_name)
        try:
            file_as_numpy = np.loadtxt(file_name, comments = '#').T
        except:
            file_as_numpy = np.loadtxt(file_name, skiprows = 2).T

        # If its index, wavelength, n, k we need to do something different. 
        if len(file_as_numpy) == 4:
            wavelengths = file_as_numpy[1]
            real_indices = file_as_numpy[2]
            imaginary_indices = file_as_numpy[3]
            file_as_numpy = np.array([wavelengths,real_indices,imaginary_indices])

    except :
        raise Exception('Could not load in file. Make sure directory is included in the input')


    wavelengths = file_as_numpy[0]

    # Truncating wl grid if necessary 
    # Any values not covered will be set to 0 in the database
    if np.max(wavelengths) < 30 or np.min(wavelengths) > 0.2:

        print('Wavelength column does not span 0.2 to 30 um')

        # If less than 30 and greater than 0.2
        if np.max(wavelengths) < 30 and np.min(wavelengths) > 0.2:

            wl_min = np.min(wavelengths)
            wl_max = np.max(wavelengths)

            idx_start = find_nearest(wl_Mie,wl_min) + 1
            idx_end = find_nearest(wl_Mie, wl_max)

            # Find nearest pulls the closest value below the given value, so we go up one index
            wl_Mie = wl_Mie[idx_start+1:idx_end]

            print('Wavelength grid will be truncated to : ' + str(np.min(wl_Mie)) + ' to '+  str(np.max(wl_Mie)))

        # If less than 30 only
        elif np.max(wavelengths) < 30 and np.min(wavelengths) <= 0.2:

            wl_min = 0.2
            wl_max = np.max(wavelengths)

            idx_start = 0
            idx_end = find_nearest(wl_Mie, wl_max)

            wl_Mie = wl_Mie[:idx_end]
            
            print('Wavelength grid will be truncated to : 0.2 to ' + str(np.max(wl_Mie)))

        # If more than 0.2 only 
        elif np.max(wavelengths) >= 30 and np.min(wavelengths) > 0.2:

            wl_min = np.min(wavelengths)
            wl_max = 30

            idx_start = find_nearest(wl_Mie,wl_min) + 1
            idx_end = 5011

            # Find nearest pulls the closest value below the given value, so we go up one index
            wl_Mie = wl_Mie[idx_start:]

            print('Wavelength grid will be truncated to : ' + str(np.min(wl_Mie)) + ' to 30')

    # Loading in the refractive indices 
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
            all_Qscats = []
            all_Qbacks = []
            all_gs = []

        z = -np.logspace(np.log10(0.1), np.log10(z_max), int(num_integral_points/2)) 
        z = np.append(z[::-1], -z)

        probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
        radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
        geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric

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

            # SSA and weighted g 
            w_hist = Qscat_hist/Qext_hist
            #g_hist = g_hist/Qscat_hist

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

        # Empty arrays to store the following values into 
        eff_ext_cross_section = np.full(5011, 1e-250)
        eff_scat_cross_section = np.full(5011, 1e-250)
        eff_abs_cross_section = np.full(5011, 1e-250)
        eff_back_cross_section = np.full(5011, 1e-250)
        eff_w = np.full(5011, 1e-250)
        eff_g = np.full(5011, 1e-250)

        # Effective Cross section is a trapezoidal integral
        eff_ext_cross_section[idx_start:idx_end] = np.trapz(probs*geometric_cross_sections*Qext_intpl, z)

        # Scattering Cross section 
        eff_scat_cross_section[idx_start:idx_end] = np.trapz(probs*geometric_cross_sections*Qscat_intpl, z)

        # Absorption Cross section
        eff_abs_cross_section[idx_start:idx_end] = eff_ext_cross_section[idx_start:idx_end] - eff_scat_cross_section[idx_start:idx_end]

        # BackScatter Cross section 
        eff_back_cross_section[idx_start:idx_end] = np.trapz(probs*geometric_cross_sections*Qback_intpl, z)

        # Effective w and g
        eff_w[idx_start:idx_end] = np.median(w_intpl, axis=1)
        eff_g[idx_start:idx_end] = np.median(g_intpl, axis=1)

        # Append everything to arrays to save
        ext_array.append(eff_ext_cross_section)
        scat_array.append(eff_scat_cross_section)
        abs_array.append(eff_abs_cross_section)
        back_array.append(eff_back_cross_section)
        w_array.append(eff_w)
        g_array.append(eff_g)

        counter += 1

    title = input_file_path + 'opacity/refractive_indices/eff_ext_Mie_' + aerosol_name
    np.save(title,ext_array,allow_pickle = True)

    title = input_file_path + 'opacity/refractive_indices/eff_scat_Mie_' + aerosol_name
    np.save(title,scat_array,allow_pickle = True)

    title = input_file_path + 'opacity/refractive_indices/eff_abs_Mie_' + aerosol_name
    np.save(title,abs_array,allow_pickle = True)

    title = input_file_path + 'opacity/refractive_indices/eff_back_Mie_' + aerosol_name
    np.save(title,back_array,allow_pickle = True)

    title = input_file_path + 'opacity/refractive_indices/eff_w_Mie_' + aerosol_name
    np.save(title,w_array,allow_pickle = True)

    title = input_file_path + 'opacity/refractive_indices/eff_g_Mie_' + aerosol_name
    np.save(title,g_array,allow_pickle = True)

    title = input_file_path + 'opacity/refractive_indices/jumbo_Mie_' + aerosol_name
    jumbo_array.append([ext_array,scat_array,abs_array,back_array,w_array,g_array])
    np.save(title,jumbo_array,allow_pickle = True)

    all_etas = []
    all_xs = []
    all_Qexts = []
    all_Qscats = []
    all_Qbacks = []
    all_gs = []
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
        
    global all_etas, all_xs, all_Qexts, all_Qscats, all_Qbacks, all_gs

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

    z = -np.logspace(np.log10(0.1), np.log10(z_max), int(num_integral_points/2)) 
    z = np.append(z[::-1], -z)

    probs = np.exp(-z**2/2) * (1/np.sqrt(2*np.pi))
    radii = r_m * np.exp(z * r_m_std_dev) # This takes the place of rm * exp(sigma z)
    geometric_cross_sections = np.pi * (radii*1e-6)**2 # Needs to be in um since its geometric

    Qext_intpl_array = []
    Qscat_intpl_array = []
    Qback_intpl_array = []
    w_intpl_array = []
    g_intpl_array = []

    # Reset the saved arrays 
    all_etas = []
    all_xs = []
    all_Qexts = []
    all_Qscats = []
    all_Qbacks = []
    all_gs = []

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
        Qext_hist, Qscat_hist, Qback_hist, g_hist = get_and_update(eta, x_hist) 

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
    Qext_intpl = np.reshape(Qext_intpl_array, (len(wavelengths), len(radii)))
    Qscat_intpl = np.reshape(Qscat_intpl_array, (len(wavelengths), len(radii)))
    Qback_intpl = np.reshape(Qback_intpl_array, (len(wavelengths), len(radii)))
    w_intpl = np.reshape(w_intpl_array, (len(wavelengths), len(radii)))
    g_intpl = np.reshape(g_intpl_array, (len(wavelengths), len(radii)))

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

    all_etas = []
    all_xs = []
    all_Qexts = []
    all_Qscats = []
    all_Qbacks = []
    all_gs = []

    return eff_ext_cross_section, eff_scat_cross_section, eff_abs_cross_section, eff_back_cross_section, eff_w, eff_g

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
    mydir = input_file_path + "opacity/refractive_indices/"
    file_list = glob.glob(mydir + "jumbo*.npy")
    file_list.sort()

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
        jumbo = np.load(title,allow_pickle = True)

        try:
            eff_ext = jumbo[0]
            eff_scat = jumbo[1]
            eff_abs = jumbo[2]
            eff_back = jumbo[3]
            eff_w = jumbo[4]
            eff_g = jumbo[5]
        except:
            eff_ext = jumbo[0][0]
            eff_scat = jumbo[0][1]
            eff_abs = jumbo[0][2]
            eff_back = jumbo[0][3]
            eff_w = jumbo[0][4]
            eff_g = jumbo[0][5]

        aerosols_dict[aerosol_list[i] + '_ext'] = eff_ext 
        aerosols_dict[aerosol_list[i] + '_abs'] = eff_abs
        aerosols_dict[aerosol_list[i] + '_scat'] = eff_scat
        aerosols_dict[aerosol_list[i] + '_back'] = eff_back 
        aerosols_dict[aerosol_list[i] + '_g'] = eff_g 
        aerosols_dict[aerosol_list[i] + '_w'] = eff_w 

    # Initialize and generate new data_base 
    database = h5py.File(input_file_path + 'opacity/aerosol_database_emission.hdf5', 'w')

    h = database.create_group('Info')
    h1 = h.create_dataset('Wavelength grid', data=wavelengths, compression='gzip', dtype='float64', shuffle=True)
    h2 = h.create_dataset('Particle Size grid', data=r_m_array, compression='gzip', dtype='float64', shuffle=True)

    h1.attrs["Variable"] = "wl"
    h2.attrs["Variable"] = "r_m"

    h1.attrs["Units"] = "um"
    h2.attrs["Units"] = "um"

    for i in range(len(aerosol_list)):

        # Print the name to show to user which one is being added 
        print(aerosol_list[i])

        g = database.create_group(aerosol_list[i])
        g1 = g.create_dataset('eff_ext', data=aerosols_dict[aerosol_list[i] + '_ext'], compression='gzip', dtype='float32', shuffle=True)
        g1.attrs["Varaible"] = "Effective Extinction Cross Section"
        g1.attrs["Units"] = "um^2"

        g2 = g.create_dataset('eff_abs', data=aerosols_dict[aerosol_list[i] + '_abs'], compression='gzip', dtype='float32', shuffle=True)
        g2.attrs["Varaible"] = "Effective Absorption Cross Section"
        g2.attrs["Units"] = "um^2"

        g3 = g.create_dataset('eff_scat', data=aerosols_dict[aerosol_list[i] + '_scat'], compression='gzip', dtype='float32', shuffle=True)
        g3.attrs["Varaible"] = "Effective Scattering Cross Section"
        g3.attrs["Units"] = "um^2"

        g4 = g.create_dataset('eff_back', data=aerosols_dict[aerosol_list[i] + '_back'], compression='gzip', dtype='float32', shuffle=True)
        g4.attrs["Varaible"] = "Effective Back Scattering Cross Section"
        g4.attrs["Units"] = "um^2"

        g5 = g.create_dataset('eff_g', data=aerosols_dict[aerosol_list[i] + '_g'], compression='gzip', dtype='float32', shuffle=True)
        g5.attrs["Varaible"] = "Effective Asymmetry Parameter"
        g5.attrs["Units"] = ""

        g6 = g.create_dataset('eff_w', data=aerosols_dict[aerosol_list[i] + '_w'], compression='gzip', dtype='float32', shuffle=True)
        g6.attrs["Varaible"] = "Effective Single Scattering Albedo"
        g6.attrs["Units"] = ""

    print('---------------------')
    print('Saving new aerosol database as')
    print(input_file_path + 'opacity/aerosol_database_emission.hdf5')
    print('---------------------')

    database.close()