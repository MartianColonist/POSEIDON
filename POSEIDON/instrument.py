# Computes instrument data points from model spectrum

import os.path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d as gauss_conv
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline as interp


#from config import planet_name, Band, App_mag, T_s, log_g_s, Met_s


def fwhm_instrument(wl_data, instrument):
    ''' 
    Evaluates full width at half maximum (FWHM) for the Point Spread 
    Function (PSF) of a given instrument mode at each bin centre point.
        
    FWHM (micron) = R_native * wl (micron)
    
    [Assumes Gaussian PSF with FWHM = instrument native spectral resolution]

    '''
        
    N_bins = len(wl_data)

    # Load reference data directory containing instrument properties
    inst_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
                               '.', 'reference_data/instruments/'))
    
    # For the below instrument modes, FWHM assumed constant as function of wavelength
    if   (instrument == 'STIS_G430'):   
        fwhm = 0.0004095 * np.ones(N_bins)  # HST STIS
    elif (instrument == 'STIS_G750'):   
        fwhm = 0.0007380 * np.ones(N_bins)  # HST STIS
    elif (instrument == 'WFC3_G280'):
        fwhm = 0.0057143 * np.ones(N_bins)  # HST WFC3
    elif (instrument == 'WFC3_G102'):
        fwhm = 0.0056350 * np.ones(N_bins)  # HST WFC3
    elif (instrument == 'WFC3_G141'):
        fwhm = 0.0106950 * np.ones(N_bins)  # HST WFC3
    elif (instrument == 'LDSS3_VPH_R'):
        fwhm = 0.0011750 * np.ones(N_bins)  # Magellan LDSS3
    
    # For JWST, we need to be a little more precise
    elif(instrument.startswith('JWST')):    

        # Find precomputed instrument spectral resolution file
        res_file = inst_dir + '/JWST/' + instrument + '_resolution.dat'
        
        # Check that file exists
        if (os.path.isfile(res_file) == False):
            print("Error! Cannot find resolution file for: " + instrument)
            raise SystemExit
            
        # Read instrument resolution file
        resolution = pd.read_csv(res_file, sep=' ', header=None)
        wl_res = np.array(resolution[0])   # (um)
        R_inst = np.array(resolution[1])   # Spectral resolution (R = wl/d_wl)
        
        # Interpolate resolution to bin centre location of each data point
        R_interp = interp(wl_res, R_inst, ext = 'extrapolate')  
        R_bin = np.array(R_interp(wl_data))
        
        # Evaluate FWHM of PSF for each data point
        fwhm = wl_data / R_bin  # (um)
    
    # For any other instruments without a known PSF, convolve with a dummy sharp PSF
    else: 
        fwhm = 0.0010000 * np.ones(N_bins) 
    
    return fwhm


def init_instrument(wl, wl_data, half_width, instrument):
    ''' 
    Initialises properties of a specified instrument. This function:
        
    1) Reads in instrument sensitivity functions.
    2) Reads in FWHM of instrument PSF.
    3) Finds indices on model wl grid closest to each data point (bin centre)
        and corresponding left/right bin edges.
    4) Pre-computes the integral of the sensitivity function over each bin
        (normalisation factor)

    These values are then stored for later usage and returned to the main
    program.
    
    '''

    # Load reference data directory containing instrument properties
    inst_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
                               '.', 'reference_data/instruments/'))
    
    # Identify instrument sensitivity function for desired instrument mode
    if (instrument == 'STIS_G430'):
        sens_file = inst_dir + '/STIS/G430L_sensitivity.dat'
    elif (instrument == 'STIS_G750'):
        sens_file = inst_dir + '/STIS/G750L_sensitivity.dat'
    elif (instrument == 'WFC3_G280'):
        sens_file = inst_dir + '/WFC3/G280_sensitivity.dat'
    elif (instrument == 'WFC3_G102'):
        sens_file = inst_dir + '/WFC3/G102_sensitivity.dat'
    elif (instrument == 'WFC3_G141'):
        sens_file = inst_dir + '/WFC3/G141_sensitivity.dat'
    elif (instrument == 'IRAC1'):
        sens_file = inst_dir + '/Spitzer/IRAC1_sensitivity.dat'
    elif (instrument == 'IRAC2'): 
        sens_file = inst_dir + '/Spitzer/IRAC2_sensitivity.dat'
    elif (instrument.startswith('JWST')): 
        sens_file = inst_dir + '/JWST/' + instrument + '_sensitivity.dat'
    
    # If instrument does not have a known sensitivity function, just use a top hat
    else: 
        sens_file = inst_dir + 'dummy_instrument_sensitivity.dat'
    
    # Verify that sensitivity file exists
    if (os.path.isfile(sens_file) == False):
        raise Exception("Error! Cannot find sensitivity file for: " + instrument)
    
    # Flag photometric bands for different treatment (no convolution with PSF)
    if (instrument in ['IRAC1', 'IRAC2']): 
        photometric = True
    else: 
        photometric = False
         
    # Read instrument sensitivity function
    transmission = pd.read_csv(sens_file, sep=' ', header=None)
    wl_trans = np.array(transmission[0])
    trans = np.array(transmission[1])

    # Transmission function evaluated at model grid locations
    sensitivity = np.zeros(len(wl))   
    
    # Interpolate instrument transmission function to model grid
    sens = interp(wl_trans, trans, ext='zeros')    
    sensitivity = sens(wl)
    sensitivity[sensitivity < 0.0] = 0.0
    
    # Compute FWHM of instrument PSF at each data point location
    fwhm = fwhm_instrument(wl_data, instrument)
    
    # PSF standard deviation (um)
    sigma_um = 0.424661*fwhm    
    
    if (photometric == False):
        
        N_bins = len(wl_data)
        
        # Compute closest indicies on model grid corresponding to bin edges and centre
        bin_left = np.zeros(N_bins).astype(np.int64)     # Closest index on model grid of left edge of each bin
        bin_cent = np.zeros(N_bins).astype(np.int64)     # Closest index on model grid of centre of each bin
        bin_right = np.zeros(N_bins).astype(np.int64)    # Closest index on model grid of right edge of each bin
        
        sigma = np.zeros(shape=(N_bins))   # Standard deviation of PSF in grid spaces
        norm = np.zeros(shape=(N_bins))    # Normalisation factor for instrument function integration
    
        # For each data point
        for n in range(N_bins):
            
            # Compute closest indicies on model grid corresponding to bin edges and centre
            bin_left[n] = np.argmin(np.abs(wl - ((wl_data[n] - half_width[n])))) 
            bin_cent[n] = np.argmin(np.abs(wl - (wl_data[n]))) 
            bin_right[n] = np.argmin(np.abs(wl - ((wl_data[n] + half_width[n]))))
            
            # Compute standard deviation of instrument PSF in grid spaces at each bin location (approx)
            dwl = 0.5 * (wl[bin_cent[n]+1] - wl[bin_cent[n]-1])
            sigma[n] = sigma_um[n]/dwl   
            
            # Compute normalisation of sensitivity function for each wl bin      
            norm[n] = trapz(sensitivity[bin_left[n]:bin_right[n]], wl[bin_left[n]:bin_right[n]])  
        
    elif (photometric == True):
        
        sigma = sigma_um   # Dummy value for return (equal to 0.0)
        
        # Compute closest indicies on model grid corresponding to bin edges and centre
        bin_left = np.argmin(np.abs(wl - ((wl_data - half_width)))) 
        bin_cent = np.argmin(np.abs(wl - (wl_data))) 
        bin_right = np.argmin(np.abs(wl - ((wl_data + half_width))))
        
        # Convert to numpy arrays
        bin_left = np.array([bin_left]).astype(np.int64)    # Closest index on model grid of left edge of each bin
        bin_cent = np.array([bin_cent]).astype(np.int64)    # Closest index on model grid of centre of each bin
        bin_right = np.array([bin_right]).astype(np.int64)  # Closest index on model grid of right edge of each bin
        
        # Compute normalisation of sensitivity function for each wl bin      
        norm = trapz(sensitivity[bin_left[0]:bin_right[0]], wl[bin_left[0]:bin_right[0]])
        norm = np.array([norm])
    
    return sigma, sensitivity, bin_left, bin_cent, bin_right, norm
    

def make_model_data(spectrum, wl, sigma, sensitivity, bin_left, bin_cent, 
                    bin_right, norm, photometric = False):                    
    ''' 
    Produces binned model points at resolution of the data. This function:
    
    1) Convolves spectrum with instrument PSF at location of each data point.
    2) Integrates convolved spectrum over instrument sensitivity function.
    3) Normalises by integral over sensitivity function to produce binned model points.

    For photometric bands, step (1) is not necessary.
    
    '''
    
    if (photometric == False):
        
        N_bins = len(bin_cent)
        data = np.zeros(shape=(N_bins))
        ymodel = np.zeros(shape=(N_bins))
        
        for n in range(N_bins):
            
            # Extend convolution beyond bin edge by max(1, 2 PSF std) model grid spaces (std rounded to integer)
            extention = max(1, int(2 * sigma[n]))   
            
            # Convolve spectrum with PSF width appropriate for a given bin 
            spectrum_conv = gauss_conv(spectrum[(bin_left[n]-extention):(bin_right[n]+extention)], 
                                       sigma=sigma[n], mode='nearest')
            integrand = spectrum_conv[extention:-extention] * sensitivity[bin_left[n]:bin_right[n]]
        
            # Integrate convolved spectrum over instrument sensitivity function
            data[n] = trapz(integrand, wl[bin_left[n]:bin_right[n]])   
            ymodel[n] = data[n]/norm[n]
        
        return ymodel
        
    elif (photometric == True):
        
        integrand = spectrum[bin_left[0]:bin_right[0]]*sensitivity[bin_left[0]:bin_right[0]]
        
        # Integrate spectrum over instrument sensitivity function
        data = trapz(integrand, wl[bin_left[0]:bin_right[0]])
        ymodel = data/norm
        
        return ymodel 
    

def bin_spectrum_to_data(spectrum, wl, data_properties):
    '''
    ADD DOCSTRING
    '''

    # Initialise combined array of binned model points (all instruments)
    ymodel = np.array([])
                
    # Generate binned model points for each instrument
    for i in range(len(data_properties['datasets'])):
            
        if (data_properties['instruments'][i] in ['IRAC1', 'IRAC2']): 
            photometric = True
        else: 
            photometric = False
            
        # Find start and end indicies of dataset_i in dataset property arrays
        idx_1 = data_properties['len_data_idx'][i]
        idx_2 = data_properties['len_data_idx'][i+1]
        
        # Compute binned transit depths for dataset_i
        ymodel_i = make_model_data(spectrum, wl, data_properties['psf_sigma'][idx_1:idx_2], 
                                   data_properties['sens'][i*len(wl):(i+1)*len(wl)], 
                                   data_properties['bin_left'][idx_1:idx_2], 
                                   data_properties['bin_cent'][idx_1:idx_2], 
                                   data_properties['bin_right'][idx_1:idx_2],
                                   data_properties['norm'][idx_1:idx_2], photometric)
                                                
        # Combine binned model points for each instrument
        ymodel = np.concatenate([ymodel, ymodel_i])    
            
    return ymodel


def R_to_wl(R_data, wl_data_min, wl_data_max):
    ''' 
    Convert a given spectral resolution to a set of wavelength data points 
    and bin half-widths.
    
    '''
    
    delta_log_wl = 1.0/R_data
    N_data = (np.log(wl_data_max) - np.log(wl_data_min)) / delta_log_wl
    N_data = np.around(N_data).astype(np.int64)
    
    log_wl_data = np.linspace(np.log(wl_data_min), np.log(wl_data_max), N_data)    
    wl_data = np.exp(log_wl_data)
    
    half_width_data = np.zeros(shape=(N_data))
    
    for n in range(N_data): 
        
        if (n==0): 
            half_width_data[n] = 0.5 * (wl_data[1] - wl_data[0])
        else: 
            half_width_data[n] = 0.5 * (wl_data[n] - wl_data[n-1])
        
    return wl_data, half_width_data, N_data


# TBD: update functions below

'''
def generate_syn_data(wl_model, spectrum, wl_start, wl_end, R_data,
                      err_data, instruments, label, Gauss_scatter = False):
    ''''''
    Generates a synthetic dataset in one or more wavelength regions. 
    
    Inputs:
    
    wl_model => model wavelength grid (um)
    spectrum => model spectrum
    wl_start => array of left edges for desired data ranges (um)
    wl_end => array of right edges for desired data ranges (um)
    R_data => array of spectral resolutions for desired datasets
    err_data => array of 1-sigma transit depth precisions for desired datasets (ppm)
    instruments => array of instrument names for each dataset (use 'None' for a flat sensitivity function)
        
    ''''''
           
    print("Creating synthetic data")
    
    # Initialise output arrays 
    wl_data_ALL = np.array([])
    half_width_data_ALL = np.array([])
    syn_data_ALL = np.array([])
    syn_err_ALL = np.array([])
    syn_ymodel_ALL = np.array([])
    
    # Generate synthetic data for each specified instrument
    for i in range(len(instruments)):
        
        print(instruments[i])
        
        if (instruments[i] in ['IRAC1', 'IRAC2']): 
            is_photometric = True
        else: 
            is_photometric = False
        
        # For a given spectral resolution, compute wavelength locations of each data point and half-bin width 
        wl_data, half_width_data, N_data = R_to_wl(R_data[i], wl_start[i], wl_end[i])
  
        N_data = len(wl_data)
        
        # Initialise the instrument properties for the synthetic dataset 
        sigma, sens, bin_left, \
        bin_cent, bin_right, norm = init_instrument(wl_model, wl_data, half_width_data, instruments[i])
        
        # Compute synthetic binned model points
        syn_ymodel = make_model_data(spectrum, wl_model, sigma, sens, bin_left, 
                                     bin_cent, bin_right, half_width_data, norm, is_photometric)
        
        # Arrays containing synthetic data and 1-sigma errors
        syn_data = np.zeros(shape=(N_data))
        syn_err = np.zeros(shape=(N_data))   
        
        # Open output file where synthetic data will be written
        if (instruments[i] != 'None'):
            f = open('../../observations/' + planet_name + '/' + planet_name +
                     '_SYNTHETIC_' + label + '_' + instruments[i] + '.dat','w')
        else:
            f = open('../../observations/' + planet_name + '/' + planet_name +
                     '_SYNTHETIC_' + label + '.dat','w')
        
        # Add Gaussian errors to binned points to produce synthetic data set
        for j in range(N_data):
            
            syn_err[j] = err_data[i]*1.0e-6   # Transit depth error for this synthetic dataset (ppm -> transit depth)
            
            if (Gauss_scatter == True):   
                err = np.random.normal(0.0, syn_err[j])   # Random draw with std = syn_err
                syn_data[j] = syn_ymodel[j] + err
            else:
                syn_data[j] = syn_ymodel[j] 
        
            # Write this datapoint to file
            f.write('%f %f %e %e \n' %(wl_data[j], half_width_data[j], syn_data[j], syn_err[j]))
            
        f.close()
        
        # Combine arrays from this dataset with previous synthetic datasets in loop
        wl_data_ALL = np.concatenate([wl_data_ALL, wl_data])
        half_width_data_ALL = np.concatenate([half_width_data_ALL, half_width_data])  
        syn_data_ALL = np.concatenate([syn_data_ALL, syn_data])
        syn_err_ALL = np.concatenate([syn_err_ALL, syn_err])
        syn_ymodel_ALL = np.concatenate([syn_ymodel_ALL, syn_ymodel])

    return syn_data_ALL, syn_err_ALL, wl_data_ALL, half_width_data_ALL, syn_ymodel


def generate_syn_data_from_file(wl_model, spectrum, instruments, err_files, 
                                label, Gauss_scatter = False):

    '''''' Generates a synthetic dataset in one or more wavelength regions. 
    
        Inputs:
        
        wl_model => model wavelength grid (m)
        spectrum => model spectrum
        err_files => file containing data wavelengths and error bars
        instruments => array of instrument names for each dataset (use 'None' for a flat sensitivity function)
          
    ''''''
           
    print("Creating synthetic data")
    
    is_photometric = []
    
    wl_data_ALL = np.array([])
    half_width_data_ALL = np.array([])
    syn_data_ALL = np.array([])
    syn_err_ALL = np.array([])
    syn_ymodel_ALL = np.array([])
    
    data_dir = '../../observations/' + planet_name + '/'

    for i in range(len(instruments)):
        
        print(instruments[i])
        
        if (instruments[i] in ['IRAC1', 'IRAC2']): is_photometric.append(True)
        else: is_photometric.append(False)
        
        # Read error file
        data = pd.read_csv((data_dir + err_files[i]), sep = ' ', header=None)

        wl_data = np.array(data).T[0]             # Wavelengths of bin centres (um)
        half_width_data = np.array(data).T[1]     # Bin half widths (um)
        err_data = np.array(data).T[3]            # Pre-simulated error bars
        
        N_data = len(wl_data)
        
        # Initialise the instrument properties for the synthetic dataset 
        sigma, sens, bin_left, bin_cent, bin_right, norm = init_instrument(wl_model, wl_data, half_width_data, instruments[i])
        
        # Compute synthetic binned model points
        syn_ymodel = make_model_data(spectrum, wl_model, sigma, sens, bin_left, bin_cent, bin_right, half_width_data, norm, is_photometric[i])
        
        # Arrays containing synthetic data and 1-sigma errors
        syn_data = np.zeros(shape=(N_data))
        syn_err = np.zeros(shape=(N_data))   
                
        # Open output file where synthetic data will be written
        if (instruments[i] != 'None'):
            f = open('../../observations/' + planet_name + '/' + planet_name +
                     '_SYNTHETIC_' + label + '_' + instruments[i] + '.dat','w')
        else:
            f = open('../../observations/' + planet_name + '/' + planet_name +
                     '_SYNTHETIC_' + label + '.dat','w')
                
        # Add Gaussian errors to binned points to produce synthetic data set
        for j in range(N_data):
            
            syn_err[j] = err_data[j]
            
            if (Gauss_scatter == True):   
                err = np.random.normal(0.0, err_data[j])
                syn_data[j] = syn_ymodel[j] + err
            else:
                syn_data[j] = syn_ymodel[j] 

            f.write('%f %f %e %e \n' %(wl_data[j], half_width_data[j], syn_data[j], syn_err[j]))
            
        f.close()
        
        wl_data_ALL = np.concatenate([wl_data_ALL, wl_data])
        half_width_data_ALL = np.concatenate([half_width_data_ALL, half_width_data])  
        syn_data_ALL = np.concatenate([syn_data_ALL, syn_data])
        syn_err_ALL = np.concatenate([syn_err_ALL, syn_err])
        syn_ymodel_ALL = np.concatenate([syn_ymodel_ALL, syn_ymodel])

    return syn_data_ALL, syn_err_ALL, wl_data_ALL, half_width_data_ALL, syn_ymodel

'''