''' 
Functions to computes various instrument properties and simulate data points.

'''

import os.path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d as gauss_conv
from scipy.integrate import trapz
from scipy.interpolate import InterpolatedUnivariateSpline as interp
from spectres import spectres


def fwhm_instrument(wl_data, instrument):
    '''
    Evaluate the full width at half maximum (FWHM) for the Point Spread 
    Function (PSF) of a given instrument mode at each bin centre wavelength.
    
    FWHM (μm) = wl (μm) / R_native 
    
    This assumes a Gaussian PSF with FWHM = native instrument spectral resolution.

    Args:
        wl_data (np.array of float): 
            Bin centre wavelengths of data points (μm).
        instrument (str):
            Instrument name corresponding to the dataset
            (e.g. WFC3_G141, JWST_NIRSpec_PRISM, JWST_NIRISS_SOSS_Ord2). 
    
    Returns:
        fwhm (np.array of float):
            Full width at half maximum as a function of wavelength (μm).

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
    elif (instrument.startswith('JWST')):    

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

    elif (instrument == 'IRTF_SpeX'):

        #fwhm_IRTF_SpeX(wl_data)  # Using the external resolution file currently

        # Find precomputed instrument spectral resolution file
        res_file = inst_dir + '/IRTF/' + instrument + '_resolution.dat'
        
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
        fwhm = 0.00001 * np.ones(N_bins) 
    
    return fwhm


def fwhm_IRTF_SpeX(wl_data):
    '''
    Calculate the wavelength dependent FWHM for the SpeX prism on the NASA
    Infrared Telescope Facility.

    Args:
        wl_data (np.array of float): 
            Bin centre wavelengths of data points (μm).
    
    Returns:
        fwhm (np.array of float):
            Full width at half maximum as a function of wavelength (μm).

    '''

    # Calculate average bin width
    delta1 = wl_data - np.roll(wl_data, 1)
    delta2 = np.abs(wl_data - np.roll(wl_data, -1))
    delta = 0.5 * (delta1 + delta2)
    
    # Approximate edge bin widths by second and penultimate bin widths
    delta[0] = delta[1]     
    delta[-1] = delta[-2]
    
    # Calculate wavelength dependent FWHM
    fwhm = 3.3 * delta   # SpeX FWHM is 3.3 times the data wavelength spacing 

    return fwhm


def init_instrument(wl, wl_data, half_width, instrument):
    '''
    Initialise required properties for a specific instrument. 
    
    This function conducts the following steps: 
        
    1) Read in the instrument sensitivity functions.
    2) Read in FWHM of the instrument PSF.
    3) Find the indices on the model wavelength grid closest to the bin centre 
       of each data point and the corresponding left/right bin edges.
    4) Pre-compute the integral of the sensitivity function over each bin
       (i.e. a normalising factor).

    These values are then stored for later usage, so this function need only be
    run once at the beginning of a retrieval.

    Args:
        wl (np.array of float):
            Model wavelength grid (μm).
        wl_data (np.array of float): 
            Bin centre wavelengths of data points (μm).
        half_width (np.array of float): 
            Bin half widths of data points (μm).
        instrument (str):
            Instrument name corresponding to the dataset
            (e.g. WFC3_G141, JWST_NIRSpec_PRISM, JWST_NIRISS_SOSS_Ord2). 
    
    Returns:
        sigma (np.array of float):
            Standard deviation of PSF for each data point (grid space unit).
        fwhm (np.array of float):
            Full width at half maximum as a function of wavelength (μm).
        sensitivity (np.array of float):
            Instrument transmission function interpolated to model wavelengths.
        bin_left (np.array of int):
            Closest index on model grid of the left bin edge for each data point.
        bin_cent (np.array of int):
            Closest index on model grid of the bin centre for each data point.
        bin_right (np.array of int):
            Closest index on model grid of the right bin edge for each data point.
        norm (np.array of float):
            Normalisation constant of the transmission function for each data bin.

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
        if (instrument == 'JWST_NIRSpec_Prism'): # Catch common misspelling of PRISM
            instrument = 'JWST_NIRSpec_PRISM'
        if ('NRS' in instrument):                # If G395H split into detectors, use common sensitivity function
            if ('G395H' in instrument):
                instrument = 'JWST_NIRSpec_G395H'
            elif ('G395M' in instrument):
                instrument = 'JWST_NIRSpec_G395M'
            elif ('G235H' in instrument):
                instrument = 'JWST_NIRSpec_G235H'
            elif ('G235M' in instrument):
                instrument = 'JWST_NIRSpec_G235M'
            elif ('G140H' in instrument):
                instrument = 'JWST_NIRSpec_G140H'
            elif ('G140M' in instrument):
                instrument = 'JWST_NIRSpec_G140M'
        sens_file = inst_dir + '/JWST/' + instrument + '_sensitivity.dat'
    
    # If instrument does not have a known sensitivity function, just use a top hat
    else:
        print("POSEIDON does not currently have an instrument transmission " +
              "function for " + instrument + ", so a box function will be used.")
        sens_file = inst_dir + '/dummy_instrument_sensitivity.dat'
    
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

    # Transmission function evaluated at model wavelengths
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
        
        # Compute closest indices on model grid corresponding to bin edges and centre
        bin_left = np.zeros(N_bins).astype(np.int64)     # Closest index on model grid of left edge of each bin
        bin_cent = np.zeros(N_bins).astype(np.int64)     # Closest index on model grid of centre of each bin
        bin_right = np.zeros(N_bins).astype(np.int64)    # Closest index on model grid of right edge of each bin
        
        sigma = np.zeros(shape=(N_bins))   # Standard deviation of PSF in grid spaces
        norm = np.zeros(shape=(N_bins))    # Normalisation factor for instrument function integration
    
        # For each data point
        for n in range(N_bins):
            
            # Compute closest indices on model grid corresponding to bin edges and centre
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
        
        # Compute closest indices on model grid corresponding to bin edges and centre
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
    
    return sigma, fwhm, sensitivity, bin_left, bin_cent, bin_right, norm
    

def make_model_data(spectrum, wl, sigma, sensitivity, bin_left, bin_cent, 
                    bin_right, norm, photometric = False):
    '''
    Produce binned model points at the same wavelengths and spectral resolution 
    as the observed data for a single instrument.

    This function conducts the following steps:
    
    1) Convolve the model with the instrument PSF for each data point.
    2) Integrate the convolved spectrum over the instrument sensitivity function.
    3) Produce binned model points via normalisation of the sensitivity function.

    For photometric bands, step (1) is not necessary.

    Args:
        spectrum (np.array of float):
            Model spectrum.
        wl (np.array of float):
            Model wavelength grid (μm).
        sigma (np.array of float):
            Standard deviation of PSF for each data point (grid space unit).
        sensitivity (np.array of float):
            Instrument transmission function interpolated to model wavelengths.
        bin_left (np.array of int):
            Closest index on model grid of the left bin edge for each data point.
        bin_cent (np.array of int):
            Closest index on model grid of the bin centre for each data point.
        bin_right (np.array of int):
            Closest index on model grid of the right bin edge for each data point.
        norm (np.array of float):
            Normalisation constant of the transmission function for each data bin.
        photometric (bool):
            If True, skip the PSF convolution (e.g. for Spitzer IRAC data).
    
    Returns:
        ymodel (np.array of float):
            Model spectrum convolved and binned to the data resolution.

    '''
    
    # For spectroscopic data
    if (photometric == False):
        
        N_bins = len(bin_cent)
        data = np.zeros(shape=(N_bins))
        ymodel = np.zeros(shape=(N_bins))
        
        for n in range(N_bins):
            
            # Extend convolution beyond bin edge by max(1, 2 PSF std) model grid spaces (std rounded to integer)
            extension = max(1, int(2 * sigma[n]))   
            
            # Convolve spectrum with PSF width appropriate for a given bin 
            spectrum_conv = gauss_conv(spectrum[(bin_left[n]-extension):(bin_right[n]+extension)], 
                                       sigma=sigma[n], mode='nearest')

            # Catch a (surprisingly common) error
            if (len(spectrum_conv[extension:-extension]) != len(sensitivity[bin_left[n]:bin_right[n]])):
                raise Exception("Error: Model wavelength range not wide enough to encompass all data.")

            integrand = spectrum_conv[extension:-extension] * sensitivity[bin_left[n]:bin_right[n]]
        
            # Integrate convolved spectrum over instrument sensitivity function
            data[n] = trapz(integrand, wl[bin_left[n]:bin_right[n]])   
            ymodel[n] = data[n]/norm[n]
            
    # For photometric data
    elif (photometric == True):
        
        integrand = spectrum[bin_left[0]:bin_right[0]]*sensitivity[bin_left[0]:bin_right[0]]
        
        # Integrate spectrum over instrument sensitivity function
        data = trapz(integrand, wl[bin_left[0]:bin_right[0]])
        ymodel = data/norm
        
    return ymodel 
    

def bin_spectrum_to_data(spectrum, wl, data_properties):
    '''
    Generate the equivalent model predicted spectrum at the spectral resolution
    of the data. This function serves as a wrapper, unpacking the POSEIDON
    data_properties dictionary and calling 'make_model_data' to handle the
    binning for each instrument separately.

    Args:
        spectrum (np.array of float or list of np.array of floats):
            Model spectrum. If a list, each element of the list is the model spectrum
            for a specific dataset.
        wl (np.array of float):
            Model wavelength grid (μm).
        data_properties (dict):
            Collection of data properties required for POSEIDON's instrument
            simulator (i.e. to create simulated binned data during retrievals).

    Returns:
        ymodel (np.array of float):
            Model spectrum convolved and binned to the data resolution.

    '''

    # Initialise combined array of binned model points (all instruments)
    ymodel = np.array([])
                
    # Generate binned model points for each instrument
    for i in range(len(data_properties['datasets'])):
            
        if (data_properties['instruments'][i] in ['IRAC1', 'IRAC2']): 
            photometric = True
        else: 
            photometric = False

        # Get spectrum for dataset_i (in case model spectrum is different for each dataset, e.g. stellar contamination)
        if type(spectrum) == list:
            spectrum_i = spectrum[i]
        else:
            spectrum_i = spectrum
        
        # Compute binned transit depths for dataset_i
        ymodel_i = make_model_data(spectrum_i, wl, data_properties['psf_sigma'][i],
                                   data_properties['sens'][i],
                                   data_properties['bin_left'][i],
                                   data_properties['bin_cent'][i],
                                   data_properties['bin_right'][i],
                                   data_properties['norm'][i], photometric)
                                                
        # Combine binned model points for each instrument
        ymodel = np.concatenate([ymodel, ymodel_i])    
            
    return ymodel


def R_to_wl(R_data, wl_data_min, wl_data_max):
    '''
    Convert a given spectral resolution to the equivalent set of wavelengths
    and bin half-widths.

    Args:
        R_data (float):
            Spectral resolution (wl/dwl) of the data.
        wl_data_min (float):
            Starting wavelength of new data grid (μm).
        wl_data_max (float):
            Ending wavelength of new data grid (μm).

    Returns:
        wl_data (np.array of float):
            New wavelength grid spaced with the desired spectral resolution (μm).
        half_width_data (np.array of float):
            Half bin width for the new wavelength grid (μm).

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
        
    return wl_data, half_width_data


def generate_syn_data_from_user(planet, wl_model, spectrum, data_dir, 
                                instrument, R_data = 100, err_data = 50, 
                                wl_start = 1.1, wl_end = 1.8, 
                                label = None, Gauss_scatter = True):
    '''
    Generate and write to file a synthetic dataset with a user specified 
    spectral resolution, precision, and wavelength range. Gaussian scatter can 
    be optionally disabled, with the data lying on the binned model points.

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        wl_model (np.array of float):
            Model wavelength grid (μm).
        spectrum (np.array of float):
            Model spectrum.
        data_dir (str):
            Directory where the synthetic datafile will be written.
        instrument (str):
            Instrument name corresponding to the dataset
            (e.g. WFC3_G141, JWST_NIRSpec_PRISM, JWST_NIRISS_SOSS_Ord2). 
        R_data (float):
            Spectral resolution (wl/dwl) of the synthetic dataset.
        err_data (float):
            Precision of the synthetic dataset, assumed constant with wl (ppm).
        wl_start (float):
            Starting wavelength of the synthetic dataset (μm).
        wl_end (float):
            Ending wavelength of the synthetic dataset (μm).
        label (str):
            Optional descriptive label to add to file name.
        Gauss_scatter (bool):
            If True, applies Gaussian scatter with 1σ = err_data.

    Returns:
        None.

    '''

    print("Creating synthetic data")

    # Unpack planet name
    planet_name = planet['planet_name']
    
    # Check if selected instrument corresponds to a photometric band
    if (instrument in ['IRAC1', 'IRAC2']): 
        is_photometric = True
    else: 
        is_photometric = False
    
    # For given R, compute wavelengths of data points and half-bin width 
    wl_data, half_bin = R_to_wl(R_data, wl_start, wl_end)

    N_data = len(wl_data)
    
    # Initialise the instrument properties for the synthetic dataset 
    sigma, fwhm, sens, bin_left, \
    bin_cent, bin_right, norm = init_instrument(wl_model, wl_data, half_bin, 
                                                instrument)
    
    # Compute synthetic binned model points
    syn_ymodel = make_model_data(spectrum, wl_model, sigma, sens, bin_left, 
                                    bin_cent, bin_right, norm, is_photometric)

    # Arrays containing synthetic data and 1-sigma errors
    syn_data = np.zeros(shape=(N_data))
    syn_err = err_data * 1.0e-6    # Convert ppm to transit depths

    # Open output file where synthetic data will be written
    if (instrument != 'None'):
        if (label is None):
            f = open(data_dir + '/' + planet_name + '_SYNTHETIC_' + 
                        instrument + '.dat', 'w')
        else:
            f = open(data_dir + '/' + planet_name + '_SYNTHETIC_' + 
                        instrument + '_' + label + '.dat', 'w')
    else:
        if (label is None):
            f = open(data_dir + '/' + planet_name + '_SYNTHETIC_.dat', 'w')
        else:
            f = open(data_dir + '/' + planet_name + '_SYNTHETIC_' + 
                        label + '.dat', 'w')
            
    # Add Gaussian errors to binned points to produce synthetic data set
    for j in range(N_data):
        
        if (Gauss_scatter == True):   
            err = np.random.normal(0.0, syn_err)
            syn_data[j] = syn_ymodel[j] + err
        else:
            syn_data[j] = syn_ymodel[j] 

        f.write('%.6f %.6f %.6e %.6e \n' %(wl_data[j], half_bin[j], 
                                           syn_data[j], syn_err))
        
    f.close()


def generate_syn_data_from_file(planet, wl_model, spectrum, data_dir, 
                                data_properties, R_to_bin = [], 
                                N_trans = [], label = None, Gauss_scatter = True):
    '''
    Generate and write to file a synthetic dataset with the same precision as
    an externally provided file. The synthetic dataset can optionally be
    rebinned and/or scaled by 1/sqrt(N) for additional transits. Rebinning or
    additional transits will adjust the data precision accordingly. If no 
    rebinning is requested, the synthetic dataset will have the same wavelength 
    grid as the external file. The user can also disable Gaussian scattering, 
    in which case the data will coincide with the binned model points.

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        wl_model (np.array of float):
            Model wavelength grid (μm).
        spectrum (np.array of float):
            Model spectrum.
        data_dir (str):
            Directory where the synthetic datafile will be written.
        instrument (str):
            Instrument name corresponding to the dataset
            (e.g. WFC3_G141, JWST_NIRSpec_PRISM, JWST_NIRISS_SOSS_Ord2). 
        R_to_bin (list of float):
            Output spectral resolution for rebinning each instrument's data.
        N_trans (list of float):
            Number of transits observed by each instrument.
        label (str):
            Optional descriptive label to add to file name.
        Gauss_scatter (bool):
            If True, applies Gaussian scatter with 1σ = err_data.

    Returns:
        None.

    '''
           
    print("Creating synthetic data")

    # Unpack planet name and instrument names
    planet_name = planet['planet_name']
    instruments = data_properties['instruments']

    # Generate dataset for each provided instrument
    for i in range(len(instruments)):
                    
        print(instruments[i])

        if (N_trans == []):
            N_trans_i = 1     # Use one transit if not specified by user
        else:
            N_trans_i = N_trans[i]

        if (R_to_bin == []):
            R_to_bin_i = None    # No binning if not specified by user
        else:
            R_to_bin_i = R_to_bin[i]
        
        # Check if selected instrument corresponds to a photometric band
        if (data_properties['instruments'][i] in ['IRAC1', 'IRAC2']): 
            photometric = True
        else: 
            photometric = False

        # Unpack data properties for this dataset 
        err_data = data_properties['err_data'][i]
        wl_data = data_properties['wl_data'][i]
        half_bin = data_properties['half_bin'][i]

        # Compute binned transit depths for dataset_i
        syn_ymodel = make_model_data(spectrum, wl_model, data_properties['psf_sigma'][i],
                                     data_properties['sens'][i],
                                     data_properties['bin_left'][i],
                                     data_properties['bin_cent'][i],
                                     data_properties['bin_right'][i],
                                     data_properties['norm'][i], photometric)

        # If simulated data will be further binned down
        if (R_to_bin_i != None):

            # Create new data wavelength grid at lower resolution
            wl_data_new, half_bin_new = R_to_wl(R_to_bin[i], wl_data[0], wl_data[-1])
            
            # Initialise the instrument properties for the synthetic dataset 
            sigma, fwhm, sens, bin_left, \
            bin_cent, bin_right, norm = init_instrument(wl_model, wl_data_new, 
                                                        half_bin_new, 
                                                        instruments[i])
            
            # Compute synthetic binned model points
            syn_ymodel_new = make_model_data(spectrum, wl_model, sigma, sens, bin_left, 
                                             bin_cent, bin_right, norm, photometric)

            # Obtain new error bars corresponding to the new spectral resolution
            _, err_data_new = spectres(wl_data_new, wl_data, syn_ymodel, 
                                       spec_errs=err_data, verbose = False)

            # Replace Spectres boundary NaNs with second and penultimate values
            err_data_new[0] = err_data_new[1]
            err_data_new[-1] = err_data_new[-2]
     

        # No further binning
        else:

            # Maintain output quantities
            wl_data_new = wl_data
            half_bin_new = half_bin
            err_data_new = err_data
            syn_ymodel_new = syn_ymodel

        # Divide error bars by sqrt(number of transits)
        err_data_new = err_data_new/np.sqrt(N_trans_i)
        
        # Find number of data points for dataset_i
        N_data = len(wl_data_new)

        # Arrays containing synthetic data and 1-sigma errors
        syn_data = np.zeros(shape=(N_data))

        # Open output file where synthetic data will be written
        if (instruments[i] != 'None'):
            if (label is None):
                f = open(data_dir + '/' + planet_name + '_SYNTHETIC_' + 
                         instruments[i] + '_N_trans_' + str(N_trans_i) + 
                         '.dat', 'w')
            else:
                f = open(data_dir + '/' + planet_name + '_SYNTHETIC_' + 
                        instruments[i] + '_' + label + '_N_trans_' + 
                        str(N_trans_i) + '.dat', 'w')
        else:
            if (label is None):
                f = open(data_dir + '/' + planet_name + '_SYNTHETIC_' + 
                         '_N_trans_' + str(N_trans_i) + '.dat', 'w')
            else:
                f = open(data_dir + '/' + planet_name + '_SYNTHETIC_' + 
                         label + '_N_trans_' + str(N_trans_i) + '.dat', 'w')
                
        # Add Gaussian errors to binned points to produce synthetic data set
        for j in range(N_data):
            
            if (Gauss_scatter == True):   
                err = np.random.normal(0.0, err_data_new[j])
                syn_data[j] = syn_ymodel_new[j] + err
            else:
                syn_data[j] = syn_ymodel_new[j] 

            f.write('%.6f %.6f %.6e %.6e \n' %(wl_data_new[j], half_bin_new[j], 
                                               syn_data[j], err_data_new[j]))
            
        f.close()
        
    return

