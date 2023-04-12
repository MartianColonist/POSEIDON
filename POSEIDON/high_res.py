from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
import pickle
from scipy import constants
from numba import jit
from astropy.io import fits
from scipy import interpolate
from .constants import R_Sun
from .constants import R_J, M_J
import numpy as np
from spectres import spectres
from sklearn.decomposition import TruncatedSVD
from scipy.ndimage import gaussian_filter1d

@jit
def get_rot_kernel(V_sin_i, wl, W_conv):
    '''
    Get rotational kernel given V sin(i) and wavelength grid of the forward model.

    Args:
        V_sin_i (float):
            TODO: V sin_i
        wl (np.array of float):
            Wavelength grid of the forward model.

    '''

    dRV = np.mean(2.0*(wl[1: ]-wl[0: -1])/(wl[1: ]+wl[0: -1]))*2.998E5
    n_ker = int(W_conv)
    half_n_ker = (n_ker - 1)//2
    rot_ker = np.zeros(n_ker)
    for ii in range(n_ker):
        ik = ii - half_n_ker
        x = ik * dRV / V_sin_i
        if np.abs(x) < 1.0:
            y = np.sqrt(1 - x**2)
            rot_ker[ii] = y
    rot_ker /= rot_ker.sum()

    return rot_ker


def log_likelihood_PCA(V_sys, K_p, dPhi, cs_p, cs_s, wl_grid, data_arr, data_scale, V_bary, Phi):
    '''
    Perform the loglikelihood calculation using singular value decompositions.
    Nord: number of spectral order.
    Nphi: number of time-resolved phases.
    Npix: number of wavelengths per spectral order.
    Typical values for (Nord, Nphi, Npix) is (44, 79, 1848).

    Args:
        V_sys (float):
            The system velocity (km/s) at which we do the loglikelihood calculation.
        K_p (float):
            The Keplerian velocity (km/s) at which we do the loglikelihood calculation.
        dPhi (float):
            Phase offset.
        cs_p (np.array of float):
            Spline representation of the observed flux of planet.
        cs_s (np.array of float): 
            Spline representation of the observed flux of star.
        wl_grid (2D np.array of float):
            2D wavelength grid of the data (Nord x Npix). Typical size ~(44, 1848).
        data_arr (3D np.array of float):
            3D Array representing the top principal components removed data.
            Shape: (Nord x Nphi x Npix)
        data_scale (3D np.array of float):
            3D Array representing the top principal components of data.
            Shape: (Nord x Nphi x Npix)
        V_bary (np.array of float):
            Array of time-resolved Earth-star velocity. We have absorbed V_sys into V_bary, so V_sys = V_sys_literature + d_V_sys.
            Shape: (Nphi, )
        Phi (np.array of float):
            Array of time-resolved phases.
            Shpae (Nphi, )
    
    Returns:
        Loglikelihood (float):
            Loglikelihood value.
    '''

    Nord, Nphi, Npix = data_arr.shape

    I = np.ones(Npix)
    N = Npix
    
    # Time-resolved total radial velocity
    RV_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (Phi + dPhi))  # V_sys is an additive term around zero
    dl_p = RV_p * 1e3 / constants.c # delta lambda, for shifting
    # RV_s = 0 # Velocity of the star is very small compared to planet's velocity and it's already be corrected
    K_s = 0.3229*1.
    RV_s = (V_sys + V_bary - K_s * np.sin(2 * np.pi * Phi)) * 0
    dl_s = RV_s * 1e3 / constants.c # delta lambda, for shifting
    
    loglikelihood = 0
    
    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for j in range(Nord): # Nord = 44 This takes 2.2 seconds to complete
        wl_slice = wl_grid[j, ].copy() # Cropped wavelengths
        Fp_Fs = np.zeros((Nphi, Npix))  # "shifted" model spectra array at each phase
        for i in range(Nphi): # This for loop takes 0.025 seconds Nphi = 79
            wl_shifted_p = wl_slice * (1.0 - dl_p[i])
            Fp = interpolate.splev(wl_shifted_p, cs_p, der=0)
            wl_shifted_s = wl_slice * (1.0 - dl_s[i])
            Fs = interpolate.splev(wl_shifted_s, cs_s, der=0)
            Fp_Fs[i, :] = Fp / Fs

        model_injected = (1 + Fp_Fs) * data_scale[j, :]

        svd = TruncatedSVD(n_components=4, n_iter=4, random_state=42).fit(model_injected)
        model_injected_PCs_removed = model_injected - (svd.transform(model_injected) @ svd.components_) # 0.008 s
        # svd.transform gives data matrix in reduced dimension (79, 5). svd.components gives the first n_components right singular vectors (5, N_wl)  
        # Original data minus PCA-ed data is equivalent to doing np.linalg.svd, setting first n_components components to zero.

        for i in range(Nphi): # This loop takes 0.001 second
            gVec = model_injected_PCs_removed[i]
            gVec -= gVec.dot(I) / Npix  # mean subtracting here...
            sg2 = gVec.dot(gVec) / Npix
            fVec = data_arr[j, i]       # already mean-subtracted
            sf2 = fVec.dot(fVec) / Npix
            R = fVec.dot(gVec) / Npix   # cross-covariance
            loglikelihood += -0.5 * N * np.log(sf2 + sg2 - 2.0 * R) # Equation 9 in paper

    return loglikelihood


def log_likelihood_sysrem(V_sys, K_p, dPhi, cs_p, wl_grid, residuals, Bs, V_bary, Phi, tmodel, uncertainties, a, b):

    Nord, Nphi, Npix = residuals.shape

    N = Nord * Nphi * Npix

    # Time-resolved total radial velocity
    RV_p = V_sys + K_p * np.sin(2 * np.pi * (Phi + dPhi))  # V_sys is an additive term around zero. Data is in rest frame of star.
    dl_p = RV_p * 1e3 / constants.c # delta lambda, for shifting
    
    # Initializing loglikelihood
    loglikelihood = 0
    CCF_sum = 0
    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for i in range(Nord):
        wl_slice = wl_grid[i]                    # Cropped wavelengths
        models_shifted = np.zeros((Nphi, Npix))  # "shifted" model spectra array at each phase
        for j in range(Nphi):
            wl_shifted_p = wl_slice * (1.0 - dl_p[j])
            # wl_shifted_p = wl_slice * np.sqrt((1.0 - dl_p[j]) / (1 + dl_p[j]))
            Fp = np.interp(wl_shifted_p, cs_p[0], cs_p[1])
            # Fp = interpolate.splev(wl_shifted_p, cs_p, der=0) # linear interpolation, einsum
            models_shifted[j] = ((1-tmodel[j]))/np.max(1-tmodel)*(-Fp) + 1 

        models_shifted = (models_shifted.T / np.median(models_shifted, axis=1)).T # divide by the median over wavelength to mimic blaze correction

        B = Bs[i]
        model_filtered = models_shifted - B @ models_shifted # filter the model
        
        if b:
            for j in range(Nphi):
                m = model_filtered[j] / uncertainties[i, j] * a
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood -= 0.5 * (m2 + f2 - 2.0*CCF) / (b ** 2)
                CCF_sum += CCF
        else: # nulled b
            for j in range(Nphi):
                m = model_filtered[j]/ uncertainties[i, j]
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood -= Npix / 2 * np.log((m2 + f2 - 2.0*CCF) / Npix) 
                CCF_sum += CCF
        # else: # nulled b and uncertainties
        #     for j in range(Nphi):
        #         m = model_filtered[j]
        #         m2 = m.dot(m)
        #         f = residuals[i, j]
        #         f2 = f.dot(f)
        #         CCF = f.dot(m)
        #         loglikelihood -= Npix / 2 * np.log((m2 + f2 - 2.0*CCF) / Npix) 

    if b:
        loglikelihood -= N * np.log(b)

    # loglikelihood -= np.sum(np.log(uncertainties))
    # loglikelihood -= N / 2 * np.log(2*np.pi)          (These two terms are normalization)
    
    return loglikelihood, CCF_sum

def log_likelihood(F_s_obs, spectrum, wl, data, model, high_res_params, 
                    high_res_param_names):
    '''
    Return the loglikelihood given the observed flux, Keplerian velocity, and centered system velocity.
    Use this function in a high resolutional rerieval.
    Nord: number of spectral order.
    Nphi: number of time-resolved phases.
    Npix: number of wavelengths per spectral order.
    Typical values for (Nord, Nphi, Npix) is (44, 79, 1848).

    Args:
        F_s_obs (np.array of float):
            Flux of the star observed at distance d = 1 pc.
        F_p_obs (np.array of float): 
            Flux of the planet observed at distance d = 1 pc.
        wl (np.array of float):
            Wavelength grid of the forward model. Typical size ~10^5 in a high-res retrieval.
        K_p (float):
            The Keplerian velocity (km/s) at which we do the loglikelihood calculation.
        V_sys (float):
            The system velocity (km/s) at which we do the loglikelihood calculation.
        log_a (float):
            Log 10 scale factor for this night. Scale = 10**log_a.
        dPhi (float):
            Phase offset.
        wl_grid (2D np.array of float):
            2D wavelength grid of the data (Nord x Npix). Typical size ~(44, 1848).
        data_arr (3D np.array of float):
            3D Array representing the top principal components removed data.
            Shape: (Nord x Nphi x Npix)
        data_scale (3D np.array of float):
            3D Array representing the top principal components of data.
            Shape: (Nord x Nphi x Npix)
        V_bary (np.array of float):
            Array of time-resolved Earth-star velocity. We have absorbed V_sys into V_bary, so V_sys = V_sys_literature + d_V_sys.
            Shape: (Nphi, )
        Phi (np.array of float):
            Array of time-resolved phases.
            Shpae (Nphi, )
    
    Returns:
        logL_Matteo (float):
            Loglikelihood given by Log(L) = -N/2 Log(s_f^2 - 2R(s) + s_g^2). Equation 9 in Brogi & Line 2019 March.
    '''

    method = data['method']
    wl_grid = data['wl_grid']
    V_bary = data['V_bary']
    Phi = data['Phi']
    spectrum_type = data['spectrum_type']

    if ('K_p' in high_res_param_names):
        K_p = high_res_params[np.where(high_res_param_names == 'K_p')[0][0]]
    else:
        K_p = model['K_p']

    if ('V_sys' in high_res_param_names):
        V_sys = high_res_params[np.where(high_res_param_names == 'V_sys')[0][0]]
    else:
        V_sys = model['V_sys']

    if ('a' in high_res_param_names):
        a = high_res_params[np.where(high_res_param_names == 'a')[0][0]]
    else:
        a = 1

    if ('dPhi' in high_res_param_names):
        dPhi = high_res_params[np.where(high_res_param_names == 'dPhi')[0][0]]
    else:
        dPhi = 0

    if ('W_conv' in high_res_param_names):
        W_conv = high_res_params[np.where(high_res_param_names == 'W_conv')[0][0]]
    else:
        W_conv = model['W_conv']

    if ('b' in high_res_param_names):
        b = high_res_params[np.where(high_res_param_names == 'b')[0][0]]
    else:
        b = None

    if method == 'pca' and spectrum_type == 'emission':
        data_scale = data['data_scale']
        data_arr = data['data_arr']
        # instrument profile convolution
        R_instrument = model['R_instrument']
        R = model['R']
        V_sin_i = data['V_sin_i']
        rot_kernel = get_rot_kernel(V_sin_i, wl, W_conv)
        F_p_rot = np.convolve(spectrum, rot_kernel, mode='same') # calibrate for planetary rotation
        xker = np.arange(-20, 21)
        sigma = (R / R_instrument) / (2 * np.sqrt(2.0 * np.log(2.0)))  # model is right now at R=250K.  IGRINS is at R~45K. We make gaussian that is R_model/R_IGRINS ~ 5.5 
        yker = np.exp(-0.5 * (xker / sigma) ** 2.0)   # instrumental broadening kernel; not understand
        yker /= yker.sum()
        F_p_conv = np.convolve(F_p_rot, yker, mode='same') * a
        cs_p = interpolate.splrep(wl, F_p_conv, s=0.0) # no need to times (R)^2 because F_p, F_s are already observed value on Earth
        F_s_conv = np.convolve(F_s_obs, yker, mode='same')
        cs_s = interpolate.splrep(wl, F_s_conv, s=0.0)
        return log_likelihood_PCA(V_sys, K_p, dPhi, cs_p, cs_s, wl_grid, data_arr, data_scale, V_bary, Phi)

    elif method == 'sysrem' and spectrum_type == 'transmission':
        residuals = data['residuals']
        Bs = data['Bs']
        tmodel = data['transit_weight']
        uncertainties = data['uncertainties']
        F_p_conv = gaussian_filter1d(spectrum, W_conv) # np.convolve use smaller kernel. Apply filter to the spectrum. And multiply by scale factor a.
        # cs_p = interpolate.splrep(wl, F_p_conv, s=0.0)
        cs_p = [wl, F_p_conv]
        return log_likelihood_sysrem(V_sys, K_p, dPhi, cs_p, wl_grid, residuals, Bs, V_bary, Phi, tmodel, uncertainties, a, b)[0]
    else:
        raise Exception('Problem with high res retreival data.')

def sysrem(data_array, stds, iter=15):

    data_transpose = data_array.T
    Npix, Nphi = data_transpose.shape

    # Create empty matrices for residuals and corresponding errors with the found dimensions such that number of rows correspond to the number of available stars, and the number of columns correspond to each specific epoch:
    residuals = np.zeros((Npix, Nphi))

    median_list = []
    # Import each of the star files
    for i, wl_channel in enumerate(data_transpose):

        median_list.append(np.median(wl_channel))

        # Calculate residuals from the ORIGINAL light curve
        channel_residual = wl_channel - np.median(wl_channel)
        # channel_residual = wl_channel

        # import the residual and error values into the matrices in the correct position (rows corresponding to stars, columns to epochs)
        residuals[i] = channel_residual

    Npix, Nphi = np.shape(residuals)

    # This medians.txt file is a 2D list with the first column being the medians
    # of stars' magnitudes at different epochs (the good ones) and their
    # standard deviations, so that they can be plotted against the results after
    # errors are taken out below.
    
    U = np.zeros((Nphi, iter+1))

    for i in range(iter):         # The number of linear systematics to remove
        w = np.zeros(Npix)
        u = np.ones(Nphi)

        # minimize a and c values for a number of iterations, iter
        for _ in range(10):

            # Using the initial guesses for each a value of each epoch, minimize c for each star
            for pix in range(Npix):
                err_squared = stds.T[pix] ** 2
                numerator = np.sum(u * residuals[pix] / err_squared)
                denominator = np.sum(u**2 / err_squared)
                w[pix] = numerator / denominator

            # Using the c values found above, minimize a for each epoch
            for phi in range(Nphi):
                err_squared = stds.T[:, phi] ** 2
                numerator = np.sum(w * residuals[:, phi] / err_squared)
                denominator = np.sum(w**2 / err_squared)
                u[phi] = numerator / denominator

        # Create a matrix for the systematic errors:
        systematic = np.zeros((Npix, Nphi))
        for pix in range(Npix):
            for phi in range(Nphi):
                systematic[pix, phi] = u[phi] * w[pix]
        
        # Remove the systematic error
        residuals = residuals - systematic

        U[:, i] = u

    # for i in range(len(residuals)):
    #     residuals[i] += median_list[i]

    U[:, -1] = np.ones(Nphi)

    return residuals.T, U

def fast_filter(data, uncertainties, iter=15):

    Nord, Nphi, Npix = data.shape
    residuals = np.zeros((Nord, Nphi, Npix))
    Us = np.zeros((Nord, Nphi, iter+1))
    
    for i, order in enumerate(data):
        stds = uncertainties[i]
        residual, U = sysrem(order, stds, iter)
        residuals[i] = residual
        Us[i] = U
    
    return residuals, Us

def make_data_cube(data, wl_grid):
    Nord, Nphi, Npix = data.shape # yup, this again--Norders x Nphases x Npixels
    # SVD/PCA method

    data_scale = np.zeros(data.shape)
    data_arr = np.zeros(data.shape)

    NPC = 4 #change the "4" to whatever. This is the number of PC's to remove

    for i in range(Nord):
        #taking only first four vectors, reconstructiong, and saving
        u,s,vh=np.linalg.svd(data[i], full_matrices = False)  #decompose
        s[NPC: ] = 0
        W=np.diag(s)
        A=np.dot(u,np.dot(W,vh))
        data_scale[i]=A

        # removing first four vectors...this is the 'processesed data'
        u, s, vh = np.linalg.svd(data[i], full_matrices = False)  #decompose--not sure why I did it again....guess you don't really need this line
        s[0: NPC] = 0
        W = np.diag(s)
        A = np.dot(u, np.dot(W, vh))
        
        # sigma clipping sort of--it really doesn't make a yuge difference.
        
        sigma = np.std(A)
        median = np.median(A)
        loc = np.where(A > 3 * sigma + median)
        A[loc] = 0 # *0.+20*sig
        loc = np.where(A < -3 * sigma + median)
        A[loc] = 0 # *0.+20*sig
        
        data_arr[i] = A

    return data_scale, data_arr


def cross_correlate(F_s_obs, F_p_obs, wl, K_p_arr, V_sys_arr, wl_grid, data_arr, data_scale, V_bary, Phi):
    '''
    Cross correlate at an array of Keplerian velocities and an array of centered system velocities given the observed flux.
    Use this function to create the cross correlation plot of detection level.
    Nord: number of spectral order.
    Nphi: number of time-resolved phases.
    Npix: number of wavelengths per spectral order.
    Typical values for (Nord, Nphi, Npix) is (44, 79, 1848).

    Args:
        F_s_obs (np.array of float):
            Flux of the star observed at distance d = 1 pc.
        F_p_obs (np.array of float): 
            Flux of the planet observed at distance d = 1 pc.
        wl (np.array of float):
            Wavelength grid of the forward model. Typical size ~10^5 in a high-res retrieval.
        K_p_arr (np.array of float):
            Array of Keplerian velocities (km/s).
        V_sys_arr (np.array of float):
            Array of centered system velocity (km/s).
        wl_grid (2D np.array of float):
            2D wavelength grid of the data (Nord x Npix). Typical size ~(44, 1848).
        data_arr (3D np.array of float):
            3D Array representing the top principal components removed data.
            Shape: (Nord x Nphi x Npix)
        data_scale (3D np.array of float):
            3D Array representing the top principal components of data.
            Shape: (Nord x Nphi x Npix)
        V_bary (np.array of float):
            Array of time-resolved Earth-star velocity. We have absorbed V_sys into V_bary, so V_sys = V_sys_literature + d_V_sys.
            Shape: (Nphi, )
        Phi (np.array of float):
            Array of time-resolved phases.
            Shpae (Nphi, )
    
    Returns:
        logL_M_arr (np.array of float):
            Array of loglikelihood given by Log(L) = -N/2 Log(s_f^2 - 2R(s) + s_g^2). Equation 9 in Brogi & Line 2019 March.
        logL_Z_arr (np.array of float):
            Array of loglikelihood given by Log(L) = -N/2 Log(1.0 - CC^2)). Equation 2 in Brogi & Line 2019 March.
        CCF_arr (float):
            Array of cross correlation value.
    '''

    dPhi = 0.0
    scale = 1.0

    # rotational coonvolutiono
    V_sin_i = 4.5
    rot_kernel = get_rot_kernel(V_sin_i, wl)
    F_p_rot = np.convolve(F_p_obs, rot_kernel, mode='same') # calibrate for planetary rotation

    # instrument profile convolustion
    xker = np.arange(-20, 21)
    sigma = 5.5/(2.* np.sqrt(2.0 * np.log(2.0)))  # nominal
    yker = np.exp(-0.5 * (xker / sigma) ** 2.0)   # instrumental broadening kernel; not understand yet
    yker /= yker.sum()
    F_p_conv = np.convolve(F_p_rot, yker, mode='same') * scale
    F_s_conv = np.convolve(F_s_obs, yker, mode='same')

    cs_p = interpolate.splrep(wl, F_p_conv, s=0.0) # no need to times (R)^2 because F_p_obs, F_s_obs are already observed value on Earth
    cs_s = interpolate.splrep(wl, F_s_conv, s=0.0)

    loglikelihood_arr = np.zeros((len(K_p_arr), len(V_sys_arr)))
    
    for i in range(len(K_p_arr)):
        for j in range(len(V_sys_arr)):
            loglikelihood = log_likelihood_PCA(V_sys_arr[j], K_p_arr[i], dPhi, cs_p, cs_s, wl_grid, data_arr, data_scale, V_bary, Phi)
            loglikelihood_arr[i, j] = loglikelihood

    return loglikelihood_arr

from scipy.optimize import minimize

def fit_uncertainties(data_raw, NPC=5):
    uncertainties = np.zeros(data_raw.shape)
    residuals = np.zeros(data_raw.shape)
    Nord = len(data_raw)
    mask = data_raw == 0
    for i in range(Nord):
        order = data_raw[i]
        svd = TruncatedSVD(n_components=NPC, n_iter=15, random_state=42).fit(order)

        residual = order - (svd.transform(order) @ svd.components_)
        residuals[i] = residual

    for i in range(Nord):
        def fun(x):
            a, b = x
            sigma = np.sqrt(a * data_raw[i] + b)
            loglikelihood = -0.5 * np.sum((residuals[i] / sigma) ** 2) - np.sum(np.log(sigma))
            return -loglikelihood
        a, b = minimize(fun, [1, 1], method='Nelder-Mead').x
        best_fit = np.sqrt(a * data_raw[i] + b)

        svd = TruncatedSVD(n_components=NPC, n_iter=15, random_state=42).fit(best_fit)

        uncertainty = svd.transform(best_fit) @ svd.components_
        uncertainties[i] = uncertainty
    uncertainties[mask] = 1e7
    return uncertainties 