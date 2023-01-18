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
    Ndet: number of spectral order.
    Nphi: number of time-resolved phases.
    Npix: number of wavelengths per spectral order.
    Typical values for (Ndet, Nphi, Npix) is (44, 79, 1848).

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
            2D wavelength grid of the data (Ndet x Npix). Typical size ~(44, 1848).
        data_arr (3D np.array of float):
            3D Array representing the top principal components removed data.
            Shape: (Ndet x Nphi x Npix)
        data_scale (3D np.array of float):
            3D Array representing the top principal components of data.
            Shape: (Ndet x Nphi x Npix)
        V_bary (np.array of float):
            Array of time-resolved Earth-star velocity. We have absorbed V_sys into V_bary, so V_sys = V_sys_literature + d_V_sys.
            Shape: (Nphi, )
        Phi (np.array of float):
            Array of time-resolved phases.
            Shpae (Nphi, )
    
    Returns:
        logL_Matteo (float):
            Loglikelihood given by Log(L) = -N/2 Log(s_f^2 - 2R(s) + s_g^2). Equation 9 in Brogi & Line 2019 March.
        logL_Zack (float):
            Loglikelihood given by Log(L) = -N/2 Log(1.0 - CC^2)). Equation 2 in Brogi & Line 2019 March.
        CCF (float):
            cross correlation value.
    '''

    Ndet, Nphi, Npix = data_arr.shape

    I = np.ones(Npix)
    N = Npix # np.array([Npix])
    
    # Time-resolved total radial velocity
    RV_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (Phi + dPhi))  # V_sys is an additive term around zero   
    dl_p = RV_p * 1e3 / constants.c # delta lambda, for shifting
    RV_s = 0 # Velocity of the star is very small compared to planet's velocity and it's already be corrected
    # RV_s = (V_sys + V_bary - K_s * np.sin(2 * np.pi * Phi)) * 0
    dl_s = RV_s * 1e3 / constants.c # delta lambda, for shifting
    
    loglikelihood = 0
    
    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for j in range(Ndet): # Ndet = 44 This takes 2.2 seconds to complete
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


def log_likelihood_sysrem(V_sys, K_p, dPhi, cs_p, wl_grid, residuals, Us, V_bary, Phi, uncertainties=None, b=None):

    Ndet = len(residuals)
    Nphi, Npix = residuals[0].shape

    I = np.ones(Npix)
    N = Npix

    # Time-resolved total radial velocity
    RV_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (Phi + dPhi))  # V_sys is an additive term around zero   
    dl_p = RV_p * 1e3 / constants.c # delta lambda, for shifting
    
    # Initializing log-likelihoods and CCFs
    loglikelihood = 0

    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for j in range(Ndet):
        wl_slice = wl_grid[j]                    # Cropped wavelengths
        models_shifted = np.zeros((Nphi, Npix))  # "shifted" model spectra array at each phase
        for i in range(Nphi):
            wl_shifted_p = wl_slice * (1.0 - dl_p[i])
            Fp = interpolate.splev(wl_shifted_p, cs_p, der=0)
            models_shifted[i] = Fp + 1
            models_shifted[i] /= np.median(models_shifted[i])

        U = Us[j]
        L = np.diag(1 / np.mean(uncertainties[j], axis=-1))
        model_filtered = models_shifted-(U @ np.linalg.inv((L @ U).T @ (L @ U)) @ (L @ U).T) @ (L @ models_shifted) # 0.002 second

        if not b:
            for i in range(Nphi): # This loop takes 0.001 second
                gVec = model_filtered[i]
                sg2 = gVec.dot(gVec / uncertainties[j, i]) / Npix 
                fVec = residuals[j][i]
                sf2 = fVec.dot(fVec / uncertainties[j, i]) / Npix
                R = fVec.dot(gVec / uncertainties[i, i]) / Npix
                loglikelihood += -0.5 * N * np.log(sf2 + sg2 - 2.0*R) # Equation 9 in paper
        else:
            for i in range(Nphi):
                m = model_filtered[i] / uncertainties[j, i]
                m2 = m.dot(m)
                f = residuals[j][i] / uncertainties[j, i]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood -= 0.5 * (m2 + f2 - 2.0*CCF) / (b ** 2)
                # loglikelihood -= np.log(uncertainties[j, i])
            loglikelihood -= N * np.log(b)
            # loglikelihood -= N / 2 * np.log(2*np.pi)
    return loglikelihood

def log_likelihood(F_s_obs, spectrum, wl, wl_grid, V_bary, Phi, V_sin_i, model, high_res_params, 
                    high_res_param_names, data_arr=None, data_scale=None, residuals=None, uncertainties=None, Us=None):
    '''
    Return the loglikelihood given the observed flux, Keplerian velocity, and centered system velocity.
    Use this function in a high resolutional rerieval.
    Ndet: number of spectral order.
    Nphi: number of time-resolved phases.
    Npix: number of wavelengths per spectral order.
    Typical values for (Ndet, Nphi, Npix) is (44, 79, 1848).

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
            2D wavelength grid of the data (Ndet x Npix). Typical size ~(44, 1848).
        data_arr (3D np.array of float):
            3D Array representing the top principal components removed data.
            Shape: (Ndet x Nphi x Npix)
        data_scale (3D np.array of float):
            3D Array representing the top principal components of data.
            Shape: (Ndet x Nphi x Npix)
        V_bary (np.array of float):
            Array of time-resolved Earth-star velocity. We have absorbed V_sys into V_bary, so V_sys = V_sys_literature + d_V_sys.
            Shape: (Nphi, )
        Phi (np.array of float):
            Array of time-resolved phases.
            Shpae (Nphi, )
    
    Returns:
        logL_Matteo (float):
            Loglikelihood given by Log(L) = -N/2 Log(s_f^2 - 2R(s) + s_g^2). Equation 9 in Brogi & Line 2019 March.
        logL_Zack (float):
            Loglikelihood given by Log(L) = -N/2 Log(1.0 - CCF^2)). Equation 2 in Brogi & Line 2019 March.
        CCF (float):
            cross correlation value.
    '''
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
        a = 0

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

    # rotational coavolution
    rot_kernel = get_rot_kernel(V_sin_i, wl, W_conv)
    
    F_p_rot = np.convolve(spectrum, rot_kernel, mode='same') # calibrate for planetary rotation

    # instrument profile convolution
    R_instrument = model['R_instrument']
    R = model['R']
    xker = np.arange(-20, 21)
    sigma = (R / R_instrument)/(2.* np.sqrt(2.0 * np.log(2.0)))  # model is right now at R=250K.  IGRINS is at R~45K. We make gaussian that is R_model/R_IGRINS ~ 5.5 
    yker = np.exp(-0.5 * (xker / sigma) ** 2.0)   # instrumental broadening kernel; not understand
    yker /= yker.sum()
    F_p_conv = np.convolve(F_p_rot, yker, mode='same') * a
    cs_p = interpolate.splrep(wl, F_p_conv, s=0.0) # no need to times (R)^2 because F_p, F_s are already observed value on Earth

    if data_arr is not None and data_scale is not None:
        F_s_conv = np.convolve(F_s_obs, yker, mode='same')
        cs_s = interpolate.splrep(wl, F_s_conv, s=0.0)
        return log_likelihood_PCA(V_sys, K_p, dPhi, cs_p, cs_s, wl_grid, data_arr, data_scale, V_bary, Phi)
    elif residuals is not None and uncertainties is not None and Us is not None:
        return log_likelihood_sysrem(V_sys, K_p, dPhi, cs_p, wl_grid, residuals, Us, V_bary, Phi, uncertainties, b)
    else:
        raise Exception('Problem in high res retreival data.')

def sysrem(data_array, iter=15):

    def generate_matrix(data_raw):

        Nphi, Npix = np.shape(data_raw)

        # Create empty matrices for residuals and corresponding errors with the found dimensions such that number of rows correspond to the number of available stars, and the number of columns correspond to each specific epoch:
        residuals = np.zeros((Nphi, Npix))
        stds = np.zeros((Nphi, Npix))

        # Import each of the star files
        for i, sepctrum in enumerate(data_raw):

            # Calculate residuals from the ORIGINAL light curve
            sepctrum_residuals = sepctrum - np.median(sepctrum)

            # For the data points with quality flags != 0,
            # set the errors to a large value
            sepctrum_std = np.copy(sepctrum.std())

            # import the residual and error values into the matrices in the correct position (rows corresponding to stars, columns to epochs)
            residuals[i] = sepctrum_residuals
            stds[i] = sepctrum_std

        return residuals, stds

    residual, stds = generate_matrix(data_array)
    Nphi, Npix = np.shape(residual)

    # This medians.txt file is a 2D list with the first column being the medians
    # of stars' magnitudes at different epochs (the good ones) and their
    # standard deviations, so that they can be plotted against the results after
    # errors are taken out below.
    
    U = np.zeros((Nphi, iter))
    W = np.zeros((Npix, iter))
    for i in range(iter):         # The number of linear systematics to remove
        u = np.zeros(Nphi)
        w = np.ones(Npix)

        # minimize a and c values for a number of iterations, iter
        for _ in range(10):

            # Using the initial guesses for each a value of each epoch, minimize c for each star
            for phi in range(Nphi):
                err_squared = stds[phi] ** 2
                numerator = np.sum(w * residual[phi] / err_squared)
                denominator = np.sum(w**2 / err_squared)
                u[phi] = numerator / denominator

            # Using the c values found above, minimize a for each epoch
            for pix in range(Npix):
                err_squared = stds[:, pix] ** 2
                numerator = np.sum(u * residual[:, pix] / err_squared)
                denominator = np.sum(u**2 / err_squared)
                w[pix] = numerator / denominator

        # Create a matrix for the systematic errors:
        systematic = np.zeros((Nphi, Npix))
        for phi in range(Nphi):
            for pix in range(Npix):
                systematic[phi, pix] = u[phi] * w[pix]

        # Remove the systematic error
        residual = residual - systematic

        U[:, i] = u
        W[:, i] = w

    return residual, U, W

def fast_filter(data, iter=15):

    Ndet, Nphi, Npix = data.shape
    residuals = np.zeros((Ndet, Nphi, Npix))
    Us = np.zeros((Ndet, Nphi, iter))
    Ws = np.zeros((Ndet, Npix, iter))
    
    for i, order in enumerate(data):
        residual, U, W = sysrem(order, iter)
        residuals[i] = residual
        Us[i] = U
        Ws[i] = W
    
    return residuals, Us, Ws

def make_data_cube(data, wl_grid):
    Ndet, Nphi, Npix = data.shape # yup, this again--Norders x Nphases x Npixels
    # SVD/PCA method

    data_scale = np.zeros(data.shape)
    data_arr = np.zeros(data.shape)

    NPC = 4 #change the "4" to whatever. This is the number of PC's to remove

    for i in range(Ndet):
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
    Ndet: number of spectral order.
    Nphi: number of time-resolved phases.
    Npix: number of wavelengths per spectral order.
    Typical values for (Ndet, Nphi, Npix) is (44, 79, 1848).

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
            2D wavelength grid of the data (Ndet x Npix). Typical size ~(44, 1848).
        data_arr (3D np.array of float):
            3D Array representing the top principal components removed data.
            Shape: (Ndet x Nphi x Npix)
        data_scale (3D np.array of float):
            3D Array representing the top principal components of data.
            Shape: (Ndet x Nphi x Npix)
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
    for i in range(Nord):
        order = data_raw[i]
        svd = TruncatedSVD(n_components=NPC, n_iter=4, random_state=42).fit(order)

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

        svd = TruncatedSVD(n_components=NPC, n_iter=4, random_state=42).fit(best_fit)

        uncertainty = svd.transform(best_fit) @ svd.components_
        uncertainties[i] = uncertainty
    return uncertainties