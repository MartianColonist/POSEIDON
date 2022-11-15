from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
import pickle
from scipy import constants
from numba import jit
from astropy.io import fits
from scipy import interpolate
from .core import create_star, create_planet, define_model, make_atmosphere, read_opacities, wl_grid_constant_R, wl_grid_line_by_line, compute_spectrum
from .constants import R_Sun
from .visuals import plot_stellar_flux
from .constants import R_J, M_J
import numpy as np
from spectres import spectres
from sklearn.decomposition import TruncatedSVD

@jit
def get_rot_kernel(V_sin_i, wl_s):
    dRV = np.mean(2.0*(wl_s[1: ]-wl_s[0: -1])/(wl_s[1: ]+wl_s[0: -1]))*2.998E5
    n_ker = 401
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


def log_likelihood_PCA(V_sys, K_p, scale, cs_p, cs_s, wl_grid, data_arr, data_scale, V_bary, Phi):

    K_s = 0.3229

    #Kstar=(Mp/Mstar*9.55E-4)*Kp  #this is mass planet/mass star
    Ndet, Nphi, Npix = data_arr.shape

    I = np.ones(Npix)
    N = Npix # np.array([Npix])
    
    # Time-resolved total radial velocity
    RV_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * Phi)  # Vsys is an additive term around zero   
    dl_p = RV_p * 1e3 / constants.c
    RV_s = (V_sys + V_bary - K_s * np.sin(2 * np.pi * Phi))*0  # Velocity of the star is very small compared to planet's velocity and it's already be corrected
    dl_s = RV_s * 1e3 / constants.c
    # Initializing log-likelihoods and CCFs
    logL_Matteo = 0
    logL_Zuck = 0
    CCF = 0
    # Looping through each phase and computing total log-L by summing logLs for each obvservation/phase
    for j in range(Ndet): # Ndet = 44 This takes 2.2 seconds to complete
        wl_slice = wl_grid[j, ].copy() # Cropped wavelengths    
        Fp_Fs = np.zeros((Nphi, Npix))  # "shifted" model spectra array at each phase
        for i in range(Nphi): # This for loop takes 0.025 seconds Nphi = 79
            wl_shifted_p = wl_slice * (1.0 - dl_p[i])
            Fp = interpolate.splev(wl_shifted_p, cs_p, der=0) * scale
            wl_shifted_s = wl_slice * (1.0 - dl_s[i])
            Fs = interpolate.splev(wl_shifted_s, cs_s, der=0)
            Fp_Fs[i, :] = Fp/Fs

        model_injected = (1 + Fp_Fs) * data_scale[j, :]  # 1??+fp/fstar is same as (fstar+fp)/fstar..tell "stretches" by transmittance
        
        # I think of this as:
        # data_scale contains the first four principle component, data_arr contains the rest. data_arr is telluric subtracted, 
        # but part of the planet signal is also subtracted along the way. Therefore, we inject planet signal into data_scale, perform
        # svd. In this way, the injected planet signal is partly subtracted again.
        # Basically it's accounting for the transmittance of the atmosphere at each wavelength.
        # Been staring at this for a while. I'll just buy it for now.

        # fastest SVD among numpy, scipy, jax, sklearn
        svd = TruncatedSVD(n_components=4, n_iter=4, random_state=42).fit(model_injected)
        model_injected_PCs_removed = model_injected - (svd.transform(model_injected) @ svd.components_)
        # svd.transform gives data matrix in reduced dimension (79, 5). svd.components gives the first n_components right singular vectors (5, N_wl)  
        # Original data minus PCA-ed data is equivalent to doing np.linalg.svd, setting first n_components components to zero.

        for i in range(Nphi): # This loop takes 0.001 second
            gVec = model_injected_PCs_removed[i, ].copy()
            gVec -= (gVec.dot(I))/float(Npix)  #mean subtracting here...
            sg2 = (gVec.dot(gVec))/float(Npix)
            fVec = data_arr[j, i, ].copy() # already mean-subtracted
            sf2 = (fVec.dot(fVec))/Npix
            R = (fVec.dot(gVec))/Npix # cross-covariance
            CC = R/np.sqrt(sf2*sg2) # cross-correlation. Normalized, must be <= 1. Could view as dot product between to unit vector
            CCF += CC
            logL_Matteo += (-0.5*N * np.log(sf2 + sg2 - 2.0*R)) # Equation 9 in paper
            logL_Zuck += (-0.5*N * np.log(1.0 - CC**2.0))	

    return logL_Matteo, logL_Zuck, CCF # returning CCF and logL values


def cross_correlate(F_s_obs, F_p_obs, wl, K_p_arr, V_sys_arr, wl_grid, data_arr, data_scale, V_bary, Phi):

    scale = 1.0

    #loading data (read_data in utility.py)

    # K_p = 192.06  # orbital velocity of planet; this is used to center trial values of K_p
    # K_star = (M_p/M_star*9.55e-4)*K_p

    # rotational coonvolutiono
    V_sin_i = 4.5
    rot_kernel = get_rot_kernel(V_sin_i, wl)
    F_p_rot = np.convolve(F_p_obs, rot_kernel, mode='same') # calibrate for planetary rotation

    # instrument profile convolustion
    xker = np.arange(-20, 21)
    sigma = 5.5/(2.* np.sqrt(2.0*np.log(2.0)))  # nominal
    yker = np.exp(-0.5 * (xker / sigma)**2.0)   # instrumental broadening kernel; not understand yet
    yker /= yker.sum()
    F_p_conv = np.convolve(F_p_rot, yker, mode='same')
    F_s_conv = np.convolve(F_s_obs, yker, mode='same')

    # cs_p = interpolate.splrep(wl, Fp_conv*(R_p*0.1)**2, s=0.0) # Commented out because why R_p * 0.1? --Roger
    # cs_s = interpolate.splrep(wl, Fstar_conv*(R_star)**2, s=0.0) 
    cs_p = interpolate.splrep(wl, F_p_conv, s=0.0) # no need to times (R)^2 because F_p_obs, F_s_obs are already observed value on Earth
    cs_s = interpolate.splrep(wl, F_s_conv, s=0.0)

    log_L_M_arr = np.zeros((len(K_p_arr), len(V_sys_arr)))
    log_L_Z_arr = np.zeros((len(K_p_arr), len(V_sys_arr)))
    CCF_arr = np.zeros((len(K_p_arr), len(V_sys_arr)))

    
    for i in range(len(K_p_arr)):
        for j in range(len(V_sys_arr)):
            log_L_M, log_L_Z, CCF1 = log_likelihood_PCA(V_sys_arr[j], K_p_arr[i], scale, cs_p, cs_s, wl_grid, data_arr, data_scale, V_bary, Phi)
            log_L_M_arr[i, j] = log_L_M
            log_L_Z_arr[i, j] = log_L_Z
            CCF_arr[i, j] = CCF1

    return log_L_M_arr, log_L_Z_arr, CCF_arr




def log_likelihood(F_s_obs, F_p_obs, wl, K_p, V_sys, wl_grid, data_arr, data_scale, V_bary, Phi):

    scale = 1.0

    # K_p = 192.06  # orbital velocity of planet; this is used to center trial values of K_p
    # K_star = (M_p/M_star*9.55e-4)*K_p

    # rotational coavolution
    V_sin_i = 4.5
    rot_kernel = get_rot_kernel(V_sin_i, wl)
    F_p_rot = np.convolve(F_p_obs, rot_kernel, mode='same') # calibrate for planetary rotation

    # instrument profile convolution
    xker = np.arange(-20, 21)
    sigma = 5.5/(2.* np.sqrt(2.0*np.log(2.0)))  # nominal
    yker = np.exp(-0.5 * (xker / sigma)**2.0)   # instrumental broadening kernel; not understand yet
    yker /= yker.sum()
    F_p_conv = np.convolve(F_p_rot, yker, mode='same')
    F_s_conv = np.convolve(F_s_obs, yker, mode='same')

    cs_p = interpolate.splrep(wl, F_p_conv, s=0.0) # no need to times (R)^2 because F_p, F_s are already observed value on Earth
    cs_s = interpolate.splrep(wl, F_s_conv, s=0.0)

    logL_M, logL_Z, CCF1 = log_likelihood_PCA(V_sys, K_p, scale, cs_p, cs_s, wl_grid, data_arr, data_scale, V_bary, Phi)
    
    return logL_M, logL_Z, CCF1