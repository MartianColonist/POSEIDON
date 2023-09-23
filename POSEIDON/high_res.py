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
    """
    Get rotational kernel given V sin(i) and wavelength grid of the forward model.

    Args:
        V_sin_i (float):
            Projected rotational velocity of a star. The component of the rotational velocity along the line of sight, which is esponsible for broadening.
        wl (np.array of float):
            Wavelength grid of the forward model.
        W_conv (int):
            Width of the rotational kernel.
    """

    dRV = np.mean(2.0 * (wl[1:] - wl[0:-1]) / (wl[1:] + wl[0:-1])) * 2.998e5
    n_ker = int(W_conv)
    half_n_ker = (n_ker - 1) // 2
    rot_ker = np.zeros(n_ker)
    for ii in range(n_ker):
        ik = ii - half_n_ker
        x = ik * dRV / V_sin_i
        if np.abs(x) < 1.0:
            y = np.sqrt(1 - x**2)
            rot_ker[ii] = y
    rot_ker /= rot_ker.sum()

    return rot_ker


def loglikelihood_PCA(V_sys, K_p, d_phi, a, wl, planet_spectrum, star_spectrum, data):
    """
    Perform the loglikelihood calculation using Principal Component Analysis (PCA). Based on M. Line 2019.
    N_order: number of spectral order.
    N_phi: number of time-resolved phases.
    N_wl: number of wavelengths per spectral order.

    Args:
        V_sys (float):
            The system velocity (km/s) at which we do the loglikelihood calculation.
        K_p (float):
            The Keplerian velocity (km/s) at which we do the loglikelihood calculation.
        d_phi (float):
            Phase offset.
        a (float):
            Scale parameter a.
        wl (np.array of float):
            Wavelength at which the model is defined.
        planet_spectrum (np.array of float):
            Observed flux of the planet (at distance = 1 pc). Should be broadened in the same way as the star_spectrum.
        star_spectrum (np.array of float):
            Observed flux of the star (at distance = 1 pc). Should be broadened in the same way as the star_spectrum.
        data (dict):
            Data dictionary read using utility.read_high_res_data. All values should be prepared beforehand.
            Has the following key-value pairss:
                data_arr (3D np.array of float):
                    3D Array representing the top principal components removed data.
                    Shape: (N_order x N_phi x N_wl)
                data_scale (3D np.array of float):
                    3D Array representing the top principal components of data.
                    Shape: (N_order x N_phi x N_wl)
                V_bary (np.array of float):
                    Array of time-resolved Earth-star velocity. We have absorbed V_sys into V_bary, so V_sys = V_sys_literature + d_V_sys.
                    Shape: (N_phi, )
                phi (np.array of float):
                    Array of time-resolved phases.
                    Shpae (N_phi, )

    Returns:
        loglikelihood_sum (float):
            Log-likelihood value.
        CCF_sum (float):
            Cross-correlation value.
    """

    data_arr = data["data_arr"]
    data_scale = data["data_scale"]
    V_bary = data["V_bary"]
    phi = data["phi"]
    wl_grid = data["wl_grid"]

    N_order, N_phi, N_wl = data_arr.shape

    # Time-resolved total radial velocity
    RV_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (phi + d_phi))
    # V_sys is an additive term around zero
    dl_p = RV_p * 1e3 / constants.c  # delta lambda, for shifting

    K_s = 0.3229
    RV_s = (
        V_sys + V_bary - K_s * np.sin(2 * np.pi * phi)
    ) * 0  # Velocity of the star is very small compared to planet's velocity and it's already be corrected
    dl_s = RV_s * 1e3 / constants.c  # delta lambda, for shifting

    loglikelihood_sum = 0
    CCF_sum = 0
    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for j in range(N_order):  # Nord = 44 This takes 2.2 seconds to complete
        wl_slice = wl_grid[j].copy()  # Cropped wavelengths
        F_p_F_s = np.zeros((N_phi, N_wl))  # "shifted" model spectra array at each phase
        for i in range(N_phi):  # This for loop takes 0.025 seconds Nphi = 79
            wl_shifted_p = wl_slice * (1.0 - dl_p[i])
            F_p = np.interp(wl_shifted_p, wl, planet_spectrum)
            wl_shifted_s = wl_slice * (1.0 - dl_s[i])
            F_s = np.interp(wl_shifted_s, wl, star_spectrum)
            F_p_F_s[i, :] = F_p / F_s

        model_injected = (1 + F_p_F_s) * data_scale[j, :]

        svd = TruncatedSVD(n_components=4, n_iter=4, random_state=42).fit(
            model_injected
        )
        models_filtered = model_injected - (
            svd.transform(model_injected) @ svd.components_
        )  # 0.008 s
        # svd.transform gives data matrix in reduced dimension (79, 5). svd.components gives the first n_components right singular vectors (5, N_wl)
        # Original data minus PCA-ed data is equivalent to doing np.linalg.svd, setting first n_components components to zero.

        for i in range(N_phi):  # This loop takes 0.001 second
            model_filtered = models_filtered[i] * a
            model_filtered -= model_filtered.mean()  # mean subtracting here...
            m2 = model_filtered.dot(model_filtered)
            planet_signal = data_arr[j, i]  # already mean-subtracted
            f2 = planet_signal.dot(planet_signal)
            R = model_filtered.dot(planet_signal)  # cross-covariance
            CCF = R / np.sqrt(m2 * f2)  # cross-correlation
            CCF_sum += CCF
            loglikelihood_sum += (
                -0.5 * N_wl * np.log((m2 + f2 - 2.0 * R) / N_wl)
            )  # Equation 9 in paper

    return loglikelihood_sum, CCF_sum


def loglikelihood_sysrem(radial_velocity, d_phi, a, b, wl, planet_spectrum, data):
    """
    Perform the loglikelihood calculation using SysRem. Based on N. Gibson 2021.
    N_order: number of spectral order.
    N_phi: number of time-resolved phases.
    N_wl: number of wavelengths per spectral order.

    Args:
        V_sys (float):
            The system velocity (km/s) at which we do the loglikelihood calculation.
        K_p (float):
            The Keplerian velocity (km/s) at which we do the loglikelihood calculation.
        d_phi (float):
            Phase offset.
        a (float):
            Scale parameter for spectrum.
        b (float or None):
            Scale parameter for noise. If None, calculate loglikelihood using nulled b (b that maximizes logL).
        wl (np.array of float):
            Wavelength at which the model is defined.
        planet_spectrum (np.array of float):
            Observed flux of the planet (at distance = 1 pc). Should be broadened in the same way as the star_spectrum.

        data (dict):
            Data dictionary read using utility.read_high_res_data. All values should be prepared beforehand.
            Has the following key-value pairs:
                residuals (3D np.array of float):
                    3D Array representing the residuals data after filtering.
                    Shape: (N_order x N_phi x N_wl)
                Bs (list of 2D np.array of float):
                    A list of basis vectors returned by fast_filter.
                    Shape: (N_order x N_phi x N_wl)
                phi (np.array of float):
                    Array of time-resolved phases.
                    Shape (N_phi, )
                transit_weight (np.array of float):
                    Transit model of the planet. A value of 1 is out of transit, and 0 is full transit.
                    Shape (N_phi, )
                uncertainties (None or 3D np.array of float):
                    Time and wavelength dependent uncertainties obtained from fit_uncertainties.
                    Can be None only if b is also None.
                    Shape: (N_order x N_phi x N_wl)

    Returns:
        Loglikelihood (float):
            Loglikelihood value.
    """

    wl_grid = data["wl_grid"]
    residuals = data["residuals"]
    Bs = data["Bs"]
    phi = data["phi"]
    transit_weight = data["transit_weight"]
    uncertainties = data.get("uncertainties")  # in case we want to null uncertainties

    N_order, N_phi, N_wl = residuals.shape

    N = residuals.size

    delta_lambda = radial_velocity * 1e3 / constants.c  # delta lambda, for shifting

    # Initializing loglikelihood
    loglikelihood_sum = 0
    CCF_sum = 0

    max_transit_depth = np.max(1 - transit_weight)
    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for i in range(N_order):
        wl_slice = wl_grid[i]  # Cropped wavelengths

        models_shifted = np.zeros(
            (N_phi, N_wl)
        )  # "shifted" model spectra array at each phase

        for j in range(N_phi):
            wl_shifted = wl_slice * (1.0 - delta_lambda[j])
            # wl_shifted_p = wl_slice * np.sqrt((1.0 - dl_p[j]) / (1 + dl_p[j]))
            F_p = np.interp(wl_shifted, wl, planet_spectrum)

            # Fp = interpolate.splev(wl_shifted_p, cs_p, der=0) # linear interpolation, einsum
            models_shifted[j] = (1 - transit_weight[j]) / max_transit_depth * (-F_p) + 1

        # divide by the median over wavelength to mimic blaze correction
        models_shifted = (models_shifted.T / np.median(models_shifted, axis=1)).T

        B = Bs[i]
        models_filtered = models_shifted - B @ models_shifted  # filter the model

        if b is not None:
            for j in range(N_phi):
                m = models_filtered[j] / uncertainties[i, j] * a
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -0.5 * (m2 + f2 - 2.0 * CCF) / (b**2)
                loglikelihood_sum += loglikelihood
                CCF_sum += CCF
        elif uncertainties is not None:  # nulled b
            for j in range(N_phi):
                m = models_filtered[j] / uncertainties[i, j] * a
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -N_wl / 2 * np.log((m2 + f2 - 2.0 * CCF) / N_wl)
                loglikelihood_sum += loglikelihood
                CCF_sum += CCF
        else:  # nulled b and uncertainties
            for j in range(N_phi):
                m = models_filtered[j]
                m2 = m.dot(m)
                f = residuals[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -N_wl / 2 * np.log((m2 + f2 - 2.0 * CCF) / N_wl)
                loglikelihood_sum += loglikelihood
                CCF_sum += CCF

    if b is not None and uncertainties is not None:
        loglikelihood_sum -= N * np.log(b)

    # loglikelihood -= np.sum(np.log(uncertainties))
    # loglikelihood -= N / 2 * np.log(2*np.pi)          (These two terms are normalization)

    return loglikelihood_sum, CCF_sum


def loglikelihood_high_res(
    wl,
    planet_spectrum,
    star_spectrum,
    data,
    model,
    high_res_params,
    high_res_param_names,
):
    """
    Return the loglikelihood given the observed flux, Keplerian velocity, and centered system velocity.
    Should only use this function in a high resolutional rerieval.

    Args:
        wl (np.array of float):
            Wavelength at which the model is defined.
        planet_spectrum (np.array of float):
            Observed flux of the planet (at distance = 1 pc). Should be broadened in the same way as the star_spectrum.
        star_spectrum (np.array of float):
            Observed flux of the star (at distance = 1 pc). Should be broadened in the same way as the star_spectrum.
        data (dict):
            Data dictionary read using utility.read_high_res_data. All values should be prepared beforehand.
            For more information, see code spec for loglikelihood_PCA or loglikelihood_sysrem.
        model (dict):
            Dictionary containing properties of the model.

    Returns:
        loglikelihood (float):
            Loglikelihood calculated based on which filtering method (specified in data['method']).
    """

    method = data["method"]
    spectrum_type = data["spectrum_type"]

    if "K_p" in high_res_param_names:
        K_p = high_res_params[np.where(high_res_param_names == "K_p")[0][0]]
    else:
        K_p = model["K_p"]

    if "V_sys" in high_res_param_names:
        V_sys = high_res_params[np.where(high_res_param_names == "V_sys")[0][0]]
    else:
        V_sys = model["V_sys"]

    if "a" in high_res_param_names:
        a = high_res_params[np.where(high_res_param_names == "a")[0][0]]
    else:
        a = model["a"]

    if "d_phi" in high_res_param_names:
        d_phi = high_res_params[np.where(high_res_param_names == "d_phi")[0][0]]
    else:
        d_phi = 0

    if "W_conv" in high_res_param_names:
        W_conv = high_res_params[np.where(high_res_param_names == "W_conv")[0][0]]
    else:
        W_conv = model.get("W_conv")

    if "b" in high_res_param_names:
        b = high_res_params[np.where(high_res_param_names == "b")[0][0]]
    else:
        b = model.get("b")  # Set a definite value or in case we want to null b

    if method is "pca" and spectrum_type is "emission":
        # instrument profile convolution
        R_instrument = model["R_instrument"]
        R = model["R"]
        V_sin_i = data["V_sin_i"]
        rot_kernel = get_rot_kernel(V_sin_i, wl, W_conv)
        F_p_rot = np.convolve(
            planet_spectrum, rot_kernel, mode="same"
        )  # calibrate for planetary rotation
        xker = np.arange(-20, 21)
        sigma = (R / R_instrument) / (
            2 * np.sqrt(2.0 * np.log(2.0))
        )  # model is right now at R=250K.  IGRINS is at R~45K. We make gaussian that is R_model/R_IGRINS ~ 5.5
        yker = np.exp(
            -0.5 * (xker / sigma) ** 2.0
        )  # instrumental broadening kernel; TODO: Understand this.
        yker /= yker.sum()
        F_p_conv = np.convolve(F_p_rot, yker, mode="same")
        F_s_conv = np.convolve(
            star_spectrum, yker, mode="same"
        )  # no need to times (R)^2 because F_p, F_s are already observed value on Earth
        loglikelihood, _ = loglikelihood_PCA(
            V_sys, K_p, d_phi, a, wl, F_p_conv, F_s_conv, data
        )
        return loglikelihood

    elif method is "sysrem" and spectrum_type is "transmission":
        if W_conv is not None:
            # np.convolve use smaller kernel. Apply filter to the spectrum. And multiply by scale factor a.
            planet_spectrum = gaussian_filter1d(planet_spectrum, W_conv)
        # Time-resolved total radial velocity
        radial_velocity = V_sys + K_p * np.sin(2 * np.pi * (data["phi"] + d_phi))
        # V_sys is an additive term around zero. Data should be in rest frame of star.
        loglikelihood, _ = loglikelihood_sysrem(
            radial_velocity, d_phi, a, b, wl, planet_spectrum, data
        )
        return loglikelihood
    else:
        raise Exception("Problem with high res retreival data.")


def sysrem(data_array, stds, N_iter=15):
    """
    SYSREM procedure adapted from https://github.com/stephtdouglas/PySysRem, originally used for detrending light curves.

    Use this function in a high resolutional rerieval.
    N_order: number of spectral order.
    N_phi: number of time-resolved phases.
    N_wl: number of wavelengths per spectral order.

    Args:
        data_array (2D np.array of float):
            Blaze-corrected data of a single order.
            Shape: (N_phi x N_wl)
        stds (np.array of float):
            Time and wavelength dependent uncertainties obtained from fit_uncertainties.
            Shape: (N_phi x N_wl)
        Niter (int):
            Number of basis vectors to consider.

    Returns:
        residuals (np.array of float):
            2D Array representing the residuals data of a single order after filtering.
            Shape: (N_phi x N_wl)

        U (2D np.array of float):
            Basis vectors obtained from SYSREM of a single vector.
            Shape: (N_phi x N_iter + 1)
    """
    data_transpose = data_array.T
    N_wl, N_phi = data_transpose.shape

    # Create empty matrices for residuals and corresponding errors with the found dimensions such that number of rows correspond to the number of available stars, and the number of columns correspond to each specific epoch:
    residuals = np.zeros((N_wl, N_phi))

    median_list = []
    # Import each of the star files
    for i, wl_channel in enumerate(data_transpose):
        median_list.append(np.median(wl_channel))

        # Calculate residuals from the ORIGINAL light curve
        channel_residual = wl_channel - np.median(wl_channel)
        # channel_residual = wl_channel

        # import the residual and error values into the matrices in the correct position (rows corresponding to stars, columns to epochs)
        residuals[i] = channel_residual

    N_wl, N_phi = np.shape(residuals)

    # This medians.txt file is a 2D list with the first column being the medians
    # of stars' magnitudes at different epochs (the good ones) and their
    # standard deviations, so that they can be plotted against the results after
    # errors are taken out below.

    U = np.zeros((N_phi, N_iter + 1))

    for i in range(N_iter):  # The number of linear systematics to remove
        w = np.zeros(N_wl)
        u = np.ones(N_phi)

        # minimize a and c values for a number of iterations, Niter
        for _ in range(10):
            # Using the initial guesses for each a value of each epoch, minimize c for each star
            for pix in range(N_wl):
                err_squared = stds.T[pix] ** 2
                numerator = np.sum(u * residuals[pix] / err_squared)
                denominator = np.sum(u**2 / err_squared)
                w[pix] = numerator / denominator

            # Using the c values found above, minimize a for each epoch
            for phi in range(N_phi):
                err_squared = stds.T[:, phi] ** 2
                numerator = np.sum(w * residuals[:, phi] / err_squared)
                denominator = np.sum(w**2 / err_squared)
                u[phi] = numerator / denominator

        # Create a matrix for the systematic errors:
        systematic = np.zeros((N_wl, N_phi))
        for pix in range(N_wl):
            for phi in range(N_phi):
                systematic[pix, phi] = u[phi] * w[pix]

        # Remove the systematic error
        residuals = residuals - systematic

        U[:, i] = u

    # for i in range(len(residuals)):
    #     residuals[i] += median_list[i]

    U[:, -1] = np.ones(N_phi)

    return residuals.T, U


def fast_filter(data, uncertainties, N_iter=15):
    """
    TODO: Add docstrings.
    Use this function in a high resolutional rerieval.

    Args:

    Returns:
        residuals (3D np.array of float):
            The residuals data of a single order after filtering.
            Shape: (N_order x N_phi x N_wl)

        Us (3D np.array of float):
            Basis vectors obtained from SYSREM of a single vector.
            Shape: (N_order x N_phi x N_iter+1)
    """
    N_order, N_phi, N_wl = data.shape
    residuals = np.zeros((N_order, N_phi, N_wl))
    Us = np.zeros(
        (N_order, N_phi, N_iter + 1)
    )  # TODO: could turn into np.ones for clarity. This corresponds to Gibson 2021. page 4625.

    for i, order in enumerate(data):
        stds = uncertainties[i]
        residual, U = sysrem(order, stds, N_iter)
        residuals[i] = residual
        Us[i] = U

    return residuals, Us


def make_data_cube(data, wl_grid):
    N_order, N_phi, N_wl = data.shape  # yup, this again--Norders x Nphases x Npixels
    # SVD/PCA method

    data_scale = np.zeros(data.shape)
    data_arr = np.zeros(data.shape)

    NPC = 4  # change the "4" to whatever. This is the number of PC's to remove

    for i in range(N_order):
        # taking only first four vectors, reconstructiong, and saving
        u, s, vh = np.linalg.svd(data[i], full_matrices=False)  # decompose
        s[NPC:] = 0
        W = np.diag(s)
        A = np.dot(u, np.dot(W, vh))
        data_scale[i] = A

        # removing first four vectors...this is the 'processesed data'
        u, s, vh = np.linalg.svd(
            data[i], full_matrices=False
        )  # decompose--not sure why I did it again....guess you don't really need this line
        s[0:NPC] = 0
        W = np.diag(s)
        A = np.dot(u, np.dot(W, vh))

        # sigma clipping sort of--it really doesn't make a yuge difference.

        sigma = np.std(A)
        median = np.median(A)
        loc = np.where(A > 3 * sigma + median)
        A[loc] = 0  # *0.+20*sig
        loc = np.where(A < -3 * sigma + median)
        A[loc] = 0  # *0.+20*sig

        data_arr[i] = A

    return data_scale, data_arr


from scipy.optimize import minimize


def fit_uncertainties(data_raw, NPC=5):
    uncertainties = np.zeros(data_raw.shape)
    residuals = np.zeros(data_raw.shape)
    N_order = len(data_raw)
    mask = data_raw == 0
    for i in range(N_order):
        order = data_raw[i]
        svd = TruncatedSVD(n_components=NPC, n_iter=15, random_state=42).fit(order)

        residual = order - (svd.transform(order) @ svd.components_)
        residuals[i] = residual

    for i in range(N_order):

        def fun(x):
            a, b = x
            sigma = np.sqrt(a * data_raw[i] + b)
            loglikelihood = -0.5 * np.sum((residuals[i] / sigma) ** 2) - np.sum(
                np.log(sigma)
            )
            return -loglikelihood

        a, b = minimize(fun, [0.5, 200], method="Nelder-Mead").x
        print(a, b)
        best_fit = np.sqrt(a * data_raw[i] + b)

        svd = TruncatedSVD(n_components=NPC, n_iter=15, random_state=42).fit(best_fit)

        uncertainty = svd.transform(best_fit) @ svd.components_
        uncertainties[i] = uncertainty
    uncertainties[mask] = 1e7
    return uncertainties


def fit_uncertainties_and_remove_outliers(data_raw, NPC=5):
    uncertainties = np.zeros(data_raw.shape)
    residuals = np.zeros(data_raw.shape)
    data_cleaned = data_raw.copy()
    N_order = len(data_raw)
    mean = data_raw.mean()
    std = data_raw.std()

    mask = (data_raw - mean) > 5 * std

    for i in range(N_order):
        order = data_raw[i]
        svd = TruncatedSVD(n_components=NPC, n_iter=15, random_state=42).fit(order)
        PCA_model = svd.transform(order) @ svd.components_
        residual = order - PCA_model
        residuals[i] = residual
        data_cleaned[i][mask[i]] = PCA_model[mask[i]]

    for i in range(N_order):

        def fun(x):
            a, b = x
            sigma = np.sqrt(a * data_cleaned[i] + b)
            loglikelihood = -0.5 * np.sum((residuals[i] / sigma) ** 2) - np.sum(
                np.log(sigma)
            )
            return -loglikelihood

        a, b = minimize(fun, [1, 1], method="Nelder-Mead").x

        best_fit = np.sqrt(a * data_cleaned[i] + b)

        svd = TruncatedSVD(n_components=NPC, n_iter=15, random_state=42).fit(best_fit)

        uncertainty = svd.transform(best_fit) @ svd.components_
        uncertainties[i] = uncertainty

    # uncertainties[mask] = 1e7
    return data_cleaned, uncertainties


def cross_correlate(Kp_arr, Vsys_arr, wl, planet_spectrum, data):
    uncertainties = data["uncertainties"]
    residuals = data["residuals"]
    phi = data["phi"]
    Bs = data["Bs"]
    wl_grid = data["wl_grid"]
    transit_weight = data["transit_weight"]
    max_transit_depth = np.max(1 - transit_weight)

    N_order, N_phi, N_wl = residuals.shape
    loglikelihood_array_final = np.zeros((len(Kp_arr), len(Vsys_arr)))
    CCF_array_final = np.zeros((len(Kp_arr), len(Vsys_arr)))
    for i in range(N_phi):
        # residual = residuals[:, i, :]
        # uncertainty = uncertainties[:, i, :]
        RV_range = np.arange(
            np.min(Kp_arr * np.sin(2 * np.pi * phi[i])) + np.min(Vsys_arr),
            np.max(Kp_arr * np.sin(2 * np.pi * phi[i])) + np.max(Vsys_arr) + 1,
        )

        loglikelihoods = np.zeros_like(RV_range)
        CCFs = np.zeros_like(RV_range)
        for j, RV in enumerate(RV_range):
            # Looping through each order and computing total log-L by summing logLs for each obvservation/order
            for k in range(N_order):
                wl_slice = wl_grid[k]  # Cropped wavelengths
                delta_lambda = RV * 1e3 / constants.c
                models_shifted = np.zeros(
                    (N_phi, N_wl)
                )  # "shifted" model spectra array at each phase

                for l in range(N_phi):
                    wl_shifted = wl_slice * (1.0 - delta_lambda)
                    # wl_shifted_p = wl_slice * np.sqrt((1.0 - dl_p[j]) / (1 + dl_p[j]))
                    F_p = np.interp(wl_shifted, wl, planet_spectrum)

                    # Fp = interpolate.splev(wl_shifted_p, cs_p, der=0) # linear interpolation, einsum
                    models_shifted[l] = (1 - transit_weight[l]) / max_transit_depth * (
                        -F_p
                    ) + 1

                # divide by the median over wavelength to mimic blaze correction
                models_shifted = (
                    models_shifted.T / np.median(models_shifted, axis=1)
                ).T

                B = Bs[k]
                models_filtered = models_shifted - B @ models_shifted

                # m = models_filtered[i] / uncertainties[k, i] * a
                # m2 = m.dot(m)
                # f = residuals[k, i] / uncertainties[k, i]
                # f2 = f.dot(f)
                # CCF = f.dot(m)
                # loglikelihood = -0.5 * (m2 + f2 - 2.0 * CCF) / (b**2)
                # loglikelihood_sum += loglikelihood
                # CCF_sum += CCF

                a = 1
                m = models_filtered[i] / uncertainties[k, i] * a
                m2 = m.dot(m)
                f = residuals[k, i] / uncertainties[k, i]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -N_wl / 2 * np.log((m2 + f2 - 2.0 * CCF) / N_wl)
                loglikelihoods[j] += loglikelihood
                CCFs[j] += CCF

        N_Kp = len(Kp_arr)
        N_Vsys = len(Vsys_arr)
        loglikelihood_array = np.zeros((N_Kp, N_Vsys))
        CCF_array = np.zeros((N_Kp, N_Vsys))

        for j, Kp in enumerate(Kp_arr):
            RV = Kp * np.sin(2 * np.pi * phi[i]) + Vsys_arr
            loglikelihood_array_final[j] += np.interp(RV, RV_range, loglikelihoods)
            CCF_array_final[j] += np.interp(RV, RV_range, CCFs)

    cross_correlation_result = [
        Kp_arr,
        Vsys_arr,
        loglikelihood_array_final,
        CCF_array_final,
    ]

    return cross_correlation_result
