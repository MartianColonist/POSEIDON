from __future__ import absolute_import, unicode_literals, print_function
import os
import numpy as np
from scipy import constants
from numba import jit
from sklearn.decomposition import TruncatedSVD
import cmasher as cmr
import matplotlib.pyplot as plt
import h5py, time, batman, matplotlib
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import minimize


def airtovac(wlA):
    # Convert wavelengths (nm) in air to wavelengths in vaccuum (empirical).
    s = 1e4 / wlA
    n = 1 + (
        0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s**2)
        + 0.0001599740894897 / (38.92568793293 - s**2)
    )
    return wlA * n


def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        data = {}
        for name in f.keys():
            data[name] = f[name][:]

    return data


def read_high_res_data(data_dir, names=None):
    if not names:
        names = os.listdir(data_dir)
    elif isinstance(names, str):
        names = [names]
    data_all = {}
    for name in names:
        processed_data_path = os.path.join(data_dir, name, "data_processed.hdf5")
        data = read_hdf5(processed_data_path)
        data_all[name] = data
    return data_all


def fit_uncertainties(flux, n_components=5, initial_guess=[0.1, 200], Print=True):
    if Print:
        print("Fitting Poisson uncertainties with {} components".format(n_components))
    uncertainties = np.zeros(flux.shape)
    nord = len(flux)
    rebuilt = PCA_rebuild(flux, n_components=n_components)
    residuals = flux - rebuilt

    for i in range(nord):
        # minimizing negative log likelihood is equivalent to maximizing log likelihood
        def neg_likelihood(x):
            a, b = x
            sigma = np.sqrt(a * flux[i] + b)
            loglikelihood = -0.5 * np.sum((residuals[i] / sigma) ** 2) - np.sum(
                np.log(sigma)
            )
            return -loglikelihood

        a, b = minimize(neg_likelihood, initial_guess, method="Nelder-Mead").x
        best_fit = np.sqrt(a * flux[i] + b)
        uncertainties[i] = best_fit

    return PCA_rebuild(uncertainties, n_components=n_components)


def blaze_correction(flux, filter_size, Print=True):
    median_filter_size, gaussian_filter_size = filter_size
    if Print:
        print(
            "Blaze correcting data with median filter size {} and gaussian filter size {}".format(
                median_filter_size, gaussian_filter_size
            )
        )
    blaze = np.zeros(flux.shape)
    nord, nphi, npix = flux.shape
    for i in range(nord):
        order = flux[i]
        median = np.median(order, axis=0)
        order_norm = order / median
        blaze[i] = order_norm

    for i in range(nord):
        for j in range(nphi):
            blaze[i][j] = median_filter(blaze[i][j], size=median_filter_size)

    for i in range(nord):
        for j in range(nphi):
            blaze[i][j] = gaussian_filter1d(blaze[i][j], sigma=gaussian_filter_size)

    flux = flux / blaze

    return flux


def prepare_high_res_data(
    data_dir,
    name,
    spectrum_type,
    method,
    flux,
    wl_grid,
    phi,
    uncertainties=None,
    transit_weight=None,
    V_bary=None,
    pca_ncomp=4,
    sysrem_niter=15,
):
    if spectrum_type == "transmission":
        if transit_weight is None:
            raise Exception(
                "Please provide transit_weight for transmission spectroscopy."
            )

    processed_data_path = os.path.join(data_dir, name, "data_processed.hdf5")

    with h5py.File(processed_data_path, "w") as f:
        print("Creating processed data at {}".format(processed_data_path))
        f.create_dataset("phi", data=phi)
        f.create_dataset("wl_grid", data=wl_grid)
        if V_bary is not None:
            f.create_dataset("V_bary", data=V_bary)

        nord, nphi, npix = flux.shape

        if spectrum_type == "emission":
            if method.lower() == "pca":
                _, residuals = make_data_cube(flux, pca_ncomp)
            elif method.lower() == "sysrem":
                residuals, Us = fast_filter(flux, uncertainties, sysrem_niter)
                Bs = np.zeros((nord, nphi, nphi))
                for i in range(nord):
                    U = Us[i]
                    L = np.diag(1 / np.mean(uncertainties[i], axis=-1))
                    B = U @ np.linalg.pinv(L @ U) @ L
                    Bs[i] = B
                f.create_dataset("Bs", data=Bs)
                f.create_dataset("uncertainties", data=uncertainties)
            f.create_dataset("flux", data=flux)
            f.create_dataset("residuals", data=residuals)

        elif spectrum_type == "transmission":
            if method.lower() == "sysrem":
                median = fit_out_transit_spec(flux, transit_weight, spec="median")
                flux /= median
                uncertainties /= median
                residuals, Us = fast_filter(flux, uncertainties, sysrem_niter)
                Bs = np.zeros((nord, nphi, nphi))
                for i in range(nord):
                    U = Us[i]
                    L = np.diag(1 / np.mean(uncertainties[i], axis=-1))
                    B = U @ np.linalg.pinv(L @ U) @ L
                    Bs[i] = B
                f.create_dataset("Bs", data=Bs)
                f.create_dataset("residuals", data=residuals)
                f.create_dataset("uncertainties", data=uncertainties)
                f.create_dataset("transit_weight", data=transit_weight)

            # # elif method == "NMF":
            # elif method == "sysrem_2020":
            #     residuals, _ = fast_filter(flux_blaze_corrected, uncertainties, niter)
            #     rebuilt = flux_blaze_corrected - residuals
            #     flux_blaze_corrected /= rebuilt
            #     uncertainties /= rebuilt
            #     # residuals = flux_blaze_corrected - np.median(flux_blaze_corrected, axis=2)[:, :, None] # mean normalize the data
            #     residuals = flux_blaze_corrected - 1  # mean normalize the data
            # elif method == "PCA":
            #     median = fit_out_transit_spec(
            #         flux_blaze_corrected, transit_weight, spec="median"
            #     )
            #     flux_blaze_corrected /= median
            #     rebuilt = PCA_rebuild(flux_blaze_corrected, n_components=10)
            #     flux_blaze_corrected /= rebuilt
            #     uncertainties /= rebuilt * median
            #     # residuals = flux_blaze_corrected - np.median(flux_blaze_corrected, axis=2)[:, :, None] # mean normalize the data
            #     residuals = flux_blaze_corrected - 1  # mean normalize the data

    return


def sysrem(data_array, uncertainties, niter=15):
    """
    SYSREM procedure adapted from https://github.com/stephtdouglas/PySysRem, originally used for detrending light curves.

    Use this function in a high resolutional rerieval.
    nord: number of spectral order.
    nphi: number of time-resolved phases.
    npix: number of wavelengths per spectral order.

    Args:
        data_array (2D np.array of float):
            Blaze-corrected data of a single order.
            Shape: (nphi x npix)
        uncertainties (np.array of float):
            Time and wavelength dependent uncertainties obtained from fit_uncertainties.
            Shape: (nphi x npix)
        Niter (int):
            Number of basis vectors to consider.

    Returns:
        residuals (np.array of float):
            2D Array representing the residuals data of a single order after filtering.
            Shape: (nphi x npix)

        U (2D np.array of float):
            Basis vectors obtained from SYSREM of a single vector.
            Shape: (nphi x N_iter + 1)
    """
    data_array = data_array.T
    uncertainties = uncertainties.T
    npix, nphi = data_array.shape

    # Create empty matrices for residuals and corresponding errors with the found dimensions such that number of rows correspond to the number of available stars, and the number of columns correspond to each specific epoch:
    residuals = np.zeros((npix, nphi))

    for i, light_curve in enumerate(data_array):
        # Calculate residuals from the ORIGINAL light curve
        residual = light_curve - np.median(light_curve)
        # import the residual and error values into the matrices in the correct position (rows corresponding to stars, columns to epochs)
        residuals[i] = residual

    U = np.zeros((nphi, niter + 1))

    for i in range(niter):  # The number of linear systematics to remove
        w = np.zeros(npix)
        u = np.ones(nphi)  # Initial guesses

        # minimize a and c values for a number of iterations. a -> u, c -> w
        # each time minimizing equation (1) in Tamuz et al. 2005
        for _ in range(10):
            for pix in range(npix):
                err_squared = uncertainties[pix] ** 2
                numerator = np.sum(u * residuals[pix] / err_squared)
                denominator = np.sum(u**2 / err_squared)
                w[pix] = numerator / denominator

            for phi in range(nphi):
                err_squared = uncertainties[:, phi] ** 2
                numerator = np.sum(w * residuals[:, phi] / err_squared)
                denominator = np.sum(w**2 / err_squared)
                u[phi] = numerator / denominator

        # Create a matrix for the systematic errors:
        systematic = np.zeros((npix, nphi))
        for pix in range(npix):
            for phi in range(nphi):
                systematic[pix, phi] = u[phi] * w[pix]

        # Remove the systematic error
        residuals = residuals - systematic

        U[:, i] = u

    U[:, -1] = np.ones(nphi)  # This corresponds to Gibson 2021. page 4625.

    return residuals.T, U


def fast_filter(flux, uncertainties, niter=15, Print=True):
    """
    TODO: Add docstrings.
    Use this function in a high resolutional rerieval.

    Args:

    Returns:
        residuals (3D np.array of float):
            The residuals data of a single order after filtering.
            Shape: (nord x nphi x npix)

        Us (3D np.array of float):
            Basis vectors obtained from SYSREM of a single vector.
            Shape: (nord x nphi x N_iter+1)
    """
    if Print:
        print("Filtering out systematics using SYSREM with {} iterations".format(niter))
    nord, nphi, npix = flux.shape
    residuals = np.zeros((nord, nphi, npix))
    Us = np.zeros((nord, nphi, niter + 1))

    for i, order in enumerate(flux):
        stds = uncertainties[i]
        residual, U = sysrem(order, stds, niter)
        residuals[i] = residual
        Us[i] = U

    return residuals, Us


def make_data_cube(data, n_components=4):
    nord, nphi, npix = data.shape

    # SVD/PCA method
    data_scale = PCA_rebuild(data, n_components=n_components)
    data_arr = data - data_scale

    for i in range(nord):
        A = data_arr[i]
        # sigma clipping sort of--it really doesn't make a yuge difference.
        sigma = np.std(A)
        median = np.median(A)
        loc = np.where(A > 3 * sigma + median)
        A[loc] = 0  # *0.+20*sig
        loc = np.where(A < -3 * sigma + median)
        A[loc] = 0  # *0.+20*sig
        data_arr[i] = A

    return data_scale, data_arr


def PCA_rebuild(flux, n_components=5):
    nord, nphi, npix = flux.shape
    rebuilt = np.zeros_like(flux)
    for i in range(nord):
        order = flux[i]
        svd = TruncatedSVD(n_components=n_components).fit(order)
        rebuilt[i] = svd.transform(order) @ svd.components_
    return rebuilt


def fit_out_transit_spec(flux, transit_weight, degree=2, spec="median", Print=True):
    nord, nphi, npix = flux.shape
    spec_fit = np.zeros_like(flux)
    out_transit = transit_weight == 1

    for i in range(nord):
        # construct a mean spectrum
        if spec == "mean":
            mean_spec = np.mean(flux[i][out_transit], axis=0)
        elif spec == "median":
            mean_spec = np.median(flux[i][out_transit], axis=0)
        else:
            raise Exception('Error: Please select "mean", "median"')

        for j in range(nphi):
            spec_fit[i, j, :] = mean_spec

    return spec_fit


def get_RV_range(Kp_range, Vsys_range, phi):
    RV_min = min(
        [
            np.min(Kp_range * np.sin(2 * np.pi * phi[i])) + np.min(Vsys_range)
            for i in range(len(phi))
        ]
    )

    RV_max = max(
        [
            np.max(Kp_range * np.sin(2 * np.pi * phi[i])) + np.max(Vsys_range)
            for i in range(len(phi))
        ]
    )

    RV_range = np.arange(RV_min, RV_max + 1)
    return RV_range


def cross_correlate(
    Kp_range, Vsys_range, RV_range, wl, planet_spectrum, data, Print=True
):
    if Print:
        time0 = time.time()
    uncertainties = data["uncertainties"]
    # uncertainties = np.ones_like(uncertainties)
    residuals = data["residuals"]
    phi = data["phi"]
    wl_grid = data["wl_grid"]

    try:
        V_bary = data["V_bary"]
    except:
        V_bary = np.zeros_like(phi)
    if "transit_weight" in data.keys():
        spectrum_type = "transmission"
        transit_weight = data["transit_weight"]
        max_transit_depth = np.max(1 - transit_weight)
    else:
        spectrum_type = "emission"

    nord, nphi, npix = residuals.shape
    CCF_Kp_Vsys = np.zeros((len(Kp_range), len(Vsys_range)))

    nRV = len(RV_range)
    CCF_phase_RV = np.zeros((nphi, nRV))
    models_shifted = np.zeros((nRV, nord, npix))
    for RV_i, RV in enumerate(RV_range):
        # Looping through each order and computing total log-L by summing logLs for each obvservation/order
        for ord_i in range(nord):
            wl_slice = wl_grid[ord_i]  # Cropped wavelengths
            delta_lambda = RV * 1e3 / constants.c
            wl_shifted = wl * (1.0 + delta_lambda)
            F_p = np.interp(wl_slice, wl_shifted, planet_spectrum)
            models_shifted[RV_i, ord_i] = F_p  # choose not to filter

    # negative of transmission spectrum gives absorption. Adding or multiplying constant does not change the CCF
    if spectrum_type == "emission":
        m = models_shifted
    elif spectrum_type == "transmission":
        m = -models_shifted

    for phi_i in range(nphi):
        for RV_i in range(nRV):
            f = residuals[:, phi_i, :]
            CCF = np.sum(f[:, :] * m[RV_i, :, :] / uncertainties[:, phi_i, :] ** 2)
            CCF_phase_RV[phi_i, RV_i] += CCF

    if spectrum_type == "transmission":
        CCF_phase_RV = (1 - transit_weight[:, None]) * CCF_phase_RV

    for Kp_i, Kp in enumerate(Kp_range):
        for phi_i in range(nphi):
            RV = Kp * np.sin(2 * np.pi * phi[phi_i]) + Vsys_range + V_bary[phi_i]
            CCF_Kp_Vsys[Kp_i] += np.interp(RV, RV_range, CCF_phase_RV[phi_i])
    if Print:
        time1 = time.time()
        print("Cross correlation took {} seconds".format(time1 - time0))
    return CCF_Kp_Vsys, CCF_phase_RV


def plot_CCF_phase_RV(
    phi,
    RV_range,
    CCF_phase_RV,
    species,
    plot_label=False,
    save_path=None,
    cmap=cmr.ember,
):
    for i in range(len(CCF_phase_RV)):
        CCF_phase_RV[i] = CCF_phase_RV[i] - np.mean(CCF_phase_RV[i])
        CCF_phase_RV[i] /= np.std(CCF_phase_RV[i])
    fig, ax = plt.subplots(figsize=(10.667, 3), constrained_layout=False)
    im = ax.imshow(
        CCF_phase_RV,
        extent=[RV_range.min(), RV_range.max(), phi.min(), phi.max()],
        aspect="auto",
        interpolation="bilinear",
        cmap=cmap,
        origin="lower",
    )
    if plot_label:
        ax.text(
            0.05,
            0.3,
            species,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="white",
            fontsize=32,
        )

    cbar = plt.colorbar(im)
    # ax.plot(
    #     np.arange(-180, -50),
    #     (np.arange(-180, -50) + 100) / (200) / (2 * np.pi)
    #     + 0.4,  # phi start from 90 degrees. sin(phi-90) -90 = -phi
    #     "--",
    #     color="red",
    #     alpha=0.5,
    # )
    # plot(V_sys_arr[loc[1]], phi[loc[0]], "xk", ms=7)
    # axis([V_sys_arr.min(), V_sys_arr.max(), phi.min(), phi.max()])
    ax.set_xlabel(r"$\rm{V_p}$ (km/s)")
    ax.set_ylabel(r"$\phi$", rotation=0, labelpad=20)
    ax.set_title(r"$\Delta$ CCF ($\sigma$)")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    return


def find_nearest_idx(array, value):
    """
    Function that will find the index of the value in the array nearest a given value

        Input: array, number

        Output: index of value in array closest to that number
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_CCF_Kp_Vsys(
    Kp_range,
    Vsys_range,
    CCF_Kp_Vsys,
    species,
    Kp,
    Vsys=0,
    RM_mask_size=10,
    plot_slice=False,
    plot_label=False,
    savefig=False,
    file_path=None,
    cmap=cmr.ember,
):
    # Expected value
    CCF_Kp_Vsys = CCF_Kp_Vsys - np.mean(CCF_Kp_Vsys)
    idx = find_nearest_idx(Vsys_range, Vsys)
    mask = np.ones(len(Vsys_range), dtype=bool)
    mask[idx - RM_mask_size : idx + RM_mask_size] = False
    stdev = np.std(CCF_Kp_Vsys[:, mask])
    maxx = (CCF_Kp_Vsys / stdev).max()

    loc = np.where(CCF_Kp_Vsys / stdev == maxx)

    colors = cmr.take_cmap_colors(cmap, 10, cmap_range=(0.1, 0.9), return_fmt="hex")
    if plot_slice:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8, 10),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [8, 2]},
        )
        ax1 = axes[0]
        ax2 = axes[1]
        idx = find_nearest_idx(Kp_range, Kp)
        slicee = CCF_Kp_Vsys[idx] / stdev
        ax2.plot(Vsys_range, slicee, c=colors[5])
        ax2.axis(
            [
                np.min(Vsys_range),
                np.max(Vsys_range),
                1.1 * slicee.min(),
                1.1 * slicee.max(),
            ]
        )
        ax2.set_xlabel(r"$\Delta$V$_{sys}$(km/s)")
        ax2.set_ylabel(r"$\Delta$ CCF ($\sigma$)")
        ax2.set_title("Slice at K$_{p}$ = " + str(Kp) + " km/s")
        ax2.axvline(x=Vsys, ls="--", color="black")
    else:
        fig, ax1 = plt.subplots(figsize=(8, 8), constrained_layout=False)
    im = ax1.imshow(
        CCF_Kp_Vsys / stdev,
        extent=[Vsys_range.min(), Vsys_range.max(), Kp_range.min(), Kp_range.max()],
        aspect=len(Vsys_range) / len(Kp_range),
        interpolation="bilinear",
        cmap=cmap,
        origin="lower",
    )
    if plot_label:
        ax1.text(
            0.05,
            0.15,
            species,
            ha="left",
            va="top",
            transform=ax1.transAxes,
            color="white",
            fontsize=32,
        )
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    ax1.axvline(x=Vsys, color="white", ls="--", lw=2)
    ax1.axhline(y=Kp, color="white", ls="--", lw=2)
    ax1.plot(Vsys_range[loc[1]], Kp_range[loc[0]], "xk", ms=15, mew=3)
    ax1.set_xlabel("$\Delta$V$_{sys}$ (km/s)")
    ax1.set_ylabel(r"K$_{p}$ (km/s)")
    ax1.set_title(r"$\Delta$ CCF ($\sigma$)")

    if savefig:
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0.1)

    return


def loglikelihood_PCA(V_sys, K_p, d_phi, a, wl, planet_spectrum, star_spectrum, data):
    """
    Perform the loglikelihood calculation using Principal Component Analysis (PCA). Based on M. Line 2019.
    nord: number of spectral order.
    nphi: number of time-resolved phases.
    npix: number of wavelengths per spectral order.

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
                    Shape: (nord x nphi x npix)
                data_scale (3D np.array of float):
                    3D Array representing the top principal components of data.
                    Shape: (nord x nphi x npix)
                V_bary (np.array of float):
                    Array of time-resolved Earth-star velocity. We have absorbed V_sys into V_bary, so V_sys = V_sys_literature + d_V_sys.
                    Shape: (nphi, )
                phi (np.array of float):
                    Array of time-resolved phases.
                    Shpae (nphi, )

    Returns:
        loglikelihood_sum (float):
            Log-likelihood value.
        CCF_sum (float):
            Cross-correlation value.
    """

    residuals = data["residuals"]
    flux = data["flux"]
    data_scale = flux - residuals
    phi = data["phi"]
    wl_grid = data["wl_grid"]

    try:
        V_bary = data["V_bary"]
    except:
        V_bary = np.zeros_like(phi)

    nord, nphi, npix = residuals.shape

    # Time-resolved total radial velocity
    radial_velocity_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (phi + d_phi))
    # V_sys is an additive term around zero
    delta_lambda_p = radial_velocity_p * 1e3 / constants.c  # delta lambda, for shifting

    K_s = 0.3229
    radial_velocity_s = (
        V_sys + V_bary - K_s * np.sin(2 * np.pi * phi) * 0
    )  # Velocity of the star is very small compared to planet's velocity and it's already be corrected
    delta_lambda_s = radial_velocity_s * 1e3 / constants.c  # delta lambda, for shifting

    loglikelihood_sum = 0
    CCF_sum = 0
    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for j in range(nord):  # Nord = 44 This takes 2.2 seconds to complete
        wl_slice = wl_grid[j]  # Cropped wavelengths
        F_p_F_s = np.zeros((nphi, npix))  # "shifted" model spectra array at each phase
        for i in range(nphi):  # This for loop takes 0.025 seconds Nphi = 79
            wl_shifted_p = wl_slice * (1.0 - delta_lambda_p[i])
            F_p = np.interp(wl_shifted_p, wl, planet_spectrum)
            wl_shifted_s = wl_slice * (1.0 - delta_lambda_s[i])
            F_s = np.interp(wl_shifted_s, wl, star_spectrum)
            F_p_F_s[i, :] = F_p / F_s

        model_injected = (1 + F_p_F_s) * data_scale[j, :]

        svd = TruncatedSVD(n_components=4, n_iter=4, random_state=42).fit(
            model_injected
        )
        models_filtered = model_injected - (
            svd.transform(model_injected) @ svd.components_
        )  # 0.008 s
        # svd.transform gives data matrix in reduced dimension (79, 5). svd.components gives the first n_components right singular vectors (5, npix)
        # Original data minus PCA-ed data is equivalent to doing np.linalg.svd, setting first n_components components to zero.

        for i in range(nphi):  # This loop takes 0.001 second
            model_filtered = models_filtered[i] * a
            model_filtered -= model_filtered.mean()  # mean subtracting here...
            m2 = model_filtered.dot(model_filtered)
            planet_signal = residuals[j, i]  # already mean-subtracted
            f2 = planet_signal.dot(planet_signal)
            R = model_filtered.dot(planet_signal)  # cross-covariance
            CCF = R / np.sqrt(m2 * f2)  # cross-correlation
            CCF_sum += CCF
            loglikelihood_sum += (
                -0.5 * npix * np.log((m2 + f2 - 2.0 * R) / npix)
            )  # Equation 9 in paper

    return loglikelihood_sum, CCF_sum


def loglikelihood_sysrem(
    V_sys, K_p, d_phi, a, b, wl, planet_spectrum, data, star_spectrum=None
):
    """
    Perform the loglikelihood calculation using SysRem. Based on N. Gibson 2021.
    nord: number of spectral order.
    nphi: number of time-resolved phases.
    npix: number of wavelengths per spectral order.

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
                    Shape: (nord x nphi x npix)
                Bs (list of 2D np.array of float):
                    A list of basis vectors returned by fast_filter.
                    Shape: (nord x nphi x npix)
                phi (np.array of float):
                    Array of time-resolved phases.
                    Shape (nphi, )
                transit_weight (np.array of float):
                    Transit model of the planet. A value of 1 is out of transit, and 0 is full transit.
                    Shape (nphi, )
                uncertainties (None or 3D np.array of float):
                    Time and wavelength dependent uncertainties obtained from fit_uncertainties.
                    Can be None only if b is also None.
                    Shape: (nord x nphi x npix)

    Returns:
        Loglikelihood (float):
            Loglikelihood value.
    """

    wl_grid = data["wl_grid"]
    residuals = data["residuals"]
    Bs = data["Bs"]
    phi = data["phi"]
    if star_spectrum is None:  # transmission
        transit_weight = data["transit_weight"]
        max_transit_depth = np.max(1 - transit_weight)
    else:  # emission
        flux = data["flux"]
        flux_star = flux - residuals

    try:
        V_bary = data["V_bary"]
    except:
        V_bary = np.zeros_like(phi)

    uncertainties = data.get("uncertainties")  # in case we want to null uncertainties

    nord, nphi, npix = residuals.shape

    N = residuals.size

    # Time-resolved total radial velocity
    radial_velocity_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (phi + d_phi))
    radial_velocity_s = V_sys + V_bary + 0

    delta_lambda_p = radial_velocity_p * 1e3 / constants.c
    delta_lambda_s = radial_velocity_s * 1e3 / constants.c

    # Initializing loglikelihood
    loglikelihood_sum = 0
    if b is not None:
        loglikelihood_sum -= N * np.log(b)

    # Looping through each order and computing total log-L by summing logLs for each obvservation/order

    for i in range(nord):
        wl_slice = wl_grid[i]  # Cropped wavelengths

        models_shifted = np.zeros(
            (nphi, npix)
        )  # "shifted" model spectra array at each phase

        for j in range(nphi):
            wl_shifted_p = wl_slice * (1.0 - delta_lambda_p[j])
            F_p = np.interp(wl_shifted_p, wl, planet_spectrum * a)
            if star_spectrum is None:
                models_shifted[j] = (1 - transit_weight[j]) / max_transit_depth * (
                    -F_p
                ) + 1
                models_shifted[j] /= np.median(
                    models_shifted[j]
                )  # divide by the median over wavelength
            else:
                wl_shifted_s = wl_slice * (1.0 - delta_lambda_s[j])
                F_s = np.interp(wl_shifted_s, wl, star_spectrum)
                models_shifted[j] = F_p / F_s * flux_star[i, j]

        B = Bs[i]
        models_filtered = models_shifted - B @ models_shifted  # filter the model

        if b is not None:
            for j in range(nphi):
                m = models_filtered[j] / uncertainties[i, j]
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -0.5 * (m2 + f2 - 2.0 * CCF) / (b**2)
                loglikelihood_sum += loglikelihood

        elif uncertainties is not None:  # nulled b
            for j in range(nphi):
                m = models_filtered[j] / uncertainties[i, j]
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -npix / 2 * np.log((m2 + f2 - 2.0 * CCF) / npix)
                loglikelihood_sum += loglikelihood

        else:  # nulled b and uncertainties
            for j in range(nphi):
                m = models_filtered[j]
                m2 = m.dot(m)
                f = residuals[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -npix / 2 * np.log((m2 + f2 - 2.0 * CCF) / npix)
                loglikelihood_sum += loglikelihood

    # loglikelihood -= np.sum(np.log(uncertainties))
    # loglikelihood -= N / 2 * np.log(2*np.pi)          (These two terms are normalization)

    return loglikelihood_sum


def loglikelihood_high_res(
    wl,
    planet_spectrum,
    star_spectrum,
    data,
    spectrum_type,
    method,
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

    K_p = high_res_params[np.where(high_res_param_names == "K_p")[0][0]]
    V_sys = high_res_params[np.where(high_res_param_names == "V_sys")[0][0]]

    if "log_alpha" in high_res_param_names:
        a = 10 ** high_res_params[np.where(high_res_param_names == "log_alpha")[0][0]]
    elif "a" in high_res_param_names:
        a = high_res_params[np.where(high_res_param_names == "a")[0][0]]
    else:
        a = 1

    if "Delta_phi" in high_res_param_names:
        d_phi = high_res_params[np.where(high_res_param_names == "Delta_phi")[0][0]]
    else:
        d_phi = 0

    if "W_conv" in high_res_param_names:
        W_conv = high_res_params[np.where(high_res_param_names == "W_conv")[0][0]]
    else:
        W_conv = None

    if "b" in high_res_param_names:
        b = high_res_params[np.where(high_res_param_names == "b")[0][0]]
    else:
        b = None  # Nulling b

    if spectrum_type == "emission":
        if W_conv is not None:
            F_p = gaussian_filter1d(planet_spectrum, W_conv)
            F_s = gaussian_filter1d(star_spectrum, W_conv)
        else:
            F_p = planet_spectrum
            F_s = star_spectrum
        loglikelihood = 0
        for key in data.keys():
            if method == "sysrem":
                loglikelihood += loglikelihood_sysrem(
                    V_sys, K_p, d_phi, a, b, wl, F_p, data[key], F_s
                )
            elif method == "PCA":
                loglikelihood, _ = loglikelihood_PCA(
                    V_sys, K_p, d_phi, a, wl, F_p, F_s, data[key]
                )
            else:
                raise Exception(
                    "Emission spectroscopy only supports sysrem and PCA for now."
                )
        return loglikelihood

    elif spectrum_type == "transmission":
        if method != "sysrem":
            raise Exception(
                "Transmission spectroscopy only supports fast filtering with sysrem (Gibson et al. 2022)."
            )
        if W_conv is not None:
            F_p = gaussian_filter1d(planet_spectrum, W_conv)
        else:
            F_p = planet_spectrum
        loglikelihood = 0
        for key in data.keys():
            # wl_vacuum = airtovac(wl * 1e4) / 1e4
            loglikelihood += loglikelihood_sysrem(
                V_sys, K_p, d_phi, a, b, wl, F_p, data[key]
            )
        return loglikelihood
    else:
        raise Exception("Spectrum type should be 'emission' or 'transmission'.")


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


def remove_outliers(wl_grid, flux):
    nord, nphi, npix = flux.shape
    cleaned_flux = flux.copy()
    # Process each order separately
    noutliers = 0
    for i in range(nord):
        # Fit a 10th order polynomial to the residual spectrum
        for j in range(nphi):
            coeffs = np.polynomial.polynomial.polyfit(wl_grid[i], flux[i, j], 10)
            fitted_spectra = np.polynomial.polynomial.polyval(wl_grid[i], coeffs)
            std = np.std(flux[i, j] - fitted_spectra)
            # Identify and replace 5σ outliers in the residual spectrum
            outliers = np.abs(flux[i, j] - fitted_spectra) > 5 * std

            cleaned_flux[i, j, outliers] = np.interp(
                wl_grid[i, outliers], wl_grid[i, ~outliers], flux[i, j, ~outliers]
            )
            noutliers += np.sum(outliers)
    print("{} outliers removed from a total of {} pixels".format(noutliers, flux.size))

    return cleaned_flux


def transit_model(R_p, R_s, a, phi):
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = 0  # time of inferior conjunction
    params.per = 1  # orbital period, dummy value
    params.rp = R_p / R_s  # planet radius (in units of stellar radii)
    params.a = a / R_s  # semi-major axis (in units of stellar radii)
    params.inc = 90.0  # orbital inclination (in degrees)
    params.ecc = 0.0  # eccentricity
    params.w = 90.0  # longitude of periastron (in degrees)
    params.limb_dark = "quadratic"  # limb darkening model
    params.u = [0, 0]  # limb darkening coefficients
    t = phi * params.per  # times at which to calculate light curve
    m = batman.TransitModel(params, t)  # initializes model
    transit_weight = m.light_curve(params)
    return transit_weight


def make_injection_data(
    data,
    data_dir,
    name,
    wl,
    planet_spectrum,
    K_p,
    V_sys,
    method,
    a=None,
    continuum=None,
    W_conv=None,
    star_spectrum=None,
):
    residuals = data["residuals"]
    flux = data["flux"]
    wl_grid = data["wl_grid"]
    phi = data["phi"]

    nord, nphi, npix = residuals.shape
    if continuum is not None and a is not None:
        planet_spectrum = (planet_spectrum - continuum) * a + continuum
    if W_conv is not None:
        planet_spectrum = gaussian_filter1d(planet_spectrum, W_conv)
    emission = star_spectrum is not None

    if emission:
        spectrum_type = "emission"
        if W_conv is not None:
            star_spectrum = gaussian_filter1d(star_spectrum, W_conv)
        transit_weight = None
    else:
        spectrum_type = "transmission"
        transit_weight = data["transit_weight"]
        max_transit_depth = np.max(1 - transit_weight)
    # Time-resolved total radial velocity
    radial_velocity = V_sys + K_p * np.sin(2 * np.pi * phi)
    delta_lambda = radial_velocity * 1e3 / constants.c

    F_p_F_s = np.zeros((nord, nphi, npix))
    F_p = np.zeros((nord, nphi, npix))

    for i in range(nord):
        wl_slice = wl_grid[i].copy()  # Cropped wavelengths
        for j in range(nphi):
            wl_shifted_p = wl_slice * (1.0 - delta_lambda[j])
            if emission:
                F_p_F_s[i, j, :] = np.interp(
                    wl_shifted_p, wl, planet_spectrum
                ) / np.interp(wl_slice, wl, star_spectrum)
            else:
                F_p[i, j, :] = (
                    -np.interp(wl_shifted_p, wl, planet_spectrum)
                    * (1 - transit_weight[j])
                    / max_transit_depth
                    + 1
                )

    if emission:
        data_injected = (1 + F_p_F_s) * (flux - residuals)
    else:
        data_injected = F_p * flux

    if method.lower() == "pca":
        uncertainties = None
    elif method.lower() == "sysrem":
        uncertainties = fit_uncertainties(
            data_injected, initial_guess=[0.1, np.mean(data_injected)]
        )

    prepare_high_res_data(
        data_dir,
        name,
        spectrum_type,
        method,
        data_injected,
        wl_grid,
        phi,
        uncertainties,
        transit_weight,
    )

    return
