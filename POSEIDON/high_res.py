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
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.ndimage import gaussian_filter1d, median_filter
import cmasher as cmr
import colormaps as cmaps
import matplotlib.pyplot as plt
import h5py
import os
from scipy.optimize import minimize
import time


def read_hdf5(file_path):
    with h5py.File(file_path, "r") as f:
        data = {}
        for name in f.keys():
            data[name] = f[name][:]

    return data


def add_high_res_data(
    data_dir, name, flux, wl_grid, phi, transit_weight, overwrite=False
):
    os.makedirs(os.path.join(data_dir, name), exist_ok=True)
    file_path = os.path.join(data_dir, name, "data_raw.hdf5")
    if os.path.exists(file_path):
        if overwrite:
            print("Overwriting data at {}".format(file_path))
            f = h5py.File(file_path, "w")
        else:
            print(
                "Raw data file already exists. Choose another name for observation or set overwrite=True to continue."
            )
            return
    else:
        print("Creating raw data at {}".format(file_path))
        f = h5py.File(file_path, "w")

    f.create_dataset("flux", data=flux)
    f.create_dataset("wl_grid", data=wl_grid)
    f.create_dataset("phi", data=phi)
    f.create_dataset("transit_weight", data=transit_weight)

    f.close()

    return


def prepare_high_res_data(
    data_dir,
    name,
    spectrum_type,
    method,
    niter=15,
    n_PC=5,
    filter_size=(15, 50),
    overwrite=False,
    Print=True,
):
    raw_data_path = os.path.join(data_dir, name, "data_raw.hdf5")
    with h5py.File(raw_data_path, "r") as f:
        flux = f["flux"][:]
        wl_grid = f["wl_grid"][:]
        phi = f["phi"][:]
        transit_weight = f["transit_weight"][:]
        norder, nphi, npix = flux.shape

    processed_data_path = os.path.join(data_dir, name, "data_processed.hdf5")

    if os.path.exists(processed_data_path):
        if overwrite:
            print("Overwriting data at {}".format(processed_data_path))
            f = h5py.File(processed_data_path, "w")
        else:
            print("Processed data file already exists. Appending on current data file.")
            print("You can set overwrite=True to overwrite everything.")
            f = h5py.File(processed_data_path, "a")
            for key in f.keys():
                if key not in ["uncertainties", "flux_blaze_corrected"]:
                    del f[key]
    else:
        print("Creating processed data at {}".format(processed_data_path))
        f = h5py.File(processed_data_path, "w")

    f.create_dataset("flux", data=flux)
    f.create_dataset("wl_grid", data=wl_grid)
    f.create_dataset("phi", data=phi)
    f.create_dataset("transit_weight", data=transit_weight)

    f.attrs["spectrum_type"] = spectrum_type
    f.attrs["method"] = method

    if "uncertainties" in f.keys():
        uncertainties = f["uncertainties"][:]
    else:
        uncertainties = fit_uncertainties(flux, n_PC, Print=Print)
        f.create_dataset("uncertainties", data=uncertainties)

    if "flux_blaze_corrected" in f.keys():
        flux_blaze_corrected = f["flux_blaze_corrected"][:]
    else:
        flux_blaze_corrected = blaze_correction(flux, filter_size, Print=Print)
        f.create_dataset("flux_blaze_corrected", data=flux_blaze_corrected)

    if spectrum_type == "emission":
        if method == "PCA":
            pass

    elif spectrum_type == "transmission":
        if method == "sysrem_2022":
            median = fit_spec(
                flux_blaze_corrected, spec="median"
            )  # TODO: check order, phase, wl
            flux_blaze_corrected /= median
            uncertainties /= median
            residuals, Us = fast_filter(flux_blaze_corrected, uncertainties, niter)

            Bs = np.zeros((norder, nphi, nphi))
            for i in range(norder):
                U = Us[i]
                L = np.diag(1 / np.mean(uncertainties[i], axis=-1))
                B = U @ np.linalg.pinv(L @ U) @ L
                Bs[i] = B
            f.create_dataset("Bs", data=Bs)
        elif method == "sysrem_2020":
            residuals, _ = fast_filter(flux_blaze_corrected, uncertainties, niter)
            rebuilt = flux_blaze_corrected - residuals
            flux_blaze_corrected /= rebuilt
            uncertainties /= rebuilt
            # residuals = flux_blaze_corrected - np.median(flux_blaze_corrected, axis=2)[:, :, None] # mean normalize the data
            residuals = flux_blaze_corrected - 1  # mean normalize the data
        elif method == "PCA":
            median = fit_spec(
                flux_blaze_corrected, spec="median"
            )  # TODO: check order, phase, wl
            flux_blaze_corrected /= median
            rebuilt = PCA_rebuild(flux_blaze_corrected, n_components=10)
            flux_blaze_corrected /= rebuilt
            uncertainties /= rebuilt * median
            # residuals = flux_blaze_corrected - np.median(flux_blaze_corrected, axis=2)[:, :, None] # mean normalize the data
            residuals = flux_blaze_corrected - 1  # mean normalize the data

        f.create_dataset("residuals", data=residuals)
        f.create_dataset("uncertainties_processed", data=uncertainties)

    f.close()
    return


def read_high_res_data(data_dir, names=None, retrieval=True):
    if not names:
        names = os.listdir(data_dir)
    elif isinstance(names, str):
        names = [names]
    data_all = {}
    for name in names:
        processed_data_path = os.path.join(data_dir, name, "data_processed.hdf5")
        data = read_hdf5(processed_data_path)
        if retrieval:
            data_all[name] = {}
            data_all[name]["residuals"] = data["residuals"]
            data_all[name]["uncertainties_processed"] = data["uncertainties_processed"]
            data_all[name]["phi"] = data["phi"]
            data_all[name]["wl_grid"] = data["wl_grid"]
            data_all[name]["transit_weight"] = data["transit_weight"]
            data_all[name]["Bs"] = data["Bs"]
        else:
            data_all[name] = data
    return data_all


def fit_uncertainties(flux, n_components=5, initial_guess=[0.5, 200], Print=True):
    if Print:
        print("Fitting Poisson uncertainties with {} components".format(n_components))
    uncertainties = np.zeros(flux.shape)
    norder = len(flux)
    rebuilt = PCA_rebuild(flux, n_components=n_components)
    residuals = flux - rebuilt

    for i in range(norder):
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
    norder, nphi, npix = flux.shape
    for i in range(norder):
        order = flux[i]
        median = np.median(order, axis=0)
        order_norm = order / median
        blaze[i] = order_norm

    for i in range(norder):
        for j in range(nphi):
            blaze[i][j] = median_filter(blaze[i][j], size=median_filter_size)

    for i in range(norder):
        for j in range(nphi):
            blaze[i][j] = gaussian_filter1d(blaze[i][j], sigma=gaussian_filter_size)

    flux = flux / blaze

    return flux


def sysrem(data_array, uncertainties, niter=15):
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
        uncertainties (np.array of float):
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
            Shape: (N_order x N_phi x N_wl)

        Us (3D np.array of float):
            Basis vectors obtained from SYSREM of a single vector.
            Shape: (N_order x N_phi x N_iter+1)
    """
    if Print:
        print("Filtering out systematics using SYSREM with {} iterations".format(niter))
    norder, nphi, npix = flux.shape
    residuals = np.zeros((norder, nphi, npix))
    Us = np.zeros((norder, nphi, niter + 1))

    for i, order in enumerate(flux):
        stds = uncertainties[i]
        residual, U = sysrem(order, stds, niter)
        residuals[i] = residual
        Us[i] = U

    return residuals, Us


def PCA_rebuild(flux, n_components=5):
    norder, nphi, npix = flux.shape
    rebuilt = np.zeros_like(flux)
    for i in range(norder):
        order = flux[i]
        svd = TruncatedSVD(n_components=n_components).fit(order)
        rebuilt[i] = svd.transform(order) @ svd.components_
    return rebuilt


def fit_spec(flux, degree=2, spec="median", Print=True):
    if Print:
        print("Fitting out {} spectrum from each exposure".format(spec))
    norder, nphi, npix = flux.shape
    spec_fit = np.zeros_like(flux)

    for i in range(norder):
        # construct a mean spectrum
        if spec == "mean":
            mean_spec = np.mean(flux[i], axis=0)
        elif spec == "median":
            mean_spec = np.median(flux[i], axis=0)
        else:
            raise Exception('Error: Please select "mean", "median"')

        for j in range(nphi):
            # fit the mean spectrum to each exposure (typically with a second order polynomial)
            mean_spec_poly_coeffs = np.polynomial.polynomial.polyfit(
                mean_spec,
                flux[i, j, :],
                degree,
            )
            # reconstruct that polynomial
            polynomial = np.polynomial.polynomial.polyval(
                mean_spec, mean_spec_poly_coeffs
            ).T
            # fit as a polynomial of the mean spectrum and AFTER fit a slope
            spec_fit[i, j, :] = polynomial

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
    uncertainties = data["uncertainties_processed"]
    residuals = data["residuals"]
    phi = data["phi"]
    wl_grid = data["wl_grid"]
    transit_weight = data["transit_weight"]
    max_transit_depth = np.max(1 - transit_weight)

    norder, nphi, npix = residuals.shape
    CCF_Kp_Vsys = np.zeros((len(Kp_range), len(Vsys_range)))

    nRV = len(RV_range)
    CCF_phase_RV = np.zeros((nphi, nRV))
    models_shifted = np.zeros((nRV, norder, npix))
    for RV_i, RV in enumerate(RV_range):
        # Looping through each order and computing total log-L by summing logLs for each obvservation/order
        for order_i in range(norder):
            wl_slice = wl_grid[order_i]  # Cropped wavelengths
            delta_lambda = RV * 1e3 / constants.c
            wl_shifted = wl * (1.0 + delta_lambda)
            F_p = np.interp(wl_slice, wl_shifted, planet_spectrum)
            models_shifted[RV_i, order_i] = F_p  # choose not to filter

    m = -models_shifted  # negative of transmission spectrum gives absorption
    m -= np.mean(m, axis=1)[:, None]  # mean normalize the model
    residuals -= np.mean(residuals, axis=2)[:, :, None]  # mean normalize the data

    for phi_i in range(nphi):
        for RV_i in range(nRV):
            f = residuals[:, phi_i, :]
            CCF = np.sum(f[:, :] * m[RV_i, :] / uncertainties[:, phi_i, :] ** 2)
            CCF_phase_RV[phi_i, RV_i] += CCF

    CCF_phase_RV = transit_weight[:, None] * CCF_phase_RV
    CCFs = np.sum(CCF_phase_RV, axis=0)
    for Kp_i, Kp in enumerate(Kp_range):
        RV = Kp * np.sin(2 * np.pi * phi[phi_i]) + Vsys_range
        CCF_Kp_Vsys[Kp_i] += np.interp(RV, RV_range, CCFs)
    if Print:
        time1 = time.time()
        print("Cross correlation took {} seconds".format(time1 - time0))
    return CCF_Kp_Vsys, CCF_phase_RV


def plot_CCF_Kp_Vsys(
    Kp_range,
    Vsys_range,
    CCF_Kp_Vsys,
    species,
    Kp,
    Vsys=0,
    RM_mask_size=10,
    plot_label=False,
    savefig=False,
    output_path=None,
):
    # Expected value
    CCF_Kp_Vsys = CCF_Kp_Vsys - np.mean(CCF_Kp_Vsys)
    idx = find_nearest_idx(Vsys_range, Vsys)
    mask = np.ones(len(Vsys_range), dtype=bool)
    mask[idx - RM_mask_size : idx + RM_mask_size] = False
    stdev = np.std(CCF_Kp_Vsys[:, mask])
    maxx = (CCF_Kp_Vsys / stdev).max()

    loc = np.where(CCF_Kp_Vsys / stdev == maxx)
    fig, ax = plt.subplots(figsize=(17, 17), constrained_layout=True)
    im = ax.imshow(
        CCF_Kp_Vsys / stdev,
        extent=[Vsys_range.min(), Vsys_range.max(), Kp_range.min(), Kp_range.max()],
        aspect=len(Vsys_range) / len(Kp_range),
        interpolation="bilinear",
        cmap=cmaps.cividis,
        origin="lower",
    )
    if plot_label:
        ax.text(
            0.1,
            0.15,
            species,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="white",
            fontsize=60,
        )

    cbar = plt.colorbar(im, shrink=0.8)
    plt.axvline(x=Vsys, color="white", ls="--", lw=2)
    plt.axhline(y=Kp, color="white", ls="--", lw=2)
    plt.plot(Vsys_range[loc[1]], Kp_range[loc[0]], "xk", ms=20, mew=4)
    ax.set_xlabel("$\Delta$V$_{sys}$ (km/s)")
    ax.set_ylabel(r"K$_{p}$ (km/s)")
    ax.set_title(r"$\Delta$ CCF ($\sigma$)")
    if savefig:
        plt.savefig(output_path + s + "_CCF_Kp_Vsys.png")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(17, 5), constrained_layout=True)
    idx = find_nearest_idx(Kp_range, Kp)

    slicee = CCF_Kp_Vsys[idx]
    ax.plot(Vsys_range, slicee)
    ax.axis(
        [
            np.min(Vsys_range),
            np.max(Vsys_range),
            1.1 * slicee.min(),
            1.1 * slicee.max(),
        ]
    )
    ax.set_xlabel(r"$\Delta$V$_{sys}$(km/s)")
    ax.set_ylabel(r"$\Delta$ CCF")
    ax.set_title("Slice at K$_{p}$ = " + str(Kp) + " km/s")
    ax.axvline(x=Vsys, ls="--", color="black")
    if savefig:
        plt.savefig(output_path + s + "_CCF_slice.png")
    plt.show()
    plt.close()


def find_nearest_idx(array, value):
    """
    Function that will find the index of the value in the array nearest a given value

        Input: array, number

        Output: index of value in array closest to that number
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_CCF_phase_RV(
    phi,
    RV_range,
    CCF_phase_RV,
    species,
    plot_label=False,
    savefig=False,
    output_path=None,
):
    for i in range(len(CCF_phase_RV)):
        CCF_phase_RV[i] = CCF_phase_RV[i] - np.mean(CCF_phase_RV[i])
        # CCF_per_phase[i] = CCF_per_phase[i] / stdev
    stdev = np.std(CCF_phase_RV)
    maxx = (CCF_phase_RV).max()
    fig, ax = plt.subplots(figsize=(17, 5), constrained_layout=True)
    im = ax.imshow(
        CCF_phase_RV,
        extent=[RV_range.min(), RV_range.max(), phi.min(), phi.max()],
        aspect="auto",
        interpolation="bilinear",
        cmap=cmaps.cividis,
        origin="lower",
    )
    if plot_label:
        ax.text(
            0.1,
            0.15,
            species,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="white",
            fontsize=60,
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
    ax.set_xlabel("Planet Radial Velocity (km/s)")
    ax.set_ylabel(r"$\phi$", rotation=0, labelpad=20)
    ax.set_title(r"$\Delta$ Cross Correlation Coefficient")
    if savefig:
        plt.savefig(output_path + species + "_CCF_phase.png")
    plt.show()
    plt.close()


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
        wl_slice = wl_grid[j]  # Cropped wavelengths
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


def loglikelihood_sysrem(V_sys, K_p, d_phi, a, b, wl, planet_spectrum, data):
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
    uncertainties = data.get(
        "uncertainties_processed"
    )  # in case we want to null uncertainties

    norder, nphi, npix = residuals.shape

    N = residuals.size

    # Time-resolved total radial velocity
    radial_velocity = V_sys + K_p * np.sin(2 * np.pi * (phi + d_phi))
    # V_sys is an additive term around zero. Data should be in rest frame of star.

    delta_lambda = radial_velocity * 1e3 / constants.c  # delta lambda, for shifting

    # Initializing loglikelihood
    loglikelihood_sum = 0
    if b is not None:
        loglikelihood_sum -= N * np.log(b)

    max_transit_depth = np.max(1 - transit_weight)
    # Looping through each order and computing total log-L by summing logLs for each obvservation/order
    for i in range(norder):
        wl_slice = wl_grid[i]  # Cropped wavelengths

        models_shifted = np.zeros(
            (nphi, npix)
        )  # "shifted" model spectra array at each phase

        for j in range(nphi):
            wl_shifted = wl_slice * (1.0 - delta_lambda[j])
            F_p = np.interp(wl_shifted, wl, planet_spectrum)
            models_shifted[j] = (1 - transit_weight[j]) / max_transit_depth * (-F_p) + 1

        # divide by the median over wavelength to mimic blaze correction
        models_shifted = (models_shifted.T / np.median(models_shifted, axis=1)).T

        B = Bs[i]
        models_filtered = (
            models_shifted - B @ models_shifted
        ) * a  # filter the model and scale by alpha

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

    method = model["method"]
    spectrum_type = model["spectrum_type"]

    if "K_p" in high_res_param_names:
        K_p = high_res_params[np.where(high_res_param_names == "K_p")[0][0]]
    else:
        K_p = model["K_p"]

    if "V_sys" in high_res_param_names:
        V_sys = high_res_params[np.where(high_res_param_names == "V_sys")[0][0]]
    else:
        V_sys = model["V_sys"]

    if "log_a" in high_res_param_names:
        a = 10 ** high_res_params[np.where(high_res_param_names == "log_a")[0][0]]
    else:
        a = 10 ** model["log_a"]

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
        b = model.get("b")  # Set a value or else we null b

    if spectrum_type is "emission":
        if method is not "PCA":
            raise Exception("Emission spectroscopy only supports PCA for now.")
        else:
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

    elif spectrum_type is "transmission":
        if method not in ["sysrem", "sysrem_2022", "SYSREM"]:
            raise Exception(
                "Transmission spectroscopy only supports fast filtering with SYSREM (Gibson et al. 2022)."
            )
        else:
            if W_conv is not None:
                planet_spectrum = gaussian_filter1d(planet_spectrum, W_conv)
            loglikelihood = 0
            for key in data.keys():
                loglikelihood += loglikelihood_sysrem(
                    V_sys, K_p, d_phi, a, b, wl, planet_spectrum, data[key]
                )
            return loglikelihood
    else:
        raise Exception("Spectrum type can only be 'emission' or 'transmission'.")


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
    norder, nphi, npix = flux.shape
    cleaned_flux = flux.copy()
    # Process each order separately
    noutliers = 0
    for i in range(norder):
        # Fit a 10th order polynomial to the residual spectrum
        for j in range(nphi):
            coeffs = np.polynomial.polynomial.polyfit(wl_grid[i], flux[i, j], 20)
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
