#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:30:43 2019

@author: pelletier
"""
import numpy as np
import pdb
import os
import sys
import glob
import time
import copy
from Constants import Rsun, Rjup, Mjup, cLight, cLight_km
import matplotlib.pylab as plt
from matplotlib.ticker import NullFormatter
import matplotlib.patheffects as PathEffects
from matplotlib.gridspec import GridSpec
from scipy.interpolate import InterpolatedUnivariateSpline as Spline_interpolate
from scipy.interpolate import interp1d as OneD_interpolator
from scipy.optimize import minimize
from PyAstronomy import pyasl
from astropy.convolution import convolve as astropy_convolve
from astropy.convolution import Gaussian1DKernel, Box1DKernel
from astropy.io import fits
import astropy.units as u
import batman
import pandas as pd
from sklearn.decomposition import PCA
import random
from copy import deepcopy


# Create a night object
class LoadNight(object):
    """
    Loads all info for a given night.
    """

    def __init__(self, night_directory="", arm="", empty=False):
        if empty:
            pass
        else:
            # define empty lists for all relevant quantities to be loaded
            self.wave, self.counts_start, self.bjds = [], [], []
            SNRs, self.BERV = [], []
            self.reduction_steps, self.variance, self.exptime = [], [], []
            # make list of all files in a night directory
            file_list = os.listdir(night_directory)
            file_list.sort()
            file_list = np.sort(np.array(file_list))
            # load all exposures
            print("Loading Exposures")
            for ii, exposure in enumerate(file_list):
                hdul = pd.HDFStore(night_directory + exposure, "r+")
                # blue or red arm of MAROONX
                if arm == "red":
                    # if red
                    file_header = hdul["header_" + arm]
                    spec = hdul["spec_" + arm]
                    ind = 67
                elif arm == "blue":
                    # if blue
                    file_header = hdul["header_" + arm]
                    spec = hdul["spec_" + arm]
                    ind = 91
                else:
                    raise Exception(
                        'Error: unrecognized arm "{}", MAROON-X arm should be "red" or "blue"'.format(
                            arm
                        )
                    )
                # data dimensions
                orders = spec.index.levels[1]
                norders = len(orders)
                npixels = spec["wavelengths"][6][ind].size
                # wavelength, flux, and uncertainties
                wave, counts, variance = np.zeros([3, norders, npixels])
                for i, order in enumerate(orders):
                    wave[-i - 1, :] = spec["wavelengths"][6][order]
                    counts[-i - 1, :] = spec["optimal_extraction"][6][order]
                    variance[-i - 1, :] = spec["optimal_var"][6][order]

                # estimate of the average SNR
                SNR_per_pix = counts / np.sqrt(variance)
                SNR = np.nanmean(SNR_per_pix, axis=1)
                # useful observational information
                bjd = float(file_header["JD_UTC_FLUXWEIGHTED_FRD"])
                exptime = float(file_header["EXPTIME"])
                BERV = (
                    float(file_header["BERV_FLUXWEIGHTED_FRD"]) / 1000
                )  # convert to km/s

                self.wave.append(wave)
                self.counts_start.append(counts)
                self.variance.append(variance)
                self.bjds.append(bjd)
                SNRs.append(SNR)
                self.exptime.append(exptime)
                self.BERV.append(BERV)
                hdul.close()

            self.nexposures = len(file_list)
            self.norders = wave.shape[0]
            self.orders = np.arange(self.norders)

            # convert lists to arrays
            self.wave = np.array(self.wave) / 1e3  # change to microns
            self.counts_start = np.array(self.counts_start)
            self.variance = np.array(self.variance)
            self.bjds = np.array(self.bjds)
            SNRs = np.array(SNRs)
            self.SNR_per_order = np.mean(SNRs, axis=0)
            self.SNR_per_exposure = np.mean(SNRs, axis=1)
            self.BERV = np.array(self.BERV)
            self.exptime = np.array(self.exptime)
            self.npixels = np.size(self.counts_start[0, 0, :])  # 4088

            self.collapse_wave()
            # already mid point in maroon-x
            # self.bjds += self.exptime/60/60/24.0/2
            self.reduction_steps = []

            print(
                "Number of exposures = {}, average exposure time = {}s, average overhead = {}s".format(
                    np.round((self.nexposures)),
                    np.mean(self.exptime),
                    np.round(
                        np.mean(np.diff(self.bjds) * 24 * 3600 - self.exptime[:-1]), 0
                    ),
                )
            )
            print(
                "Night integration duration = {} hours".format(
                    np.round(
                        (self.bjds[-1] - self.bjds[0]) * 24
                        + self.exptime[-1] / 60 / 60,
                        1,
                    )
                )
            )

    # if all wavelength solutions are the same during the night, just take one of them
    # otherwise pick the highest SNR one and interpolate the otherns onto it.
    def collapse_wave(self):
        self.wavemin = self.wave.min()
        self.wavemax = self.wave.max()
        if np.sum(self.wave - self.wave[0, :, :]) == 0.0:
            self.wave = self.wave[0, :, :]
        else:
            print("Bringing all specs to the same wavelength grid")
            # choose wavelength solution at middle of time series
            midpoint_index = int(self.nexposures / 2)
            wave = self.wave[midpoint_index, :, :]
            counts_common_wave = np.zeros([self.nexposures, self.norders, self.npixels])
            for exp in range(self.nexposures):
                for order in range(self.norders):
                    # try spline interp.  In rare cases where wavelength decreases, linearly interp
                    try:
                        interp_func = Spline_interpolate(
                            self.wave[exp, order, :], self.counts_start[exp, order, :]
                        )
                        counts_common_wave[exp, order, :] = interp_func(wave[order, :])
                    except:
                        counts_common_wave[exp, order, :] = np.interp(
                            wave[order, :],
                            self.wave[exp, order, :],
                            self.counts_start[exp, order, :],
                        )
            self.wave = wave
            self.counts_start = counts_common_wave

    # first function to run after extracting data
    # sets bad regions of data to nans
    def discard_bad_data(self):
        """
        Function to get rid of bad regions of data

        """
        # add small value to bring sligtly negative values to being positive
        self.counts_start[self.counts_start < 0.5] = np.nan
        # set columns containing nans to all nans
        means = np.mean(self.counts_start, axis=0)
        for order in self.orders:
            self.counts_start[:, order, np.isnan(means[order])] = np.nan
        # also eliminate edge pixels

    def calc_planet_net_RV_shift(self):
        """
        Function to calculate the planet and star velocities during each exposure

        """
        # period in days
        perDay = self.per / 60 / 60 / 24
        # planet phases assuming circular orbit (0 = mid transit, 0.5 = mid eclipse)
        self.planet_phases = ((self.bjds - self.tt) / perDay) - np.floor(
            (self.bjds - self.tt) / perDay
        )
        print("Calculating RV assuming circular orbit (from phase)")
        RV = calc_Vcirc_from_phase(Kp=self.Kp_0, phases=self.planet_phases)
        print(
            "Phase change from {0} to {1} during night".format(
                np.round(self.planet_phases[0], 3), np.round(self.planet_phases[-1], 3)
            )
        )
        self.planet_vels = RV
        print(
            "Velocity change from {0}km/s to {1}km/s ({2}km/s per hour)".format(
                np.round(self.planet_vels[0], 2),
                np.round(self.planet_vels[-1], 2),
                np.round(
                    (self.planet_vels[-1] - self.planet_vels[0])
                    / (self.bjds[-1] - self.bjds[0])
                    / 24,
                    2,
                ),
            )
        )
        self.net_pla_shifts = self.planet_vels + self.Vsys_0 - self.BERV

        self.star_vels = (
            calc_Vcirc_from_phase(Kp=-self.kRV, phases=self.planet_phases) / 1000
        )  # km/s

    def calc_transit_properties(self, use_Master_Out_only=True):
        """
        Function to calculate which exposures to use for the out-of-transit 'master' spectrum used to fit out of every exposure.
        Also calculates the phases at which ingress/egress occur if the data contains a transit or eclipse

        use_Master_Out_only: True or False, whether to use all exposures for the 'master' spectrum, or only those out of transit / in eclipse

        """

        self.master_spec_use_exposures = np.arange(self.nexposures)

        # implemented following Gandhi et al. 2022 (Fig.4, section 2.4)
        phase = self.planet_phases
        p2 = flip_transit_phases(phase)
        phase_smooth = np.linspace(p2[0], p2[-1], 5000)
        phase_smooth[np.where(phase_smooth < 0.0)[0]] += 1
        Rs = self.Rstar
        a = self.ap
        Rp = self.Rp
        b = self.b

        # Ehrenreich+2020
        u1 = 0.393
        u2 = 0.2219

        # east and west points
        x_e = a * np.sin(2 * np.pi * phase) / Rs - Rp / Rs
        x_w = a * np.sin(2 * np.pi * phase) / Rs + Rp / Rs
        mu_e = np.sqrt(1 - x_e**2 - b**2)
        mu_w = np.sqrt(1 - x_w**2 - b**2)
        mu_e[np.isnan(mu_e)] = 0.0
        mu_w[np.isnan(mu_w)] = 0.0
        mu_avg = (mu_e + mu_w) / 2
        mu_avg[mu_avg == 0] = np.nan
        # limb darkening (0 outside of transit)
        LD = 1 - u1 * (1 - mu_avg) - u2 * (1 - mu_avg) ** 2
        LD[np.isnan(LD)] = 0.0

        self.exposure_weights = LD * -1

        self.out_of_transit_exposures = np.where(LD == 0.0)[0]
        self.in_transit_exposures = np.where(LD != 0.0)[0]

        #
        x_smooth = a * np.sin(2 * np.pi * phase_smooth) / Rs
        mu_smooth = np.sqrt(1 - x_smooth**2 - b**2)
        # limb darkening (0 outside of transit)
        LD_smooth = 1 - u1 * (1 - mu_smooth) - u2 * (1 - mu_smooth) ** 2
        LD_smooth[np.isnan(LD_smooth)] = 0.0

        ite = np.where(LD_smooth != 0.0)[0]
        self.phase_ingress = phase_smooth[ite[0]]
        self.phase_egress = phase_smooth[ite[-1]]

    def select_orders(self, use_orders):
        """
        Function to select which spectral orders to be used in the analysis.  Also updates the night min/max wave range

        use_orders: 'all' to use all orders, or provide an array of orders

        """

        if use_orders == "all":
            self.orders = self.orders
            print("Using all ({}) spectral orders".format(len(self.orders)))
        else:
            self.orders = use_orders
            print("Using provided spectral orders:")
            print(use_orders)

            self.wavemin = self.wave[self.orders[0]].min()
            self.wavemax = self.wave[self.orders[-1]].max()

    def update_orders(self, flx, cutoff=0, Print=True):
        """
        Function to update the spectral orders to be used in the analysis.  Rejects orders that have all or nearly all columns that are masked out

        flx: Data cube to be checked for bad orders
        cutoff: Fraction under which an error will be thrown out if more that this percentage of the data is nans (0.2 means that if less than 20% of the data of this order is good - reject this order)
        Print: whther or not to print which orders are rejected

        """

        # useful_fraction_per_order = np.count_nonzero(np.isfinite(flx),axis=(0,2))/self.nexposures/self.useful_pixels_per_order
        useful_fraction_per_order = (
            np.count_nonzero(np.isfinite(flx), axis=(0, 2))
            / self.nexposures
            / self.npixels
        )
        np.nan_to_num(useful_fraction_per_order, 0)
        # update orders
        new_orders = []
        not_included_orders = []
        #        for order in range(self.norders):
        for order in self.orders:
            if (
                useful_fraction_per_order[order] <= cutoff
            ):  # if order is all nans don't include
                # not_included_orders.append(int(order+1))
                not_included_orders.append(int(order))
            #                print('Not including order %s' % int(order+1) )
            else:
                new_orders.append(order)
        if Print:
            if len(not_included_orders) > 0:
                if cutoff:
                    print(
                        "Cutting off orders with less than {}% of useful pixels masked".format(
                            cutoff * 100
                        )
                    )
                print("Not including orders %s" % not_included_orders)
        self.orders = np.array(new_orders)

    def shift_to_frame(self, flux, RVs):
        # shifted_mat = np.zeros([self.nexposures,self.norders,self.npixels])
        shifted_mat = np.zeros([self.nexposures, self.norders, self.npixels]) * np.nan
        for order in self.orders:
            # shift the planet model for each of these velocities
            planet_wave_matrix = self.wave[order] * (1.0 + RVs[:, None] / cLight_km)
            for exp in range(self.nexposures):
                finite_inds = np.isfinite(flux[exp, order])
                # avoid extrapolation
                shifted_mat[exp, order, :] = np.interp(
                    self.wave[order],
                    planet_wave_matrix[exp, finite_inds],
                    flux[exp, order, finite_inds],
                    right=np.nan,
                    left=np.nan,
                )

            # cut off edge effects by setting to nan all columns that have at least 1 nan value
            order_mean = np.mean(shifted_mat[:, order, :], axis=0)
            order_nan_cols = np.where(np.isnan(order_mean))[0]
            shifted_mat[:, order, order_nan_cols] = np.nan

        return shifted_mat

    def shift_summedCCFs(self, ccf_grid, summedCCFs, RVs):
        shifted_mat = np.zeros_like(summedCCFs, dtype=self.numerical_precision)

        # planet_wave_matrix = self.wave[order]*(1.0 + RVs[:,None]/cLight_km)

        for exp in range(self.nexposures):
            # finite_inds = np.isfinite(flux[exp,order])
            new_RVs = ccf_grid + RVs[exp]

            shifted_mat[exp, :] = np.interp(ccf_grid, new_RVs, summedCCFs[exp])

        return shifted_mat

    def CCF_BERV_correction(self, RV_range, CCF_mat, Print=True):
        """
        Function to move the cross-correlation matrix from the telluric frame to the barycentric frame.
        This step is done in CCF space as opposed to flux space to not interpolate the data directly.
        Removing the BERV is also useful to add different nights together
        """

        if self.frame == "earth":
            if Print:
                print("Correcting CCFs for BERV")
            nmod, norder, nexp, nrvs = CCF_mat.shape
            CCF_mat_shifted = np.zeros([nmod, norder, nexp, nrvs])
            for mod in np.arange(nmod):
                for order in np.arange(norder):
                    for exp in np.arange(nexp):
                        CCF_mat_shifted[mod, order, exp, :] = np.interp(
                            RV_range,
                            RV_range + self.BERV[exp],
                            CCF_mat[mod, order, exp, :],
                            right=np.nan,
                            left=np.nan,
                        )

            CCF_mat_shifted = np.nan_to_num(CCF_mat_shifted, 0)
            return CCF_mat_shifted
        else:
            return CCF_mat

    def get_noise_model(
        self,
        flux_start,
        n_pcas=5,
        initial_guess=np.array([1, 0]),
        bounds=((0, None), (0, None)),
    ):
        # start with the raw flux counts (initial data without blaze correction)
        print("Fitting noise model")
        # run it through a 5 PCA reduction
        means = np.nanmean(flux_start, axis=2)
        PCA = self.PCA_by_order_TD(flux_start - means[:, :, None], Npcs_removed=n_pcas)
        rebuilt = PCA + means[:, :, None]
        residuals = flux_start - rebuilt
        # residuals are now essentially noise. Fit noise model to each order
        sigma_model = np.zeros([self.nexposures, self.norders, self.npixels]) * np.nan
        As = np.zeros(self.norders) * np.nan
        Bs = np.zeros(self.norders) * np.nan
        for order in self.orders:
            flux_start_order = flux_start[:, order, :].clip(min=0.1)
            # find nans from the first exposure
            # default bounds = positive numbers
            soln = minimize(
                gibson_likelihood,
                initial_guess,
                args=(residuals[:, order, :], flux_start_order),
                bounds=bounds,
            )

            best_a, best_b = soln.x
            As[order] = best_a
            Bs[order] = best_b
            sigma_model_order_ini = gibson_sigma(flux_start_order, best_a, best_b)

            sigma_means = np.nanmean(sigma_model_order_ini, axis=1)
            # print(order)
            sigma_PCA5 = self.PCA_TD_single_order(
                sigma_model_order_ini - sigma_means[:, None], 5
            )
            finitecols = np.where(np.isfinite(flux_start_order[0, :]))[0]
            sigma_model[:, order, finitecols] = (
                sigma_PCA5[:, finitecols] + sigma_means[:, None]
            )

        return sigma_model  # , As, Bs

    # =============================================================================
    #     # Function based on Brogi et al. 2014
    #     # looks for outliers in TIME and then fixes them by interpolating in WAVELENGTH
    #     # spline interpolates single outliers
    #     # linearly interpolates 2-3 point outliers
    #     # masks large (>4 consecutive) outlier segments
    #     # Median normalize, subtract mean spec, apply pixel correction on residuals
    # =============================================================================
    def correct_bad_pixels(self, flux, sigma=10):
        #        print('Bad Pixel Correction: correcting >%s sigma outliers' % sigma)
        # first median normalize everything
        # flux = flux.copy()
        medians = np.nanmedian(flux, axis=(1, 2))

        flx = flux / medians[:, None, None]

        # fit and remove the mean spectrum
        spec_fit = self.fit_mean_spec_slope(flux, degree_meanSpec=1, spec="median")
        residuals = flux / spec_fit
        # standard deviation of each spectral channel
        stdarray = np.nanstd(residuals, axis=0)

        axethickness = 2
        tickthickness = 2
        labelfontsize = 10
        tickfontsize = 10
        direction = "in"
        ticklength = 3

        N_bad_pixels = 0
        big_segment_counter = 0
        for exposure in np.arange(self.nexposures):
            for order in self.orders:
                # Find all bad pixel
                bad_pixels = findOutliers(
                    residuals[exposure, order], sigma=sigma, stdarray=stdarray[order]
                )
                bad_pixels_indices = np.where(bad_pixels == True)[0]
                # keep track of how many there are in total
                N_bad_pixels += len(bad_pixels_indices)
                # split into segments
                segments = split_list(bad_pixels_indices)
                if segments[0].size == 0:
                    pass
                else:
                    for seg in segments:
                        # if ONE INDIVIDUAL, correct it with Spline interpolation
                        if np.size(seg) == 1.0:
                            bad_pixel = seg[0]
                            # how many surrounding points to use in the interpotation
                            lower = 25
                            upper = 25
                            # if too close to the edge, use what is available
                            if bad_pixel < 25:
                                lower = bad_pixel
                            if bad_pixel > 4088 - 25:
                                upper = bad_pixel

                            # make a wave/flux array surrounding but not including the bad pixel
                            wave = self.wave[
                                order, bad_pixel - lower : bad_pixel + upper
                            ]
                            flxx = flx[
                                exposure, order, bad_pixel - lower : bad_pixel + upper
                            ]
                            bad_pix_flux = flx[exposure, order, bad_pixel]
                            if np.isnan(bad_pix_flux) or bad_pix_flux.sum():
                                # I think this happens if it flags a bad pixel in the exposure, but the residuals are nan because the mean spec is nan (i.e. some other exposure has nans at this wavelength)
                                # print('Flagged bad pixel is nan')
                                pass
                            else:
                                short_array_bad_pix = np.where(flxx == bad_pix_flux)[0]
                                wave1 = np.delete(wave, short_array_bad_pix)
                                flux1 = np.delete(flxx, short_array_bad_pix)
                                finite = np.isfinite(flux1)
                                wave2 = wave1[finite]
                                flux2 = flux1[finite]
                                # interpolate new value
                                if flux2.size == 0:
                                    flx[exposure, order, bad_pixel] = np.nan
                                elif len(wave2) < 4:
                                    flx[exposure, order, bad_pixel] = np.nan
                                else:
                                    interp_func = Spline_interpolate(wave2, flux2)
                                    fixed_pix = interp_func(
                                        wave[short_array_bad_pix][0]
                                    )
                                    #                                med = np.nanmedian(flx[exposure,order,:][flx[exposure,order,:] > np.nanmedian(flx[exposure,order,:])])
                                    #                                old_residual = residuals[bad_pixel]
                                    old_residual = residuals[exposure, order, bad_pixel]
                                    #                                new_residual = fixed_pix-med-spec_fit[exposure,order,bad_pixel]
                                    new_residual = (
                                        fixed_pix - spec_fit[exposure, order, bad_pixel]
                                    )
                                    # if the new value is actually worse than the old one, don't use it
                                    if np.abs(new_residual) > np.abs(old_residual):
                                        Status = "Not Approved"
                                    else:
                                        Status = "Approved"
                                        flx[exposure, order, bad_pixel] = fixed_pix

                        # if two to three neighboring pixels, correct them with linear interpolation
                        elif np.size(seg) < 4:
                            bad_pixel_2_or_3 = seg
                            # how many surrounding points to use in the interpotation
                            lower = 25
                            upper = 25
                            # if too close to the edge, use what is available
                            if bad_pixel_2_or_3[0] < lower:
                                lower = bad_pixel_2_or_3[0]
                            if bad_pixel_2_or_3[-1] > 4088 - upper:
                                upper = 4088 - bad_pixel_2_or_3[-1]

                            # make a wave/flux array surrounding but not including the bad pixel
                            wave = self.wave[
                                order,
                                bad_pixel_2_or_3[0]
                                - lower : bad_pixel_2_or_3[-1]
                                + upper,
                            ]
                            flxx = flx[
                                exposure,
                                order,
                                bad_pixel_2_or_3[0]
                                - lower : bad_pixel_2_or_3[-1]
                                + upper,
                            ]
                            bad_pix_flux = flx[exposure, order, bad_pixel_2_or_3]
                            # find out which indices correspond to the bad pixels
                            short_array_bad_pix = []
                            if (
                                np.isnan(bad_pix_flux.sum())
                                or np.isnan(flxx.sum())
                                or bad_pix_flux.sum()
                            ):
                                #                                print('Flagged bad pixel small group is nan')
                                pass
                            else:
                                for val in bad_pix_flux:
                                    short_array_bad_pix.append(
                                        np.where(flxx == val)[0][0]
                                    )
                                short_array_bad_pix = np.array(short_array_bad_pix)
                                # delete them for the interpolation
                                wave1 = np.delete(wave, short_array_bad_pix)
                                flux1 = np.delete(flxx, short_array_bad_pix)
                                finite = np.isfinite(flux1)
                                wave2 = wave1[finite]
                                flux2 = flux1[finite]
                                # interpolate new value
                                interp_func = OneD_interpolator(wave2, flux2)
                                try:
                                    fixed_pixs = interp_func(wave[short_array_bad_pix])
                                except:  # if too close to the edge just those channels to nan
                                    fixed_pixs = (
                                        np.zeros_like(
                                            wave[short_array_bad_pix],
                                            dtype=self.numerical_precision,
                                        )
                                        * np.nan
                                    )
                                    flx[:, order, bad_pixel_2_or_3] = fixed_pixs

                                #                                med = np.nanmedian(flx[exposure,order,:][flx[exposure,order,:] > np.nanmedian(flx[exposure,order,:])])

                                old_residuals = residuals[
                                    exposure, order, bad_pixel_2_or_3
                                ]
                                #                                new_residuals = fixed_pixs-med-spec_fit[exposure,order,bad_pixel_2_or_3]
                                new_residuals = (
                                    fixed_pixs
                                    - spec_fit[exposure, order, bad_pixel_2_or_3]
                                )
                                # if the new value is actually worse than the old one, don't use it
                                if np.sum(np.abs(new_residuals)) > np.sum(
                                    np.abs(old_residuals)
                                ):
                                    Status = "Not Approved"
                                else:
                                    Status = "Approved"
                                    flx[exposure, order, bad_pixel_2_or_3] = fixed_pixs

                        else:
                            #                            print('Big (%s) Segment' % int(np.size(seg)) )
                            big_segment_counter += 1
                            flx[:, order, seg] = np.nan

        print("Total number of big segments = {}".format(int(big_segment_counter)))
        print("Total number of outliers = {}".format(N_bad_pixels))
        # re-add medians
        return flx * medians[:, None, None]

    def fit_mean_spec_slope(self, flux, degree_meanSpec=2, spec="median", Print=True):
        if Print:
            print("Fitting out {} spectrum from each exposure".format(spec))
        flx = flux.copy()

        # mean_spec_fit = np.zeros([self.nexposures, self.norders, self.npixels])*np.nan
        mean_spec_fit = np.empty([self.nexposures, self.norders, self.npixels])
        mean_spec_fit.fill(np.nan)
        for order in self.orders:
            # print(order)
            # construct a mean spectrum
            # if spec == 'mean':
            #     mean_spec = np.nanmean(flx[:,order,:], axis=0)
            # elif spec == 'median':
            #     mean_spec = np.nanmedian(flx[:,order,:], axis=0)
            if spec == "mean":
                mean_spec = np.mean(
                    flx[self.master_spec_use_exposures, order, :], axis=0
                )
            elif spec == "median":
                mean_spec = np.median(
                    flx[self.master_spec_use_exposures, order, :], axis=0
                )
            else:
                raise Exception('Error: Please select "mean", "median"')
            # pick only the non-nan ones (as each row should have the same non points, check the nans in the first row)
            # even if only using out-of-transit, calculate nans for all exposures
            non_nans = np.isfinite(np.mean(flx[:, order, :], axis=0))
            # don't fit if order is only nans
            if np.any(non_nans) == True:
                for exposure in np.arange(self.nexposures):
                    # fit the mean spectrum to each exposure (typically with a second order polynomial)
                    mean_spec_poly_coeffs = np.polyfit(
                        mean_spec[non_nans],
                        flx[exposure, order, non_nans],
                        degree_meanSpec,
                    )[::-1]
                    # reconstruct that polynomial
                    polynomial = np.polynomial.polynomial.polyval(
                        mean_spec[non_nans], mean_spec_poly_coeffs
                    ).T
                    # fit as a polynomial of the mean spectrum and AFTER fit a slope
                    mean_spec_fit[exposure, order, non_nans] = polynomial

        return mean_spec_fit

    # prepare the data before the processing (things done here not re-done in modelling processing)
    def spec_alignment(
        self,
        flux,
        spectrum_alignment="norm",
        keep_steps=False,
        box_width=21,
        gaussian_width=50,
    ):
        if keep_steps:
            try:
                # if reduction_steps exists:
                self.reduction_steps.append(flux + 0)
            except:
                # if reduction_steps does not exist, make it
                self.reduction_steps = []
                self.reduction_steps.append(flux + 0)

        if spectrum_alignment == "hpf":
            # print('Bringing all specs to a common blaze with a high pass filter of box width {}, smoothed by a Gaussian filter of standard deviation {}'.format(prep_case['common_blaze_hpf_box_width'],prep_case['common_blaze_hpf_gauss_std'] ) )
            print("Bringing all specs to a common continuum")
            med_spec = np.nanmedian(flux, axis=0)
            flux1 = flux / med_spec
            # 11 and 50 Gibson et al. 2020
            # hpf1 = self.highpass_filter(flux1, 'Double', 21, 50)
            #  Gibson+2022
            hpf1 = self.highpass_filter(flux1, "Double", box_width, gaussian_width)
            flux /= hpf1

        elif spectrum_alignment == "norm":
            print("Normalizing each spectrum by the median of the 500 brightest pixels")
            normalization = self.normalize_each_spec(flux, nbrightest=500)
            flux /= normalization
        #            normed_flux_corrected = copy.deepcopy(flux)

        else:
            print("Warning: No spectrum alignment was applied")

        if keep_steps:
            self.reduction_steps.append(flux + 0)

        return flux

    def process_data(
        self, flux, Npcs_removed=10, keep_steps=False, sigmas=None, Print=True
    ):
        # print('Applying Telluric mask')
        flux *= self.tel_mask

        # self.finite_dat_inds  = np.array([   np.where(np.isfinite( np.mean(flux[:,order,:],axis=0) ) )[0] for order in  range(self.norders) ])
        self.finite_dat_inds = np.array(
            [
                np.where(np.isfinite(np.mean(flux[:, order, :], axis=0)))[0]
                for order in range(self.norders)
            ],
            dtype=object,
        )

        spec_fit = self.fit_mean_spec_slope(flux)
        flux /= spec_fit
        if keep_steps:
            self.reduction_steps.append(flux + 0)

        # if np.sum(process_case['PCAs']) != 0:
        if Npcs_removed:
            if Print:
                print("Applying PCA: removing {} components".format(Npcs_removed))

            mean_time = np.nanmean(flux, axis=2)
            rebuilt = self.PCA_by_order_TD1(
                flux - mean_time[:, :, None], Npcs_removed=Npcs_removed
            )
            rebuilt += mean_time[:, :, None]

            flux /= rebuilt
            if keep_steps:
                self.reduction_steps.append(flux + 0)

        self.high_std_mask = self.build_high_std_mask(
            flux, sigma=4, per_order=True, Print=Print
        )
        flux *= self.high_std_mask

        return flux

    def highpass_filter(self, flx, hpf_type="Gaussian", resPower=100, resPower_2nd=5):
        #        print('High pass filtering')
        t0 = time.time()
        flux = flx * 1.0
        highpass_fit = np.ones([self.nexposures, self.norders, self.npixels]) * np.nan

        if hpf_type == "Box":
            for order in self.orders:
                for exp in range(self.nexposures):
                    inds = np.isfinite(flux[exp, order, :])
                    left_side_mean = np.median(
                        flux[exp, order, inds][: int(resPower / 2)]
                    )
                    right_side_mean = np.median(
                        flux[exp, order, inds][-int(resPower / 2) :]
                    )
                    yy = np.hstack(
                        [
                            np.ones(int(resPower / 2)) * left_side_mean,
                            flux[exp, order, inds],
                            np.ones(int(resPower / 2) - 1) * right_side_mean,
                        ]
                    )
                    highpass_fit[exp, order, inds] = running_mean(yy, resPower)
        elif hpf_type == "Gaussian":
            for order in self.orders:
                for exp in range(self.nexposures):
                    highpass_fit[exp, order, :] = astropy_convolve(
                        flux[exp, order, :],
                        Gaussian1DKernel(resPower / 2.35),
                        boundary="extend",
                        preserve_nan=True,
                    )
        elif hpf_type == "Double":
            for order in self.orders:
                gi = np.isnan(flux[:, order, :])
                aa = running_filter(flux[:, order, :], np.nanmedian, resPower)
                aa[gi] = np.nan
                kernel = Gaussian1DKernel(resPower_2nd)
                for exp in range(self.nexposures):
                    highpass_fit[exp, order, :] = astropy_convolve(
                        aa.data[exp], kernel, boundary="extend", preserve_nan=True
                    )
        else:
            raise Exception(
                "Highpass filter type must be either 'Box', 'Gaussian' or 'Double' "
            )

        print(
            "{} High Pass took {} minutes".format(
                hpf_type, str(np.round((time.time() - t0) / 60.0, 3))
            )
        )
        return highpass_fit

    # Prepare the thermal model for the cross-correlation
    def Prep_model_for_CCF(self, model_wave, model_flux, resolution=85000):
        n_dim = np.ndim(model_flux)
        if n_dim == 1:
            print("Preparing model for Cross-Correlation")
            model_wave, broadened_model_flux = BroadenSpec(
                wave=model_wave, flux=model_flux, end_resolution=resolution, Print=True
            )
        else:
            n_models = np.shape(model_flux)[0]
            print("Preparing %s models for cross-correlation" % n_models)
            broadened_model_flux = np.zeros([n_models, len(model_wave)])
            for i in range(n_models):
                model_wave, broadened_model_flux[i, :] = BroadenSpec(
                    wave=model_wave, flux=model_flux[i, :], end_resolution=resolution
                )

        # convert from dppm to d
        broadened_model_flux /= 1e6

        # average planet RV shift between 2 exposures
        avg_pla_RV = np.mean(np.diff(self.net_pla_shifts))
        # portion of the delta RV that occurs during the exposure (not during overheads)
        fraction = np.mean(self.exptime) / (np.mean(np.diff(self.bjds)) * 60 * 60 * 24)
        # calculate by how much the planet shifts during 1 exposure
        pla_RV_during_exposure = avg_pla_RV * fraction
        #        print('Average delta RV during 1 exposure = %s km/s, roughly %s pixels' % (np.round(np.abs(pla_RV_during_exposure),3),np.round(np.abs(pla_RV_during_exposure/2),3)) )

        # delta lambda over lambda = v over c

        # if shift over 1 pixel is significant, convolve
        try:
            model_velocity_resolution = (
                cLight_km * (model_wave[1] - model_wave[0]) / model_wave[0]
            )
            # determine how many model wavelength points the planet delta RV during 1 exposure corresponds to
            box_size = np.abs(pla_RV_during_exposure / model_velocity_resolution)
            if n_dim == 1:
                model_flux_avg = astropy_convolve(
                    broadened_model_flux, Box1DKernel(box_size), boundary="extend"
                )
            else:
                model_flux_avg = np.zeros([n_models, len(model_wave)])
                for i in range(n_models):
                    model_flux_avg[i, :] = astropy_convolve(
                        broadened_model_flux[i, :],
                        Box1DKernel(box_size),
                        boundary="extend",
                    )
        # otherwise if minor, convolution may not work so interpolate
        except:
            shifted_plus = model_wave * (1.0 + pla_RV_during_exposure / 2 / cLight_km)
            shifted_minus = model_wave * (1.0 - pla_RV_during_exposure / 2 / cLight_km)
            # take flux as average of flux at start and end of exposure
            if n_dim == 1:
                flux_pos = np.interp(model_wave, shifted_plus, broadened_model_flux)
                flux_neg = np.interp(model_wave, shifted_minus, broadened_model_flux)
            else:
                flux_pos = np.zeros([n_models, len(model_wave)])
                flux_neg = np.zeros([n_models, len(model_wave)])
                for i in range(n_models):
                    flux_pos[i, :] = np.interp(
                        model_wave, shifted_plus, broadened_model_flux[i, :]
                    )
                    flux_neg[i, :] = np.interp(
                        model_wave, shifted_minus, broadened_model_flux[i, :]
                    )
            model_flux_avg = (flux_pos + flux_neg) / 2.0
            model_flux_avg /= model_lpf

        return model_wave, model_flux_avg

    # time domain PCA (1st component ~ airmass)
    def PCA_by_order_TD1(self, flux, Npcs_removed):
        mattt_rebuilt = np.zeros([self.nexposures, self.norders, self.npixels])
        for order in self.orders:
            if isinstance(Npcs_removed, (int, float)):
                n_pca = Npcs_removed
            else:
                n_pca = int(Npcs_removed[order])
            pcs, coefficients, npca_removed = PCA_decompose(
                matrix=flux[
                    :, order, np.array(self.finite_dat_inds[order], dtype=int)
                ].T,
                n_components=n_pca,
            )
            mattt_rebuilt[
                :, order, np.array(self.finite_dat_inds[order], dtype=int)
            ] = PCA_rebuild(
                pcs=pcs, coefficients=coefficients, n_pcs=int(npca_removed)
            ).T

        return mattt_rebuilt

    # time domain PCA (1st component ~ airmass)
    def PCA_by_order_TD(self, flux, Npcs_removed):
        mattt_rebuilt = np.zeros([self.nexposures, self.norders, self.npixels])
        # loop over relevant orders
        for order in self.orders:
            if isinstance(Npcs_removed, (int, float)):
                n_pca = Npcs_removed
            else:
                n_pca = int(Npcs_removed[order])
            mattt_rebuilt[:, order, :] = self.PCA_TD_single_order(
                flux[:, order, :], n_pca=n_pca
            )
        return mattt_rebuilt

    # time domain PCA (1st component ~ airmass) for a single order
    def PCA_TD_single_order(self, flux, n_pca):
        data_norm = flux.T
        mattt_rebuilt = np.zeros([self.npixels, self.nexposures])
        # take only columns that do not have nan values
        finitecols = np.isfinite(np.mean(data_norm[:, :], axis=1))
        # decompose into principal components
        pcs, coefficients, Npcs_removed = PCA_decompose(
            matrix=data_norm[finitecols], n_components=n_pca
        )
        # rebuild the data with only N pcas
        mattt_rebuilt[finitecols] = PCA_rebuild(
            pcs=pcs, coefficients=coefficients, n_pcs=int(Npcs_removed)
        )
        # re-transpose to go back to original shape
        return mattt_rebuilt.T

    # function to mask certain wavelength regions
    def build_regions_mask(self, regions=[[]], returnMask=False):
        # first check if a telluric mask is already defined (as would be if self.build_threshold_mask() has been run)
        try:
            mask = self.tel_mask
        # if not, define it
        except:
            mask = np.ones([self.norders, self.npixels])

        pre_regions = np.round(
            100
            * np.count_nonzero(mask[self.orders] * 0.0)
            / np.size(mask[self.orders]),
            1,
        )

        if regions == None:
            print(
                "Warning: TelMask includes 'regions' but params['regions'] is set to None"
            )
        else:
            # loop over all orders, flag regions in
            for order in range(self.norders):
                for region in regions:
                    region_min = region[0]
                    region_max = region[1]
                    # if out of range
                    if self.wave[order].max() < region_min:
                        pass
                    elif self.wave[order].min() > region_max:
                        pass
                    # if in range, apply mask
                    else:
                        result = np.where(
                            (self.wave[order] > region_min)
                            & (self.wave[order] < region_max)
                        )[0]
                        mask[order, result] = np.nan

        self.tel_mask = mask

    def apply_tel_mask(self, flux):
        return flux * self.tel_mask

    def build_high_std_mask(self, flux, sigma=3.0, per_order=False, Print=True):
        # Step 7: remove columns that have a standard deviation more than 3 times the average standard deviation    Removed: (of that order)
        mask = np.ones([self.norders, self.npixels])
        tot = 0
        tot_finite = 0
        if per_order == True:
            #            std_per_order = np.nanstd(flux, axis=(0,2) )
            for order in self.orders:
                std_per_wave = np.nanstd(flux[:, order, :], axis=0)
                finite = np.where(np.isfinite(std_per_wave))[0]
                tot_finite += len(finite)
                polyfit = gaussian_filter(np.ma.array(std_per_wave), 20)
                flat_std_per_wave = std_per_wave - polyfit
                std = np.nanstd(flat_std_per_wave)
                high_variance_cols = np.where(flat_std_per_wave > sigma * std)[0]
                mask[order, high_variance_cols] = np.nan
                tot += len(high_variance_cols)
        else:
            std = np.nanstd(flux)
            for order in self.orders:
                std_per_wave = np.nanstd(flux[:, order, :], axis=0)
                finite = np.where(np.isfinite(std_per_wave))[0]
                #            high_variance_cols = np.where(std_per_wave > sigma*std_per_order[order] )[0]
                high_variance_cols = np.where(std_per_wave > sigma * std)[0]
                mask[order, high_variance_cols] = np.nan
                tot += len(high_variance_cols)
                tot_finite += len(finite)
        #                if len(high_variance_cols) > 0:
        #                    print('Removing {} columns from order {}'.format(len(high_variance_cols), int(order+1) ) )
        #            print('Removing {} high standard deviation columns'.format(tot ))
        self.percent_masked_std = np.round(100 * tot / tot_finite, 1)
        if Print:
            print(
                "Masking >{} sigma standard deviation columns ({} columns total ({}% of the data)).".format(
                    sigma, tot, self.percent_masked_std
                )
            )
        return mask

    # multi model in for loop at end
    # CCF_G20, logL_G20, CCF_BL19, logL_BL19
    def CCF(self, fluxes, pla_mod_wave, pla_mod_flux, Planet_RVs, sigmas, Print=True):
        if Print:
            print("Starting Cross-Correlation")
        flxes_dat = fluxes.copy() + 0

        self.RV_range = Planet_RVs  # update
        if len(pla_mod_flux.shape) == 2:
            oneD = False
        else:
            oneD = True
        nmodels = int(np.size(pla_mod_flux) / np.size(pla_mod_wave))

        CCF_mat = np.zeros(
            [nmodels, self.norders, self.nexposures, Planet_RVs.size], dtype="float32"
        )

        # mask is so that the standard deviation of the model is only calculated on pixels where the data is non-nan
        mask = np.ones([self.norders, self.npixels], dtype="float32") * np.nan
        # re-calculate finite indices (different from stored finite_cols that is calculated before application of standard deviation mask)
        avg_spec = np.mean(flxes_dat, axis=0)
        # avg_spec = np.nanmean(flxes_dat,axis=0)
        finite_dat_inds = np.array(
            [
                np.where(np.isfinite(avg_spec[order, :]))[0]
                for order in range(self.norders)
            ],
            dtype=object,
        )
        for order in range(self.norders):
            mask[order, finite_dat_inds[order]] = 1.0
        wa = np.ma.array(self.wave * mask)
        # mean normalize the data
        flxes_dat -= np.nanmean(flxes_dat, axis=2)[:, :, None]

        # calculate number of points
        N = np.count_nonzero(~np.isnan(flxes_dat[0]), axis=1)
        N_tot = np.sum(N)
        # calculate standard deviation of data (only needs to be done once)
        sigma_f = np.sqrt(np.nansum(flxes_dat**2, axis=2))
        t0 = time.time()

        for i, primary_velocity in enumerate(Planet_RVs):
            # shift combined model by v_primary
            shifted_comb_model_wave = pla_mod_wave * (
                1.0 + primary_velocity / cLight_km
            )

            flux_mod = np.zeros([nmodels, self.norders, self.npixels], dtype="float32")
            for j in range(nmodels):
                if oneD:
                    flx_mod = np.interp(wa, shifted_comb_model_wave, pla_mod_flux)
                else:
                    flx_mod = np.interp(wa, shifted_comb_model_wave, pla_mod_flux[j, :])
                flux_mod[j, :, :] = flx_mod - np.nanmean(flx_mod, axis=1)[:, None]

            CCF_rv = np.zeros([nmodels, self.nexposures, self.norders], dtype="float32")

            for ordr in self.orders:
                flx_dat_ordr = flxes_dat[:, ordr, finite_dat_inds[ordr]]

                for mod in range(nmodels):
                    flux_mod_ordr = flux_mod[mod, ordr, finite_dat_inds[ordr]]

                    CCF_rv[mod, :, ordr] = np.sum(
                        flx_dat_ordr[:, :]
                        * flux_mod_ordr
                        / sigmas[:, ordr, finite_dat_inds[ordr]] ** 2,
                        axis=1,
                    )
            CCF_mat[:, :, :, i] = np.nan_to_num(np.transpose(CCF_rv, axes=(0, 2, 1)), 0)

        if Print:
            print(
                "Cross-Correlation: took {} minutes".format(
                    str(np.round((time.time() - t0) / 60.0, 3))
                )
            )

        return CCF_mat

    def calc_SNRmap(self, CCF_mat, exposure_weights=None, box_half_width=10):
        if exposure_weights is None:
            exposure_weights = np.ones(self.nexposures)

        self.SummedCCF_mat = np.sum(CCF_mat, axis=1)
        self.KpVsysMapCCF, map_std = self.calc_significance_map(
            self.SummedCCF_mat,
            self.Vsys_range,
            self.Kp_range,
            box_half_width=box_half_width,
            exposure_weights=exposure_weights,
        )
        self.SNRmapCCF = (
            self.KpVsysMapCCF - np.median(self.KpVsysMapCCF, axis=(1, 2))[:, None, None]
        ) / map_std[:, None, None]

    # default RM_center (0) means vsys
    def calc_CCF_mask(self, RM_mask=0, RM_center=0):
        self.CCF_mask = np.ones([self.nexposures, len(self.RV_range)])
        if RM_center:
            mid = RM_center
        else:
            mid = self.Vsys_0
        if RM_mask:
            print(
                "Applying mask {}km/s mask centered at {}km/s".format(
                    RM_mask, RM_center
                )
            )
            self.CCF_mask[
                :, np.where(np.abs(self.RV_range - np.round(mid)) < RM_mask)[0]
            ] = 0

    def calc_significance_map(
        self, CCmat, Vsys_range, Kp_range, box_half_width=5, exposure_weights=None
    ):
        if len(np.shape(CCmat)) == 3:
            n_models = np.shape(CCmat)[0]
            KpVsysMat = np.zeros([int(n_models), Vsys_range.size, Kp_range.size])
        else:
            n_models = 1
            KpVsysMat = np.zeros([Vsys_range.size, Kp_range.size])

        for exp in np.arange(self.nexposures):
            try:
                use_inds = np.nonzero(self.CCF_mask[exp])[0]
            except:
                # use_inds = np.arange(self.nexposures)
                # use_inds = np.arange(Vsys_range.size)
                use_inds = np.arange(self.RV_range.size)
                # print('USING ALL RVs FOR CCF MAP')
            # scale the planet_vels calculated at pla.Kp to all other Kps
            base = Kp_range * (self.planet_vels[exp] / self.Kp_0)
            # repeat the base of Vsecs for each Vsys, each of these will have a Vsys added to it
            Kp_Vsys1 = np.tile(base, (Vsys_range.size, 1))
            Vsysss = np.repeat(Vsys_range, Kp_range.size).reshape(
                Vsys_range.size, Kp_range.size
            )
            # calculate the Vsec corresponding to every point in the Kp/Vsys matrix
            # if self.frame == 'star':
            #     Vsecs = Kp_Vsys1 + Vsysss
            # else:
            #     Vsecs = Kp_Vsys1 + Vsysss - self.BERV[exp]
            Vsecs = Kp_Vsys1 + Vsysss
            # make interpolation function of likelihood
            if n_models == 1:
                # Calculate likelihood at every point on the Kp/Vsys matrix
                #                interp_func = Spline_interpolate(self.RV_range, CCmat[0,exp])
                #                Kp_Vsys = interp_func(Vsecs)
                Kp_Vsys = np.interp(
                    Vsecs, self.RV_range[use_inds], CCmat[0, exp, use_inds]
                )
                # Kp_Vsys = np.interp(Vsecs, self.RV_range, CCmat[0,exp])
                if exposure_weights is None:
                    KpVsysMat += Kp_Vsys
                else:
                    KpVsysMat += Kp_Vsys * exposure_weights[exp]
            else:
                for model in range(n_models):
                    Kp_Vsys = np.interp(
                        Vsecs, self.RV_range[use_inds], CCmat[model, exp, use_inds]
                    )
                    # Kp_Vsys = np.interp(Vsecs, self.RV_range, CCmat[model,exp])
                    if exposure_weights is None:
                        KpVsysMat[model] += Kp_Vsys
                    else:
                        KpVsysMat[model] += Kp_Vsys * exposure_weights[exp]

        #        print('Using box width = %s' % str(int(box_half_width)*2 + 1) )

        ind_nearest_Kp = find_nearest(Kp_range, self.Kp_0)
        ind_nearest_Vsys = find_nearest(Vsys_range, self.Vsys_0)

        night_cut = KpVsysMat.copy()
        # how big a box around the planet not to include in standard deviation calculation
        buf = box_half_width
        # peak region, don't count this when calculating std
        if n_models == 1:
            # night_cut[0,ind_nearest_Vsys-buf:ind_nearest_Vsys+buf, ind_nearest_Kp-buf:ind_nearest_Kp+buf] = np.nan
            night_cut[0, ind_nearest_Vsys - buf : ind_nearest_Vsys + buf, :] = np.nan
            std = np.array([np.nanstd(night_cut[0])])
        else:
            # night_cut[:,ind_nearest_Vsys-buf:ind_nearest_Vsys+buf, ind_nearest_Kp-buf:ind_nearest_Kp+buf] = np.nan
            night_cut[:, ind_nearest_Vsys - buf : ind_nearest_Vsys + buf, :] = np.nan
            std = np.nanstd(night_cut, axis=(1, 2))

        return KpVsysMat, std

    def save(self, filename="night"):
        # delete all the big stuff
        wave = self.wave
        del self.wave
        counts_start = self.counts_start
        del self.counts_start
        variance = self.variance
        del self.variance
        flux = self.flux
        del self.flux
        CCF_mat = self.CCF_mat
        del self.CCF_mat
        KpVsysMapCCF = self.KpVsysMapCCF
        del self.KpVsysMapCCF
        SNRmapCCF = self.SNRmapCCF
        del self.SNRmapCCF
        SummedCCF_mat = self.SummedCCF_mat
        del self.SummedCCF_mat
        CCF_mask = self.CCF_mask
        del self.CCF_mask
        high_std_mask = self.high_std_mask
        del self.high_std_mask
        finite_dat_inds = self.finite_dat_inds
        del self.finite_dat_inds
        tel_mask = self.tel_mask
        del self.tel_mask
        reduction_steps = self.reduction_steps
        del self.reduction_steps
        # save night object
        savepickle(self, filename + ".pkl")
        # re-add all that was deleted
        self.wave = wave
        self.counts_start = counts_start
        self.variance = variance
        self.flux = flux
        self.CCF_mat = CCF_mat
        self.KpVsysMapCCF = KpVsysMapCCF
        self.SummedCCF_mat = SummedCCF_mat
        self.SNRmapCCF = SNRmapCCF
        self.CCF_mask = CCF_mask
        self.high_std_mask = high_std_mask
        self.finite_dat_inds = finite_dat_inds
        self.tel_mask = tel_mask
        self.reduction_steps = reduction_steps

    # %% Plotting Functions

    def plotCCOrbit2(
        self,
        CCmap,
        cmap="gray_r",
        mean_norm=True,
        xlim=[None, None],
        apply_CCF_mask=False,
        savepath=None,
        saveformat="png",
    ):
        summed_CCmap = CCmap + 0
        if mean_norm:
            summed_CCmap -= np.mean(summed_CCmap, axis=1)[:, None]

        if apply_CCF_mask:
            summed_CCmap *= self.CCF_mask

        phases = flip_transit_phases(self.planet_phases)

        # pdb.set_trace()
        # Y1, X1 = np.meshgrid(np.arange(self.nexposures),self.RV_range )
        Y1, X1 = np.meshgrid(phases, self.RV_range)
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        # im = ax[0].pcolormesh(X1,Y1, np.transpose(summed_CCmap ), cmap=cmap )
        ax[0].pcolormesh(X1, Y1, np.transpose(summed_CCmap), cmap=cmap)
        #        fig.colorbar(im, ax=ax[0])

        ax[0].set_xlim(xlim)

        # pp1 = (self.net_pla_shifts-self.pla.vsys)*(np.ones(self.RV_range.size)[:,None])
        pp1 = (self.planet_vels + self.Vsys_0) * (np.ones(self.RV_range.size)[:, None])

        rest_frame_summed_CCF_1 = np.zeros_like(summed_CCmap)
        for exp in range(self.nexposures):
            interped_CCF_1 = np.interp(
                X1[:, exp], X1[:, exp] - pp1[:, exp], summed_CCmap[exp]
            )
            rest_frame_summed_CCF_1[exp, :] = interped_CCF_1

        # pcm11_1=ax[1].pcolormesh(X1, Y1, np.transpose(rest_frame_summed_CCF_1), cmap=cmap)
        ax[1].pcolormesh(X1, Y1, np.transpose(rest_frame_summed_CCF_1), cmap=cmap)
        ax[1].set_xlim([-100, 100])

        ax[0].tick_params(labelbottom=True, labeltop=False, top="on")
        ax[1].tick_params(labelbottom=True, labelleft=False)
        #        rax = ax[0].twiny()
        #        rax.plot(self.RV_range, np.zeros_like(self.RV_range)*np.nan)
        #        rax.set_xlabel('RV Shift (km/s)')
        ax[0].set_title("Earth Frame")
        ax[0].set_xlabel("RV Shift (km/s)")
        # ax[0].set_ylabel('Exposure #')
        ax[0].set_ylabel("Phase")
        ax[1].set_title("Planet Rest Frame")
        ax[1].set_xlabel("RV Shift (km/s)")
        # ax[1].set_ylabel('Exposure #')

        ax[0].axhline(self.phase_ingress - 1, linestyle="--", lw=1.0)
        ax[0].axhline(self.phase_egress, linestyle="--", lw=1.0)
        ax[1].axhline(self.phase_ingress - 1, linestyle="--", lw=1.0)
        ax[1].axhline(self.phase_egress, linestyle="--", lw=1.0)

        fig.tight_layout()

        if savepath:
            plt.savefig(savepath + "." + saveformat)

    def plotKpVsys(
        self,
        KpVsysMat,
        cmap="viridis",
        cmaptype="contour",
        title="",
        xlim=None,
        ylim=[20, None],
        cbar_ticks=None,
        Kpslice=None,
        Vsysslice=None,
        savepath=None,
        printSNR=True,
        saveformat="png",
    ):
        ticklength = 5
        axethickness = 2
        tickthickness = 2
        tickfontsize = 14
        linewidth = 3
        axlinewidth = 1
        labelfontsize = 16
        clbfontsize = 14
        direction = "in"

        fig, ax = plt.subplots(figsize=(14, 8), sharex=True)
        ax.set_axis_off()
        plt.figure(fig.number)

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left + width + 0.013
        left_h = left + width + 0.01
        bottom_width = 0.14
        top_width = 0.2

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, top_width]
        rect_histy = [left_h, bottom, bottom_width, height]

        ax0 = plt.axes(rect_scatter)
        axtop = plt.axes(rect_histx)
        axright = plt.axes(rect_histy)

        # no labels
        nullfmt = NullFormatter()
        axtop.xaxis.set_major_formatter(nullfmt)
        axright.yaxis.set_major_formatter(nullfmt)

        if Kpslice == None:
            Kpslice = self.Kp_0
        if Vsysslice == None:
            Vsysslice = self.Vsys_0

        ind_nearest_Kp = find_nearest(self.Kp_range, Kpslice)
        ind_nearest_Vsys = find_nearest(self.Vsys_range, Vsysslice)

        night = KpVsysMat

        axtop.plot(self.Vsys_range, night[:, ind_nearest_Kp], linewidth=linewidth)
        axright.plot(night[ind_nearest_Vsys, :], self.Kp_range, linewidth=linewidth)

        axtop.plot(
            self.Vsys_range,
            night[:, ind_nearest_Kp],
            linewidth=linewidth,
            color="k",
            label="Both",
        )
        axright.plot(
            night[ind_nearest_Vsys, :],
            self.Kp_range,
            linewidth=linewidth,
            color="k",
            label="Both",
        )

        axright.axhline(Kpslice, color="k", linestyle="--", linewidth=axlinewidth)
        # axtop.axvline(self.pla.vsys, color='k', linestyle = '--', linewidth=axlinewidth)
        axtop.axvline(Vsysslice, color="k", linestyle="--", linewidth=axlinewidth)
        axtop.set_xlim([self.Vsys_range.min(), self.Vsys_range.max()])
        axright.set_ylim([self.Kp_range.min(), self.Kp_range.max()])

        X1, Y1 = np.meshgrid(self.Kp_range, self.Vsys_range)

        #        im = ax0.contourf(Y1,X1, night/std, 12 , cmap=plt.cm.get_cmap(cmap))#, levels=12 )
        #        im = ax0.contourf(Y1,X1, night/std , cmap=cmap)#, levels=12 )

        if printSNR:
            ax0.text(
                Y1.max() * 1.05,
                X1.max() * 1.03,
                r"$\Delta$SNR = {}".format(
                    str(np.round(np.abs(night.max() / night.min()), 3))
                ),
            )

        if cmaptype == "contour":
            im = ax0.contourf(
                Y1, X1, night, 200, cmap=cmap, zorder=-20
            )  # , levels=12 )
        elif cmaptype == "mesh":
            im = ax0.pcolormesh(Y1, X1, night, cmap=cmap)
        # Botttom Label
        cbaxes = fig.add_axes([left_h + bottom_width + 0.01, bottom, 0.02, height])
        #        ticks = [-3,-1,1,3,5]
        if cbar_ticks:
            cb = plt.colorbar(im, cax=cbaxes, ticks=cbar_ticks)
        else:
            cb = plt.colorbar(im, cax=cbaxes)
        cb.ax.minorticks_off()

        cb.ax.tick_params(
            width=tickthickness, labelsize=clbfontsize
        )  # , direction='out', which='major')
        cb.set_label(r"S/N", fontsize=labelfontsize)

        # ax.plot(pla.vsys, pla.Kp, marker = 'x', color = 'b', markersize = 10, mew=3)
        ax0.axhline(Kpslice, color="w", linestyle="--", linewidth=axlinewidth)
        # ax0.axvline(self.pla.vsys, color='w', linestyle = '--', linewidth=axlinewidth)
        ax0.axvline(Vsysslice, color="w", linestyle="--", linewidth=axlinewidth)
        ax0.set_xlabel(r"$V_{\mathrm{sys}}$ [km/s]", fontsize=labelfontsize)
        ax0.set_ylabel(r"$K_p$ [km/s]", fontsize=labelfontsize)

        axtop.set_ylabel(r"S/N", fontsize=labelfontsize)
        axright.set_xlabel(r"S/N", fontsize=labelfontsize)

        ax0.tick_params(
            top="on",
            right="on",
            bottom="on",
            left="on",
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
        )
        axtop.tick_params(
            top="on",
            right="on",
            bottom="on",
            left="on",
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
        )
        axright.tick_params(
            top="on",
            right="on",
            bottom="on",
            left="on",
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
        )

        axtop.set_title(title, fontsize=labelfontsize)

        for axis in ["top", "bottom", "left", "right"]:
            ax0.spines[axis].set_linewidth(axethickness)
            axright.spines[axis].set_linewidth(axethickness)
            axtop.spines[axis].set_linewidth(axethickness)

        ax0.set_xlim(xlim)
        axtop.set_xlim(xlim)

        ax0.set_ylim(ylim)
        axright.set_ylim(ylim)

        ax0.minorticks_off()
        axtop.minorticks_off()
        axright.minorticks_off()

        for c in im.collections:
            c.set_edgecolor("face")

        ax0.set_rasterization_zorder(-10)

        if savepath:
            plt.savefig(savepath + "." + saveformat)

    def plotCCFpanels(
        self,
        CCmap,
        KpVsysMat,
        mean_norm=False,
        phase_label=None,
        xlim=[None, None],
        map_ylim=[None, None],
        Kpslice=None,
        map_cmap="viridis",
        trail_cmap="gray_r",
        mol_label=None,
        vmin_buffer=1,
        vmax_buffer=1,
        linewidth=3,
        labelpad=1,
        slice_ticks=[],
        cbar_ticks=[],
        cbar_ticklabels=[],
        cbar_loc="out",
        apply_CCF_mask=False,
        show_slice=True,
        print_slice=False,
        savepath=None,
    ):
        ticklength = 4
        axethickness = 2.5
        tickthickness = 2.5
        tickfontsize = 15
        axlinewidth = 1
        labelfontsize = 17
        clbfontsize = 15
        direction = "in"

        if phase_label == "deg":
            factor = 360
        else:
            factor = 1

        if show_slice:
            fig, ax11 = plt.subplots(figsize=(5, 6), sharex=True)
            # fig, ax11 = plt.subplots(figsize = (5,5.7))

            gs = GridSpec(8, 1)
            ax00 = plt.subplot(gs[:2, :])
            ax = plt.subplot(gs[2:6, :])
            ax22 = plt.subplot(gs[6:, :])
            bottom = 0.085
        else:
            fig, ax11 = plt.subplots(figsize=(5, 5))
            gs = GridSpec(6, 1)
            ax00 = plt.subplot(gs[:2, :])
            ax = plt.subplot(gs[2:, :])
            bottom = 0.11

        if cbar_loc == "out" or cbar_loc == "right":
            right = 0.88
        else:
            # right = 0.963
            right = 0.975

        # plt.subplots_adjust(top=0.999, bottom=bottom, left = 0.152, right=right ,hspace=0.)
        plt.subplots_adjust(
            top=0.998, bottom=bottom, left=0.148, right=right, hspace=0.0
        )

        summed_CCmap = CCmap + 0
        if mean_norm:
            summed_CCmap -= np.mean(summed_CCmap, axis=1)[:, None]

        ######
        # top panel
        #####
        YY_two, XX_two = np.meshgrid(self.planet_phases * factor, self.RV_range)

        if apply_CCF_mask:
            summed_CCmap *= self.CCF_mask

        im1 = ax00.pcolormesh(
            XX_two,
            YY_two,
            np.transpose(summed_CCmap),
            cmap=trail_cmap,
            vmin=np.percentile(summed_CCmap, 0.01) / vmin_buffer,
            vmax=np.percentile(summed_CCmap, 99.9) / vmax_buffer,
            rasterized=True,
        )

        ax00.set_rasterization_zorder(-10)

        ax00.set_xlim(xlim)

        ax00.tick_params(
            labelbottom=False,
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
        )
        # ax00.set_xlabel('RV Shift (km/s)')
        ax00.set_ylabel("Phase", labelpad=labelpad, fontsize=labelfontsize)

        if self.spectype == "transit":
            ax00.axhline(self.phase_ingress * factor, linestyle="--", lw=1.0, color="w")
            ax00.axhline(self.phase_egress * factor, linestyle="--", lw=1.0, color="w")
        ax00.minorticks_off()

        show_trail = False
        if show_trail:
            # phase_smooth = np.linspace(0, 1, 1000)
            phase_smooth = np.linspace(
                self.planet_phases.min(), self.planet_phases.max(), 1000
            )
            v_planet_smooth = calc_Vcirc_from_phase(Kp=self.pla.Kp, phases=phase_smooth)

            vels_smooth = v_planet_smooth + self.pla.vsys  # - vbarys_smooth

            bu = 0.004
            # phase_smooth_cut = phase_smooth.copy()
            # ind = (phase_smooth <= self.planet_phases.max()+bu) & (phase_smooth >= self.planet_phases.min()-bu)
            # phase_smooth_cut[ind] = np.nan

            ax00.plot(vels_smooth, phase_smooth, linestyle=":", color="b", linewidth=1)
            # ax00.plot(vels_smooth, phase_smooth_cut, linestyle=':', color='b', linewidth=1)

        phase_yticks = ax00.get_yticks()
        tick_strs = []
        for tick in phase_yticks:
            if phase_label == "deg":
                if tick > 0:
                    tick_strs.append(
                        r"$+$" + str(int(np.round(tick, 2))) + r"$\degree$"
                    )
                elif tick < 0:
                    tick_strs.append(
                        r"$-$" + str(int(np.abs(np.round(tick, 2)))) + r"$\degree$"
                    )
                else:
                    tick_strs.append(str(int(np.round(tick, 2))) + r"$\degree$")
            else:
                if tick > 0 and tick < 0.2:
                    tick_strs.append(r"$+$" + str(np.round(tick, 2)))
                elif tick < 0 and tick > 0.8:
                    tick_strs.append(r"$-$" + str(np.abs(np.round(tick, 2))))
                else:
                    tick_strs.append(str(np.round(tick, 2)))

        ax00.set_yticklabels(tick_strs)

        # pos_tick_inds = np.where(phase_yticks>0)[0]
        # pos_ticks = phase_yticks[np.where(phase_yticks>0)[0]]
        # tick_strs = []
        # for tick in pos_ticks:
        #     tick_strs.append('+'+str(tick))
        # phase_yticks_str = phase_yticks.type('str')
        # for ind in pos_tick_inds:
        #     phase_yticks_str[ind] = '+' + str(np.round(phase_yticks[ind],2))

        ######
        # middle panel
        #####
        if Kpslice == None:
            Kpslice = self.pla.Kp

        try:
            bot_ind = find_nearest(self.Kp_range, map_ylim[0])
        except:
            bot_ind = 0
        try:
            top_ind = find_nearest(self.Kp_range, map_ylim[-1])
        except:
            top_ind = -1

        X1, Y1 = np.meshgrid(self.Kp_range[bot_ind:top_ind], self.Vsys_range)
        im = ax.contourf(
            Y1, X1, KpVsysMat[:, bot_ind:top_ind], 200, cmap=map_cmap, zorder=-20
        )

        # X1, Y1 = np.meshgrid(self.Kp_range, self.Vsys_range)
        # im = ax.pcolormesh(Y1,X1,  KpVsysMat , cmap=map_cmap )
        # im = ax.contourf(Y1,X1,  KpVsysMat, 200, cmap=map_cmap, zorder=-20 )

        if cbar_loc == "out" or cbar_loc == "right":
            cbaxes = fig.add_axes(
                [
                    ax.get_position().x0 + ax.get_position().width + 0.01,
                    ax.get_position().y0,
                    0.03,
                    ax.get_position().height,
                ]
            )
            cbar_label_color = "k"
            makecbar = True
        elif cbar_loc == "in":
            # cbaxes = fig.add_axes([ax.get_position().x0 + ax.get_position().width - 0.13, ax.get_position().y0 + 0.02, 0.04, ax.get_position().height/2])
            cbaxes = fig.add_axes(
                [
                    ax.get_position().x0 + ax.get_position().width - 0.145,
                    ax.get_position().y0 + 0.02,
                    0.04,
                    ax.get_position().height / 2,
                ]
            )
            cbar_label_color = "w"
            makecbar = True
        else:
            makecbar = False

        if makecbar:
            cb = fig.colorbar(im, cax=cbaxes, pad=0.01)

            if len(cbar_ticks) > 0:
                cb.set_ticks(cbar_ticks)
            if len(cbar_ticklabels) > 0:
                cb.set_ticklabels(cbar_ticklabels)

            cb.ax.minorticks_off()

            cb.ax.tick_params(
                labelsize=clbfontsize,
                width=tickthickness,
                direction=direction,
                length=ticklength,
                color="k",
            )
            cb.ax.tick_params(labelcolor=cbar_label_color)
            cb.outline.set_color("k")
            cb.outline.set_linewidth(axethickness)
            # [i.set_linewidth(3) for i in ax.spines.itervalues()]

            # for axis in ['top','bottom','left','right']:
            #     cb.ax.spines[axis].set_linewidth(axethickness)

        if mol_label:
            txt = ax.text(
                xlim[0] + np.diff(xlim) * 0.05,
                map_ylim[0] + np.diff(map_ylim) * 0.85,
                mol_label,
                color="w",
                weight="bold",
                fontsize=25,
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="k")])

        # cb.outline.set_edgecolor('w')
        # cb.ax.yaxis.set_tick_params(color='w')
        # cb.set_label(r'S/N ($\sigma$)', labelpad=labelpad,  fontsize=labelfontsize)

        # ax.plot(pla.vsys, pla.Kp, marker = 'x', color = 'b', markersize = 10, mew=3)
        # ax.axhline(self.pla.Kp, color='w', linestyle = '--', linewidth=axlinewidth)
        # ax.axvline(self.pla.vsys, color='w', linestyle = '--', linewidth=axlinewidth)
        ax.plot(
            self.pla.vsys,
            Kpslice,
            marker="+",
            markersize=5,
            markeredgewidth=1,
            color="k",
        )
        # ax.plot(self.pla.vsys, self.pla.Kp, marker='+', markersize=7, markeredgewidth=1.5, color='w')
        # ax.plot(self.pla.vsys, self.pla.Kp, marker='+', markersize=5, markeredgewidth=1.5, color='#ff7f0e')

        # ax.set_xlabel(r'V$_{\mathrm{sys}}$ (km/s)', fontsize=labelfontsize)
        # ax.set_ylabel(r'$K_p$ (km/s)', fontsize=labelfontsize)
        ax.set_ylabel(r"$K_p$ [km/s]", fontsize=labelfontsize, labelpad=0)

        ax.tick_params(
            labelbottom=False,
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
        )

        ax.set_xlim(xlim)

        # ax.set_ylim(map_ylim)

        for c in im.collections:
            c.set_edgecolor("face")

        ax.set_rasterization_zorder(-10)
        # fig.subplots_adjust(right=0.98, top=0.94)
        ax.minorticks_off()

        ######
        # bottom panel
        #####
        if show_slice:
            ind_nearest_Kp = find_nearest(self.Kp_range, Kpslice)

            night = KpVsysMat

            # ax22.plot(self.Vsys_range, night[:,ind_nearest_Kp], linewidth = linewidth )

            ax22.plot(
                self.Vsys_range,
                night[:, ind_nearest_Kp],
                color="#1f77b4",
                linewidth=linewidth,
            )

            ax22.axvline(
                self.pla.vsys, color="k", linestyle="--", linewidth=axlinewidth
            )
            # ax22.axvline(self.pla.vsys, color='w', linestyle = '--', linewidth=axlinewidth)
            # ax22.set_xlim(xlim)
            # ax22.set_xlabel(r'$V_{\mathrm{sys}}$ (km/s)', fontsize=labelfontsize)
            # ax22.set_ylabel(r'S/N ($\sigma$)',  fontsize=labelfontsize)
            ax22.set_xlabel(
                r"$V_{\mathrm{sys}}$ [km/s]", fontsize=labelfontsize, labelpad=0
            )
            ax22.set_ylabel(r"S/N [$\sigma$]", fontsize=labelfontsize)
            ax22.minorticks_off()
            ax22.tick_params(
                width=tickthickness,
                labelsize=tickfontsize,
                direction=direction,
                length=ticklength,
            )
            if len(slice_ticks) > 0:
                ax22.set_yticks(slice_ticks)
            if print_slice:
                # ax22.text(np.min(self.Vsys_range)*0.95, np.max(night[:,ind_nearest_Kp])*0.78, r'$K_p$ = {}$\,$km/s'.format(Kpslice), fontsize=tickfontsize)
                lim = ax22.get_ylim()
                height = np.diff(lim)[0]
                top = lim[1]
                bot = lim[0]
                ax22.text(
                    np.max(self.Vsys_range) * 0.2,
                    top - height * 0.17,
                    r"$K_p$ = {}$\,$km/s".format(Kpslice),
                    fontsize=tickfontsize,
                )
                # ax22.text(np.max(self.Vsys_range)*0.28, np.max(night[:,ind_nearest_Kp])*0.78, r'$K_p$ = {}$\,$km/s'.format(Kpslice), fontsize=tickfontsize)
                # ax22.text(97, bot - height*0.153, r'100'.format(Kpslice), fontsize=tickfontsize, horizontalalignment='center')

            # for tick in ax22.xaxis.get_majorticklabels():
            #     tick.set_horizontalalignment("right")
            # ax22.set_xticks([-95,-50,0,50,95])
            # xticks = ax22.get_xticks()
            # ax22.set_xticks(xticks[:-1])
            ax22.set_xlim(xlim)
            # ax_ = fig.add_subplot(111)
            # # ax22.text(xticks[-1], ax22.get_ylim()[0]/1.1, r'100'.format(Kpslice), fontsize=tickfontsize, horizontalalignment='center')
            # ax_.text(0.2, 0.2, r'100'.format(Kpslice), fontsize=tickfontsize, horizontalalignment='center')

            for axis in ["top", "bottom", "left", "right"]:
                ax22.spines[axis].set_linewidth(axethickness)
        else:
            ax.tick_params(
                labelbottom=True,
                width=tickthickness,
                labelsize=tickfontsize,
                direction=direction,
                length=ticklength,
            )
            # ax.set_xlabel(r'$V_{\mathrm{sys}}$ (km/s)', fontsize=labelfontsize)
            ax.set_xlabel(
                r"$V_{\mathrm{sys}}$ [km/s]", fontsize=labelfontsize, labelpad=0
            )

            # lim = ax.get_ylim()
            # height = np.diff(lim)[0]
            # top = lim[1]
            # bot = lim[0]
            # ax.text(97, bot - height*0.07, r'100'.format(Kpslice), fontsize=tickfontsize, horizontalalignment='center')
            # xticks = ax.get_xticks()
            # ax.set_xticks(xticks[:-1])

        for axis in ["top", "bottom", "left", "right"]:
            ax00.spines[axis].set_linewidth(axethickness)
            ax.spines[axis].set_linewidth(axethickness)

        if savepath:
            plt.savefig(savepath + "." + saveformat)

    def plotReductionSteps_with_sigma(
        self,
        stepsList,
        order=None,
        cmap="gist_heat",
        save=False,
        savepath=None,
        showlegends=False,
    ):
        if order is None:
            # order = int(self.norders/2)
            order = 10

        inds = np.arange(self.npixels)

        wave = self.wave[order, inds]

        wave *= 1000

        XX1, YY1 = np.meshgrid(np.arange(self.nexposures), wave)

        axethickness = 1
        tickthickness = 1
        labelfontsize = 7
        tickfontsize = 7
        outlinewidth = 1

        direction = "in"
        tickcolor = "k"
        ticklength = 2

        mm = 1 / 25.4
        # fig, ax = plt.subplots(len(stepsList), 1, sharex=True, figsize=(6,8))
        fig, ax = plt.subplots(
            len(stepsList), 1, sharex=True, figsize=(120 * mm, 140 * mm)
        )

        # Remove horizontal space between axes
        fig.subplots_adjust(hspace=0)

        # letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        letters = ["a", "b", "c", "d", "e", "f", "g"]

        # if include the legend then leave more room for it on the right
        if showlegends:
            fig.subplots_adjust(left=0.07, right=1.075, top=0.92, bottom=0.087)
        else:
            # fig.subplots_adjust(left=0.108, right=0.96, top=0.99, bottom=0.072)
            fig.subplots_adjust(
                top=0.995, bottom=0.053, left=0.093, right=0.993, hspace=0.0, wspace=0.2
            )

        ax[0].set_xlim([wave[10], wave[-10]])

        for i in range(len(stepsList) - 1):
            if i == 0:
                ax[i].tick_params(
                    labelbottom=None,
                    labeltop=None,
                    top=None,
                    right=True,
                    bottom="on",
                    left="on",
                    width=tickthickness,
                    labelsize=tickfontsize,
                    direction=direction,
                    length=ticklength,
                    pad=2,
                )
            else:
                ax[i].tick_params(
                    labelbottom=None,
                    labeltop=None,
                    top="on",
                    right=True,
                    bottom="on",
                    left="on",
                    width=tickthickness,
                    labelsize=tickfontsize,
                    direction=direction,
                    length=ticklength,
                    pad=2,
                )

            im = ax[i].pcolormesh(
                YY1,
                XX1,
                np.transpose(stepsList[i][:, order, inds]),
                cmap=cmap,
                vmin=np.nanpercentile(stepsList[i][:, order, inds], 1),
                vmax=np.nanpercentile(stepsList[i][:, order, inds], 99),
                rasterized=True,
            )
            ax[i].set_rasterization_zorder(-10)
            if showlegends:
                pad = 0.01

                if np.nanmean(stepsList[i][:, order, inds]) > 10000:
                    if np.any(overwride_cbar_ticks[i]):
                        cb = fig.colorbar(
                            im,
                            ax=ax[i],
                            pad=pad,
                            aspect=10,
                            format="%.2e",
                            ticks=overwride_cbar_ticks[i],
                        )  # , ticks=[-3,-2,-1,0,1,2,3,4])#, pad=colorbar_pad)
                        cb.ax.set_yticklabels(
                            overwride_cbar_tick_labels[i]
                        )  # , fontsize=tickfontsize)
                    else:
                        cb = plt.colorbar(
                            im, ax=ax[i], pad=pad, aspect=10, format="%.2e"
                        )  # , ticks=[-3,-2,-1,0,1,2,3,4])#, pad=colorbar_pad)

                    # cb = plt.colorbar(im, ax=ax[i], pad=0.065, aspect=10)
                    # cb.formatter.set_powerlimits((-2, 2))
                    cb.ax.tick_params(
                        width=tickthickness,
                        labelsize=tickfontsize,
                        direction=direction,
                        length=ticklength,
                    )
                else:
                    cb = plt.colorbar(im, ax=ax[i], pad=pad, aspect=10)
                    cb.ax.tick_params(
                        width=tickthickness,
                        labelsize=tickfontsize,
                        direction=direction,
                        length=ticklength,
                    )
                cb.minorticks_off()

            ax[i].set_ylabel("exposure #", fontsize=labelfontsize, labelpad=4)

            ax[i].minorticks_off()

            x_limits = ax[i].get_xlim()
            y_limits = ax[i].get_ylim()
            ax_width = np.diff(x_limits)[0]
            ax_height = np.diff(y_limits)[0]

            # txt  = ax[i].text(x_limits[0]+ax_width*0.03, y_limits[-1]-ax_height*0.35, letters[i], color='w', weight='bold', fontsize = LetterFontsize)
            # txt.set_path_effects([PathEffects.withStroke(linewidth=outlinewidth, foreground='k')])
            ax[i].text(
                x_limits[0] - np.diff(x_limits) / 10,
                y_limits[1] * 0.9,
                letters[i],
                color="k",
                weight="bold",
                fontname="DejaVu Sans",
                fontsize=8,
            )

            for axis in ["top", "bottom", "left", "right"]:
                ax[i].spines[axis].set_linewidth(axethickness)

        # sigma_mean = np.nanmedian(stepsList[-1][:,order,:],axis=0)
        # ax[-1].plot(wave, sigma_mean, lw=0.5, color='#1f77b4')
        ax[-1].plot(wave, stepsList[-1][-1, order, :], lw=0.5, color="#1f77b4")
        # ax[-1].plot(wave, sigma_mean, lw=2.5)
        ax[-1].minorticks_off()

        ax[-1].text(
            ax[-1].get_xlim()[0] - np.diff(ax[-1].get_xlim()) / 10,
            ax[-1].get_ylim()[1] * 0.9,
            letters[i + 1],
            color="k",
            weight="bold",
            fontname="DejaVu Sans",
            fontsize=8,
        )
        ax[-1].set_ylabel("noise", fontsize=labelfontsize, labelpad=4)

        for axis in ["top", "bottom", "left", "right"]:
            ax[-1].spines[axis].set_linewidth(axethickness)

        x_limits = ax[-1].get_xlim()
        y_limits = ax[-1].get_ylim()
        ax_width = np.diff(x_limits)[0]
        ax_height = np.diff(y_limits)[0]

        ax[-1].tick_params(
            labelbottom="on",
            labeltop=None,
            top="on",
            right=True,
            bottom="on",
            left="on",
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
        )

        ax[-1].tick_params(axis="y", which="major", pad=1)

        ax[-1].set_xlabel(
            r"wavelength (nm)", fontsize=labelfontsize
        )  # , labelpad=0.03)

        if save:
            if savepath:
                plt.savefig(
                    savepath + ".tif",
                    dpi=300,
                    format="tiff",
                    pil_kwargs={"compression": "tiff_lzw"},
                )
                # plt.savefig(savepath + '.pdf',dpi=300)


def gen_common_phase_grid(nights):
    phase_min = np.zeros(len(nights))
    phase_max = np.zeros(len(nights))
    phase_delta = np.zeros(len(nights))
    all_phases = np.empty(0)
    # transit = 0
    for i, night in enumerate(nights):
        planet_phases = flip_transit_phases(night.planet_phases)
        phase_min[i] = np.min(planet_phases)
        phase_max[i] = np.max(planet_phases)
        phase_delta[i] = np.min(np.diff(planet_phases))
        all_phases = np.append(all_phases, planet_phases)

    phase_min = np.min(phase_min)
    phase_max = np.max(phase_max)
    min_delta_phase = np.min(phase_delta)

    phase_grid = np.arange(
        phase_min - min_delta_phase, phase_max + min_delta_phase * 2, min_delta_phase
    )

    keep_inds = []
    for i, phase in enumerate(phase_grid):
        ind = find_nearest(all_phases, phase)
        if np.abs(phase - all_phases[ind]) < min_delta_phase:
            keep_inds.append(i)

    negative_inds = np.where(phase_grid < 0)[0]
    phase_grid[negative_inds] += 1

    return phase_grid[keep_inds]


# function to combine  multiple nights together
def combine_nights(nights, phases):
    nmols, ph, nRVs = nights[0].SummedCCF_mat.shape

    SummedCCF_mat = np.zeros([nmols, len(phases), nRVs])

    # interpolate each night onto common phase grid
    for night in nights:
        # if transit, flip phases for interpolation
        phases2 = flip_transit_phases(phases)
        planet_phases2 = flip_transit_phases(night.planet_phases)

        for j in range(nmols):
            for i in range(nRVs):
                rv_ccf = np.interp(
                    phases2,
                    planet_phases2,
                    night.SummedCCF_mat[j, :, i],
                    left=0,
                    right=0,
                )
                SummedCCF_mat[j, :, i] += rv_ccf

    ni = LoadNight(empty=True)
    ni.Kp_0 = night.Kp_0
    ni.Vsys_0 = night.Vsys_0
    ni.tt = night.tt
    ni.per = night.per
    ni.kRV = night.kRV
    # ni.incli = night.incli
    # ni.a_R = night.a_R
    ni.Rstar = night.Rstar
    ni.Rp = night.Rp
    ni.b = night.b
    ni.ap = night.ap

    ni.nexposures = len(phases)
    ni.RV_range = night.RV_range
    ni.planet_vels = calc_Vcirc_from_phase(phases=phases, Kp=night.Kp_0)
    ni.phase_ingress = night.phase_ingress
    ni.phase_egress = night.phase_egress
    ni.BERV = np.zeros(ni.nexposures)
    # ni.bjds = np.linspace(pla.TimeOfPeri - pla.perDay()*np.abs(phase_min),pla.TimeOfPeri + pla.perDay()*np.abs(phase_max),ni.nexposures)
    # ni.bjds = np.ones(ni.nexposures)
    ni.bjds = phases * ni.per / 24 / 3600 + ni.tt
    ni.net_pla_shifts = ni.planet_vels + ni.Vsys_0 - ni.BERV
    ni.planet_phases = phases
    ni.Kp_range = night.Kp_range
    ni.Vsys_range = night.Vsys_range

    ni.SummedCCF_mat = SummedCCF_mat

    # KpVsysMat_all = np.zeros([nmols,len(phases),ni.Vsys_range.size,ni.Kp_range.size])
    ni.calc_transit_properties()
    ni.calc_CCF_mask(RM_mask=10, RM_center=0)

    ni.KpVsysMapCCF, map_std = ni.calc_significance_map(
        ni.SummedCCF_mat,
        ni.Vsys_range,
        ni.Kp_range,
        box_half_width=40,
        exposure_weights=ni.exposure_weights,
    )
    ni.SNRmapCCF = (
        ni.KpVsysMapCCF - np.median(ni.KpVsysMapCCF, axis=(1, 2))[:, None, None]
    ) / map_std[:, None, None]

    # phase-fold CCF matrix to compute SNR maps
    # night.calc_SNRmap(CCF_mat=CCF_mat, order_weights=order_weights, exposure_weights=exposure_weights, box_half_width=10)

    return ni


"""
Function for transits the turns phases of 0.8, 0.9 to -0.2, -0.1 etc..

    Input: phase array near transit and cutoff

    Output: phases with negative phases before transit
"""


def flip_transit_phases(phases, cutoff=0.5):
    flipped_phases = phases + 0
    inds = np.where(phases > cutoff)[0]
    neg_phases = phases[inds] - 1
    flipped_phases[inds] = neg_phases
    return flipped_phases


def gaussian(x, mean, sigma):
    G = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / np.sqrt(2 * np.pi * sigma**2)
    return G


def plotTP(
    p_layers,
    T_samples,
    ax=None,
    xlim=[None, None],
    ylim=[1e-7, 1e2],
    Tlabel="bottom",
    Plabel="left",
    color=None,
    axethickness=2,
    lw=2,
    fontsize=16,
    tickthickness=2,
    tickfontsize=16,
    direction="in",
    ticklength=3,
    labelpad=0,
    pad=3.5,
    smooth_sigma=1,
    alpha=1,
    showTPpoints=None,
    TPmarkersize=5,
    ylabel_coord=None,
    returnTP=False,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    levels = [2.5, 16, 50, 84, 97.5]
    perc = np.nanpercentile(T_samples, levels, axis=0).T

    smooth_perc = np.zeros([len(p_layers), 5])

    n_elements = np.round(3 * smooth_sigma) * 2 + 1
    kernel = gaussian(
        np.arange(n_elements), mean=(n_elements - 1) / 2, sigma=smooth_sigma
    )

    for i in np.arange(5):
        smooth_perc[:, i] = astropy_convolve(perc[:, i], kernel, boundary="extend")

    xnew2 = np.linspace(np.log10(p_layers).min(), np.log10(p_layers).max(), 1000)
    fine_perc = np.interp(xnew2, np.log10(p_layers), smooth_perc[:, 2])

    ax.set_xlabel("temperature (K)", fontsize=fontsize)
    ax.set_ylabel("pressure (bar)", fontsize=fontsize, labelpad=labelpad)

    try:
        ax.yaxis.set_label_coords(ylabel_coord[0], ylabel_coord[1])
    except:
        pass

    ax.set_yscale("log")

    if color:
        ax.fill_betweenx(
            p_layers / 1e5,
            smooth_perc[:, 0],
            smooth_perc[:, 4],
            color=color,
            alpha=alpha - 0.1,
            zorder=-10,
            edgecolor=None,
        )
        ax.fill_betweenx(
            p_layers / 1e5,
            smooth_perc[:, 1],
            smooth_perc[:, 3],
            color=color,
            alpha=alpha,
            zorder=-9,
            edgecolor=None,
        )
        ax.plot(
            smooth_perc[:, 2],
            p_layers / 1e5,
            color=color,
            lw=lw,
            zorder=-8,
            alpha=alpha + 0.15,
        )
    else:
        ax.fill_betweenx(
            p_layers / 1e5,
            smooth_perc[:, 0],
            smooth_perc[:, 4],
            color=[0.9, 0.9, 1],
            alpha=alpha,
            zorder=-10,
        )
        ax.fill_betweenx(
            p_layers / 1e5,
            smooth_perc[:, 1],
            smooth_perc[:, 3],
            color=[0.7, 0.7, 1],
            alpha=alpha,
            zorder=-9,
        )
        ax.plot(smooth_perc[:, 2], p_layers / 1e5, color="blue", lw=lw, zorder=-8)

    if np.any(showTPpoints):
        inds = []
        for i, Ppoint in enumerate(showTPpoints):
            ind = int((np.abs(xnew2 - Ppoint)).argmin())
            inds.append(ind)

        ax.plot(
            fine_perc[inds],
            10 ** xnew2[inds] / 1e5,
            linestyle="",
            markersize=TPmarkersize,
            marker="o",
            color="k",
            zorder=99,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()

    if Tlabel == "top":
        ax.xaxis.set_label_position("top")
        ax.tick_params(
            labelbottom=False,
            labeltop=True,
            top="on",
            right="on",
            bottom="on",
            left="on",
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
            pad=pad,
        )
    else:
        ax.tick_params(
            labelbottom=True,
            labeltop=False,
            top="on",
            right="on",
            bottom="on",
            left="on",
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
            pad=pad,
        )

    if Plabel == "right":
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            labelleft=False,
            labelright=True,
            top="on",
            right="on",
            bottom="on",
            left="on",
            width=tickthickness,
            labelsize=tickfontsize,
            direction=direction,
            length=ticklength,
            pad=pad,
        )

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(axethickness)

    ax.minorticks_off()

    if returnTP:
        return fig, ax, smooth_perc
    else:
        return fig, ax


"""
Calculate the circular velocity given the phase from point of conjunction (transit)
Does not take into account the eccentricity
Brogi & Line 2019 Eq. 1

    Input: Planet phases and planet (positive) Kp.  Give star Kp as negative

    Output: the planet radial velocity

"""


def calc_Vcirc_from_phase(phases, Kp):
    return Kp * np.sin(2 * np.pi * phases)


"""
Gibson noise model & likelihood function

"""


def gibson_sigma(F, a=1, b=0):
    return np.sqrt(a * F + 10**b)


def gibson_likelihood(theta, R, F):
    a, b = theta
    sig = gibson_sigma(F, a, b)
    return -1.0 * (-0.5 * np.nansum((R / sig) ** 2) - np.nansum(np.log(sig)))


"""
Function to find _ sigma outliers in an array.  Same as the simple case of remOutliers in auxbenneke, but allows for nan values, and returns indOutlier instead of indKeep

    Input: array of values, sigma limit

    Output: the indices of all the outliers
"""


def findOutliers(y, sigma, stdarray=None):
    med = np.nanmedian(y)
    if stdarray is not None:
        std = stdarray
    else:
        std = np.nanstd(y)
    indOutlier = np.logical_or(y > med + sigma * std, y < med - sigma * std)
    #    indKeep = np.logical_not(indOutlier)
    return indOutlier  # indKeep


"""
Function that splits a list of integers into segments

    Input: list of indices

    Output: list of arrays of group of indices

    ex: input = [3,4,5,22,23,54]    ouput = [array([3,4,5]), array([22,23]), array([54]) ]
"""


def split_list(mylist):
    """Function to do the initial split"""
    # calculate differences
    d = np.diff(mylist)
    # when the differences are not 1 save that location
    # we need a +1 to get make up for the lost element
    breaks = list(np.arange(len(mylist) - 1)[d != 1] + 1)
    slices = zip([0] + breaks, breaks + [len(mylist)])
    # slice up the list
    int_list = [mylist[a:b] for a, b in slices]
    # chop up long sequences
    # flatten the list once
    return int_list


"""
Function that returns the running mean of an array

    Input: An array (y) and the size of the box of the running mean (resPower)

    Output: the running mean y at every point in the array

"""


def running_mean(y, N):
    cumsum = np.cumsum(np.insert(y, 0, 0))
    cont = (cumsum[N:] - cumsum[:-N]) / float(N)
    return cont


"""
Function that broadens a flux array to a lower spectral resolution.
Also uncludes the option of rotational broadening

Input: wave/flux arrays on a fixed wavelength resolution grid (typically a scarlet model), desired resolution (i.e. the resolution of the instrument - 75000 for SPIRou), (optional) start resolution of the model (typically 125 or 250k)

"""


def BroadenSpec(
    wave,
    flux,
    end_resolution=70000,
    start_resolution=None,
    sigma_width=5,
    rot_broad=False,
    Print=False,
):
    # if the initial resolution of the model is not given, calculate it from the wavelength array of the model
    if start_resolution is None:
        # if start resolution is not defined then wave must be
        start_resolution = np.mean(wave[:-1]) / np.mean(np.diff(wave[:-1]))
        if Print == True:
            print(
                "Calculated Model Resolution = {} --> Downgrading to R = {}".format(
                    np.round(start_resolution), np.round(end_resolution)
                )
            )

    if rot_broad:
        rotker = RotKerTransitCloudy(
            pl_rad=pla.Rp / Rjup * u.jupiterRad,
            pl_mass=pla.Mp / Mjup * u.jupiterMass,
            t_eq=pla.Teq00 * u.K,
            omega=2 * np.pi / pla.perDay() / u.day,
            resolution=end_resolution,
        )
        kernel = rotker.return_kernel(start_resolution, n_os=100, pad=4)
    else:
        # determine FWHM of the Gaussian that we want to use to convolve our model with to downgrade its resolution
        FWHM = start_resolution / float(end_resolution)
        # convert that FWHM into sigma (standard conversion equation - wikipedia)
        sigma = FWHM / 2.35482  # / (2 * np.sqrt(2 * np.log(2)) )
        # determine width of Gaussian to use for convolution: go out sigma_width*sigma elements in each direction
        n_elements = np.round(sigma_width * sigma) * 2 + 1
        # generate convolution kernel
        kernel = gaussian(np.arange(n_elements), mean=(n_elements - 1) / 2, sigma=sigma)

    # perform convolution with broadening kernel
    # broad_flux = np.convolve(flux, kernel, mode='same')
    broad_flux = astropy_convolve(flux, kernel, boundary="extend")
    return wave, broad_flux


"""
Antoine Darveau-Bernier's toy model to broaden a transmission spectrum
"""


class RotKerTransitCloudy:
    def __init__(
        self,
        pl_rad,
        pl_mass,
        t_eq,
        omega,
        resolution,
        left_val=1,
        right_val=1,
        step_smooth=1,
        v_mid=0,
        mu=None,
    ):
        """Rotational kernel for transit at dv constant (constant resolution)
        pl_rad: scalar astropy quantity, planet radius
        pl_mass: scalar astropy quantity, planet mass
        t_eq: scalar astropy quantity, planet equilibrium temperature
        omega: array-like astropy quantity, 1 or 2 elements
            Rotation frequency. If 2 elements, different for
            each hemisphere. The first element will be for
            the negative speed (blue-shifted), and the
            second for positive speed (red-shifted)
        resolution: scalar (float or int)
            spectral resolution
        left_val: float
            Transmission value at the bluer part of the kernel.
            Between 0 and 1. Default is 1 (no clouds)
        right_val: float
            Transmission value at the redest part of the kernel.
            Between 0 and 1. Default is 1 (no clouds)
        step_smooth: float
            fwhm of the gaussian kernel to smooth the clouds
            transmission step function. Default is 1.
        v_mid: float
            velocity where the step occurs in the clouds
            transmission step function. Default is 0.
        mu: scalar astropy quantity, mean molecular mass
        """
        if mu is None:
            mu = 2 * u.u
        g_surf = const.G * pl_mass / pl_rad**2
        scale_height = const.k_B * t_eq / (mu * g_surf)
        z_h = 5 * scale_height.to("m").value
        r_p = pl_rad.to("m").value
        res_elem = const.c / resolution
        res_elem = res_elem.to("m/s").value
        omega = omega.to("1/s").value
        self.res_elem = res_elem
        self.omega = omega
        self.z_h = z_h
        self.r_p = r_p
        self.left_val = left_val
        self.right_val = right_val
        self.step_smooth = step_smooth
        self.v_mid = v_mid

    def get_ker(self, n_os=None, pad=7):
        """
        n_os: scalar, oversampling (to sample the kernel)
        pad: scalar
            pad around the kernel in units of resolution
            elements. Values of the pad are set to zero.
        """
        res_elem = self.res_elem
        omega = self.omega
        z_h = self.z_h
        r_p = self.r_p
        clouds_args = (self.left_val, self.right_val, self.step_smooth, self.v_mid)
        ker_h_len = (r_p + z_h) * omega.max()
        v_max = ker_h_len + pad * res_elem
        if n_os is None:
            # Find adequate sampling
            delta_v = np.abs(z_h * omega.min() / 100)
        else:
            delta_v = res_elem / n_os
        v_grid = np.arange(-v_max, v_max, delta_v)
        v_grid -= np.mean(v_grid)
        if omega.size == 2:
            kernel = np.zeros_like(v_grid)
            idx_minus = v_grid < 0
            # Negative v
            args = (v_grid[idx_minus], omega[0], r_p, z_h)
            kernel[idx_minus] = _get_rot_ker_tr_v(*args)
            # Positive (remaining index)
            args = (v_grid[~idx_minus], omega[1], r_p, z_h)
            kernel[~idx_minus] = _get_rot_ker_tr_v(*args)
        else:
            kernel = _get_rot_ker_tr_v(v_grid, omega, r_p, z_h)
        # normalize
        kernel /= kernel.sum()

        # Get cloud transmission function
        idx_valid = kernel > 0
        clouds = np.ones_like(kernel) * np.nan
        clouds[idx_valid] = box_smoothed_step(v_grid[idx_valid], *clouds_args)
        kernel[idx_valid] = kernel[idx_valid] * clouds[idx_valid]

        return v_grid, kernel, clouds

    def degrade_ker(self, fwhm_km=None, **kwargs):
        """kwargs are passed to get_ker method"""
        if fwhm_km:
            fwhm = fwhm_km * 1000
            # print('Overwriting FWHM')
        else:
            fwhm = self.res_elem
        v_grid, rot_ker, _ = self.get_ker(**kwargs)
        norm = rot_ker.sum()
        gauss_ker = gauss(v_grid, 0.0, FWHM=fwhm)
        out_ker = np.convolve(rot_ker, gauss_ker, mode="same")
        out_ker /= out_ker.sum()
        out_ker *= norm
        # return v_grid, out_ker
        return v_grid, out_ker, gauss_ker, rot_ker

    def resample(self, res_sampling, **kwargs):
        """
        res_sampling: resolution of the sampling needed
        kwargs are passed to degrade_ker method
        """
        dv_new = cst.c / res_sampling
        # dv_new = dv_new.to('m/s').value
        v_grid, kernel, ker_g, ker_rot = self.degrade_ker(**kwargs)
        norm = kernel.sum()
        ker_spl = interp1d(v_grid, kernel, kind="linear")
        v_grid = np.arange(v_grid.min(), v_grid.max(), dv_new)
        out_ker = ker_spl(v_grid)
        out_ker /= out_ker.sum()
        out_ker *= norm
        return out_ker
        # return out_ker, v_grid

    def return_kernel(self, res_sampling, **kwargs):
        dv_new = cst.c / res_sampling

        v_grid, kernel, ker_g, ker_rot = self.degrade_ker(**kwargs)

        v_grid2 = np.arange(v_grid.min(), v_grid.max(), dv_new)
        v_grid2 -= np.mean(v_grid2)
        # if even, remove 1 point to make odd
        # want odd so that kernel is centered (not offset by half a pixel)
        if len(v_grid2) % 2 == 0:
            v_grid2 += dv_new / 2
            v_grid2 = v_grid2[:-1]

        norm = kernel.sum()
        ker_spl = interp1d(v_grid, kernel, kind="linear")
        out_ker_both = ker_spl(v_grid2)
        out_ker_both /= out_ker_both.sum()
        out_ker_both *= norm

        return out_ker_both

    def return_gauss_kernel(self, res_sampling, return_v_grid=False, **kwargs):
        dv_new = cst.c / res_sampling

        v_grid, kernel, ker_g, ker_rot = self.degrade_ker(**kwargs)

        v_grid2 = np.arange(v_grid.min(), v_grid.max(), dv_new)
        v_grid2 -= np.mean(v_grid2)
        # if even, remove 1 point to make odd
        # want odd so that kernel is centered (not offset by half a pixel)
        if len(v_grid2) % 2 == 0:
            v_grid2 += dv_new / 2
            v_grid2 = v_grid2[:-1]

        norm = ker_g.sum()
        ker_spl = interp1d(v_grid, ker_g, kind="linear")
        out_ker_gauss = ker_spl(v_grid2)
        out_ker_gauss /= out_ker_gauss.sum()
        out_ker_gauss *= norm

        if return_v_grid:
            return v_grid2, out_ker_gauss
        else:
            return out_ker_gauss

    def return_fwhm_kernel(self, res_sampling, fwhm_km, return_v_grid=False, **kwargs):
        dv_new = cst.c / res_sampling

        v_grid, kernel, ker_g, ker_rot = self.degrade_ker(fwhm_km=fwhm_km, **kwargs)

        v_grid2 = np.arange(v_grid.min(), v_grid.max(), dv_new)
        v_grid2 -= np.mean(v_grid2)
        # if even, remove 1 point to make odd
        # want odd so that kernel is centered (not offset by half a pixel)
        if len(v_grid2) % 2 == 0:
            v_grid2 += dv_new / 2
            v_grid2 = v_grid2[:-1]

        norm = ker_g.sum()
        ker_spl = interp1d(v_grid, ker_g, kind="linear")
        out_ker_gauss = ker_spl(v_grid2)
        out_ker_gauss /= out_ker_gauss.sum()
        out_ker_gauss *= norm

        if return_v_grid:
            return v_grid2, out_ker_gauss
        else:
            return out_ker_gauss

    def return_all_kernels(self, res_sampling, plot=False, **kwargs):
        """
        res_sampling: resolution of the sampling needed
        kwargs are passed to degrade_ker method
        """
        dv_new = cst.c / res_sampling
        # dv_new = dv_new.to('m/s').value
        v_grid, kernel, ker_g, ker_rot = self.degrade_ker(**kwargs)

        v_grid2 = np.arange(v_grid.min(), v_grid.max(), dv_new)
        v_grid2 -= np.mean(v_grid2)
        # if even, remove 1 point to make odd
        # want odd so that kernel is centered (not offset by half a pixel)
        if len(v_grid2) % 2 == 0:
            v_grid2 += dv_new / 2
            v_grid2 = v_grid2[:-1]

        norm = kernel.sum()
        ker_spl = interp1d(v_grid, kernel, kind="linear")
        out_ker_both = ker_spl(v_grid2)
        out_ker_both /= out_ker_both.sum()
        out_ker_both *= norm

        norm2 = ker_g.sum()
        ker_spl2 = interp1d(v_grid, ker_g, kind="linear")
        out_ker_gaus = ker_spl2(v_grid2)
        out_ker_gaus /= out_ker_gaus.sum()
        out_ker_gaus *= norm2

        norm3 = ker_rot.sum()
        ker_spl3 = interp1d(v_grid, ker_rot, kind="linear")
        out_ker_rot = ker_spl3(v_grid2)
        out_ker_rot /= out_ker_rot.sum()
        out_ker_rot *= norm3

        if plot:
            Rjup_km = 69911
            v = self.omega * (1.83 * Rjup_km)

            fig, ax = plt.subplots()
            # ax.plot(v_grid, kernel)
            # ax.plot(v_grid, ker_g)
            # ax.plot(v_grid, ker_rot)
            ax.plot(v_grid2 / 1e3, out_ker_both, label="Combined")
            ax.plot(
                v_grid2 / 1e3,
                out_ker_gaus,
                label="Instrumental ({})".format(int(cLight / self.res_elem)),
            )
            ax.plot(
                v_grid2 / 1e3,
                out_ker_rot,
                label="Rotational ({} km/s)".format(np.round(v, 2)),
            )
            ax.set_xlabel("dv (km/s)")
            ax.set_ylabel("Kernel")
            ax.legend(fontsize=14)

        return (
            v_grid2,
            out_ker_both,
            out_ker_gaus,
            out_ker_rot,
        )  # , v_grid, kernel, ker_g, ker_rot

    def show(self, **kwargs):
        res_elem = self.res_elem
        v_grid, kernel, clouds = self.get_ker(**kwargs)
        gauss_ker = gauss(v_grid, 0.0, FWHM=res_elem)
        gauss_ker /= gauss_ker.sum()
        # _, ker_degraded = self.degrade_ker(**kwargs)
        # _, ker_degraded, ker_rot = self.degrade_ker(**kwargs)
        _, ker_degraded, ker_g, ker_rot = self.degrade_ker(**kwargs)
        fig = plt.figure()
        plt.xlabel("dv [km/s]")

        lines = plt.plot(
            v_grid / 1e3,
            gauss_ker,
            "--",
            color="gray",
            label="Instrumental resolution element",
        )
        # plt.axvline(res_elem/2e3, linestyle='--', color='gray')
        # plt.axvline(-res_elem/2e3, linestyle='--', color='gray')
        new_line = plt.plot(v_grid / 1e3, kernel, label="Rotation kernel")
        lines += new_line
        new_line = plt.plot(v_grid / 1e3, ker_degraded, label="Instrumental * rotation")
        lines += new_line
        twin_ax = plt.gca().twinx()
        new_line = twin_ax.plot(
            v_grid / 1e3,
            clouds,
            label="Transmission clouds",
            color="g",
            alpha=0.5,
            linestyle="--",
        )
        lines += new_line

        labs = [line.get_label() for line in lines]
        # plt.legend(lines, labs, loc='upper left', fontsize=11)
        plt.xlabel("dv [km/s]")
        plt.ylabel("Kernel")
        twin_ax.set_ylim(-0.05, 1.05)


def _get_rot_ker_tr_v(v_grid, omega, r_p, z_h):
    x_v = v_grid / omega
    out = np.zeros_like(x_v)
    idx = np.abs(x_v) < r_p
    arg1 = (r_p + z_h) ** 2 - (x_v) ** 2
    arg2 = r_p**2 - (x_v) ** 2
    out[idx] = (np.sqrt(arg1[idx]) - np.sqrt(arg2[idx])) / z_h
    idx = (np.abs(x_v) >= r_p) & (np.abs(x_v) <= (r_p + z_h))
    arg1 = (r_p + z_h) ** 2 - (x_v) ** 2
    out[idx] = np.sqrt(arg1[idx]) / z_h
    return out


def gauss(x, mean=0, sigma=1, FWHM=None):
    if FWHM is not None:
        sigma = fwhm2sigma(FWHM)  # FWHM / (2 * np.sqrt(2 * np.log(2)))

    if x.ndim > 1:
        mean = np.expand_dims(mean, axis=-1)
        sigma = np.expand_dims(sigma, axis=-1)

    G = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / np.sqrt(2 * cst.pi * sigma**2)

    if x.ndim > 1:
        G /= G.sum(axis=-1)[:, None]  # Normalization
    else:
        G /= G.sum()

    return G


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def box_smoothed_step(v_grid, left_val, right_val, box_width, v_mid=0):
    # box width in units of dv
    dv_grid = v_grid[1] - v_grid[0]
    box_width = box_width / dv_grid

    # Gaussian smoothing kernel
    g_ker = Gaussian1DKernel(box_width).array

    # Apply step funnction
    y_val = np.zeros(v_grid.shape)
    y_val[v_grid < v_mid] = left_val
    y_val[v_grid >= v_mid] = right_val

    # Pad with connstants at the boundaries
    pad_size = np.ceil(len(g_ker) / 2).astype(int) - 1
    pad_left = np.repeat(y_val[0], pad_size)
    pad_right = np.repeat(y_val[-1], pad_size)
    y_val = np.concatenate([pad_left, y_val, pad_right])

    # Smooth the step function
    y_smooth = np.convolve(y_val, g_ker, mode="valid")

    # Normalize so that if the function was equal
    # to 1 everywhere
    y_ones = np.convolve(np.ones_like(y_val), g_ker, mode="valid")
    norm = y_ones.sum()
    y_smooth = y_smooth / norm * len(y_smooth)

    return y_smooth


def running_filter_1D(data1, func, length, cval=np.nan, verbose=True):
    data = deepcopy(data1)
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(fill_value=np.nan)

    if length % 2 != 1:
        raise ValueError("length should be odd.")

    n = len(data)
    sides = int((length - 1) / 2)
    data_ext = np.pad(data, sides, "constant", constant_values=cval)
    # Generate index for each interval to apply filter
    # Same as [np.arange(i,i+length) for i in range(n)] but quicker
    index = np.arange(length)[None, :] + np.arange(n)[:, None]

    try:
        out = func(data_ext[index], axis=1)
    except TypeError as e:
        if verbose:
            print(e, "so may take longer.")
            print("Consider using numpy func with axis keyword.")
        out = [func(x) for x in data_ext[index]]

    #     mask = ~np.isfinite(out) | np.isnan(data)

    return out


def running_filter_1D_by_part(data, func, length, part=int(3e6), **kwargs):
    n = len(data)
    step = int(part / length)
    out = [
        running_filter_1D(data[i : i + step], func, length, **kwargs)
        for i in range(0, n, step)
    ]

    return np.concatenate(out)


def running_filter(data, func, length, cval=np.nan, **kwargs):
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(fill_value=np.nan)

    # Make sure we do not exceed memory
    if length * data.shape[-1] > 2e6:
        filter_1D = running_filter_1D_by_part
    else:
        filter_1D = running_filter_1D

    if data.ndim > 1:
        out = [filter_1D(dat, func, length, cval=cval, **kwargs) for dat in data]
    else:
        out = filter_1D(data, func, length, cval=cval, **kwargs)

    mask = ~np.isfinite(out) | np.isnan(data)

    return np.ma.array(out, mask=mask)


def gaussian_filter(f, width=5):
    return astropy_convolve(
        f, Gaussian1DKernel(width), boundary="extend", mask=f.mask, preserve_nan=True
    )


"""
Function that decomposes a matrix in its first N components

    Input: A matrix and a number of components

    Output: the first N components and coefficients

"""


def PCA_decompose(matrix, n_components=None, return_variance=False):
    # if no components given, break down into all of them
    if n_components == None:
        n_components = np.shape(matrix)[1]
    pca = PCA(n_components=n_components)
    coefficients = pca.fit_transform(matrix)
    pcs = pca.components_
    N_components_removed = pca.n_components_
    # if return_variance:
    #     explained_variance = pca.explained_variance_ratio_
    #     return pcs, coefficients, explained_variance
    # else:
    return pcs, coefficients, N_components_removed


"""
Function that, given the components and coefficients, rebuilds the first N components of a matrix

    Input: The principal components and coefficients as outputted by PCA_decompose, and the number of components to be used to rebuild it

    Output: the reconstructed first N components of a matrix

"""


def PCA_rebuild(pcs, coefficients, n_pcs):
    comps = pcs[:n_pcs, :]
    rebuilt = 0.0
    for i in range(n_pcs):
        rebuilt += comps[i, :][None, :] * coefficients[:, i][:, None]

    return rebuilt


"""
Function that will find the index of the value in the array nearest a given value

    Input: array, number

    Output: index of value in array closest to that number
"""


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx  # array[idx]


"""
Function to save a pickle file

    Input: the object to be saved and the filename it is to be saved under

    Output: None
"""


def savepickle(obj, filename):
    """Save a pickle to a given filename.  If it can't be saved by
    pickle, return -1 -- otherwise return the file object.

    To save multiple objects in one file, use (e.g.) a dict:

       tools.savepickle(dict(a=[1,2], b='eggs'), filename)
    """
    # 2011-05-21 11:22 IJMC: Created from loadpickle
    # 2011-05-28 09:36 IJMC: Added dict example

    # good = True

    f = open(filename, "wb")
    pickle.dump(obj, f)
    f.close()

    return f
