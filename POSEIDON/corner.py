'''
Functions to generate corner plots.

Contributions from:
    
Johannes Buchner [PyMultiNest] (C) 2013-2019
Josh Speagle [Dynesty] (MIT licensed)
Ryan MacDonald [POSEIDON modifications] (2021-2025)

'''

from __future__ import absolute_import, unicode_literals, print_function, division

import numpy
import logging
import types
import math
import pymultinest
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter as norm_kde

from POSEIDON.utility import generate_latex_param_names, round_sig_figs
from POSEIDON.constants import R_J

try:
    str_type = types.StringTypes
    float_type = types.FloatType
    int_type = types.IntType
except:
    str_type = str
    float_type = float
    int_type = int

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Global dictionary to store maximum histogram values for each parameter
global_max_hist_values = {}

def _quantile(x, q, weights=None):
    '''
    Compute (weighted) quantiles from an input set of samples.

    Args:
        x (np.array of float):
            Input samples.
        q (np.array of float): 
            The list of quantiles to compute (ranging from 0 to 1).
        weights (np.array of float):
            The associated weight from each sample.
    
    Returns:
        quantiles (np.array of float):
            The weighted sample quantiles computed at 'q'.
    
    '''

    # Initial check
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    # If no weights provided, this simply calls 'np.percentile'
    if weights is None:

        return np.percentile(x, list(100.0 * q))

    # If weights are provided, compute the weighted quantiles
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)              # Sort samples
        sw = weights[idx]                # Sort weights
        cdf = np.cumsum(sw)[:-1]         # Compute CDF
        cdf /= cdf[-1]                   # Normalise CDF
        cdf = np.append(0, cdf)          # Ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()

        return quantiles


def resample_equal(samples, weights, rstate=None):
    '''
    Resample a new set of points from the weighted set of inputs, such that 
    they all have equal weight.

    Each input sample appears in the output array either
    'floor(weights[i] * nsamples)' or 'ceil(weights[i] * nsamples)' times,
    with 'floor' or 'ceil' randomly selected (weighted by proximity).

    Note: implements the systematic resampling method described in Hol, Schon, 
          and Gustafsson (2006): doi:10.1109/NSSPW.2006.4378824.

    Args:
        samples (np.array of float):
            Set of unequally weighted samples.
        weights (np.array of float):
            Corresponding weight of each sample.
        rstate (np.random.RandomState):
            Numpy 'RandomState' instance.
    
    Returns:
        equal_weight_samples (np.array of float):
            New set of samples with equal weights.
    
    '''

    if rstate is None:
        rstate = np.random

    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("Weights do not sum to 1.")

    # Make N subdivisions and choose positions with a consistent random offset.
    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples

    # Resample the data.
    idx = np.zeros(nsamples, dtype=np.int64)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    return samples[idx]


def _hist2d(x, y, smooth=0.02, span=None, weights=None, levels=None,
            ax=None, colour='gray', plot_datapoints=False, plot_density=True,
            plot_contours=True, no_fill_contours=False, fill_contours=True,
            contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
            **kwargs):
    '''
    Internal function called by the 'cornerplot' function to generate a 2D 
    histogram / contour of samples.

    Args:
        x (np.array of float):
            Sample positions in the first dimension.
        y (np.array of float):
            Sample positions in the second dimension.
        smooth (float):
            Gaussian smoothing factor for 2D contours.
        span (list of tuples or float):
            A list where each element is either a length-2 tuple containing
            lower and upper bounds or a float from `(0., 1.]` giving the
            fraction of (weighted) samples to include. If a fraction is provided,
            the bounds are chosen to be equal-tailed. If not specified, defaults 
            to +/- 5σ range. Example: span = [(0., 10.), 0.95, (5., 6.)].
        weights (np.array of float):
            Weights associated with the samples.
        levels (np.array of float):
            The contour levels to draw. Default are [1σ, 2σ, 3σ].
        ax (matplotlib axis object):
            A matplotlib axis instance on which to add the 2-D histogram.
            If not provided, a figure will be generated.
        colour (str):
            The matplotlib-style colour used to draw lines, colour cells,
            and contours. Default is 'gray'.
        plot_datapoints (bool):
            Whether to plot the individual data points. Default is False.
        plot_density (bool):
            Whether to draw the density colourmap. Default is True.
        plot_contours (bool):
            Whether to draw the contours. Default is True.
        no_fill_contours (bool):
            Whether to add absolutely no filling to the contours. This differs
            from 'fill_contours = False', which still adds a white fill at the
            densest points. Default is False.
        fill_contours (bool):
            Whether to fill the contours. Default is True.
        contour_kwargs (dict):
            Any additional keyword arguments to pass to the 'contour' method.
        contourf_kwargs (dict):
            Any additional keyword arguments to pass to the 'contourf' method.
        data_kwargs (dict):
            Any additional keyword arguments to pass to the 'plot' method when
            adding the individual data points.
    
    Returns:
        None.
    
    '''

    if ax is None:
        ax = plt.gca()

    # Determine plotting bounds
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the centre
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [colour, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This 'colour map' is the list of colours for the contour levels if the
    # contours are filled
    rgba_colour = colorConverter.to_rgba(colour)
    contour_cmap = [list(rgba_colour) for l in levels] + [rgba_colour]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # Initialize smoothing
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth, smooth]
    bins = []
    svalues = []
    for s in smooth:
        if isinstance(s, int_type):
            # If 's' is an integer, the weighted histogram has
            # 's' bins within the provided bounds
            bins.append(s)
            svalues.append(0.)
        else:
            # If 's' is a float, oversample the data relative to the
            # smoothing filter by a factor of 2, then use a Gaussian
            # filter to smooth the results
            bins.append(int(round(2. / s)))
            svalues.append(2.)

    # We'll make the 2D histogram to directly estimate the density
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, span)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range.")

    # Smooth the results
    if not np.all(svalues == 0.):
        H = norm_kde(H, svalues)

    # Compute the density levels
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centres
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", colour)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colours
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", colour)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(span[0])
    ax.set_ylim(span[1])


def cornerplot(results, span=None, quantiles=[0.1587, 0.5, 0.8413],
               colour_plt='purple', colour_quantile='blue', smooth_hist=30, 
               smooth_corr=0.02, hist_kwargs=None, hist2d_kwargs=None, 
               labels=None, param_names=None, label_kwargs=None,
               show_titles=True, title_kwargs=None, truths=None, 
               truth_colour='red', truth_kwargs=None, max_n_ticks=5, 
               top_ticks=False, use_math_text=False, verbose=False, 
               fig=None, model_i = None, 
               two_sigma_upper_limits = [], two_sigma_lower_limits = []):
    '''
    Generate a corner plot of the 1D and 2D marginalised posteriors.

    Args:
        results (dict):
            Results dictionary containing the samples and weights from a
            nested sampling retrieval.
        span (list of tuples or float):
            A list where each element is either a length-2 tuple containing
            lower and upper bounds or a float from `(0., 1.]` giving the
            fraction of (weighted) samples to include. If a fraction is provided,
            the bounds are chosen to be equal-tailed. If not specified, defaults
            to +/- 5σ range. Example: span = [(0., 10.), 0.95, (5., 6.)].
        quantiles (np.array of float):
            A list of fractional quantiles to overplot on the 1D marginalised
            posteriors as vertical dashed lines. Default is '[0.1587, 0.5, 0.8413]'
            (spanning the 68% / 1σ confidence interval).
        colour_plt (str):
            Matplotlib-style colour for the histograms and probability contours.
        colour_quantile (str):
            Matplotlib-style  for the vertical dashed quantile lines.
        smooth_hist (float or int):
            The standard deviation for the Gaussian kernel used to smooth the 1D
            histograms, expressed as a fraction of the span, if a float provided.
            If an integer is provided instead, this will instead default to a
            simple (weighted) histogram with 'bins=smooth'. Default is 30 bins.
        smooth_corr (float):
            The standard deviation for the Gaussian kernel used to smooth the 2D
            contours, expressed as a fraction of the span, if a float provided.
            Default is 2% smoothing.
        hist_kwargs (dict):
            Extra keyword arguments to send to the 1D histograms.
        hist2d_kwargs (dict):
            Extra keyword arguments to send to the 2D contours.
        labels (np.array of str):
            A list of names for each parameter. If not provided, the default
            name used when plotting will follow the math module 'x_i' style.
        param_names (np.array of str):
            List of parameter names used by POSEIDON for this retrieval.
        label_kwargs (dict):
            Extra keyword arguments that will be sent to the matplotlib axes
            'set_xlabel' and 'set_ylabel' methods.
        show_titles (bool):
            Whether to display a title above each 1D marginalised posterior
            showing the median along with the upper/lower bounds associated
            with the 1σ confidence interval. Default is True.
        title_kwargs (dict):
            Extra keyword arguments that will be sent to the matplotlib axes
            'set_title' command.
        truths (list of float):
            A list of reference values that will be overplotted on the traces
            and marginalised 1D histograms as solid horizontal/vertical lines.
            Individual values can be exempt using 'None'. Default is 'None'.
        truth_colour (str or list of str):
            Matplotlib-style colour (either a single colour or a different
            value for each subplot) used when plotting 'truths'. Default is 'red'.
        truth_kwargs (dict):
            Extra keyword arguments that will be used for plotting the vertical
            and horizontal lines with 'truths'.
        max_n_ticks (int):
            Maximum number of ticks allowed. Default is '5'.
        top_ticks (bool):
            Whether to label the top (rather than bottom) ticks. Default is False.
        use_math_text (bool):
            Whether the axis tick labels for very large/small exponents should be
            displayed as powers of 10 rather than using 'e'. Default is False.
        verbose (bool):
            Whether to print the values of the computed quantiles associated with
            each parameter. Default is False.
        fig (matplotlib figure object):
            If provided, overplot the traces and marginalised 1D histograms
            onto the provided figure. Otherwise, by default an internal figure
            is generated.
        two_sigma_upper_limits (list of bool):
            If True for any parameter, the 2σ upper limit will be plotted instead of the 1σ range.
        two_sigma_lower_limits (list of bool):
            If True for any parameter, the 2σ lower limit will be plotted instead of the 1σ range.

    Returns:
        cornerplot (matplotlib figure, matplotlib axes objects):
            Output corner plot.

    '''

    # Initialise values
    if quantiles is None:
        quantiles = []
    if truth_kwargs is None:
        truth_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if title_kwargs is None:
        title_kwargs = dict()
    if hist_kwargs is None:
        hist_kwargs = dict()
    if hist2d_kwargs is None:
        hist2d_kwargs = dict()

    # Set defaults
    hist_kwargs['alpha'] = hist_kwargs.get('alpha', 0.6)
    hist2d_kwargs['alpha'] = hist2d_kwargs.get('alpha', 0.6)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)
    truth_kwargs['alpha'] = truth_kwargs.get('alpha', 0.7)
    
    title_fmt_default = '.2f'

    # Extract weighted samples
    samples = results['samples']
    try:
        weights = np.exp(results['logwt'] - results['logz'][-1])
    except:
        weights = results['weights']

    # Deal with 1D results. A number of extra catches are also here in case 
    # users are trying to plot other results besides the `Results` instance
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"
    ndim, nsamps = samples.shape

    # Check weights
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Determine plotting bounds
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)

    # Set labels
    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    # Setting up smoothing
    if (isinstance(smooth_hist, int_type) or isinstance(smooth_hist, float_type)):
        smooth_hist = [smooth_hist for i in range(ndim)]
    if (isinstance(smooth_corr, int_type) or isinstance(smooth_corr, float_type)):
        smooth_corr = [smooth_corr for i in range(ndim)]

    # Setup axis layout (from `corner.py`)
    factor = 2.0  # size of side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # size of width/height margin
    plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
    dim = lbdim + plotdim + trdim  # total size
        
    # Initialize figure.
    if (fig is None and model_i is None) or (fig is None and model_i == 0):
        fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))
    elif fig is not None and model_i is not None:
        try:
            fig, axes = fig
            axes = np.array(axes).reshape((ndim, ndim))
            # show_titles = False
        except:
            raise ValueError("Mismatch between axes and dimension.")
    else:
        raise ValueError(
            "Must provide both existing figure and overplot index when overplotting."
        )

    # Format figure
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Determine the maximum value of all histograms for each parameter
    for i, x in enumerate(samples):
        param_name = param_names[i]
        sx_hist = smooth_hist[i]
        if isinstance(sx_hist, int_type):
            n, b = np.histogram(x, bins=sx_hist, weights=weights, range=np.sort(span[i]))
        else:
            bins = int(round(10. / sx_hist))
            n, b = np.histogram(x, bins=bins, weights=weights, range=np.sort(span[i]))
            n = norm_kde(n, 10.)
        
        # Update the global maximum histogram value for this parameter
        if model_i is not None:
            if param_name not in global_max_hist_values:
                global_max_hist_values[param_name] = max(n)
            else:
                global_max_hist_values[param_name] = max(global_max_hist_values[param_name], max(n))

    # Determine the maximum value of all histograms for each parameter
    max_hist_values = np.zeros(ndim)
    for i, x in enumerate(samples):
        sx_hist = smooth_hist[i]
        if isinstance(sx_hist, int_type):
            n, b = np.histogram(x, bins=sx_hist, weights=weights, range=np.sort(span[i]))
        else:
            bins = int(round(10. / sx_hist))
            n, b = np.histogram(x, bins=bins, weights=weights, range=np.sort(span[i]))
            n = norm_kde(n, 10.)
        max_hist_values[i] = max(max_hist_values[i], max(n))

    # Plotting
    for i, x in enumerate(samples):
        
        param_name = param_names[i]
        
        if np.shape(samples)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]

        # Plot the 1-D marginalized posteriors.

        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                   prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())
        
        # Label axes
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        if i < ndim - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.set_xlabel(labels[i], **label_kwargs)
            ax.xaxis.set_label_coords(0.5, -0.3)
        
        # Generate distribution
        sx_hist = smooth_hist[i]
        if isinstance(sx_hist, int_type):
            # If `sx` is an integer, plot a weighted histogram with
            # `sx` bins within the provided bounds.
            n, b, _ = ax.hist(x, bins=sx_hist, weights=weights, color=colour_plt,
                              range=np.sort(span[i]), **hist_kwargs)
        else:
            # If `sx` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / sx_hist))
            n, b = np.histogram(x, bins=bins, weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            b0 = 0.5 * (b[1:] + b[:-1])
            n, b, _ = ax.hist(b0, bins=b, weights=n,
                              range=np.sort(span[i]), color=colour_plt,
                              **hist_kwargs)
            
        # Set the y-axis limit based on the global maximum value
        if model_i is not None:
            ax.set_ylim([0., global_max_hist_values[param_name] * 1.05])
        else:
            ax.set_ylim([0., max_hist_values[i] * 1.05])

        # Plot quantiles
        if quantiles is not None and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)

            # Plot median and one sigma limits by default
            if ((len(two_sigma_upper_limits) == 0) and (len(two_sigma_lower_limits) == 0)):
                for i_q, q in enumerate(qs):
                    if (quantiles[i_q] == 0.5):   # For median
                        ax.axvline(q, lw=2, ls="-", alpha=0.7, color=colour_quantile)
                    else:
                        ax.axvline(q, lw=1, ls="dashed", color=colour_quantile)
                if verbose:
                    print("Quantiles:")
                    print(labels[i], [blob for blob in zip(quantiles, qs)])
            else:
                if (param_name in two_sigma_upper_limits):
                    qh = _quantile(x, [0.95], weights=weights)[0]
                    
                    # Plot arrow for upper limit
                    if model_i is not None:
                        arrow_y_pos = 0.9 - 0.05 * model_i
                    else:
                        arrow_y_pos = 0.9
                    ax.axvline(qh, lw=2, ls="-", color=colour_quantile)
                    ax.annotate('', xy=(qh, arrow_y_pos), 
                                xytext=((qh - (0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]))), arrow_y_pos), 
                                xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                                arrowprops=dict(facecolor=colour_quantile, color = colour_quantile, 
                                                edgecolor=colour_quantile, arrowstyle='<|-', lw=2, ls='-',
                                                shrinkA=0, shrinkB=0))
                    if verbose:
                        print("Quantiles:")
                        print(labels[i] + ": 2 sigma upper = " + str(qh))
                elif (param_name in two_sigma_lower_limits):
                    ql = _quantile(x, [0.05], weights=weights)[0]

                    # Plot arrow for lower limit
                    if model_i is not None:
                        arrow_y_pos = 0.9 - 0.05 * model_i
                    else:
                        arrow_y_pos = 0.9
                    ax.axvline(ql, lw=2, ls="-", color=colour_quantile)
                    ax.annotate('', xy=(ql + (0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0])), arrow_y_pos), 
                                xytext=(ql, arrow_y_pos), 
                                xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                                arrowprops=dict(facecolor=colour_quantile, color = colour_quantile, 
                                                edgecolor=colour_quantile, arrowstyle='-|>', lw=2, ls='-',
                                                shrinkA=0, shrinkB=0))
                    if verbose:
                        print("Quantiles:")
                        print(labels[i] + ": 2 sigma lower = " + str(ql))
                else:
                    for i_q, q in enumerate(qs):
                        if (quantiles[i_q] == 0.5):   # For median
                            ax.axvline(q, lw=2, ls="-", alpha=0.7, color=colour_quantile)
                        else:
                            ax.axvline(q, lw=1, ls="dashed", color=colour_quantile)
                    if verbose:
                        print("Quantiles:")
                        print(labels[i], [blob for blob in zip(quantiles, qs)])
        
        # Add truth value(s)
        if truths is not None and truths[i] is not None:
            try:
                [ax.axvline(t, color=truth_colour, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], color=truth_colour, **truth_kwargs)
        
        # Set titles
        if show_titles:
            title = None

            # Plot one sigma limits by default
            if ((len(two_sigma_upper_limits) == 0) and (len(two_sigma_lower_limits) == 0)):
                ql, qm, qh = _quantile(x, quantiles, weights=weights)
                q_minus, q_plus = qm - ql, qh - qm

            # Plot 2 sigma upper/lower limits where user flags the given parameter
            else:
                if (param_name in two_sigma_upper_limits):
                    qh = _quantile(x, [0.95], weights=weights)[0]
                elif (param_name in two_sigma_lower_limits):
                    ql = _quantile(x, [0.05], weights=weights)[0]
                else:
                    ql, qm, qh = _quantile(x, quantiles, weights=weights)
                    q_minus, q_plus = qm - ql, qh - qm

            if (("T" in param_name) or ("T_" in param_name)) and ("log" not in param_name):
                title_fmt = ".0f"
            elif (param_name == "a") or (param_name == "b"):  # for high res scaling parameters
                title_fmt = ".2f"
            elif "delta_rel" in param_name:
                title_fmt = ".0f"
            elif "R_p_ref" in param_name:
                label_exponent = round_sig_figs(np.floor(np.log10(np.abs(0.5 * (qh - ql)))), 1)
                if label_exponent == -2.0:
                    title_fmt = ".2f"
                elif label_exponent == -3.0:
                    title_fmt = ".3f"
                elif label_exponent == -4.0:
                    title_fmt = ".4f"
                else:
                    title_fmt = ".2f"
            elif "d" in param_name:
                title_fmt = ".3f"
            else:
                title_fmt = title_fmt_default

            fmt = "{{0:{0}}}".format(title_fmt).format

            # Title has +/- 1 sigma limits by default
            if ((len(two_sigma_upper_limits) == 0) and (len(two_sigma_lower_limits) == 0)):
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)

            # Title has 2 sigma upper/lower limits where user flags the given parameter
            else:
                if (param_name in two_sigma_upper_limits):
                    title = r"${{{0}}}$"
                    title = title.format(fmt(qh))
                    title = "{0} < {1}".format(labels[i], title)
                elif (param_name in two_sigma_lower_limits):
                    title = r"${{{0}}}$"
                    title = title.format(fmt(ql))
                    title = "{0} > {1}".format(labels[i], title)
                else:
                    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                    title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                    title = "{0} = {1}".format(labels[i], title)

            if model_i is not None:
                ax.text(0.5, 1 + model_i * 0.1,
                        title,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color=colour_plt,
                        transform=ax.transAxes,
                        **title_kwargs,
                       )
            else:
                ax.set_title(title, **title_kwargs)
        else:
            ax.set_title(None)

        for j, y in enumerate(samples):
            if np.shape(samples)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]

            # Plot the 2-D marginalized posteriors.

            # Setup axes.
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
            # Label axes.
            sf = ScalarFormatter(useMathText=use_math_text, useOffset=False)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if i < ndim - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.set_ylabel(labels[i], **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
            # Generate distribution.
            sx_corr = smooth_corr[j]
            sy_corr = smooth_corr[j]
            check_ix = isinstance(sx_corr, int_type)
            check_iy = isinstance(sy_corr, int_type)
            if check_ix and check_iy:
                fill_contours = False
                plot_contours = False
            else:
                fill_contours = True
                plot_contours = True
            hist2d_kwargs['fill_contours'] = hist2d_kwargs.get('fill_contours',
                                                               fill_contours)
            hist2d_kwargs['plot_contours'] = hist2d_kwargs.get('plot_contours',
                                                               plot_contours)
            _hist2d(y, x, ax=ax, span=[span[j], span[i]],
                    weights=weights, colour=colour_plt, smooth=[sy_corr, sx_corr],
                    **hist2d_kwargs)
            
            # Add truth values
            if truths is not None:
                if truths[j] is not None:
                    try:
                        [ax.axvline(t, color=truth_colour, **truth_kwargs)
                         for t in truths[j]]
                    except:
                        ax.axvline(truths[j], color=truth_colour,
                                   **truth_kwargs)
                if truths[i] is not None:
                    try:
                        [ax.axhline(t, color=truth_colour, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axhline(truths[i], color=truth_colour,
                                   **truth_kwargs)

    return (fig, axes)


def generate_cornerplot(planet, model, params_to_plot = None, 
                        retrieval_name = None, true_vals = None,
                        colour_scheme = '#984ea3', span = None, corner_name = None,
                        two_sigma_upper_limits = [], two_sigma_lower_limits = [],
                        ):
    '''
    Generate giant triangle plot of doom to visualise the results of a 
    POSEIDON retrieval.

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        model (dict):
            Dictionary containing the description of the POSEIDON model.
        retrieval_name (str):
            Optional retrieval name suffix after the model name.
        true_vals (list of float):
            True values of parameters to overplot.
        colour_scheme (str with hex code):
            Desired colour for the histograms and probability contours.
        span (list of tuples of float):
            Range to plot for each parameter (overrules default +/- 5σ range).
        corner_name (str):
            Optional file name prefix for the corner plot.
        two_sigma_upper_limits (list of str):
            List of parameters for which the 2σ upper limit will be plotted instead of the 1σ range.
        two_sigma_lower_limits (list of str):
            List of parameters for which the 2σ lower limit will be plotted instead of the 1σ range.
    
    Returns:
        fig (matplotlib figure object):
            Your new triangle plot of doom. Use responsibly!

    '''


    # Only generate a cornerplot using the first core
    if (rank == 0):

        # Confirm valid inputs
        if (len(two_sigma_lower_limits) != 0) and (len(two_sigma_lower_limits) != 0):
            for param in two_sigma_upper_limits:
                if (param in two_sigma_lower_limits):
                    raise Exception("Cannot have both a two sigma lower and upper limit for a given parameter.")
    
        # Unpack planet name
        planet_name = planet['planet_name']

        # Unpack model properties
        model_name = model['model_name']
        param_names = model['param_names']
        n_params = len(param_names)

        if (retrieval_name is None):
            retrieval_name = model_name
        else:
            retrieval_name = model_name + '_' + retrieval_name

        # Identify output directory location
        output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'

        # Load relevant output directory
        
        output_prefix = (output_dir + 'MultiNest_raw/' + retrieval_name + '-')

        # Run PyMultiNest analyser to extract posterior samples and model evidence
        a = pymultinest.Analyzer(n_params, outputfiles_basename = output_prefix,
                                 verbose = False)
        s = a.get_stats()
        
        print('Generating corner plot ...')

        # Extract quantities needed to use the Dynesty corner plotting script
        data = a.get_data()
        i = data[:, 1].argsort()[::-1]
        if params_to_plot is not None:
            indices_to_plot = [
                list(param_names).index(param) for param in params_to_plot
            ]
            samples = data[i][:, [2 + index for index in indices_to_plot]]
        else:
            params_to_plot = param_names
            samples = data[i, 2:]
        weights = data[i, 0]
        loglike = data[i, 1]
        Z = s["global evidence"]
        logvol = np.log(weights) + 0.5 * loglike + Z
        logvol = logvol - logvol.max()
        
        # Package results dictionary expected by the cornerplot function
        results = dict(samples=samples, weights=weights, logvol=logvol)
        
        # Calculate 2D levels for 1, 2, 3 sigma contours    
        levels = 1.0 - np.exp(-0.5 * np.array([1.0, 2.0, 3.0]) ** 2)

        # Generate LaTeX names for each parameter for plot axes
        params_latex = generate_latex_param_names(params_to_plot)
        
        # Generate corner plot
        fig, axes = cornerplot(results, 
                               quantiles=[0.1587, 0.5, 0.8413],  
                               smooth_hist=30, 
                               smooth_corr=0.02, 
                               colour_plt=colour_scheme,
                               colour_quantile='royalblue',
                               show_titles=True,\
                               labels=params_latex, 
                               param_names=params_to_plot,
                               truths=true_vals, 
                               span=span,
                               truth_colour='green',
                               label_kwargs={'fontsize': 18}, 
                               hist_kwargs={'histtype':'stepfilled','edgecolor': None},
                               hist2d_kwargs={'plot_contours': True,
                                              'fill_contours': True,
                                              'levels': levels,
                                              'plot_datapoints': False},
                               two_sigma_upper_limits=two_sigma_upper_limits,
                               two_sigma_lower_limits=two_sigma_lower_limits
                              )

        # Set plot file name
        if (corner_name is None):
            results_prefix = output_dir + 'results/' + retrieval_name
        else:
            results_prefix = output_dir + 'results/' + corner_name

        # Save corner plot in results directories
        plt.savefig(results_prefix + '_corner.pdf', bbox_inches='tight')

        return (fig, axes)


def generate_overplot(planet, models, params_to_plot = [], 
                      model_display_names = None, true_vals = None,
                      truth_colour = 'green', colour_schemes = ['purple', 'green'], 
                      span = None, overplot_name = None,
                      two_sigma_upper_limits = [], two_sigma_lower_limits = []):
    '''
    Generate overplotted giant triangle plot of doom to visualise the results 
    of multiple POSEIDON retrievals.

    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        model (dict):
            Dictionary containing the description of the POSEIDON model.
        retrieval_name (str):
            Optional retrieval name suffix after the model name.
        true_vals (list of float):
            True values of parameters to overplot.
        colour_scheme (str with hex code):
            Desired colour for the histograms and probability contours.
        span (list of tuples of float):
            Range to plot for each parameter (overrules default +/- 5σ range).
        overplot_name (str):
            Optional file name prefix for the overplot.
        two_sigma_upper_limits (list of str):
            List of parameters for which the 2σ upper limit will be plotted instead of the 1σ range.
        two_sigma_lower_limits (list of str):
            List of parameters for which the 2σ lower limit will be plotted instead of the 1σ range.

    Returns:
        fig (matplotlib figure object):
            Your new triangle plot of doom. Use responsibly!

    '''

    # Only generate a cornerplot using the first core
    if rank == 0:

        # Check for correct settings
        if ((len(colour_schemes) != len(models))):
            raise Exception("Number of colours does not match number of models.")
        if (len(two_sigma_lower_limits) != 0) and (len(two_sigma_lower_limits) != 0):
            for param in two_sigma_upper_limits:
                if (param in two_sigma_lower_limits):
                    raise Exception("Cannot have both a two sigma lower and upper limit for a given parameter.")

        # Unpack planet name
        planet_name = planet["planet_name"]

        existing_fig = None

        if model_display_names is None:
            model_display_names = [model["model_name"] for model in models]

        # Loop over each retrieval model
        for model_i, model in enumerate(models):

            # Unpack model properties
            model_name = model["model_name"]
            param_names = model["param_names"]
            n_params = len(param_names)

            # Identify output directory location
            output_dir = "./POSEIDON_output/" + planet_name + "/retrievals/"

            # Load relevant output directory

            output_prefix = output_dir + "MultiNest_raw/" + model_name + "-"

            # Run PyMultiNest analyser to extract posterior samples and model evidence
            a = pymultinest.Analyzer(
                n_params, outputfiles_basename=output_prefix, verbose=False
            )
            s = a.get_stats()

            print(f"Generating corner plot {model_i+1}...")

            # Extract quantities needed to use the Dynesty corner plotting script
            data = a.get_data()
            i = data[:, 1].argsort()[::-1]
            if params_to_plot is not None:
                indices_to_plot = [
                    list(param_names).index(param) for param in params_to_plot
                ]
                samples = data[i][:, [2 + index for index in indices_to_plot]]
            else:
                params_to_plot = param_names
                samples = data[i, 2:]
            weights = data[i, 0]
            loglike = data[i, 1]
            Z = s["global evidence"]
            logvol = np.log(weights) + 0.5 * loglike + Z
            logvol = logvol - logvol.max()

            # Package results dictionary expected by the cornerplot function
            results = dict(samples=samples, weights=weights, logvol=logvol)

            # Calculate 2D levels for 1, 2, 3 sigma contours
            levels = 1.0 - np.exp(-0.5 * np.array([1.0, 2.0, 3.0]) ** 2)

            # Generate LaTeX names for each parameter for plot axes
            params_latex = generate_latex_param_names(params_to_plot)

            # Generate corner plot
            existing_fig = cornerplot(results,
                                      quantiles=[0.1587, 0.5, 0.8413],
                                      smooth_hist=30,
                                      smooth_corr=0.02,
                                      colour_plt=colour_schemes[model_i],
                                      colour_quantile=colour_schemes[model_i],
                                      show_titles=True,
                                      labels=params_latex,
                                      param_names=params_to_plot,
                                      truths=true_vals,
                                      span=span,
                                      truth_colour=truth_colour,
                                      label_kwargs={"fontsize": 18},
                                      hist_kwargs={"histtype": "stepfilled", "edgecolor": None},
                                      hist2d_kwargs={"plot_contours": True,
                                                    "fill_contours": True,
                                                    "levels": levels,
                                                    "plot_datapoints": False,
                                                    },
                                      fig=existing_fig,
                                      model_i=model_i,
                                      two_sigma_upper_limits=two_sigma_upper_limits,
                                      two_sigma_lower_limits=two_sigma_lower_limits,
                                     )

            existing_fig[0].text(0.7, (0.75 + 0.05 * model_i),
                                 model_display_names[model_i],
                                 horizontalalignment="left",
                                 fontsize=20,
                                 color=colour_schemes[model_i],
                                 )

        # Save corner plot in results directory
        if (overplot_name is None):
            overplot_name = ''
            for model_display_name in model_display_names:
                overplot_name += model_display_name + '_'
            results_prefix = output_dir + "results/" + overplot_name
        else:
            results_prefix = output_dir + "results/" + overplot_name + '_'

        plt.savefig(results_prefix + "corner_overplot.pdf", bbox_inches="tight")

        return existing_fig
