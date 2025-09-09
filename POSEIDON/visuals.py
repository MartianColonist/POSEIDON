'''
Plotting routines to visualise POSEIDON output.

'''

import os
import numpy as np
import scipy.constants as sc
import colorsys
import matplotlib
from pylab import rcParams
import pymultinest
import matplotlib.style
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
                              ScalarFormatter, NullFormatter, MaxNLocator
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .corner import _quantile
from .atmosphere import count_atoms

plt.style.use('classic')
plt.rc('font', family = 'serif')
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['figure.facecolor'] = 'white'

import warnings

# Suppress invalid warning about x limits which sometimes occur for spectra plots
warnings.filterwarnings("ignore", message = "Attempted to set non-positive left " + 
                                            "xlim on a log-scaled axis.\n" + 
                                            "Invalid limit will be ignored.")

warnings.filterwarnings("ignore", message = "This figure includes Axes that are " +
                                            "not compatible with tight_layout, " +
                                            "so results might be incorrect.")

from .utility import bin_spectrum, generate_latex_param_names, round_sig_figs, \
                     confidence_intervals, create_directories, plot_collection
from .instrument import bin_spectrum_to_data
from .parameters import split_params
from .retrieval import get_retrieved_atmosphere
from .species_data import solar_abundances


# Define some more flexible linestyles for convenience
linestyles = {'loosely dotted':        (0, (1, 10)),
              'dotted':                (0, (1, 1)),
              'densely dotted':        (0, (1, 1)),
              'loosely dashed':        (0, (5, 10)),
              'dashed':                (0, (5, 5)),
              'densely dashed':        (0, (5, 1)),
              'loosely dashdotted':    (0, (3, 10, 1, 10)),
              'dashdotted':            (0, (3, 5, 1, 5)),
              'densely dashdotted':    (0, (3, 1, 1, 1)),
              'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
              'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
              'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
              }


def scale_lightness(colour_name, scale):
    ''' 
    Scale the lightness of a colour by the given factor.
    
    Args:
        colour_name (str): 
            The name of the colour to be scaled, in matplotlib colour format.
        scale (float): 
            The factor by which to scale the lightness of the colour (< 1 makes
            the colour darker).
    
    Returns:
        tuple: 
            A tuple containing the RGB values of the scaled colour.
    '''
    
    # Convert colour name to RBG value
    rgb = matplotlib.colors.ColorConverter.to_rgb(colour_name)
    
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Manipulate h, l, s values and return as RGB
    return colorsys.hls_to_rgb(h, min(1, l * scale), s = s)


def plot_transit(ax, R_p, R_s, b_p, r, T, phi, phi_edge, theta, theta_edge,
                 perspective, y_p = 0.0, plot_labels = True, show_star = False,
                 annotate_Rp = False, back_colour = 'white'):
    '''
    Subfunction used by the 'plot_geometry' function below. This function plots
    a 2D slice through an exoplanet and its atmosphere (to scale) from various 
    observer perspectives.

    Args:
        ax (matplotlib axis object):
            A matplotlib axis instance.
        R_p (float):
            White light planetary radius.
        r (3D np.array of float):
            Radial distance profile (m).
        T (3D np.array of float):
            Temperature profile (K).
        phi (np.array of float):
            Mid-sector angles (radians).
        phi_edge (np.array of float):
            Boundary angles for each sector (radians).
        dphi (np.array of float):
            Angular width of each sector (radians).
        theta (np.array of float):
            Mid-zone angles (radians).
        theta_edge (np.array of float):
            Boundary angles for each zone (radians).
        dtheta (np.array of float):
            Angular width of each zone (radians).
        perspective (str):
            Observer viewing perspective for 2D slice.
            (Options: terminator / day-night).
        y_p (float):
            Projected coordinate of planet centre from observer perspective
        plot_labels (bool):
            If False, removes text labels from the plot.
        show_star (bool):
            If True, plots the star in the background from observer perspective
        annotate_Rp (bool):
            If True, adds an arrow to the terminator perspective plot showing
            the radius of the planet (works best when show_star = True).
        back_colour (str):
            Background colour of figure.

    Returns:
        p (matplotlib PatchCollection):
            Patch collection containing atmosphere slices and temperature 
            colour bar. The figure itself is written to the provided axis.

    '''

    ax.set_facecolor(back_colour)

    ax.axis('equal')
    
    # First, average sectors / zones to find radii and temperatures for terminator plane and polar slices
    N_layers = r.shape[0]
    N_sectors_init = r.shape[1]
    N_zones_init = r.shape[2]
    
    # Pick out radial extents for north-south pole slice
    if (N_sectors_init > 1):
        r_pole = 0.5*(r[:,(N_sectors_init//2)-1,:] + r[:,(N_sectors_init//2),:])  # Average sectors adjacent to north pole
        T_pole = 0.5*(T[:,(N_sectors_init//2)-1,:] + T[:,(N_sectors_init//2),:])  # Average sectors adjacent to north pole
    else:
        r_pole = r[:,0,:]
        T_pole = T[:,0,:]
    
    # Pick out radial extents for terminator slice
    if (N_zones_init > 1):
        r_term = 0.5*(r[:,:,(N_zones_init//2)-1] + r[:,:,(N_zones_init//2)])  # Average zones adjacent to terminator plane
        T_term = 0.5*(T[:,:,(N_zones_init//2)-1] + T[:,:,(N_zones_init//2)])  # Average zones adjacent to terminator plane
    else:
        r_term = r[:,:,0]
        T_term = T[:,:,0]

    r_max = np.max(r)/R_p
    r_pole_max = np.max(r_pole)/R_p
    r_term_max = np.max(r_term)/R_p

    # Slice through the terminator plane
    if (perspective == 'terminator'):

        # In single sector case, can skip edge-finding calculations
        if (len(phi) == 1):
            phi_prime_edge_all = np.array([-90.0, 90.0, -90.0])
            
        else:
        
            # Convert phi into phi_prime = pi/2 - phi (angle w.r.t. y axis)
            phi_prime_N = np.pi/2.0 - phi
            phi_prime_S = (-1.0*phi_prime_N)[::-1]   # South pole symmetric about y axis
            phi_prime_all = np.append(phi_prime_N, phi_prime_S)
            
            # Same for sector boundaries
            phi_prime_edge_N = np.pi/2.0 - phi_edge[:-1]      # Don't need equator
            phi_prime_edge_S = -1.0*phi_prime_edge_N[::-1]    # South pole symmetric about y axis
            phi_prime_edge_all = np.append(phi_prime_edge_N, phi_prime_edge_S)  
            phi_prime_edge_all *= 180/np.pi                 # Convert to degrees for wedges plot
            phi_prime_edge_all = phi_prime_edge_all[:-1]   # Remove boundaries at equator
            phi_prime_edge_all[0] = phi_prime_edge_all[-1]
            
        # Plot star
        if (show_star == True):
            star = Circle((-y_p/R_p, -b_p/R_p), R_s/R_p, facecolor='gold', 
                           edgecolor='None', alpha=0.8)
            ax.add_artist(star)
        
        patches = []
        T_colors = []
        
        # After removing equatorial edges, compute number of azimuthal sectors
        N_sectors = len(phi_prime_edge_all)-1
        
        for j_all in range(N_sectors):   # Twice to cover North and South pole sectors
            
            # Find equivalent sector in northern hemisphere
            if (j_all >= N_sectors/2):
                j = N_sectors - j_all
            else:
                j = j_all
                
            # Special case for uniform atmosphere, where we only have one zone
            if (len(phi) == 1):
                j = 0
                
            # Show one layer in plot for every 11 atmospheric layers (~ each decade in pressure)
            for i in range(0, N_layers-11, 11):
                   
                # Plot full atmosphere for this sector
                planet_atm = Wedge((0.0, 0.0), r_term[i+11,j]/R_p, phi_prime_edge_all[j_all+1],  # Wedges plots w.r.t x axis and in degrees
                                   phi_prime_edge_all[j_all], edgecolor='None', 
                                   width = (r_term[i+11,j] - r_term[i,j])/R_p)
            
                patches.append(planet_atm)
                T_colors.append(np.mean(T_term[i:i+11,j])) 
            
            # Plot planet core (circular, below atmosphere)
            planet_core = Circle((0.0, 0.0), r[0,0,0]/R_p, facecolor='#1E202C', edgecolor='None')
            ax.add_artist(planet_core)
        
        ax.set_xlabel(r'y ($R_p$)', fontsize = 16)
        ax.set_ylabel(r'z ($R_p$)', fontsize = 16)
        
        # Plot atmosphere segment collection
        if (show_star == True):
            p = PatchCollection(patches, cmap=matplotlib.cm.RdBu_r, alpha=1.0, 
                                edgecolor=colorConverter.to_rgba('black', alpha=0.4), 
                                lw=0.1, zorder = 10, rasterized = True)

            # Add text label to indicate system geometry is shown
            ax.text(0.04, 0.96, 'System Geometry', horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes, color = 'black', fontsize = 16)

        else:
            p = PatchCollection(patches, cmap=matplotlib.cm.RdYlBu_r, alpha=1.0, 
                                edgecolor=colorConverter.to_rgba('black', alpha=0.1), 
                                lw=0.1, zorder = 10, rasterized = True)

        # Colour each segment according to atmospheric temperature
        colors = np.array(T_colors)
        p.set_array(colors)
        if (N_zones_init > 1):
            p.set_clim([0.98*np.min(T_pole), 1.02*np.max(T_pole)])
        elif (N_sectors_init > 1):
            p.set_clim([0.98*np.min(T_term), 1.02*np.max(T_term)])
        else:
            p.set_clim([0.98*np.min(T_pole), 1.02*np.max(T_pole)])

        ax.add_collection(p)
        
        # Add labels
        if (plot_labels == True):
            ax.set_title("Terminator Plane", fontsize = 16, pad=10)
            ax.text(0.04, 0.90, 'Evening', horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)
            ax.text(0.96, 0.90, 'Morning', horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)

        ax.set_xlim([-1.4*r_max, 1.4*r_max])
        ax.set_ylim([-1.4*r_max, 1.4*r_max])

        if (annotate_Rp == True):
            ax.annotate(s='', xy=(0.0, 0.0), xytext=(-1.0/np.sqrt(2), -1.0/np.sqrt(2)), 
                        arrowprops=dict(arrowstyle='<->', color='white', alpha=1.0), bbox=dict(fc='none', ec='none'))
            ax.text(-0.50, -0.15, r'$R_{\rm{p}}$', horizontalalignment = 'left', 
                    verticalalignment = 'top', fontsize = 14, color='white')


    # Slice through the north-south pole plane
    elif (perspective == 'day-night'):
    
        # In single zone case, can skip edge-finding calculations
        if (len(theta) == 1):
            theta_prime_edge_all = np.array([-90.0, 90.0, -90.0])
            
        else:
        
            # Convert theta into theta_prime = pi/2 - theta (angle w.r.t. x axis)
            theta_prime_N = np.pi/2.0 - theta
            theta_prime_S = (-1.0*theta_prime_N)[::-1]   # South pole symmetric about x axis
            theta_prime_all = np.append(theta_prime_N, theta_prime_S)
            
            # Same for sector boundaries
            theta_prime_edge_N = np.pi/2.0 - theta_edge[:-1]      # Don't need equator
            theta_prime_edge_S = -1.0*theta_prime_edge_N[::-1]    # South pole symmetric about y axis
            theta_prime_edge_all = np.append(theta_prime_edge_N, theta_prime_edge_S)  
            theta_prime_edge_all *= 180/np.pi                 # Convert to degrees for wedges plot
            
            theta_prime_edge_all = theta_prime_edge_all[:-1]   # Remove boundaries at equator
            theta_prime_edge_all[0] = theta_prime_edge_all[-1]
                
        patches = []
        T_colors = []
        
        # After removing equatorial edges, compute number of azimuthal sectors
        N_zones = len(theta_prime_edge_all)-1
        
        for k_all in range(N_zones):   # Twice to cover North and South pole sectors
            
            # Find equivalent sector in northern hemisphere
            if (k_all >= N_zones/2):
                k = N_zones - k_all
            else:
                k = k_all
                
            # Special case for uniform atmosphere, where we only have one zone
            if (len(theta) == 1):
                k = 0
                
            # Show one layer in plot for every 11 atmospheric layers (~ each decade in pressure)
            for i in range(0, N_layers-11, 11):
                
                planet_atm = Wedge((0.0, 0.0), r_pole[i+11,k]/R_p, theta_prime_edge_all[k_all+1],  # Wedges plots w.r.t x axis and in degrees
                                   theta_prime_edge_all[k_all], edgecolor='None', 
                                   width = (r_pole[i+11,k] - r_pole[i,k])/R_p)
            
                patches.append(planet_atm)
                T_colors.append(np.mean(T_pole[i:i+11,k]))

            # Plot planet core (circular, below atmosphere)
            planet_core = Circle((0.0, 0.0), r[0,0,0]/R_p, facecolor='black', edgecolor='None')
            ax.add_artist(planet_core)
        
        ax.set_title("Day-Night Transition", fontsize = 16, pad=10)
        
        ax.set_xlabel(r'x ($R_p$)', fontsize = 16)
        ax.set_ylabel(r'z ($R_p$)', fontsize = 16)
        
        # Plot atmosphere segment collection
        p = PatchCollection(patches, cmap=matplotlib.cm.RdYlBu_r, alpha=1.0, 
                            edgecolor=colorConverter.to_rgba('black', alpha=0.1), 
                            lw=0.1, zorder = 10, rasterized = True)
        
        # Colour each segment according to atmospheric temperature
        colors = np.array(T_colors)
        p.set_array(colors)
        if (N_zones_init > 1):
            p.set_clim([0.98*np.min(T_pole), 1.02*np.max(T_pole)])
        elif (N_sectors_init > 1):
            p.set_clim([0.98*np.min(T_term), 1.02*np.max(T_term)])
        else:
            p.set_clim([0.98*np.min(T_pole), 1.02*np.max(T_pole)])

        ax.add_collection(p)

        # Add labels
        if (plot_labels == True):
            ax.text(0.12, 0.97, 'Star', horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)
            ax.annotate('', xy=(0.04, 0.92), xytext=(0.30, 0.92), 
                        xycoords = 'axes fraction', textcoords = 'axes fraction',
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.8))
            ax.text(0.92, 0.97, 'Observer', horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)
            ax.annotate('', xy=(0.96, 0.92), xytext=(0.70, 0.92), 
                        xycoords = 'axes fraction', textcoords = 'axes fraction',
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.8))

            ax.text(0.05, 0.80, 'Day', horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)
            ax.text(0.50, 0.98, 'Terminator', horizontalalignment='center', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)
            ax.text(0.95, 0.80, 'Night', horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)

        ax.set_xlim([-1.4*r_max, 1.4*r_max])
        ax.set_ylim([-1.4*r_max, 1.4*r_max])

    return p
    
   
def plot_geometry(planet, star, model, atmosphere, plot_labels = True):
    '''
    Plots two 2D slice plots through the planetary atmosphere (to scale),
    coloured according to the temperature field. The left panel corresponds
    to a slice through the terminator plane, while the right panel is a
    slice through the north pole - observer plane.

    Args:
        planet (dict):
            POSEIDON planet properties dictionary.
        star (dict):
            POSEIDON stellar properties dictionary (currently unused).
        model (dict):
            POSEIDON model properties dictionary.
        atmosphere (dict):
            POSEIDON atmospheric properties dictionary.
        plot_labels (bool)
            If False, removes text labels from the plot.

    Returns:
        fig (matplotlib figure object):
            The geometric slice plot.

    '''

    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']
    R_p = planet['planet_radius']
    b_p = planet['planet_impact_parameter']
    R_s = star['R_s']
    r = atmosphere['r']
    T = atmosphere['T']
    phi = atmosphere['phi']
    dphi = atmosphere['dphi']
    phi_edge = atmosphere['phi_edge']
    theta = atmosphere['theta']
    dtheta = atmosphere['dtheta']
    theta_edge = atmosphere['theta_edge']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Create figure
    fig_combined = plt.figure(constrained_layout=True, figsize=(12, 6))  

    # Deploy the magic function
    axd = fig_combined.subplot_mosaic(
        """
        BA
        """
    )

    ax1 = axd['A']
    ax2 = axd['B']

    # Plot terminator plane on LHS axis
    p = plot_transit(ax1, R_p, R_s, b_p, r, T, phi, phi_edge, theta, 
                     theta_edge, 'terminator', plot_labels) 

    # Plot side perspective on RHS axis
    _ = plot_transit(ax2, R_p, R_s, b_p, r, T, phi, phi_edge, theta, 
                     theta_edge, 'day-night', plot_labels) 
    
    # Plot temperature colourbar
    cbaxes = fig_combined.add_axes([1.01, 0.131, 0.015, 0.786]) 
    cb = plt.colorbar(p, cax = cbaxes)  
    tick_locator = ticker.MaxNLocator(nbins=8)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.formatter.set_useOffset(False)
    cb.ax.set_title(r'$T \, \, \rm{(K)}$', horizontalalignment='left', pad=10)
    
  #  plt.tight_layout()

    # Write figure to file
    file_name = output_dir + planet_name + '_' + model_name + '_Geometry.png'

    plt.savefig(file_name, bbox_inches='tight', dpi=800)

    return fig_combined


def plot_geometry_spectrum_mixed(planet, star, model, atmosphere, spectra,
                                 y_p = 0.0, plot_labels = False, 
                                 show_star = True, annotate_Rp = True, 
                                 back_colour = 'black', data_properties = None,
                                 show_data = False, plot_full_res = True,
                                 bin_spectra = True, 
                                 R_to_bin = 100, wl_min = None, wl_max = None,
                                 y_min = None, y_max = None,
                                 y_unit = 'transit_depth', plt_label = None, 
                                 colour_list = [], spectra_labels = [], 
                                 data_colour_list = [], data_labels = [],
                                 data_marker_list = [], 
                                 data_marker_size_list = [], wl_axis = 'log', 
                                 figure_shape = 'default', 
                                 legend_location = 'upper right', legend_box = True):
    '''
    Plots two 2D slice plots through the planetary atmosphere (to scale).

    Args:
        planet (dict):
            POSEIDON planet properties dictionary.
        star (dict):
            POSEIDON stellar properties dictionary (currently unused).
        model (dict):
            POSEIDON model properties dictionary.
        atmosphere (dict):
            POSEIDON atmospheric properties dictionary.
        spectra (list):
            List of spectra to plot.
        y_p (float):
            Projected coordinate of planet centre from observer perspective.
        plot_labels (bool):
            If False, removes text labels from the plot.
        show_star (bool):
            If True, plots the star in the background from observer perspective.
        annotate_Rp (bool):
            If True, adds an arrow to the terminator perspective plot showing
            the radius of the planet (works best when show_star = True).
        back_colour (str):
            Background colour of figure.
        data_properties (dict, optional): 
            Dictionary containing data properties.
        show_data (bool): 
            If True, shows data on the right-hand side plot.
        plot_full_res (bool): 
            If True, shows full resolution spectra.
        bin_spectra (bool): 
            If True, bins spectra.
        R_to_bin (int): 
            Binning spectral resolution.
        wl_min (float): 
            Minimum wavelength for plotting.
        wl_max (float): 
            Maximum wavelength for plotting.
        y_min (float): 
            Minimum y-axis value for plotting.
        y_max (float): 
            Maximum y-axis value for plotting.
        y_unit (str): 
            Unit for y-axis. Default is 'transit_depth'.
        plt_label (str): 
            Label for the plot.
        colour_list (list): 
            List of colours for plotting spectra.
        spectra_labels (list): 
            List of labels for spectra.
        data_colour_list (list): 
            List of colours for data points.
        data_labels (list): 
            List of labels for data points.
        data_marker_list (list): 
            List of markers for data points.
        data_marker_size_list (list): 
            List of marker sizes for data points.
        wl_axis (str): 
            Axis for wavelength.
        figure_shape (str): 
            Shape of the figure.
        legend_location (str):
            Location of the legend.
        legend_box (bool):
            If True, shows legend box.

    Returns:
        fig (matplotlib figure object):
            The geometric slice plot.

    '''

    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    R_p = planet['planet_radius']
    b_p = planet['planet_impact_parameter']
    R_s = star['R_s']
    r = atmosphere['r']
    T = atmosphere['T']
    phi = atmosphere['phi']
    phi_edge = atmosphere['phi_edge']
    theta = atmosphere['theta']
    theta_edge = atmosphere['theta_edge']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1.5]) 
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Plot transit geometry on LHS axis
    p = plot_transit(ax1, R_p, R_s, b_p, r, T, phi, phi_edge, theta, 
                     theta_edge, 'terminator', y_p, plot_labels, show_star,
                     annotate_Rp, back_colour) 

    # Plot spectrum on RHS axis
    plot_spectra(spectra, planet, data_properties, show_data, plot_full_res, 
                 bin_spectra, R_to_bin, wl_min, wl_max, y_min, y_max, y_unit, 
                 plt_label, colour_list, spectra_labels, data_colour_list,
                 data_labels, data_marker_list, data_marker_size_list, wl_axis, 
                 figure_shape, legend_location, legend_box, ax2, save_fig = False)

    # Save Figure to file
    file_name = (output_dir + planet_name + '_' + plt_label + '_geometry_spectra.png')

    fig.savefig(file_name, bbox_inches = 'tight', dpi = 800)


def plot_PT(planet, model, atmosphere, show_profiles = [],
            PT_label = None, log_P_min = None, log_P_max = None, T_min = None,
            T_max = None, colour = 'darkblue', legend_location = 'lower left',
            ax = None):
    '''
    Plot the pressure-temperature (P-T) profiles defining the atmosphere.
    
    Args:
        planet (dict): 
            POSEIDON planet properties dictionary.
        model (dict): 
            POSEIDON model properties dictionary.
        atmosphere (dict): 
            Dictionary containing atmospheric properties.
        show_profiles (list, optional): 
            List of profiles to plot. Default is an empty list.
            For a 1D model, a single P-T profile is plotted. For 2D or 3D models,
            the user can specify the regions for which the P-T profiles should be
            plotted. This is handled through 'show_profiles'.
            Valid choices for 2D and 3D models:
                2D Day-Night: ['day', 'night', 'terminator']
                2D Evening-Morning: ['morning', 'evening', 'average']
                3D: ['evening-day', 'evening-night', 'evening-terminator', 
					 'morning-day', 'morning-night', 'morning-terminator',
					 'terminator-average']
            Any subset of the above can be passed via 'show_profiles'.
		PT_label (str, optional): 
            Label for the P-T profile.
		log_P_min (float, optional):
            Minimum value for the log10 pressure.
		log_P_max (float, optional):
            Maximum value for the log10 pressure.
		T_min (float, optional):
            Minimum temperature to plot.
		T_max (float, optional):
            Maximum temperature to plot.
		colour (str, optional):
            Colour of the plotted P-T profile.
		legend_location (str, optional):
            Location of the legend. Default is 'lower left'.
        ax (matplotlib axis object, optional):
            Matplotlib axis provided externally.
	
    Returns:
		fig (matplotlib figure object):
            The P-T profile plot.

    '''
    
    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']
    P = atmosphere['P']
    T = atmosphere['T']
    Atmosphere_dimension = model['Atmosphere_dimension']
    TwoD_type = model['TwoD_type']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Find minimum and maximum temperatures in atmosphere
    if (T_min == None):
        T_min = np.floor(np.min(T)/100)*100 - 200.0    # Round down to nearest 100

    if (T_max == None):
        T_max = np.ceil(np.max(T)/100)*100 + 200.0     # Round up to nearest 100
        
    # Find range to plot
    T_range = T_max - T_min    
    
    # Calculate appropriate axis spacing
    if (T_range >= 500.0):
        major_spacing = max(np.around((T_range/10), -2), 100.0)
    elif (T_range < 500.0):
        major_spacing = max(np.around((T_range/10), -1), 10.0)
        
    minor_spacing = major_spacing/10

    if (log_P_min == None):
        log_P_min = np.log10(np.min(P))
    if (log_P_max == None):
        log_P_max = np.log10(np.max(P))
    
    # create figure
    fig = plt.figure()

    if (ax == None):
        ax = plt.gca()
    else:
        ax = ax
    
    # Assign axis spacing
    xmajorLocator_PT = MultipleLocator(major_spacing)
    xminorLocator_PT = MultipleLocator(minor_spacing)
        
    ax.xaxis.set_major_locator(xmajorLocator_PT)
    ax.xaxis.set_minor_locator(xminorLocator_PT)
    
    # Plot P-T profiles
    
    # 1D temperature profile
    if (Atmosphere_dimension == 1):
        if (PT_label == None):
            ax.semilogy(T[:,0,0], P, lw=1.5, color = colour, label = model_name)
        else:
            ax.semilogy(T[:,0,0], P, lw=1.5, color = colour, label = PT_label)

    # 2D temperature profile
    elif (Atmosphere_dimension == 2):
        
        # Day-Night gradient
        if (TwoD_type == 'D-N'):
            
            # If user didn't specify which profiles to plot, plot all of them
            if (len(show_profiles) == 0):
                show_profiles = ['day', 'night', 'terminator']
            
            for profile_to_plot in show_profiles:
                
                if (profile_to_plot not in ['day', 'night', 'terminator']):
                    print ("Ignoring invalid profile '" + profile_to_plot +
                           "', since only 'day', 'night', and 'terminator' " +
                           "are valid for a 2D day-night model.")
                    
                if (profile_to_plot == 'day'):
                    ax.semilogy(T[:,0,0], P, lw=1.5, color = 'darkred', label='Day')
                if (profile_to_plot == 'night'):
                    ax.semilogy(T[:,0,-1], P, lw=1.5, color = 'darkblue', label='Night')
                if (profile_to_plot == 'terminator'):
                    ax.semilogy(0.5*(T[:,0,0]+T[:,0,-1]), P, lw=1.5, ls='--', 
                                color = 'darkorange', label='Terminator')
                   
        # Evening-Morning gradient
        elif (TwoD_type == 'E-M'):
            
            # If user didn't specify which profiles to plot, plot all of them
            if (len(show_profiles) == 0):
                show_profiles = ['evening', 'morning', 'average']
                
            for profile_to_plot in show_profiles:
                
                if (profile_to_plot not in ['evening', 'morning', 'average']):
                    print ("Ignoring invalid profile '" + profile_to_plot +
                           "', since only 'evening', 'morning', and 'average' " +
                           "are valid for a 2D evening-morning model.")
                    
                if (profile_to_plot == 'evening'):
                    ax.semilogy(T[:,0,0], P, lw=1.5, color = 'darkred', label='Evening')
                if (profile_to_plot == 'morning'):
                    ax.semilogy(T[:,-1,0], P, lw=1.5, color = 'darkblue', label='Morning')
                if (profile_to_plot == 'average'):
                    ax.semilogy(0.5*(T[:,0,0]+T[:,-1,0]), P, lw=1.5, ls='--', 
                                color = 'darkorange', label='Average')
                    
    # 3D temperature profile
    elif (Atmosphere_dimension == 3):
        
        # If user didn't specify which profiles to plot, plot all of them
        if (len(show_profiles) == 0):
            show_profiles = ['evening-day', 'evening-night', 'evening-terminator', 
                             'morning-day', 'morning-night', 'morning-terminator',
                             'terminator-average']
        
        for profile_to_plot in show_profiles:
            
            if (profile_to_plot not in ['evening-day', 'evening-night', 
                                        'evening-terminator', 'morning-day',
                                        'morning-night', 'morning-terminator',
                                        'terminator-average']):
                print ("Ignoring invalid profile' " + profile_to_plot +
                       "', since only combinations of 'evening-' or 'morning-' " +
                       "and 'day', 'night', or 'terminator' are valid for " +
                       "a 3D model. A global average can be plotted via " +
                       "'terminator-average'.")
                
            if (profile_to_plot == 'evening-day'):
                ax.semilogy(T[:,0,0], P, lw=1.5, color = 'crimson', label='Evening: Day')
            if (profile_to_plot == 'evening-night'):
                ax.semilogy(T[:,0,-1], P, lw=1.5, color = 'red', label='Evening: Night')
            if (profile_to_plot == 'evening-terminator'):
                ax.semilogy(0.5*(T[:,0,0]+T[:,0,-1]), P, lw=1.5, ls=':', 
                            color = 'darkred', label='Evening: Terminator')
            if (profile_to_plot == 'morning-day'):
                ax.semilogy(T[:,-1,0], P, lw=1.5, color = 'navy', label='Morning: Day')
            if (profile_to_plot == 'morning-night'):
                ax.semilogy(T[:,-1,-1], P, lw=1.5, color = 'cyan', label='Morning: Night')
            if (profile_to_plot == 'morning-terminator'):
                ax.semilogy(0.5*(T[:,-1,0]+T[:,-1,-1]), P, lw=1.5, ls=':', 
                            color = 'darkblue', label='Morning: Terminator')
            if (profile_to_plot == 'terminator-average'):
                ax.semilogy(0.25*(T[:,0,0]+T[:,0,-1]+T[:,-1,0]+T[:,-1,-1]), P, 
                            lw=1.5, ls='--', color = 'darkorange', label='Terminator Average')
            
    # Common plot settings for all profiles
    ax.invert_yaxis()            
    ax.set_xlabel(r'Temperature (K)', fontsize = 16)
    ax.set_xlim(T_min, T_max)
    ax.set_ylabel(r'Pressure (bar)', fontsize = 16)
    ax.set_ylim(np.power(10.0, log_P_max), np.power(10.0, log_P_min))  
    ax.tick_params(labelsize=12)
    
    # Add legend
    legend = ax.legend(loc=legend_location, shadow=True, prop={'size':10}, ncol=1, 
                       frameon=False, columnspacing=1.0)
    
    fig.set_size_inches(9.0, 9.0)
        
    # Write figure to file
    file_name = output_dir + planet_name + '_' + model_name + '_PT.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_chem(planet, model, atmosphere, plot_species = [], 
              colour_list = [], show_profiles = [],
              log_X_min = None, log_X_max = None,
              log_P_min = None, log_P_max = None,
              legend_title = None, legend_location = 'upper right'):  
    ''' 
    Plot the mixing ratio profiles defining the atmosphere.
    
    The user specifies which chemical species to plot via the list
    'plot_species'. The colours used for each species can be specified
    by the user via 'colour_list', or else default colours will be used.
    This function supports plotting up to 8 chemical species.

    Args:
        planet (dict): 
            Dictionary containing planet properties.
        model (dict):
            Dictionary containing model properties.
        atmosphere (dict):
            Dictionary containing atmospheric properties.
        plot_species (list, optional):
            List of chemical species to plot. If not specified, default to all 
            chemical species in the model (including bulk species).
        colour_list (list, optional):
            List of colours to use for each species in plot_species.
            Default is a predefined list, if the user doesn't provide one.
        show_profiles (list, optional):
            List of chemical profiles to plot. Default is an empty list.
            For a 1D model, a single P-T profile is plotted. For 2D or 3D models,
            the user can specify the regions for which the P-T profiles should be
            plotted. This is handled through 'show_profiles'.
            Valid choices for 2D and 3D models:
                2D Day-Night: ['day', 'night', 'terminator']
                2D Evening-Morning: ['morning', 'evening', 'average']
                3D: ['evening-day', 'evening-night', 'evening-terminator', 
					 'morning-day', 'morning-night', 'morning-terminator',
					 'terminator-average']
            Any subset of the above can be passed via 'show_profiles'.
        log_X_min (float, optional):
            Minimum log10 mixing ratio to plot. If not specified, the range is 
            calculated automatically.
        log_X_max (float, optional):
            Minimum log10 mixing ratio to plot. If not specified, the range is 
            calculated automatically.
        log_P_min (float, optional):
            Minimum log10 pressure to plot. If not specified, the range is 
            calculated automatically.
        log_P_max (float, optional):
            Minimum log10 pressure to plot. If not specified, the range is 
            calculated automatically.
        legend_title (str, optional):
            Title for the legend. Defaults to the model name if not provided.
        legend_location (str, optional):
            Location of the legend. Default is 'upper right'.

        Returns:
            fig (matplotlib figure object):
                Chemical mixing ratio plot.

    '''
    
    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']
    chemical_species = model['chemical_species']
    P = atmosphere['P']
    log_X = np.log10(atmosphere['X'])
    Atmosphere_dimension = model['Atmosphere_dimension']
    TwoD_type = model['TwoD_type']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'
    
    # If the user did not specify which species to plot, plot all of them
    if (len(plot_species) == 0):
        plot_species = chemical_species

    # Quick validity checks for plotting
    if (len(plot_species) > 8):
        raise Exception("Max number of concurrent species on plot is 8.\n"
                        "Please specify species to plot via plot_species = [LIST]")
    if ((len(colour_list) != 0) and (len(plot_species) != len(colour_list))):
        raise Exception("Number of colours does not match number of species.")
    for q, species in enumerate(plot_species):
        if (species not in chemical_species):
            raise Exception(species + " not included in this model.")

    # Find minimum and maximum mixing ratios in atmosphere
    if (log_X_min == None):
        log_X_min = -1
        for q, species in enumerate(plot_species):
            log_X_min = min(log_X_min, (np.floor(np.min(log_X[chemical_species == species,:,0,0][0])) - 1.0))

    if (log_X_max == None):
        log_X_max = -10
        for q, species in enumerate(plot_species):
            log_X_max = max(log_X_max, (min((np.ceil(np.max(log_X[chemical_species == species,:,0,0][0])) + 1.0), 0.0)))
    
    # When range is small, extend axes +/- 1 dex either side
    if (log_X_min == log_X_max):
        log_X_min = log_X_min - 1.0
        log_X_max = log_X_max + 1.0
        
    # Find range to plot
    log_X_range = log_X_max - log_X_min
    
    # Calculate appropriate axis spacing
    if (log_X_range <= 10):
        major_spacing = 1.0
    else:
        major_spacing = 2.0
    minor_spacing = major_spacing/10

    if (log_P_min == None):
        log_P_min = np.log10(np.min(P))
    if (log_P_max == None):
        log_P_max = np.log10(np.max(P))
    
    # Define colours for mixing ratio profiles (default or user choice)
    if (len(colour_list) == 0):   # If user did not specify a custom colour list
        colours = ['royalblue', 'darkgreen', 'magenta', 'crimson', 'darkgrey', 
                   'black', 'darkorange', 'navy']
    else:
        colours = colour_list
    
    # Find LaTeX code for each chemical species to plot
    latex_species = generate_latex_param_names(plot_species)
    
    # create figure
    fig = plt.figure()  
    ax = plt.gca()
    
    # Assign axis spacing
    xmajorLocator_X = MultipleLocator(major_spacing)
    xminorLocator_X = MultipleLocator(minor_spacing)
        
    ax.xaxis.set_major_locator(xmajorLocator_X)
    ax.xaxis.set_minor_locator(xminorLocator_X)
    
    # Plot mixing ratio profiles
    
    # 1D temperature profile
    if (Atmosphere_dimension == 1):

        for q, species in enumerate(plot_species):
            ax.semilogy(log_X[chemical_species == species,:,0,0][0], P, 
                        lw=1.5, color = colours[q], label=latex_species[q])

    # 2D temperature profile
    elif (Atmosphere_dimension == 2):
        
        if (len(show_profiles) == 0):
            raise Exception("For 2D or 3D models, you need to specify " +
                            "which regions to plot.")
        
        for q, species in enumerate(plot_species):

            # Day-Night gradient
            if (TwoD_type == 'D-N'):
                    
                # If dayside and nightside the same, only plot one profile
                if (np.all(log_X[chemical_species == species,:,0,0][0] ==
                           log_X[chemical_species == species,:,0,-1][0])):
                    ax.semilogy(log_X[chemical_species == species,:,0,0][0],
                                P, lw=1.5, ls='-', color = colours[q],
                                label=latex_species[q])

                # Do the same for bulk species
                elif (species in ['H2', 'He']):
                    ax.semilogy(log_X[chemical_species == species,:,0,0][0],
                                P, lw=1.5, ls='-', color = colours[q],
                                label=latex_species[q])
                
                # Otherwise, plot user choices
                else:
                    
                    for profile_to_plot in show_profiles:
                        
                        if (profile_to_plot not in ['day', 'night', 'terminator']):
                            print ("Ignoring invalid profile '" + profile_to_plot +
                                   "', since only 'day', 'night', and 'terminator' " +
                                   "are valid for a 2D day-night model.")

                        if (profile_to_plot == 'day'):
                            ax.semilogy(log_X[chemical_species == species,:,0,0][0], 
                                        P, lw=3.0, ls=':', color = colours[q], 
                                        label=latex_species[q] + ' (Day)')
                        if (profile_to_plot == 'night'):
                            ax.semilogy(log_X[chemical_species == species,:,0,-1][0], 
                                        P, lw=1.5, ls='--', color = colours[q],
                                        label=latex_species[q] + ' (Night)')
                        if (profile_to_plot == 'terminator'):
                            ax.semilogy(0.5*(log_X[chemical_species == species,:,0,0][0] +
                                             log_X[chemical_species == species,:,0,-1][0]), 
                                        P, lw=1.5, ls='-', color = colours[q],
                                        label=latex_species[q] + ' (Terminator)')
                            
            # Evening-Morning gradient
            elif (TwoD_type == 'E-M'):
                    
                # If evening and morning the same, only plot one profile
                if (np.all(log_X[chemical_species == species,:,0,0][0] ==
                           log_X[chemical_species == species,:,-1,0][0])):
                    ax.semilogy(log_X[chemical_species == species,:,0,0][0],
                                P, lw=1.5, ls='-', color = colours[q],
                                label=latex_species[q])

                # Do the same for bulk species
                elif (species in ['H2', 'He']):
                    ax.semilogy(log_X[chemical_species == species,:,0,0][0],
                                P, lw=1.5, ls='-', color = colours[q],
                                label=latex_species[q])
                
                # Otherwise, plot user choices
                else:
                    
                    for profile_to_plot in show_profiles:
                        
                        if (profile_to_plot not in ['evening', 'morning', 'average']):
                            print ("Ignoring invalid profile '" + profile_to_plot +
                                   "', since only 'evening', 'morning', and 'average' " +
                                   "are valid for a 2D evening-morning model.")
                            
                        if (profile_to_plot == 'evening'):
                            ax.semilogy(log_X[chemical_species == species,:,0,0][0], 
                                        P, lw=3.0, ls=':', color = colours[q], 
                                        label=latex_species[q] + ' (Evening)')
                        if (profile_to_plot == 'morning'):
                            ax.semilogy(log_X[chemical_species == species,:,-1,0][0], 
                                        P, lw=1.5, ls='--', color = colours[q],
                                        label=latex_species[q] + ' (Morning)')
                        if (profile_to_plot == 'average'):
                            ax.semilogy(0.5*(log_X[chemical_species == species,:,0,0][0] +
                                             log_X[chemical_species == species,:,-1,0][0]), 
                                        P, lw=1.5, ls='-', color = colours[q],
                                        label=latex_species[q] + ' (Average)')
  
    # 3D temperature profile
    elif (Atmosphere_dimension == 3):
        
        if (len(show_profiles) == 0):
            raise Exception("For 2D or 3D models, you need to specify " +
                            "which regions to plot.")
        
        for q, species in enumerate(plot_species):
            
            # If all profiles the same, only plot one profile
            if ((np.all(log_X[chemical_species == species,:,0,0][0] ==
                        log_X[chemical_species == species,:,-1,0][0])) and
                (np.all(log_X[chemical_species == species,:,0,0][0] ==
                       log_X[chemical_species == species,:,0,-1][0]))):
                ax.semilogy(log_X[chemical_species == species,:,0,0][0],
                            P, lw=1.5, ls='-', color = colours[q],
                            label=latex_species[q])
            
            # Otherwise, plot user choices
            else:

                for profile_to_plot in show_profiles:
                    
                    if (profile_to_plot not in ['evening-day', 'evening-night', 
                                                'evening-terminator', 'morning-day',
                                                'morning-night', 'morning-terminator',
                                                'terminator-average']):
                        print ("Ignoring invalid profile '" + profile_to_plot +
                               "', since only combinations of 'evening-' or 'morning-' " +
                               "and 'day', 'night', or 'terminator' are valid for " +
                               "a 3D model. A global average can be plotted via " +
                               "'terminator-average'.")
                        
                    if (profile_to_plot == 'evening-day'):
                        ax.semilogy(log_X[chemical_species == species,:,0,0][0], 
                                    P, lw=1.5, ls=linestyles['loosely dashed'], color = colours[q], 
                                    label=latex_species[q] + ' (Evening: Day)')
                    if (profile_to_plot == 'evening-night'):
                        ax.semilogy(log_X[chemical_species == species,:,0,-1][0], 
                                    P, lw=1.5, ls=linestyles['dashed'], color = colours[q], 
                                    label=latex_species[q] + ' (Evening: Night)')
                    if (profile_to_plot == 'evening-terminator'):
                        ax.semilogy(0.5*(log_X[chemical_species == species,:,0,0][0] +
                                         log_X[chemical_species == species,:,0,-1][0]), 
                                    P, lw=1.5, ls=linestyles['densely dashed'], color = colours[q], 
                                    label=latex_species[q] + ' (Evening: Terminator)')
                    if (profile_to_plot == 'morning-day'):
                        ax.semilogy(log_X[chemical_species == species,:,-1,0][0], 
                                    P, lw=1.5, ls=linestyles['loosely dashdotted'], color = colours[q], 
                                    label=latex_species[q] + ' (Morning: Day)')
                    if (profile_to_plot == 'morning-night'):
                        ax.semilogy(log_X[chemical_species == species,:,-1,-1][0], 
                                    P, lw=1.5, ls=linestyles['dashdotted'], color = colours[q], 
                                    label=latex_species[q] + ' (Morning: Night)')
                    if (profile_to_plot == 'morning-terminator'):
                        ax.semilogy(0.5*(log_X[chemical_species == species,:,-1,0][0] + 
                                         log_X[chemical_species == species,:,-1,-1][0]), 
                                    P, lw=1.5, ls=linestyles['densely dashdotted'], color = colours[q], 
                                    label=latex_species[q] + ' (Morning: Terminator)')
                    if (profile_to_plot == 'terminator-average'):
                        ax.semilogy(0.25*(log_X[chemical_species == species,:,0,0][0] + 
                                          log_X[chemical_species == species,:,0,-1][0] + 
                                          log_X[chemical_species == species,:,-1,0][0] + 
                                          log_X[chemical_species == species,:,-1,-1][0]), 
                                    P, lw=1.5, ls=linestyles['dotted'], color = colours[q], 
                                    label=latex_species[q] + ' (Terminator Average)')
  
    # Common plot settings for all profiles
    ax.invert_yaxis()            
    ax.set_xlabel(r'Mixing Ratios (log $X_{\rm{i}}$)', fontsize = 20)
    ax.set_xlim(log_X_min, log_X_max)  
    ax.set_ylabel(r'Pressure (bar)', fontsize = 20)
    ax.set_ylim(np.power(10.0, log_P_max), np.power(10.0, log_P_min))  
    ax.tick_params(labelsize=12)

    # Add legend
    legend = ax.legend(loc=legend_location, shadow=True, prop={'size':14}, 
                       frameon=True, columnspacing=1.0, title = legend_title,
                       title_fontsize = 16)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    if (legend_location == 'upper left'):
        legend.set_bbox_to_anchor([0.02, 0.98], transform=None)
    elif (legend_location == 'upper right'):
        legend.set_bbox_to_anchor([0.98, 0.98], transform=None)
    elif (legend_location == 'lower left'):
        legend.set_bbox_to_anchor([0.02, 0.02], transform=None)
    elif (legend_location == 'lower right'):
        legend.set_bbox_to_anchor([0.98, 0.02], transform=None)
    
    fig.set_size_inches(9.0, 9.0)

    # Write figure to file
    file_name = output_dir + planet_name + '_' + model_name + '_chem.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def set_spectrum_wl_ticks(wl_min, wl_max, wl_axis = 'log'):
    '''
    Calculates default x axis tick spacing for spectra plots in POSEIDON.
    
    Args:
        wl_min (float):
            The minimum wavelength to plot.
        wl_max (float):
            The maximum wavelength to plot.
        wl_axis (str, optional):
            The type of x-axis to use ('log' or 'linear').
            
    Returns:
        np.array:
            The x axis tick values for the given wavelength range.

    '''

    wl_range = wl_max - wl_min

    if (wl_max < wl_min):
        raise Exception("Error: max wavelength must be greater than min wavelength.")

    # For plots over a wide wavelength range
    if (wl_range > 0.2):
        if (wl_max <= 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
            wl_ticks_2 = np.array([])
            wl_ticks_3 = np.array([])
            wl_ticks_4 = np.array([])
        elif (wl_max <= 2.0):
            if (wl_min < 1.0):
                wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
                wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
            else:
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*5)/5.0, round_sig_figs(wl_max, 2)+0.01, 0.2)))
            wl_ticks_3 = np.array([])
            wl_ticks_4 = np.array([])
        elif (wl_max <= 3.0):
            if (wl_min < 1.0):
                wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
                wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.5)
            else:
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*5)/5.0, round_sig_figs(wl_max, 2)+0.01, 0.2)))
            wl_ticks_3 = np.array([])
            wl_ticks_4 = np.array([])
        elif (wl_max <= 5.0):
            if (wl_min < 1.0):
                wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
                wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
                wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
            elif (wl_min < 3.0):
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*5)/5.0, 3.0, 0.2)))
                if (wl_axis == 'log'):
                    wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 0.5)
                elif (wl_axis == 'linear'):
                    wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
            else:
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.array([])
                wl_ticks_3 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*5)/5.0, round_sig_figs(wl_max, 2)+0.01, 0.2)))
            wl_ticks_4 = np.array([])
        elif (wl_max <= 10.0):
            if (wl_min < 1.0):
                if (wl_axis == 'log'):
                    wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
                    wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
                    wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
                elif (wl_axis == 'linear'):
                    wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 1.0)
                    wl_ticks_2 = np.arange(1.0, 3.0, 1.0)            
                    wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
            elif (wl_min < 3.0):
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*2)/2.0, 3.0, 0.5)))
                if (wl_axis == 'log'):
                    wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
                elif (wl_axis == 'linear'):
                    wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 0.5)
            else:
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.array([])
                if (wl_axis == 'log'):
                    wl_ticks_3 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*2)/2.0, round_sig_figs(wl_max, 2)+0.01, 0.5)))
                elif (wl_axis == 'linear'):
                    wl_ticks_3 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*2)/2.0, round_sig_figs(wl_max, 2)+0.01, 0.5)))
            wl_ticks_4 = np.array([])
        else:
            if (wl_min < 1.0):
                if (wl_axis == 'log'):
                    wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
                    wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
                    wl_ticks_3 = np.arange(3.0, 10.0, 1.0)
                    wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 2)+0.01, 2.0)
                elif (wl_axis == 'linear'):
                    wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 1.0)
                    wl_ticks_2 = np.arange(1.0, 3.0, 1.0)            
                    wl_ticks_3 = np.arange(3.0, 10.0, 1.0)
                    wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 3)+0.01, 1.0)
            elif (wl_min < 3.0):
                wl_ticks_1 = np.array([])
                if (wl_axis == 'log'): 
                    wl_ticks_2 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min*2)/2.0, 3.0, 0.5)))
                    wl_ticks_3 = np.arange(3.0, 10.0, 1.0)
                    wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 2)+0.01, 2.0)
                elif (wl_axis == 'linear'):
                    wl_ticks_2 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min), 3.0, 1.0)))
                    wl_ticks_3 = np.arange(3.0, 10.0, 1.0)
                    wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 3)+0.01, 1.0)
            elif (wl_min < 10.0):
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.array([])
                if (wl_axis == 'log'):
                    wl_ticks_3 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min), 10.0, 1.0)))
                    wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 3)+0.01, 2.0)
                elif (wl_axis == 'linear'):
                    wl_ticks_3 = np.concatenate(([round_sig_figs(wl_min, 2)], np.arange(np.ceil(wl_min), 10.0, 1.0)))
                    wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 3)+0.01, 1.0)
            else:
                wl_ticks_1 = np.array([])
                wl_ticks_2 = np.array([])
                wl_ticks_3 = np.array([])
                if (wl_axis == 'log'):
                    wl_ticks_4 = np.concatenate(([round_sig_figs(wl_min, 3)], np.arange(np.ceil(wl_min), round_sig_figs(wl_max, 3)+0.01, 1.0)))
                elif (wl_axis == 'linear'):
                    wl_ticks_4 = np.concatenate(([round_sig_figs(wl_min, 3)], np.arange(np.ceil(wl_min*2)/2.0, round_sig_figs(wl_max, 3)+0.01, 0.5)))

        wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3, wl_ticks_4))

    # For high-resolution (zoomed in) spectra
    else:

        # Aim for 10 x-axis labels
        wl_spacing = round_sig_figs((wl_max - wl_min), 1)/10
        
        major_exponent = round_sig_figs(np.floor(np.log10(np.abs(wl_spacing))), 1)
        
        # If last digit of x labels would be 3,6,7,8,or 9, bump up to 10
        if (wl_spacing > 5*np.power(10, major_exponent)):
            wl_spacing = 1*np.power(10, major_exponent+1)
        elif (wl_spacing == 3*np.power(10, major_exponent)):
            wl_spacing = 2*np.power(10, major_exponent)

        wl_ticks = np.arange(round_sig_figs(wl_min, 3), round_sig_figs(wl_max, 3)+0.0001, wl_spacing)

    return wl_ticks


def plot_spectra(spectra, planet, data_properties = None, show_data = False,
                 plot_full_res = True, bin_spectra = True, R_to_bin = 100,
                 plt_label = None, show_planet_name = True, 
                 wl_min = None, wl_max = None, y_min = None, y_max = None,
                 y_unit = 'transit_depth', colour_list = [],
                 spectra_labels = [], data_colour_list = [],
                 data_labels = [], data_marker_list = [],
                 data_marker_size_list = [], data_alpha_list = [],
                 data_eline_alpha_list = [], data_edge_width_list = [],
                 data_eline_colour_list = [], data_eline_width_list = [],
                 line_width_list = [], line_style_list = [], line_alpha_list = [],
                 text_annotations = [], annotation_pos = [],
                 err_colour = 'black', wl_axis = 'log', 
                 figure_shape = 'default', 
                 show_legend = True, legend_location = 'upper right',
                 legend_box = True, legend_line_size = [], legend_n_columns = 0,
                 ax = None, save_fig = True, model = None, 
                 show_data_bin_width = True, show_data_cap = True,
                 add_retrieved_offsets = False, verbose_offsets = True,
                 add_retrieved_error_inflation = False,
                 xlabels = True, ylabels = True, 
                 x_tick_fontsize = 12, x_label_fontsize = 16,
                 y_tick_fontsize = 12, y_label_fontsize = 16,
                 legend_fontsize = 10, plt_label_fontsize = 14,
                 planet_name_fontsize = 16, plot_style = 'standard',
                 fill_between = [], fill_between_alpha = 0.5, fill_to_spectrum = [],
                 ):
    ''' 
    Plot a collection of individual model spectra. This function can plot
    transmission or emission spectra, according to the user's choice of 'y_unit'.

    Args:
        spectra (list of tuples):
            A list of model spectra to be plotted, each with the format
            (wl, spectrum).
        planet (dict):
            POSEIDON planet properties dictionary.
        data_properties (dict, optional):
            POSEIDON observational data properties dictionary.
        show_data (bool, optional):
            Flag indicating whether to plot the observational data.
        plot_full_res (bool, optional):
            Flag indicating whether to plot full resolution model spectra.
        bin_spectra (bool, optional):
            Flag indicating whether to bin model spectra to the resolution
            specified by 'R_to_bin'.
        R_to_bin (int, optional):
            Spectral resolution (R = wl/dwl) to bin the model spectra to.
        plt_label (str, optional):
            The label for the plot.
        show_planet_name (bool, optional):
            Flag indicating whether to include the planet name in the top left.
        wl_min (float, optional):
            The minimum wavelength to plot.
        wl_max (float, optional):
            The maximum wavelength to plot.
        y_min (float, optional):
            The minimum value for the y-axis.
        y_max (float, optional):
            The maximum value for the y-axis.
        y_unit (str, optional):
            The unit of the y-axis
            (Options: 'transit_depth', 'eclipse_depth', '(Rp/Rs)^2',
            '(Rp/R*)^2', 'Fp/Fs', 'Fp/F*', 'Fp', 'Fs', 'F*').
        colour_list (list, optional):
            A list of colours for the model spectra.
        spectra_labels (list, optional):
            A list of labels for the model spectra.
        data_colour_list (list, optional):
            A list of colours for the observational data points.
        data_labels (list, optional):
            A list of labels for the observational data.
        data_marker_list (list, optional):
            A list of marker styles for the observational data.
        data_marker_size_list (list, optional):
            A list of marker sizes for the observational data.
        data_alpha_list (list of float, optional):
            Alpha values for the central circle colours on each data point
            (defaults to 1.0 if not specified).
        data_eline_alpha_list (list of float, optional):
            Alpha values for the error bar colours on each data point
            (defaults to 0.8 if not specified).
        data_edge_width_list (list of float, optional):
            Border line width for the central circle on each data point
            (defaults to 0.8 if not specified).
        data_eline_colour_list (list of str, optional):
            Colours for data error bars (defaults to 'black' if not specified).
        data_eline_width_list (list of float, optional):
            Line widths for error bars (defaults to 1.0 if not specified).
        line_width_list (list of float, optional):
            Line widths for binned spectra (defaults to 2.0 if not specified).
        line_style_list (list of str, optional):
            Line styles for binned spectra (defaults to '-' if not specified).
        line_alpha_list (list of float, optional):
            Alpha values for binned spectra (defaults to 0.8 if not specified).
        text_annotations (list of str, optional):
            A list of text annotations for Figure decoration (e.g. molecule names)
        annotation_pos (list of tuples of str, optional):
            (x, y) locations of the text annotations in the previous argument.
        err_colour (str, optional):
            Colour of the data error bars if they are all the same (you can use 
            data_eline_colour_list to have different colours).
        wl_axis (str, optional):
            The type of x-axis to use ('log' or 'linear').
        figure_shape (str, optional):
            The shape of the figure ('default' or 'wide' - the latter is 16:9).
        show_legend (bool, optional):
            If False, will not plot legend.
        legend_location (str, optional):
            The location of the legend ('upper left', 'upper right',
            'lower left', 'lower right','outside right').
        legend_box (bool, optional):
            Flag indicating whether to plot a box surrounding the figure legend.
        legend_line_size (list of float, optional):
            Size of lines in the legend. Put 1 for data points
        legend_n_columns (integer):
            Manually set the number of columns for the legend.
        ax (matplotlib axis object, optional):
            Matplotlib axis provided externally.
        save_fig (bool, optional):
            If True, saves a PDF in the POSEIDON output folder.
        model (dict, optional):
            POSEIDON model dictionary. Required to be defined for offsets to be added.
        show_data_bin_width (bool, optional):
            Flag indicating whether to plot x bin widths for data points.
        show_data_cap (bool, optional):
            Flag indicating whether to plot the error bar caps on the data points.
        add_retrieved_offsets (bool, optional):
            Plots data with retrieved offset values.
        add_retrieved_error_inflation (bool, optional):
            Plots data error bars including retrieved error inflation value.
        verbose offsets (bool, optional):
            Will print out offsets applied to which datasets.
        x_labels (bool):
            If False, will remove x_ticks labels and x_label.
        y_labels (bool):
            If False, will remove y_ticks labels and y_label.
        x_tick_fontsize (int, optional):
            Font size for x-axis tick labels.
        x_label_fontsize (int, optional):
            Font size for x-axis label.
        y_tick_fontsize (int, optional):
            Font size for y-axis tick labels.
        y_label_fontsize (int, optional):
            Font size for y-axis label.
        legend_fontsize (int, optional):
            Font size for the legend.
        plt_label_fontsize (int, optional):
            Font size for the plot label.
        planet_name_fontsize (int, optional):
            Font size for the planet name.
        plot_style (str, optional):
            (Experimental!) plot style ('standard' or 'fancy').
        fill_between (list of bools, optional):
            If True, spectrum will have a fill color from its 
            line to 0 or fill_to_spectrum.
        fill_between_alpha (int, optional):
            Alpha of the fill region.
        fill_to_spectrum (list of ints, optional):
            If non-empty, will fill spectra to this spectrum (instead of 0).

    Returns:
        fig (matplotlib figure object):
            The spectra plot.
    
    '''
    
    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', '(Rp/R*)', 'transit_depth',
                   'transit_depth_ppm']):
        plot_type = 'transmission'
    elif (y_unit in ['Rp/Rs', 'Rp/R*', '(Rp/Rs)', '(Rp/R*)']):
        plot_type = 'planet_star_radius_ratio'
    elif (y_unit in ['time_average_transit_depth']):
        plot_type = 'time_average_transmission'
    elif (y_unit in ['Fp/Fs', 'Fp/F*', 'eclipse_depth', 'eclipse_depth_ppm']):
        plot_type = 'emission'
    elif (y_unit in ['Fp',  'Fs', 'F*']):
        plot_type = 'direct_emission'
    elif (y_unit in ['T_bright']):
        plot_type = 'brightness_temp'
    else:
        raise Exception("Unexpected y unit. Did you mean 'transit_depth' " +
                       "or 'eclipse_depth'?")
    
    # Find number of spectra to plot
    N_spectra = len(spectra)

    # Unpack model and atmospheric properties
    if (planet != None):
        planet_name = planet['planet_name']
    else:
        planet_name = ''

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Quick validity checks for plotting
    if (N_spectra == 0):
        raise Exception("Must provide at least one spectrum to plot!")
    if ((N_spectra > 8) and len(colour_list) == 0):
        raise Exception("Max number of concurrent spectra to plot is 8 with default colour list")
    if ((len(colour_list) != 0) and (N_spectra != len(colour_list))):
        raise Exception("Number of colours does not match number of spectra.")
    if ((len(spectra_labels) != 0) and (N_spectra != len(spectra_labels))):
        raise Exception("Number of model labels does not match number of spectra.")
    if ((len(text_annotations) != 0) and (len(text_annotations) != len(annotation_pos))):
        raise Exception("Number of annotation labels does not match provided positions.")
    if ((len(line_width_list) != 0) and (N_spectra != len(line_width_list))):
        raise Exception("Number of line widths does not match number of spectra.")
    if ((len(line_style_list) != 0) and (N_spectra != len(line_style_list))):
        raise Exception("Number of line styles does not match number of spectra.")
    if ((len(line_alpha_list) != 0) and (N_spectra != len(line_alpha_list))):
        raise Exception("Number of line alphas does not match number of spectra.")
    if ((fill_between != []) and (N_spectra != len(fill_between))):
        raise Exception("Bools in fill_between array must equal number of spectra.")
        
    # Define colours for plotted spectra (default or user choice)
    if (len(colour_list) == 0):   # If user did not specify a custom colour list
        colours = ['green', 'red', 'black', 'darkgrey', 'navy', 
                   'brown', 'goldenrod', 'magenta']
    else:
        colours = colour_list

    # Load default spectrum line width if not specified by the user
    if (len(line_width_list) == 0):
        if (plot_full_res == True):
            line_widths = np.full(N_spectra, 1.0)    # Default spectrum line width
        else:
            line_widths = np.full(N_spectra, 2.0)
    else:
        line_widths = line_width_list

    # Load default spectrum line style if not specified by the user
    if (len(line_style_list) == 0):
        line_styles = np.full(N_spectra, '-')    # Default spectrum line style
    else:
        line_styles = line_style_list

    # Load default spectrum line alpha if not specified by the user
    if (len(line_alpha_list) == 0):
        line_alphas = np.full(N_spectra, 0.8)    # Default spectrum line alpha
    else:
        line_alphas = line_alpha_list

    # Unpack data properties (if provided)
    if ((data_properties != None) and (show_data == True)):

        datasets = data_properties['datasets']
        instruments = data_properties['instruments']
        ydata = data_properties['ydata']
        err_data = data_properties['err_data']
        wl_data = data_properties['wl_data']
        bin_size = data_properties['half_bin']

        # Find number of datasets to plot
        N_datasets = len(datasets)
            
        # Quick validity checks for plotting
        if (N_datasets == 0):
            raise Exception("Must provide at least one dataset to plot!")
        if ((N_datasets > 5) and len(data_colour_list) == 0) or ((N_datasets > 5) and len(data_marker_list) == 0):
            raise Exception("Max number of concurrent datasets to plot is 5 with default data colours and markers.")
        if ((len(data_colour_list) != 0) and (N_datasets != len(data_colour_list))):
            raise Exception("Number of colours does not match number of datasets.")
        if ((len(data_labels) != 0) and (N_datasets != len(data_labels))):
            raise Exception("Number of dataset labels does not match number of datasets.")
        if ((len(data_marker_list) != 0) and (N_datasets != len(data_marker_list))):
            raise Exception("Number of dataset markers does not match number of datasets.")
        if ((len(data_marker_size_list) != 0) and (N_datasets != len(data_marker_size_list))):
            raise Exception("Number of dataset marker sizes does not match number of datasets.")
        if ((len(data_eline_colour_list) != 0) and (len(data_eline_colour_list) != N_datasets)):
            raise Exception("Number of error bar colours must match number of datasets.")

        # Define colours for plotted spectra (default or user choice)
        if (len(data_colour_list) == 0):   # If user did not specify a custom colour list
            data_colours = ['orange', 'lime', 'cyan', 'magenta', 'brown']
        else:
            data_colours = data_colour_list

        # Define data marker symbols (default or user choice)
        if (len(data_marker_list) == 0):   # If user did not specify a custom colour list
            if (N_datasets <= 5):
                data_markers = ['o', 's', 'D', '*', 'X']
            else:
                data_markers = np.full(N_datasets, 'o')
        else:
            data_markers = data_marker_list

        # Define data marker sizes (default or user choice)
        if (len(data_marker_size_list) == 0):
            data_markers_size = np.full(N_datasets, 3.0)   # Default data marker size
        else:
            data_markers_size = data_marker_size_list

        # Define data marker alpha (default or user choice)
        if (data_alpha_list == []):
            data_alphas = np.full(N_datasets, 1.0)   # Default data alpha
        else:
            data_alphas = data_alpha_list

        # Define data marker alpha (default or user choice)
        if (data_eline_alpha_list == []):
            data_eline_alphas = np.full(N_datasets, 0.8)   # Default error bar alpha
        else:
            data_eline_alphas = data_eline_alpha_list

        # Define data edge widths (default or user choice)
        if (data_edge_width_list == []):
            data_edge_widths = np.full(N_datasets, 0.8)   # Default data marker edge width
        else:
            data_edge_widths = data_edge_width_list

        # Define error bar line width (default or user choice)
        if (data_eline_width_list == []):
            data_eline_widths = np.full(N_datasets, 1.0)   # Default error line width
        else:
            data_eline_widths = data_eline_width_list

        #***** Apply any retrieved offsets to the data *****#

        if (add_retrieved_offsets == True):

            # Check model has been defined
            if (model == None):
                raise Exception('Please provide model to plot offsets')
            
            offset_datasets = model['offsets_applied']
            model_name = model['model_name']
            
            # Avoid overwriting the data points
            ydata_to_plot = np.array(ydata)

            # Add offsets for a single dataset 
            if (offset_datasets == 'single_dataset'):
                
                ### Unpack offset data properties (TBD: turn into function?) ###
                
                # offset_1_end == 0 is the default value for offset_1 array (meaning that the original offset_datasets was used)
                # The only difference is that the offset_1 setting can have multiple datasets with same offset

                if (data_properties['offset_1_end'] == 0):
                    offset_start, offset_end = data_properties['offset_start'], data_properties['offset_end']
                else:
                    offset_start, offset_end = data_properties['offset_1_start'], data_properties['offset_1_end']

                # Catch offsets for one dataset
                if isinstance(offset_start, np.int64):
                    offset_start, offset_end = np.array([offset_start]), np.array([offset_end])

                # Retrieve offset value from results file
                results_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'
                results_file_name = model_name + '_results.txt'

                # Open results file to find retrieved median offset value
                with open(results_dir + results_file_name, 'r') as f:
                    for line in f:
                        if ('delta_rel' in line):
                            delta_rel = float(line.split()[2])

                        # Stop reading file after 1 sigma constraints
                        if ('2 σ constraints' in line):
                            break

                for start, end in zip(offset_start, offset_end):
                    # Note: offsets are in ppm
                    ydata_to_plot[start:end] = ydata[start:end] - delta_rel*1e-6
                
                # If this is true, will append the offset applied to the dataset to the data labels 
                if (verbose_offsets == True):
                    if (data_properties['offset_1_end'] == 0):
                        print('Applied ' + str(delta_rel) + ' ppm offset to offset_datasets')
                    else:
                        print('Applied ' + str(delta_rel) + ' ppm offset to offset_1_datasets')
            
            # Add multiple offsets
            elif ((offset_datasets == 'two_datasets') or (offset_datasets == 'three_datasets')):
                #print('in two datasets')     

                # Unpack offset data properties
                if ((offset_datasets == 'two_datasets') and (data_properties['offset_1_start'] != 0)):
                    offset_start_list = ['offset_1_start', 'offset_2_start']
                    offset_end_list = ['offset_1_end', 'offset_2_end']
                elif ((offset_datasets == 'three_datasets') and (data_properties['offset_1_start'] != 0)):
                    offset_start_list = ['offset_1_start', 'offset_2_start', 'offset_3_start']
                    offset_end_list = ['offset_1_end', 'offset_2_end', 'offset_3_end']

                offset_start_end = []

                if (data_properties['offset_1_start'] != 0):
                    for start_name, end_name in zip(offset_start_list, offset_end_list):
                        offset_start, offset_end = data_properties[start_name], data_properties[end_name]

                        print(offset_start, offset_end)

                        # Catch zero offsets, not defined as arrays
                        if isinstance(offset_start, np.int64):
                            offset_start, offset_end = np.array([offset_start]), np.array([offset_end])
                        
                    #    if(len(offset_start) == 0):
                        offset_start_end.append((offset_start[0], offset_end[-1]))
                
                else:
                    for i in range(len(data_properties['offset_start'])):
                        offset_start, offset_end = data_properties['offset_start'][i], data_properties['offset_end'][i]

                        offset_start_end.append((offset_start, offset_end))

                # Retrieve offset value from results file
                results_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'
                results_file_name = model_name + '_results.txt'

                # Create empty array for relative offsets (max. number of offsets is currently 3)
                delta_rel_array = np.zeros(3)

                # Open results file to find retrieved median offset value
                with open(results_dir + results_file_name, 'r') as f:
                    for line in f:
                        if ('delta_rel_1' in line):
                            delta_rel_array[0] = line.split()[2]
                        if ('delta_rel_2' in line):
                            delta_rel_array[1] = line.split()[2]
                        if ('delta_rel_3' in line):
                            delta_rel_array[2] = line.split()[2]

                        # Stop reading file after 1 sigma constraints
                        if ('2 σ constraints' in line):
                            break

                # Add relative offset to ydata (note: offsets are subtracted)
                for delta_rel, (offset_start, offset_end) in zip(delta_rel_array, offset_start_end):
                    # Note: offsets are in ppm
                    ydata_to_plot[offset_start:offset_end] = ydata[offset_start:offset_end] - delta_rel*1e-6

                if (verbose_offsets == True):
                    print('Applied ' + str(delta_rel_array[0]) + ' ppm offset to offset_1_datasets')
                    print('Applied ' + str(delta_rel_array[1]) + ' ppm offset to offset_2_datasets')

                    if (offset_datasets == 'three_datasets'):
                        print('Applied ' + str(delta_rel_array[2]) + ' ppm offset to offset_3_datasets')
            
            # Continue plotting if no offsets are found
            elif offset_datasets == None:
                print('No offsets found, plotting data without offsets')
            
        else:
            ydata_to_plot = ydata
        
        #***** Apply retrieved error inflation parameter to data *****#

        if (add_retrieved_error_inflation == True):

            # Check model has been defined
            if (model == None):
                raise Exception('Please provide model to plot error inflated data')
            
            error_inflation = model['error_inflation']
            model_name = model['model_name']

            # Add offsets for a single dataset 
            if (error_inflation == None):
                error_inflation_params = []
            else:
                if (error_inflation == 'Line15'):
                    error_inflation_params = ['b']
                elif (error_inflation == 'Piette20'):
                    error_inflation_params = ['x_tol']
                elif ('Line15' in error_inflation) and ('Piette20' in error_inflation):
                    error_inflation_params = ['b', 'x_tol']
            
            # Retrieve offset value from results file
            results_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'
            results_file_name = model_name + '_results.txt'

            # Inflate error bars in the plot by the media retrieved error inflation parameter(s)
            if (error_inflation == None):
                err_data_to_plot = err_data
            else:
                err_inflation_param_values = []

                # Open results file to find retrieved median error inflation value
                with open(results_dir + results_file_name, 'r') as f:
                    for line in f:
                        for error_inflation_parameter in error_inflation_params:
                            if (((error_inflation_parameter in line)) and (len(error_inflation_parameter) == len(line.split()[0]))):
                                err_inflation_param_values += [float(line.split()[2])]  # Median error inflation parameter

                        # Stop reading file after 1 sigma constraints
                        if ('2 σ constraints' in line):
                            break

                # Apply error inflation to the data (Line+2015 prescription)
                if (error_inflation == 'Line15'):

                    # Calculate effective error bars including the median error inflation parameter
                    err_data_to_plot = np.sqrt(err_data**2 + np.power(10.0, err_inflation_param_values[0]))

                # Apply error inflation to the data (Piette+2020 prescription)
                elif (error_inflation == 'Piette20'):

                    # Extract median spectrum and wavelength grid
                    (spec_med, wl) = spectra[0]

                    # Bin the median spectrum to the data resolution
                    ymodel_median = bin_spectrum_to_data(spec_med, wl, data_properties)

                    # Calculate effective error bars including the median error inflation parameter
                    err_data_to_plot = np.sqrt(err_data**2 + (err_inflation_param_values[0] * ymodel_median)**2)

                # Apply both error inflation prescriptions to data (Line+2015 & Piette+2020)
                elif (('Line15' in error_inflation) and ('Piette20' in error_inflation)):

                    # Extract median spectrum and wavelength grid
                    (spec_med, wl) = spectra[0]

                    # Bin the median spectrum to the data resolution
                    ymodel_median = bin_spectrum_to_data(spec_med, wl, data_properties)

                    # Calculate effective error bars including the median error inflation parameter
                    err_data_to_plot = np.sqrt(err_data**2 + np.power(10.0, err_inflation_param_values[0]) +
                                            (err_inflation_param_values[1] * ymodel_median)**2)
        
        else:
            err_data_to_plot = err_data

    # If the user did not specify a wavelength range, find min and max from input models
    if (wl_min == None):
        
        wl_min = 1e10   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            wl_min_i = np.min(spectra[i][1])
            wl_min = min(wl_min, wl_min_i)
            
    # If the user did not specify a wavelength range, find min and max from input models
    if (wl_max == None):
        
        wl_max = 1e-10  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            wl_max_i = np.max(spectra[i][1])
            wl_max = max(wl_max, wl_max_i)

    wl_range = wl_max - wl_min
   
    # If the user did not specify a y range, find min and max from input models
    if (y_min == None):
        
        y_min_plt = 1e10   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            y_min_i = np.min(spectra[i][0])
            y_min_plt = min(y_min_plt, y_min_i)
            
        # Check if the lowest data point falls below the current y-limit
        if (show_data == True):
            if (y_min_plt > min(ydata - err_data)):
                
                y_min_plt = min(ydata - err_data)
            
        y_min_plt = 0.995*y_min_plt  # Extend slightly below
        
    else:
        y_min_plt = y_min

    if (y_max == None):
        
        y_max_plt = 1e-10  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            y_max_i = np.max(spectra[i][0])
            y_max_plt = max(y_max_plt, y_max_i)
            
        # Check if the highest data point falls above the current y-limit
        if (show_data == True):
            if (y_max_plt < max(ydata + err_data)):
                
                y_max_plt = max(ydata + err_data)
            
        y_max_plt = 1.005*y_max_plt  # Extend slightly above
        
    else:
        y_max_plt = y_max
        
    #***** Format x and y ticks *****#

    # Create x formatting objects
    if (wl_range >= 0.2):                      
        if (wl_max < 1.0):                         # Plotting the optical range
            xmajorLocator = MultipleLocator(0.1)
            xminorLocator = MultipleLocator(0.02)
        else:                                      # Plot extends into the infrared
            xmajorLocator = MultipleLocator(1.0)
            xminorLocator = MultipleLocator(0.1)
    elif ((wl_range < 0.2) and (wl_range >= 0.02)):   # High-resolution zoomed plots
        xmajorLocator = MultipleLocator(0.01)
        xminorLocator = MultipleLocator(0.002)
    else:                                             # Super high-resolution
        xmajorLocator = MultipleLocator(0.001)
        xminorLocator = MultipleLocator(0.0002)

    xmajorFormatter = FormatStrFormatter('%g')
    xminorFormatter = NullFormatter()
    
    # Aim for 10 major y-axis labels
    ymajor_spacing = round_sig_figs((y_max_plt - y_min_plt), 1)/10
    yminor_spacing = ymajor_spacing/10

    if (np.abs(np.log10(ymajor_spacing)) <= 10.0):    
        major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 1)
        minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 1)
    else:    # Bug fix for surface spectra where Fp > 1e10 
        major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 2)
        minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 2)

    # If last digit of y labels would be a multiple of 6,7,8,or 9, bump up to 10
    if (ymajor_spacing > 5*np.power(10, major_exponent)):
        ymajor_spacing = 1*np.power(10, major_exponent+1)
    elif (ymajor_spacing == 3*np.power(10, major_exponent)):
        ymajor_spacing = 2*np.power(10, major_exponent)
    if (yminor_spacing > 5*np.power(10, minor_exponent)):
        yminor_spacing = 1*np.power(10, minor_exponent+1)
    elif (yminor_spacing == 3*np.power(10, minor_exponent)):
        yminor_spacing = 2*np.power(10, minor_exponent)

    # Refine y range to be a multiple of the tick spacing (only if range not specified by user)
    if (y_min == None):
        y_min_plt = np.floor(y_min_plt/ymajor_spacing)*ymajor_spacing
    if (y_max == None):
        y_max_plt = np.ceil(y_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    y_range = [y_min_plt, y_max_plt]

    # Create y formatting objects
    ymajorLocator   = MultipleLocator(ymajor_spacing)

    if ((plot_type == 'planet_star_radius_ratio') or (y_min_plt > 0.10)):
        ymajorFormatter = ScalarFormatter(useMathText=False)
    else:
        ymajorFormatter = ScalarFormatter(useMathText=True)
        ymajorFormatter.set_powerlimits((0,0))

    yminorLocator = MultipleLocator(yminor_spacing)

    # Generate figure and axes
    fig = plt.figure()

    # Set figure size
    if (figure_shape == 'default'):
        fig.set_size_inches(8.0, 6.0)    # Default Matplotlib figure size
    elif (figure_shape == 'wide'):
        fig.set_size_inches(10.667, 6.0)    # 16:9 widescreen format (for two column figures)
    elif (legend_location == 'outside right'):
        fig.set_size_inches(12, 8.0) 
    else:
        raise Exception("Unsupported Figure shape - please use 'default' or 'wide'")
    
    if (ax == None):
        ax1 = plt.gca()
    else:
        ax1 = ax
    
    # Set x axis to be linear or logarithmic
    ax1.set_xscale(wl_axis)

    # Assign formatter objects to axes
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    ax1.yaxis.set_minor_locator(yminorLocator)
    
    for i in range(N_spectra):
        
        # Extract spectrum and wavelength grid
        (spec, wl) = spectra[i]
        
        # If user did not specify a model label, just call them "Model 1, 2" etc.
        if (len(spectra_labels) == 0):
            if (N_spectra == 1):
                label_i = r'Spectrum'
            else:
                label_i = r'Spectrum ' + str(i+1)
        else:
            label_i = spectra_labels[i]
        
        # Plot spectrum at full model resolution
        if (plot_full_res == True):
            ax1.plot(wl, spec, lw = 0.5, alpha = 0.4, zorder = i,
                     color = colours[i], label = label_i)

        # Plot smoothed (binned) version of the model
        if (bin_spectra == True):
            
            N_plotted_binned = 0  # Counter for number of plotted binned spectra

            if len(fill_to_spectrum) != 0:
                wl_binned, fill_to_spectrum_binned, _ = bin_spectrum(wl, fill_to_spectrum, R_to_bin)
            
            # Calculate binned wavelength and spectrum grid
            wl_binned, spec_binned, _ = bin_spectrum(wl, spec, R_to_bin)

            if (plot_full_res == True):
                colour_binned = scale_lightness(colours[i], 0.4)
                label_i += ' (R = ' + str(R_to_bin) + ')'
            else:
                colour_binned = colours[i]

            # Plot binned spectrum
            ax1.plot(wl_binned, spec_binned, lw = line_widths[i], 
                     alpha = line_alphas[i], 
                     color = colour_binned, 
                     zorder = N_spectra+N_plotted_binned, 
                     label = label_i,
                     linestyle = line_styles[i])
            
            if len(fill_between) != 0:
                if fill_between[i] == True:
                    if len(fill_to_spectrum) == 0:
                        ax1.fill_between(wl_binned, spec_binned, y2 = 0, 
                                         alpha=fill_between_alpha,
                                         color = colour_binned)
                    else:
                        ax1.fill_between(wl_binned, spec_binned, y2 = fill_to_spectrum,
                                         alpha=fill_between_alpha,
                                         color = colour_binned)  
            
            N_plotted_binned += 1

    # Overplot datapoints
    if (show_data == True):

        for i in range(N_datasets):
            
            # If user did not specify dataset labels, use the instrument names
            if (len(data_labels) == 0):
                label_i = instruments[i]
            else:
                label_i = data_labels[i]
            
            # Find start and end indices of dataset_i in dataset property arrays
            idx_start = data_properties['len_data_idx'][i]
            idx_end = data_properties['len_data_idx'][i+1]

            # Extract the ith dataset
            wl_data_i = wl_data[idx_start:idx_end]
            ydata_i = ydata_to_plot[idx_start:idx_end]
            err_data_i = err_data_to_plot[idx_start:idx_end]
            bin_size_i = bin_size[idx_start:idx_end]

            if (show_data_cap == True):
                capsize = 2
            else:
                capsize = 0

            # Plot dataset
            if (show_data_bin_width == True):
                x_bin_size = bin_size_i
            else:
                x_bin_size = None

            if len(data_eline_colour_list) == 0:
                markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr = err_data_i, 
                                                   xerr = x_bin_size, marker = data_markers[i], 
                                                   markersize = data_markers_size[i], 
                                                   capsize = capsize, ls='none',
                                                   elinewidth = data_eline_widths[i], 
                                                   color = data_colours[i], 
                                                   alpha = data_eline_alphas[i],
                                                   ecolor = err_colour, label=label_i,
                                                   markeredgewidth = data_edge_widths[i],
                                                   zorder = 100)
            else:
                markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr = err_data_i, 
                                                   xerr = x_bin_size, marker = data_markers[i], 
                                                   markersize = data_markers_size[i], 
                                                   capsize = capsize, ls='none', 
                                                   elinewidth = data_eline_widths[i], 
                                                   color = data_colours[i], 
                                                   alpha = data_eline_alphas[i],
                                                   ecolor = data_eline_colour_list[i], label=label_i,
                                                   markeredgewidth = data_edge_widths[i],
                                                   zorder = 100)

            [markers.set_alpha(data_alphas[i])]

    # Plot text annotations
    if (len(text_annotations) != 0):

        for i in range(len(text_annotations)):

            # Plot each annotation at the location provided by the user
            ax1.text(annotation_pos[i][0], annotation_pos[i][1], 
                     text_annotations[i], fontsize=14, color = 'black')

    # Set axis ranges
    ax1.set_xlim([wl_min, wl_max])
    ax1.set_ylim([y_range[0], y_range[1]])
        
    # Set axis labels
    if (xlabels == True):
        ax1.set_xlabel(r'Wavelength (μm)', fontsize = x_label_fontsize)

    if (ylabels == True):
        if (plot_type == 'transmission'):
            if (y_unit == 'transit_depth_ppm'):
                ax1.set_ylabel(r'Transit Depth (ppm)', fontsize = y_label_fontsize)
            else:
                if (y_min_plt < 0.10):
                    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = y_label_fontsize)
                else:
                    ax1.set_ylabel(r'Transit Depth', fontsize = y_label_fontsize)
        elif (plot_type == 'planet_star_radius_ratio'):
            ax1.set_ylabel(r'$R_p/R_*$', fontsize = y_label_fontsize)
        elif (plot_type == 'time_average_transmission'):
            ax1.set_ylabel(r'Average Transit Depth', fontsize =  y_label_fontsize)
        elif (plot_type == 'emission'):
            if (y_unit == 'eclipse_depth_ppm'):
                ax1.set_ylabel(r'Eclipse Depth $(ppm)$', fontsize = y_label_fontsize)
            else:
                ax1.set_ylabel(r'Emission Spectrum $(F_p/F_*)$', fontsize = y_label_fontsize)
        elif (plot_type == 'direct_emission'):
            if (y_unit == 'Fp'):
                ax1.set_ylabel(r'$F_{\rm{p}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = y_label_fontsize)
            elif (y_unit in ['Fs', 'F*']):
                ax1.set_ylabel(r'$F_{\rm{s}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = y_label_fontsize)
        elif (plot_type == 'brightness_temp'):
            ax1.set_ylabel(r'Brightness Temperature (K)', fontsize =  y_label_fontsize)

    # Add planet name label
    if (show_planet_name == True):
        ax1.text(0.02, 0.96, planet_name, horizontalalignment = 'left', 
                 verticalalignment = 'top', transform = ax1.transAxes, fontsize = planet_name_fontsize)

    # Add plot label
    if (plt_label != None):
        if (show_planet_name == True):
            ax1.text(0.03, 0.90, plt_label, horizontalalignment = 'left', 
                     verticalalignment = 'top', transform = ax1.transAxes, fontsize = plt_label_fontsize)
        else:
            ax1.text(0.03, 0.96, plt_label, horizontalalignment = 'left', 
                     verticalalignment = 'top', transform = ax1.transAxes, fontsize = plt_label_fontsize)

    # Decide at which wavelengths to place major tick labels
    wl_ticks = set_spectrum_wl_ticks(wl_min, wl_max, wl_axis)
        
    # Plot wl tick labels
    if (xlabels == True):
        ax1.set_xticks(wl_ticks)
    else:
        ax1.set_xticks(wl_ticks)
        ax1.tick_params(labelbottom=False)  
    
    # If ylabels is False, don't show them
    if (ylabels == False):
        ax1.tick_params(labelleft=False)  

    # Set the x and y tick font sizes
    ax1.tick_params(axis='x', labelsize=x_tick_fontsize)
    ax1.tick_params(axis='y', labelsize=y_tick_fontsize)
    
    # Switch to two columns if many spectra are being plotted
    if (legend_n_columns == 0):
        if (N_spectra >= 6):
            n_columns = 2
        else:
            n_columns = 1
    else:
        n_columns = legend_n_columns

    # Add box around legend
    if show_legend == True:
        if (legend_box == True):
            frameon = True
            framefacecolour = '0.9'
        else:
            frameon = False
            framefacecolour = None

        # Add legend
        if isinstance(legend_location, tuple):
            legend = ax1.legend(loc = 'center', shadow = True, prop = {'size': legend_fontsize},
                                ncol = n_columns, frameon = frameon, bbox_to_anchor = legend_location)
        elif legend_location == 'outside right':
            legend = ax1.legend(loc='center left', shadow = True, prop = {'size':legend_fontsize}, 
                                ncol = n_columns, frameon = frameon, bbox_to_anchor = (1, 0.5))
        else:
            legend = ax1.legend(loc = legend_location, shadow = True, prop={'size': legend_fontsize},
                                ncol = n_columns, frameon = frameon)  # Legend settings

        frame = legend.get_frame()
        frame.set_facecolor(framefacecolour)

        legend.set_zorder(200)   # Make legend always appear in front of everything

        # Set legend line width
        if len(legend_line_size) == 0:
            try:
                for legline in legend.legend_handles:
                    if ((plot_full_res == True) or (show_data == True)):
                        legline.set_linewidth(1.0)
                    else:
                        legline.set_linewidth(2.0)
            except AttributeError:
                for legline in legend.legendHandles:
                    if ((plot_full_res == True) or (show_data == True)):
                        legline.set_linewidth(1.0)
                    else:
                        legline.set_linewidth(2.0)
        
        # Let user define line width in legend 
        else:
            # Check legend line size length
            try:
                if (len(legend_line_size) != len(legend.legend_handles)):
                    raise Exception("Make sure legend_line_size length is equal to number of handles.")
            except:
                # weird attribute error
                if (len(legend_line_size) != len(legend.legendHandles)):
                    raise Exception("Make sure legend_line_size length is equal to number of handles.")
            try:
                for i in range(len(legend.legend_handles)):
                    legline = legend.legend_handles[i]
                    legline.set_linewidth(legend_line_size[i])
            except AttributeError:
                for i in range(len(legend.legendHandles)):
                    legline = legend.legendHandles[i]
                    legline.set_linewidth(legend_line_size[i])
    
    plt.tight_layout()

    # Write figure to file
    if (save_fig == True):
        if (plt_label == None):
            file_name = (output_dir + planet_name + '_' + plot_type + '_spectra.pdf')
        else:
            file_name = (output_dir + planet_name + '_' + plt_label + '_' +
                        plot_type + '_spectra.pdf')

        plt.savefig(file_name, bbox_inches = 'tight')

    return fig


def plot_data(data, planet_name, wl_min = None, wl_max = None, 
              y_min = None, y_max = None, y_unit = 'transit_depth',
              plt_label = None, data_colour_list = [], data_labels = [], 
              data_marker_list = [], data_marker_size_list = [],
              err_colour = 'black', wl_axis = 'log', figure_shape = 'default', 
              legend_location = 'upper right', legend_box = True,
              show_data_bin_width = True, show_data_cap = True,
              data_alpha = 0.8, data_edge_width = 0.8,
              ax = None, save_fig = True,
              ):
    ''' 
    Plot a collection of datasets. This function can plot transmission or 
    emission datasets, according to the user's choice of 'y_unit'.
    
    Args:
        data (dict):
            POSEIDON observational data properties dictionary.
        planet_name (str):
            Name of the planet.
        wl_min (float, optional):
            The minimum wavelength to plot.
        wl_max (float, optional):
            The maximum wavelength to plot.
        y_min (float, optional):
            The minimum value for the y-axis.
        y_max (float, optional):
            The maximum value for the y-axis.
        y_unit (str, optional):
            The unit of the y-axis
            (Options: 'transit_depth', 'eclipse_depth', '(Rp/Rs)^2', 
            '(Rp/R*)^2', 'Fp/Fs', 'Fp/F*', 'Fp').
        plt_label (str, optional):
            The label for the plot.
        data_colour_list (list, optional):
            A list of colours for the observational data points.
        data_labels (list, optional):
            A list of labels for the observational data.
        data_marker_list (list, optional):
            A list of marker styles for the observational data.
        data_marker_size_list (list, optional):
            A list of marker sizes for the observational data.
        err_colour (string, optional):
            Colour of the data error bars (white works best for a dark background)
        wl_axis (str, optional):
            The type of x-axis to use ('log' or 'linear').
        figure_shape (str, optional):
            The shape of the figure ('default' or 'wide' - the latter is 16:9).
        legend_location (str, optional):
            The location of the legend ('upper left', 'upper right', 
            'lower left', 'lower right', 'outside right').
        legend_box (bool, optional):
            Flag indicating whether to plot a box surrounding the figure legend.
        show_data_bin_width (bool, optional):
            Flag indicating whether to plot x bin widths for data points.
        show_data_cap (bool, optional):
            Flag indicating whether to show the caps on the data error bars.
        data_alpha (float, optional):
            Alpha for the central circle colours on each data point.
        data_edge_width (float, optional):
            Border line width for the central circle on each data point.
        ax (matplotlib axis object, optional):
            Matplotlib axis provided externally.
        save_fig (bool, optional):
            If True, saves a PDF in the POSEIDON output folder.

    Returns:
        fig (matplotlib figure object):
            The data plot.
    
    '''

    base_dir = './'

    # Create output directories (if not already present)
    create_directories(base_dir, planet_name)

    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', '(Rp/R*)', 'transit_depth',
                   'transit_depth_ppm']):
        plot_type = 'transmission'
    elif (y_unit in ['Rp/Rs', 'Rp/R*', '(Rp/Rs)', '(Rp/R*)']):
        plot_type = 'planet_star_radius_ratio'
    elif (y_unit in ['Fp/Fs', 'Fp/F*', 'eclipse_depth', 'eclipse_depth_ppm']):
        plot_type = 'emission'
    elif (y_unit in ['Fp']):
        plot_type = 'direct_emission'
    else:
        raise Exception("Unexpected y unit. Did you mean 'transit_depth' " +
                       "or 'eclipse_depth'?")

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/' + 'plots/'

    # Unpack data properties
    datasets = data['datasets']
    instruments = data['instruments']
    ydata = data['ydata']
    err_data = data['err_data']
    wl_data = data['wl_data']
    bin_size = data['half_bin']
    
    # Find number of datasets to plot
    N_datasets = len(datasets)
         
    # Quick validity checks for plotting
    if (N_datasets == 0):
        raise Exception("Must provide at least one dataset to plot!")
    if ((N_datasets > 7) and len(data_colour_list) == 0) or ((N_datasets > 7) and len(data_marker_list) == 0):
        raise Exception("Max number of concurrent datasets to plot is 7 with default data colours and markers.")
    if ((len(data_colour_list) != 0) and (N_datasets != len(data_colour_list))):
        raise Exception("Number of colours does not match number of datasets.")
    if ((len(data_labels) != 0) and (N_datasets != len(data_labels))):
        raise Exception("Number of dataset labels does not match number of datasets.")
    if ((len(data_marker_list) != 0) and (N_datasets != len(data_marker_list))):
        raise Exception("Number of dataset markers does not match number of datasets.")
    if ((len(data_marker_size_list) != 0) and (N_datasets != len(data_marker_size_list))):
        raise Exception("Number of dataset marker sizes does not match number of datasets.")
        
    # Define colours for plotted spectra (default or user choice)
    if (len(data_colour_list) == 0):   # If user did not specify a custom colour list
        colours = ['orange', 'lime', 'cyan', 'magenta', 'brown', 'grey', 'purple']
    else:
        colours = data_colour_list

    # Define data marker symbols (default or user choice)
    if (len(data_marker_list) == 0):   # If user did not specify a custom colour list
        data_markers = ['o', 's', 'D', '*', 'X', 'o', 'o']
    else:
        data_markers = data_marker_list

    # Define data marker sizes (default or user choice)
    if (len(data_marker_size_list) == 0):   # If user did not specify a custom colour list
        data_markers_size = [3, 3, 3, 3, 3, 3, 3]
    else:
        data_markers_size = data_marker_size_list
       
    # If the user did not specify a wavelength range, find min and max from input data
    if (wl_min == None):
        wl_min = np.min(wl_data - 4*bin_size)  # Minimum at twice the bin width for the shortest wavelength data
    else:
        wl_min = wl_min
 
    if (wl_max == None):
        wl_max = np.max(wl_data + 4*bin_size)  # Maximum at twice the bin width for the longest wavelength data
    else:
        wl_max = wl_max

    # If the user did not specify a y range, find min and max from data
    if (y_min == None):
        y_min_plt = 0.995 * np.min(ydata - err_data) # Extend slightly below
    else:
        y_min_plt = y_min

    if (y_max == None):
        y_max_plt = 1.005 * np.max(ydata + err_data) # Extend slightly above
    else:
        y_max_plt = y_max
        
    #***** Format x and y ticks *****#

    # Create x formatting objects
    if (wl_max < 1.0):    # If plotting over the optical range
        xmajorLocator = MultipleLocator(0.1)
        xminorLocator = MultipleLocator(0.02)
        
    else:                 # If plot extends into the infrared
        xmajorLocator = MultipleLocator(1.0)
        xminorLocator = MultipleLocator(0.1)
            
    xmajorFormatter = FormatStrFormatter('%g')
    xminorFormatter = NullFormatter()
    
    # Aim for 10 major y-axis labels
    ymajor_spacing = round_sig_figs((y_max_plt - y_min_plt), 1)/10
    yminor_spacing = ymajor_spacing/10
    
    if (np.abs(np.log10(ymajor_spacing)) <= 10.0):    
        major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 1)
        minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 1)
    else:    # Bug fix for surface spectra where Fp > 1e10 
        major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 2)
        minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 2)
    
    # If last digit of y labels would be a multiple of 6,7,8,or 9, bump up to 10
    if (ymajor_spacing > 5*np.power(10, major_exponent)):
        ymajor_spacing = 1*np.power(10, major_exponent+1)
    elif (ymajor_spacing == 3*np.power(10, major_exponent)):
        ymajor_spacing = 2*np.power(10, major_exponent)
    if (yminor_spacing > 5*np.power(10, minor_exponent)):
        yminor_spacing = 1*np.power(10, minor_exponent+1)
    elif (yminor_spacing == 3*np.power(10, minor_exponent)):
        yminor_spacing = 2*np.power(10, minor_exponent)
    
    # Refine y range to be a multiple of the tick spacing (only if range not specified by user)
    if (y_min == None):
        y_min_plt = np.floor(y_min_plt/ymajor_spacing)*ymajor_spacing
    if (y_max == None):
        y_max_plt = np.ceil(y_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    y_range = [y_min_plt, y_max_plt]

    # Create y formatting objects
    ymajorLocator = MultipleLocator(ymajor_spacing)

    if ((plot_type == 'planet_star_radius_ratio') or (y_min_plt > 0.10)):
        ymajorFormatter = ScalarFormatter(useMathText=False)
    else:
        ymajorFormatter = ScalarFormatter(useMathText=True)
        ymajorFormatter.set_powerlimits((0,0))

    yminorLocator = MultipleLocator(yminor_spacing)

    # Generate figure and axes
    fig = plt.figure()

    # Set figure size
    if (figure_shape == 'default'):
        fig.set_size_inches(8.0, 6.0)    # Default Matplotlib figure size
    elif (figure_shape == 'wide'):
        fig.set_size_inches(10.667, 6.0)    # 16:9 widescreen format (for two column figures) 
    elif (legend_location == 'outside right'):
        fig.set_size_inches(12, 8.0) 

    
    if (ax == None):
        ax1 = plt.gca()
    else:
        ax1 = ax
    
    # Set x axis to be linear or logarithmic
    ax1.set_xscale(wl_axis)

    # Assign formatter objects to axes
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    ax1.yaxis.set_minor_locator(yminorLocator)
    
    for i in range(N_datasets):
        
        # If user did not specify dataset labels, use the instrument names
        if (len(data_labels) == 0):
            label_i = instruments[i]
        else:
            label_i = data_labels[i]
        
        # Find start and end indices of dataset_i in dataset property arrays
        idx_start = data['len_data_idx'][i]
        idx_end = data['len_data_idx'][i+1]

        # Extract the ith dataset
        wl_data_i = wl_data[idx_start:idx_end]
        ydata_i = ydata[idx_start:idx_end]
        err_data_i = err_data[idx_start:idx_end]
        bin_size_i = bin_size[idx_start:idx_end]

        if (show_data_cap == True):
            capsize = 2
        else:
            capsize = 0

        # Plot dataset
        if (show_data_bin_width == True):
            markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr=err_data_i, 
                                               xerr=bin_size_i, marker=data_markers[i], 
                                               markersize=data_markers_size[i], 
                                               capsize=capsize, ls='none', elinewidth=0.8, 
                                               color=colours[i], alpha = data_alpha,
                                               ecolor = err_colour, label=label_i,
                                               markeredgewidth = data_edge_width,)

        else:
            markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr=err_data_i, 
                                               marker=data_markers[i], 
                                               markersize=data_markers_size[i], 
                                               capsize=capsize, ls='none', elinewidth=0.8, 
                                               color=colours[i], alpha = data_alpha,
                                               ecolor = err_colour, label=label_i,
                                               markeredgewidth = data_edge_width)

        [markers.set_alpha(1.0)]
            
    # Set axis ranges
    ax1.set_xlim([wl_min, wl_max])
    ax1.set_ylim([y_range[0], y_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)

    if (plot_type == 'transmission'):
        if (y_unit == 'transit_depth_ppm'):
            ax1.set_ylabel(r'Transit Depth (ppm)', fontsize = 16)
        else:
            if (y_min_plt < 0.10):
                ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)
            else:
                ax1.set_ylabel(r'Transit Depth', fontsize = 16)
    elif (plot_type == 'planet_star_radius_ratio'):
        ax1.set_ylabel(r'$R_p/R_*$', fontsize = 16)
    elif (plot_type == 'emission'):
        if (y_unit == 'eclipse_depth_ppm'):
            ax1.set_ylabel(r'Eclipse Depth $(ppm)$', fontsize = 16)
        else:
            ax1.set_ylabel(r'Emission Spectrum $(F_p/F_*)$', fontsize = 16)
    elif (plot_type == 'direct_emission'):
        if (y_unit == 'Fp'):
            ax1.set_ylabel(r'$F_{\rm{p}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = 16)
        elif (y_unit in ['Fs', 'F*']):
            ax1.set_ylabel(r'$F_{\rm{s}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

    # Add planet name label
    ax1.text(0.02, 0.96, planet_name, horizontalalignment='left', 
             verticalalignment='top', transform=ax1.transAxes, fontsize = 16)
  
    # Add plot label
    if (plt_label != None):
        ax1.text(0.03, 0.90, plt_label, horizontalalignment='left', 
                 verticalalignment='top', transform=ax1.transAxes, fontsize = 14)

    # Decide at which wavelengths to place major tick labels
    wl_ticks = set_spectrum_wl_ticks(wl_min, wl_max, wl_axis)
        
    # Plot wl tick labels
    ax1.set_xticks(wl_ticks)

    # Add box around legend
    if (legend_box == True) and (legend_location != 'outside right'):
        legend = ax1.legend(loc = legend_location, shadow = True, prop = {'size':10}, 
                            ncol = 1, frameon = True)    # Legend settings
        frame = legend.get_frame()
        frame.set_facecolor('0.90') 
    elif legend_location == 'outside right':
        legend = ax1.legend(loc='center left', shadow = True, prop = {'size':10}, 
                            ncol = 1, frameon=False,bbox_to_anchor=(1, 0.5))  
    else:
        legend = ax1.legend(loc=legend_location, shadow = True, prop = {'size':10}, 
                            ncol = 1, frameon = False)    # Legend settings
        
    plt.tight_layout()
    
    try:
        for legline in legend.legend_handles:
            legline.set_linewidth(1.0)
    except AttributeError:
        for legline in legend.legend_handles:
            legline.set_linewidth(1.0)

    # Write figure to file
    if (save_fig == True):
        if (plt_label == None):
            file_name = (output_dir + planet_name +
                        '_data.pdf')
        else:
            file_name = (output_dir + planet_name + '_' + plt_label + 
                        '_data.pdf')

        plt.savefig(file_name, bbox_inches = 'tight')

    return fig


def plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1, 
                           spectra_high1, spectra_high2, planet_name,
                           data_properties, R_to_bin = 100, plt_label = None,
                           show_ymodel = True, show_planet_name = True,
                           wl_min = None, wl_max = None, y_min = None, y_max = None, 
                           y_unit = 'transit_depth', colour_list = [], 
                           spectra_labels = [], data_colour_list = [], 
                           data_labels = [], data_marker_list = [], 
                           data_marker_size_list = [], data_alpha_list = [], 
                           data_eline_alpha_list = [], data_edge_width_list = [],
                           data_eline_colour_list = [], data_eline_width_list = [],
                           line_width_list = [], line_style_list = [], line_alpha_list = [],
                           binned_colour_list = [], 
                           text_annotations = [], annotation_pos = [], 
                           err_colour = 'black', wl_axis = 'log', 
                           figure_shape = 'default',
                           show_legend = True, legend_location = 'upper right', 
                           legend_box = False, legend_line_size = [], legend_n_columns = 0,
                           ax = None, save_fig = True, model = None,
                           show_data_bin_width = True, show_data_cap = True,
                           sigma_to_plot = 2, 
                           add_retrieved_offsets = False, verbose_offsets = True,
                           add_retrieved_error_inflation = False,
                           xlabels = True, ylabels = True,  
                           x_tick_fontsize = 12, x_label_fontsize = 16, 
                           y_tick_fontsize = 12, y_label_fontsize = 16,
                           legend_fontsize = 10, plt_label_fontsize = 14,
                           planet_name_fontsize = 16, plot_style = 'standard',
                           ):
    ''' 
    Plot a collection of individual model spectra. This function can plot
    transmission or emission spectra, according to the user's choice of 'y_unit'.
    
    Args:
        spectra_median (list of tuples): 
            A list of median spectra to be plotted, each with the format 
            (wl, spec_median).
        spectra_low2 (list of tuples): 
            Corresponding list of -2σ confidence intervals on the retrieved 
            spectra, each with the format (wl, spec_low2).
        spectra_low1 (list of tuples): 
            Corresponding list of -1σ confidence intervals on the retrieved 
            spectra, each with the format (wl, spec_low1).
        spectra_high1 (list of tuples): 
            Corresponding list of +1σ confidence intervals on the retrieved 
            spectra, each with the format (wl, spec_high1).
        spectra_high2 (list of tuples): 
            Corresponding list of +2σ confidence intervals on the retrieved 
            spectra, each with the format (wl, spec_high2).
        planet_name (str):
            Planet name to overplot on figure.
        data_properties (dict, optional):
            POSEIDON observational data properties dictionary.
        R_to_bin (int, optional):
            Spectral resolution (R = wl/dwl) to bin the model spectra to.
        plt_label (str, optional):
            The label for the plot.
        show_ymodel (bool, optional):
            Flag indicating whether to plot the median retrieved spectra binned 
            to the data resolution.
        show_planet_name (bool, optional):
            Flag indicating whether to include the planet name in the top left.
        wl_min (float, optional):
            The minimum wavelength to plot.
        wl_max (float, optional):
            The maximum wavelength to plot.
        y_min (float, optional):
            The minimum value for the y-axis.
        y_max (float, optional):
            The maximum value for the y-axis.
        y_unit (str, optional):
            The unit of the y-axis
            (Options: 'transit_depth', 'eclipse_depth', '(Rp/Rs)^2', 
            '(Rp/R*)^2', 'Fp/Fs', 'Fp/F*', 'Fp').
        colour_list (list, optional):
            A list of colours for the model spectra.
        spectra_labels (list, optional):
            A list of labels for the model spectra.
        data_colour_list (list, optional):
            A list of colours for the observational data points.
        data_labels (list, optional):
            A list of labels for the observational data.
        data_marker_list (list, optional):
            A list of marker styles for the observational data.
        data_marker_size_list (list, optional):
            A list of marker sizes for the observational data.
        data_alpha_list (list of float, optional):
            Alpha values for the central circle colours on each data point
            (defaults to 1.0 if not specified).
        data_eline_alpha_list (list of float, optional):
            Alpha values for the error bar colours on each data point
            (defaults to 0.8 if not specified).
        data_edge_width_list (list of float, optional):
            Border line width for the central circle on each data point
            (defaults to 0.8 if not specified).
        data_eline_colour_list (list of str, optional):
            Colours for data error bars (defaults to 'black' if not specified).
        data_eline_width_list (list of float, optional):
            Line widths for error bars (defaults to 1.0 if not specified).
        line_width_list (list of float, optional):
            Line widths for median spectra (defaults to 1.0 if not specified).
        line_style_list (list of str, optional):
            Line styles for median spectra (defaults to '-' if not specified).
        line_alpha_list (list of float, optional):
            Alpha values for median spectra (defaults to 0.8 if not specified).     
        binned_colour_list (list, optional):
            A list of colours for the binned models.
        text_annotations (list of str, optional):
            A list of text annotations for Figure decoration (e.g. molecule names)
        annotation_pos (list of tuples of str, optional):
            (x, y) locations of the text annotations in the previous argument.
        err_colour (string, optional):
            Colour of the data error bars (white works best for a dark background)
        wl_axis (str, optional):
            The type of x-axis to use ('log' or 'linear').
        figure_shape (str, optional):
            The shape of the figure ('default' or 'wide' - the latter is 16:9).
        show_legend (bool, optional):
            If False, will not plot legend.
        legend_location (str, optional):
            The location of the legend ('upper left', 'upper right', 
            'lower left', 'lower right', 'outside right').
        legend_box (bool, optional):
            Flag indicating whether to plot a box surrounding the figure legend.
        legend_line_size (list of float, optional):
            Size of lines in the legend. Put 1 for data points
        legend_n_columns (integer):
            Manually set the number of columns for the legend.
        ax (matplotlib axis object, optional):
            Matplotlib axis provided externally.
        save_fig (bool, optional):
            If True, saves a PDF in the POSEIDON output folder.
        model (dict, optional):
            POSEIDON model dictionary. Required to be defined for offsets to be added.
        show_data_bin_width (bool, optional):
            Flag indicating whether to plot x bin widths for data points.
        show_data_cap (bool, optional):
            Flag indicating whether to plot the error bar caps on the data points.
        sigma_to_plot (int, optional):
            How many sigma contours to plot (0 for only median, 1 for median and 
            1 sigma, or 2 for median, 1 sigma, and 2 sigma).
        add_retrieved_offsets (bool, optional):
            Plots data with retrieved offset values.
        add_retrieved_error_inflation (bool, optional):
            Plots data error bars including retrieved error inflation value.
        verbose offsets (bool, optional):
            Will print out offsets applied to which datasets.
        x_labels (bool, optional):
            If false, will remove x_ticks and x_label.
        y_labels (bool, optional):
            If false, will remove y_ticks and y_label.
        legend_n_columns (integer):
            Manually set the number of columns for the legend.
        x_tick_fontsize (int, optional):
            Font size for x-axis tick labels.
        x_label_fontsize (int, optional):
            Font size for x-axis label.
        y_tick_fontsize (int, optional):
            Font size for y-axis tick labels.
        y_label_fontsize (int, optional):
            Font size for y-axis label.
        legend_fontsize (int, optional):
            Font size for the legend.
        plt_label_fontsize (int, optional):
            Font size for the plot label.
        planet_name_fontsize (int, optional):
            Font size for the planet name.
        plot_style (str, optional):
            (Experimental!) plot style ('standard' or 'fancy').
     
    Returns:
        fig (matplotlib figure object):
            The retrieved spectra plot.
    
    '''

    if (plot_style == 'fancy'):
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['lines.markersize'] = 3
        plt.rcParams['lines.markeredgewidth'] = 0
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.serif'] = 'DejaVu Sans'
        plt.rcParams['mathtext.fontset'] = 'dejavusans'

    else:
        plt.style.use('classic')
        plt.rc('font', family = 'serif')
        matplotlib.rcParams['svg.fonttype'] = 'none'
        matplotlib.rcParams['figure.facecolor'] = 'white'   

    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', '(Rp/R*)', 'transit_depth',
                   'transit_depth_ppm']):
        plot_type = 'transmission'
    elif (y_unit in ['Rp/Rs', 'Rp/R*', '(Rp/Rs)', '(Rp/R*)']):
        plot_type = 'planet_star_radius_ratio'
    elif (y_unit in ['Fp/Fs', 'Fp/F*', 'eclipse_depth', 'eclipse_depth_ppm']):
        plot_type = 'emission'
    elif (y_unit in ['Fp']):
        plot_type = 'direct_emission'
    else:
        raise Exception("Unexpected y unit. Did you mean 'transit_depth' " +
                       "or 'eclipse_depth'?")
    
    # Find number of spectra to plot
    N_spectra = len(spectra_median)

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'
         
    # Quick spectra validity checks for plotting
    if (N_spectra == 0):
        raise Exception("Must provide at least one spectrum to plot!")
    if (N_spectra > 5) and (len(colour_list) == 0):
        raise Exception("Max number of concurrent retrieved spectra to plot is 5 with default colour list.")
    if ((len(colour_list) != 0) and (N_spectra != len(colour_list))):
        raise Exception("Number of colours does not match number of spectra.")
    if ((len(binned_colour_list) != 0) and (N_spectra != len(binned_colour_list))):
        raise Exception("Number of binned model colours does not match number of spectra.")
    if ((len(spectra_labels) != 0) and (N_spectra != len(spectra_labels))):
        raise Exception("Number of model labels does not match number of spectra.")
    if ((len(line_width_list) != 0) and (N_spectra != len(line_width_list))):
        raise Exception("Number of line widths does not match number of spectra.")
    if ((len(line_style_list) != 0) and (N_spectra != len(line_style_list))):
        raise Exception("Number of line styles does not match number of spectra.")
    if ((len(line_alpha_list) != 0) and (N_spectra != len(line_alpha_list))):
        raise Exception("Number of line alphas does not match number of spectra.")

    # Define colours for plotted spectra (default or user choice)
    if (len(colour_list) == 0):   # If user did not specify a custom colour list
        colours = ['purple', 'darkorange', 'green', 'royalblue', 'grey']
    else:
        colours = colour_list

    # Define colours for binned model points (default or user choice)
    if (len(binned_colour_list) == 0):
        binned_colours = ['gold', 'pink', 'cyan', 'lime', 'white']
    else:
        binned_colours = binned_colour_list

    # Load default median spectrum line width if not specified by the user
    if (len(line_width_list) == 0):
        line_widths = np.full(N_spectra, 1.0)    # Default spectrum line width
    else:
        line_widths = line_width_list

    # Load default median spectrum line style if not specified by the user
    if (len(line_style_list) == 0):
        line_styles = np.full(N_spectra, '-')    # Default spectrum line style
    else:
        line_styles = line_style_list

    # Load default median spectrum line alpha if not specified by the user
    if (len(line_alpha_list) == 0):
        line_alphas = np.full(N_spectra, 1.0)    # Default spectrum line alpha
    else:
        line_alphas = line_alpha_list

    # Unpack data properties (if provided)
    datasets = data_properties['datasets']
    instruments = data_properties['instruments']
    ydata = data_properties['ydata']
    err_data = data_properties['err_data']
    wl_data = data_properties['wl_data']
    bin_size = data_properties['half_bin']

    # Find number of datasets to plot
    N_datasets = len(datasets)
        
    # Quick data validity checks for plotting
    if (N_datasets == 0):
        raise Exception("Must provide at least one dataset to plot!")
    if ((N_datasets > 10) and len(data_colour_list) == 0) or ((N_datasets > 10) and len(data_marker_list) == 0):
        raise Exception("Max number of concurrent datasets to plot is 10 with default data colours and markers.")
    if ((len(data_colour_list) != 0) and (N_datasets != len(data_colour_list))):
        raise Exception("Number of colours does not match number of datasets.")
    if ((len(data_labels) != 0) and (N_datasets != len(data_labels))):
        raise Exception("Number of dataset labels does not match number of datasets.")
    if ((len(data_marker_list) != 0) and (N_datasets != len(data_marker_list))):
        raise Exception("Number of dataset markers does not match number of datasets.")
    if ((len(data_marker_size_list) != 0) and (N_datasets != len(data_marker_size_list))):
        raise Exception("Number of dataset marker sizes does not match number of datasets.")
    if ((len(data_alpha_list) != 0) and (N_datasets != len(data_alpha_list))):
        raise Exception("Number of dataset alpha values does not match number of datasets.")
    if ((len(data_eline_alpha_list) != 0) and (N_datasets != len(data_eline_alpha_list))):
        raise Exception("Number of dataset alpha values does not match number of datasets.")
    if ((len(data_edge_width_list) != 0) and (N_datasets != len(data_edge_width_list))):
        raise Exception("Number of dataset marker sizes does not match number of datasets.")
    if ((len(data_eline_width_list) != 0) and (N_datasets != len(data_eline_width_list))):
        raise Exception("Number of error bar line widths does not match number of spectra.")
    if ((len(text_annotations) != 0) and (len(text_annotations) != len(annotation_pos))):
        raise Exception("Number of annotation labels does not match provided positions.")
    if ((len(data_eline_colour_list) != 0) and (len(data_eline_colour_list) != N_datasets)):
        raise Exception("Number of error bar colours must match number of datasets.")

    # Define colours for plotted spectra (default or user choice)
    if (len(data_colour_list) == 0):   # If user did not specify a custom colour list
        data_colours = ['lime', 'cyan', 'magenta', 'orange', 'brown', 'crimson',
                        'forestgreen', 'deepskyblue', 'grey', 'whitesmoke']
    else:
        data_colours = data_colour_list

    # Define data marker symbols (default or user choice)
    if (len(data_marker_list) == 0):   # If user did not specify a custom colour list
        if (N_datasets <= 5):
            data_markers = ['o', 's', 'D', '*', 'X']
        else:
            data_markers = np.full(N_datasets, 'o')
    else:
        data_markers = data_marker_list

    # Define data marker sizes (default or user choice)
    if (len(data_marker_size_list) == 0):
        data_markers_size = np.full(N_datasets, 3.0)   # Default data marker size
    else:
        data_markers_size = data_marker_size_list

    # Define data marker alpha (default or user choice)
    if (len(data_alpha_list) == 0):
        data_alphas = np.full(N_datasets, 1.0)   # Default data alpha
    else:
        data_alphas = data_alpha_list

    # Define data marker alpha (default or user choice)
    if (len(data_eline_alpha_list) == 0):
        data_eline_alphas = np.full(N_datasets, 0.8)   # Default error bar alpha
    else:
        data_eline_alphas = data_eline_alpha_list

    # Define data edge widths (default or user choice)
    if (len(data_edge_width_list) == 0):
        data_edge_widths = np.full(N_datasets, 0.8)   # Default data marker edge width
    else:
        data_edge_widths = data_edge_width_list

    # Define error bar line width (default or user choice)
    if (len(data_eline_width_list) == 0):
        data_eline_widths = np.full(N_datasets, 1.0)   # Default error line width
    else:
        data_eline_widths = data_eline_width_list

    #***** Apply any retrieved offsets to the data *****#

    if (add_retrieved_offsets == True):

        # Check model has been defined
        if (model == None):
            raise Exception('Please provide model to plot offsets')
        
        offset_datasets = model['offsets_applied']
        model_name = model['model_name']
        
        # Avoid overwriting the data points
        ydata_to_plot = np.array(ydata)

        # Add offsets for a single dataset 
        if (offset_datasets == 'single_dataset'):
            
            ### Unpack offset data properties (TBD: turn into function?) ###
            
            # offset_1_end == 0 is the default value for offset_1 array (meaning that the original offset_datasets was used)
            # The only difference is that the offset_1 setting can have multiple datasets with same offset

            if (data_properties['offset_1_end'] == 0):
                offset_start, offset_end = data_properties['offset_start'], data_properties['offset_end']
            else:
                offset_start, offset_end = data_properties['offset_1_start'], data_properties['offset_1_end']

            # Catch offsets for one dataset
            if isinstance(offset_start, np.int64):
                offset_start, offset_end = np.array([offset_start]), np.array([offset_end])

            # Retrieve offset value from results file
            results_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'
            results_file_name = model_name + '_results.txt'

            # Open results file to find retrieved median offset value
            with open(results_dir + results_file_name, 'r') as f:
                for line in f:
                    if ('delta_rel' in line):
                        delta_rel = float(line.split()[2])

                    # Stop reading file after 1 sigma constraints
                    if ('2 σ constraints' in line):
                        break

            for start, end in zip(offset_start, offset_end):
                # Note: offsets are in ppm
                ydata_to_plot[start:end] = ydata[start:end] - delta_rel*1e-6
            
            # If this is true, will append the offset applied to the dataset to the data labels 
            if (verbose_offsets == True):
                if (data_properties['offset_1_end'] == 0):
                    print('Applied ' + str(delta_rel) + ' ppm offset to offset_datasets')
                else:
                    print('Applied ' + str(delta_rel) + ' ppm offset to offset_1_datasets')
        
        # Add multiple offsets
        elif ((offset_datasets == 'two_datasets') or (offset_datasets == 'three_datasets')):
            #print('in two datasets')     

            # Unpack offset data properties
            if ((offset_datasets == 'two_datasets') and (data_properties['offset_1_start'] != 0)):
                offset_start_list = ['offset_1_start', 'offset_2_start']
                offset_end_list = ['offset_1_end', 'offset_2_end']
            elif ((offset_datasets == 'three_datasets') and (data_properties['offset_1_start'] != 0)):
                offset_start_list = ['offset_1_start', 'offset_2_start', 'offset_3_start']
                offset_end_list = ['offset_1_end', 'offset_2_end', 'offset_3_end']

            offset_start_end = []

            if (data_properties['offset_1_start'] != 0):
                for start_name, end_name in zip(offset_start_list, offset_end_list):
                    offset_start, offset_end = data_properties[start_name], data_properties[end_name]

                    print(offset_start, offset_end)

                    # Catch zero offsets, not defined as arrays
                    if isinstance(offset_start, np.int64):
                        offset_start, offset_end = np.array([offset_start]), np.array([offset_end])
                    
                #    if(len(offset_start) == 0):
                    offset_start_end.append((offset_start[0], offset_end[-1]))
            
            else:
                for i in range(len(data_properties['offset_start'])):
                    offset_start, offset_end = data_properties['offset_start'][i], data_properties['offset_end'][i]

                    offset_start_end.append((offset_start, offset_end))

            # Retrieve offset value from results file
            results_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'
            results_file_name = model_name + '_results.txt'

            # Create empty array for relative offsets (max. number of offsets is currently 3)
            delta_rel_array = np.zeros(3)

            # Open results file to find retrieved median offset value
            with open(results_dir + results_file_name, 'r') as f:
                for line in f:
                    if ('delta_rel_1' in line):
                        delta_rel_array[0] = line.split()[2]
                    if ('delta_rel_2' in line):
                        delta_rel_array[1] = line.split()[2]
                    if ('delta_rel_3' in line):
                        delta_rel_array[2] = line.split()[2]

                    # Stop reading file after 1 sigma constraints
                    if ('2 σ constraints' in line):
                        break

            # Add relative offset to ydata (note: offsets are subtracted)
            for delta_rel, (offset_start, offset_end) in zip(delta_rel_array, offset_start_end):
                # Note: offsets are in ppm
                ydata_to_plot[offset_start:offset_end] = ydata[offset_start:offset_end] - delta_rel*1e-6

            if (verbose_offsets == True):
                print('Applied ' + str(delta_rel_array[0]) + ' ppm offset to offset_1_datasets')
                print('Applied ' + str(delta_rel_array[1]) + ' ppm offset to offset_2_datasets')

                if (offset_datasets == 'three_datasets'):
                    print('Applied ' + str(delta_rel_array[2]) + ' ppm offset to offset_3_datasets')
        
        # Continue plotting if no offsets are found
        elif offset_datasets == None:
            print('No offsets found, plotting data without offsets')
        
    else:
        ydata_to_plot = ydata

    #***** Apply retrieved error inflation parameter to data *****#

    if (add_retrieved_error_inflation == True):

        # Check model has been defined
        if (model == None):
            raise Exception('Please provide model to plot error inflated data')
        
        error_inflation = model['error_inflation']
        model_name = model['model_name']

        # Add offsets for a single dataset 
        if (error_inflation == None):
            error_inflation_params = []
        else:
            if (error_inflation == 'Line15'):
                error_inflation_params = ['b']
            elif (error_inflation == 'Piette20'):
                error_inflation_params = ['x_tol']
            elif ('Line15' in error_inflation) and ('Piette20' in error_inflation):
                error_inflation_params = ['b', 'x_tol']
        
        # Retrieve offset value from results file
        results_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'
        results_file_name = model_name + '_results.txt'

        # Inflate error bars in the plot by the media retrieved error inflation parameter(s)
        if (error_inflation == None):
            err_data_to_plot = err_data
        else:
            err_inflation_param_values = []

            # Open results file to find retrieved median error inflation value
            with open(results_dir + results_file_name, 'r') as f:
                for line in f:
                    for error_inflation_parameter in error_inflation_params:
                        if (((error_inflation_parameter in line)) and (len(error_inflation_parameter) == len(line.split()[0]))):
                            err_inflation_param_values += [float(line.split()[2])]  # Median error inflation parameter

                    # Stop reading file after 1 sigma constraints
                    if ('2 σ constraints' in line):
                        break

            # Apply error inflation to the data (Line+2015 prescription)
            if (error_inflation == 'Line15'):

                # Calculate effective error bars including the median error inflation parameter
                err_data_to_plot = np.sqrt(err_data**2 + np.power(10.0, err_inflation_param_values[0]))

            # Apply error inflation to the data (Piette+2020 prescription)
            elif (error_inflation == 'Piette20'):

                # Extract median spectrum and wavelength grid
                (spec_med, wl) = spectra_median[0]

                # Bin the median spectrum to the data resolution
                ymodel_median = bin_spectrum_to_data(spec_med, wl, data_properties)

                # Calculate effective error bars including the median error inflation parameter
                err_data_to_plot = np.sqrt(err_data**2 + (err_inflation_param_values[0] * ymodel_median)**2)

            # Apply both error inflation prescriptions to data (Line+2015 & Piette+2020)
            elif (('Line15' in error_inflation) and ('Piette20' in error_inflation)):

                # Extract median spectrum and wavelength grid
                (spec_med, wl) = spectra_median[0]

                # Bin the median spectrum to the data resolution
                ymodel_median = bin_spectrum_to_data(spec_med, wl, data_properties)

                # Calculate effective error bars including the median error inflation parameter
                err_data_to_plot = np.sqrt(err_data**2 + np.power(10.0, err_inflation_param_values[0]) +
                                           (err_inflation_param_values[1] * ymodel_median)**2)
    
    else:
        err_data_to_plot = err_data

    #***** Find desirable y range for plot *****#

    # If the user did not specify a wavelength range, find min and max from input models
    if (wl_min == None):
        
        wl_min = 1e10   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            wl_min_i = np.min(spectra_median[i][1])
            wl_min = min(wl_min, wl_min_i)
            
    # If the user did not specify a wavelength range, find min and max from input models
    if (wl_max == None):
        
        wl_max = 1e-10  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            wl_max_i = np.max(spectra_median[i][1])
            wl_max = max(wl_max, wl_max_i)

    # If the user did not specify a y range, find min and max from input models
    if (y_min == None):
        
        y_min_plt = 1e10   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):

            (spec_low2, wl) = spectra_low2[i]
            _, spec_low2_binned, _ = bin_spectrum(wl, spec_low2, R_to_bin)

            y_min_i = np.min(spec_low2_binned)
            y_min_plt = min(y_min_plt, y_min_i)
            
        # Check if the lowest data point falls below the current y-limit
        if (y_min_plt > min(ydata_to_plot - err_data)):
            y_min_plt = min(ydata_to_plot - err_data)
            
        y_min_plt = 0.995*y_min_plt  # Extend slightly below
        
    else:
        y_min_plt = y_min

    if (y_max == None):
        
        y_max_plt = 1e-10  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):

            (spec_high2, wl) = spectra_high2[i]
            _, spec_high2_binned, _ = bin_spectrum(wl, spec_high2, R_to_bin)

            y_max_i = np.max(spec_high2_binned)
            y_max_plt = max(y_max_plt, y_max_i)
            
        # Check if the highest data point falls above the current y-limit
        if (y_max_plt < max(ydata + err_data)):
            y_max_plt = max(ydata + err_data)
            
        y_max_plt = 1.040*y_max_plt  # Extend slightly above
        
    else:
        y_max_plt = y_max

    #***** Format x and y ticks *****#

    # Create x formatting objects
    if (wl_max < 1.0):    # If plotting over the optical range
        xmajorLocator = MultipleLocator(0.1)
        xminorLocator = MultipleLocator(0.02)
        
    else:                 # If plot extends into the infrared
        xmajorLocator = MultipleLocator(1.0)
        xminorLocator = MultipleLocator(0.1)
            
    xmajorFormatter = FormatStrFormatter('%g')
    xminorFormatter = NullFormatter()
    
    # Aim for 10 major y-axis labels
    ymajor_spacing = round_sig_figs((y_max_plt - y_min_plt), 1)/10
    yminor_spacing = ymajor_spacing/10

    if (np.abs(np.log10(ymajor_spacing)) <= 10.0):    
        major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 1)
        minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 1)
    else:    # Bug fix for surface spectra where Fp > 1e10 
        major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 2)
        minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 2)
    
    # If last digit of y labels would be a multiple of 6,7,8,or 9, bump up to 10
    if (ymajor_spacing > 5*np.power(10, major_exponent)):
        ymajor_spacing = 1*np.power(10, major_exponent+1)
    elif (ymajor_spacing == 3*np.power(10, major_exponent)):
        ymajor_spacing = 2*np.power(10, major_exponent)
    if (yminor_spacing > 5*np.power(10, minor_exponent)):
        yminor_spacing = 1*np.power(10, minor_exponent+1)
    elif (yminor_spacing == 3*np.power(10, minor_exponent)):
        yminor_spacing = 2*np.power(10, minor_exponent)

    # Refine y range to be a multiple of the tick spacing (only if range not specified by user)
    if (y_min == None):
        y_min_plt = np.floor(y_min_plt/ymajor_spacing)*ymajor_spacing
    if (y_max == None):
        y_max_plt = np.ceil(y_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    y_range = [y_min_plt, y_max_plt]

    # Create y formatting objects
    ymajorLocator = MultipleLocator(ymajor_spacing)

    if ((plot_type == 'planet_star_radius_ratio') or (y_min_plt > 0.10)):
        ymajorFormatter = ScalarFormatter(useMathText=False)
    else:
        ymajorFormatter = ScalarFormatter(useMathText=True)
        ymajorFormatter.set_powerlimits((0,0))

    yminorLocator = MultipleLocator(yminor_spacing)

    # Generate figure and axes
    fig = plt.figure()

    # Set figure size
    if (figure_shape == 'default'):
        fig.set_size_inches(8.0, 6.0)    # Default Matplotlib figure size
    elif (figure_shape == 'wide'):
        fig.set_size_inches(10.667, 6.0)    # 16:9 widescreen format (for two column figures) 
    elif (legend_location == 'outside right'):
        fig.set_size_inches(12, 8.0) 

    if (ax == None):
        ax1 = plt.gca()
    else:
        ax1 = ax
    
    # Set x axis to be linear or logarithmic
    ax1.set_xscale(wl_axis)

    # Assign formatter objects to axes
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    ax1.yaxis.set_minor_locator(yminorLocator)
    
    for i in range(N_spectra):
        
        # Extract spectrum and wavelength grid
        (spec_med, wl) = spectra_median[i]
        (spec_low1, wl) = spectra_low1[i]
        (spec_low2, wl) = spectra_low2[i]
        (spec_high1, wl) = spectra_high1[i]
        (spec_high2, wl) = spectra_high2[i]
        
        # If user did not specify a model label, just call them "Model 1, 2" etc.
        if (len(spectra_labels) == 0):
            if (N_spectra == 1):
                label_i = r'Retrieved Spectrum'
            else:
                label_i = r'Retrieved Spectrum ' + str(i+1)
        else:
            label_i = spectra_labels[i]
        
        # Calculate binned wavelength and retrieved spectra confidence intervals
        wl_binned, spec_med_binned, _ = bin_spectrum(wl, spec_med, R_to_bin)
        wl_binned, spec_low1_binned, _ = bin_spectrum(wl, spec_low1, R_to_bin)
        wl_binned, spec_low2_binned, _ = bin_spectrum(wl, spec_low2, R_to_bin)
        wl_binned, spec_high1_binned, _ = bin_spectrum(wl, spec_high1, R_to_bin)
        wl_binned, spec_high2_binned, _ = bin_spectrum(wl, spec_high2, R_to_bin)
        
        # Only add sigma intervals to legend for one model (avoids clutter)
        if (N_spectra == 1):
            label_med = label_i + r' (Median)'
            label_one_sig = label_i + r' ($1 \sigma$)'
            label_two_sig = label_i + r' ($2 \sigma$)'
        else:
            label_med = label_i
            label_one_sig = ''
            label_two_sig = ''

        # Plot median retrieved spectrum
        ax1.plot(wl_binned, spec_med_binned, lw = line_widths[i],
                 alpha = line_alphas[i],
                 color = scale_lightness(colours[i], 1.0), 
                 label = label_med,
                 linestyle = line_styles[i])
        
        # Plot +/- 1σ confidence region
        if sigma_to_plot == 1 or sigma_to_plot == 2:
            ax1.fill_between(wl_binned, spec_low1_binned, spec_high1_binned,
                            lw=0.0, alpha=0.5, facecolor=colours[i],  
                            label = label_one_sig)

        # Plot +/- 2σ sigma confidence region
        if sigma_to_plot ==2 :
            ax1.fill_between(wl_binned, spec_low2_binned, spec_high2_binned,
                            lw=0.0, alpha=0.2, facecolor=colours[i],  
                            label = label_two_sig)

        # Overplot median model, binned to resolution of the observations
        if (show_ymodel == True):
            ymodel_median = bin_spectrum_to_data(spec_med, wl, data_properties)

            ax1.scatter(wl_data, ymodel_median, color = binned_colours[i], 
                        s=5, marker='D', lw=0.5, alpha=0.8, edgecolor='black',
                        label = label_i + r' (Binned)', zorder = 200)
            
    # Overplot datapoints
    for i in range(N_datasets):
        
        # If user did not specify dataset labels, use the instrument names
        if (len(data_labels) == 0):
            label_i = instruments[i]
        else:
            label_i = data_labels[i]
        
        # Find start and end indices of dataset_i in dataset property arrays
        idx_start = data_properties['len_data_idx'][i]
        idx_end = data_properties['len_data_idx'][i+1]

        # Extract the ith dataset
        wl_data_i = wl_data[idx_start:idx_end]
        ydata_i = ydata_to_plot[idx_start:idx_end]
        err_data_i = err_data_to_plot[idx_start:idx_end]
        bin_size_i = bin_size[idx_start:idx_end]

        if (show_data_cap == True):
            capsize = 2
        else:
            capsize = 0

        # Plot dataset
        if (show_data_bin_width == True):
            x_bin_size = bin_size_i
        else:
            x_bin_size = None

        if (len(data_eline_colour_list) == 0):
            markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr = err_data_i, 
                                            xerr = x_bin_size, marker = data_markers[i], 
                                            markersize = data_markers_size[i], 
                                            capsize = capsize, ls='none',
                                            elinewidth = data_eline_widths[i], 
                                            color = data_colours[i], 
                                            alpha = data_eline_alphas[i],
                                            ecolor = err_colour, label=label_i,
                                            markeredgewidth = data_edge_widths[i],
                                            zorder = 100)
        else:
            markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr = err_data_i, 
                                            xerr = x_bin_size, marker = data_markers[i], 
                                            markersize = data_markers_size[i], 
                                            capsize = capsize, ls='none', 
                                            elinewidth = data_eline_widths[i], 
                                            color = data_colours[i], 
                                            alpha = data_eline_alphas[i],
                                            ecolor = data_eline_colour_list[i], label=label_i,
                                            markeredgewidth = data_edge_widths[i],
                                            zorder = 100)

        [markers.set_alpha(data_alphas[i])]

    # Plot text annotations
    if (len(text_annotations) != 0):

        for i in range(len(text_annotations)):

            # Plot each annotation at the location provided by the user
            ax1.text(annotation_pos[i][0], annotation_pos[i][1], 
                     text_annotations[i], fontsize=14, color = 'black')
    
    # Set axis ranges
    ax1.set_xlim([wl_min, wl_max])
    ax1.set_ylim([y_range[0], y_range[1]])
        
    # Set axis labels
    if (xlabels == True):
        ax1.set_xlabel(r'Wavelength (μm)', fontsize = x_label_fontsize)

    if (ylabels == True):
        if (plot_type == 'transmission'):
            if (y_unit == 'transit_depth_ppm'):
                ax1.set_ylabel(r'Transit Depth (ppm)', fontsize = y_label_fontsize)
            else:
                if (y_min_plt < 0.10):
                    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = y_label_fontsize)
                else:
                    ax1.set_ylabel(r'Transit Depth', fontsize = y_label_fontsize)
        elif (plot_type == 'planet_star_radius_ratio'):
            ax1.set_ylabel(r'$R_p/R_*$', fontsize = y_label_fontsize)
        elif (plot_type == 'emission'):
            if (y_unit == 'eclipse_depth_ppm'):
                ax1.set_ylabel(r'Eclipse Depth $(ppm)$', fontsize = y_label_fontsize)
            else:
                ax1.set_ylabel(r'Emission Spectrum $(F_p/F_*)$', fontsize = y_label_fontsize)
        elif (plot_type == 'direct_emission'):
            if (y_unit == 'Fp'):
                ax1.set_ylabel(r'$F_{\rm{p}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = y_label_fontsize)
            elif (y_unit in ['Fs', 'F*']):
                ax1.set_ylabel(r'$F_{\rm{s}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = y_label_fontsize)

    # Add planet name label
    if (show_planet_name == True):
        ax1.text(0.02, 0.96, planet_name, horizontalalignment = 'left', 
                 verticalalignment = 'top', transform = ax1.transAxes, fontsize = planet_name_fontsize)

    # Add plot label
    if (plt_label != None):
        if (show_planet_name == True):
            ax1.text(0.03, 0.90, plt_label, horizontalalignment = 'left', 
                     verticalalignment = 'top', transform = ax1.transAxes, fontsize = plt_label_fontsize)
        else:
            ax1.text(0.03, 0.96, plt_label, horizontalalignment = 'left', 
                     verticalalignment = 'top', transform = ax1.transAxes, fontsize = plt_label_fontsize)

    # Decide at which wavelengths to place major tick labels
    wl_ticks = set_spectrum_wl_ticks(wl_min, wl_max, wl_axis)
        
    # Plot wl tick labels
    if (xlabels == True):
        ax1.set_xticks(wl_ticks)
    else:
        ax1.set_xticks(wl_ticks)
        ax1.tick_params(labelbottom=False)  
    
    # If ylabels is False, don't show them
    if (ylabels == False):
        ax1.tick_params(labelleft=False)

    # Set the x and y tick font sizes
    ax1.tick_params(axis='x', labelsize=x_tick_fontsize)
    ax1.tick_params(axis='y', labelsize=y_tick_fontsize)

    # Switch to two columns if many spectra are being plotted
    if (legend_n_columns == 0):
        if (N_spectra >= 6):
            n_columns = 2
        else:
            n_columns = 1
    else:
        n_columns = legend_n_columns

    # Assign legend box settings
    if show_legend == True:
        if (legend_box == True):
            frameon = True
            framefacecolour = '0.9'
        else:
            frameon = False
            framefacecolour = None

        # Add legend
        if isinstance(legend_location, tuple):
            legend = ax1.legend(loc = 'center', shadow = True, prop = {'size': legend_fontsize},
                                ncol = n_columns, frameon = frameon, bbox_to_anchor = legend_location)
        elif legend_location == 'outside right':
            legend = ax1.legend(loc='center left', shadow = True, prop = {'size':legend_fontsize}, 
                                ncol = n_columns, frameon = frameon, bbox_to_anchor = (1, 0.5))
        else:
            legend = ax1.legend(loc = legend_location, shadow = True, prop={'size': legend_fontsize},
                                ncol = n_columns, frameon = frameon)  # Legend settings

        frame = legend.get_frame()
        frame.set_facecolor(framefacecolour)
        
        legend.set_zorder(200)   # Make legend always appear in front of everything

        # Set legend line width
        if len(legend_line_size) != 0:
            # Check legend line size length
            try:
                if (len(legend_line_size) != len(legend.legend_handles)):
                    raise Exception("Make sure legend_line_size length is equal to number of handles.")
            except:
                # weird attribute error
                if (len(legend_line_size) != len(legend.legendHandles)):
                    raise Exception("Make sure legend_line_size length is equal to number of handles.")
            try:
                for i in range(len(legend.legend_handles)):
                    legline = legend.legend_handles[i]
                    legline.set_linewidth(legend_line_size[i])
            except AttributeError:
                for i in range(len(legend.legendHandles)):
                    legline = legend.legendHandles[i]
                    legline.set_linewidth(legend_line_size[i])

    plt.tight_layout()

    # Write figure to file
    if (save_fig == True):
        if (plt_label == None):
            file_name = output_dir + planet_name + '_retrieved_spectra.pdf'
        else:
            file_name = output_dir + planet_name + '_' + plt_label + '_retrieved_spectra.pdf'

        plt.savefig(file_name, bbox_inches = 'tight')

    return fig


def plot_PT_retrieved(planet_name, PT_median, PT_low2, PT_low1, PT_high1,
                      PT_high2, T_true = None, Atmosphere_dimension = 1, 
                      TwoD_type = None, plt_label = None, show_profiles = [],
                      PT_labels = [], colour_list = [], log_P_min = None,
                      log_P_max = None, T_min = None, T_max = None,
                      legend_location = 'lower left',
                      ax = None, save_fig = True,
                      sigma_to_plot = 2,
                      show_legend = True,
                      custom_ticks = [],
                      ylabels = True,
                      retrieved_log_P_surf = [],
                      log_P_surf_sigma_upper_lower = 'upper',
                      log_P_surf_histogram_list = []):
    '''
    Plot retrieved Pressure-Temperature (P-T) profiles.
    
    Args:
        planet_name (str): 
            The name of the planet.
        PT_median (list of tuples): 
            List of tuples containing the median temperature and pressure grids 
            for each model, each with the format (T_median, P).
        PT_low2 (list of tuples): 
            Corresponding list of -2σ confidence intervals on the retrieved 
            temperature, each with the format (T_low2, P).
        PT_low1 (list of tuples): 
            Corresponding list of -1σ confidence intervals on the retrieved 
            temperature, each with the format (T_low1, P).
        PT_high1 (list of tuples):
            Corresponding list of +1σ confidence intervals on the retrieved 
            temperature, each with the format (T_high1, P).
        PT_high2 (list of tuples): 
            Corresponding list of +2σ confidence intervals on the retrieved 
            temperature, each with the format (T_high2, P).
        T_true (np.array, optional): 
            True temperature profile (optional).
        Atmosphere_dimension (int, optional): 
            Dimensionality of the atmospheric model.
        TwoD_type (str, optional): 
            If 'Atmosphere_dimension' = 2, the type of 2D model
            (Options: 'D-N' for day-night, 'E-M' for evening-morning).
        plt_label (list, optional): 
            List of labels for each model.
        show_profiles (list, optional): 
            If model is 2D or 3D, which profiles to plot.
        PT_labels (list, optional): 
            List of labels for each retrieved P-T profile.
        colour_list (list, optional): 
            List of colours for each retrieved P-T profile.
		log_P_min (float, optional):
            Minimum value for the log10 pressure.
		log_P_max (float, optional):
            Maximum value for the log10 pressure.
		T_min (float, optional):
            Minimum temperature to plot.
		T_max (float, optional):
            Maximum temperature to plot.
		legend_location (str, optional):
            Location of the legend. Default is 'lower left'.
		show_legend (bool, optional):
            If False, will not show legend.
        custom_ticks (list, optional): 
            Major and minor ticks
        ylabels (bool, optional):
            If False, will not plot y ticks
        retrieved_log_P_surf (list, optional):
            Will overplot 1 sigma P_surf (Retrieved log_P_surf, one_sigma_positive, one_sigma_negative)
        log_P_surf_sigma_upper_lower (str, optional):
            Will set things depending on if its an upper or lower limit or unconstrained
	
    Returns:
		fig (matplotlib figure object):
            The retrieved P-T profile plot.

    '''

    # Find number of P-T profiles to plot
    N_PT = len(PT_median)

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Quick validity checks for plotting
    if (N_PT == 0):
        raise Exception("Must provide at least one P-T profile to plot!")
    if (N_PT > 4):
        raise Exception("Max number of concurrent retrieved P-T profiles to plot is 4.")
    if ((len(colour_list) != 0) and (N_PT != len(colour_list))):
        raise Exception("Number of colours does not match number of P-T profiles.")
    if ((len(PT_labels) != 0) and (N_PT != len(PT_labels))):
        raise Exception("Number of model labels does not match number of P-T profiles.")

    # Define colours for plotted spectra (default or user choice)
    if (len(colour_list) == 0):   # If user did not specify a custom colour list
        colours = ['purple', 'darkorange', 'green']
    else:
        colours = colour_list

    # If the user did not specify a temperature range, find min and max from input models
    if (T_min == None):
        
        T_min = 1e10   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_PT):
            
            T_min_i = np.min(PT_low2[i][0])
            T_min = min(T_min, T_min_i)

        T_min = np.floor(T_min/100)*100 - 200.0    # Round down to nearest 100
            
    # If the user did not specify a temperature range, find min and max from input models
    if (T_max == None):
        
        T_max = 1e-10  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_PT):
            
            T_max_i = np.max(PT_high2[i][0])
            T_max = max(T_max, T_max_i)

        T_max = np.ceil(T_max/100)*100 + 200.0     # Round up to nearest 100

    # Find range to plot
    T_range = T_max - T_min
    
    # Calculate appropriate axis spacing
    if (len(custom_ticks) == 0):
        if (T_range >= 500.0):
            major_spacing = max(np.around((T_range/10), -2), 100.0)
        elif (T_range < 500.0):
            major_spacing = max(np.around((T_range/10), -1), 10.0)
            
        minor_spacing = major_spacing/10
    else:
        major_spacing = custom_ticks[0]
        minor_spacing = custom_ticks[1]

    # Load pressure grid
    P = PT_median[0][1]

    if (log_P_min == None):
        log_P_min = np.log10(np.min(P))
    if (log_P_max == None):
        log_P_max = np.log10(np.max(P))
    
    # create figure
    fig = plt.figure()

    if (ax == None):
        ax1 = plt.gca()
    else:
        ax1 = ax
    
    # Assign axis spacing
    xmajorLocator_PT = MultipleLocator(major_spacing)
    xminorLocator_PT = MultipleLocator(minor_spacing)
        
    ax1.xaxis.set_major_locator(xmajorLocator_PT)
    ax1.xaxis.set_minor_locator(xminorLocator_PT)
    
    #***** Plot P-T profiles *****#
    
    # 1D temperature profile
    if (Atmosphere_dimension > 1):
        raise Exception("This function does not currently support " + 
                        "multidimensional retrievals.")
        
    else:

        # Loop over retrieved P-T profiles
        for i in range(N_PT):
            
            # Extract temperature and pressure grid
            (T_med, P) = PT_median[i]
            (T_low1, P) = PT_low1[i]
            (T_low2, P) = PT_low2[i]
            (T_high1, P) = PT_high1[i]
            (T_high2, P) = PT_high2[i]
            
            # If user did not specify a model label, just call them "Model 1, 2" etc.
            if (len(PT_labels) == 0):
                if (N_PT == 1):
                    label_i = r'Retrieved P-T Profile'
                else:
                    label_i = r'Retrieved P-T Profile ' + str(i+1)
            else:
                label_i = PT_labels[i]
            
            # Only add sigma intervals to legend for one model (avoids clutter)
            if (N_PT == 1):
                label_med = label_i + r' (Median)'
                label_one_sig = label_i + r' ($1 \sigma$)'
                label_two_sig = label_i + r' ($2 \sigma$)'
            else:
                label_med = label_i
                label_one_sig = ''
                label_two_sig = ''

            # Plot median retrieved spectrum
            ax1.semilogy(T_med, P, lw = 1.5, color = scale_lightness(colours[i], 1.0), 
                        label = label_med)
            
            # Plot +/- 1σ confidence region
            if sigma_to_plot == 1 or sigma_to_plot == 2:
                ax1.fill_betweenx(P, T_low1, T_high1, lw = 0.0, alpha = 0.5, 
                                facecolor = colours[i], label = label_one_sig)

            # Plot +/- 2σ sigma confidence region
            if sigma_to_plot == 2:
                ax1.fill_betweenx(P, T_low2, T_high2, lw = 0.0, alpha = 0.2, 
                                facecolor = colours[i], label = label_two_sig)

        # Plot actual (true) P-T profile
        if (T_true != None):
            ax1.semilogy(T_true, P, lw = 1.5, color = 'crimson', label = 'True')

    # Plot the retrieved surface pressure
    # This assumes the distribution is a tailed distribution (rn)
    if (len(retrieved_log_P_surf) != 0):

        median_P_surf = 10**retrieved_log_P_surf[0]

        # Note this is before things are flipped, so in order to keep it less confusing its top and bottom in traditional sense
        # So this is actually the higher pressure, and will be bottom once axis if flipped

        P_surf_high = 10**(retrieved_log_P_surf[0] + retrieved_log_P_surf[1])
        P_surf_low = 10**(retrieved_log_P_surf[0] + retrieved_log_P_surf[2])

        # If the arrow is up, it will point from the highest pressure to the lowest
        # If the arrow is down, it will point from the lowest pressure to the highest 

        if log_P_surf_sigma_upper_lower == 'upper':
            #ax1.axhspan(P_surf_low,np.max(P), lw = 0.0, alpha = 0.5, color = 'darkgray')
            ax1.axhline(P_surf_low, lw = 3.0, color = scale_lightness(colours[i], 1.0), label = 'Surface Pressure ($1 \sigma$ upper)')
            ax1.axhspan(P_surf_low,median_P_surf, lw = 0.0, alpha = 0.5, color = colours[i],  hatch = 'xx')
            ax1.axhspan(median_P_surf,P_surf_high, lw = 0.0, alpha = 0.25, color = colours[i],  hatch = 'x')
            #ax1.axhspan(P_surf_low,np.max(P), lw = 0.0, alpha = 0.5, color = 'darkgray')
            ax1.axhspan(P_surf_high,np.max(P), lw = 0.0, alpha = 0.5, color = 'darkgray', hatch = '+++')

        elif log_P_surf_sigma_upper_lower == 'lower':
            ax1.axhspan(median_P_surf,np.max(P), lw = 0.0, alpha = 0.5, color = 'darkgray')
            ax1.axhline(P_surf_high, lw = 3.0, color = scale_lightness(colours[i], 1.0), label = 'Surface Pressure ($1 \sigma$ lower)')
            ax1.axhspan(P_surf_high,median_P_surf, lw = 0.0, alpha = 0.5, color = colours[i],  hatch = 'xx')
            ax1.axhspan(median_P_surf,P_surf_low, lw = 0.0, alpha = 0.25, color = colours[i],  hatch = 'x')
            ax1.axhspan(P_surf_high,np.max(P), lw = 0.0, alpha = 0.5, color = 'darkgray', hatch = '+++')

        ax1.axhline(median_P_surf, lw = 1.0, color = scale_lightness(colours[i], 1.0), label = 'Surface Pressure (Median)')

        # Draw arrow (either up or down to P_surf_low)
        #if arrow_P_surf == 'up':
        #    x = arrow_x
        #    y = median_P_surf
        #    dx = 0
        #    dy = -(median_P_surf - P_surf_low)
        #    ax1.arrow(x,y,dx,dy, color = colours[i],length_includes_head = True,
        #  head_width=50, head_length=0.1*np.abs(dy))
        #elif arrow_P_surf == 'down':
        #    x = arrow_x
        #    dy = median_P_surf - P_surf_low
        #    dx = 0
        #    y = P_surf_low
        #    ax1.arrow(x,y,dx,dy, color = colours[i],length_includes_head = True,
        #  head_width=50, head_length=0.1*np.abs(dy))

    if len(log_P_surf_histogram_list) != 0:
        log_P_surf_histogram_bool = log_P_surf_histogram_list[0]
        if log_P_surf_histogram_bool == True:
            planet = log_P_surf_histogram_list[1]
            model =  log_P_surf_histogram_list[2]
            log_P_surf_title = log_P_surf_histogram_list[3]

            ax_histy = ax1.inset_axes([0.85, 0, 0.15, 1])

            _ = plot_histograms(planet, [model], plot_parameters = ['log_P_surf'], 
                                span = ((-6,2)),
                                N_bins = [50],
                                parameter_colour_list = colour_list,
                                axes = [ax_histy], save_fig = False,
                                tick_labelsize = 14,               
                                title_fontsize = 16,                        
                                alpha_hist = 0.7,
                                show_title = True,
                                orientation = 'horizontal',
                                custom_labels = ['']
                                )
            
            ax_histy.set_ylim(ax_histy.get_ylim()[::-1])

            ax_histy.set_xticks([])
            ax_histy.yaxis.set_label_position("right")
            if log_P_surf_title == True:
                ax_histy.set_ylabel('log P$_{\mathrm{surf}}$', rotation = 270, labelpad = 20, fontsize = 15)




    # Common plot settings for all profiles
    ax1.invert_yaxis()            
    ax1.set_xlabel(r'Temperature (K)', fontsize = 16)
    ax1.set_xlim(T_min, T_max)
    ax1.set_ylim(np.power(10.0, log_P_max), np.power(10.0, log_P_min))

    # If ylabels = False, don't show the ticks
    if ylabels == False:
        ax1.tick_params(labelleft=False) 
    # Else, set the ylabel 
    else:
        ax1.set_ylabel(r'Pressure (bar)', fontsize = 16)

    ax1.tick_params(labelsize=12)
    
    # Add legend
    if show_legend == True:
        legend = ax1.legend(loc=legend_location, shadow=True, prop={'size':10}, ncol=1, 
                        frameon=False, columnspacing=1.0)
    
    fig.set_size_inches(9.0, 9.0)

    # Write figure to file
    if (save_fig == True):
        if (plt_label == None):
            file_name = output_dir + planet_name + '_retrieved_PT.pdf'
        else:
            file_name = output_dir + planet_name + '_' + plt_label + '_retrieved_PT.pdf'

        plt.savefig(file_name, bbox_inches = 'tight')

    return fig


def plot_chem_retrieved(planet_name, chemical_species, log_Xs_median, 
                        log_Xs_low2, log_Xs_low1, log_Xs_high1, log_Xs_high2, 
                        log_X_true = None, plot_species = [], plot_two_sigma = False,
                        Atmosphere_dimension = 1, TwoD_type = None, plt_label = None, 
                        show_profiles = [], model_labels = [], colour_list = [],
                        log_P_min = None, log_P_max = None, log_X_min = None, 
                        log_X_max = None):
    '''
    Plot retrieved mixing ratio profiles.
    
    Args:
        planet_name (str): 
            The name of the planet.
        chemical_species (list, optional):
            List of chemical species to plot. If not specified, default to all 
            chemical species in the model (including bulk species).
        log_Xs_median (list of tuples): 
            List of tuples containing the median retrieved log10 mixing ratio
            for each chemical species (for a single model) and its corresponding 
            pressure grid, each with the format (log10 X_median, P).
        log_Xs_low2 (list of tuples): 
            Corresponding list of -2σ confidence intervals on the retrieved 
            log10 mixing ratio for each chemical species, each with the 
            form (log10 X_low2, P).
        log_Xs_low1 (list of tuples): 
            Corresponding list of -1σ confidence intervals on the retrieved 
            log10 mixing ratio for each chemical species, each with the 
            form (log10 X_low1, P).
        log_Xs_high1 (list of tuples): 
            Corresponding list of +1σ confidence intervals on the retrieved 
            log10 mixing ratio for each chemical species, each with the 
            form (log10 X_high1, P).
        log_Xs_high2 (list of tuples): 
            Corresponding list of +2σ confidence intervals on the retrieved 
            log10 mixing ratio for each chemical species, each with the 
            form (log10 X_high2, P).
        log_X_true (2D np.array, optional): 
            True log10 mixing ratio profiles for each chemical species.
        plot_species (list, optional):
            List of chemical species to plot. If not specified, default to all 
            chemical species in the model (including bulk species).
        plot_two_sigma (bool, optional):
            If False, only plots the median and +/- 1σ confidence intervals for
            each chemical species (default behaviour to avoid clutter).
        Atmosphere_dimension (int, optional): 
            Dimensionality of the atmospheric model.
        TwoD_type (str, optional): 
            If 'Atmosphere_dimension' = 2, the type of 2D model
            (Options: 'D-N' for day-night, 'E-M' for evening-morning).
        plt_label (list, optional): 
            List of labels for each model.
        show_profiles (list, optional): 
            If model is 2D or 3D, which profiles to plot.
        model_labels (list, optional): 
            List of labels for each retrieved chemical profile (only one model
            currently supported).
        colour_list (list, optional): 
            List of colours for each retrieved chemical profile.
		    log_P_min (float, optional):
            Minimum value for the log10 pressure.
		    log_P_max (float, optional):
            Maximum value for the log10 pressure.
		    log_X_min (float, optional):
            Minimum log10 mixing ratio to plot.
		    log_X_max (float, optional):
            Maximum log10 mixing ratio to plot.
		    legend_location (str, optional):
            Location of the legend. Default is 'lower left'.
        log_P_min (float, optional):
            Minimum value for the log10 pressure.
        log_P_max (float, optional):
            Maximum value for the log10 pressure.
        log_X_min (float, optional):
            Minimum log10 mixing ratio to plot.
        log_X_max (float, optional):
            Maximum log10 mixing ratio to plot.
	
    Returns:
		    fig (matplotlib figure object):
            The retrieved mixing ratio profile plot.

    '''
  
    # Find number of mixing ratio model profiles to plot
    N_chem = len(log_Xs_median)

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # If the user did not specify which species to plot, plot all of them
    if (len(plot_species) == 0):
        plot_species = chemical_species

    # Quick validity checks for plotting
    if (N_chem > 1):
        raise Exception("Only 1 set of mixing ratio profiles can be plotted currently.")
    if (len(plot_species) > 8):
        raise Exception("Max number of concurrent species on plot is 8.\n"
                        "Please specify species to plot via plot_species = [LIST]")
    if ((len(colour_list) != 0) and (len(plot_species) != len(colour_list))):
        raise Exception("Number of colours does not match number of species.")
    if ((len(model_labels) != 0) and (N_chem != len(model_labels))):
        raise Exception("Number of model labels does not match number of mixing ratio profiles.")
    for q, species in enumerate(plot_species):
        if (species not in chemical_species):
            raise Exception(species + " not included in this model.")

    # Define colours for mixing ratio profiles (default or user choice)
    if (len(colour_list) == 0):   # If user did not specify a custom colour list
        colours = ['royalblue', 'darkgreen', 'magenta', 'crimson', 'darkgrey', 
                   'black', 'darkorange', 'navy']
    else:
        colours = colour_list

    # If the user did not specify a mixing ratio range, find min and max from input models
    if (log_X_min == None):
        
        log_X_min = 0.0   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_chem):
            
            log_X_min_i = np.min(log_Xs_low2[i][0])
            log_X_min = min(log_X_min, log_X_min_i)
            
    # If the user did not specify a mixing ratio range, find min and max from input models
    if (log_X_max == None):
        
        log_X_max = -50.0  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_chem):
            
            log_X_max_i = np.max(log_Xs_high2[i][0])
            log_X_max = max(log_X_max, log_X_max_i)

    # Find minimum and maximum mixing ratios in atmosphere
    log_X_min = np.floor(log_X_min) - 1.0
    log_X_max = min((np.ceil(log_X_max) + 1.0), 0.0)

    # When range is small, extend axes +/- 1 dex either side
    if (log_X_min == log_X_max):
        log_X_min = log_X_min - 1.0
        log_X_max = log_X_max + 1.0
        
    # Find range to plot
    log_X_range = log_X_max - log_X_min    
    
    # Calculate appropriate axis spacing
    major_spacing = 1.0
    minor_spacing = major_spacing/10

    # Load pressure grid
    P = log_Xs_median[0][1]

    if (log_P_min == None):
        log_P_min = np.log10(np.min(P))
    if (log_P_max == None):
        log_P_max = np.log10(np.max(P))
    
    # Find LaTeX code for each chemical species to plot
    latex_species = generate_latex_param_names(plot_species)
    
    # create figure
    fig = plt.figure()  
    ax = plt.gca()
    
    # Assign axis spacing
    xmajorLocator_X = MultipleLocator(major_spacing)
    xminorLocator_X = MultipleLocator(minor_spacing)
        
    ax.xaxis.set_major_locator(xmajorLocator_X)
    ax.xaxis.set_minor_locator(xminorLocator_X)
    
    #***** Plot mixing ratio profiles *****#
    
    # 1D temperature profile
    if (Atmosphere_dimension > 1):
        raise Exception("This function does not currently support " + 
                        "multidimensional retrievals.")
        
    else:

        # Loop over retrieved mixing ratio profile models
        for i in range(N_chem):
            
            # Extract mixing ratio and pressure grid
            (log_X_med, P) = log_Xs_median[i]
            (log_X_low1, P) = log_Xs_low1[i]
            (log_X_low2, P) = log_Xs_low2[i]
            (log_X_high1, P) = log_Xs_high1[i]
            (log_X_high2, P) = log_Xs_high2[i]

            # Plot the profile for each species in turn
            for q, species in enumerate(plot_species):
 
                # If user did not specify a model label, just call them "Model 1, 2" etc.
                if (len(model_labels) == 0):
                    if (N_chem == 1):
                        label_i = r'Retrieved ' + latex_species[q]
                    else:
                        label_i = r'Retrieved ' + latex_species[q] + str(i+1)
                else:
                    label_i = latex_species[q] + ' ' + model_labels[i]
            
                # Don't add sigma intervals to legend (avoids clutter)
                label_med = label_i
                label_one_sig = ''
                label_two_sig = ''

                # Plot median retrieved mixing ratio profile
                ax.semilogy(log_X_med[chemical_species == species,:][0], P, 
                            lw = 1.5, color = colours[q],
                            label = label_med)

                # Plot +/- 1σ confidence region
                ax.fill_betweenx(P, log_X_low1[chemical_species == species,:][0], 
                                 log_X_high1[chemical_species == species,:][0],
                                 lw = 0.0, alpha = 0.4, facecolor = colours[q],
                                 label = label_one_sig)

                # Plot +/- 2σ confidence region
                if (plot_two_sigma == True):
                    ax.fill_betweenx(P, log_X_low2[chemical_species == species,:][0], 
                                    log_X_high2[chemical_species == species,:][0],
                                    lw = 0.0, alpha = 0.2, facecolor = colours[q],
                                    label = label_two_sig)

                # Plot actual (true) mixing ratio profile
                if (log_X_true != None):

                    ax.semilogy(log_X_true[chemical_species == species,:][0], P, 
                                lw = 1.5, color = colours[q], ls = linestyles['dashed'],
                                label = r'True ' + latex_species[q])

    # Common plot settings for all profiles
    ax.invert_yaxis()            
    ax.set_xlabel(r'Mixing Ratios (log $X_{\rm{i}}$)', fontsize = 20)
    ax.set_xlim(log_X_min, log_X_max)  
    ax.set_ylabel(r'Pressure (bar)', fontsize = 20)
    ax.set_ylim(np.power(10.0, log_P_max), np.power(10.0, log_P_min))

    ax.tick_params(labelsize=12)
        
    # Add legend
    legend = ax.legend(loc='upper right', shadow=True, prop={'size':14}, ncol=1,
                       frameon=True, columnspacing=1.0)
    frame = legend.get_frame()
    frame.set_facecolor('0.90') 
    
    fig.set_size_inches(9.0, 9.0)

    # Write figure to file
    if (plt_label == None):
        file_name = output_dir + planet_name + '_retrieved_chem.pdf'
    else:
        file_name = output_dir + planet_name + '_' + plt_label + '_retrieved_chem.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_stellar_flux(flux, wl, wl_min = None, wl_max = None, flux_min = None,
                      flux_max = None, flux_axis = 'linear', wl_axis = 'log'):
    '''
    Straightforward function to plot an emergent stellar spectrum.

    Args:
        flux (np.array): 
            Surface flux of the star as a function of wavelength.
        wl (np.array): 
            Corresponding wavelength array.
        wl_min (float, optional):
            Minimum wavelength for x axis.
        wl_max (float, optional):
            Maximum wavelength for x axis.
        flux_min (float, optional):
            Minimum flux for y axis.
        flux_max (float, optional):
            Maximum flux for y axis.
        flux_axis (str, optional):
            'linear' or 'log' axis scaling for the y-axis. Default is 'linear'.
        wl_axis (str, optional):
            'linear' or 'log' axis scaling for the x-axis. Default is 'log'.
    
    Returns:
        fig (matplotlib figure object):
            The simplest stellar flux plot you've ever seen.

    '''

    print("WARNING: This function is deprecated and will be removed in a future " + 
          "version of POSEIDON.")
    
    # Initialise figure
    fig = plt.figure()  
    ax = plt.gca()

    # Format axes
    ax.set_yscale(flux_axis)
    ax.set_xscale(wl_axis)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

    # Plot the spectrum
    ax.plot(wl, flux, lw=1, alpha=0.8, label=r'Stellar Flux')

    # Add axis labels
    ax.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax.set_ylabel(r'Surface Flux (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

    # Check if user has specified x and y ranges
    if (wl_min == None):
        wl_min = min(wl)
    if (wl_max == None):
        wl_max = max(wl)
    if (flux_min == None):
        flux_min = min(flux)
    if (flux_max == None):
        flux_max = max(flux)   

    # Set x and y ranges
    ax.set_xlim([wl_min, wl_max])
    ax.set_ylim([flux_min, flux_max])

    # add legend
    ax.legend(loc='upper right', shadow=True, prop={'size':10}, ncol=1, frameon=False)
    
    return fig


def plot_histogram(nbins, vals, colour, ax, shrink_factor, x_max_array, alpha_hist, orientation):
    '''
    Function to plot a histogram of parameter values.

    Args:
        nbins (int): 
            Number of bins for the histogram.
        vals (list): 
            List of parameter values for each model.
        colour (str): 
            Colour for the histogram.
        ax (matplotlib axis object): 
            Axis to plot on.
        shrink_factor (float): 
            Factor to shrink the y-axis.
        x_max_array (np.array): 
            Array of maximum x values for scaling.
        alpha_hist (float): 
            Alpha value for histogram bars.

    Returns:
        low3, low2, low1, median, high1, high2, high3: 
            Confidence intervals for the parameter values.
    
    '''

    
  #  weights = np.ones_like(vals)/float(len(vals))
    
    # Plot histogram
    x,w,patches = ax.hist(vals, bins=nbins, color=colour, histtype='stepfilled', 
                          alpha=alpha_hist, edgecolor='None', density=True, stacked=True,
                          orientation = orientation)

    # Plot histogram border
    x,w,patches = ax.hist(vals, bins=nbins, histtype='stepfilled', lw = 0.8, 
                          facecolor='None', density=True, stacked=True,
                          orientation = orientation)
    
    if orientation == 'vertical':
        x_max = np.max(x_max_array)
            
        ax.set_ylim(0, (1.1+shrink_factor)*x_max)
    
    else:
        x_max = np.max(x_max_array)
            
        ax.set_xlim(0, (1.1+shrink_factor)*x_max)
    
    low3, low2, low1, median, high1, high2, high3 = confidence_intervals(len(vals), vals, 0)
    
    return low3, low2, low1, median, high1, high2, high3


def plot_parameter_panel(ax, param_vals, N_bins, param_min, param_max, 
                         colour, x_max_array, alpha_hist, orientation):
    '''
    Setup function to plot the histogram panel for a given parameter.

    Args:
        ax (matplotlib axis object): 
            Axis to plot on.
        param_vals (list): 
            List of parameter values for each model.
        N_bins (int): 
            Number of bins for the histogram.
        param_min (float): 
            Minimum value for the parameter.
        param_max (float): 
            Maximum value for the parameter.
        colour (str): 
            Colour for the histogram.
        x_max_array (np.array): 
            Array of maximum x values for scaling.
        alpha_hist (float): 
            Alpha value for histogram bars.

    Returns:
        low1 (float): 
            Lower 1σ confidence interval.
        median (float): 
            Median value.
        high1 (float): 
            Upper 1σ confidence interval.
    '''
    
    
    # Plot histogram
    _, low2, low1, median, high1, high2, _ = plot_histogram(N_bins, param_vals, colour, ax, 0.0, x_max_array, alpha_hist, orientation)

    # Adjust x-axis extent
    if orientation == 'vertical':
        ax.set_xlim(param_min, param_max)
    else:
        ax.set_ylim(param_min, param_max)

    ax.tick_params(axis='both', which='major', labelsize=8)

    return low1, median, high1


def plot_retrieved_parameters(axes_in, param_vals, plot_parameters, parameter_colour_list, 
                              retrieval_colour_list, retrieval_labels, span, truths, 
                              N_rows, N_columns, N_bins,
                              vertical_lines, vertical_lines_colors, 
                              tick_labelsize = 8, 
                              title_fontsize = 12, title_vert_spacing = 0.2,
                              custom_labels = [], custom_ticks = [],
                              alpha_hist = 0.4, show_title = True,
                              two_sigma_upper_limits_full = [], two_sigma_lower_limits_full = [],
                              orientation = 'vertical'
                              ):
    '''
    Plot retrieved parameters as histograms.

    Args:
        axes_in (list): 
            List of axes to plot on. If empty, new axes will be created.
        param_vals (list): 
            List of parameter values for each model.
        plot_parameters (list): 
            List of parameters to plot.
        parameter_colour_list (list): 
            List of colours for each parameter.
        retrieval_colour_list (list): 
            List of colours for each retrieval.
        retrieval_labels (list): 
            List of labels for each retrieval.
        span (list): 
            List of min and max values for each parameter.
        truths (list): 
            True values for each parameter.
        N_rows (int): 
            Number of rows in the plot grid.
        N_columns (int): 
            Number of columns in the plot grid.
        N_bins (list): 
            Number of bins for each histogram.
        vertical_lines (list): 
            Vertical lines to plot on the histograms.
        vertical_lines_colors (list): 
            Colours for the vertical lines.
        tick_labelsize (int, optional):
            Font size for tick labels. Default is 8.
        title_fontsize (int, optional):
            Font size for titles. Default is 12.
        title_vert_spacing (float, optional):
            Vertical spacing between titles. Default is 0.2.
        custom_labels (list, optional):
            Custom labels for the parameters. Default is empty list.
        custom_ticks (list, optional):
            Custom ticks for the parameters. Default is empty list.
        alpha_hist (float, optional):
            Alpha value for histogram bars. Default is 0.4.
        show_title (bool, optional):
            Whether to show titles on the plots. Default is True.
        two_sigma_upper_limits_full (1D or 2D list of str, optional):
            Upper limits for two sigma confidence intervals. Default is empty list.
        two_sigma_lower_limits_full (1D or 2D list of str, optional):
            Lower limits for two sigma confidence intervals. Default is empty list.

    Returns:
        fig (matplotlib figure object):
            The retrieved parameters plot.
    '''

    N_params = len(plot_parameters)
    N_models = len(param_vals)

    # If user doesn't specify number of rows or columns, place 3 histograms on each row
    if ((N_rows == None) or (N_columns == None)):
        N_columns = 3
        N_rows =  1 + (N_params - 1)//N_columns

    # Initialise multi-panel grid
    fig = plt.figure()

    gs = gridspec.GridSpec(N_rows, N_columns)

    fig.set_size_inches(2.5*N_columns, 2.5*N_rows)
    
    # Latex code for parameter labels

    if (len(custom_labels) == 0):
        param_labels = generate_latex_param_names(plot_parameters)
    else:
        param_labels = custom_labels

    # Determine histogram bounds (defaults to +/- 5σ)
    if (len(span) == 0):
        span = [0.999999426697 for q in range(N_params)]
    span = list(span)
    
    #***** Generate panels *****#
    
    # For each parameter
    for q in range(len(plot_parameters)):

        param = plot_parameters[q]
        param_label = param_labels[q]
        
        row_idx = q // N_columns
        column_idx = q - (row_idx * N_columns)

        if (len(axes_in) == 0):
            ax = plt.subplot(gs[row_idx, column_idx:column_idx+1])
        else:
            ax = axes_in[q]

        # Set number of significant figures for titles
        if ((('T' in param) or ('T_' in param)) and ('log' not in param)):
            title_fmt = '.0f'
        elif (param == 'a') or (param == 'b'):
            title_fmt = ".2f"
        elif ('delta_rel' in param):
            title_fmt = ".0f"
        elif (param == 'R_p_ref'):
            label_exponent = round_sig_figs(np.floor(np.log10(np.abs(0.5 * (qh - ql)))), 1)
            if label_exponent == -2.0:
                title_fmt = ".2f"
            elif label_exponent == -3.0:
                title_fmt = ".3f"
            elif label_exponent == -4.0:
                title_fmt = ".4f"
            else:
                title_fmt = ".2f"
        elif (param == 'd'):
            title_fmt = ".3f"
        else:
            title_fmt = '.2f'

        # Find the maximum x to set the y off of 
        x_max_array = []

        for m in range(N_models):
            
            param_vals_m = param_vals[m]
            
            if (N_models == 1):
                colour = parameter_colour_list[q]   # Each species has a different colour
            else:
                colour = retrieval_colour_list[m]   # Each retrieval has a different colour

            # Set minimum and maximum mixing ratio plot limits
            try:
                # If there is only one plot parameter, this doesn't work since the list isn't a list of lists 
                # i.e. if len (plot_parameters = 1) then span = (-5,-1) and if >2 ((-5,-1), (-5,-1)) etc
                if (len(plot_parameters) == 1):
                    try:
                        param_min, param_max = span[0], span[1]
                    except:
                        param_min, param_max = span[q]
                else:
                    param_min, param_max = span[q]
            
            # Lij: I'm not sure what this code does (why is there a try except here?) but I tried to fix 
            #      for len(plot_parameters) == 1
            except:
                if (len(plot_parameters) == 1):
                    quant = [0.5 - 0.5 * span, 0.5 + 0.5 * span]
                    span = _quantile(param_vals_m[:], quant)
                    param_min = span[0]
                    param_max = span[1]
                else:
                    quant = [0.5 - 0.5 * span[q], 0.5 + 0.5 * span[q]]
                    span[q] = _quantile(param_vals_m[:,q], quant)
                    param_min = span[q][0]
                    param_max = span[q][1]

            x,w,patches = ax.hist(param_vals_m[:,q], bins=N_bins[q], color=colour, histtype='stepfilled', 
                                  alpha=0.0, edgecolor='None', density=True, stacked=True,
                                  orientation = 'vertical')
            
            x_max_array.append(x.max())

        # For each retrieval
        for m in range(N_models):

            if (N_models == 1):
                title_colour = 'black'
                constraint_colour = 'dimgray'
            else:
                title_colour = retrieval_colour_list[m]
                constraint_colour = retrieval_colour_list[m]

            param_vals_m = param_vals[m]
            
            if (N_models == 1):
                colour = parameter_colour_list[q]   # Each species has a different colour
            else:
                colour = retrieval_colour_list[m]   # Each retrieval has a different colour

            # Set minimum and maximum mixing ratio plot limits
            try:
                param_min, param_max = span[q]
            except:
                quant = [0.5 - 0.5 * span[q], 0.5 + 0.5 * span[q]]
                span[q] = _quantile(param_vals_m[:,q], quant)
                param_min = span[q][0]
                param_max = span[q][1]

            # Plot histogram
            low1, median, high1 = plot_parameter_panel(ax, param_vals_m[:,q], N_bins[q],
                                                       param_min, param_max, colour, x_max_array = x_max_array,
                                                       alpha_hist = alpha_hist, orientation = orientation)

            # Add retrieval model labels to top left panel
            if ((row_idx == 0) and (column_idx == 0) and (len(retrieval_labels) != 0)):
                ax.text(0.10, (0.94 - m*0.10), retrieval_labels[m], color=colour, 
                        fontsize = 10, horizontalalignment='left', 
                        verticalalignment='top', transform=ax.transAxes)
                
            # Plot retrieved parameter value as title
            if (show_title == True):
                title = None

                fmt = "{{0:{0}}}".format(title_fmt).format

                # Plot one sigma limits by default
                if ((len(two_sigma_upper_limits_full) == 0) and (len(two_sigma_lower_limits_full) == 0)):
                                    
                    # Add title
                    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                    title = title.format(fmt(median), fmt((median-low1)), fmt((high1-median)))
                    title = "{0} = {1}".format(param_label, title)
                  #  title = "{0}".format(title)

                  #  ax.set_title(title, fontsize = title_fontsize)

                    # Plot median and +/- 1σ confidence intervals
                    if orientation == 'vertical':
                        ax.axvline(median, lw=2, ls="-", alpha=0.7, color=constraint_colour)
                        ax.axvline(low1, lw=1, ls="dashed", color=constraint_colour)
                        ax.axvline(high1, lw=1, ls="dashed", color=constraint_colour)
                    else:
                        ax.axhline(median, lw=2, ls="-", alpha=0.7, color=parameter_colour_list[q])
                        ax.axhline(low1, lw=1, ls="dashed", color=parameter_colour_list[q])
                        ax.axhline(high1, lw=1, ls="dashed", color=parameter_colour_list[q])  

                # Title has 2 sigma upper/lower limits where user flags the given parameter
                else:
                    
                    # If you have multiple models and want them to have different 
                    # options (i.e one model is upper limit, one is lower limit)
                    # this just checks for that and pulls out the 
                    # model options in the loop
                    # otherwise it uses the 1D array for all the models

                    is_list_of_lists_upper = all(isinstance(item, list) for item in two_sigma_upper_limits_full)
                    is_list_of_lists_lower = all(isinstance(item, list) for item in two_sigma_lower_limits_full)

                    if (is_list_of_lists_upper == True) and (len(two_sigma_upper_limits_full) != 0):
                        two_sigma_upper_limits = two_sigma_upper_limits_full[m]
                    else:
                        two_sigma_upper_limits = two_sigma_upper_limits_full

                    if (is_list_of_lists_lower == True) and (len(two_sigma_lower_limits_full) != 0):
                        two_sigma_lower_limits = two_sigma_lower_limits_full[m]
                    else:
                        two_sigma_lower_limits = two_sigma_lower_limits_full
                    
                    if (param in two_sigma_upper_limits):

                        # Find 95th percentile
                        qh = _quantile(param_vals_m[:,q], [0.95])[0]

                        # Add title
                        title = r"${{{0}}}$"
                        title = title.format(fmt(qh))
                        title = "{0} < {1}".format(param_label, title)

                        # Plot arrow for upper limit
                        ax.axvline(qh, lw=2, ls="-", color=constraint_colour, alpha=0.8)
                        ax.annotate('', xy=(qh, (0.9 - 0.1 * m)),
                                    xytext=((qh - (0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]))), (0.9 - 0.1 * m)), 
                                    xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                                    arrowprops=dict(facecolor=constraint_colour, color=constraint_colour, 
                                                    edgecolor=constraint_colour, arrowstyle='<|-', 
                                                    lw=2, ls='-', shrinkA=0, shrinkB=0),
                                    alpha=0.8)

                    elif (param in two_sigma_lower_limits):

                        # Find 5th percentile
                        ql = _quantile(param_vals_m[:,q], [0.05])[0]

                        # Add title
                        title = r"${{{0}}}$"
                        title = title.format(fmt(ql))
                        title = "{0} > {1}".format(param_label, title)

                        # Plot arrow for lower limit
                        ax.axvline(ql, lw=2, ls="-", color=constraint_colour, alpha=0.8)
                        ax.annotate('', xy=(ql + (0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0])), (0.9 - 0.1 * m)), 
                                    xytext=(ql, (0.9 - 0.1 * m)), 
                                    xycoords=('data', 'axes fraction'), textcoords=('data', 'axes fraction'),
                                    arrowprops=dict(facecolor=constraint_colour, color=constraint_colour, 
                                                    edgecolor=constraint_colour, arrowstyle='-|>', 
                                                    lw=2, ls='-', shrinkA=0, shrinkB=0),
                                    alpha=0.8)

                    else:

                        # Add title
                        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                        title = title.format(fmt(median), fmt((median-low1)), fmt((high1-median)))
                        title = "{0} = {1}".format(param_label, title)

                        # Plot median and +/- 1σ confidence intervals
                        if orientation == 'vertical':
                            ax.axvline(median, lw=2, ls="-", alpha=0.7, color=constraint_colour)
                            ax.axvline(low1, lw=1, ls="dashed", color=constraint_colour)
                            ax.axvline(high1, lw=1, ls="dashed", color=constraint_colour)
                        else:
                            ax.axhline(median, lw=2, ls="-", alpha=0.7, color=constraint_colour)
                            ax.axhline(low1, lw=1, ls="dashed", color=constraint_colour)
                            ax.axhline(high1, lw=1, ls="dashed", color=constraint_colour)                     

                # Plot title
                
                # I prefer it flipped so that its in the order as plot_retrieved_spectra (EM)
                top_y = 1.05 + ((N_models-1)*0.2)

                #ax.text(0.5, 1.05 + (m * 0.2),
                #        title, horizontalalignment = "center", verticalalignment = "bottom",
                #        color = title_colour, transform = ax.transAxes, fontsize = title_fontsize,
                #       )
                
                if orientation == 'vertical':
                    ax.text(0.5, top_y - (m * 0.2),
                            title, horizontalalignment = "center", verticalalignment = "bottom",
                            color = title_colour, transform = ax.transAxes, fontsize = title_fontsize,
                        )

            else:

                # Add param name to x-axis label instead
                ax.set_xlabel(param_label, fontsize = title_fontsize)

            ax.set_yticks([])
            ax.tick_params(axis='both', which='major', labelsize = tick_labelsize)
            
            if('log_r_m' in param):
                xmajorLocator = MultipleLocator(1)
                xminorLocator = MultipleLocator(0.5)
                ax.xaxis.set_major_locator(xmajorLocator)
                ax.xaxis.set_minor_locator(xminorLocator)
            
            # Better axis label spacing for temperatures
            if ('T' in param):
                if ((param_max - param_min) < 100):
                    xmajor_interval = 20
                    xminor_interval = 10
                elif (((param_max - param_min) >= 100) and ((param_max - param_min) < 200)):
                    xmajor_interval = 50
                    xminor_interval = 10
                elif (((param_max - param_min) >= 200) and ((param_max - param_min) < 400)):
                    xmajor_interval = 100
                    xminor_interval = 50
                elif (((param_max - param_min) >= 400) and ((param_max - param_min) < 800)):
                    xmajor_interval = 200
                    xminor_interval = 100
                elif (((param_max - param_min) >= 800) and ((param_max - param_min) < 2000)):
                    xmajor_interval = 400
                    xminor_interval = 200
                elif ((param_max - param_min) >= 2000):
                    xmajor_interval = 1000
                    xminor_interval = 500

                xmajorLocator = MultipleLocator(xmajor_interval)
                xminorLocator = MultipleLocator(xminor_interval)
                ax.xaxis.set_major_locator(xmajorLocator)
                ax.xaxis.set_minor_locator(xminorLocator)

            # Better axis label spacing for mixing ratios
            if ('log_' in param):
                if ((param_max - param_min) <= 0.5):
                    xmajor_interval = 0.1
                    xminor_interval = 0.02
                elif ((param_max - param_min) <= 1.0):
                    xmajor_interval = 0.2
                    xminor_interval = 0.05
                elif (((param_max - param_min) > 1.0) and ((param_max - param_min) <= 2.0)):
                    xmajor_interval = 0.5
                    xminor_interval = 0.1
                elif (((param_max - param_min) > 2.0) and ((param_max - param_min) <= 4.0)):
                    xmajor_interval = 1.0
                    xminor_interval = 0.2
                elif (((param_max - param_min) > 4.0) and ((param_max - param_min) <= 8.0)):
                    xmajor_interval = 2.0
                    xminor_interval = 0.5
                elif ((param_max - param_min) >= 8.0):
                    xmajor_interval = 4.0
                    xminor_interval = 1.0

                xmajorLocator = MultipleLocator(xmajor_interval)
                xminorLocator = MultipleLocator(xminor_interval)
                ax.xaxis.set_major_locator(xmajorLocator)
                ax.xaxis.set_minor_locator(xminorLocator)

          #  if ('log_X' in param) and ('base' not in param):
          #      xmajorLocator = MultipleLocator(5)
          #      xminorLocator = MultipleLocator(2.5)
          #      ax.xaxis.set_major_locator(xmajorLocator)
          #      ax.xaxis.set_minor_locator(xminorLocator)

            if (len(custom_ticks) != 0):
                xmajorLocator = MultipleLocator(custom_ticks[q][0])
                xminorLocator = MultipleLocator(custom_ticks[q][1])
                ax.xaxis.set_major_locator(xmajorLocator)
                ax.xaxis.set_minor_locator(xminorLocator)

        # Overplot true value
        if (len(truths) != 0):
            ax.axvline(x=truths[q], linewidth=1.5, linestyle='-', color='crimson', alpha=0.8)

        if (len(vertical_lines) != 0):
            for n in range(len(vertical_lines)):
                ax.axvline(x=vertical_lines[n][q], linewidth=1.5, linestyle='-', color=vertical_lines_colors[n], alpha=0.8)

    return fig

  
def elemental_ratio_samples(all_species, X_vals, element_1, element_2):
    '''
    Helper function to calculate the abundance ratio between any two elements 
    in the atmosphere.
    
    Example: to compute the C/O ratio, use element_1 = 'C' and element_2 = 'O'.

    Args:
        all_species (np.array of str):
            List of all chemical species included in the model.
        X_vals (2D np.array of float):
            Mixing ratio samples.
        element_1 (str):
            First element in ratio.
        element_2 (str):
            First element in ratio.

    Returns:
        element_ratio (np.array of float):
            Abundance ratio samples.
    
    '''

    # Store shape of mixing ratio array
    N_samples, N_species = np.shape(X_vals)

    # Initialise element ratio array
    element_ratio = np.zeros(shape=(N_samples))

    # Loop through atmosphere
    for i in range(N_samples):

        element_1_abundance = 0.0   # First element in ratio
        element_2_abundance = 0.0   # Second element in ratio

        for q in range(N_species): 
            
            # Extract name and mixing ratio of molecule 'q'
            molecule_q = all_species[q] 
            X_q = X_vals[i,q]

            # Count how many atoms of each element are in this molecule
            counts = count_atoms(molecule_q)

            # Loop over elements
            for element, count in counts.items():

                # Add abundances of element 1 and 2 to the total
                if (element == element_1):
                    element_1_abundance += count * X_q
                elif (element == element_2):
                    element_2_abundance += count * X_q

        # Compute the element ratio
        element_ratio[i] = element_1_abundance / element_2_abundance

    return element_ratio


def plot_histograms(planet, models, plot_parameters,
                    parameter_colour_list = [], retrieval_colour_list = [], 
                    retrieval_labels = [], span = [], truths = [], N_bins = [], 
                    N_rows = None, N_columns = None, axes = [], 
                    retrieval_codes = [], external_samples = [],
                    external_param_names = [], plt_label = None, 
                    save_fig = True, show_title = True,
                    vertical_lines = [], vertical_line_colors = [],
                    tick_labelsize = None, 
                    title_fontsize = None, title_vert_spacing = None,
                    custom_labels = [], custom_ticks = [],
                    alpha_hist = 0.4, 
                    two_sigma_upper_limits = [], two_sigma_lower_limits = [],
                    orientation = 'vertical'):
    '''
    Plot a set of histograms from one or more retrievals.

    Args:
        planet (dict):
            Dictionary containing the planet properties.
        models (list of dicts):
            List of dictionaries containing the model properties.
        plot_parameters (list of str):
            List of parameters to plot.
        parameter_colour_list (list of str, optional):
            List of colours for each parameter.
        retrieval_colour_list (list of str, optional):
            List of colours for each retrieval model.
        retrieval_labels (list of str, optional):
            List of labels for each retrieval model.
        span (list of float, optional):
            Span for each parameter to plot.
        truths (list of float, optional):
            True values for each parameter.
        N_bins (list of int, optional):
            Number of bins for each histogram.
        N_rows (int, optional):
            Number of rows in the figure. Default is None.
        N_columns (int, optional):
            Number of columns in the figure. Default is None.
        axes (list of matplotlib axes, optional):
            List of axes to plot on. Default is empty list.
        retrieval_codes (list of str, optional):
            List of retrieval codes for each model. Default is empty list.
        external_samples (list of np.array, optional):
            List of external samples for each model. Default is empty list.
        external_param_names (list of list of str, optional):
            List of external parameter names for each model. Default is empty list.
        plt_label (str, optional):
            Label for the plot file name. Default is None.
        save_fig (bool, optional):
            Whether to save the figure or not. Default is True.
        show_title (bool, optional):
            Whether to show the title or not. Default is True.
        vertical_lines (list of float, optional):
            List of vertical lines to plot. Default is empty list.
        vertical_line_colors (list of str, optional):
            List of colors for vertical lines. Default is empty list.
        tick_labelsize (int, optional):
            Font size for tick labels. If None and axes provided, will auto-scale 
            based on figure size. Default is None.
        title_fontsize (int, optional):
            Font size for titles. If None and axes provided, will auto-scale 
            based on figure size. Default is None.
        title_vert_spacing (float, optional):
            Vertical spacing between titles. If None and axes provided, will auto-scale 
            based on title font size. Default is None.
        custom_labels (list of str, optional):
            Custom labels for the parameters. Default is empty list.
        custom_ticks (list of list of float, optional):
            Custom ticks for the x-axis. Default is empty list.
        alpha_hist (float, optional):
            Transparency for the histograms. Default is 0.4.
        two_sigma_upper_limits (1D or 2D list of str, optional):
            List of parameters with two sigma upper limits. Default is empty list.
            If 1D, will apply two_sigma_upper_limit to all models. If 2D, will 
            only do it for specific models. 
        two_sigma_lower_limits (1D or 2D list of str, optional):
            List of parameters with two sigma lower limits. Default is empty list.
            If 1D, will apply two_sigma_lower_limit to all models. If 2D, will 
            only do it for specific models. 

    '''

    N_models = len(models)
    N_params_to_plot = len(plot_parameters)

    # Check user provided settings are valid
    if (N_models > 10):
        raise Exception("Max supported number of retrieval models is 10.")
    if (N_models == 1) and (len(parameter_colour_list) == 0):
        parameter_colour_list = ['darkblue', 'darkgreen', 'orangered', 'magenta',
                                 'saddlebrown', 'grey', 'brown']
    elif (N_models == 1) and (len(parameter_colour_list) != 0):
        if (len(plot_parameters) != 0):
            if (len(parameter_colour_list) != len(plot_parameters)):
                raise Exception("Number of parameter colours does not match the " + 
                                "requested number of parameters to plot.")
    elif (N_models >= 2) and (len(retrieval_colour_list) == 0):
        retrieval_colour_list = ['purple', 'dodgerblue', 'forestgreen']
    elif (N_models >= 2) and (len(retrieval_colour_list) != 0):
        if (len(retrieval_colour_list) != N_models):
            raise Exception("Number of retrieval colours does not match the " +
                            "number of retrieval models.")
    if (len(two_sigma_lower_limits) != 0) and (len(two_sigma_lower_limits) != 0):
        for param in two_sigma_upper_limits:
            if (param in two_sigma_lower_limits):
                raise Exception("Cannot have both a two sigma lower and upper limit for a given parameter.")

    
    param_vals = []    # List to store parameter values for all models, samples, and parameters

    # For each retrieval
    for m in range(N_models):

        model = models[m]

        if ((len(retrieval_codes) == 0) or (retrieval_codes[m] == 'POSEIDON')):

            # Unpack model and atmospheric properties
            planet_name = planet['planet_name']
            model_name = model['model_name']
            chemical_species = model['chemical_species']
            param_species = model['param_species']
            bulk_species = model['bulk_species']
            X_param_names = model['X_param_names']
            Atmosphere_dimension = model['Atmosphere_dimension']
            N_params_cum = model['N_params_cum']
            disable_atmosphere = model['disable_atmosphere']
            mass_unit = model['mass_unit']
            N_species = len(chemical_species)
            
            # Unpack number of free parameters
            param_names = model['param_names']

            # Identify output directory location
            output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'
                
            # Identify directory location where the plot will be saved
            plot_dir = './POSEIDON_output/' + planet_name + '/plots/'

            # Load relevant output directory
            output_prefix = model_name + '-'

            # Change directory into MultiNest result file folder
            cwd = os.getcwd()
            os.chdir(output_dir + 'MultiNest_raw/')
            
            # Run PyMultiNest analyser to extract posterior samples
            analyzer = pymultinest.Analyzer(len(param_names), outputfiles_basename = output_prefix,
                                            verbose = False)
            samples = analyzer.get_equal_weighted_posterior()[:,:-1]

            # Change directory back to directory where user's python script is located
            os.chdir(cwd)
            #os.chdir('../../../../')

            # Find total number of available posterior samples from MultiNest 
            N_samples = len(samples[:,0])

            if (Atmosphere_dimension > 1):
                print("Note: this function is not currently configured for bulk gas " +
                      "mixing ratios or element ratios for multidimensional retrievals")

            # Create array to store the composition of the atmosphere  
            X_stored = np.zeros(shape=(N_samples, N_species))
            mu_stored = np.zeros(shape=(N_samples))
            
            if (disable_atmosphere == False):
                
                # Only generates atmospheres, which is very slow, if its
                # mu, mmw, or a elemental ratio
                if ('mu' in plot_parameters) or ('mmw' in plot_parameters) or ('/' in str(plot_parameters)):
                        
                    # Load mixing ratios and mean molecular weight samples
                    for i in range(N_samples):

                        atmosphere_i = get_retrieved_atmosphere(planet, model, np.logspace(np.log10(100.0), np.log10(1e-6), 100),
                                                                specific_param_values = samples[i], R_p_ref_set=planet['planet_radius'])
                        
                        X_stored[i,:] = atmosphere_i['X'][:,0,0,0]
                        mu_stored[i] = atmosphere_i['mu'][0,0,0]/sc.u

        # Or load samples in directly from external code
        else:
            param_names = np.array(external_param_names[m])
            samples = external_samples[m]
            N_samples = len(samples[:,0])

        # Create array to store parameter values for model m
        param_samples_m = np.zeros(shape=(N_samples, N_params_to_plot))
        
        for q in range(N_params_to_plot):

            param = plot_parameters[q]

            # QUICK FIX DO NOT KEEP THIS
            # This is to compare old Grant et al 2023 results to new results
            # I changed the parameter name
            if param == 'log_P_top_slab_SiO2':
                try:
                    param_samples_m[:,q] = samples[:,np.where(param_names == param)[0][0]]
                except:
                    param = 'log_P_cloud_SiO2'
                    param_samples_m[:,q] = samples[:,np.where(param_names == param)[0][0]]

            else:

                # For parameters in retrieval model, load from the samples
                if (param in param_names):
                    param_samples_m[:,q] = samples[:,np.where(param_names == param)[0][0]]
                
                # Derived mean molecular weight samples
                elif (param in ['mu', 'mmw']):
                    if (disable_atmosphere == True):
                        raise Exception("Error: can't plot the mean molecular weight " +
                                        "for a model with no atmosphere!")
                    else:
                        param_samples_m[:,q] = mu_stored

                # Elemental ratios
                elif ('/' in param):
                    if (disable_atmosphere == True):
                        raise Exception("Error: can't plot an elemental ratio for " +
                                        "a model with no atmosphere!")
                    else:
                        if ('log_' in param):
                            ratio = param.split('log_')[1]
                        else:
                            ratio = param
                        elements = ratio.split('/')   # Split into constituent elements
                        element_1, element_2 = elements

                        # For metallicity, sum the abundances of elements heavier than He
                        if (ratio == 'M/H') or (ratio == 'log_M/H'):
                            numerator = np.zeros(N_samples)
                            denominator = 0.0

                            # Automatically detect which elements are present in the atmosphere
                            elements_in_atmosphere = set()
                            
                            # Scan all chemical species to find which elements are present
                            for species in chemical_species:
                                counts = count_atoms(species)
                                for element in counts.keys():
                                    # Only include metals (exclude H and He)
                                    if element not in ['H', 'He']:
                                        elements_in_atmosphere.add(element)
                            
                            # Convert to sorted list for consistent ordering
                            elements_in_atmosphere = sorted(list(elements_in_atmosphere))
                            
                            # Only include elements that have solar abundance data
                            available_elements = []
                            for element_i in elements_in_atmosphere:
                                if element_i in solar_abundances:
                                    available_elements.append(element_i)
                            
                            # Calculate metallicity using detected elements
                            for element_i in available_elements:

                                # Sum elemental ratios for the atmosphere
                                numerator += elemental_ratio_samples(chemical_species, X_stored, 
                                                                     element_i, element_2)

                                # Sum solar abundances for normalisation
                                denominator += 10**(solar_abundances[element_i]-12.0)

                            # Divide summed atmospheric abundances by summed solar abundances for the metallicity
                            element_ratio_norm = numerator / denominator

                        # Other elemental ratios
                        else:
                            element_ratio = elemental_ratio_samples(chemical_species, X_stored, 
                                                                    element_1, element_2)
                            if (element_2 == 'H'):
                                element_ratio_norm = element_ratio / 10**(solar_abundances[element_1]-12.0)
                            else:
                                element_ratio_norm = element_ratio

                        # Plot either the log of the elemental ratio or the ratio itself
                        if ('log_' in param): 
                            param_samples_m[:,q] = np.log10(element_ratio_norm)
                        else:
                            param_samples_m[:,q] = element_ratio_norm

                # Filler gas
                else:
                    if ('log_' in param):
                        filler_species = param.split('log_')[1]
                        if (filler_species == 'H2+He'):
                            param_samples_m[:,q] = np.log10(X_stored[:,0])
                        elif (filler_species == 'H2'):
                            param_samples_m[:,q] = np.log10(X_stored[:,0])
                        elif (filler_species == 'He'):
                            param_samples_m[:,q] = np.log10(X_stored[:,1])
                        else:
                            param_samples_m[:,q] = np.log10(X_stored[:,0])

        param_vals.append(param_samples_m)

    # Auto-scale font sizes based on figure size if user provided axes but no explicit font settings
    if (len(axes) > 0 and (title_fontsize is None or tick_labelsize is None or title_vert_spacing is None)):
        
        # Get the figure from the first axis
        fig_for_scaling = axes[0].get_figure()
        fig_width, fig_height = fig_for_scaling.get_size_inches()
        
        # Get actual subplot dimensions in inches
        bbox = axes[0].get_position()  # Get position in figure coordinates (0-1)
        subplot_height_inches = bbox.height * fig_height
        subplot_width_inches = bbox.width * fig_width
        
        # Calculate font sizes based on actual subplot height
        # Allocate space: ~15% for title, ~15% for x-axis labels, 60% for histogram
        title_space_inches = subplot_height_inches * 0.15
        xaxis_space_inches = subplot_height_inches * 0.15
        histogram_space_inches = subplot_height_inches * 0.60
        
        if (title_fontsize is None):

            # Scale title font based on available title space
            title_fontsize = max(6, min(14, int(title_space_inches * 72 * 0.5)))
            
        if (tick_labelsize is None):
        
            # Scale tick labels based on available x-axis space
            tick_labelsize = max(7, min(14, int(xaxis_space_inches * 72 * 0.6)))
        
        # Set title_vert_spacing to be very small for constrained layouts
        if (title_vert_spacing is None):
            title_vert_spacing = 0.12 
    
    # Set default values if still None and no axes provided
    if title_fontsize is None:
        title_fontsize = 12
    if tick_labelsize is None:
        tick_labelsize = 8
    if title_vert_spacing is None:
        title_vert_spacing = 0.2

    fig = plot_retrieved_parameters(axes, param_vals, plot_parameters, 
                                    parameter_colour_list, retrieval_colour_list, 
                                    retrieval_labels, span, truths, 
                                    N_rows, N_columns, N_bins,
                                    vertical_lines, vertical_line_colors, 
                                    tick_labelsize = tick_labelsize, 
                                    title_fontsize = title_fontsize,
                                    title_vert_spacing = title_vert_spacing,
                                    custom_labels = custom_labels,
                                    custom_ticks = custom_ticks,
                                    alpha_hist = alpha_hist,
                                    show_title = show_title,
                                    two_sigma_upper_limits_full = two_sigma_upper_limits,
                                    two_sigma_lower_limits_full = two_sigma_lower_limits,
                                    orientation = orientation
                                    )
    
    # Save figure to file
    if (save_fig == True):
        if (plt_label == None):
            file_name = (plot_dir + planet_name + '_histograms.png')
        else:
            file_name = (plot_dir + planet_name + '_' + plt_label + '_histograms.png')

        fig.savefig(file_name, bbox_inches='tight', dpi=800)

    return fig


def vary_one_parameter_PT(model, planet, param_name, vary_list,
                          P, P_ref, R_p_ref, PT_params_og, 
                          log_X_params_og, cloud_params_og,
                          ax = None,legend_location = 'upper right'):
    '''
    This function is used in the tutorial notebooks to show how turning a knob 
    on a parameter changes the resulting PT profile.

    Args:
        model (dict):
            A specific description of a given POSEIDON model.
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        param_name (string):
            Name of the parameter to vary
        vary_list (array of float):
            Array containing values to test
        P (np.array of float):
            Model pressure grid (bar).
        P_ref (float):
            Reference pressure (bar).
        R_p_ref (float):
            Planet radius corresponding to reference pressure (m).
        PT_params_og (np.array of float):
            Original parameters defining the pressure-temperature field.
        log_X_params_og (np.array of float):
            Original parameters defining the log-mixing ratio field.
        cloud_params_og (np.array of float):
            Original parameters defining atmospheric aerosols.
        legend_location (str, optional):
            The location of the legend ('upper left', 'upper right', 
            'lower left', 'lower right').
        ax (matplotlib axis object, optional):
            Matplotlib axis provided externally.

    Returns: 
        Outputs a plot of resultant spectra with the param_name at the vary_list values.

    '''

    from POSEIDON.core import define_model
    from POSEIDON.core import make_atmosphere
    from POSEIDON.core import compute_spectrum
    from POSEIDON.visuals import plot_spectra
    from POSEIDON.utility import plot_collection
    import matplotlib.pyplot as plt

    spectra_array = []
    spectra_labels = []

    colour_list = ['red','orange','yellow','green','blue','purple']

    # create figure
    fig = plt.figure()

    if (ax == None):
        ax = plt.gca()
    else:
        ax = ax

    # Real spectrum 
    model_name = 'Vary-One-Thing'
    bulk_species = ['H2','He']
    species_list = model['param_species']
    param_species = species_list

    if model['cloud_model'] != 'Mie':

        model = define_model(model_name,bulk_species,param_species,
                                PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                                cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                                cloud_dim = model['cloud_dim'])

    else:
        aerosol_species = model['aerosol_species']

        model = define_model(model_name,bulk_species,param_species,
                        PT_profile = model['PT_profile'], X_profile = model['X_profile'],
                        cloud_model = model['cloud_model'], cloud_type = model['cloud_type'],
                        cloud_dim = model['cloud_dim'],
                        aerosol_species = aerosol_species, 
                        scattering = model['scattering'],
                        reflection = model['reflection'])


    index = np.argwhere(model['PT_param_names'] == param_name)[0][0]

    for i in range(len(vary_list)):

        PT_params = np.copy(PT_params_og)
        PT_params[index] = vary_list[i]

        PT_label = param_name + ' = ' + str(vary_list[i])
        
        atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, log_X_params_og, cloud_params_og)

        P = atmosphere['P']
        T = atmosphere['T']

        ax.semilogy(T[:,0,0], P, lw=1.5, color = colour_list[i], label = PT_label)
   
    #Common plot settings for all profiles
    ax.invert_yaxis()            
    ax.set_xlabel(r'Temperature (K)', fontsize = 16)
    #ax.set_xlim(T_min, T_max)
    ax.set_ylabel(r'Pressure (bar)', fontsize = 16)
    #ax.set_ylim(np.power(10.0, log_P_max), np.power(10.0, log_P_min))  
    ax.tick_params(labelsize=12)
    
    # Add legend
    legend = ax.legend(loc=legend_location, shadow=True, prop={'size':10}, ncol=1, 
                       frameon=False, columnspacing=1.0)
    
    fig.set_size_inches(9.0, 9.0)

def vary_one_parameter(model, planet, star, atmosphere, opac, wl, param_name, vary_list, spectrum_type = 'transmission', 
                    y_unit = 'transit_depth', **plot_kwargs):
    
    '''
    This function is utilized in tutorial noteooks to show how turning a knob on a parameter changes a resultant spectrum

    Args:
        model (dict):
            A specific description of a given POSEIDON model.
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        star (dict):
            Collection of stellar properties used by POSEIDON.
        atmosphere (dict):
            Collection of atmospheric properties used by POSEIDON.
        opac (dict):
            Collection of cross sections and other opacity sources.
        wl (np.array of float):
            Model wavelength grid (μm).
        param_name (string):
            Name of the parameter to vary
        vary_list (array of float):
            Array containing values to test
        spectrum_type (str):
            The type of spectrum for POSEIDON to compute
            (Options: transmission / emission / direct_emission / 
                    transmission_time_average).
        y_unit (str):
            The unit of the y-axis
            (Options: 'transit_depth', 'eclipse_depth', '(Rp/Rs)^2', 
            '(Rp/R*)^2', 'Fp/Fs', 'Fp/F*', 'Fp').
        

    Returns: 
        Outputs a plot of resultant spectra with the param_name at the vary_list values.

    '''
    from POSEIDON.core import compute_spectrum
    from POSEIDON.utility import _update_model, _update_atmosphere

    # Array that holds each spectrum and their label
    spectra_array = []
    spectra_labels = []

    assert param_name in model['param_names'], f"{param_name} is not in the param list. Check model['param_names']"

    for i in range(len(vary_list)):

        if param_name in planet.keys():
            planet[param_name] = vary_list[i]
        elif param_name in star.keys():
            star[param_name] = vary_list[i]
        elif param_name in model.keys():
            model[param_name] = vary_list[i]
        elif param_name in model['cloud_param_names']:
            cloud_index = np.argwhere(model['cloud_param_names'] == param_name)[0][0]
            atmosphere['cloud_params'][cloud_index] = vary_list[i]
        elif param_name in model['PT_param_names']:
            PT_index = np.argwhere(model['PT_param_names'] == param_name)[0][0]
            atmosphere['PT_params'][PT_index] = vary_list[i]
        elif param_name in model['X_param_names']:
            X_index = np.argwhere(model['X_param_names'] == param_name)[0][0]
            atmosphere['log_X_params'][X_index] = vary_list[i]
        elif param_name in model['surface_param_names']:
            surface_index = np.argwhere(model['surface_param_names'] == param_name)[0][0]
            atmosphere['surface_params'][surface_index] = vary_list[i]
        elif param_name in model['geometry_param_names']:
            geometry_index = np.argwhere(model['geometry_param_names'] == param_name)[0][0]
            atmosphere['geometry_params'][geometry_index] = vary_list[i]
        elif param_name in atmosphere.keys():
            atmosphere[param_name] = vary_list[i]
        else:
            raise Exception(f"{param_name} not found in planet, star, model or atmosphere dictionaries.")
        
        model = _update_model(model)
        atmosphere = _update_atmosphere(planet, model, atmosphere)
    
        spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                    spectrum_type = spectrum_type)
            
        spectra_array.append(spectrum)
        label = param_name + ' = ' + str(vary_list[i])
        spectra_labels.append(label)


    # Plot
    for s in range(len(spectra_array)):
        if s == 0:
            spectra = plot_collection(spectra_array[s], wl, collection = [])
        else:
            spectra = plot_collection(spectra_array[s], wl, collection = spectra)

    label = 'Varying ' + param_name
    
    fig = plot_spectra(spectra, planet, R_to_bin = 100,
                       plt_label = label,
                       spectra_labels = spectra_labels,
                       plot_full_res = False,
                       save_fig = False,
                       y_unit = y_unit,
                       **plot_kwargs)