'''
Plotting routines to visualise POSEIDON output.

'''

from enum import unique
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
                              ScalarFormatter, NullFormatter
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
                     confidence_intervals
from .instrument import bin_spectrum_to_data
from .parameters import split_params
              
# Define some more flexible linestyles for convenience
linestyles = {
              'loosely dotted':        (0, (1, 10)),
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


def plot_transit(ax, R_p, r, T, phi, phi_edge, dphi, theta, theta_edge, dtheta, 
                 perspective, plot_labels = True):
    '''
    
    '''

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
   #     star = Circle((-y_p/R_p, -b_p/R_p), R_s/R_p, facecolor='gold', edgecolor='None', alpha=0.8)
   #     ax.add_artist(star)
        
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
            planet_core = Circle((0.0, 0.0), r[0,0,0]/R_p, facecolor='black', edgecolor='None')
            ax.add_artist(planet_core)
        
        ax.set_title("Terminator Plane", fontsize = 16, pad=10)
        
        ax.set_xlabel(r'y ($R_p$)', fontsize = 16)
        ax.set_ylabel(r'z ($R_p$)', fontsize = 16)
        
        # Plot atmosphere segment collection
        p = PatchCollection(patches, cmap=matplotlib.cm.RdYlBu_r, alpha=1.0, 
                            edgecolor=colorConverter.to_rgba('black', alpha=0.4), 
                            lw=0.1, zorder = 10)

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
            ax.text(0.04, 0.90, 'Evening', horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)
            ax.text(0.96, 0.90, 'Morning', horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes, fontsize = 14)

        ax.set_xlim([-1.4*r_max, 1.4*r_max])
        ax.set_ylim([-1.4*r_max, 1.4*r_max])
  
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
                            edgecolor=colorConverter.to_rgba('black', alpha=0.4), 
                            lw=0.1, zorder = 10)
        
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
    to a slice through the terminator plane, whilst the right panel is a
    slice through the north pole - observer plane.

    Args:
        planet (dict of str: various):
            Collection of planetary properties used by POSEIDON.
        star (dict of str: various):
            Collection of stellar properties used by POSEIDON.
        model (dict of str: various):
            A specific description of a given POSEIDON model.
        atmosphere (dict of str: various):
            Collection of atmospheric properties.
        plot_labels (bool)
            If False, removes text labels from the plot.

    Returns:
        fig (matplotlib figure object)

    '''


    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']
    R_p = planet['planet_radius']
  #  R_s = star['stellar_radius']
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
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # Plot terminator plane on LHS axis
    p = plot_transit(ax1, R_p, r, T, phi, phi_edge, dphi, theta, 
                     theta_edge, dtheta, 'terminator', plot_labels) 

    # Plot side perspective on RHS axis
    _ = plot_transit(ax2, R_p, r, T, phi, phi_edge, dphi, theta, 
                     theta_edge, dtheta, 'day-night', plot_labels) 
    
    # Plot temperature colourbar
    cbaxes = fig.add_axes([1.01, 0.131, 0.015, 0.786]) 
    cb = plt.colorbar(p, cax = cbaxes)  
    tick_locator = ticker.MaxNLocator(nbins=8)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.formatter.set_useOffset(False)
    cb.ax.set_title(r'$T \, \, \rm{(K)}$', horizontalalignment='left', pad=10)
    
    plt.tight_layout()

    # Write figure to file
    file_name = output_dir + planet_name + '_' + model_name + '_Geometry.png'

    plt.savefig(file_name, bbox_inches='tight', dpi=800)

    return fig


def plot_PT(planet, model, atmosphere, show_profiles = [],
            PT_label = None, log_P_min = None, log_P_max = None, T_min = None,
            T_max = None, colour = 'darkblue', legend_location = 'lower left'):
    ''' 
    Plot the pressure-temperature (P-T) profiles defining the atmosphere.
    
    For a 1D model, a single P-T profile is plotted. For 2D or 3D models,
    the user can specify the regions for which the P-T profiles should be
    plotted. This is handled through 'show_profiles'.
    
    Valid choices for 2D and 3D models:
        
    * 2D Day-Night:
        -> show_profiles = ['day', 'night', 'terminator']
        
    * 2D Evening-Morning:
        -> show_profiles = ['morning', 'evening', 'average']
        
    * 3D:
        -> show_profiles = ['evening-day', 'evening-night', 'evening-terminator', 
                            'morning-day', 'morning-night', 'morning-terminator',
                            'terminator-average']
        
    Any subset of the above can be passed via 'show_profiles'.
        
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
    ax = plt.gca()
    
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
            if (show_profiles == []):
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
            if (show_profiles == []):
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
        if (show_profiles == []):
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
    ax.set_xlabel(r'Temperature (K)', fontsize = 20)
    ax.set_xlim(T_min, T_max)
    ax.set_ylabel(r'Pressure (bar)', fontsize = 20)
    ax.set_ylim(np.power(10.0, log_P_max), np.power(10.0, log_P_min))  
    ax.tick_params(labelsize=12)
    
    # Add legend
    legend = ax.legend(loc=legend_location, shadow=True, prop={'size':14}, ncol=1, 
                       frameon=False, columnspacing=1.0)
    
    fig.set_size_inches(9.0, 9.0)
    
    #plt.tight_layout()
    
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
    by the user via 'colour_list', or else default colours will be used
    (for up to 8 species).

    For a 1D model, a single chemical profile is plotted. For 2D or 3D 
    models, the user needs to specify the regions from which the profiles
    should be plotted. This is handled through 'show_profiles'.
    
    Valid choices for 2D and 3D models:
        
    * 2D Day-Night:
        -> show_profiles = ['day', 'night', 'terminator']
        
    * 2D Evening-Morning:
        -> show_profiles = ['morning', 'evening', 'average']
        
    * 3D:
        -> show_profiles = ['evening-day', 'evening-night', 'evening-terminator', 
                            'morning-day', 'morning-night', 'morning-terminator',
                            'terminator-average']
        
    Any subset of the above can be passed via 'show_profiles'.
        
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
    if ((colour_list != []) and (len(plot_species) != len(colour_list))):
        raise Exception("Number of colours does not match number of species.")
    for q, species in enumerate(plot_species):
        if (species not in chemical_species):
            raise Exception(species + " not included in this model.")

    # Find minimum and maximum mixing ratios in atmosphere
    if (log_X_min == None):
        log_X_min = np.floor(np.min(log_X)) - 1.0
    if (log_X_max == None):
        log_X_max = min((np.ceil(np.max(log_X)) + 1.0), 0.0)
    
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
    if (colour_list == []):   # If user did not specify a custom colour list
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
        legend.set_bbox_to_anchor([0.98, 0.98], transform=None)
    
    fig.set_size_inches(9.0, 9.0)

    #plt.tight_layout()

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

        wl_ticks = np.arange(round_sig_figs(wl_min, 2), round_sig_figs(wl_max, 2)+0.01, wl_spacing)

    return wl_ticks

    
def plot_spectra(spectra, planet, data_properties = None, show_data = False,
                 plot_full_res = True, bin_spectra = True, R_to_bin = 100, 
                 wl_min = None, wl_max = None, y_min = None, y_max = None,
                 y_unit = 'transit_depth', plt_label = None, 
                 colour_list = [], spectra_labels = [], data_colour_list = [],
                 data_labels = [], data_marker_list = [], 
                 data_marker_size_list = [], wl_axis = 'log', 
                 figure_shape = 'default', legend_location = 'upper right'):
    ''' 
    Plot a collection of individual model spectra. This function can plot
    transmission or emission spectra, according to the user's choice of 'y_unit'.
    
    Args:
        spectra (list): 
            A list of model spectra to be plotted, each with the form 
            (wavelength, spectrum).
        planet (dict):
            POSEIDON planet property dictionary.
        data_properties (dict, optional):
            POSEIDON observational data property dictionary.
        show_data (bool, optional):
            Flag indicating whether to plot the observational data.
        plot_full_res (bool, optional):
            Flag indicating whether to plot full resolution model spectra.
        bin_spectra (bool, optional):
            Flag indicating whether to bin model spectra to the resolution
            specified by 'R_to_bin'.
        R_to_bin (int, optional):
            Spectral resolution (R = wl/dwl) to bin the model spectra to. 
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
        wl_axis (str, optional):
            The type of x-axis to use ('log' or 'linear').
        figure_shape (str, optional):
            The shape of the figure ('default' or 'wide' - the latter is 16:9).
        legend_location (str, optional):
            The location of the legend ('upper left', 'upper right', 
            'lower left', 'lower right').

    Returns:
        fig (matplotlib figure object):
            The spectra plot.
    
    '''

    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', 'transit_depth']):
        plot_type = 'transmission'
    elif (y_unit in ['Fp/Fs', 'Fp/F*', 'eclipse_depth']):
        plot_type = 'emission'
    elif (y_unit in ['Fp']):
        plot_type = 'direct_emission'
    else:
        raise Exception("Unexpected y unit. Did you mean 'transit_depth' " +
                       "or 'eclipse_depth'?")
    
    # Find number of spectra to plot
    N_spectra = len(spectra)

    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Quick validity checks for plotting
    if (N_spectra == 0):
        raise Exception("Must provide at least one spectrum to plot!")
    if (N_spectra > 6):
        raise Exception("Max number of concurrent spectra to plot is 6.")
    if ((colour_list != []) and (N_spectra != len(colour_list))):
        raise Exception("Number of colours does not match number of spectra.")
    if ((spectra_labels != []) and (N_spectra != len(spectra_labels))):
        raise Exception("Number of model labels does not match number of spectra.")
        
    # Define colours for plotted spectra (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['green', 'red', 'black', 'darkgrey', 'navy', 'brown']
    else:
        colours = colour_list

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
        if (N_datasets > 6):
            raise Exception("Max number of concurrent datasets to plot is 6.")
        if ((data_colour_list != []) and (N_datasets != len(data_colour_list))):
            raise Exception("Number of colours does not match number of datasets.")
        if ((data_labels != []) and (N_datasets != len(data_labels))):
            raise Exception("Number of dataset labels does not match number of datasets.")
        if ((data_marker_list != []) and (N_datasets != len(data_marker_list))):
            raise Exception("Number of dataset markers does not match number of datasets.")
        if ((data_marker_size_list != []) and (N_datasets != len(data_marker_size_list))):
            raise Exception("Number of dataset marker sizes does not match number of datasets.")
            
        # Define colours for plotted spectra (default or user choice)
        if (data_colour_list == []):   # If user did not specify a custom colour list
            data_colours = ['orange', 'lime', 'cyan', 'magenta', 'brown', 'black']
        else:
            data_colours = data_colour_list

        # Define data marker symbols (default or user choice)
        if (data_marker_list == []):   # If user did not specify a custom colour list
            data_markers = ['o', 's', 'D', '*', 'X', 'p']
        else:
            data_markers = data_marker_list

        # Define data marker sizes (default or user choice)
        if (data_marker_size_list == []):   # If user did not specify a custom colour list
            data_markers_size = [3, 3, 3, 3, 3, 3]
        else:
            data_markers_size = data_marker_size_list
        
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
            
            transit_depth_min_i = np.min(spectra[i][0])
            y_min_plt = min(y_min_plt, transit_depth_min_i)
            
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
            
            transit_depth_max_i = np.max(spectra[i][0])
            y_max_plt = max(y_max_plt, transit_depth_max_i)
            
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
    
    major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 1)
    minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 1)
    
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
    else:
        raise Exception("Unsupported Figure shape - please use 'default' or 'wide'")
    
    ax1 = plt.gca()
    
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
        if (spectra_labels == []):
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
            
            # Calculate binned wavelength and spectrum grid
            wl_binned, spec_binned, _ = bin_spectrum(wl, spec, R_to_bin)

            if (plot_full_res == True):
                colour_binned = scale_lightness(colours[i], 0.4)
                lw_binned = 1.0
                label_i += ' (R = ' + str(R_to_bin) + ')'
            else:
                colour_binned = colours[i]
                lw_binned = 2.0

            # Plot binned spectrum
            ax1.plot(wl_binned, spec_binned, lw = lw_binned, alpha = 0.8, 
                     color = colour_binned, 
                     zorder = N_spectra+N_plotted_binned, 
                     label = label_i)
            
            N_plotted_binned += 1

    # Overplot datapoints
    if (show_data == True):

        for i in range(N_datasets):
            
            # If user did not specify dataset labels, use the instrument names
            if (data_labels == []):
                label_i = instruments[i]
            else:
                label_i = data_labels[i]
            
            # Find start and end indices of dataset_i in dataset property arrays
            idx_start = data_properties['len_data_idx'][i]
            idx_end = data_properties['len_data_idx'][i+1]

            # Extract the ith dataset
            wl_data_i = wl_data[idx_start:idx_end]
            ydata_i = ydata[idx_start:idx_end]
            err_data_i = err_data[idx_start:idx_end]
            bin_size_i = bin_size[idx_start:idx_end]

            # Plot dataset
            markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr = err_data_i, 
                                               xerr = bin_size_i, marker = data_markers[i], 
                                               markersize = data_markers_size[i], 
                                               capsize = 2, ls = 'none', elinewidth = 0.8, 
                                               color = data_colours[i], alpha = 0.8,
                                               ecolor = 'black', label = label_i,
                                               zorder = 100)

            [markers.set_alpha(1.0)]

    # Set axis ranges
    ax1.set_xlim([wl_min, wl_max])
    ax1.set_ylim([y_range[0], y_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)

    if (plot_type == 'transmission'):
        ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)
    elif (plot_type == 'emission'):
        ax1.set_ylabel(r'Emission Spectrum $(F_p/F_*)$', fontsize = 16)
    elif (plot_type == 'direct_emission'):
        ax1.set_ylabel(r'$F_{\rm{p}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

    # Add planet name label
    ax1.text(0.02, 0.96, planet_name, horizontalalignment = 'left', 
             verticalalignment = 'top', transform = ax1.transAxes, fontsize = 16)
  
    # Add plot label
    if (plt_label != None):
        ax1.text(0.03, 0.90, plt_label, horizontalalignment = 'left', 
                 verticalalignment = 'top', transform = ax1.transAxes, fontsize = 14)

    # Decide at which wavelengths to place major tick labels
    wl_ticks = set_spectrum_wl_ticks(wl_min, wl_max, wl_axis)
        
    # Plot wl tick labels
    ax1.set_xticks(wl_ticks)
    
    legend = ax1.legend(loc = legend_location, shadow = True, prop = {'size':10}, 
                        ncol = 1, frameon = True)    #legend settings

    frame = legend.get_frame()
    frame.set_facecolor('0.90') 
        
    for legline in legend.legendHandles:
        if ((plot_full_res == True) or (show_data == True)):
            legline.set_linewidth(1.0)
        else:
            legline.set_linewidth(2.0)
    
    plt.tight_layout()

    # Write figure to file
    if (plt_label == None):
        file_name = (output_dir + planet_name + '_' + plot_type + '_spectra.pdf')
    else:
        file_name = (output_dir + planet_name + '_' + plt_label + '_' +
                     plot_type + '_spectra.pdf')

    plt.savefig(file_name, bbox_inches = 'tight')

    return fig


def plot_data(data, planet, wl_min = None, wl_max = None, 
              transit_depth_min = None, transit_depth_max = None, 
              FpFs_min = None, FpFs_max = None, y_unit = 'transit_depth',
              plt_label = None, colour_list = [], data_labels = [], 
              data_marker_list = [], data_marker_size_list = [],
              wl_axis = 'log', figure_shape = 'default'):
    ''' 
    Plot a collection of datasets.
    
    '''

    # Unpack planet name
    planet_name = planet['planet_name']

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
    if (N_datasets > 6):
        raise Exception("Max number of concurrent datasets to plot is 6.")
    if ((colour_list != []) and (N_datasets != len(colour_list))):
        raise Exception("Number of colours does not match number of datasets.")
    if ((data_labels != []) and (N_datasets != len(data_labels))):
        raise Exception("Number of dataset labels does not match number of datasets.")
    if ((data_marker_list != []) and (N_datasets != len(data_marker_list))):
        raise Exception("Number of dataset markers does not match number of datasets.")
    if ((data_marker_size_list != []) and (N_datasets != len(data_marker_size_list))):
        raise Exception("Number of dataset marker sizes does not match number of datasets.")
        
    # Define colours for plotted spectra (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['orange', 'lime', 'cyan', 'magenta', 'brown', 'black']
    else:
        colours = colour_list

    # Define data marker symbols (default or user choice)
    if (data_marker_list == []):   # If user did not specify a custom colour list
        data_markers = ['o', 's', 'D', '*', 'X', 'p']
    else:
        data_markers = data_marker_list

    # Define data marker sizes (default or user choice)
    if (data_marker_size_list == []):   # If user did not specify a custom colour list
        data_markers_size = [3, 3, 3, 3, 3, 3]
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

    # If the user did not specify a transit depth range, find min and max from input models
    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', 'transit_depth']):
    
        if (transit_depth_min == None):
            y_min_plt = 0.995 * np.min(ydata - err_data) # Extend slightly below
        else:
            y_min_plt = transit_depth_min

        if (transit_depth_max == None):
            y_max_plt = 1.005 * np.max(ydata + err_data) # Extend slightly above
        else:
            y_max_plt = transit_depth_max

    # If the user did not specify an Fp/Fs range, find min and max from input models
    elif (y_unit in ['Fp/Fs', 'Fp/F*', 'Fp']):

        if (FpFs_min == None):
            y_min_plt = 0.995 * np.min(ydata - err_data) # Extend slightly below
        else:
            y_min_plt = FpFs_min

        if (FpFs_max == None):
            y_max_plt = 1.005 * np.max(ydata + err_data) # Extend slightly above
        else:
            y_max_plt = FpFs_max
        
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
    
    major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 1)
    minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 1)
    
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
    if ((transit_depth_min == None) and (FpFs_min == None)):
        y_min_plt = np.floor(y_min_plt/ymajor_spacing)*ymajor_spacing
    if ((transit_depth_max == None) and (FpFs_max == None)):
        y_max_plt = np.ceil(y_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    y_range = [y_min_plt, y_max_plt]

    # Create y formatting objects
    ymajorLocator   = MultipleLocator(ymajor_spacing)
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
    
    ax1 = plt.gca()
    
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
        if (data_labels == []):
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

        # Plot dataset
        markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr=err_data_i, 
                                            xerr=bin_size_i, marker=data_markers[i], 
                                            markersize=data_markers_size[i], 
                                            capsize=2, ls='none', elinewidth=0.8, 
                                            color=colours[i], alpha = 0.8,
                                            ecolor = 'black', label=label_i)

        [markers.set_alpha(1.0)]
            
    # Set axis ranges
    ax1.set_xlim([wl_min, wl_max])
    ax1.set_ylim([y_range[0], y_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)

    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', 'transit_depth']):
        ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)
    elif (y_unit in ['Fp/Fs', 'Fp/F*']):
        ax1.set_ylabel(r'Emission Spectrum $(F_p/F_*)$', fontsize = 16)
    elif (y_unit in ['Fp']):
        ax1.set_ylabel(r'$F_{\rm{p}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

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
    
    legend = ax1.legend(loc='upper right', shadow=True, prop={'size':10}, 
                        ncol=1, frameon=True)    #legend settings
  #  legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
    frame = legend.get_frame()
    frame.set_facecolor('0.90') 
        
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)
    
    plt.tight_layout()

    # Write figure to file
    if (plt_label == None):
        file_name = (output_dir + planet_name +
                     '_data.pdf')
    else:
        file_name = (output_dir + planet_name + '_' + plt_label + 
                     '_data.pdf')

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1, 
                           spectra_high1, spectra_high2, planet_name, 
                           data_properties, R_to_bin = 100, plt_label = None,
                           show_ymodel = True, wl_min = None, wl_max = None, 
                           transit_depth_min = None, transit_depth_max = None,
                           FpFs_min = None, FpFs_max = None, y_unit = 'transit_depth', 
                           colour_list = [], spectra_labels = [],
                           data_colour_list = [], data_labels = [],
                           data_marker_list = [], data_marker_size_list = [],
                           wl_axis = 'log', figure_shape = 'default',
                           legend_location = 'upper right'):
    ''' 
    Plot retrieved transmission spectra.
    
    '''

    # Find number of spectra to plot
    N_spectra = len(spectra_median)

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'
         
    # Quick validity checks for plotting
    if (N_spectra == 0):
        raise Exception("Must provide at least one spectrum to plot!")
    if (N_spectra > 3):
        raise Exception("Max number of concurrent retrieved spectra to plot is 3.")
    if ((colour_list != []) and (N_spectra != len(colour_list))):
        raise Exception("Number of colours does not match number of spectra.")
    if ((spectra_labels != []) and (N_spectra != len(spectra_labels))):
        raise Exception("Number of model labels does not match number of spectra.")

    # Define colours for plotted spectra (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['purple', 'darkorange', 'green']
    else:
        colours = colour_list

    binned_colours = ['gold', 'pink', 'cyan']

    # Unpack data properties (if provided)
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
    if (N_datasets > 6):
        raise Exception("Max number of concurrent datasets to plot is 6.")
    if ((data_colour_list != []) and (N_datasets != len(data_colour_list))):
        raise Exception("Number of colours does not match number of datasets.")
    if ((data_labels != []) and (N_datasets != len(data_labels))):
        raise Exception("Number of dataset labels does not match number of datasets.")
    if ((data_marker_list != []) and (N_datasets != len(data_marker_list))):
        raise Exception("Number of dataset markers does not match number of datasets.")
    if ((data_marker_size_list != []) and (N_datasets != len(data_marker_size_list))):
        raise Exception("Number of dataset marker sizes does not match number of datasets.")
        
    # Define colours for plotted spectra (default or user choice)
    if (data_colour_list == []):   # If user did not specify a custom colour list
        data_colours = ['lime', 'cyan', 'magenta', 'orange', 'brown', 'black']
    else:
        data_colours = data_colour_list

    # Define data marker symbols (default or user choice)
    if (data_marker_list == []):   # If user did not specify a custom colour list
        data_markers = ['o', 's', 'D', '*', 'X', 'p']
    else:
        data_markers = data_marker_list

    # Define data marker sizes (default or user choice)
    if (data_marker_size_list == []):   # If user did not specify a custom colour list
        data_markers_size = [3, 3, 3, 3, 3, 3]
    else:
        data_markers_size = data_marker_size_list
                
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

    # If the user did not specify a transit depth range, find min and max from input models
    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', 'transit_depth']):
    
        if (transit_depth_min == None):
            
            y_min_plt = 1e10   # Dummy value
            
            # Loop over each model, finding the most extreme min / max range 
            for i in range(N_spectra):

                (spec_low2, wl) = spectra_low2[i]
                _, spec_low2_binned, _ = bin_spectrum(wl, spec_low2, R_to_bin)

                transit_depth_min_i = np.min(spec_low2_binned)
                y_min_plt = min(y_min_plt, transit_depth_min_i)
                
            # Check if the lowest data point falls below the current y-limit
            if (y_min_plt > min(ydata - err_data)):
                y_min_plt = min(ydata - err_data)
                
            y_min_plt = 0.995*y_min_plt  # Extend slightly below
            
        else:
            y_min_plt = transit_depth_min

        if (transit_depth_max == None):
            
            y_max_plt = 1e-10  # Dummy value
            
            # Loop over each model, finding the most extreme min / max range 
            for i in range(N_spectra):

                (spec_high2, wl) = spectra_high2[i]
                _, spec_high2_binned, _ = bin_spectrum(wl, spec_high2, R_to_bin)

                transit_depth_max_i = np.max(spec_high2_binned)
                y_max_plt = max(y_max_plt, transit_depth_max_i)
                
            # Check if the highest data point falls above the current y-limit
            if (y_max_plt < max(ydata + err_data)):
                y_max_plt = max(ydata + err_data)
                
            y_max_plt = 1.040*y_max_plt  # Extend slightly above
            
        else:
            y_max_plt = transit_depth_max

    # If the user did not specify an Fp/Fs range, find min and max from input models
    elif (y_unit in ['Fp/Fs', 'Fp/F*', 'Fp']):

        if (FpFs_min == None):
            
            y_min_plt = 1e10   # Dummy value
            
            # Loop over each model, finding the most extreme min / max range 
            for i in range(N_spectra):

                (spec_low2, wl) = spectra_low2[i]
                _, spec_low2_binned, _ = bin_spectrum(wl, spec_low2, R_to_bin)

                FpFs_min_i = np.min(spec_low2_binned)
                y_min_plt = min(y_min_plt, FpFs_min_i)
                
            # Check if the lowest data point falls below the current y-limit
            if (y_min_plt > min(ydata - err_data)):
                y_min_plt = min(ydata - err_data)
                
            y_min_plt = 0.995*y_min_plt  # Extend slightly below
            
        else:
            y_min_plt = FpFs_min

        if (FpFs_max == None):
            
            y_max_plt = 1e-10  # Dummy value
            
            # Loop over each model, finding the most extreme min / max range 
            for i in range(N_spectra):

                (spec_high2, wl) = spectra_high2[i]
                _, spec_high2_binned, _ = bin_spectrum(wl, spec_high2, R_to_bin)

                FpFs_max_i = np.max(spec_high2_binned)
                y_max_plt = max(y_max_plt, FpFs_max_i)
                
            # Check if the highest data point falls above the current y-limit
            if (y_max_plt < max(ydata + err_data)):
                y_max_plt = max(ydata + err_data)
                
            y_max_plt = 1.040*y_max_plt  # Extend slightly above
            
        else:
            y_max_plt = FpFs_max

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
    
    major_exponent = round_sig_figs(np.floor(np.log10(np.abs(ymajor_spacing))), 1)
    minor_exponent = round_sig_figs(np.floor(np.log10(np.abs(yminor_spacing))), 1)
    
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
    if ((transit_depth_min == None) and (FpFs_min == None)):
        y_min_plt = np.floor(y_min_plt/ymajor_spacing)*ymajor_spacing
    if ((transit_depth_max == None) and (FpFs_max == None)):
        y_max_plt = np.ceil(y_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    y_range = [y_min_plt, y_max_plt]

    # Create y formatting objects
    ymajorLocator   = MultipleLocator(ymajor_spacing)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
    yminorLocator = MultipleLocator(yminor_spacing)

#    ymajorLocator_H   = MultipleLocator(1)
#    ymajorFormatter_H = FormatStrFormatter('%.0f')
#    yminorLocator_H   = MultipleLocator(0.2)

    # Generate figure and axes
    fig = plt.figure()

    # Set figure size
    if (figure_shape == 'default'):
        fig.set_size_inches(8.0, 6.0)    # Default Matplotlib figure size
    elif (figure_shape == 'wide'):
        fig.set_size_inches(10.667, 6.0)    # 16:9 widescreen format (for two column figures) 
    
    ax1 = plt.gca()
    
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
        if (spectra_labels == []):
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
        ax1.plot(wl_binned, spec_med_binned, lw = 1.0,  
                 color = scale_lightness(colours[i], 1.0), 
                 label = label_med)
        
        # Plot +/- 1σ confidence region
        ax1.fill_between(wl_binned, spec_low1_binned, spec_high1_binned,
                         lw=0.0, alpha=0.5, facecolor=colours[i],  
                         label = label_one_sig)

        # Plot +/- 2σ sigma confidence region
        ax1.fill_between(wl_binned, spec_low2_binned, spec_high2_binned,
                         lw=0.0, alpha=0.2, facecolor=colours[i],  
                         label = label_two_sig)

        # Overplot median model, binned to resolution of the observations
        if (show_ymodel == True):
            ymodel_median = bin_spectrum_to_data(spec_med, wl, data_properties)

            ax1.scatter(wl_data, ymodel_median, color = binned_colours[i], 
                        s=5, marker='D', lw=0.1, alpha=0.8, edgecolor='black',
                        label = label_i + r' (Binned)', zorder = 200)
            
    # Overplot datapoints
    for i in range(N_datasets):
        
        # If user did not specify dataset labels, use the instrument names
        if (data_labels == []):
            label_i = instruments[i]
        else:
            label_i = data_labels[i]
        
        # Find start and end indices of dataset_i in dataset property arrays
        idx_start = data_properties['len_data_idx'][i]
        idx_end = data_properties['len_data_idx'][i+1]

        # Extract the ith dataset
        wl_data_i = wl_data[idx_start:idx_end]
        ydata_i = ydata[idx_start:idx_end]
        err_data_i = err_data[idx_start:idx_end]
        bin_size_i = bin_size[idx_start:idx_end]

        # Plot dataset
        markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr=err_data_i, 
                                            xerr=bin_size_i, marker=data_markers[i], 
                                            markersize=data_markers_size[i], 
                                            capsize=2, ls='none', elinewidth=0.8, 
                                            color=data_colours[i], alpha = 0.8,
                                            ecolor = 'black', label=label_i,
                                            zorder = 100)

        [markers.set_alpha(1.0)]
    
    # Set axis ranges
    ax1.set_xlim([wl_min, wl_max])
    ax1.set_ylim([y_range[0], y_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)

    if (y_unit in ['(Rp/Rs)^2', '(Rp/R*)^2', 'transit_depth']):
        ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)
    elif (y_unit in ['Fp/Fs', 'Fp/F*']):
        ax1.set_ylabel(r'Emission Spectrum $(F_p/F_*)$', fontsize = 16)
    elif (y_unit in ['Fp']):
        ax1.set_ylabel(r'$F_{\rm{p}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

    # Add planet name label
    ax1.text(0.02, 0.96, planet_name, horizontalalignment='left', 
             verticalalignment='top', transform=ax1.transAxes, fontsize = 16)

    # Decide at which wavelengths to place major tick labels
    wl_ticks = set_spectrum_wl_ticks(wl_min, wl_max, wl_axis)
        
    # Plot wl tick labels
    ax1.set_xticks(wl_ticks)

    legend = ax1.legend(loc=legend_location, shadow=True, prop={'size':10}, 
                        ncol=1, frameon=False)    #legend settings
  #  legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
  #  frame = legend.get_frame()
  #  frame.set_facecolor('0.90') 
            
    plt.tight_layout()

    # Write figure to file
    if (plt_label == None):
        file_name = output_dir + planet_name + '_retrieved_spectra.pdf'
    else:
        file_name = output_dir + planet_name + '_' + plt_label + '_retrieved_spectra.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_PT_retrieved(planet_name, PT_median, PT_low2, PT_low1, PT_high1,
                      PT_high2, T_true = None, Atmosphere_dimension = 1, 
                      TwoD_type = None, plt_label = None, show_profiles = [],
                      PT_labels = [], colour_list = [], log_P_min = None,
                      log_P_max = None, T_min = None, T_max = None,
                      legend_location = 'lower left'):
        
    ''' Plot a retrieved Pressure-Temperature (P-T) profile.
        
    '''

    # Find number of P-T profiles to plot
    N_PT = len(PT_median)

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Quick validity checks for plotting
    if (N_PT == 0):
        raise Exception("Must provide at least one P-T profile to plot!")
    if (N_PT > 3):
        raise Exception("Max number of concurrent retrieved P-T profiles to plot is 3.")
    if ((colour_list != []) and (N_PT != len(colour_list))):
        raise Exception("Number of colours does not match number of P-T profiles.")
    if ((PT_labels != []) and (N_PT != len(PT_labels))):
        raise Exception("Number of model labels does not match number of P-T profiles.")

    # Define colours for plotted spectra (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
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
    if (T_range >= 500.0):
        major_spacing = max(np.around((T_range/10), -2), 100.0)
    elif (T_range < 500.0):
        major_spacing = max(np.around((T_range/10), -1), 10.0)
        
    minor_spacing = major_spacing/10

    # Load pressure grid
    P = PT_median[0][1]

    if (log_P_min == None):
        log_P_min = np.log10(np.min(P))
    if (log_P_max == None):
        log_P_max = np.log10(np.max(P))
    
    # create figure
    fig = plt.figure()  
    ax = plt.gca()
    
    # Assign axis spacing
    xmajorLocator_PT = MultipleLocator(major_spacing)
    xminorLocator_PT = MultipleLocator(minor_spacing)
        
    ax.xaxis.set_major_locator(xmajorLocator_PT)
    ax.xaxis.set_minor_locator(xminorLocator_PT)
    
    #***** Plot P-T profiles *****#
    
    # 1D temperature profile
    if (Atmosphere_dimension > 1):
        raise Exception("This function does not support multidimensional retrievals.")
        
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
            if (PT_labels == []):
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
            ax.semilogy(T_med, P, lw = 1.5, color = scale_lightness(colours[i], 1.0), 
                        label = label_med)
            
            # Plot +/- 1σ confidence region
            ax.fill_betweenx(P, T_low1, T_high1, lw = 0.0, alpha = 0.5, 
                            facecolor = colours[i], label = label_one_sig)

            # Plot +/- 2σ sigma confidence region
            ax.fill_betweenx(P, T_low2, T_high2, lw = 0.0, alpha = 0.2, 
                            facecolor = colours[i], label = label_two_sig)

        # Plot actual (true) P-T profile
        if (T_true != None):
            ax.semilogy(T_true, P, lw = 1.5, color = 'crimson', label = 'True')

    # Common plot settings for all profiles
    ax.invert_yaxis()            
    ax.set_xlabel(r'Temperature (K)', fontsize = 20)
    ax.set_xlim(T_min, T_max)
    ax.set_ylabel(r'Pressure (bar)', fontsize = 20)
    ax.set_ylim(np.power(10.0, log_P_max), np.power(10.0, log_P_min)) 

    ax.tick_params(labelsize=12)
    
    # Add legend
    legend = ax.legend(loc=legend_location, shadow=True, prop={'size':14}, ncol=1, 
                       frameon=False, columnspacing=1.0)
    
    fig.set_size_inches(9.0, 9.0)

    #plt.tight_layout()

    #legend.set_bbox_to_anchor([0.20, 0.10], transform=None)

    # Write figure to file
    if (plt_label == None):
        file_name = output_dir + planet_name + '_retrieved_PT.pdf'
    else:
        file_name = output_dir + planet_name + '_' + plt_label + '_retrieved_PT.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_chem_retrieved(planet_name, chemical_species, log_Xs_median, 
                        log_Xs_low2, log_Xs_low1, log_Xs_high1, log_Xs_high2, 
                        log_X_true = None, plot_species = [], plot_two_sigma = False,
                        Atmosphere_dimension = 1, TwoD_type = None, plt_label = None, 
                        show_profiles = [], model_labels = [], colour_list = [],
                        log_P_min = None, log_P_max = None, log_X_min = None, 
                        log_X_max = None):
        
    ''' Plot retrieved mixing ratio profiles.
        
    '''
  
    # Find number of mixing ratio model profiles to plot
    N_chem = len(log_Xs_median)

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # If the user did not specify which species to plot, plot all of them
    if (len(plot_species) == 0):
        plot_species = chemical_species

    # Quick validity checks for plotting
    if (N_chem >= 2):
        raise Exception("Only 1 set of mixing ratio profiles can be plotted currently.")
    if (len(plot_species) > 8):
        raise Exception("Max number of concurrent species on plot is 8.\n"
                        "Please specify species to plot via plot_species = [LIST]")
    if ((colour_list != []) and (len(plot_species) != len(colour_list))):
        raise Exception("Number of colours does not match number of species.")
    if ((model_labels != []) and (N_chem != len(model_labels))):
        raise Exception("Number of model labels does not match number of mixing ratio profiles.")
    for q, species in enumerate(plot_species):
        if (species not in chemical_species):
            raise Exception(species + " not included in this model.")

    # Define colours for mixing ratio profiles (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
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
        raise Exception("This function does not support multidimensional retrievals.")
        
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
                if (model_labels == []):
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

    #plt.tight_layout()

    #legend.set_bbox_to_anchor([0.20, 0.10], transform=None)

    # Write figure to file
    if (plt_label == None):
        file_name = output_dir + planet_name + '_retrieved_chem.pdf'
    else:
        file_name = output_dir + planet_name + '_' + plt_label + '_retrieved_chem.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_stellar_flux(Flux, wl, wl_min = None, wl_max = None, flux_min = None,
                      flux_max = None, flux_axis = 'linear', wl_axis = 'log'):
    
    fig = plt.figure()  
        
    ax = plt.gca()

    ax.set_yscale(flux_axis)
    ax.set_xscale(wl_axis)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

    ax.plot(wl, Flux, lw=1, alpha=0.8, label=r'Stellar Flux')

    ax.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax.set_ylabel(r'Surface Flux (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

    if (wl_min == None):
        wl_min = min(wl)
    if (wl_max == None):
        wl_max = max(wl)
    if (flux_min == None):
        flux_min = min(Flux)
    if (flux_max == None):
        flux_max = max(Flux)   

    ax.set_xlim([wl_min, wl_max])
    ax.set_ylim([flux_min, flux_max])

    ax.legend(loc='upper right', shadow=True, prop={'size':10}, ncol=1, frameon=False)
    
    return fig


def plot_FpFs(planet, model, FpFs, wl, R_to_bin = 100):

    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Create y formatting objects
    ymajorLocator   = MultipleLocator(1.0e-4)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
    yminorLocator = MultipleLocator(1.0e-5)
    
    fig = plt.figure()  
        
    ax = plt.gca()

    ax.set_xscale("log")

    # Assign formatter objects to axes
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.yaxis.set_minor_locator(yminorLocator)

    ax.plot(wl, FpFs, lw=0.5, alpha=0.4, color = 'crimson', label=r'Flux Ratio')

    # Calculate binned wavelength and spectrum grid
    wl_binned, FpFs_binned, _ = bin_spectrum(wl, FpFs, R_to_bin)

    # Plot binned spectrum
    ax.plot(wl_binned, FpFs_binned, lw=1.0, alpha=0.8, 
                color=scale_lightness('crimson', 0.4),
                label='Flux Ratio' + ' (R = ' + str(R_to_bin) + ')')

    # Decide at which wavelengths to place major tick labels
    wl_min = min(wl)
    wl_max = max(wl)

    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 2.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 3.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 3)+0.01, 0.5)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 10.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        wl_ticks_4 = np.array([])
    else:
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, 10.0, 1.0)
        wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 2)+0.01, 2.0)

    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3, wl_ticks_4))
    
    # Plot wl tick labels
    ax.set_xticks(wl_ticks)

    ax.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax.set_ylabel(r'$F_{\rm{p}} / F_*$', fontsize = 16)

    ax.set_xlim([min(wl), max(wl)])

    ax.legend(loc='upper left', shadow=True, prop={'size':10}, ncol=1, frameon=False)
    
    # Write figure to file
    file_name = output_dir + model_name + '_emission_spectra.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_Fp(planet, model, Fp, wl, R_to_bin = 100):

    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Create y formatting objects
  #  ymajorLocator   = MultipleLocator(1.0e-4)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
  #  yminorLocator = MultipleLocator(1.0e-5)
    
    fig = plt.figure()  
        
    ax = plt.gca()

   # ax.set_xscale("log")

    # Assign formatter objects to axes
  #  ax.xaxis.set_major_formatter(ScalarFormatter())
  #  ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
  #  ax.yaxis.set_minor_locator(yminorLocator)

    ax.plot(wl, Fp, lw=0.5, alpha=0.4, color = 'crimson', 
            label='Flux (R = 15,000)')

    # Calculate binned wavelength and spectrum grid
    wl_binned, Fp_binned, _ = bin_spectrum(wl, Fp, R_to_bin)

    # Plot binned spectrum
    ax.plot(wl_binned, Fp_binned, lw=1.0, alpha=0.8, 
            color=scale_lightness('crimson', 0.4),
            label='Flux' + ' (R = ' + str(R_to_bin) + ')')

    # Decide at which wavelengths to place major tick labels
    wl_min = min(wl)
    wl_max = max(wl)

    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 2.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 3.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 3)+0.01, 0.5)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 10.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        wl_ticks_4 = np.array([])
    else:
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, 10.0, 1.0)
        wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 2)+0.01, 2.0)

    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3, wl_ticks_4))
    
    # Plot wl tick labels
    ax.set_xticks(wl_ticks)

    ax.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax.set_ylabel(r'$F_{\rm{p}}$ (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

    ax.set_xlim([min(wl), max(wl)])

    ax.legend(loc='upper right', shadow=True, prop={'size':10}, ncol=1, frameon=False)
    
    # Write figure to file
    file_name = output_dir + model_name + '_emission_spectra.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_chem_histogram(nbins, X_i_vals, colour, oldax, shrink_factor):
    
    weights = np.ones_like(X_i_vals)/float(len(X_i_vals))
    
    x,w,patches = oldax.hist(X_i_vals, bins=nbins, color=colour, histtype='stepfilled', alpha=0.4, edgecolor='None', weights=weights, density=True, stacked=True)
    x,w,patches = oldax.hist(X_i_vals, bins=nbins, histtype='stepfilled', lw = 0.8, facecolor='None', weights=weights, density=True, stacked=True)
        
    oldax.set_ylim(0, (1.1+shrink_factor)*x.max())
    
    low3, low2, low1, median, high1, high2, high3 = confidence_intervals(len(X_i_vals), X_i_vals, 0)
    
    return low1, median, high1

'''
def plot_retrieved_element_ratios(X_vals, planet_name, model_name, chemical_species,
                                  abundance_fmt = '.2f'):

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/results/'
    
    fig = plt.figure()  
    fig.set_size_inches(5, 2.5)
    #fig.set_size_inches(10.3, 7.5)
       
    # Initialise histogram grid (use subplot2grid for irregular spacings)
    gs = gridspec.GridSpec(1, 2) 
   
    # Colours for histograms
    colours = ['dodgerblue', 'orangered']
   
    # Names of the parameters to be plotted
    parameters = []
    parameters.append(r"$\rm{log}(\rm{O / H}) \, \, (\times \, \rm{solar})$")
    parameters.append(r"$\rm{C / O}$")

    N_samples, _ = np.shape(X_vals)
   
    # Empty arrays for mu, O/H, and C/O
    O_to_H_vals = np.zeros(shape=(N_samples))
    C_to_O_vals = np.zeros(shape=(N_samples))
   
    O_to_H_solar = np.power(10.0, (8.69-12.0))  # Asplund (2009) ~ 4.9e-4  (Present day photosphere value)
   
    # Work out mean molecular mass and metallicity for each retrieved chemistry vector (functions defined in profile.py)
    for i in range(N_samples):
   
        O_to_H_vals[i] = compute_O_to_H(X_vals[i,:], chemical_species) / O_to_H_solar
        C_to_O_vals[i] = compute_C_to_O(X_vals[i,:], chemical_species)
   
    for i in range(len(parameters)):
   
        count = i+1

        plt.subplot(gs[0,i])
   
        oldax = plt.gca()
   
        if (count == 1): low1, median, high1 = plot_chem_histogram(40, np.log10(O_to_H_vals), colours[i], oldax, 0.0)
        if (count == 2): low1, median, high1 = plot_chem_histogram(40, C_to_O_vals, colours[i], oldax, 0.0)
            
        fmt = "{{0:{0}}}".format(abundance_fmt).format
        overlay = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        overlay = overlay.format(fmt(median), fmt((median-low1)), fmt((high1-median)))

        newax = plt.gcf().add_axes(oldax.get_position(), sharex=oldax, frameon=False)
        newax.set_ylim(0, 1)
   	
        ylim = newax.get_ylim()
        y = ylim[0] + 0.06*(ylim[1] - ylim[0])
   
        newax.errorbar(x=median, y=y,
                       xerr=np.transpose([[median - low1, high1 - median]]), 
                       color='lightgreen', ecolor='green', markersize=3, 
                       markeredgewidth = 0.6, linewidth=0.9, capthick=0.9, capsize=1.7, marker='s')
                      
        oldax.set_yticks([])
       
        oldax.tick_params(axis='both', which='major', labelsize=8)
        newax.tick_params(axis='both', which='major', labelsize=8)

        median = round_sig_figs(median, 2)
        upper_sigma = round_sig_figs((high1-median), 2)
        lower_sigma = round_sig_figs((median-low1), 2)

        # Custom plotting options for each histogram          
        if (count == 1):
            oldax.set_xlim([-0.2, 2.6])
            newax.text(0.06, 0.96, r'O/H', color='navy', fontsize = 10, 
                       horizontalalignment='left', verticalalignment='top', transform=newax.transAxes)
            newax.text(0.96, 0.96, overlay, color='navy', fontsize = 10,
                       horizontalalignment='right', verticalalignment='top', transform=newax.transAxes)
            newax.axvline(x=0.0, linewidth=1.5, linestyle='-', color='crimson', alpha=0.8)
            newax.set_yticklabels([])
            
        if (count == 2):
            oldax.set_xlim([0.0, 1.0])
            newax.text(0.06, 0.96, r'C/O', color='maroon', fontsize = 10, 
                       horizontalalignment='left', verticalalignment='top', transform=newax.transAxes)
            newax.text(0.96, 0.96, overlay, color='maroon', fontsize = 10,
                       horizontalalignment='right', verticalalignment='top', transform=newax.transAxes)
            newax.axvline(x=0.55, linewidth=1.5, linestyle='-', color='crimson', alpha=0.8)
            newax.set_yticklabels([])

        plt.xlabel(parameters[i], fontsize = 12, labelpad = 1)
        
  #  plt.tight_layout(pad = 0.7, w_pad = 0.2, h_pad = 0.8)
        
    # Write figure to file
    file_name = output_dir + model_name + '_element_ratios.png'

    plt.savefig(file_name, bbox_inches='tight', dpi=300)


def plot_composition(planet, model):

    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']
    param_species = model['param_species']
    N_params_cum = model['N_params_cum']

    # Unpack number of free parameters
    param_names = model['param_names']
    n_params = len(param_names)

    # Identify output directory location
    output_dir = './POSEIDON_output/' + planet_name + '/retrievals/'

    # Load relevant output directory
    output_prefix = model_name + '-'

    # Change directory into MultiNest result file folder
    os.chdir(output_dir + 'MultiNest_raw/')
    
    # Run PyMultiNest analyser to extract posterior samples
    analyzer = pymultinest.Analyzer(n_params, outputfiles_basename = output_prefix,
                                    verbose = False)
    samples = analyzer.get_equal_weighted_posterior()[:,:-1]

    # Change directory back to directory where user's python script is located
    os.chdir('../../../../')

    # Find total number of available posterior samples from MultiNest 
    N_samples = len(samples[:,0])
    N_species = len(param_species)

    log_X_stored = np.zeros(shape=(N_samples, N_species))
    
    # Generate spectrum and PT profiles from selected samples
    for i in range(N_samples):

        # Convert MultiNest parameter samples into POSEIDON function inputs
        _, _, log_X_stored[i,:], _, _, _, _, _ = split_params(samples[i], 
                                                              N_params_cum)

    # Plot elemental ratios
    plot_retrieved_element_ratios(np.power(10.0, log_X_stored), planet_name, 
                                  model_name, param_species)
'''