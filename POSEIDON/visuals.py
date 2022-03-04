# Plotting routines to visualise POSEIDON output*****

from enum import unique
import pymultinest
import numpy as np
import h5py
import scipy.constants as sc
from scipy.ndimage import gaussian_filter1d as gauss_conv
import colorsys
import matplotlib
from pylab import rcParams
import matplotlib.style
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, \
                              FuncFormatter, ScalarFormatter, NullFormatter
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('classic')
plt.rc('font', family='serif')
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['figure.facecolor']='white'

import warnings

# Suppress invalid warning about x limits which sometimes occur for spectra plots
warnings.filterwarnings("ignore", message="Attempted to set non-positive left " + 
                                          "xlim on a log-scaled axis.\n" + 
                                          "Invalid limit will be ignored.")

from .utility import confidence_intervals, bin_spectrum_fast, closest_index, \
                     generate_latex_param_names, round_sig_figs, file_name_check
from .absorption import H_minus_bound_free, H_minus_free_free
from .instrument import bin_spectrum_to_data


# import warnings

# warnings.filterwarnings( "ignore", module = "matplotlib\..*" )



              
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
    
    # Convert colour name to RBG value
    rgb = matplotlib.colors.ColorConverter.to_rgb(colour_name)
    
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Manipulate h, l, s values and return as RGB
    
    return colorsys.hls_to_rgb(h, min(1, l * scale), s = s)


def plot_transit(ax, R_p, R_s, r, T, phi, phi_edge, dphi, theta, theta_edge, dtheta, 
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

        # In single zone case, can skip edge-finding calculations
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
            
        ax.set_xlim([-1.2*r_max, 1.2*r_max])
        ax.set_ylim([-1.2*r_max, 1.2*r_max])
        
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
        p.set_clim([0.98*np.min(T_pole), 1.02*np.max(T_pole)])
        ax.add_collection(p)
        
        # Add labels
        if (plot_labels == True):
            ax.text(0.7*r_term_max, 1.02*r_term_max, 'Morning', fontsize = 14)
            ax.text(-1.1*r_term_max, 1.02*r_term_max, 'Evening', fontsize = 14)     
  
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
            
        ax.set_xlim([-1.2*r_max, 1.2*r_max])
        ax.set_ylim([-1.2*r_max, 1.2*r_max])
        
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
        p.set_clim([0.98*np.min(T_pole), 1.02*np.max(T_pole)])
        ax.add_collection(p)

        # Add labels
        if (plot_labels == True):
            ax.text(0.7*r_pole_max, 0.90*r_pole_max, 'Night', fontsize = 14)
            ax.text(-0.28*r_pole_max, 1.08*r_pole_max, 'Terminator', fontsize = 14)
            ax.text(-0.9*r_pole_max, 0.90*r_pole_max, 'Day', fontsize = 14)
            
            ax.text(-0.9*r_pole_max, 1.07*r_pole_max, 'Star', fontsize = 14)
            ax.annotate(s='', xy=(-1.1*r_pole_max, 1.02*r_pole_max), xytext=(-0.5*r_pole_max, 1.02*r_pole_max), 
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.8))
            ax.text(0.55*r_pole_max, 1.07*r_pole_max, 'Observer', fontsize = 14)
            ax.annotate(s='', xy=(0.5*r_pole_max, 1.02*r_pole_max), xytext=(1.1*r_pole_max, 1.02*r_pole_max), 
                        arrowprops=dict(arrowstyle='<-', color='black', alpha=0.8))

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
    R_s = star['stellar_radius']
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
    p = plot_transit(ax1, R_p, R_s, r, T, phi, phi_edge, dphi, theta, 
                     theta_edge, dtheta, 'terminator', plot_labels) 

    # Plot side perspective on RHS axis
    _ = plot_transit(ax2, R_p, R_s, r, T, phi, phi_edge, dphi, theta, 
                     theta_edge, dtheta, 'day-night', plot_labels) 
    
    # Plot temperature colourbar
    cbaxes = fig.add_axes([1.01, 0.131, 0.015, 0.786]) 
    cb = plt.colorbar(p, cax = cbaxes)  
    tick_locator = ticker.MaxNLocator(nbins=8)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.formatter.set_useOffset(False)
    
    plt.tight_layout()
    
    # Write figure to file
    plt.savefig(output_dir + model_name + '_Geometry.png', 
                bbox_inches='tight', dpi=800)

    return fig


def plot_PT(planet, model, atmosphere, show_profiles = []):
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
    T_min = np.floor(np.min(T)/100)*100 - 200.0    # Round down to nearest 100
    T_max = np.ceil(np.max(T)/100)*100 + 200.0     # Round up to nearest 100
        
    # Find range to plot
    T_range = T_max - T_min    
    
    # Calculate appropriate axis spacing
    if (T_range >= 500.0):
        major_spacing = max(np.around((T_range/10), -2), 100.0)
    elif (T_range < 500.0):
        major_spacing = max(np.around((T_range/10), -1), 10.0)
        
    minor_spacing = major_spacing/10
    
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
        ax.semilogy(T[:,0,0], P, lw=1.5, color = 'darkblue', label='P-T Profile')

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
    ax.tick_params(labelsize=12)
    
    # Add legend
    legend = ax.legend(loc='lower left', shadow=True, prop={'size':14}, ncol=1, 
                       frameon=False, columnspacing=1.0)
    
    fig.set_size_inches(9.0, 9.0)
    
    #plt.tight_layout()
    
    # Write figure to file
    plt.savefig(output_dir + model_name + '_PT.pdf', bbox_inches='tight')

    return fig
    

def plot_chem(planet, model, atmosphere, plot_species = [], 
              colour_list = [], show_profiles = []):    
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
    log_X_min = np.floor(np.min(log_X)) - 1.0
    log_X_max = min((np.ceil(np.max(log_X)) + 1.0), 0.0)
    
    # When range is small, extend axes +/- 1 dex either side
    if (log_X_min == log_X_max):
        log_X_min = log_X_min - 1.0
        log_X_max = log_X_max + 1.0
        
    # Find range to plot
    log_X_range = log_X_max - log_X_min    
    
    # Calculate appropriate axis spacing
    major_spacing = 1.0
    minor_spacing = major_spacing/10
    
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
                
                # Otherwise, plot user choices
                else:
                    
                    for profile_to_plot in show_profiles:
                        
                        if (profile_to_plot not in ['day', 'night', 'terminator']):
                            print ("Ignoring invalid profile '" + profile_to_plot +
                                   "', since only 'day', 'night', and 'terminator' " +
                                   "are valid for a 2D day-night model.")

                        if (profile_to_plot == 'day'):
                            ax.semilogy(log_X[chemical_species == species,:,0,0][0], 
                                        P, lw=1.5, ls=':', color = colours[q], 
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
                
                # Otherwise, plot user choices
                else:
                    
                    for profile_to_plot in show_profiles:
                        
                        if (profile_to_plot not in ['evening', 'morning', 'average']):
                            print ("Ignoring invalid profile '" + profile_to_plot +
                                   "', since only 'evening', 'morning', and 'average' " +
                                   "are valid for a 2D evening-morning model.")
                            
                        if (profile_to_plot == 'evening'):
                            ax.semilogy(log_X[chemical_species == species,:,0,0][0], 
                                        P, lw=1.5, ls=':', color = colours[q], 
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
    ax.tick_params(labelsize=12)
        
    # Add legend
    legend = ax.legend(loc='upper right', shadow=True, prop={'size':14}, 
                       frameon=True, columnspacing=1.0)
    frame = legend.get_frame()
    frame.set_facecolor('0.90') 
    
    fig.set_size_inches(9.0, 9.0)

    #plt.tight_layout()

    # Write figure to file
    plt.savefig(output_dir + model_name + '_chem.pdf', bbox_inches='tight')

    return fig


def plot_spectra(spectra, planet, model, data_properties = None,
                 plot_full_res = True, bin_spectra = True, R_to_bin = 100, 
                 wl_min = None, wl_max = None, transit_depth_min = None,
                 transit_depth_max = None, show_data = False, 
                 colour_list = [], spectra_labels = []):
    ''' 
    Plot a collection of individual model transmission spectra.
    
    '''
    
    # Find number of spectra to plot
    N_spectra = len(spectra)

    # Unpack model and atmospheric properties
    planet_name = planet['planet_name']
    model_name = model['model_name']

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Unpack data properties (if provided)
    if (data_properties is not None):
        ydata = data_properties['ydata']
        err_data = data_properties['err_data']
        wl_data = data_properties['wl_data']
        bin_size = data_properties['half_bin']
  
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
        colours = ['green', 'red', 'blue', 'brown', 'black', 'darkgrey']
    else:
        colours = colour_list
                
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

    # Set x range
    wl_range = [wl_min, wl_max]
    
    # If the user did not specify a transit depth range, find min and max from input models
    if (transit_depth_min == None):
        
        transit_depth_min_plt = 1e10   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            transit_depth_min_i = np.min(spectra[i][0])
            transit_depth_min_plt = min(transit_depth_min_plt, transit_depth_min_i)
            
        # Check if the lowest data point falls below the current y-limit
        if (show_data == True):
            if (transit_depth_min_plt > min(ydata - err_data)):
                
                transit_depth_min_plt = min(ydata - err_data)
            
        transit_depth_min_plt = 0.995*transit_depth_min_plt  # Extend slightly below
        
    else:
        transit_depth_min_plt = transit_depth_min

    # If the user did not specify a transit depth range, find min and max from input models
    if (transit_depth_max == None):
        
        transit_depth_max_plt = 1e-10  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):
            
            transit_depth_max_i = np.max(spectra[i][0])
            transit_depth_max_plt = max(transit_depth_max_plt, transit_depth_max_i)
            
        # Check if the highest data point falls above the current y-limit
        if (show_data == True):
            if (transit_depth_max_plt < max(ydata + err_data)):
                
                transit_depth_max_plt = max(ydata + err_data)
            
        transit_depth_max_plt = 1.005*transit_depth_max_plt  # Extend slightly above
        
    else:
        transit_depth_max_plt = transit_depth_max
        
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
    ymajor_spacing = round_sig_figs((transit_depth_max_plt - transit_depth_min_plt), 1)/10
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
    if (transit_depth_min == None):
        transit_depth_min_plt = np.floor(transit_depth_min_plt/ymajor_spacing)*ymajor_spacing
    if (transit_depth_max == None):
        transit_depth_max_plt = np.ceil(transit_depth_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    transit_depth_range = [transit_depth_min_plt, transit_depth_max_plt]
    
    # Place planet name in top left corner
    planet_name_x_position = 0.008*(wl_range[1]-wl_range[0]) + wl_range[0]
    planet_name_y_position = (0.92*(transit_depth_range[1]-transit_depth_range[0]) + 
                                    transit_depth_range[0])

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
    
    ax1 = plt.gca()
    
    # Set x axis to be logarithmic by default
    ax1.set_xscale("log")

    # Assign formatter objects to axes
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    ax1.yaxis.set_minor_locator(yminorLocator)
    
  #  ax2 = ax1.twinx()
 
  #  ax2.yaxis.set_major_locator(ymajorLocator_H)
  #  ax2.yaxis.set_major_formatter(ymajorFormatter_H)
  #  ax2.yaxis.set_minor_locator(yminorLocator_H)
    
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
            ax1.plot(wl, spec, lw=0.5, alpha=0.4, zorder=i,
                     color=colours[i], label=label_i)

        # Plot smoothed (binned) version of the model
        if (bin_spectra == True):
            
            N_plotted_binned = 0  # Counter for number of plotted binned spectra
            
            # Calculate binned wavelength and spectrum grid
            wl_binned, spec_binned = bin_spectrum_fast(wl, spec, R_to_bin)

            # Plot binned spectrum
            ax1.plot(wl_binned, spec_binned, lw=1.0, alpha=0.8, 
                     color=scale_lightness(colours[i], 0.4), 
                     zorder=N_spectra+N_plotted_binned, 
                     label=label_i + ' (R = ' + str(R_to_bin) + ')')
            
            N_plotted_binned += 1

    # Overplot datapoints
    if (show_data == True):
        markers, caps, bars = ax1.errorbar(wl_data, ydata, yerr=err_data, 
                                           xerr=bin_size, marker='o', 
                                           markersize=3, capsize=2, 
                                           ls='none', color='orange', 
                                           elinewidth=0.8, ecolor = 'black', 
                                           alpha=0.8, label=r'Data') 
        [markers.set_alpha(1.0)]
        
    # Overplot a particular model, binned to resolution of the observations
   # if (show_ymodel == True):
   #     ax1.scatter(wl_data, ymodel, color = 'gold', s=5, marker='D', 
   #                 lw=0.1, alpha=0.8, edgecolor='black', label=r'Binned Model')
    
    # Set axis ranges
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    # Add planet name label
    ax1.text(planet_name_x_position, planet_name_y_position, planet_name, fontsize = 16)
   # ax1.text(planet_name_x_position, planet_name_y_position, model_name, fontsize = 16)


    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
    elif (wl_max <= 2.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
    elif (wl_max <= 3.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 3)+0.01, 0.5)
        wl_ticks_3 = np.array([])
    else:
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        
    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3))
    
    # Plot wl tick labels
    ax1.set_xticks(wl_ticks)
    
    # Compute equivalent scale height for secondary axis
 #   base_depth = (R_p*R_p)/(R_s*R_s)
    
    #photosphere_T = 560.0
 #   photosphere_T = T_eq
    
#    H_sc = (sc.k*photosphere_T)/(mu*g_0)
#    depth_values = np.array([transit_depth_range[0], transit_depth_range[1]])
#    N_sc = ((depth_values - base_depth)*(R_s*R_s))/(2.0*R_p*H_sc)
    
#    ax2.set_ylim([N_sc[0], N_sc[1]])
#    ax2.set_ylabel(r'$\mathrm{Scale \, \, Heights}$', fontsize = 16)
    
    legend = ax1.legend(loc='upper right', shadow=True, prop={'size':10}, 
                        ncol=1, frameon=True)    #legend settings
  #  legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
    frame = legend.get_frame()
    frame.set_facecolor('0.90') 
        
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)
    
    plt.tight_layout()

    
    # Write figure to file
    file_name = output_dir + model_name + '_transmission_spectra.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_data(data, planet, wl_min = None, wl_max = None, 
              transit_depth_min = None, transit_depth_max = None, 
              colour_list = [], data_labels = []):
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
        
    # Define colours for plotted spectra (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['orange', 'lime', 'cyan', 'magenta', 'brown', 'black']
    else:
        colours = colour_list
                
    # If the user did not specify a wavelength range, find min and max from input data
    if (wl_min == None):
        wl_min_plt = np.min(wl_data - 4*bin_size)  # Minimum at twice the bin width for the shortest wavelength data
    else:
        wl_min_plt = wl_min
 
    if (wl_max == None):
        wl_max_plt = np.max(wl_data + 4*bin_size)  # Maximum at twice the bin width for the longest wavelength data
    else:
        wl_max_plt = wl_max

    # Set x range
    wl_range = [wl_min_plt, wl_max_plt]
    
    # If the user did not specify a transit depth range, find min and max from input models
    if (transit_depth_min == None):
        transit_depth_min_plt = 0.995 * np.min(ydata - err_data) # Extend slightly below
    else:
        transit_depth_min_plt = transit_depth_min

    if (transit_depth_max == None):
        transit_depth_max_plt = 1.005 * np.max(ydata + err_data) # Extend slightly above
    else:
        transit_depth_max_plt = transit_depth_max
        
    #***** Format x and y ticks *****#

    # Create x formatting objects
    if (wl_max_plt < 1.0):    # If plotting over the optical range
        xmajorLocator = MultipleLocator(0.1)
        xminorLocator = MultipleLocator(0.02)
        
    else:                 # If plot extends into the infrared
        xmajorLocator = MultipleLocator(1.0)
        xminorLocator = MultipleLocator(0.1)
            
    xmajorFormatter = FormatStrFormatter('%g')
    xminorFormatter = NullFormatter()
    
    # Aim for 10 major y-axis labels
    ymajor_spacing = round_sig_figs((transit_depth_max_plt - transit_depth_min_plt), 1)/10
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
    if (transit_depth_min == None):
        transit_depth_min_plt = np.floor(transit_depth_min_plt/ymajor_spacing)*ymajor_spacing
    if (transit_depth_max == None):
        transit_depth_max_plt = np.ceil(transit_depth_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    transit_depth_range = [transit_depth_min_plt, transit_depth_max_plt]
    
    # Place planet name in top left corner
  #  planet_name_x_position = 0.008*(wl_range[1]-wl_range[0]) + wl_range[0]
  #  planet_name_y_position = (0.92*(transit_depth_range[1]-transit_depth_range[0]) + 
  #                                  transit_depth_range[0])

    # Create y formatting objects
    ymajorLocator   = MultipleLocator(ymajor_spacing)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
    yminorLocator = MultipleLocator(yminor_spacing)

    # Generate figure and axes
    fig = plt.figure()  
    
    ax1 = plt.gca()
    
    # Set x axis to be logarithmic by default
    ax1.set_xscale("log")

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
        
        # Find start and end indicies of dataset_i in dataset property arrays
        idx_start = data['len_data_idx'][i]
        idx_end = data['len_data_idx'][i+1]

        # Extract the ith dataset
        wl_data_i = wl_data[idx_start:idx_end]
        ydata_i = ydata[idx_start:idx_end]
        err_data_i = err_data[idx_start:idx_end]
        bin_size_i = bin_size[idx_start:idx_end]

        # Plot dataset
        markers, caps, bars = ax1.errorbar(wl_data_i, ydata_i, yerr=err_data_i, 
                                           xerr=bin_size_i, marker='o', 
                                           markersize=3, capsize=2, ls='none', 
                                           color=colours[i], elinewidth=0.8, 
                                           ecolor = 'black', alpha=0.8, 
                                           label=label_i) 
        [markers.set_alpha(1.0)]
            
    # Set axis ranges
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    # Add planet name label
  #  ax1.text(planet_name_x_position, planet_name_y_position, planet_name, fontsize = 16)

    # Decide at which wavelengths to place major tick labels
    if (wl_max_plt <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min_plt, 1), round_sig_figs(wl_max_plt, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
    elif (wl_max_plt <= 2.0):
        if (wl_min_plt < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min_plt, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max_plt, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
    elif (wl_max_plt <= 3.0):
        if (wl_min_plt < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min_plt, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max_plt, 2)+0.01, 0.5)
        wl_ticks_3 = np.array([])
    else:
        if (wl_min_plt < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min_plt, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max_plt, 2)+0.01, 1.0)
        
    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3))
    
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
    file_name = output_dir + planet_name + '_data.pdf'
    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1, 
                           spectra_high1, spectra_high2, planet_name, 
                           data_properties, R_to_bin = 100, 
                           show_ymodel = True, wl_min = None, wl_max = None, 
                           transit_depth_min = None, transit_depth_max = None, 
                           colour_list = [], spectra_labels = []):
    ''' 
    Plot retrieved transmission spectra.
    
    '''

    # Find number of spectra to plot
    N_spectra = len(spectra_median)

    # Identify output directory location where the plot will be saved
    output_dir = './POSEIDON_output/' + planet_name + '/plots/'

    # Unpack data properties (if provided)
    if (data_properties is not None):
        ydata = data_properties['ydata']
        err_data = data_properties['err_data']
        wl_data = data_properties['wl_data']
        bin_size = data_properties['half_bin']
         
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

    # Set x range
    wl_range = [wl_min, wl_max]
    
    # If the user did not specify a transit depth range, find min and max from input models
    if (transit_depth_min == None):
        
        transit_depth_min_plt = 1e10   # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):

            (spec_low2, wl) = spectra_low2[i]
            _, spec_low2_binned = bin_spectrum_fast(wl, spec_low2, R_to_bin)

            transit_depth_min_i = np.min(spec_low2_binned)
            transit_depth_min_plt = min(transit_depth_min_plt, transit_depth_min_i)
            
        # Check if the lowest data point falls below the current y-limit
        if (transit_depth_min_plt > min(ydata - err_data)):
            transit_depth_min_plt = min(ydata - err_data)
            
        transit_depth_min_plt = 0.995*transit_depth_min_plt  # Extend slightly below
        
    else:
        transit_depth_min_plt = transit_depth_min

    # If the user did not specify a transit depth range, find min and max from input models
    if (transit_depth_max == None):
        
        transit_depth_max_plt = 1e-10  # Dummy value
        
        # Loop over each model, finding the most extreme min / max range 
        for i in range(N_spectra):

            (spec_high2, wl) = spectra_high2[i]
            _, spec_high2_binned = bin_spectrum_fast(wl, spec_high2, R_to_bin)

            transit_depth_max_i = np.max(spec_high2_binned)
            transit_depth_max_plt = max(transit_depth_max_plt, transit_depth_max_i)
            
        # Check if the highest data point falls above the current y-limit
        if (transit_depth_max_plt < max(ydata + err_data)):
            transit_depth_max_plt = max(ydata + err_data)
            
        transit_depth_max_plt = 1.020*transit_depth_max_plt  # Extend slightly above
        
    else:
        transit_depth_max_plt = transit_depth_max

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
    ymajor_spacing = round_sig_figs((transit_depth_max_plt - transit_depth_min_plt), 1)/10
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
    if (transit_depth_min == None):
        transit_depth_min_plt = np.floor(transit_depth_min_plt/ymajor_spacing)*ymajor_spacing
    if (transit_depth_max == None):
        transit_depth_max_plt = np.ceil(transit_depth_max_plt/ymajor_spacing)*ymajor_spacing
 
    # Set y range
    transit_depth_range = [transit_depth_min_plt, transit_depth_max_plt]
    
    # Place planet name in top left corner
    planet_name_x_position = 0.008*(wl_range[1]-wl_range[0]) + wl_range[0]
    planet_name_y_position = (0.92*(transit_depth_range[1]-transit_depth_range[0]) + 
                                    transit_depth_range[0])

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
    
    ax1 = plt.gca()
    
    # Set x axis to be logarithmic by default
    ax1.set_xscale("log")

    # Assign formatter objects to axes
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    ax1.yaxis.set_minor_locator(yminorLocator)
    
  #  ax2 = ax1.twinx()
 
  #  ax2.yaxis.set_major_locator(ymajorLocator_H)
  #  ax2.yaxis.set_major_formatter(ymajorFormatter_H)
  #  ax2.yaxis.set_minor_locator(yminorLocator_H)
                         
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
        wl_binned, spec_med_binned = bin_spectrum_fast(wl, spec_med, R_to_bin)
        wl_binned, spec_low1_binned = bin_spectrum_fast(wl, spec_low1, R_to_bin)
        wl_binned, spec_low2_binned = bin_spectrum_fast(wl, spec_low2, R_to_bin)
        wl_binned, spec_high1_binned = bin_spectrum_fast(wl, spec_high1, R_to_bin)
        wl_binned, spec_high2_binned = bin_spectrum_fast(wl, spec_high2, R_to_bin)
        
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
                        s=5, marker='D', lw=0.1, alpha=0.8, zorder=100, edgecolor='black',
                        label = label_i + r' (Binned)')
            
    # Overplot datapoints
    markers, caps, bars = ax1.errorbar(wl_data, ydata, yerr=err_data, xerr=bin_size, 
                                       marker='o', markersize=3, capsize=2, 
                                       ls='none', color='black', elinewidth=0.8, 
                                       ecolor='black', alpha=0.8, label=r'Data') 
    [markers.set_alpha(1.0)]
    
    # Set axis ranges
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    # Add planet name label
    ax1.text(planet_name_x_position, planet_name_y_position, planet_name, fontsize = 16)
   # ax1.text(planet_name_x_position, planet_name_y_position, model_name, fontsize = 16)


    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
    elif (wl_max <= 2.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
    elif (wl_max <= 3.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 3)+0.01, 0.5)
        wl_ticks_3 = np.array([])
    else:
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        
    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3))
    
    # Plot wl tick labels
    ax1.set_xticks(wl_ticks)
    
    # Compute equivalent scale height for secondary axis
 #   base_depth = (R_p*R_p)/(R_s*R_s)
    
 #   photosphere_T = T_eq
    
#    H_sc = (sc.k*photosphere_T)/(mu*g_0)
#    depth_values = np.array([transit_depth_range[0], transit_depth_range[1]])
#    N_sc = ((depth_values - base_depth)*(R_s*R_s))/(2.0*R_p*H_sc)
    
#    ax2.set_ylim([N_sc[0], N_sc[1]])
#    ax2.set_ylabel(r'$\mathrm{Scale \, \, Heights}$', fontsize = 16)

    legend = ax1.legend(loc='upper right', shadow=True, prop={'size':10}, 
                        ncol=1, frameon=False)    #legend settings
  #  legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
  #  frame = legend.get_frame()
  #  frame.set_facecolor('0.90') 
            
    plt.tight_layout()


    # Write figure to file
    file_name = output_dir + planet_name + '_retrieved_spectra.pdf'

    plt.savefig(file_name, bbox_inches='tight')

    return fig


def plot_stellar_flux(Flux, wl):
    
    fig = plt.figure()  
        
    ax = plt.gca()

    ax.set_xscale("log")

    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.plot(wl, Flux, lw=1, alpha=0.8, label=r'Stellar Flux')

    ax.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax.set_ylabel(r'Surface Flux (W m$^{-2}$ m$^{-1}$)', fontsize = 16)

    ax.set_xlim([min(wl), max(wl)])

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
    wl_binned, FpFs_binned = bin_spectrum_fast(wl, FpFs, R_to_bin)

    # Plot binned spectrum
    ax.plot(wl_binned, FpFs_binned, lw=1.0, alpha=0.8, 
                color=scale_lightness('crimson', 0.4),
                label='Flux Ratio' + ' (R = ' + str(R_to_bin) + ')')

    # Decide at which wavelengths to place major tick labels
    wl_min = min(wl)
    wl_max = max(wl)

    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
    elif (wl_max <= 2.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
    elif (wl_max <= 3.0):
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 3)+0.01, 0.5)
        wl_ticks_3 = np.array([])
    else:
        if (wl_min < 1.0):
            wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0, 0.2)
        else:
            wl_ticks_1 = np.array([])
        wl_ticks_2 = np.arange(1.0, 3.0, 0.5)
        wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        
    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3))
    
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