# *****visuals.py - various plotting routines to visualise POSEIDON output*****

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
from matplotlib.ticker import MultipleLocator, AutoLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter, NullFormatter
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('classic')
plt.rc('font', family='serif')
matplotlib.rcParams['svg.fonttype'] = 'none'

from config import R_p, R_s, planet_name, b_p, g_0, R, T_eq, load_observations
from utility import confidence_intervals, bin_spectrum_fast, closest_index, \
                    generate_latex_param_names, round_sig_figs
from atmosphere import compute_mean_mol_mass, compute_metallicity, compute_C_to_O, \
                       compute_O_to_H, compute_C_to_H, compute_N_to_H
from absorption import H_minus_bound_free, H_minus_free_free
              
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


def plot_transit_NEW(ax, r, T, phi, phi_edge, dphi, theta, theta_edge, dtheta, 
                     perspective):
    
#    N_sectors = len(phi)
#    N_zones = len(theta)
    
    # Specify planet impact paramter and 
 #   b_p = 0.0*R_s
 #   y_p = -0.7*R_s
    
    # Find distance between stellar and planetary centres
 #   d = np.sqrt(y_p**2 + b_p**2)/R_s
 

    
    ax.axis('equal')
    
    # First, average sectors / zones to find radii and temperatures for terminator plane and polar slices
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
        
        
    # Looking at nightside
    if (perspective == 'observer'):
    
        # Pick out radial extents for nightside zone
        r_night = r[:,:,-1]
        
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
                
            # Plot 9 layers (~ each decade in pressure)
            for i in range(0, r_night.shape[0]-11, 11):
                   
                # Plot full atmosphere for this sector
                planet_atm = Wedge((0.0, 0.0), r_night[i+11,j]/R_p, phi_prime_edge_all[j_all+1],  # Wedges plots w.r.t x axis and in degrees
                                   phi_prime_edge_all[j_all], edgecolor='None', 
                                   width = (r_night[i+11,j] - r_night[i,j])/R_p)
            
                patches.append(planet_atm)
                T_colors.append(0.5*(T[i,j,-1] + T[i+11,j,-1]))
            
            # Plot planet core (circular, below atmosphere)
            planet_core = Circle((0.0, 0.0), r[0,0,0]/R_p, facecolor='black', edgecolor='None')
            ax.add_artist(planet_core)
            
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])
        
        ax.set_title("Observer's Perspective", fontsize = 16, pad=10)
        
        ax.set_xlabel(r'y ($R_p$)', fontsize = 16)
        ax.set_ylabel(r'z ($R_p$)', fontsize = 16)
        
        # Plot atmosphere segment collection
        p = PatchCollection(patches, cmap=matplotlib.cm.RdYlBu_r, alpha=1.0, 
                            edgecolor=colorConverter.to_rgba('black', alpha=0.1), 
                            lw=0.1, zorder = 10)
        
        # Colour each segment according to atmospheric temperature
        colors = np.array(T_colors)
        p.set_array(colors)
   #     p.set_clim([np.min(T), np.max(T)])
        p.set_clim([600, 1400])

        ax.add_collection(p)
        
        ax.text(0.50, 1.3, 'Morning', fontsize = 14)
        ax.text(-1.20, 1.3, 'Evening', fontsize = 14)
        
  #      plt.colorbar(p)      
  
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
                
            # Plot 9 layers (~ each decade in pressure)
            for i in range(0, r_term.shape[0]-11, 11):
                   
                # Plot full atmosphere for this sector
                planet_atm = Wedge((0.0, 0.0), r_term[i+11,j]/R_p, phi_prime_edge_all[j_all+1],  # Wedges plots w.r.t x axis and in degrees
                                   phi_prime_edge_all[j_all], edgecolor='None', 
                                   width = (r_term[i+11,j] - r_term[i,j])/R_p)
            
                patches.append(planet_atm)
                T_colors.append(0.5*(T_term[i,j] + T_term[i+11,j]))
            
            # Plot planet core (circular, below atmosphere)
            planet_core = Circle((0.0, 0.0), r[0,0,0]/R_p, facecolor='black', edgecolor='None')
            ax.add_artist(planet_core)
            
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-1.8, 1.8])
        
        ax.set_title("Terminator Plane", fontsize = 16, pad=10)
        
        ax.set_xlabel(r'y ($R_p$)', fontsize = 16)
        ax.set_ylabel(r'z ($R_p$)', fontsize = 16)
        
        # Plot atmosphere segment collection
        p = PatchCollection(patches, cmap=matplotlib.cm.RdYlBu_r, alpha=1.0, 
                            edgecolor=colorConverter.to_rgba('black', alpha=0.1), 
                            lw=0.1, zorder = 10)
        
        # Colour each segment according to atmospheric temperature
        colors = np.array(T_colors)
        p.set_array(colors)
   #     p.set_clim([0.98*np.min(T_term), 1.02*np.max(T_term)])
        p.set_clim([600, 1400])

        ax.add_collection(p)
        
        ax.text(0.50, 1.2, 'Morning', fontsize = 14)
        ax.text(-1.40, 1.2, 'Evening', fontsize = 14)
        
  #      plt.colorbar(p)      
  
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
        
        # Plot star
   #     star = Circle((-y_p/R_p, -b_p/R_p), R_s/R_p, facecolor='gold', edgecolor='None', alpha=0.8)
   #     ax.add_artist(star)
        
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
                
            # Plot 9 layers (~ each decade in pressure)
            for i in range(0, r_pole.shape[0]-11, 11):
                
                planet_atm = Wedge((0.0, 0.0), r_pole[i+11,k]/R_p, theta_prime_edge_all[k_all+1],  # Wedges plots w.r.t x axis and in degrees
                                   theta_prime_edge_all[k_all], edgecolor='None', 
                                   width = (r_pole[i+11,k] - r_pole[i,k])/R_p)
            
                patches.append(planet_atm)
                T_colors.append(0.5*(T_pole[i,k] + T_pole[i+11,k]))

            # Plot planet core (circular, below atmosphere)
            planet_core = Circle((0.0, 0.0), r[0,0,0]/R_p, facecolor='black', edgecolor='None')
            ax.add_artist(planet_core)
            
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-1.8, 1.8])
        
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
   #     p.set_clim([0.98*np.min(T_term), 1.02*np.max(T_term)])
        p.set_clim([600, 1400])

        ax.add_collection(p)

        ax.text(0.90, 1.0, 'Night', fontsize = 14)
        ax.text(-0.37, 1.35, 'Terminator', fontsize = 14)
        ax.text(-1.20, 1.0, 'Day', fontsize = 14)
        
        ax.text(-1.05, 1.28, 'Star', fontsize = 14)
        ax.annotate(s='', xy=(-1.3, 1.2), xytext=(-0.6, 1.2), 
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.8))
        ax.text(0.68, 1.28, 'Observer', fontsize = 14)
        ax.annotate(s='', xy=(0.6, 1.2), xytext=(1.3, 1.2), 
                    arrowprops=dict(arrowstyle='<-', color='black', alpha=0.8))
   
   #     plt.colorbar(p)
   
    return p
    
   
def plot_geometry(r, T, phi, phi_edge, dphi, theta, theta_edge, dtheta,
                  plt_tag = None):
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # Plot observer's perspective on LHS axis
    p = plot_transit_NEW(ax1, r, T, phi, phi_edge, dphi, theta, theta_edge, dtheta, 
                     'terminator') 
 #   p = plot_transit_NEW(ax1, r, T, phi, phi_edge, dphi, theta, theta_edge, dtheta, 
 #                        'observer') 
    
    # Plot side perspective on RHS axis
    _ = plot_transit_NEW(ax2, r, T, phi, phi_edge, dphi, theta, theta_edge, dtheta, 
                     'day-night') 
    
    # Plot temperature colourbar
    cbaxes = fig.add_axes([1.01, 0.131, 0.015, 0.786]) 
    cb = plt.colorbar(p, cax = cbaxes)  
    tick_locator = ticker.MaxNLocator(nbins=8)
    cb.locator = tick_locator
    cb.update_ticks()
    
    plt.tight_layout()
    
    # Write figure to file
    if (plt_tag == None):
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_Geometry.png', bbox_inches='tight', dpi=800)
    else:
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_' + plt_tag + '_Geometry.png', bbox_inches='tight', dpi=800)

    
def plot_transit(y_p, r, ax):
    
    # Find number of sectors with different radial extent
    N_sectors = r.shape[1]

        
    N_phi = 36
    dphi = (2.0*np.pi)/N_phi
    #phi = (dphi * np.arange(N_phi)) + (dphi/2.0)
    #phi = np.append(0.0, phi, phi[0])
    
    # Find distance between stellar and planetary centres
    d = np.sqrt(y_p**2 + b_p**2)/R_s
    
    if (N_sectors == 2):
        
        phi_1 = (dphi * np.arange(N_phi/2)) + (dphi/2.0)
        phi_2 = (dphi * np.arange(N_phi/2)) + (dphi/2.0) + np.pi
        phi = np.concatenate(([0.0], phi_1, [np.pi], phi_2, [2.0*np.pi]))
        
        ax.axis('equal')
        
        phi_edge = np.array([0.0, np.pi, 2.0*np.pi])       # Boundaries between East and West
        phi_edge_deg = (phi_edge * (180.0/np.pi)) - 90.0   # Convert to degrees, rotate so w.r.t x axis
        
        # Plot star
 #       star = Circle((-y_p/R_p, -b_p/R_p), R_s/R_p, facecolor='gold', edgecolor='None', alpha=0.8)
        star = Circle((9.0, -b_p/R_p), R_s/R_p, facecolor='gold', edgecolor='None', alpha=0.8)
        ax.add_artist(star)
        
        for j in range(N_sectors):
            
            # Identify polar angles lying in this sector
            phi_sector = phi[((phi >= phi_edge[j]) & (phi <= phi_edge[j+1]))]
    
            # Compute x and y coordiates for each atmospheric grid point
            x = np.outer(r[:,j], np.sin(phi_sector))/R_p
            y = np.outer(r[:,j], np.cos(phi_sector))/R_p
            
            # Plot full atmosphere for this sector
            planet_atm = Wedge((0.0, 0.0), r[-1,j]/R_p, (phi_edge_deg[j]),  # Wedges plots w.r.t x axis and in degrees
                               (phi_edge_deg[j+1]), facecolor='cyan',      
                               edgecolor='None', alpha=0.4)

            planet_atm_edge = Wedge((0.0, 0.0), r[-1,j]/R_p, (phi_edge_deg[j]),  # Wedges plots w.r.t x axis and in degrees
                                    (phi_edge_deg[j+1]), facecolor='None',      
                                    edgecolor='grey', lw=0.1)                           

            ax.add_artist(planet_atm)
            ax.add_artist(planet_atm_edge)
            
            # Plot a selection of the atmospheric layer grid for this sector
            for i in range(0, r.shape[0], 11):
                
                ax.plot(x[i,:], y[i,:], color='blue', ls='-', lw=0.1, zorder=5)
                ax.scatter(x[i,1:-1], y[i,1:-1], color='crimson', s=0.1, zorder=10)
            
            # Plot planet core (circular, below atmosphere)
            planet_core = Circle((0.0, 0.0), r[0,0]/R_p, facecolor='black', edgecolor='None')
            ax.add_artist(planet_core)
        
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])
            
   #     if (d >= 0.8):
   #         ax.set_xlim([-2.0, 2.0])
   #         ax.set_ylim([-2.0, 2.0])
        
   #     else:
   #         ax.set_xlim([-10.0, 10.0])
   #         ax.set_ylim([-10.0, 10.0])
        
        ax.set_xlabel(r'x ($R_p$)', fontsize = 16)
        ax.set_ylabel(r'y ($R_p$)', fontsize = 16)
        
        if (d >= 0.8):
            ax.text(0.85, 1.5, r'd =' + '%0.3f' % d + r'$R_{*}$', fontsize = 14)
            
        else:
            ax.text(0.85, 1.5, r'd =' + '%0.3f' % d + r'$R_{*}$', fontsize = 14)
    
    elif (N_sectors == 1):
        
        phi = (dphi * np.arange(N_phi)) + (dphi/2.0)
        phi = np.append(phi, phi[0])
        
        r_atm = r[:,0]
    
        x = np.outer(r_atm, np.sin(phi))/R_p
        y = np.outer(r_atm, np.cos(phi))/R_p
        
        ax.axis('equal')
        
        star = plt.Circle((-y_p/R_p, -b_p/R_p), R_s/R_p, facecolor='gold', edgecolor='None', alpha=0.8)
        planet_core = plt.Circle((0.0, 0.0), r_atm[0]/R_p, facecolor='black', edgecolor='None')
        planet_atm = plt.Circle((0.0, 0.0), r_atm[-1]/R_p, facecolor='cyan', edgecolor='None', alpha=0.4)
        planet_atm_edge = plt.Circle((0.0, 0.0), r_atm[-1]/R_p, facecolor='None', edgecolor='blue', lw=0.1)
        
        ax.add_artist(star)
        ax.add_artist(planet_atm)
        ax.add_artist(planet_atm_edge)
        ax.add_artist(planet_core)
        
        d = np.sqrt(y_p**2 + b_p**2)/R_s
            
        for i in range(0, len(r_atm), 11):
            
            ax.plot(x[i,:], y[i,:], color='blue', ls='-', lw=0.1, zorder=5)
 #           ax.scatter(x[i,:], y[i,:], color='crimson', s=0.1, zorder=10)
        
        if (d >= 0.8):
            ax.set_xlim([-2.0, 2.0])
            ax.set_ylim([-2.0, 2.0])
        
        else:
            ax.set_xlim([-10.0, 10.0])
            ax.set_ylim([-10.0, 10.0])
        
        ax.set_xlabel(r'x ($R_p$)', fontsize = 16)
        ax.set_ylabel(r'y ($R_p$)', fontsize = 16)
        
        if (d >= 0.8):
            ax.text(0.85, 1.5, r'd =' + '%0.3f' % d + r'$R_{*}$', fontsize = 14)
            
        else:
            ax.text(4.25, 7.5, r'd =' + '%0.3f' % d + r'$R_{*}$', fontsize = 14)
    

def plot_spectrum_time_resolved(spectrum, ymodel, ydata, err_data, bin_size, wl_data, wl, mu, r, y_p, i):
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1]) 
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    #***** Plot transit on RHS axis first *****
    plot_transit(y_p, r, ax2)
    
    #***** Now plot spectrum *****
    wl_um = wl   # Wavelengths in microns
    
    wl_range = [0.4, 5.0]
    transit_depth_range = [1.40e-2, 1.80e-2]
    
    planet_name_x_position = 0.008*(wl_range[1]-wl_range[0]) + wl_range[0]
    planet_name_y_position = 0.94*(transit_depth_range[1]-transit_depth_range[0]) + transit_depth_range[0]
    
    # Tick formatting
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()
    ymajorLocator   = MultipleLocator(0.0005)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
    yminorLocator   = MultipleLocator(0.0001)

    # Generate figure and axes
    #fig = plt.figure()  
    
    ax1.set_xscale("log")

    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_major_formatter(xmajorFormatter)
    ax1.xaxis.set_minor_locator(xminorLocator)
    ax1.xaxis.set_minor_formatter(xminorFormatter)
    ax1.yaxis.set_major_locator(ymajorLocator)
    ax1.yaxis.set_major_formatter(ymajorFormatter)
    ax1.yaxis.set_minor_locator(yminorLocator)
    
#    wl_R100, spectrum_R100 = bin_spectrum_fast(wl, spectrum, wl_range[0], wl_range[-1], 100.0)
#    wl_R100_um = wl_R100
    
    ax1.plot(wl_um, spectrum, lw=1.0, alpha=0.4, color = 'green', zorder=3, label=r'POSEIDON (line-by-line)')  
    
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    ax1.set_xlabel(r'Wavelength ' + r'(μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    ax1.text(planet_name_x_position, planet_name_y_position, r'HD 209458b', fontsize = 14)
    #ax1.text(1.05*planet_name_x_position, 0.996*planet_name_y_position, r'y =' + str(y_p/R_s) + r'$R_{*}$', fontsize = 14)
    ax1.set_xticks([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    
    legend = ax1.legend(loc='lower right', shadow=True, prop={'size':10}, ncol=1, frameon=False)    #legend settings
        
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)
    
    plt.tight_layout()
    
 #   plt.savefig('../../output/plots/Transit_animation/' + planet_name + '_INGRESS_y' + '%0.3f' % (y_p/R_s) + '.png', bbox_inches='tight', fmt='pdf', dpi=300)
 #   plt.savefig('../../output/plots/Transit_animation/' + planet_name + '_Animation_frame_' + str(i) + '.png', bbox_inches='tight', fmt='pdf', dpi=500)
    plt.savefig('../../output/plots/' + planet_name + '_' + str(i) + '.png', bbox_inches='tight', fmt='png', dpi=400)
 
#    plt.close()


def plot_spectra(spectra, ymodel, wl_data, ydata, err_data, bin_size,
                 plot_full_res = True, bin_spectra = True, R_bin = 100, 
                 show_ymodel = False, wl_min = None, wl_max = None, 
                 transit_depth_min = None, transit_depth_max = None, 
                 colour_list = [], model_labels = [], plt_tag = None):
    
    ''' Plot a collection of individual model transmission spectra.
    
    '''
    
    # Find number of spectra to plot
    N_spectra = len(spectra)
 
    # Format planet name
    if ('HD' in planet_name):
        planet_tag = planet_name[:2] + ' ' + planet_name[2:]
    elif ('HAT' in planet_name):
        planet_tag = planet_name[:3] + '-P-' + planet_name[4:]
    elif ('WASP' in planet_name):
        planet_tag = planet_name[:4] + '-' + planet_name[4:]
    else:
        planet_tag = planet_name
        
    # Quick validity checks for plotting
    if (N_spectra == 0):
        raise Exception("Must provide at least one spectrum to plot!")
    if (N_spectra > 6):
        raise Exception("Max number of concurrent spectra to plot is 6.")
    if ((colour_list != []) and (N_spectra != len(colour_list))):
        raise Exception("Number of colours does not match number of spectra.")
    if ((model_labels != []) and (N_spectra != len(model_labels))):
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
        if (load_observations == True):
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
        if (load_observations == True):
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
        if (model_labels == []):
            label_i = r'Model ' + str(i)
        else:
            label_i = model_labels[i]
        
        # Plot spectrum at full model resolution
        if (plot_full_res == True):
            ax1.plot(wl, spec, lw=0.5, alpha=0.4, zorder=i,
                     color=colours[i], label=label_i)

        # Plot smoothed (binned) version of the model
        if (bin_spectra == True):
            
            N_plotted_binned = 0  # Counter for number of plotted binned spectra
            
            # Calculate binned wavelength and spectrum grid
            wl_binned, spec_binned = bin_spectrum_fast(wl, spec, R_bin)

            # Plot binned spectrum
            ax1.plot(wl_binned, spec_binned, lw=1.0, alpha=0.8, 
                     color=scale_lightness(colours[i], 0.4), 
                     zorder=N_spectra+N_plotted_binned, 
                     label=label_i + ' (R = ' + str(R_bin) + ')')
            
            N_plotted_binned += 1

    # Overplot datapoints
    if (load_observations == True):
        markers, caps, bars = ax1.errorbar(wl_data, ydata, yerr=err_data, xerr=0, 
                                           marker='o', markersize=3, capsize=2, 
                                           ls='none', color='orange', 
                                           elinewidth=0.8, ecolor = 'black', 
                                           alpha=0.8, label=r'Sim. data') 
        [markers.set_alpha(1.0)]
        
    # Overplot a particular model, binned to resolution of the observations
    if (show_ymodel == True):
        ax1.scatter(wl_data, ymodel, color = 'gold', s=5, marker='D', 
                    lw=0.1, alpha=0.8, edgecolor='black', label=r'Binned Model')
    
    # Set axis ranges
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    # Add planet name label
 #   ax1.text(planet_name_x_position, planet_name_y_position, planet_tag, fontsize = 16)
    ax1.text(planet_name_x_position, planet_name_y_position, '2D Warm Neptune', fontsize = 16)


    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
    elif (wl_max <= 2.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
    elif (wl_max <= 3.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.5)
        wl_ticks_3 = np.array([])
    else:
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, 3.0+0.01, 0.5)
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
    
    legend = ax1.legend(loc='lower right', shadow=True, prop={'size':10}, ncol=1, frameon=False)    #legend settings
    #legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
    #frame = legend.get_frame()
    #frame.set_facecolor('0.90') 
        
 #   for legline in legend.legendHandles:
 #       legline.set_linewidth(1.0)
    
    plt.tight_layout()
    
    # Write figure to file
    if (plt_tag == None):
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_spectrum.pdf', bbox_inches='tight', dpi=500)
    else:
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_' + plt_tag + '_spectrum.pdf', bbox_inches='tight', dpi=500)
    
    
def plot_evening_morning_opening_angle(spectrum_1, spectrum_2, spectrum_3, spectrum_4, 
                                       spectrum_5, spectrum_6, spectrum_7, spectrum_8,
                                       spectrum_9, spectrum_10, ymodel, 
                                       ydata, err_data, bin_size, wl_data, wl, wl_2, mu):
     
    wl_um = wl   # Wavelengths in microns
    
    wl_range = [min(wl_um), max(wl_um)]
    transit_depth_range = [1.38e-2, 1.68e-2]
    
    planet_name_x_position = 0.008*(wl_range[1]-wl_range[0]) + wl_range[0]
    planet_name_y_position = 0.92*(transit_depth_range[1]-transit_depth_range[0]) + transit_depth_range[0]
    
    # Tick formatting
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%g')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()

    ymajorLocator   = MultipleLocator(0.0002)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
    yminorLocator   = MultipleLocator(0.00004)

    ymajorLocator_H   = MultipleLocator(1)
    ymajorFormatter_H = FormatStrFormatter('%.0f')
    yminorLocator_H   = MultipleLocator(0.2)

    # Generate figure and axes
    fig = plt.figure()  
    
    ax1 = plt.gca()
    
    ax1.set_xscale("log")

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
    
    R_smooth = 100.0
    wl_smooth_min = 0.41
    wl_smooth_max = 5.1
    
    wl_R100, spec_1_R100 = bin_spectrum_fast(wl, spectrum_1, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_2_R100 = bin_spectrum_fast(wl, spectrum_2, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_3_R100 = bin_spectrum_fast(wl, spectrum_3, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_4_R100 = bin_spectrum_fast(wl, spectrum_4, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_5_R100 = bin_spectrum_fast(wl, spectrum_5, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_6_R100 = bin_spectrum_fast(wl, spectrum_6, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_7_R100 = bin_spectrum_fast(wl, spectrum_7, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_8_R100 = bin_spectrum_fast(wl, spectrum_8, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_9_R100 = bin_spectrum_fast(wl, spectrum_9, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_10_R100 = bin_spectrum_fast(wl, spectrum_10, wl_smooth_min, wl_smooth_max, R_smooth)
    
    print(np.max(np.abs(spec_8_R100 - spec_1_R100))*1.0e6)
    
    wl_R100_um = wl_R100
    
#    ax1.plot(wl_um, spectrum_1, lw=0.5, alpha=0.4, color = 'green', zorder=3, label=r'Demo spectrum')
#    ax1.plot(wl_um, spectrum_8, lw=0.5, alpha=0.4, color = 'red', zorder=3, label=r'Validate') 
    ax1.plot(wl_R100_um, spec_1_R100, lw=1.0, alpha=0.8, color = 'darkgreen', zorder=10, label=r'$\alpha = 0^{\circ}$')
    ax1.plot(wl_R100_um, spec_2_R100, lw=1.0, alpha=0.8, color = 'crimson', zorder=3, label=r'$\alpha = 5^{\circ}$')
    ax1.plot(wl_R100_um, spec_3_R100, lw=1.0, alpha=0.8, color = 'darkblue', zorder=3, label=r'$\alpha = 10^{\circ}$')
    ax1.plot(wl_R100_um, spec_4_R100, lw=1.0, alpha=0.8, color = 'black', zorder=3, label=r'$\alpha = 20^{\circ}$')
    ax1.plot(wl_R100_um, spec_5_R100, lw=1.0, alpha=0.8, color = 'brown', zorder=3, label=r'$\alpha = 40^{\circ}$')
    ax1.plot(wl_R100_um, spec_6_R100, lw=1.0, alpha=0.8, color = 'darkgrey', zorder=3, label=r'$\alpha = 80^{\circ}$')
    ax1.plot(wl_R100_um, spec_7_R100, lw=1.0, alpha=0.8, color = 'magenta', zorder=3, label=r'$\alpha = 120^{\circ}$')
    ax1.plot(wl_R100_um, spec_8_R100, lw=1.0, alpha=0.8, color = 'purple', zorder=3, label=r'$\alpha = 180^{\circ}$')
    ax1.plot(wl_R100_um, spec_9_R100, lw=2.0, alpha=0.8, color = 'royalblue', ls=':', zorder=9, label=r'1D average')
    ax1.plot(wl_R100_um, spec_10_R100, lw=2.0, alpha=0.8, color = 'darkgoldenrod', ls=':', zorder=9, label=r'1D log(average)')

    if (load_observations == True):
        ax1.errorbar(wl_data, ydata, yerr = err_data, xerr = 0, marker='o', markersize=2, capsize=2, zorder=12, ls='none', color='black', elinewidth=0.6, alpha=0.8, label=r'Synthetic data') 
    
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    ax1.set_xlabel(r'Wavelength ' + r'(μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    ax1.text(planet_name_x_position, planet_name_y_position, r'HD 209458b', fontsize = 16)
    ax1.text(1.01*planet_name_x_position, 0.990*planet_name_y_position, r'Variable Morning-Evening Terminator Opening Angle', fontsize = 14)
    ax1.text(1.02*planet_name_x_position, 0.980*planet_name_y_position, r'Abundance Gradient', fontsize = 14)

    ax1.set_xticks([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5])
    
    # Compute equivalent scale height for secondary axis
    base_depth = (R_p*R_p)/(R_s*R_s)
    
    #photosphere_T = 560.0
    photosphere_T = T_eq
    
#    H_sc = (sc.k*photosphere_T)/(mu*g_0)
#    depth_values = np.array([transit_depth_range[0], transit_depth_range[1]])
#    N_sc = ((depth_values - base_depth)*(R_s*R_s))/(2.0*R_p*H_sc)
    
#    ax2.set_ylim([N_sc[0], N_sc[1]])
#    ax2.set_ylabel(r'$\mathrm{Scale \, \, Heights}$', fontsize = 16)
    
    legend = ax1.legend(loc='lower right', shadow=True, prop={'size':10}, ncol=5, frameon=False)    #legend settings
    #legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
    #frame = legend.get_frame()
    #frame.set_facecolor('0.90') 
        
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)
    
    plt.tight_layout()
    
    plt.savefig('../../output/plots/' + planet_name + '_spectrum_morning_evening_demo.pdf', bbox_inches='tight', fmt='pdf', dpi=500)   
    

def plot_day_night_opening_angle(spectrum_1, spectrum_2, spectrum_3, spectrum_4, 
                                 spectrum_5, spectrum_6, spectrum_7, spectrum_8,
                                 spectrum_9, spectrum_10, ymodel, 
                                 ydata, err_data, bin_size, wl_data, wl, wl_2, mu):
     
    wl_um = wl   # Wavelengths in microns
    
    wl_range = [min(wl_um), max(wl_um)]
    transit_depth_range = [1.38e-2, 1.70e-2]
    
    planet_name_x_position = 0.008*(wl_range[1]-wl_range[0]) + wl_range[0]
    planet_name_y_position = 0.92*(transit_depth_range[1]-transit_depth_range[0]) + transit_depth_range[0]
    
    # Tick formatting
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%g')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()

    ymajorLocator   = MultipleLocator(0.0002)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
    yminorLocator   = MultipleLocator(0.00004)

    ymajorLocator_H   = MultipleLocator(1)
    ymajorFormatter_H = FormatStrFormatter('%.0f')
    yminorLocator_H   = MultipleLocator(0.2)

    # Generate figure and axes
    fig = plt.figure()  
    
    ax1 = plt.gca()
    
    ax1.set_xscale("log")

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
    
    R_smooth = 100.0
    wl_smooth_min = 0.41
    wl_smooth_max = 5.1
    
    wl_R100, spec_1_R100 = bin_spectrum_fast(wl, spectrum_1, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_2_R100 = bin_spectrum_fast(wl, spectrum_2, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_3_R100 = bin_spectrum_fast(wl, spectrum_3, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_4_R100 = bin_spectrum_fast(wl, spectrum_4, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_5_R100 = bin_spectrum_fast(wl, spectrum_5, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_6_R100 = bin_spectrum_fast(wl, spectrum_6, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_7_R100 = bin_spectrum_fast(wl, spectrum_7, wl_smooth_min, wl_smooth_max, R_smooth)
    wl_R100, spec_8_R100 = bin_spectrum_fast(wl, spectrum_8, wl_smooth_min, wl_smooth_max, R_smooth)
    
    print(np.max(np.abs(spec_7_R100 - spec_1_R100))*1.0e6)
    
    wl_R100_um = wl_R100
    
 #   ax1.plot(wl_um, spectrum_1, lw=0.5, alpha=0.4, color = 'green', zorder=3, label=r'Demo spectrum')
    ax1.plot(wl_R100_um, spec_1_R100, lw=1.0, alpha=0.8, color = 'darkgreen', zorder=10, label=r'$\beta = 0^{\circ}$')
 #   ax1.plot(wl_um, spectrum_2, lw=0.5, alpha=0.4, color = 'red', zorder=3, label=r'Validate') 
    ax1.plot(wl_R100_um, spec_2_R100, lw=1.0, alpha=0.8, color = 'crimson', zorder=3, label=r'$\beta = 5^{\circ}$')
    ax1.plot(wl_R100_um, spec_3_R100, lw=1.0, alpha=0.8, color = 'darkblue', zorder=3, label=r'$\beta = 10^{\circ}$')
    ax1.plot(wl_R100_um, spec_4_R100, lw=1.0, alpha=0.8, color = 'black', zorder=3, label=r'$\beta = 20^{\circ}$')
    ax1.plot(wl_R100_um, spec_5_R100, lw=1.0, alpha=0.8, color = 'brown', zorder=3, label=r'$\beta = 40^{\circ}$')
    ax1.plot(wl_R100_um, spec_6_R100, lw=1.0, alpha=0.8, color = 'darkgrey', zorder=3, label=r'$\beta = 80^{\circ}$')
    ax1.plot(wl_R100_um, spec_7_R100, lw=1.0, alpha=0.8, color = 'magenta', zorder=3, label=r'$\beta = 120^{\circ}$')
    ax1.plot(wl_R100_um, spec_8_R100, lw=0.8, alpha=1.0, color = 'cyan', ls=':', zorder=11, label=r'1D terminator average')
  #  ax1.plot(wl_R100_um, spec_8_R100, lw=0.5, alpha=0.8, color = 'gold', ls=':', zorder=11, label=r'1D terminator average')

    if (load_observations == True):
        ax1.errorbar(wl_data, ydata, yerr = err_data, xerr = 0, marker='o', markersize=2, capsize=2, zorder=12, ls='none', color='black', elinewidth=0.6, alpha=0.8, label=r'Synthetic data') 
    
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    ax1.set_xlabel(r'Wavelength ' + r'(μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    ax1.text(planet_name_x_position, planet_name_y_position, r'HD 209458b', fontsize = 16)
    ax1.text(1.01*planet_name_x_position, 0.990*planet_name_y_position, r'Variable Day-Night Terminator Opening Angle', fontsize = 14)
#    ax1.text(1.02*planet_name_x_position, 0.980*planet_name_y_position, r'Uniform Abundances', fontsize = 14)
    ax1.text(1.02*planet_name_x_position, 0.980*planet_name_y_position, r'Abundance Gradient', fontsize = 14)

    ax1.set_xticks([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5])
    
    # Compute equivalent scale height for secondary axis
    base_depth = (R_p*R_p)/(R_s*R_s)
    
    #photosphere_T = 560.0
    photosphere_T = T_eq
    
#    H_sc = (sc.k*photosphere_T)/(mu*g_0)
#    depth_values = np.array([transit_depth_range[0], transit_depth_range[1]])
#    N_sc = ((depth_values - base_depth)*(R_s*R_s))/(2.0*R_p*H_sc)
    
#    ax2.set_ylim([N_sc[0], N_sc[1]])
#    ax2.set_ylabel(r'$\mathrm{Scale \, \, Heights}$', fontsize = 16)
    
    legend = ax1.legend(loc='lower right', shadow=True, prop={'size':10}, ncol=4, frameon=False)    #legend settings
    #legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
    #frame = legend.get_frame()
    #frame.set_facecolor('0.90') 
        
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)
    
    plt.tight_layout()
    
    #plt.savefig('../../output/plots/Transmission_R300.svg', bbox_inches='tight', fmt='svg', dpi=1000)
    
    #plt.savefig('../../output/plots/' + planet_name + '_Spectrum_VIS+IR.svg', bbox_inches='tight', fmt='svg', dpi=1000)
    #plt.savefig('../../output/plots/' + planet_name + '_Validate_spectrum_2.svg', bbox_inches='tight', fmt='pdf', dpi=1000)
    plt.savefig('../../output/plots/' + planet_name + '_spectrum_day_night_demo.pdf', bbox_inches='tight', fmt='pdf', dpi=500)   
    
    
def plot_spectrum_chem_removed(spectrum_1, spectrum_2, spectrum_3, spectrum_4, 
                               spectrum_5, spectrum_6, spectrum_7, spectrum_8,
                               spectrum_9, spectrum_10, spectrum_11, spectrum_12,
                               ymodel, ydata, err_data, bin_size, wl_data, wl, mu):
     
    wl_um = wl   # Wavelengths in microns  
    
    wl_range = [0.385, 5.15]
    transit_depth_range = [1.21e-2, 1.45e-2]
    
    mode_arrow_y = 1.215e-2
    mode_label_y = 1.22e-2
        
    planet_name_x_position = 0.008*(wl_range[1]-wl_range[0]) + wl_range[0]
    planet_name_y_position = 0.92*(transit_depth_range[1]-transit_depth_range[0]) + transit_depth_range[0]
    
    # Tick formatting
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()
    ymajorLocator   = MultipleLocator(0.0002)
    ymajorFormatter = ScalarFormatter(useMathText=True)
    ymajorFormatter.set_powerlimits((0,0))
    yminorLocator   = MultipleLocator(0.00002)

    ymajorLocator_H   = MultipleLocator(2)
    ymajorFormatter_H = FormatStrFormatter('%.0f')
    yminorLocator_H   = MultipleLocator(0.2)

    # Generate figure and axes
    fig = plt.figure()  
    fig.set_size_inches(8.27, 4.651875)
    
    ax1 = plt.gca()
    
    ax1.set_xscale("log")

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
    
    # Bin high resolution down to low resolution
    R_plt = 100.0
    
    wl_plt, spec_1_plt = bin_spectrum_fast(wl, spectrum_1, wl_range[0], wl_range[1], R_plt)
    wl_plt, spec_2_plt = bin_spectrum_fast(wl, spectrum_2, wl_range[0], wl_range[1], R_plt)
    wl_plt, spec_3_plt = bin_spectrum_fast(wl, spectrum_3, wl_range[0], wl_range[1], R_plt)
    wl_plt, spec_4_plt = bin_spectrum_fast(wl, spectrum_4, wl_range[0], wl_range[1], R_plt)
    wl_plt, spec_5_plt = bin_spectrum_fast(wl, spectrum_5, wl_range[0], wl_range[1], R_plt)
    wl_plt, spec_6_plt = bin_spectrum_fast(wl, spectrum_6, wl_range[0], wl_range[1], R_plt)

    wl_plt_um = wl_plt
     
    ax1.plot(wl_plt_um, spec_1_plt, lw=4.0, alpha=0.4, color = 'green', zorder=1, label=r'Best-fit Model') 
    ax1.plot(wl_plt_um, spec_5_plt, lw=1.5, alpha=0.8, color = 'grey', zorder=5, label=r'Faculae')  
    ax1.plot(wl_plt_um, spec_2_plt, lw=1.5, alpha=0.8, color = 'crimson', zorder=4, label=r'TiO + faculae')  
    ax1.plot(wl_plt_um, spec_3_plt, lw=1.5, alpha=0.8, color = 'dodgerblue', zorder=2, label=r'H$_2$O + faculae')   
    ax1.plot(wl_plt_um, spec_4_plt, lw=1.5, alpha=0.8, color = 'chocolate', zorder=3, label=r'HCN + faculae') 
    ax1.plot(wl_plt_um, spec_6_plt, lw=1.5, alpha=0.4, color = 'purple', zorder=6, label=r'Atmosphere') 
 #   ax1.plot(wl_plt_um, spec_7_plt, lw=1.0, alpha=0.6, color = 'crimson', zorder=7, label=r'HCN') 
 #   ax1.plot(wl_plt_um, spec_8_plt, lw=1.0, alpha=0.6, color = 'purple', zorder=8, label=r'H2S') 
 #   ax1.plot(wl_plt_um, spec_9_plt, lw=1.0, alpha=0.6, color = 'chocolate', zorder=9, label=r'PH3') 
 #   ax1.plot(wl_plt_um, spec_10_plt, lw=1.0, alpha=0.6, color = 'limegreen', zorder=8, label=r'PO') 
 #   ax1.plot(wl_plt_um, spec_11_plt, lw=1.0, alpha=0.6, color = 'brown', zorder=9, label=r'PN')  
 
  #  ax1.plot(wl_plt_um, spec_5_plt, lw=1.0, alpha=1.0, color = 'grey', zorder=10, label=r'H$_2$ + faculae') 
     
    ax1.scatter(wl_data, ymodel, color = 'gold', s=5, zorder=15, marker='D', edgecolor='black', alpha=0.8, lw=0.1, label=r'Binned Model')
    ax1.errorbar(wl_data, ydata, yerr = err_data, xerr = bin_size, marker='o', markersize=3, capsize=2, zorder=14, ls='none', color='black', elinewidth=0.8, alpha=1.0, label=r'Observations')
    
 #   ax1.annotate(s='', xy=(0.31, mode_arrow_y), xytext=(0.57, mode_arrow_y), arrowprops=dict(arrowstyle='<->', color='blueviolet', alpha=0.8))
 #   ax1.annotate(s='', xy=(0.5, mode_arrow_y+0.016e-2), xytext=(1.08, mode_arrow_y+0.016e-2), arrowprops=dict(arrowstyle='<->', color='m', alpha=0.8))
    ax1.annotate(s='', xy=(0.385, mode_arrow_y), xytext=(1.0, mode_arrow_y), arrowprops=dict(arrowstyle='<->', color='navy', alpha=0.8))
    ax1.annotate(s='', xy=(1.05, mode_arrow_y), xytext=(1.75, mode_arrow_y), arrowprops=dict(arrowstyle='<->', color='green', alpha=0.8))
    ax1.annotate(s='', xy=(3.2, mode_arrow_y), xytext=(5.1, mode_arrow_y), arrowprops=dict(arrowstyle='<->', color='orangered', alpha=0.8))
        
 #   ax1.text(0.365, mode_label_y, r'STIS G430', fontsize=8, color='blueviolet')
 #   ax1.text(0.64, mode_label_y+0.016e-2, r'STIS G750', fontsize=8, color='m')
    ax1.text(0.405, mode_label_y, r'ACCESS + LRG-BEASTS + GMOS + FORS2', fontsize=7.5, color='darkblue')
    ax1.text(1.19, mode_label_y, r'WFC3 G141', fontsize=8, color='darkgreen')
    ax1.text(3.70, mode_label_y, r'Spitzer', fontsize=8, color='orangered')  
    
#    ax1.annotate(s='', xy=(0.62, 4.47e-3), xytext=(1.02, 4.47e-3), arrowprops=dict(arrowstyle='<->', color='purple', alpha=0.8))
#    ax1.annotate(s='', xy=(0.82, 4.44e-3), xytext=(2.86, 4.44e-3), arrowprops=dict(arrowstyle='<->', color='blueviolet', alpha=0.8))
#    ax1.annotate(s='', xy=(2.37, 4.50e-3), xytext=(4.10, 4.50e-3), arrowprops=dict(arrowstyle='<->', color='mediumseagreen', alpha=0.8))
#    ax1.annotate(s='', xy=(3.80, 4.53e-3), xytext=(5.10, 4.53e-3), arrowprops=dict(arrowstyle='<->', color='darkolivegreen', alpha=0.8))
#    ax1.annotate(s='', xy=(2.85, 4.44e-3), xytext=(5.22, 4.44e-3), arrowprops=dict(arrowstyle='<->', color='royalblue', alpha=0.8))
#    ax1.annotate(s='', xy=(4.95, 4.44e-3), xytext=(11.1, 4.44e-3), arrowprops=dict(arrowstyle='<->', color='darkred', alpha=0.8))

 #   ax1.text(1.0, 1.59e-2, r'H$_2$O', fontsize=7)  
 #   ax1.text(1.0, 1.57e-2, r'CH$_4$', fontsize=7) 
 #   ax1.text(1.0, 1.55e-2, r'CO$_2$', fontsize=7) 
 #   ax1.text(1.0, 1.53e-2, r'NH$_3$', fontsize=7) 

    ax1.set_xticks([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])     
        
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    ax1.set_xlabel(r'Wavelength ' + r'(μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    ax1.text(planet_name_x_position, planet_name_y_position, r'WASP-103b', fontsize = 14)
    ax1.text(planet_name_x_position, 0.991*planet_name_y_position, r'Spectral Contributions', fontsize = 12)
  
    # Compute equivalent scale height for secondary axis
 #   base_depth = (R_p*R_p)/(R_s*R_s)
    
 #   photosphere_T = 1000.0  # Retrieved photosphere temperature
    
 #   H_sc = (sc.k*photosphere_T)/(mu*g_0)
 #   depth_values = np.array([transit_depth_range[0], transit_depth_range[1]])
 #   N_sc = ((depth_values - base_depth)*(R_s*R_s))/(2.0*R_p*H_sc)
    
 #   ax2.set_ylim([N_sc[0], N_sc[1]])
 #   ax2.set_ylabel(r'$\mathrm{Scale \, \, Heights}$', fontsize = 16)
    
    legend = ax1.legend(loc='lower right', shadow=False, prop={'size':8}, ncol=2, frameon=False)    #legend settings
    for line in legend.get_lines(): line.set_linewidth(2.0)
    frame = legend.get_frame()
 #   frame.set_facecolor('0.90')
    legend.set_bbox_to_anchor([0.98, 0.10], transform=None)
    
    # Add inset for Spitzer observations
    if (1==2):
        ax3 = inset_axes(ax1, height='17%', width='17%',loc=1, borderpad=0.8)
                
        xmaj_Locator   = MultipleLocator(0.5)
        xmaj_Formatter = FormatStrFormatter('%.1f')
        xmin_Locator   = MultipleLocator(0.1)
    
        ymaj_Locator   = MultipleLocator(0.2)
        #ymaj_Formatter = ScalarFormatter(useMathText=True)
        #ymaj_Formatter.set_powerlimits((0,0))
        ymin_Locator   = MultipleLocator(0.04)
                
        ax3.xaxis.set_major_locator(xmaj_Locator)
        ax3.xaxis.set_major_formatter(xmaj_Formatter)
        ax3.xaxis.set_minor_locator(xmin_Locator)
        ax3.yaxis.set_major_locator(ymaj_Locator)
        #ax3.yaxis.set_major_formatter(ymaj_Formatter)
        ax3.yaxis.set_minor_locator(ymin_Locator)
        
        ax3.plot(wl_um, spectrum_2*1.0e3, lw=0.6, alpha=0.6, color = 'blue', zorder=2)  
        ax3.plot(wl_um, spectrum_3*1.0e3, lw=0.6, alpha=0.6, color = 'crimson', zorder=3)  
        ax3.plot(wl_um, spectrum_4*1.0e3, lw=0.6, alpha=0.6, color = 'purple', zorder=4)  
        ax3.plot(wl_um, spectrum_5*1.0e3, lw=0.6, alpha=0.6, color = 'chocolate', zorder=5)
        ax3.plot(wl_um, spectrum_6*1.0e3, lw=0.6, alpha=1.0, color = 'grey', zorder=6)  
        ax3.plot(wl_um, spectrum_1*1.0e3, lw=1.8, alpha=0.4, color = 'green', zorder=1)   
        ax3.scatter(wl_data[47:], ymodel[47:]*1.0e3, color = 'gold', s=5, zorder=11, marker='D', edgecolor='black', alpha=0.8, lw=0.1)
        ax3.errorbar(wl_data[47:], ydata[47:]*1.0e3, yerr = err_data[47:]*1.0e3, xerr = bin_size[47:], marker='o', markersize=3, capsize=2, zorder=9, ls='none', color='black', elinewidth=0.8)
                
        ax3.set_xlim([3.0, 5.2])
        ax3.set_ylim([4.60, 5.40])
                
        ax3.tick_params(axis='both', which='major', labelsize=5)
                
        ax3.set_xlabel(r'$\mathrm{Wavelength}$ ' + r'($\mu m$)', fontsize = 5, labelpad = 1.0)
        ax3.set_ylabel(r'$\mathrm{Transit \, \, Depth \, \, } \times 10^3$', fontsize = 5, labelpad = 1.0)

    plt.tight_layout()
    
 #   plt.savefig('../../output/plots/' + planet_name + '_spectral_contributions.svg', bbox_inches='tight', fmt='svg', dpi=1000)
    plt.savefig('../../output/plots/' + planet_name + '_spectral_contributions.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)
   
    
def plot_Fp(Fp, wl):
    
    R_smooth = 100.0
    wl_smooth_min = 0.52
    wl_smooth_max = 3.0
    
    wl_R100, Fp_R100 = bin_spectrum_fast(wl, Fp, wl_smooth_min, wl_smooth_max, R_smooth)
    
    wl_um = wl
    wl_R100_um = wl_R100
    
    ax = plt.gca()
    ax.plot(wl_um, Fp, lw=0.5, alpha=0.6, label=r'$F_p$ (line-by-line)')
    ax.plot(wl_R100_um, Fp_R100, lw=1, color = 'crimson', alpha=0.8, label=r'$F_p$ (binned, R=100)')
    ax.set_xlabel(r'Wavelength ' + r'($\mu m$)', fontsize = 16)
    ax.set_ylabel(r'Surface Flux ' + r'($W \, m^{-2} \, sr^{-1} \, m^{-1}$)', fontsize = 16)
    ax.set_xlim([min(wl), max(wl)])
    ax.legend(loc='upper right', shadow=True, prop={'size':10}, ncol=1, frameon=False)
    
    plt.savefig('../../output/plots/' + planet_name + 'Fp_test.pdf', fmt='pdf', dpi=400)
    
    
def plot_spectra_retrieved(spectra_median, spectra_low2, spectra_low1, 
                           spectra_high1, spectra_high2, ymodel_median, 
                           wl_data, ydata, err_data, bin_size, R_bin = 100, 
                           show_ymodel = True, spectrum_true = None, 
                           wl_true = None, wl_min = None, wl_max = None, 
                           transit_depth_min = None, transit_depth_max = None, 
                           colour_list = [], model_labels = [], plt_tag = None):
    
    ''' Plot retrieved transmission spectra.
    
    '''
        
    # Find number of spectra to plot
    N_spectra = len(spectra_median)
 
    # Format planet name
    if ('HD' in planet_name):
        planet_tag = planet_name[:2] + ' ' + planet_name[2:]
    elif ('HAT' in planet_name):
        planet_tag = planet_name[:3] + '-P-' + planet_name[4:]
    elif ('WASP' in planet_name):
        planet_tag = planet_name[:4] + '-' + planet_name[4:]
    else:
        planet_tag = planet_name
        
    # Quick validity checks for plotting
    if (N_spectra == 0):
        raise Exception("Must provide at least one spectrum to plot!")
    if (N_spectra > 3):
        raise Exception("Max number of concurrent retrieved spectra to plot is 3.")
    if ((colour_list != []) and (N_spectra != len(colour_list))):
        raise Exception("Number of colours does not match number of spectra.")
    if ((model_labels != []) and (N_spectra != len(model_labels))):
        raise Exception("Number of model labels does not match number of spectra.")
        
    # Define colours for plotted spectra (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['purple', 'red', 'green', 'brown', 'black', 'darkgrey']
    else:
        colours = colour_list
                
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
            
            transit_depth_min_i = np.min(spectra_median[i][0])
            transit_depth_min_plt = min(transit_depth_min_plt, transit_depth_min_i)
            
        # Check if the lowest data point falls below the current y-limit
        if (load_observations == True):
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
            
            transit_depth_max_i = np.max(spectra_median[i][0])
            transit_depth_max_plt = max(transit_depth_max_plt, transit_depth_max_i)
            
        # Check if the highest data point falls above the current y-limit
        if (load_observations == True):
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
    
    # If comparing to an actual (true) spectrum, bin then plot it
    if (spectrum_true != None):

        wl_true_binned, \
        spec_true_binned = bin_spectrum_fast(wl_true, spectrum_true, R_bin)

        # Plot actual (true) spectrum
        ax1.plot(wl_true, spec_true_binned, lw=1.0, color=colours[i], 
                 label=r'True Model')
                     
    for i in range(N_spectra):
        
        # Extract spectrum and wavelength grid
        (spec_med, wl) = spectra_median[i]
        (spec_low1, wl) = spectra_low1[i]
        (spec_low2, wl) = spectra_low2[i]
        (spec_high1, wl) = spectra_high1[i]
        (spec_high2, wl) = spectra_high2[i]
        
        # If user did not specify a model label, just call them "Model 1, 2" etc.
        if (model_labels == []):
            label_i = ''
        else:
            label_i = '(' + model_labels[i] + ')'
        
        # Calculate binned wavelength and retrieved spectra confidence intervals
        wl_binned, spec_med_binned = bin_spectrum_fast(wl, spec_med, R_bin)
        wl_binned, spec_low1_binned = bin_spectrum_fast(wl, spec_low1, R_bin)
        wl_binned, spec_low2_binned = bin_spectrum_fast(wl, spec_low2, R_bin)
        wl_binned, spec_high1_binned = bin_spectrum_fast(wl, spec_high1, R_bin)
        wl_binned, spec_high2_binned = bin_spectrum_fast(wl, spec_high2, R_bin)
        
        # Plot median retrieved spectrum
        ax1.plot(wl_binned, spec_med_binned, lw=1.0,  
                 color=scale_lightness(colours[i], 1.0), 
                 label=r'Median ' + label_i)
        
        # Plot +/- 1 sigma confidence region
        ax1.fill_between(wl_binned, spec_low1_binned, spec_high1_binned,
                         lw=0.0, alpha=0.5, facecolor=colours[i],  
                         label=r'$1 \sigma$ ' + label_i)

        # Plot +/- 2 sigma confidence region
        ax1.fill_between(wl_binned, spec_low2_binned, spec_high2_binned,
                         lw=0.0, alpha=0.2, facecolor=colours[i],  
                         label=r'$2 \sigma$ ' + label_i)
            
    # Overplot datapoints
    if (load_observations == True):
        ax1.errorbar(wl_data, ydata, yerr=err_data, xerr=bin_size, marker='o', 
                     markersize=3, capsize=2, ls='none', color='black', 
                     elinewidth=0.8, alpha=0.8, label=r'Observations') 
        
    # Overplot median model, binned to resolution of the observations
    if (show_ymodel == True):
        ax1.scatter(wl_data, ymodel_median, color = 'gold', s=5, marker='D', 
                    lw=0.1, alpha=0.8, zorder=100, edgecolor='black',
                    label=r'Binned Model')
    
    # Set axis ranges
    ax1.set_xlim([wl_range[0], wl_range[1]])
    ax1.set_ylim([transit_depth_range[0], transit_depth_range[1]])
        
    # Set axis labels
    ax1.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    ax1.set_ylabel(r'Transit Depth $(R_p/R_*)^2$', fontsize = 16)

    # Add planet name label
    ax1.text(planet_name_x_position, planet_name_y_position, planet_tag, fontsize = 16)
        
    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
    elif (wl_max <= 2.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
    elif (wl_max <= 3.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.5)
        wl_ticks_3 = np.array([])
    else:
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, 3.0+0.01, 0.5)
        wl_ticks_3 = np.arange(3.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        
    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3))
        
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
    
    legend = ax1.legend(loc='upper right', shadow=True, prop={'size':10}, ncol=1, frameon=False)    #legend settings
    #legend.set_bbox_to_anchor([0.75, 0.98], transform=None)
    #frame = legend.get_frame()
    #frame.set_facecolor('0.90') 
        
 #   for legline in legend.legendHandles:
 #       legline.set_linewidth(1.0)
    
    plt.tight_layout()
    
    # Write figure to file
    if (plt_tag == None):
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name + 
                    '_spectrum_retrieved.pdf', bbox_inches='tight', dpi=500)
    else:
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name + 
                    '_' + plt_tag + '_spectrum_retrieved.pdf', bbox_inches='tight', dpi=500)
                 
    
def plot_PT_profiles(P, T, Atmosphere_dimension, TwoD_type, show_profiles=[],
                     plt_tag = None):
    
    ''' Plot the pressure-temperature (P-T) profiles defining the atmosphere.
    
        For a 1D model, a single P-T profile is plotted. For 2D or 3D models,
        the user needs to specify the regions from which the P-T profiles
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
        
    legend = ax.legend(loc='lower left', shadow=True, prop={'size':14}, ncol=1, frameon=False, columnspacing=1.0)     #legend settings
    
    fig.set_size_inches(9.0, 9.0)
    
    #plt.tight_layout()
    
    
    #***** For illustrative diagram *****#
    if (1 == 70):
        
        ax.annotate(s='', xy=(1500.0, 5.0e-6), xytext=(2300.0, 5.0e-6), arrowprops=dict(arrowstyle='<->', lw=2, color='darkgreen', alpha=0.8))
        ax.axhline(y=1.0e-5, lw=2, linestyle=':', alpha=0.4, color='purple')
        ax.axhline(y=1.0e1, lw=2, linestyle=':', alpha=0.4, color='purple')
    
        ax.text(1330, 6.0e-6, r'$P_{\rm{high}}$', fontsize = 20)
        ax.text(1330, 6.0e-0, r'$P_{\rm{deep}}$', fontsize = 20)
    
        ax.text(1530, 6.0e-7, r'$T_{\rm{night}}$', color='darkblue', fontsize = 20)
        ax.text(2330, 6.0e-7, r'$T_{\rm{day}}$', color='darkred', fontsize = 20)
        ax.text(1930, 6.0e-7, r'$T_{\rm{term}}$', color='darkorange', fontsize = 20)
        ax.text(2530, 5.0e1, r'$T_{\rm{deep}}$', color='black', fontsize = 20)
        ax.text(1930, 3.0e-6, r'$\Delta T_{\rm{DN}}$', color='darkgreen', fontsize = 20)

#    legend.set_bbox_to_anchor([0.20, 0.10], transform=None)

    # Write figure to file
    if (plt_tag == None):
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name + 
                    '_PT.pdf', bbox_inches='tight', dpi=500)
    else:
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_' + plt_tag + '_PT.pdf', bbox_inches='tight', dpi=500)
        
    
def plot_PT_retrieved(P, T_median, T_low2, T_low1, T_high1, T_high2, 
                      Atmosphere_dimension, T_true = None, plt_tag = None):
        
    ''' Plot a retrieved Pressure-Temperature (P-T) profile.
        
    '''

    # Find minimum and maximum temperatures in atmosphere
    T_min = np.floor(np.min(T_low2)/100)*100 - 200.0    # Round down to nearest 100
    T_max = np.ceil(np.max(T_high2)/100)*100 + 200.0     # Round up to nearest 100
        
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
    if (Atmosphere_dimension > 1):
        raise Exception("This function does not support multidimensional retrievals.")
        
    else:
        
        # Plot actual (true) P-T profile
        if (T_true != None):
            ax.semilogy(T_true, P, lw=1.5, color = 'crimson', label='True')

        # Plot the median, +/- 1 sigma, and +/- 2 sigma confidence intervals 
        ax.semilogy(T_median, P, lw=1.5, color = 'darkblue', label=r'Median')
        ax.fill_betweenx(P, T_low1, T_high1, facecolor='purple', alpha=0.5, linewidth=0.0, label=r'$1 \sigma$')
        ax.fill_betweenx(P, T_low2, T_high2, facecolor='purple', alpha=0.2, linewidth=0.0, label=r'$2 \sigma$')

        # Common plot settings
        ax.invert_yaxis()            
        ax.set_xlabel(r'Temperature (K)', fontsize = 20)
        ax.set_xlim(T_min, T_max)  
        ax.set_ylabel(r'Pressure (bar)', fontsize = 20)
        ax.tick_params(labelsize=12)
            
        legend = ax.legend(loc='upper right', shadow=True, prop={'size':14}, ncol=1, frameon=False, columnspacing=1.0)     #legend settings
        
        fig.set_size_inches(9.0, 9.0)
    
        #plt.tight_layout()
    
        #legend.set_bbox_to_anchor([0.20, 0.10], transform=None)
    
    # Write figure to file
    if (plt_tag == None):
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_PT_retrieved.pdf', bbox_inches='tight', dpi=500)
    else:
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_' + plt_tag + '_PT_retrieved.pdf', bbox_inches='tight', dpi=500)
        
    
def plot_X_profiles(P, log_X, Atmosphere_dimension, TwoD_type, chemical_species, 
                    plot_species=[], colour_list=[], show_profiles=[],
                    plt_tag = None):
    
    ''' Plot the mixing ratio profiles defining the atmosphere.
    
        The user species which chemical species to plot via the list
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
    
    # Quick validity checks for plotting
    if (len(plot_species) == 0):
        raise Exception("Must specify at least one species to plot!")
    if (len(plot_species) > 8):
        raise Exception("Max number of concurrent species to plot is 8.")
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
                       frameon=True, columnspacing=1.0)     #legend settings
    frame = legend.get_frame()
    frame.set_facecolor('0.90') 
    
    fig.set_size_inches(9.0, 9.0)
    
#    ax.invert_yaxis()            
#    ax.set_xlabel(r'Mixing Ratios (log $X_{\rm{i}}$)', fontsize = 16)
#    ax.set_xlim(log_X_min, log_X_max)  
#    ax.set_ylabel(r'Pressure (bar)', fontsize = 16)
#    ax.tick_params(labelsize=12)
        
#    ax.legend(loc='upper right', shadow=True, prop={'size':12},frameon=False, columnspacing=1.0)     #legend settings
    
    
    #plt.tight_layout()
    
    # Write figure to file
    if (plt_tag == None):
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_X_profiles.pdf', bbox_inches='tight', dpi=500)
    else:
        plt.savefig('../../output/plots/' + planet_name + '/' + planet_name +
                    '_' + plt_tag + '_X_profiles.pdf', bbox_inches='tight', dpi=500)
        

def plot_chem_histogram(nbins, X_i_vals, colour, oldax, shrink_factor):
    
    weights = np.ones_like(X_i_vals)/float(len(X_i_vals))
    
    x,w,patches = oldax.hist(X_i_vals, bins=nbins, color=colour, histtype='stepfilled', alpha=0.4, edgecolor='None', weights=weights, density=True, stacked=True)
    x,w,patches = oldax.hist(X_i_vals, bins=nbins, histtype='stepfilled', lw = 0.8, facecolor='None', weights=weights, density=True, stacked=True)
        
    oldax.set_ylim(0, (1.1+shrink_factor)*x.max())
    
    low3, low2, low1, median, high1, high2, high3 = confidence_intervals(len(X_i_vals), X_i_vals, 0)
    
    return low1, median, high1


def plot_retrieval_panel(X_vals_1, X_vals_2, X_vals_3, N_bin_1, N_bin_2, N_bin_3, 
                         included_species, param, parameter_label, x_label, ax, q, 
                         X_range, true_val, colours, shrink_factor):    
    
    # Specify location of species label
    param_name_x_position = 0.84*(X_range[1]-X_range[0]) + X_range[0]
    param_name_y_position = 0.92
    
    # Specify location of retrieval type label
    retrieval_type_x_position = 0.05*(X_range[1]-X_range[0]) + X_range[0]
    retrieval_type_y_position = 0.92
    
    retrieval_x_position_1 = 0.05*(X_range[1]-X_range[0]) + X_range[0]
    retrieval_x_position_2 = 0.05*(X_range[1]-X_range[0]) + X_range[0]
    retrieval_x_position_3 = 0.05*(X_range[1]-X_range[0]) + X_range[0]
    
    # For each retrieval
    for i in range(3):
        
        if (i==0):
            N_bin_plt = N_bin_1
            X_vals_plt = X_vals_1[:,np.where(included_species == param)[0][0]]
        elif (i==1):
            N_bin_plt = N_bin_2
            X_vals_plt = X_vals_2[:,np.where(included_species == param)[0][0]]
        elif (i==2):
            N_bin_plt = N_bin_3
            X_vals_plt = X_vals_3[:,np.where(included_species == param)[0][0]]
            
        low1, median, high1, high2 = plot_chem_histogram(N_bin_plt, np.log10(X_vals_plt), colours[i], ax, shrink_factor)

        # Store the median, 1 sigma regions for error bar plotting later
        if   (i == 0): low1_posterior_1, median_posterior_1, high1_posterior_1, high2_posterior_1 = low1, median, high1, high2
        elif (i == 1): low1_posterior_2, median_posterior_2, high1_posterior_2, high2_posterior_2 = low1, median, high1, high2
        elif (i == 2): low1_posterior_3, median_posterior_3, high1_posterior_3, high2_posterior_3 = low1, median, high1, high2

        # Create new axis for error bar
        newax = plt.gcf().add_axes(ax.get_position(), sharex=ax, frameon=False)
        newax.set_ylim(0, 1)
        
        # Position of error bar
        ylim = newax.get_ylim()
    
        # Remove y-ticks on original axis
        ax.set_yticks([])
            
   #     oldax.tick_params(axis='both', which='major', labelsize=5)
   #     newax.tick_params(axis='both', which='major', labelsize=5)
   
    # Adjust x-axis extent
    ax.set_xlim(X_range[0], X_range[1])
    
    # Add parameter name to plot
    newax.text(param_name_x_position, param_name_y_position, parameter_label, fontsize = 16)
        
    # Add x-axis label
    ax.set_xlabel(x_label, fontsize = 16)
    
    # Add true parameter value
    newax.axvline(x=true_val, linewidth=3.0, linestyle='-', color='r', alpha=0.5)
             
    # For first plot in row
    if (q in [0]):
        newax.set_ylabel(r'Probability density (normalized)', fontsize = 16)
            
        # Add retrieval data type
        newax.text(retrieval_type_x_position, retrieval_type_y_position, 
                   'Retrieved Abundances', fontsize = 15)
            
    if (q == 1):
        
        # Add retrieval type names
        newax.text(retrieval_x_position_1, retrieval_type_y_position, 
                   '35 ppm', color = colours[0], fontsize = 14)
        newax.text(retrieval_x_position_2, retrieval_type_y_position-0.06, 
                   '20 ppm', color = colours[1], fontsize = 14)
        newax.text(retrieval_x_position_3, retrieval_type_y_position-0.12, 
                   '15 ppm', color = colours[2], fontsize = 14)
        
    # For other plots in row
    elif (q != 0):
        newax.set_yticklabels([])    # Remove y-axis tick marks
        
   #     newax.axvline(x=-3.3, linewidth=3.0, linestyle=':', color='r', alpha=0.5)
   
    # Plot upper limit        
#    newax.axvline(x=high2_posterior_1, linewidth=2, linestyle='-', color='b')
        
#    arrow_delta_right = 0.02*np.abs(high2_posterior_1 - X_range[1]) # Start arrow 2% along plot area left of 2 sigma boundary
#    arrow_delta_left = 0.40*np.abs(high2_posterior_1- X_range[0])  # End arrow 20% along plot area right of 2 sigma boundary
                
#    newax.annotate(s='', xy=(high2_posterior_1+arrow_delta_right, 0.5), 
#                   xytext=(high2_posterior_1-arrow_delta_left, 0.5), 
#                   arrowprops=dict(arrowstyle='<-', color='blue')) 
   
    # Plot error bars
    newax.errorbar(x=median_posterior_1, y = (ylim[0] + 0.09*(ylim[1] - ylim[0])),
                   xerr=np.transpose([[median_posterior_1 - low1_posterior_1, high1_posterior_1 - median_posterior_1]]), 
                   color=colours[0], ecolor=colours[0], markersize=5, markeredgewidth = 0.8, 
                   linewidth=1.1, capthick=1.1, capsize=1.9, marker='s')
    newax.errorbar(x=median_posterior_2, y = (ylim[0] + 0.06*(ylim[1] - ylim[0])),
                   xerr=np.transpose([[median_posterior_2 - low1_posterior_2, high1_posterior_2 - median_posterior_2]]), 
                   color=colours[1], ecolor=colours[1], markersize=5, markeredgewidth = 0.8, 
                   linewidth=1.1, capthick=1.1, capsize=1.9, marker='s')
    newax.errorbar(x=median_posterior_3, y = (ylim[0] + 0.03*(ylim[1] - ylim[0])),
                   xerr=np.transpose([[median_posterior_3 - low1_posterior_3, high1_posterior_3 - median_posterior_3]]), 
                   color=colours[2], ecolor=colours[2], markersize=5, markeredgewidth = 0.8, 
                   linewidth=1.1, capthick=1.1, capsize=1.9, marker='s')  
 
    
def plot_abundances_comparison(X_vals_1, X_vals_2, X_vals_3, included_species):
    
    # Initialise multi-panel grid
    fig = plt.figure()
    
    fig.set_size_inches(10.0, 5.0)
    
    gs = gridspec.GridSpec(1, 2) 

    #***** Define settings for multi-panel plot *****#
    
    # Chemical species to plot
    species = ['HCN', 'NH3']
    
    # Latex code for soecuies labels
    species_labels = ['HCN', 'NH$_3$']
    
    # x-axis labels
    x_labels = [r'log $(X_{\mathrm{HCN}})$', r'log $(X_{\mathrm{NH_{3}}})$']
    
    # Horizontal ranges on retrieved mixing ratio panels
    X_ranges = [[-10.0, -3.0],
                [-10.0, -3.0]]
    
    # Number of bins for histogram of each chemical species
    N_bins_1 = [60, 60]
    N_bins_2 = [80, 90]
    N_bins_3 = [90, 100]
    
    # Mixing ratio histogram colours for each retrieval code
    colours = ['green', 'blue', 'purple']
    
    # True parameter values
    true_vals = [-5.7, -6.0]
    
    # Factors to alter normalisation of different retrieval posteriors for nicer plot
    shrink_factors = [0.0, 0.16]
    
    #***** Generate panels *****#
    
    # For each species
    for q in range(len(species)):
        
        ax = plt.subplot(gs[0, q:q+1])
            
        plot_retrieval_panel(X_vals_1, X_vals_2, X_vals_3, N_bins_1[q], N_bins_2[q], N_bins_3[q],
                             included_species, species[q], species_labels[q], x_labels[q],
                             ax, q, X_ranges[q], true_vals[q], colours, shrink_factors[q])
    
    plt.tight_layout()
    
    plt.savefig('../../output/plots/HD209458bb_HCN_NH3_abundances.pdf', bbox_inches='tight', fmt='pdf', dpi=400)
    

def plot_opacity(P, T, database = 'High-T', plot_species=[], colour_list=[], 
                 plt_tag = None, smooth = False, smooth_factor = 100, 
                 wl_min = None, wl_max = None, sigma_min = None, sigma_max = None):
    
    ''' A visualisation routine to produce plots of cross sections.
    
    '''
    
    # Quick validity checks for plotting
    if (len(plot_species) == 0):
        raise Exception("Must specify at least one species to plot!")
    if (len(plot_species) > 9):
        raise Exception("Max number of concurrent species to plot is 9.")
    if ((colour_list != []) and (len(plot_species) != len(colour_list))):
        raise Exception("Number of colours does not match number of species.")
        
    # If the user did not specify a wavelength range
    if (wl_min == None):
        wl_min = 0.4   
            
    # If the user did not specify a wavelength range
    if (wl_max == None):
        wl_max = 20.0

    # Set x range
    wl_range = [wl_min, wl_max]
    
    # If the user did not specify a cross section range
    if (sigma_min == None):
        sigma_min_plt = 1.0e-30   # Dummy value
    else:
        sigma_min_plt = sigma_min
    
    # If the user did not specify a cross section range
    if (sigma_max == None):
        sigma_max_plt = 1.0e-16   # Dummy value
    else:
        sigma_max_plt = sigma_max

    
    # Define colours for mixing ratio profiles (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['darkorange', 'navy', 'purple', 'black', 'dimgrey',
                   'royalblue', 'darkgreen', 'magenta', 'crimson']
    else:
        colours = colour_list

    # Find LaTeX code for each chemical species to plot
    latex_species = generate_latex_param_names(plot_species)
    
    # Initialise plot
    plt.figure(figsize=(9.2,6))
    ax = plt.gca()    
    ax.set_xscale("log")
    
    # Create x formatting objects
    if (wl_max < 1.0):    # If plotting over the optical range
        xmajorLocator = MultipleLocator(0.1)
        xminorLocator = MultipleLocator(0.02)
        
    else:                 # If plot extends into the infrared
        xmajorLocator = MultipleLocator(1.0)
        xminorLocator = MultipleLocator(0.1)
            
    xmajorFormatter = FormatStrFormatter('%g')
    xminorFormatter = NullFormatter()
        
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_formatter(xminorFormatter)
    
    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 2.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 3.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.5)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 10.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, 2.0+0.01, 0.5)
        wl_ticks_3 = np.arange(2.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        wl_ticks_4 = np.array([])
    else:
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, 2.0+0.01, 0.5)
        wl_ticks_3 = np.arange(2.0, 10+0.01, 1.0)
        wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 2)+0.01, 2.0)
   
    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3, wl_ticks_4))
    
    # Open opacity database
    if (database == 'High-T'):
        opac_file = h5py.File('../../opacity/Opacity_database_0.01cm-1.hdf5', 'r')
    elif (database == 'Temperate'):
        opac_file = h5py.File('../../opacity/Opacity_database_0.01cm-1_TEMPERATE.hdf5', 'r')
    
    # Plot each cross section
    for q in range(len(plot_species)):
        
        species = plot_species[q]  # Species to plot cross section
        colour = colours[q]        # Colour of cross section for plot

        # Load cross section and grids
        if (species == 'H-'):
            wl_plt = np.linspace(wl_min, wl_max, 10000)
            sig_plt = H_minus_bound_free(wl_plt)
            latex_species[q] = 'H$^{-}$ (bf)'

        else:
            sig = np.power(10.0, np.array(opac_file[species + '/log(sigma)']))
            wl_plt = 1.0e4/np.array(opac_file[species+ '/nu'])
            T_grid = np.array(opac_file[species + '/T'])   
            log_P_grid = np.array(opac_file[species + '/log(P)'])   
            
            # Find nearest entry for desired T and P to plot
            idx_T = closest_index(T, T_grid[0], T_grid[-1], len(T_grid))
            idx_P = closest_index(np.log10(P), log_P_grid[0], log_P_grid[-1], len(log_P_grid))    
        
            sig_plt = sig[idx_P, idx_T,:]
        
            del sig
        
        # Gaussian smooth cross section for clarity
        if ((smooth == True) and (species not in ['H-'])):
            sig_plt = gauss_conv(sig_plt, sigma=smooth_factor, mode='nearest')
        
        # Plot cross section
        plt.semilogy(wl_plt, sig_plt, lw=1.5, alpha=0.8, color=colour, 
                     label=latex_species[q])
        
    # Set axis limits
    plt.ylim([sigma_min_plt, sigma_max_plt])
    plt.xlim([wl_min, wl_max])
    
    # Plot wl tick labels
    ax.set_xticks(wl_ticks)
    
    # Place planet name
    if (database == 'High-T'):
        label_x_position = np.power(10.0, (0.42*(np.log10(wl_range[1])-np.log10(wl_range[0])) + np.log10(wl_range[0])))
        label_y_position = np.power(10.0, (0.92*(np.log10(sigma_max_plt)-np.log10(sigma_min_plt)) + np.log10(sigma_min_plt)))
    elif (database == 'Temperate'):
        label_x_position = np.power(10.0, (0.04*(np.log10(wl_range[1])-np.log10(wl_range[0])) + np.log10(wl_range[0])))
        label_y_position = np.power(10.0, (0.92*(np.log10(sigma_max_plt)-np.log10(sigma_min_plt)) + np.log10(sigma_min_plt)))
    

    # Add axis labels
    ax.set_ylabel(r'Cross Section (m$^{2}$ / species)', fontsize = 16)
    ax.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    
    
    if (plt_tag != None):
        ax.text(label_x_position, label_y_position, plt_tag, fontsize = 16)
        
  #  ax.text(0.41, 2.0e-15, (r'$\mathrm{T = }$' + str(T) + r'$\mathrm{K \, \, P = }$' + str(P*1000) + r'$\mathrm{mbar}$'), fontsize = 12)
    
    # Add legend
    if (len(plot_species) > 6):
        n_columns = 3
    else:
        n_columns = 2
        
    if (database == 'High-T'):
        legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':10}, ncol=n_columns)
        legend.set_bbox_to_anchor([0.95, 0.90], transform=None)
    elif (database == 'Temperate'):
        legend = plt.legend(loc='upper left', shadow=False, frameon=False, prop={'size':10}, ncol=n_columns)
        legend.set_bbox_to_anchor([0.05, 0.90], transform=None)
    
    for legline in legend.legendHandles:
        legline.set_linewidth(1.5)    

    # Close opacity file
    opac_file.close()
    
    # Write figure to file
    plt.savefig('../../output/plots/Cross_sections_' + str(T) + 'K_' + 
                str(P*1000) + 'mbar.pdf', bbox_inches='tight', dpi=500)


def plot_continuum(T, plot_type = 'High-T', plot_pair=[], colour_list=[], 
                   plt_tag = None, smooth = False, smooth_factor = 100, 
                   wl_min = None, wl_max = None, cont_min = None, cont_max = None):
        
    ''' A visualisation routine to produce plots of binary cross sections.
    
    '''
    
    # Quick validity checks for plotting
    if (len(plot_pair) == 0):
        raise Exception("Must specify at least one species to plot!")
    if (len(plot_pair) > 9):
        raise Exception("Max number of concurrent species to plot is 9.")
    if ((colour_list != []) and (len(plot_pair) != len(colour_list))):
        raise Exception("Number of colours does not match number of species.")
        
    # If the user did not specify a wavelength range
    if (wl_min == None):
        wl_min = 0.4   
            
    # If the user did not specify a wavelength range
    if (wl_max == None):
        wl_max = 20.0

    # Set x range
    wl_range = [wl_min, wl_max]
    
    # If the user did not specify a cross section range
    if (cont_min == None):
        cont_min_plt = 1.0e-60   # Dummy value
    else:
         cont_min_plt = cont_min
    
    # If the user did not specify a cross section range
    if (cont_max == None):
        cont_max_plt = 1.0e-40   # Dummy value
    else:
        cont_max_plt = cont_max

    
    # Define colours for mixing ratio profiles (default or user choice)
    if (colour_list == []):   # If user did not specify a custom colour list
        colours = ['darkorange', 'navy', 'purple', 'black', 'dimgrey',
                   'royalblue', 'darkgreen', 'magenta', 'crimson']
    else:
        colours = colour_list

    # Find LaTeX code for each chemical species to plot
    latex_species = generate_latex_param_names(plot_pair)
    
    # Initialise plot
    plt.figure(figsize=(9.2,6))
    ax = plt.gca()    
    ax.set_xscale("log")
    
    # Create x formatting objects
    if (wl_max < 1.0):    # If plotting over the optical range
        xmajorLocator = MultipleLocator(0.1)
        xminorLocator = MultipleLocator(0.02)
        
    else:                 # If plot extends into the infrared
        xmajorLocator = MultipleLocator(1.0)
        xminorLocator = MultipleLocator(0.1)
            
    xmajorFormatter = FormatStrFormatter('%g')
    xminorFormatter = NullFormatter()
        
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_formatter(xminorFormatter)
    
    # Decide at which wavelengths to place major tick labels
    if (wl_max <= 1.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), round_sig_figs(wl_max, 2)+0.01, 0.1)
        wl_ticks_2 = np.array([])
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 2.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.2)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 3.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, round_sig_figs(wl_max, 2)+0.01, 0.5)
        wl_ticks_3 = np.array([])
        wl_ticks_4 = np.array([])
    elif (wl_max <= 10.0):
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, 2.0+0.01, 0.5)
        wl_ticks_3 = np.arange(2.0, round_sig_figs(wl_max, 2)+0.01, 1.0)
        wl_ticks_4 = np.array([])
    else:
        wl_ticks_1 = np.arange(round_sig_figs(wl_min, 1), 1.0+0.01, 0.2)
        wl_ticks_2 = np.arange(1.0, 2.0+0.01, 0.5)
        wl_ticks_3 = np.arange(2.0, 10+0.01, 1.0)
        wl_ticks_4 = np.arange(10.0, round_sig_figs(wl_max, 2)+0.01, 2.0)
   
    wl_ticks = np.concatenate((wl_ticks_1, wl_ticks_2, wl_ticks_3, wl_ticks_4))
    
    # Open opacity database
    cia_file = h5py.File('../../opacity/Opacity_database_cia.hdf5', 'r')
    
    # Plot each cross section
    for q in range(len(plot_pair)):
        
        pair = plot_pair[q]  # Species to plot cross section
        colour = colours[q]        # Colour of cross section for plot

        # Load cross section and grids
        if (pair == 'H-'):
            wl_plt = np.linspace(wl_min, wl_max, 10000)
            cont_plt = H_minus_free_free(wl_plt, np.array([T]))[0,:]
            latex_species[q] = 'H$^{-}$ (ff)'

        else:
            cont = np.power(10.0, np.array(cia_file[pair + '/log(cia)']))
            wl_plt = 1.0e4/np.array(cia_file[pair + '/nu'])
            T_grid = np.array(cia_file[pair + '/T'])   
            
            # Find nearest entry for desired T to plot
            idx_T = closest_index(T, T_grid[0], T_grid[-1], len(T_grid))
        
            cont_plt = cont[idx_T,:]
        
            del cont
        
        # Gaussian smooth cross section for clarity
        if (smooth == True):
            cont_plt = gauss_conv(cont_plt, sigma=smooth_factor, mode='nearest')
        
        # Plot cross section
        plt.semilogy(wl_plt, cont_plt, lw=1.5, alpha=0.8, color=colour, 
                     label=latex_species[q])
        
    # Set axis limits
    plt.ylim([cont_min_plt, cont_max_plt])
    plt.xlim([wl_min, wl_max])
    
    # Plot wl tick labels
    ax.set_xticks(wl_ticks)
    
    # Place planet name
    if (plot_type == 'High-T'):
        label_x_position = np.power(10.0, (0.42*(np.log10(wl_range[1])-np.log10(wl_range[0])) + np.log10(wl_range[0])))
        label_y_position = np.power(10.0, (0.92*(np.log10(cont_max_plt)-np.log10(cont_min_plt)) + np.log10(cont_min_plt)))
    elif (plot_type == 'Temperate'):
        label_x_position = np.power(10.0, (0.04*(np.log10(wl_range[1])-np.log10(wl_range[0])) + np.log10(wl_range[0])))
        label_y_position = np.power(10.0, (0.92*(np.log10(cont_max_plt)-np.log10(cont_min_plt)) + np.log10(cont_min_plt)))
    

    # Add axis labels
    ax.set_ylabel(r'Binary Cross Section (m$^{5}$ / pair)', fontsize = 16)
    ax.set_xlabel(r'Wavelength (μm)', fontsize = 16)
    
    
    if (plt_tag != None):
        ax.text(label_x_position, label_y_position, plt_tag, fontsize = 16)
        
  #  ax.text(0.41, 2.0e-15, (r'$\mathrm{T = }$' + str(T) + r'$\mathrm{K \, \, P = }$' + str(P*1000) + r'$\mathrm{mbar}$'), fontsize = 12)
    
    # Add legend
    if (len(plot_pair) > 6):
        n_columns = 3
    else:
        n_columns = 2
        
    if (plot_type == 'High-T'):
        legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':10}, ncol=n_columns)
        legend.set_bbox_to_anchor([0.95, 0.90], transform=None)
    elif (plot_type == 'Temperate'):
        legend = plt.legend(loc='upper left', shadow=False, frameon=False, prop={'size':10}, ncol=n_columns)
        legend.set_bbox_to_anchor([0.05, 0.90], transform=None)
    
    for legline in legend.legendHandles:
        legline.set_linewidth(1.5)    

    # Close opacity file
    cia_file.close()
    
    # Write figure to file
    plt.savefig('../../output/plots/Continuum_absorption_' + str(T) + 'K.pdf', 
                bbox_inches='tight', dpi=500)


'''  
plot_opacity(1, 2000, database = 'High-T', 
             plot_species=['TiO', 'VO', 'H-', 'Na', 'K', 'H2O', 'CH4', 'CO', 'CO2'], 
             colour_list=['darkorange', 'navy', 'purple', 'black', 'dimgrey',
                          'royalblue', 'darkgreen', 'magenta', 'crimson'], 
             plt_tag = 'High Temperature Cross Sections', smooth=True, smooth_factor = 100,
             wl_min = 0.4,  wl_max = 14.0, sigma_min = 1.0e-30, sigma_max = 1.0e-16)
'''
'''
plot_opacity(1, 300, database = 'Temperate', 
             plot_species=['O3', 'O2', 'H2O', 'CH4', 'CO', 'CO2'], 
             colour_list=['darkorange', 'black', 'royalblue', 'darkgreen', 
                          'magenta', 'crimson'], 
             plt_tag = 'Low Temperature Cross Sections', smooth=True, smooth_factor = 500,
             wl_min = 0.4,  wl_max = 14.0, sigma_min = 1.0e-32, sigma_max = 1.0e-20)
'''
'''
plot_continuum(2000, plot_type = 'High-T', 
               plot_pair=['H2-He', 'H2-H2', 'H2-H', 'H-'], 
               colour_list=['darkorange', 'navy', 'darkgreen', 'purple'],
               plt_tag = 'High Temperature Pair Absorption', smooth = False, 
               wl_min = 0.4, wl_max = 14.0, cont_min = 1.0e-60, cont_max = 1.0e-42)
'''
'''
plot_continuum(300, plot_type = 'Temperate', 
               plot_pair=['O2-O2', 'O2-N2', 'O2-CO2', 'N2-N2', 'N2-H2O', 
                          'N2-H2', 'CO2-CO2', 'CO2-H2', 'CO2-CH4'], 
               colour_list=['darkorange', 'navy', 'dimgrey', 'black', 'purple',
                            'royalblue', 'darkgreen', 'magenta', 'crimson'],
               plt_tag = 'Low Temperature Pair Absorption', smooth = False, 
               wl_min = 0.4, wl_max = 14.0, cont_min = 1.0e-57, cont_max = 1.0e-52)
'''


