# Radiative transfer calculations for generating transmission spectra

import numpy as np
from numba.core.decorators import jit
from .absorption import extinction, extinction_LBL

from .utility import prior_index, closest_index

#from utility import prior_index
#from atmosphere import profiles
#from instrument import make_model_data
#from geometry import angular_grids
#from stellar import stellar_contamination_single_spot
#from parameters import unpack_cloud_params, unpack_geometry_params
#from config import N_D, R_s, T_s, b_p, opacity_treatment, TwoD_type, \
#                   N_slice_EM, N_slice_DN, term_transition, stellar_contam, \
#                   rad_transfer, load_observations, P_deep
                   

@jit(nopython=True)
def zone_boundaries(N_b, N_sectors, N_zones, b, r_up, k_zone_back,
                    theta_edge_min, theta_edge_max):
    
    ''' Compute maximum and minimal radial extent a ray at impact 
        parameter b experiences in each sector and zone.
        
    '''
    
    r_min = np.zeros(shape=(N_b, N_sectors, N_zones))
    r_max = np.zeros(shape=(N_b, N_sectors, N_zones))
    
    for k in range(N_zones):
        
        # Trigonometry to compute maximum r, given b and angle to terminator
        denom_min = np.cos(theta_edge_min[k])
        denom_max = np.cos(theta_edge_max[k])
        
        # Extract index of background atmosphere arrays this sub-zone is in (e.g. radial arrays)
        k_in = k_zone_back[k]   # Only differs from k when a cloud splits a zone
            
        for i in range(N_b):
            
            # Trigonometry to compute maximum r, given b and angle to terminator
            r_min_geom = b[i]/(denom_min + 1.0e-250)   # Denominator one loop up for efficiency
            r_max_geom = b[i]/(denom_max + 1.0e-250)   # Additive factor prevents division by zeros for dayside and nightside
                                       
            for j in range(N_sectors):
                
                # If geometric expressions go above the maximum altitude, set to top of atmosphere
                r_min[i,j,k] = np.minimum(r_up[-1,j,k_in], r_min_geom)
    
                # If in the dayside or nightside, max radial extent given by top of atmosphere
                if ((k == 0) or (k == N_zones-1)):
                    r_max[i,j,k] = r_up[-1,j,k_in]   # Top of atmosphere in sector j, zone k
                
                else:   # For all other zones
                    
                    # If geometric expressions go above the maximum altitude, set to top of atmosphere
                    r_max[i,j,k] = np.minimum(r_up[-1,j,k_in], r_max_geom)
                
    return r_min, r_max


@jit(nopython=True)
def path_distribution_geometric(b, r_up, r_low, dr, i_bot, j_sector_back, 
                                N_layers, N_sectors_back, N_zones_back, 
                                N_zones, N_phi, k_zone_back, theta_edge_all):
    
    ''' Compute the path distribution tensor analytically for the geometric
        limit (rays travel in straight lines).
    
    '''

    # Store length of impact parameter vector    
    N_b = b.shape[0]
    
    # Initialise path distribution tensor
    Path = np.zeros(shape=(N_b, N_phi, N_zones, N_layers))
    
    # Compute squared radial layer boundary vectors
    r_up_sq = r_up * r_up
    r_low_sq = r_low * r_low
    
    # Initialise squared impact parameter array
    b_sq = b * b
    
    # If the rays traverse only a single zone, symmetry gives a factor of 2 in path length
    if (N_zones == 1):
        symmetry_factor = 2.0   # Factor of 2 for ray paths into and out of atmosphere
    elif (N_zones >= 2):
        symmetry_factor = 1.0   # No factor of 2 when inwards and outwards directions separately treated 
    
    # For a uniform atmosphere or a sharp transition, the calculation is simple
    if (N_zones <= 2) and (N_zones == N_zones_back):  # Latter condition ensures a non-uniform cloud hasn't split a zone
        
        # For each terminator sector
        for j in range(N_phi):
            
            # Refresh sector count
            j_sector_last = -1  # This counts the angular index where the transmissivity was last computed
            
            # Find which asymmetric terminator sector this angle lies in
            j_sector_back_in = j_sector_back[j]
            
            # If the path distribution has not yet been computed for this sector
            if (j_sector_back_in != j_sector_last):
            
                # For each zone along tangent ray
                for k in range(N_zones):
                        
                    # For each ray impact parameter
                    for i in range(N_b):
                    
                        # For each atmosphere layer
                        for l in range(i_bot, N_layers):
                        
                            if (b[i] < r_up[l,j_sector_back_in,k]): 
                                
                                s1 = np.sqrt(r_up_sq[l,j_sector_back_in,k] - b_sq[i])
                            
                                if (b[i] > r_low[l,j_sector_back_in,k]): 
                                    
                                    s2 = 0.0
                                
                                else:
                                    
                                    s2 = np.sqrt(r_low_sq[l,j_sector_back_in,k] - b_sq[i])
                                
                                Path[i,j,k,l] = symmetry_factor * (s1 - s2)/dr[l,j_sector_back_in,k]
                            
                            else:
                            
                                Path[i,j,k,l] = 0.0
                                
            # Copy path distribution if sector 'j' is identical to the last
            else:
                
                Path[:,j,:,:] = Path[:,j-1,:,:]
                                
            # Update angular sector pre-computation completion index
            j_sector_last = j_sector_back_in
                            
    # For a day-night transition region, we need to carefully handle the geometry
    else:
        
        #***** Define minimum and maximum angles of zone boundaries *****#
        # Max angles given by removing the terminator plane from array
        theta_edge_max = np.delete(theta_edge_all, np.where(theta_edge_all == 0.0)[0])
        
        # Min angles given by clipping equators and adding an extra 0.0 (two zones adjacent to the terminator plane share theta = 0)
        theta_edge_min = np.sort(np.append(theta_edge_all[1:-1], 0.0))

        # Compute maximum and minimum radial extent each ray can possesses in each zone
        r_min, r_max = zone_boundaries(N_b, N_sectors_back, N_zones, b, r_up, 
                                       k_zone_back, theta_edge_min, theta_edge_max)
        
        # Compute squared radial zone boundary vectors
        r_min_sq = r_min * r_min
        r_max_sq = r_max * r_max

        # For each terminator sector
        for j in range(N_phi):
            
            # Refresh sector count
            j_sector_last = -1  # This counts the angular index where the transmissivity was last computed
            
            # Find which asymmetric terminator sector this angle lies in
            j_sector_back_in = j_sector_back[j]
            
            # If the path distribution has not yet been computed for this sector
            if (j_sector_back_in != j_sector_last):
            
                # For each zone along tangent ray
                for k in range(N_zones):
                    
                    # Extract index of background atmosphere arrays this sub-zone is in (e.g. radial arrays)
                    k_in = k_zone_back[k]   # Only differs from k when a cloud splits a zone
                    
                    # For each ray impact parameter
                    for i in range(N_b):
                    
                        # For each atmosphere layer
                        for l in range(i_bot, N_layers):
                            
                            # Check for layers falling outside of region sampled by ray
                            if ((r_low[l,j_sector_back_in,k_in] >= r_max[i,j_sector_back_in,k]) or
                                (r_up[l,j_sector_back_in,k_in] < r_min[i,j_sector_back_in,k]) or
                                (b[i] > r_max[i,j_sector_back_in,k])):             
                                
                                Path[i,j,k,l] = 0.0  # No path if layer outside region
                                
                            # For other cases, we always subtract two terms to compute traversed distance
                            else:
                                
                                if (r_up[l,j_sector_back_in,k_in] >= r_max[i,j_sector_back_in,k]): 
    
                                    s1 = np.sqrt(r_max_sq[i,j_sector_back_in,k] - b_sq[i])
                                    s2 = 0.0
                                
                                else:   #  elif (r_up[l,j,k_in] < r_max[i,j,k]):
                                    
                                    s2 = np.sqrt(r_up_sq[l,j_sector_back_in,k_in] - b_sq[i])
                                    s1 = 0.0
                                    
                                if (r_low[l,j_sector_back_in,k_in] >= r_min[i,j_sector_back_in,k]): 
                                    
                                    s3 = np.sqrt(r_low_sq[l,j_sector_back_in,k_in] - b_sq[i])
                                    s4 = 0.0
                                    
                                else:     #  elif (r_low[l,j,k_in] < r_min[i,j,k]): 
                                    
                                    s4 = np.sqrt(r_min_sq[i,j_sector_back_in,k] - b_sq[i])
                                    s3 = 0.0
                                
                                # Compute the path distribution (only two terms of s1,2,3,4 are non-zero)
                                Path[i,j,k,l] = symmetry_factor * (s1 + s2 - s3 - s4)/dr[l,j_sector_back_in,k_in]
                                
            # Copy path distribution if sector 'j' is identical to the last
            else:
                
                Path[:,j,:,:] = Path[:,j-1,:,:]
                                
            # Update angular sector pre-computation completion index
            j_sector_last = j_sector_back_in
            
    
    return Path


@jit(nopython=True)
def extend_rad_transfer_grids(phi_edge, theta_edge, R_s, d, R_max, f_cloud, 
                              phi_0, theta_0, N_sectors_back, N_zones_back, 
                              enable_deck, N_phi_max = 36):
    
    ''' Extend the background atmosphere geometric grids (north hemisphere)
        to produce the full geometric grids used for radiative transfer.
        
        This function first duplicates the north hemisphere to symmetrically
        extend to the south hemisphere. Then, additional sectors and zones
        are added for cloudy models at the angular locations where they
        slice the existing background sectors / zones.
        
        In cases where the planet only partially transits the stellar disc
        (e.g. grazing transits, ingress, or egress) the 'N_phi_max' parameter
        specifies how many azimuthal sectors the atmosphere is spatially 
        resolved into. In this case, the nearest background 2D / 3D atmosphere
        sector is placed on the fine grid during radiative transfer.
        
    '''
    
    # Specify maximum number of angles used for azimuthal integration
  #  N_phi_max = 36   # 10 degree slices
    
    #***** Background atmosphere geometry grids *****#
    
    # Given north-south symmetry, rotate azimuthal coordinate by 90 degrees anti-clockwise    
    # Transform phi into phi_prime = pi/2 + phi (angle w.r.t. West equator / -ve y axis)
    
    # Sector boundaries
    phi_edge_N = np.pi/2.0 + phi_edge[:-1]                  # Northern hemisphere
    phi_edge_S = (-1.0*phi_edge_N)[::-1] + 2.0*np.pi        # Southern hemisphere
    phi_edge_back = np.append(phi_edge_N, phi_edge_S)  
    
    # Background zones are specific to a given sector, so no transformations needed   
    theta_edge_back = theta_edge        
    
    #***** Cloud geometry *****#
    
    # We need to add an extra zone if a non-uniform cloud slices an existing zone
    
    # Treat day-night cloud geometry first

    # Convert angle describing onset of clouds to radians (measured from terminator plane towards nightside)
    theta_0 = ((np.pi/180.0) * theta_0)
    
    # The cloud boundary will define a new zone, if it cuts an existing zone
    theta_edge_all = theta_edge_back     # Copy the background zone boundaries
    
    # If the background atmosphere is uniform, add boundary at terminator plane
    if ((N_zones_back == 1) and (enable_deck == 1)):
        theta_edge_all = np.append(theta_edge_all, 0.0)

    # Only define a new zone boundary if the cloud cuts an existing zone
    if (np.any(theta_edge_all == theta_0) == False):
        theta_edge_all = np.append(theta_edge_all, theta_0)

    # Resort zone boundaries in increasing order
    theta_edge_all = np.sort(theta_edge_all)
    
    # Compute angular width of each sector
    dtheta_all = np.diff(theta_edge_all)
                
    # Compute array of angles defining centre of each zone (w.r.t. terminator plane)
    theta_all = (-np.pi/2.0) + np.cumsum(dtheta_all) - (dtheta_all/2.0)
    
    # Redefine theta_all to be the new grid of zones
    theta_grid = theta_all
    
    # Store number of distinct zones for path distribution calculation
    N_zones = len(theta_grid) 

    # Define array specifying which zones contain clouds (0 -> clear, 1-> cloudy)
    cloudy_zones = np.zeros(N_zones).astype(np.int64)
    
    # Count cloudy zones working from dayside to nightside (theta > 0 from terminator plane towards night)
    cloudy_zones[(theta_grid >= theta_0)] = 1
    
    # Initialise array containing zone indices each angular slice falls in
    k_zone_back = np.zeros(shape=(N_zones)).astype(np.int64)
    
    # For each zone angle
    for k in range(N_zones):
        
        # Find the zone in which this angle lies, store for later referencing in radiative transfer
        k_zone_back[k] = prior_index(theta_grid[k], theta_edge_back, 0)   # Background atmosphere only 
    
    # Now treat azimuthal evening-morning cloud geometry
    
    # Consider how many extra sectors are required to deal with clouds
    
    # Extract azimuthal angles describing cloud start and angular extent
    phi_0, phi_c = ((np.pi/180.0) * phi_0), (2.0*np.pi * f_cloud)
    
    # Transform phi into phi_prime = pi/2 + phi (angle w.r.t. West equator / -ve y axis)
    phi_0 = np.pi/2.0 + phi_0   # Not needed for phi_c, since that is a difference / angular extent of the cloud, so same for phi_prime_c
    
    # Account for cyclic boundary condition, making phi_prime +ve (e.g. for phi < -90)
    phi_0 = np.mod(phi_0, 2.0*np.pi)  
    
    # Store end point of cloud deck 
    phi_end = np.mod((phi_0 + phi_c), 2.0*np.pi)
    
    # The cloud boundaries define new sectors, if they lie between existing boundaries
    phi_edge_all = phi_edge_back     # Copy the background sector boundaries

    # Only define a new sector boundary if the cloud cuts an existing sector
    if (np.any(phi_edge_back == phi_0) == False):
        phi_edge_all = np.append(phi_edge_all, phi_0)
    if (np.any(phi_edge_back == phi_end) == False):
        phi_edge_all = np.append(phi_edge_all, phi_end)

    # Resort sector boundaries in increasing order
    phi_edge_all = np.sort(phi_edge_all)
    
    # Compute angular width of each sector
    dphi_all = np.diff(phi_edge_all)
                
    # Compute array of angles defining centre of each sector (w.r.t. west equator)
    phi_all = 0.0 + np.cumsum(dphi_all) - (dphi_all/2.0)
    
    # Store number of distinct sectors for azimuthal integrals
    N_sectors = len(phi_all) 

    # Define array specifying which sectors contain clouds (0 -> clear, 1-> cloudy)
    cloudy_sectors = np.zeros(N_sectors).astype(np.int64)
    
    # Count cloudy sectors clockwise where cloud does not cover West equator, anti-clockwise if it does
    if ((phi_0 + phi_c) < 2.0*np.pi):      # Cloud does not loop around and cover West equator
        cloudy_sectors[(phi_all >= phi_0) & (phi_all <= phi_end)] = 1   # Clockwise
    elif ((phi_0 + phi_c) >= 2.0*np.pi):   # Cloud covers the West equator
        cloudy_sectors[(phi_all <= phi_end) | (phi_all >= phi_0)] = 1   # Anti-clockwise

    # With the new axial grid created, consider if it needs to have a finer
    # resolution in cases where the planet partially overlaps the star
    
    # If planet entirely overlaps star
    if (d <= (R_s - R_max)):
        
        N_phi = N_sectors         # Angular integration is effectively just an average in this case        
        dphi_grid = dphi_all
        phi_grid = phi_all
        
    # If planet partially overlaps star
    elif (d > (R_s - R_max)):
            
        N_phi = N_phi_max                                    # Divide terminator into slices defined by N_phi_max (top of this function)
        dphi_0 = (2.0*np.pi)/N_phi                           # Polar angle integration element resolution
        dphi_grid = dphi_0 * np.ones(N_phi)                  # Polar integration element array (trivial here, all elements dphi_0)
        phi_grid = np.cumsum(dphi_grid) - (dphi_grid/2.0)    # Angles in centre of each area element
  
    # Now create arrays storing which original sector and background sector
    # a given angle lies in (to avoid computing transmissivities multiple times)
    
    # Initialise array containing sector indices each angular slice falls in
    j_sector = np.zeros(shape=(N_phi)).astype(np.int64)
    j_sector_back = np.zeros(shape=(N_phi)).astype(np.int64)
    
    # For each polar angle
    for j in range(N_phi):
        
        # Find the terminator sector in which this angle lies
        j_sector_in = prior_index(phi_grid[j], phi_edge_all, 0)         # All sectors (including clouds)
        j_sector_back_in = prior_index(phi_grid[j], phi_edge_back, 0)   # Background atmosphere only 
        
        # Find equivalent background sector in northern hemisphere
        if (j_sector_back_in >= N_sectors_back):
            j_sector_back_in = 2*(N_sectors_back - 1) - j_sector_back_in
            
        # Store sector indices for later referencing in radiative transfer
        j_sector[j] = j_sector_in                
        j_sector_back[j] = j_sector_back_in
        
    
    return phi_grid, dphi_grid, theta_grid, theta_edge_all, N_sectors, N_zones, \
           N_phi, j_sector, j_sector_back, k_zone_back, cloudy_sectors, cloudy_zones
        


@jit(nopython=True)
def compute_tau_vert(N_phi, N_layers, N_zones, N_wl, j_sector, j_sector_back,
                     k_zone_back, cloudy_zones, cloudy_sectors, kappa_clear, 
                     kappa_cloud, dr):
    
    ''' Compute the vertical optical depth array.
    
    '''
    
    tau_vert = np.zeros(shape=(N_layers, N_phi, N_zones, N_wl))

    # For each sector around terminator
    for j in range(N_phi):
        
        # Refresh sector count
        j_sector_last = -1  # This counts the angular index where the transmissivity was last computed
        
        # Find which asymmetric terminator sector this angle lies in
        j_sector_in = j_sector[j]
        j_sector_back_in = j_sector_back[j]
        
        # If tau_vert has not yet been computed for this sector
        if (j_sector_in != j_sector_last):
        
            # For each zone along line of sight
            for k in range(N_zones):
                
                # Extract index of background atmosphere this sub-zone is in
                k_zone_back_in = k_zone_back[k]   # Only differs from k when a cloud splits a zone
            
                # If zone and sector angles lie within cloudy region
                if ((cloudy_zones[k] == 1) and             # If zone is cloudy and
                    (cloudy_sectors[j_sector_in] == 1)):   # If sector is cloudy   
                    
                    # For each wavelength
                    for q in range(N_wl):
                            
                        # Populate vertical optical depth tensor
                        tau_vert[:,j,k,q] = ((kappa_clear[:,j_sector_back_in,k_zone_back_in,q] +
                                              kappa_cloud[:,j_sector_back_in,k_zone_back_in,q]) * 
                                              dr[:,j_sector_back_in,k_zone_back_in])
                            
                # For clear regions, do not need to add cloud opacity
                elif ((cloudy_zones[k] == 0) or              # If zone is clear or
                      (cloudy_sectors[j_sector_in] == 0)):   # If sector is clear  
                
                    # For each wavelength
                    for q in range(N_wl):
                        
                        # Populate vertical optical depth tensor
                        tau_vert[:,j,k,q] = (kappa_clear[:,j_sector_back_in,k_zone_back_in,q] * 
                                             dr[:,j_sector_back_in,k_zone_back_in])
                        
        # Copy tau_vert if sector 'j' is identical to the last
        else:
            
            tau_vert[:,j,:,:] = tau_vert[:,j-1,:,:]
                            
        # Update angular sector pre-computation completion index
        j_sector_last = j_sector_in
        
        
    return tau_vert


@jit(nopython = True)
def delta_ray_geom(N_phi, N_b, b, b_p, y_p, phi_grid, R_s_sq):
    
    ''' Compute the ray tracing factor in the geometric limit.
    
    '''
    
    delta_ray = np.zeros(shape=(N_b, N_phi))
    
    # For each polar angle
    for j in range(N_phi):

        # For each atmospheric layer
        for i in range(N_b):
            
            # Compute distance from stellar centre to centre of area element
            d_ij_sq = (b[i]**2 + b_p**2 + y_p**2 + 
                       2.0*b[i]*(b_p*np.cos(phi_grid[j] - np.pi/2.0) + 
                                 y_p*np.sin(phi_grid[j] - np.pi/2.0)))
            
            # If planet area element has star in the background
            if (d_ij_sq <= R_s_sq):
                
                # Ray traces back to stellar surface => 1
                delta_ray[i,j] = 1.0
                
            # If area element falls off stellar surface
            else:
                
                # No illumination => 0
                delta_ray[i,j] = 0.0   
                
    return delta_ray


#@jit(nopython = True)
def TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud, enable_deck, 
            enable_haze, b_p, y_p, R_s, f_cloud, phi_0, theta_0, phi_edge, theta_edge):

    ''' Compute the transit depth in the geometric limit by solving the
        radiative transfer equation through the planetary atmosphere.
        
        V3.0: Tensor implementation from MacDonald & Lewis (2022)
       
        Inputs:
           
        b_p => impact parameter of planetary orbit
        y_p => perpendicular distance from planet centre to point where d = b_p
               (alternatively, y coordinate of planet centre in stellar z-y plane)
        phi_0 => reduced angle where cloud deck begins relative to z axis (radians/(2*pi))
        phi_c => reduced angular extent of cloud deck (radians/(2*pi))
        P => pressure of each layer
        r => radius at centre of each layer
        r_up => radius of top edge of each layer
        r_low => radius of top edge of each layer
        dr => radial thickness of each layer
        R_p =>
        n => atmosphere number density grid
        X => atmosphere volume mixing ratios
        wl => model wavelength grid
        kappa_chem => extinction coefficient due to chemistry
        kappa_Rayleigh => extinction coefficient due to Rayleigh scattering
        kappa_haze => extinction coefficient due to a scattering haze
        kappa_cloud => extinction coefficient due to a grey cloud opacity
        enable_deck => 1 if opaque cloud deck enabled
        enable_haze => 1 if scattering haze enabled
        offset => additive transit depth offset
        N_sectors_back => 
       
        Outputs:
           
        trans_depth => transmission spectrum
       
    '''
    
    #***** Step 1: Initialise key quantities *****#
    
    # Compute projected distance from stellar centre to planet centre (z-y plane)
    d_sq = (b_p**2 + y_p**2)
    d = np.sqrt(d_sq) 
    
    # Load number of wavelengths where transit depth desired
    N_wl = len(wl)
    
    # Initialise transit depth array
    transit_depth = np.zeros(shape=(N_wl))
    
    # Compute squared stellar radius
    R_s_sq = R_s*R_s

    # Store number of layers
    N_layers = len(P)
    
    # Find index of deep pressure below which atmosphere is homogenous (usually 10 bar)
    i_bot = 0   #np.argmin(np.abs(P - P_deep))
    
    # Find index of dayside sector with maximum radial extent
    j_top = np.argmax(r[-1,:,0])
    
    # Compute maximum radial extent of atmosphere
    R_max = r_up[-1,j_top,0]    # Maximal radial extent across all sectors
    R_max_sq = R_max*R_max      # Squared maximal radial extent
    
    # Initialise impact parameter array to radial array in sector with maximal extent
    b = r_up[:,j_top,0]    # Impact parameters given by upper layer boundaries in dayside
    db = dr[:,j_top,0]     # Differential impact parameter array 
    N_b = b.shape[0]       # Length of impact parameter array  
  
  #  N_b = 1000
  #  b = np.linspace(r[0,j_top,0], r[-1,j_top,0], N_b)
  #  db = np.diff(b, append=[(b[-1] + (b[-1] - b[-2]))])
        
    # Print maximum terminator opening angle that can be probed
 #   beta_max = (2.0*np.arccos(b[0]/R_max))*(180.0/np.pi)   
 #   print("Beta_max = " + str(beta_max))
    
    # Store number of distinct background sectors for azithumal integrals
    N_sectors_back = r.shape[1] 
    
    # Store number of distinct background zones
    N_zones_back = r.shape[2]
 
    # Extend the north hemisphere grid to account for both hemispheres and clouds
    phi_grid, dphi_grid, theta_grid, \
    theta_edge_all, N_sectors, N_zones, \
    N_phi, j_sector, j_sector_back, \
    k_zone_back, cloudy_sectors, \
    cloudy_zones = extend_rad_transfer_grids(phi_edge, theta_edge, R_s, d, R_max,
                                             f_cloud, phi_0, theta_0, N_sectors_back, 
                                             N_zones_back, enable_deck)
        
    #***** Step 2: Compute planetary area overlapping the star *****#
    
    # If planet does not overlap star, do not need to do any computations
    if (d >= (R_s + R_max)):
        
        return np.zeros(shape=(N_wl)) # Transit depth zero at all wavelengths
    
    # If planet fully overlaps star
    elif (d <= (R_s - R_max)):
        
        # Area of overlap just pi*R_max^2 in this case
        A_overlap = np.pi * R_max_sq
        
    # In all other cases with partial overlap (e.g. ingress/egress or grazing)
    elif (d > (R_s - R_max)) and (d < (R_s + R_max)):
        
        # Compute angles from star-planet line to R_p = R_s intersection
        phi_1 = np.arccos((d_sq + R_max_sq - R_s_sq)/(2 * d * R_max))  # Angle at planet centre
        phi_2 = np.arccos((d_sq + R_s_sq - R_max_sq)/(2 * d * R_s))    # Angle at star centre
            
        # Evaluate the overlapping area analytically
        A_overlap = (R_max_sq * (phi_1 - 0.5 * np.sin(2.0 * phi_1)) +
                     R_s_sq   * (phi_2 - 0.5 * np.sin(2.0 * phi_2)))
        
    #***** Step 3: Calculate the delta_ray matrix *****#
    
    delta_ray = delta_ray_geom(N_phi, N_b, b, b_p, y_p, phi_grid, R_s_sq)
         
    #***** Step 4: Calculate atmosphere area matrices *****#

    # Populate elements of atmosphere area matrix
    dA_atm = np.outer((b * db), dphi_grid)
                
    # Find overlapping area matrix of atmosphere (zero if rays don't intersect the star)
    dA_atm_overlap = delta_ray * dA_atm
    
    #***** Step 5: Calculate path distribution tensor *****#
    
    Path = path_distribution_geometric(b, r_up, r_low, dr, i_bot, j_sector_back, 
                                       N_layers, N_sectors_back, N_zones_back, 
                                       N_zones, N_phi, k_zone_back, theta_edge_all)
   
    #***** Step 6: Calculate vertical optical depth tensor *****#
    
    tau_vert = compute_tau_vert(N_phi, N_layers, N_zones, N_wl, j_sector, 
                                j_sector_back, k_zone_back, cloudy_zones, 
                                cloudy_sectors, kappa_clear, kappa_cloud, dr)

    #***** Step 7: Calculate transmittance tensor *****#
        
    Trans = np.zeros(shape=(N_b, N_phi, N_wl))
    
    # For each sector around terminator
    for j in range(N_phi):
        
        # Refresh sector count
        j_sector_last = -1  # This counts the angular index where the transmissivity was last computed
        
        # Find which asymmetric terminator sector this angle lies in
        j_sector_in = j_sector[j]
        
        # If transmittance has not yet been computed for this sector
        if (j_sector_in != j_sector_last):

            Trans[:,j,:] = np.exp(-1.0*np.tensordot(Path[:,j,:,:], tau_vert[:,j,:,:], axes=([2,1],[0,1])))
        
        # Copy transmittance if sector 'j' is identical to the last
        else:

            Trans[:,j,:] = Trans[:,j-1,:]
                            
        # Update angular sector pre-computation completion index
        j_sector_last = j_sector_in

    # Delete vertical optical depth and path distribution tensors to free memory
    del tau_vert, Path
    
    #***** Step 8: Finally, compute the transmission spectrum *****#
    
    # Calculate effective overlapping area of atmosphere at each wavelength
    A_atm_overlap_eff = np.tensordot(Trans, dA_atm_overlap, axes=([0,1],[0,1]))

    # Compute the transmission spectrum
    transit_depth = (A_overlap - A_atm_overlap_eff)/(np.pi * R_s_sq)
                
    
    return transit_depth


