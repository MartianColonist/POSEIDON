''' 
Radiative transfer calculations for generating emission spectra.

'''

import numpy as np
import scipy.constants as sc
from numba import jit, cuda
import math
import os

from .utility import mock_missing, interp_GPU

try:
    import cupy as cp
except ImportError:
    cp = mock_missing('cupy')

block = int(os.environ['block'])
thread = int(os.environ['thread'])


@jit(nopython = True)
def planck_lambda_arr(T, wl):
    '''
    Compute the Planck function spectral radiance for a range of model
    wavelengths and atmospheric temperatures.

    Args:
        T (np.array of float):
            Array of temperatures in each atmospheric layer (K).
        wl (np.array of float): 
            Wavelength grid (μm).
    
    Returns:
        B_lambda (2D np.array of float):
            Planck function spectral radiance as a function of layer temperature
            and wavelength in SI units (W/m^2/sr/m).

    '''
    
    # Define Planck function array
    B_lambda = np.zeros(shape=(len(T),len(wl)))  # (Temperature, wavelength)

    # Convert wavelength array to m
    wl_m = wl * 1.0e-6
    
    # Second radiative constant
    c_2 = (sc.h * sc.c) / sc.k
    
    # For each wavelength
    for k in range(len(wl)):
        
        # Precompute Planck function coefficient prefactor
        coeff = (2.0 * sc.h * sc.c**2) / (wl_m[k]**5)
        
        # For each atmospheric layer
        for i in range(len(T)):
            
            # Evaluate Planck function spectral radiance
            B_lambda[i,k] = coeff * (1.0 / (np.exp(c_2 / (wl_m[k] * T[i])) - 1.0))
            
    return B_lambda


@cuda.jit
def planck_lambda_arr_GPU(T, wl, B_lambda):
    '''
    GPU variant of the 'planck_lambda_arr' function.

    Compute the Planck function spectral radiance for a range of model
    wavelengths and atmospheric temperatures.
    Args:
        T (np.array of float):
            Array of temperatures in each atmospheric layer (K).
        wl (np.array of float): 
            Wavelength grid (μm).
    
    Returns:
        B_lambda (2D np.array of float):
            Planck function spectral radiance as a function of layer temperature
            and wavelength in SI units (W/m^2/sr/m).
    '''
    
    thread = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    # Second radiative constant
    c_2 = (sc.h * sc.c) / sc.k
    
    # For each wavelength
    for k in range(thread, len(wl), stride):
        
        # Precompute Planck function coefficient prefactor
        coeff = (2.0 * sc.h * sc.c**2) / (wl[k]**5)
        
        # For each atmospheric layer
        for i in range(len(T)):
            
            # Evaluate Planck function spectral radiance
            B_lambda[i,k] = coeff * (1.0 / (math.exp(c_2 / (wl[k] * T[i])) - 1.0))


@jit(nopython = True)
def emission_single_stream(T, dz, wl, kappa, Gauss_quad = 2):
    '''
    Compute the emergent top-of-atmosphere flux from a planet or brown dwarf.

    This function  considers only pure thermal emission (i.e. no scattering).

    Args:
        T (np.array of float):
            Temperatures in each atmospheric layer (K).
        dz (np.array of float):
            Vertical extent of each atmospheric layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        kappa (2D np.array of float):
            Extinction coefficient in each layer as a function of wavelength (m^-1).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            (Options: 2 / 3).
    
    Returns:
        F (np.array of float):
            Spectral surface flux in SI units (W/m^2/sr/m).

    '''
    
    # Load weights and cos(theta) values for desired Gaussian quadrature scheme
    if (Gauss_quad == 2):
        W = np.array([0.5, 0.5])
        mu = np.array([0.5 - 0.5*np.sqrt(1.0/3.0), 0.5 + 0.5*np.sqrt(1.0/3.0)])
    elif (Gauss_quad == 3):
        W = np.array([5.0/18.0, 4.0/9.0, 5.0/18.0])
        mu = np.array([0.5 - 0.5*np.sqrt(3.0/5.0), 0.5, 0.5 + 0.5*np.sqrt(3.0/5.0)])
    
    # Calculate Planck function in each layer and each wavelength
    B = planck_lambda_arr(T, wl)
    
    # Initial intensity at the base of the atmosphere is a Planck function 
    I = np.ones(shape=(len(mu),len(wl))) * B[0,:]
    
    # Initialise surface flux array
    F = np.zeros(len(wl))

    # Initialise differential optical depth array
    dtau = np.zeros(shape=(len(T), len(wl)))
    
    # For each wavelength
    for k in range(len(wl)):
    
        # For each ray travelling at mu = cos(theta)
        for j in range(len(mu)):

            # For each atmospheric layer
            for i in range(len(T)):
    
                # Compute vertical optical depth across the layer
                dtau_vert = kappa[i,k] * dz[i]
                dtau[i,k] = dtau_vert
                
                # Compute transmissivity of the layer
                Trans = np.exp((-1.0 * dtau_vert)/mu[j])
                
                # Solve for emergent intensity from the layer top
                I[j,k] = Trans * I[j,k] + (1.0 - Trans) * B[i,k]
                
            # Add contribution of this ray/angle to the surface flux
            F[k] += 2.0 * np.pi * mu[j] * I[j,k] * W[j]
    
    return F, dtau

@jit(nopython = True)
def emission_single_stream_w_albedo(T, dz, wl, kappa, Gauss_quad = 2, 
                                    surf_reflect = [], index_below_P_surf = 0):
    '''
    Compute the emergent top-of-atmosphere flux from a planet or brown dwarf.

    This function  considers only pure thermal emission (i.e. no scattering).

    Args:
        T (np.array of float):
            Temperatures in each atmospheric layer (K).
        dz (np.array of float):
            Vertical extent of each atmospheric layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        kappa (2D np.array of float):
            Extinction coefficient in each layer as a function of wavelength (m^-1).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            (Options: 2 / 3).
        surf_reflect : numpy.ndarray    
            Surface reflectivity as a function of wavelength.
        index_below_P_surf : int
            Index below P_surf, so that the blackbody can be computed. 
            (Note that P_surf can be between two pressure levels, we take the lower one)
    
    Returns:
        F (np.array of float):
            Spectral surface flux in SI units (W/m^2/sr/m).

    '''
    
    # Load weights and cos(theta) values for desired Gaussian quadrature scheme
    if (Gauss_quad == 2):
        W = np.array([0.5, 0.5])
        mu = np.array([0.5 - 0.5*np.sqrt(1.0/3.0), 0.5 + 0.5*np.sqrt(1.0/3.0)])
    elif (Gauss_quad == 3):
        W = np.array([5.0/18.0, 4.0/9.0, 5.0/18.0])
        mu = np.array([0.5 - 0.5*np.sqrt(3.0/5.0), 0.5, 0.5 + 0.5*np.sqrt(3.0/5.0)])

    # Surface Emissivity 
    emissivity = 1.0 - surf_reflect #Emissivity is 1 - surface reflectivity

    # Calculate Planck function in each layer and each wavelength
    B = planck_lambda_arr(T, wl)

    # The blackbody at the hard surface/top of the cloud deck is 
    # emissivity * blackbody 
    B[index_below_P_surf,:] = B[index_below_P_surf,:]*emissivity 
    
    # Initial intensity at the base of the atmosphere is a Planck function 
    I = np.ones(shape=(len(mu),len(wl))) * B[0,:]
    
    # Initialise surface flux array
    F = np.zeros(len(wl))

    # Initialise differential optical depth array
    dtau = np.zeros(shape=(len(T), len(wl)))
    
    # For each wavelength
    for k in range(len(wl)):
    
        # For each ray travelling at mu = cos(theta)
        for j in range(len(mu)):

            # For each atmospheric layer
            for i in range(len(T)):
    
                # Compute vertical optical depth across the layer
                dtau_vert = kappa[i,k] * dz[i]
                dtau[i,k] = dtau_vert
                
                # Compute transmissivity of the layer
                Trans = np.exp((-1.0 * dtau_vert)/mu[j])
                
                # Solve for emergent intensity from the layer top
                I[j,k] = Trans * I[j,k] + (1.0 - Trans) * B[i,k]
                
            # Add contribution of this ray/angle to the surface flux
            F[k] += 2.0 * np.pi * mu[j] * I[j,k] * W[j]
    
    return F, dtau


def emission_single_stream_GPU(T, dz, wl, kappa, Gauss_quad = 2):
    '''
    GPU variant of the 'emission_rad_transfer' function.

    Compute the emergent top-of-atmosphere flux from a planet or brown dwarf.

    This function considers only pure thermal emission (i.e. no scattering).

    Args:
        T (np.array of float):
            Temperatures in each atmospheric layer (K).
        dz (np.array of float):
            Vertical extent of each atmospheric layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        kappa (2D np.array of float):
            Extinction coefficient in each layer as a function of wavelength (m^-1).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            (Options: 2 / 3).
    
    Returns:
        F (np.array of float):
            Spectral surface flux in SI units (W/m^2/sr/m).
    '''
    
    # Load weights and cos(theta) values for desired Gaussian quadrature scheme
    if (Gauss_quad == 2):
        W = cp.array([0.5, 0.5])
        mu = cp.array([0.5 - 0.5*cp.sqrt(1.0/3.0), 0.5 + 0.5*cp.sqrt(1.0/3.0)])
    elif (Gauss_quad == 3):
        W = cp.array([5.0/18.0, 4.0/9.0, 5.0/18.0])
        mu = cp.array([0.5 - 0.5*cp.sqrt(3.0/5.0), 0.5, 0.5 + 0.5*cp.sqrt(3.0/5.0)])
    
    # Define temperature, wavelength, and Planck function arrays on GPU
    T = cp.asarray(T)
    wl_m = cp.asarray(wl) * 1.0e-6
    B = cp.zeros((len(T), len(wl_m)))

    # Calculate Planck function in each layer and each wavelength    
    planck_lambda_arr_GPU[block, thread](T, wl_m, B)
    
    # Initial intensity at the base of the atmosphere is a Planck function 
    I = cp.ones(shape=(len(mu),len(wl))) * B[0,:]
    F_p = cp.zeros(len(wl))
    
    # Initialise differential optical depth array
    dtau = cp.zeros(shape=(len(T), len(wl)))
    
    @cuda.jit
    def helper_kernel(T, dz, wl, kappa, W, mu, B, I, F, dtau):
        thread = cuda.grid(1)
        stride = cuda.gridsize(1)

        # For each wavelength
        for k in range(thread, len(wl), stride):

            # For each ray travelling at mu = cos(theta)
            for j in range(len(mu)):

                # For each atmospheric layer
                for i in range(len(T)):
        
                    # Compute vertical optical depth across the layer
                    dtau_vert = kappa[i,k] * dz[i]
                    dtau[i,k] = dtau_vert
                    
                    # Compute transmissivity of the layer
                    Trans = math.exp((-1.0 * dtau_vert)/mu[j])
                    
                    # Solve for emergent intensity from the layer top
                    I[j,k] = Trans * I[j,k] + (1.0 - Trans) * B[i,k]
                    
                # Add contribution of this ray/angle to the surface flux
                cuda.atomic.add(F, k, 2.0 * math.pi * mu[j] * I[j,k] * W[j])

    helper_kernel[block, thread](T, cp.asarray(dz), cp.asarray(wl), kappa, W, mu, B, I, F_p, dtau)

    return cp.asnumpy(F_p), dtau


@jit(nopython = True)
def determine_photosphere_radii(dtau, r_low, wl, photosphere_tau = 2/3):
    '''
    Interpolate optical depth to find the radius corresponding to the
    photosphere (by default at tau = 2/3).

    Args:
        dtau (2D np.array of float):
            Vertical optical depth across each layer (starting from the top of
            the atmosphere) as a function of layer and wavelength.
        r_low (np.array of float):
            Radius at the lower boundary of each layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        photosphere_tau (float):
            Optical depth to determine photosphere radius.
    
    Returns:
        R_p_eff (np.array of float):
            Photosphere radius as a function of wavelength (m).

    '''

    # Initialise photosphere radius array
    R_p_eff = np.zeros(len(wl))

    # Calculate photosphere radius at tau = 2/3 for each wavelength
    for k in range(len(wl)):

        # Find cumulative optical depth from top of atmosphere down at each wavelength
        tau_lambda = np.cumsum(dtau[:,k])

        # Interpolate layer boundary radii to find radius where tau = 2/3 (lower because we are integrating down) 
        R_p_eff[k] = np.interp(photosphere_tau, tau_lambda, r_low)

    return R_p_eff


@cuda.jit
def determine_photosphere_radii_GPU(tau_lambda, r_low, wl, R_p_eff, photosphere_tau = 2/3):
    '''
    GPU variant of the 'determine_photosphere_radii' function.

    Interpolate optical depth to find the radius corresponding to the
    photosphere (by default at tau = 2/3).

    Args:
        dtau (2D np.array of float):
            Vertical optical depth across each layer (starting from the top of
            the atmosphere) as a function of layer and wavelength.
        r_low (np.array of float):
            Radius at the lower boundary of each layer (m).
        wl (np.array of float): 
            Wavelength grid (μm).
        photosphere_tau (float):
            Optical depth to determine photosphere radius.
    
    Returns:
        R_p_eff (np.array of float):
            Photosphere radius as a function of wavelength (m).
    '''
    
    thread = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Calculate photosphere radius at tau = 2/3 for each wavelength
    for k in range(thread, len(wl), stride):

        # Interpolate layer boundary radii to find radius where tau = 2/3 (lower because we are integrating down) 
        R_p_eff[k] = interp_GPU(photosphere_tau, tau_lambda[:,k], r_low)




##### Functions adapted from PICASO (https://github.com/natashabatalha/picaso) #####
# IMPORTANT : WE DON"T INCLUDE RAMAN SCATTERING (yet)

@jit(nopython=True, cache=True, fastmath=True)
def slice_gt(array, lim):
    '''
    Function to replace values with upper or lower limit
    '''

    for i in range(array.shape[0]):
        new = array[i,:] 
        new[np.where(new>lim)] = lim
        array[i,:] = new     
    return array


@jit(nopython=True, cache=True)
def setup_tri_diag(N_layer, N_wl ,c_plus_up, c_minus_up, 
                   c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                   gamma, dtau, exptrm_positive,  exptrm_minus):
    """
    Before we can solve the tridiagonal matrix (See Toon+1989) section
    "SOLUTION OF THE TwO-STREAM EQUATIONS FOR MULTIPLE LAYERS", we 
    need to set up the coefficients. 
    Parameters
    ----------
    N_layer : int 
        number of layers in the model 
    N_wl : int 
        number of wavelength points
    c_plus_up : array 
        c-plus evaluated at the top of the atmosphere 
    c_minus_up : array 
        c_minus evaluated at the top of the atmosphere 
    c_plus_down : array 
        c_plus evaluated at the bottom of the atmosphere 
    c_minus_down : array 
        c_minus evaluated at the bottom of the atmosphere 
    b_top : array 
        The diffuse radiation into the model at the top of the atmosphere
    b_surface : array
        The diffuse radiation into the model at the bottom. Includes emission, reflection 
        of the unattenuated portion of the direct beam  
    surf_reflect : array 
        Surface reflectivity 
    g1 : array 
        table 1 toon et al 1989
    g2 : array 
        table 1 toon et al 1989
    g3 : array 
        table 1 toon et al 1989
    lamba : array 
        Eqn 21 toon et al 1989 
    gamma : array 
        Eqn 22 toon et al 1989
    dtau : array 
        Opacity per layer
    exptrm_positive : array 
        Eqn 44, exponential terms needed for tridiagonal rotated layered, clipped at 35 
    exptrm_minus : array 
        Eqn 44, exponential terms needed for tridiagonal rotated layered, clipped at 35 
    Returns
    -------
    array 
        coefficient of the positive exponential term 
    
    """
    L = 2 * N_layer

    #EQN 44 

    e1 = exptrm_positive + gamma*exptrm_minus
    e2 = exptrm_positive - gamma*exptrm_minus
    e3 = gamma*exptrm_positive + exptrm_minus
    e4 = gamma*exptrm_positive - exptrm_minus


    #now build terms 
    A = np.zeros((L,N_wl)) 
    B = np.zeros((L,N_wl )) 
    C = np.zeros((L,N_wl )) 
    D = np.zeros((L,N_wl )) 

    A[0,:] = 0.0
    B[0,:] = gamma[0,:] + 1.0
    C[0,:] = gamma[0,:] - 1.0
    D[0,:] = b_top - c_minus_up[0,:]

    #even terms, not including the last !CMM1 = UP
    A[1::2,:][:-1] = (e1[:-1,:]+e3[:-1,:]) * (gamma[1:,:]-1.0) #always good
    B[1::2,:][:-1] = (e2[:-1,:]+e4[:-1,:]) * (gamma[1:,:]-1.0)
    C[1::2,:][:-1] = 2.0 * (1.0-gamma[1:,:]**2)          #always good 
    D[1::2,:][:-1] =((gamma[1:,:]-1.0)*(c_plus_up[1:,:] - c_plus_down[:-1,:]) + 
                            (1.0-gamma[1:,:])*(c_minus_down[:-1,:] - c_minus_up[1:,:]))
    #import pickle as pk
    #pk.dump({'GAMA_1':(gama[1:,:]-1.0), 'CPM1':c_plus_up[1:,:] , 'CP':c_plus_down[:-1,:], '1_GAMA':(1.0-gama[1:,:]), 
    #   'CM':c_minus_down[:-1,:],'CMM1':c_minus_up[1:,:],'Deven':D[1::2,:][:-1]}, open('../testing_notebooks/GFLUX_even_D_terms.pk','wb'))
    
    #odd terms, not including the first 
    A[::2,:][1:] = 2.0*(1.0-gamma[:-1,:]**2)
    B[::2,:][1:] = (e1[:-1,:]-e3[:-1,:]) * (gamma[1:,:]+1.0)
    C[::2,:][1:] = (e1[:-1,:]+e3[:-1,:]) * (gamma[1:,:]-1.0)
    D[::2,:][1:] = (e3[:-1,:]*(c_plus_up[1:,:] - c_plus_down[:-1,:]) + 
                            e1[:-1,:]*(c_minus_down[:-1,:] - c_minus_up[1:,:]))

    #last term [L-1]
    A[-1,:] = e1[-1,:]-surf_reflect*e3[-1,:]
    B[-1,:] = e2[-1,:]-surf_reflect*e4[-1,:]
    C[-1,:] = 0.0
    D[-1,:] = b_surface-c_plus_down[-1,:] + surf_reflect*c_minus_down[-1,:]

    return A, B, C, D


@jit(nopython=True, cache=True)
def tri_diag_solve(l, a, b, c, d):
    ''''
    Tridiagonal Matrix Algorithm solver, a b c d can be NumPy array type or Python list type.
    refer to this wiki_ and to this explanation_. 
    
    .. _wiki: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    .. _explanation: http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    
    A, B, C and D refer to: 
    .. math:: A(I)*X(I-1) + B(I)*X(I) + C(I)*X(I+1) = D(I)
    This solver returns X. 
    Parameters
    ----------
    A : array or list 
    B : array or list 
    C : array or list 
    C : array or list 
    Returns
    -------
    array 
        Solution, x 
    '''

    AS, DS, CS, DS,XK = np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l), np.zeros(l) # copy arrays

    AS[-1] = a[-1]/b[-1]
    DS[-1] = d[-1]/b[-1]

    for i in range(l-2, -1, -1):
        x = 1.0 / (b[i] - c[i] * AS[i+1])
        AS[i] = a[i] * x
        DS[i] = (d[i]-c[i] * DS[i+1]) * x
    XK[0] = DS[0]
    for i in range(1,l):
        XK[i] = DS[i] - AS[i] * XK[i-1]
    return XK


@jit(nopython=True, cache=True)
def emission_Toon(P, T, wl, dtau_tot, 
                  kappa_Ray, kappa_cloud, kappa_tot,
                  w_cloud, g_cloud, zone_idx,
                  surf_reflect,
                  kappa_cloud_seperate,
                  hard_surface = 0, tridiagonal = 0, 
                  Gauss_quad = 5, numt = 1,
                  T_surf = 0,):
    
    ###############################################
    # ORIGINAL PICASO PREAMBLE (fluxes.py, get_thermal_1d())
    ###############################################
    '''
    This function uses the source function method, which is outlined here : 
    https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    
    The result of this routine is the top of the atmosphere thermal flux as 
    a function of gauss and chebychev points across the disk. 
    Everything here is in CGS units:
    Fluxes - erg/s/cm^3
    Temperature - K 
    Wave grid - cm-1
    Pressure ; dyne/cm2
    Reminder: Flux = pi * Intensity, so if you are trying to compare the result of this with 
    a black body you will need to compare with pi * BB !

    Adapted from get_thermal_1d
    
    Parameters
    ----------
    nlevel : int 
        Number of levels which occur at the grid points (not to be confused with layers which are
        mid points)
    wno : numpy.ndarray
        Wavenumber grid in inverse cm 
    N_wl : int 
        Number of wavenumber points 
    numg : int 
        Number of gauss points (think longitude points)
    numt : int 
        Number of chebychev points (think latitude points)
    tlevel : numpy.ndarray
        Temperature as a function of level (not layer)
    dtau : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the per layer optical depth. 
    w0 : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the single scattering albedo of 
        the atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    cosb : numpy.ndarray
        This is a matrix of nlayer by nwave. This describes the asymmetry of the 
        atmosphere. Note this is free of any Raman scattering or any d-eddington correction 
        that is sometimes included in reflected light calculations.
    plevel : numpy.ndarray
        Pressure for each level (not layer, which is midpoints). CGS units (dyne/cm2)
    ubar1 : numpy.ndarray
        This is a matrix of ng by nt. This describes the outgoing incident angles and is generally
        computed in `picaso.disco`
    surf_reflect : numpy.ndarray    
        Surface reflectivity as a function of wavenumber. 
    hard_surface : int
        0 for no hard surface (e.g. Jupiter/Neptune), 1 for hard surface (terrestrial)
    tridiagonal : int 
        0 for tridiagonal, 1 for pentadiagonal
    Returns
    -------
    numpy.ndarray
        Thermal flux in CGS units (erg/cm3/s) in a matrix that is 
        numg x numt x nwno
    '''
    
    ###############################################
    # POSEIDON PREAMBLE
    ###############################################

    # From PICASO we redefine we define or compute the following original inputs : 
    # 
    # nlevel as N_layer + 1 = len(P) + 1
    # wno = 1/wl
    # nwno as N_wl = len(wl)
    # 
    # numg = Num Gauss angles = Gauss_quad = 5
    # numt = Num Chebychev angles =  1 
    # we define our system to default to the 1x10 scheme where tangle = 0 and gangle = 5. See : 
    # https://natashabatalha.github.io/picaso/notebooks/8_SphericalIntegration.html?highlight=geometry
    #
    # tlevel = T_level and is converted below 
    #
    # dtau = dtau_tot (differential extinction optical depth (from absorption + scattering) across each later)
    # IMPORTANT : We only consider a 1D model, meaning the dtau and taus don't vary with gauss weight
    # 
    # w0 = w_tot      calculated below (weighted, combined single scattering albedo)
    # cosb = g_tot    calculated below (weighted, combined scattering asymmetry parameter)
    # 
    # plevel = P_level and is converted below 
    #
    # ubar 1 is calculated below (outgoing incident angles)
    # IMPORTANT : We end up redefining iubar = ubar1[ng,nt] to iubar = gangle[ng]
    # 
    # surf_reflect = 0s if surface is false or its a gray surface
    # otherwise, it is constant or from lab data
    #
    # hard_surface = 0 (gas giants) or 1 (terrestrial worlds)
    # tridiagonal = 0 (support only for tridiagonal matrices)
    #
    # We additionally loop over the gauss angles at the end
    # and apply the gass weight, a step performed in justdoit.py , picaso()
    #
    # Similar to PICASO, we don't use any delta-eddington nor raman scattering for emission.
    #############################################################################  

    # In order to account for multiple clouds, we compute sums in this loop
    kappa_cloud_w_cloud_sum = np.zeros_like(kappa_cloud)
    kappa_cloud_w_cloud_g_cloud_sum = np.zeros_like(kappa_cloud)

    for aerosol in range(len(kappa_cloud_seperate)):

        # Creating the sum for w_tot 

        # In order to account for w_cloud = 1, which causes numerical errors, we take this as well
        w_cloud[aerosol,:,0,zone_idx,:] = w_cloud[aerosol,:,0,zone_idx,:] * 0.99999

        # kappa_mie_w_cloud sum used in numerator of w_tot and denominator of g_tot
        kappa_cloud_w_cloud_sum[:,0,zone_idx,:] += kappa_cloud_seperate[aerosol,:,0,zone_idx,:] * w_cloud[aerosol,:,0,zone_idx,:]

        # kappa_cloud_w_cloud_g_cloud_sum used in numberator of g_tot 
        kappa_cloud_w_cloud_g_cloud_sum[:,0,zone_idx,:] += kappa_cloud_seperate[aerosol,:,0,zone_idx,:] * w_cloud[aerosol,:,0,zone_idx,:] * g_cloud[aerosol,:,0,zone_idx,:]

    # Numba doesn't like it when you multiply numpy arrays that are 1d vs 2d 

    # Calculate weighted, combined single scattering albedo
    # From W0_no_raman in optics.py, compute_opacity()
    # W0_no_raman = (TAURAY*0.99999 + TAUCLD*single_scattering_cld) / (TAUGAS + TAURAY + TAUCLD)
    # or 
    # w = ((tau_ray * w_Ray) + (tau_Mie * w_Mie))/(tau_tot) 
    # where w_Ray = 1 (perfect scatterers), and taus = kappas

    w_tot = ((0.99999 * kappa_Ray[:,0,zone_idx,:]) + (kappa_cloud_w_cloud_sum[:,0,zone_idx,:]))/kappa_tot

    # Calculate weighted, combined scattering asymmetry parameter
    # From COSB in optics.py, compute_opacity()
    # COSB = ftau_cld*asym_factor_cld 
    #      = (single_scattering_cld * TAUCLD)/(single_scattering_cld * TAUCLD + TAURAY)) * asym_factor_cloud
    # or 
    # g = ((w_Ray*delta_tau_Ray*g_Ray) + (w_Mie * delta_tau_Mie * g_Mie))/ (w_ray*delta_tau_ray + w_Mie*delta_tau_Mie)
    # where g_Ray = 0 (isotropic scatterers), w_rRay = 1, and taus = kappas 
    g_tot = (kappa_cloud_w_cloud_g_cloud_sum[:,0,zone_idx,:])/(kappa_cloud_w_cloud_sum[:,0,zone_idx,:]+(0.99999 * kappa_Ray[:,0,zone_idx,:]))

    # Compute planet flux including scattering (function expects 0 index to be top of atmosphere, so flip all the axis)
    P = np.flipud(P)
    T = np.flipud(T)
    dtau_tot = np.flipud(dtau_tot)
    w_tot = np.flipud(w_tot)
    g_tot = np.flipud(g_tot)   

    N_wl = len(wl)
    N_layer = len(P)
    N_level = N_layer + 1 # Number of levels (one more than the number of layers)

    T_level = np.zeros(N_level)
    log_P_level = np.zeros(N_level)

    # Calculate temperatures at layer boundaries ('levels')
    T_level[1:-1] = (T[1:] + T[:-1]) / 2.0
    T_level[0] = T_level[1] - (T_level[2] - T_level[1])
    T_level[-1] = T_level[-2] + (T_level[-2] - T_level[-3])

    # Calculate pressures at layer boundaries ('levels')
    log_P = np.log10(P)
    log_P_level[1:-1] = (log_P[1:] + log_P[:-1]) / 2.0
    log_P_level[0] = log_P_level[1] - (log_P_level[2] - log_P_level[1])
    log_P_level[-1] = log_P_level[-2] + (log_P_level[-2] - log_P_level[-3])
    P_level = np.power(10.0, log_P_level)

    # Load Gaussian quadrature mu and weights
    # gangle, gweight from get_angles_1d(num_gangle) 
    if (Gauss_quad == 5):
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])

    # Only 5th order Gaussian quadrature is currently supported
    else:
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])      

    # Calculate whatever ubar1 represents from the Gaussian angles
    # Taken from compute_disco(ng, nt, gangle, tangle, phase)
    # Here we assume the symmetric case where chebychev angles = 1 (equatorial latitude) so tangle = 0
    
    cos_theta = 1.0
    longitude = np.arcsin((gangle-(cos_theta-1.0)/(cos_theta+1.0))/(2.0/(cos_theta+1)))
    colatitude = np.arccos(0.0)              # Colatitude = 90-latitude 
    f = np.sin(colatitude)                   # Define to eliminate repetition
    ubar1 = np.outer(np.cos(longitude), f) 

    ###############################################
    # IMPORTED PICASO CODE *
    #
    # * variable names are changed, and we loop over gauss angles at end 
    ###############################################

    # Store hemispheric mean prefactor
    mu1 = 0.5  # From Table 1 Toon

  #  dtau = np.flip(dtau, axis=0)
  #  T_level = np.flip(T_level)
  #  P_level = np.flip(P_level)

    # Calculate matrix of blackbodies 
    all_b = planck_lambda_arr(T_level, wl)   # Returns N_level by N_wl

    # Calculate functions of black bodies appearing in Toon+1989 equations 
    b0 = all_b[0:-1,:]
    b1 = (all_b[1:,:] - b0) / dtau_tot # Eqn 26, Toon+1989

    # Hemispheric mean parameters from Table 1 of Toon+1989 
    # Changed 
    g1 = 2.0 - (w_tot * (1 + g_tot))
    g2 = w_tot * (1 - g_tot)

    alpha = np.sqrt((1.0 - w_tot) / (1.0 - (w_tot * g_tot)))
    lamda = np.sqrt(g1**2 - g2**2) #eqn 21 toon 
    gamma = (g1-lamda)/g2 # #eqn 22 toon
    
    g1_plus_g2 = 1.0/(g1+g2) #second half of eqn.27

    #same as with reflected light, compute c_plus and c_minus 
    #these are eqns 27a & b in Toon89
    #_ups are evaluated at lower optical depth, TOA
    #_dows are evaluated at higher optical depth, bottom of atmosphere
    c_plus_up = 2*np.pi*mu1*(b0 + b1* g1_plus_g2) 
    c_minus_up = 2*np.pi*mu1*(b0 - b1* g1_plus_g2)
    #NOTE: to keep consistent with Toon, we keep these 2pis here. However, 
    #in 3d cases where we no long assume azimuthal symmetry, we divide out 
    #by 2pi when we multiply out the weights as seen in disco.compress_thermal 

    c_plus_down = 2*np.pi*mu1*(b0 + b1 * dtau_tot + b1 * g1_plus_g2) 
    c_minus_down = 2*np.pi*mu1*(b0 + b1 * dtau_tot - b1 * g1_plus_g2)

    #calculate exponential terms needed for the tridiagonal rotated layered method
    exptrm = lamda*dtau_tot

    #save from overflow 
    exptrm = slice_gt(exptrm, 35.0) 

    exptrm_positive = np.exp(exptrm) 
    exptrm_minus = 1.0/exptrm_positive

    #for flux heating calculations, the energy balance solver 
    #does not like a fixed zero at the TOA. 
    #to avoid a discontinuous kink at the last atmosphere
    #layer we create this "fake" boundary condition
    #we imagine that the atmosphere continues up at an isothermal T and that 
    #there is optical depth from above the top to infinity 
    tau_top = dtau_tot[0,:]*P_level[0]/(P_level[1]-P_level[0]) #tried this.. no luck*exp(-1)# #tautop=dtau[0]*np.exp(-1)
    b_top = (1.0 - np.exp(-tau_top / mu1 )) * all_b[0,:] * np.pi #  Btop=(1.-np.exp(-tautop/ubari))*B[0]
    #print('hard_surface=',hard_surface)
    if hard_surface:
        emissivity = 1.0 - surf_reflect #Emissivity is 1 - surface reflectivity
        #b_surface =  emissivity*planck_lambda_arr([T_surf], wl)[-1,:]*np.pi #for terrestrial, hard surface
        #b_surface =  emissivity*all_b[-1,:]*np.pi #for terrestrial, hard surface 
        b_surface =  emissivity*all_b[-1,:]*np.pi #for terrestrial, hard surface 
    else: 
        b_surface= (all_b[-1,:] + b1[-1,:]*mu1)*np.pi #(for non terrestrial)

    #Now we need the terms for the tridiagonal rotated layered method
    if tridiagonal==0:
        A, B, C, D = setup_tri_diag(N_layer, N_wl,  c_plus_up, c_minus_up, 
                                    c_plus_down, c_minus_down, b_top, b_surface, 
                                    surf_reflect, gamma, dtau_tot, exptrm_positive,  exptrm_minus) 
    #else:
    #   A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
    #                       c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
    #                        gamma, dtau, 
    #                       exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 
    positive = np.zeros((N_layer, N_wl))
    negative = np.zeros((N_layer, N_wl))
    #========================= Start loop over wavelength =========================
    L = N_layer + N_layer
    
    for w in range(N_wl):
        #coefficient of positive and negative exponential terms 
        if tridiagonal==0:
            X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
            #unmix the coefficients
            positive[:,w] = X[::2] + X[1::2] #Y1+Y2 in toon (table 3)
            negative[:,w] = X[::2] - X[1::2] #Y1-Y2 in toon (table 3)
        #else:
        #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
        #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
        #   negative[:,w] = X[::2] - X[1::2]

    #if you stop here this is regular ole 2 stream 
    f_up = (positive * exptrm_positive + gamma * negative * exptrm_minus + c_plus_up)
    #flux_minus  = gamma*positive*exptrm_positive + negative*exptrm_minus + c_minus_down
    #flux_plus  = positive*exptrm_positive + gamma*negative*exptrm_minus + c_plus_down
    #flux = zeros((2*nlevel, nwno))
    #flux[0,:] = (gamma*positive + negative + c_minus_down)[0,:]
    #flux[1,:] = (positive + gamma*negative + c_plus_down)[0,:]
    #flux[2::2, :] = flux_minus
    #flux[3::2, :] = flux_plus


    #calculate everything from Table 3 toon
    #from here forward is source function technique in toon
    G = (1/mu1 - lamda)*positive     
    H = gamma*(lamda + 1/mu1)*negative 
    J = gamma*(lamda + 1/mu1)*positive 
    K = (1/mu1 - lamda)*negative     
    alpha1 = 2*np.pi*(b0+b1*(g1_plus_g2 - mu1)) 
    alpha2 = 2*np.pi*b1 
    sigma1 = 2*np.pi*(b0-b1*(g1_plus_g2 - mu1)) 
    sigma2 = 2*np.pi*b1 

    int_minus = np.zeros((N_level,N_wl))
    int_plus = np.zeros((N_level,N_wl))
    int_minus_mdpt = np.zeros((N_level,N_wl))
    int_plus_mdpt = np.zeros((N_level,N_wl))
    #intensity = np.zeros((Gauss_quad, numt, N_level, N_wl))

    exptrm_positive_mdpt = np.exp(0.5*exptrm) 
    exptrm_minus_mdpt = 1/exptrm_positive_mdpt 

    #================ START CRAZE LOOP OVER ANGLE #================
    int_at_top = np.zeros((Gauss_quad, numt, N_wl)) #get intensity 
    int_down = np.zeros((Gauss_quad, numt, N_wl))

    F = np.zeros(N_wl)   # Emergent flux

    #work through building eqn 55 in toon (tons of book keeping exponentials)
    for ng in range(Gauss_quad):
        for nt in range(numt): 
            #flux_out[ng,nt,:,:] = flux

         #   iubar = ubar1[ng,nt]
            iubar = gangle[ng]

            #intensity boundary conditions
            if hard_surface:
                emissivity = 1.0 - surf_reflect #Emissivity is 1 - surface reflectivity
                int_plus[-1,:] = emissivity*all_b[-1,:] *2*np.pi  # terrestrial flux /pi = intensity
            else:
                int_plus[-1,:] = ( all_b[-1,:] + b1[-1,:] * iubar)*2*np.pi #no hard surface 

            int_minus[0,:] =  (1 - np.exp(-tau_top / iubar)) * all_b[0,:] *2*np.pi
            
            exptrm_angle = np.exp( - dtau_tot / iubar)
            exptrm_angle_mdpt = np.exp( -0.5 * dtau_tot / iubar) 

            for itop in range(N_layer):

                #disbanning this for now because we dont need it in the thermal emission code
                #EQN 56,toon
            #    int_minus[itop+1,:]=(int_minus[itop,:]*exptrm_angle[itop,:]+
            #                         (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
            #                         (K[itop,:]/(lamda[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
            #                         sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
            #                         sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau_tot[itop,:]-iubar) )

            #    int_minus_mdpt[itop,:]=(int_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
            #                            (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
            #                            (K[itop,:]/(-lamda[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
            #                            sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
            #                            sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau_tot[itop,:]-iubar))

                ibot=N_layer-1-itop
                #EQN 55,toon
                int_plus[ibot,:]=(int_plus[ibot+1,:]*exptrm_angle[ibot,:]+
                                  (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle[ibot,:]-1.0)+
                                  (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(1.0-exptrm_minus[ibot,:] * exptrm_angle[ibot,:])+
                                  alpha1[ibot,:]*(1.-exptrm_angle[ibot,:])+
                                  alpha2[ibot,:]*(iubar-(dtau_tot[ibot,:]+iubar)*exptrm_angle[ibot,:]) )

                int_plus_mdpt[ibot,:]=(int_plus[ibot+1,:]*exptrm_angle_mdpt[ibot,:]+
                                       (G[ibot,:]/(lamda[ibot,:]*iubar-1.0))*(exptrm_positive[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_positive_mdpt[ibot,:])-
                                       (H[ibot,:]/(lamda[ibot,:]*iubar+1.0))*(exptrm_minus[ibot,:]*exptrm_angle_mdpt[ibot,:]-exptrm_minus_mdpt[ibot,:])+
                                       alpha1[ibot,:]*(1.-exptrm_angle_mdpt[ibot,:])+
                                       alpha2[ibot,:]*(iubar+0.5*dtau_tot[ibot,:]-(dtau_tot[ibot,:]+iubar)*exptrm_angle_mdpt[ibot,:])  )

            int_at_top[ng,nt,:] = int_plus_mdpt[0,:] #N_level by N_wl 
            #intensity[ng,nt,:,:] = int_plus

            #to get the convective heat flux 
            #flux_minus_mdpt_disco[ng,nt,:,:] = flux_minus_mdpt #N_level by N_wl
            #flux_plus_mdpt_disco[ng,nt,:,:] = int_plus_mdpt #N_level by N_wl

    for ng in range(Gauss_quad):

        F += (int_at_top[ng,0,:] * gweight[ng]) 

    return F, dtau_tot

@jit(nopython=True, cache=True)
def numba_cumsum(mat):
    """Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
    cumsum 
    """
    new_mat = np.zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:,i] = np.cumsum(mat[:,i])
    return new_mat

@jit(nopython=True, cache=True)
def reflection_Toon(P, wl, dtau_tot,
                    kappa_Ray, kappa_cloud, kappa_tot,
                    w_cloud, g_cloud, zone_idx,
                    surf_reflect,
                    single_phase = 3, multi_phase = 0,
                    frac_a = 1, frac_b = -1, frac_c = 2, constant_back = -0.5, constant_forward = 1,
                    Gauss_quad = 5, numt = 1,
                    toon_coefficients=0, tridiagonal=0, b_top=0):

    ###############################################
    # ORIGINAL PICASO PREAMBLE (get_reflected_1d)
    ###############################################
    """
    Computes toon fluxes given tau and everything is 1 dimensional. This is the exact same function 
    as `get_flux_geom_3d` but is kept separately so we don't have to do unecessary indexing for fast
    retrievals. 
    Parameters
    ----------
    nlevel : int 
        Number of levels in the model 
    wno : array of float 
        Wave number grid in cm -1 
    nwno : int 
        Number of wave points
    numg : int 
        Number of Gauss angles 
    numt : int 
        Number of Chebyshev angles 
    DTAU : ndarray of float
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    TAU : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT D-Eddington Correction
        Dimensions=# level by # wave        
    W0 : ndarray of float 
        This is the single scattering albedo, from scattering, clouds, raman, etc 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    COSB : ndarray of float 
        This is the asymmetry factor 
        WITHOUT D-Eddington Correction
        Dimensions=# layer by # wave
    GCOS2 : ndarray of float 
        Parameter that allows us to directly include Rayleigh scattering 
        = 0.5*tau_rayleigh/(tau_rayleigh + tau_cloud)
    ftau_cld : ndarray of float 
        Fraction of cloud extinction to total 
        = tau_cloud/(tau_rayleigh + tau_cloud)
    ftau_ray : ndarray of float 
        Fraction of rayleigh extinction to total 
        = tau_rayleigh/(tau_rayleigh + tau_cloud)
    dtau_og : ndarray of float 
        This is the opacity contained within each individual layer (defined at midpoints of "levels")
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# layer by # wave
    tau_og : ndarray of float
        This is the cumulative summed opacity 
        WITHOUT the delta eddington correction, if it was specified by user
        Dimensions=# level by # wave    
    w0_og : ndarray of float 
        Same as w0 but WITHOUT the delta eddington correction, if it was specified by user  
    cosb_og : ndarray of float 
        Same as cosbar buth WITHOUT the delta eddington correction, if it was specified by user
    surf_reflect : float 
        Surface reflectivity 
    ubar0 : ndarray of float 
        matrix of cosine of the incident angle from geometric.json
    ubar1 : ndarray of float 
        matrix of cosine of the observer angles
    cos_theta : float 
        Cosine of the phase angle of the planet 
    F0PI : array 
        Downward incident solar radiation
    single_phase : str 
        Single scattering phase function, default is the two-term henyey-greenstein phase function
    multi_phase : str 
        Multiple scattering phase function, defulat is N=2 Legendre polynomial 
    frac_a : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_b : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C)
    frac_c : float 
        (Optional), If using the TTHG phase function. Must specify the functional form for fraction 
        of forward to back scattering (A + B * gcosb^C), Default is : 1 - gcosb^2
    constant_back : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of back scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    constant_forward : float 
        (Optional), If using the TTHG phase function. Must specify the assymetry of forward scatterer. 
        Remember, the output of A & M code does not separate back and forward scattering.
    tridiagonal : int 
        0 for tridiagonal, 1 for pentadiagonal 
    toon_coefficients : int     
        0 for quadrature (default) 1 for eddington

    Returns
    -------
    intensity at the top of the atmosphere for all the different ubar1 and ubar2 
    To Do
    -----
    - F0PI Solar flux shouldn't always be 1.. Follow up to make sure that this isn't a bad 
          hardwiring to solar, despite "relative albedo"
    """
 
    ###############################################
    # POSEIDON PREAMBLE
    ###############################################

    # From PICASO, we define or compute the following original inputs : 
    # 
    # nlevel as N_layer + 1 = len(P) + 1
    # wno = 1/wl
    # nwno as N_wl = len(wl)
    #
    # numg = Num Gauss angles (longitude) as Gauss_quad = 5
    # numt = Num Chebychev angles (latitude) =  1
    # we define our system to default to the 1x10 scheme where where tangle = 0 and gangle = 5
    # the weights are defined below
    # https://natashabatalha.github.io/picaso/notebooks/8_SphericalIntegration.html?highlight=geometry
    #
    # dtau = delta eddington version of dtau_og (calculated below), renamed dtau_dedd
    # tau = delta eddington version of tau (calculated below), renamed tau_dedd
    # IMPORTANT : We only consider a 1D model, meaning the dtau and taus don't vary with gauss weight
    #
    # w0 = delta eddington version of w0_og (calculated below), renamed w_dedd
    # cosb = delta eddington version of cosb_og (calculated below), renamed g_dedd
    # IMPORTANT : Delta Eddington is used in reflection tridiagonal + multiple scattering
    # but NOT single scattering 
    #
    # gcos2 = asymmetry parameter of Rayleigh scattering, calculated below
    #
    # ftau_cld = fractional scattering due to clouds, calculated below (see g_tot in emission_Toon)
    # ftau_ray = fractional scattering due to Rayleigh, claculated below 
    #
    # Since we use delta eddington, these are actually our initial inputs 
    # dtau_og = dtau_tot (differential extinction optical depth (from absorption + scattering) across each later)
    # tau_og  = summed d_tau across each layer, calculated below 
    # w0_og = w_tot      calculated below (weighted, combined single scattering albedo)
    # cosb_og = g_cloud  (IMPORTANT : in reflection, we don't take the weighted version)
    #
    # surf_reflect = 0s if surface is false or its a gray surface
    # otherwise, it is constant or from lab data
    #
    # ubar0 (ingoing incident angles) are calculated below with phase_angle 0
    # ubar1 (outgoing incident angles) are calculated below with phase_angle = 0
    # IMPORTANT : We don't redefine iubar here, like we do for emission_Toon
    # 
    # cos_theta is set to 1 (phase_angle = 0)
    #
    # F0PI = np.zeros(nwno) + 1. by default
    # 
    # These are the default settings from PICASO
    # See equations described in section 2.1 of Batalha 2019
    # single_phase = 'TTHG_ray' ( = 3)
    # multi_phase = 'N=2' ( = 0)
    # frac_a = 1
    # frac_b = -1
    # frac_c = 2
    # constant_back = -1/2
    # constant_forward = 1
    # toon_coefficients = 0 (quadtrature)
    # tridiagonal = 0 (only support for tridiagonal matrices)
    # b_top = 0 (The diffuse radiation into the model at the top of the atmosphere, default boundary condition for the triadonalg matrix)

    # We additionally loop over the gauss angles at the end
    # and apply the gass weight, a step performed in justdoit.py , picaso()
    # We also compute the albedo, a step performed in justdoit.py, picaso(), compress_disco()
    ############################################################################################################

    N_wl = len(wl)
    N_layer = len(P)
    N_level = N_layer + 1 # Number of levels (one more than the number of layers)

    # From optics.py, compute_opacity 
    # We calculate the ftaus, tau, and delta_eddington corrections 

    # In order to account for w_cloud = 1, which causes numerical errors, we take this as well
    w_cloud[:,0,zone_idx,:] = w_cloud[:,0,zone_idx,:] * 0.99999

    # ftau_cld 

    # From emission_Toon we figured out that 
    # cosb = g_tot = ftau_cld * g_cloud 
    # however, the above is only valid in emission. In reflection, g_cloud remains g_cloud
    # but we can still use the above equation to figure out ftau_cld in POSEIDON terms 
    # in emission_Toon 
    # g_tot = ((w_cloud * kappa_cloud) / ((w_cloud * kappa_cloud) + kappa_Ray)) * g_cloud 
    # it follows that 
    # ftau_cld = ((w_cloud * kappa_cloud) / ((w_cloud * kappa_cloud) + kappa_Ray))

    ftau_cld = ((w_cloud[:,0,zone_idx,:] * kappa_cloud[:,0,zone_idx,:]) / ((w_cloud[:,0,zone_idx,:] * kappa_cloud[:,0,zone_idx,:]) + kappa_Ray[:,0,zone_idx,:])) * g_cloud[:,0,zone_idx,:]

    # gcos2 
    # ftau_ray = TAURAY/(TAURAY + single_scattering_cld * TAUCLD)
    # GCOS2 = 0.5*ftau_ray #Hansen & Travis 1974 for Rayleigh scattering 

    ftau_ray = kappa_Ray[:,0,zone_idx,:]/(kappa_Ray[:,0,zone_idx,:] + g_cloud[:,0,zone_idx,:] * kappa_cloud[:,0,zone_idx,:])
    gcos2 = 0.5*ftau_ray

    # w_tot is the same as it is in emission_Toon (weighted version)
    w_tot = (0.99999 * kappa_Ray[:,0,zone_idx,:] + (kappa_cloud[:,0,zone_idx,:] * w_cloud[:,0,zone_idx,:]))/kappa_tot
     
    # Function expects 0 index to be top of atmosphere, so flip all the axis
    P = np.flipud(P)
    dtau_tot = np.flipud(dtau_tot)
    w_tot = np.flipud(w_tot)
    g_cloud = np.flipud(g_cloud)  
    ftau_cld = np.flipud(ftau_cld) 
    ftau_ray = np.flipud(ftau_ray)

    # Remake g_cloud so that its always [:,0,zone_idx,:]
    g_cloud = g_cloud[:,0,zone_idx,:]
    
    # tau 
    # This sums up the taus starting at the top
    # In the original picaso : TAU = np.zeros((nlayer+1, nwno,ngauss))
    # Where each ngauss column is exactly the same for contiuum and rayleigh
    # but not the same for molecular opacities 
    # In POSEIDON we don't use the gauss quad as a dimension to loop over
    # So we remove that dependence here 
    tau = np.zeros((N_layer+1, N_wl))
    tau[1:,:]=numba_cumsum(dtau_tot[:,:])
 
    # Delta Eddington Correction
    # We take the default number of streams to be 2 
    stream = 2
    f_deltaM = g_cloud**stream
    w_dedd=w_tot*(1.-f_deltaM)/(1.0-w_tot*f_deltaM)
    g_dedd=(g_cloud-f_deltaM)/(1.-f_deltaM)
    dtau_dedd=dtau_tot*(1.-w_tot*f_deltaM) 

    tau_dedd = np.zeros((N_layer+1, N_wl))
    tau_dedd[1:,:]=numba_cumsum(dtau_dedd[:,:])
    
    # Load Gaussian quadrature mu and weights
    # gangle, gweight from disco.py, get_angles_1d(num_gangle) 
    # tangle, tweight = 0 and 1 (equator only)
    if (Gauss_quad == 5):
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])

    # Only 5th order Gaussian quadrature is currently supported
    else:
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])      

    # Calculate what ubar1 and ubar0 represents from the Gaussian angles
    # Taken from compute_disco(ng, nt, gangle, tangle, phase)
    # Here we assume the symmetric case where chebychev angles = 1 (equatorial latitude) so tangle = 0
    phase_angle = 0
    cos_theta = 1.0     #cos(phase_angle)
    longitude = np.arcsin((gangle-(cos_theta-1.0)/(cos_theta+1.0))/(2.0/(cos_theta+1)))
    colatitude = np.arccos(0.0)              # Colatitude = 90-latitude, 0 at equator 
    f = np.sin(colatitude)                   # Define to eliminate repetition
    ubar0 = np.outer(np.cos(longitude-phase_angle) , f) #ng by nt 
    ubar1 = np.outer(np.cos(longitude), f) 

 
    # FOPI is just an array of 1s (solar)
    F0PI = np.zeros(N_wl) + 1

    ###############################################
    # IMPORTED PICASO CODE (with variable name changes, and loop at end)
    ###############################################

    #what we want : intensity at the top as a function of all the different angles

    xint_at_top = np.zeros((Gauss_quad, numt, N_wl))
    #intensity = zeros((numg, numt, nlevel, nwno))

    flux_out = np.zeros((Gauss_quad, numt, 2*N_level, N_wl))

    #now define terms of Toon et al 1989 quadrature Table 1 
    #https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/JD094iD13p16287
    #see table of terms 

    # terms not dependent on incident angle
    # Taken from table 1, these are the coefficients for eddington and quadrature
    # The default is quadtrature 
    # Different from emission, where we assume hemispheric g1 and g2

    sq3 = np.sqrt(3.)
    if toon_coefficients == 1:#eddington
        g1  = (7-w_dedd*(4+3*ftau_cld*g_dedd))/4 #(sq3*0.5)*(2. - w0*(1.+cosb)) #table 1 # 
        g2  = -(1-w_dedd*(4-3*ftau_cld*g_dedd))/4 #(sq3*w0*0.5)*(1.-cosb)        #table 1 # 

    # DEFAULT
    elif toon_coefficients == 0:#quadrature
        g1  = (sq3*0.5)*(2. - w_dedd*(1.+ftau_cld*g_dedd)) #table 1 # 
        g2  = (sq3*w_dedd*0.5)*(1.-ftau_cld*g_dedd)        #table 1 # 

    lamda = np.sqrt(g1**2 - g2**2)         #eqn 21
    gama  = (g1-lamda)/g2               #eqn 22

    #================ START CRAZE LOOP OVER ANGLE #================
    for ng in range(Gauss_quad):
        for nt in range(numt):
            u1 = ubar1[ng,nt]
            u0 = ubar0[ng,nt]
            if toon_coefficients == 1 : #eddington
                g3  = (2-3*ftau_cld*g_dedd*u0)/4#0.5*(1.-sq3*cosb*ubar0[ng, nt]) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
            elif toon_coefficients == 0 :#quadrature
                g3  = 0.5*(1.-sq3*ftau_cld*g_dedd*u0) #  #table 1 #ubar has dimensions [gauss angles by tchebyshev angles ]
    
            # now calculate c_plus and c_minus (equation 23 and 24 toon)
            g4 = 1.0 - g3
            denominator = lamda**2 - 1.0/u0**2.0

            #everything but the exponential 
            a_minus = F0PI*w_dedd* (g4*(g1 + 1.0/u0) +g2*g3 ) / denominator
            a_plus  = F0PI*w_dedd*(g3*(g1-1.0/u0) +g2*g4) / denominator

            #add in exponential to get full eqn
            #_up is the terms evaluated at lower optical depths (higher altitudes)
            #_down is terms evaluated at higher optical depths (lower altitudes)
            x = np.exp(-tau_dedd[:-1,:]/u0)
            c_minus_up = a_minus*x #CMM1
            c_plus_up  = a_plus*x #CPM1
            x = np.exp(-tau_dedd[1:,:]/u0)
            c_minus_down = a_minus*x #CM
            c_plus_down  = a_plus*x #CP

            #calculate exponential terms needed for the tridiagonal rotated layered method
            exptrm = lamda*dtau_dedd
            #save from overflow 
            exptrm = slice_gt(exptrm, 35.0) 

            exptrm_positive =np.exp(exptrm) #EP
            exptrm_minus = 1.0/exptrm_positive#EM


            #boundary conditions 
            #b_top = 0.0                                       

            b_surface = 0. + surf_reflect*u0*F0PI*np.exp(-tau_dedd[-1, :]/u0)

            #Now we need the terms for the tridiagonal rotated layered method
            if tridiagonal==0:
                A, B, C, D = setup_tri_diag(N_layer,N_wl, c_plus_up, c_minus_up, 
                                    c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
                                     gama, dtau_dedd, 
                                    exptrm_positive,  exptrm_minus) 

            #else:
            #   A_, B_, C_, D_, E_, F_ = setup_pent_diag(nlayer,nwno,  c_plus_up, c_minus_up, 
            #                       c_plus_down, c_minus_down, b_top, b_surface, surf_reflect,
            #                        gama, dtau, 
            #                       exptrm_positive,  exptrm_minus, g1,g2,exptrm,lamda) 

            positive = np.zeros((N_layer, N_wl))
            negative = np.zeros((N_layer, N_wl))
            #========================= Start loop over wavelength =========================
            L = N_layer + N_layer

            for w in range(N_wl):
                #coefficient of posive and negative exponential terms 
                if tridiagonal==0:
                    X = tri_diag_solve(L, A[:,w], B[:,w], C[:,w], D[:,w])
                    #unmix the coefficients
                    positive[:,w] = X[::2] + X[1::2] 
                    negative[:,w] = X[::2] - X[1::2]

                #else: 
                #   X = pent_diag_solve(L, A_[:,w], B_[:,w], C_[:,w], D_[:,w], E_[:,w], F_[:,w])
                    #unmix the coefficients
                #   positive[:,w] = exptrm_minus[:,w] * (X[::2] + X[1::2])
                #   negative[:,w] = X[::2] - X[1::2]

            #========================= End loop over wavelength =========================

            #use expression for bottom flux to get the flux_plus and flux_minus at last
            #bottom layer
            flux_zero  = positive[-1,:]*exptrm_positive[-1,:] + gama[-1,:]*negative[-1,:]*exptrm_minus[-1,:] + c_plus_down[-1,:]
            flux_minus  = gama*positive*exptrm_positive + negative*exptrm_minus + c_minus_down
            flux_plus  = positive*exptrm_positive + gama*negative*exptrm_minus + c_plus_down
            flux = np.zeros((2*N_level, N_wl))
            flux[0,:] = (gama*positive + negative + a_minus)[0,:]
            flux[1,:] = (positive + gama*negative + a_plus)[0,:]
            flux[2::2, :] = flux_minus
            flux[3::2, :] = flux_plus
            flux_out[ng,nt,:,:] = flux

            xint = np.zeros((N_level,N_wl))
            xint[-1,:] = flux_zero/np.pi

            ################################ BEGIN OPTIONS FOR MULTIPLE SCATTERING####################

            #Legendre polynomials for the Phase function due to multiple scatterers 
            if multi_phase ==0:#'N=2':
                #ubar2 is defined to deal with the integration over the second moment of the 
                #intensity. It is FIT TO PURE RAYLEIGH LIMIT, ~(1/sqrt(3))^(1/2)
                #this is a decent assumption because our second order legendre polynomial 
                #is forced to be equal to the rayleigh phase function
                ubar2 = 0.767  # 
                multi_plus = (1.0+1.5*ftau_cld*g_dedd*u1 #! was 3
                                + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
                multi_minus = (1.-1.5*ftau_cld*g_dedd*u1 
                                + gcos2*(3.0*ubar2*ubar2*u1*u1 - 1.0)/2.0)
            elif multi_phase ==1:#'N=1':
                multi_plus = 1.0+1.5*ftau_cld*g_dedd*u1  
                multi_minus = 1.-1.5*ftau_cld*g_dedd*u1


            ################################ END OPTIONS FOR MULTIPLE SCATTERING####################

            G=positive*(multi_plus+gama*multi_minus)    * w_dedd
            H=negative*(gama*multi_plus+multi_minus)    *w_dedd
            A=(multi_plus*c_plus_up+multi_minus*c_minus_up) *w_dedd

            G=G*0.5/np.pi
            H=H*0.5/np.pi
            A=A*0.5/np.pi

            ################################ BEGIN OPTIONS FOR DIRECT SCATTERING####################
            #define f (fraction of forward to back scattering), 
            #g_forward (forward asymmetry), g_back (backward asym)
            #needed for everything except the OTHG
            # if single_phase!=1: 
            g_forward = constant_forward*g_cloud
            g_back = constant_back*g_cloud#-
            f = frac_a + frac_b*g_back**frac_c

            # NOTE ABOUT HG function: we are translating to the frame of the downward propagating beam
            # Therefore our HG phase function becomes:
            # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2+2*cosb_og*cos_theta)**3) 
            # as opposed to the traditional:
            # p_single=(1-cosb_og**2)/sqrt((1+cosb_og**2-2*cosb_og*cos_theta)**3) (NOTICE NEGATIVE FROM COS_THETA)

            #if single_phase==0:#'cahoy':
            #    #Phase function for single scattering albedo frum Solar beam
            #    #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
            #          #first term of TTHG: forward scattering
            #    p_single=(f * (1-g_forward**2)
            #                    /np.sqrt((1+g_cloud**2+2*g_cloud*cos_theta)**3) 
            #                    #second term of TTHG: backward scattering
            #                    +(1-f)*(1-g_back**2)
            #                    /np.sqrt((1+(-g_cloud/2.)**2+2*(-g_cloud/2.)*cos_theta)**3)+
            #                    #rayleigh phase function
            #                    (gcos2))
            #elif single_phase==1:#'OTHG':
            #    p_single=(1-g_cloud**2)/np.sqrt((1+g_cloud**2+2*g_cloud*cos_theta)**3) 
            #elif single_phase==2:#'TTHG':
            #    #Phase function for single scattering albedo frum Solar beam
            #    #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
            #          #first term of TTHG: forward scattering
            #    p_single=(f * (1-g_forward**2)
            #                    /np.sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
            #                    #second term of TTHG: backward scattering
            #                    +(1-f)*(1-g_back**2)
            #                    /np.sqrt((1+g_back**2+2*g_back*cos_theta)**3))
            #elif single_phase==3:#'TTHG_ray':
                #Phase function for single scattering albedo frum Solar beam
                #uses the Two term Henyey-Greenstein function with the additiona rayleigh component 
                #first term of TTHG: forward scattering
            p_single=(ftau_cld*(f * (1-g_forward**2)
                                            /np.sqrt((1+g_forward**2+2*g_forward*cos_theta)**3) 
                                            #second term of TTHG: backward scattering
                                            +(1-f)*(1-g_back**2)
                                            /np.sqrt((1+g_back**2+2*g_back*cos_theta)**3))+            
                                            #rayleigh phase function
                                            ftau_ray*(0.75*(1+cos_theta**2.0)))
                

            
            #removing single form option from code 
            #single_form : int 
            #    form of the phase function can either be written as an 'explicit' (0) henyey greinstein 
            #    or it can be written as a 'legendre' (1) expansion. Default is 'explicit'=0

            #if single_form==1:
            #    TAU = tau; DTAU = dtau; W0 = w0
            #    p_single = 0*p_single
            #    Pu0 = legP(-u0) # legendre polynomials for -u0
            #    Pu1 = legP(u1) # legendre polynomials for -u0
            #    maxterm = 2
            #    for l in range(maxterm):
            #        w = (2*l+1) * cosb_og**l
            #        w_single = (w - (2*l+1)*cosb_og**maxterm) / (1 - cosb_og**maxterm) 
            #        p_single = p_single + w_single * Pu0[l]*Pu1[l]
            #else:
            #    TAU = tau_og; DTAU = dtau_og; W0 = w0_og

            ################################ END OPTIONS FOR DIRECT SCATTERING####################

            single_scat = np.zeros((N_level,N_wl))
            multi_scat = np.zeros((N_level,N_wl))

            for i in range(N_layer-1,-1,-1):
                single_scat[i,:] = ((w_tot[i,:]*F0PI/(4.*np.pi))
                        *(p_single[i,:])*np.exp(-tau[i,:]/u0)
                        *(1. - np.exp(-dtau_tot[i,:]*(u0+u1)
                        /(u0*u1)))*
                        (u0/(u0+u1)))

                multi_scat[i,:] = (A[i,:]*(1. - np.exp(-dtau_dedd[i,:] *(u0+1*u1)/(u0*u1)))*
                        (u0/(u0+1*u1))
                        +G[i,:]*(np.exp(exptrm[i,:]*1-dtau_dedd[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                        +H[i,:]*(1. - np.exp(-exptrm[i,:]*1-dtau_dedd[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0)
                        )

                #direct beam
                xint[i,:] =( xint[i+1,:]*np.exp(-dtau_dedd[i,:]/u1) 
                        #single scattering albedo from sun beam (from ubar0 to ubar1)
                        +(w_tot[i,:]*F0PI/(4.*np.pi))
                        *(p_single[i,:])*np.exp(-tau[i,:]/u0)
                        *(1. - np.exp(-dtau_tot[i,:]*(u0+u1)
                        /(u0*u1)))*
                        (u0/(u0+u1))
                        #multiple scattering terms p_single
                        +A[i,:]*(1. - np.exp(-dtau_dedd[i,:] *(u0+1*u1)/(u0*u1)))*
                        (u0/(u0+1*u1))
                        +G[i,:]*(np.exp(exptrm[i,:]*1-dtau_dedd[i,:]/u1) - 1.0)/(lamda[i,:]*1*u1 - 1.0)
                        +H[i,:]*(1. - np.exp(-exptrm[i,:]*1-dtau_dedd[i,:]/u1))/(lamda[i,:]*1*u1 + 1.0)
                        )

            xint_at_top[ng,nt,:] = xint[0,:]
            #intensity[ng,nt,:,:] = xint

#    import IPython; IPython.embed()
#    import sys; sys.exit()

    # Apply the gauss weights 
    # IMPORTANT : In picaso.py, when get_reflected_1d is called
    # It first uses the gauss weights that go with correlated k
    # So we don't use these weights

    # Now we want to get the abedo 
    # This is taken from the compress_disco functions 

    # Default tangle and tweight
    tangle,tweight = np.array([0]), np.array([1])
 
    ng = len(gweight)
    nt = len(tweight)
    if nt==1 : sym_fac = 2*np.pi #azimuthal symmetry  
    else: sym_fac = 1
    albedo=np.zeros(N_wl)
    #for w in range(nwno):
    #   albedo[w] = 0.5 * sum((xint_at_top[:,:,w]*tweight).T*gweight)
    for ig in range(ng): 
        for it in range(nt): 
            albedo = albedo + xint_at_top[ig,it,:] * gweight[ig] * tweight[it]
    albedo = sym_fac * 0.5 * albedo /F0PI * (cos_theta + 1.0)

    return albedo

#@jit(nopython = True)
def emission_bare_surface(T_surf, wl, surf_reflect):
    '''
    Compute the emergent top-of-atmosphere flux from a bare rock 

    This function  considers only pure thermal emission (i.e. no scattering).
    Uses gauss weights of 5 to stay consistent with emission_Toon

    Args:
        T (np.array of float):
            Temperatures in each atmospheric layer (K).
        wl (np.array of float): 
            Wavelength grid (μm).
        surf_reflect : numpy.ndarray    
            Surface reflectivity as a function of wavelength. 
    
    Returns:
        F (np.array of float):
            Spectral surface flux in SI units (W/m^2/sr/m).

    '''
    
    # Calculate Planck function in each layer and each wavelength
    T = np.array([T_surf])
    B = planck_lambda_arr(T, wl)

    # Need emissivity 
    emissivity = 1.0 - surf_reflect #Emissivity is 1 - surface reflectivity

    # For a bare body rock, F = pi * I * emissivity
    # Instead of 2 pi you need pi so that its just hemisphere 
    # The weights, gauss weights, just end up being 0.5 which goes down to pi * I
    F = B[0,:] * emissivity * np.pi
    
    return F

#@jit(nopython = True)
def reflection_bare_surface(wl, surf_reflect, Gauss_quad = 5):
    '''
    Compute the emergent top-of-atmosphere flux from a bare rock 

    This function  considers only pure thermal emission (i.e. no scattering).
    Uses gauss weights of 5 to stay consistent with emission_Toon

    Args:
        wl (np.array of float): 
            Wavelength grid (μm).
        surf_reflect : numpy.ndarray    
            Surface reflectivity as a function of wavelength. 
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            (Options: 2 / 3).
    
    Returns:
        F (np.array of float):
            Spectral surface flux in SI units (W/m^2/sr/m).

    '''
    
    N_wl = len(wl)

    if (Gauss_quad == 5):
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])

    # Only 5th order Gaussian quadrature is currently supported
    else:
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])  

    # Default tangle and tweight
    tangle,tweight = np.array([0]), np.array([1])

    cos_theta = 1.0     #cos(phase_angle)

    # FOPI is just an array of 1s (solar)
    F0PI = np.zeros(N_wl) + 1

    ng = len(gweight)
    nt = len(tweight)
    if nt==1 : sym_fac = 2*np.pi #azimuthal symmetry  
    else: sym_fac = 1
    albedo=np.zeros(N_wl)

    phase_angle = 0
    cos_theta = 1.0     #cos(phase_angle)
    longitude = np.arcsin((gangle-(cos_theta-1.0)/(cos_theta+1.0))/(2.0/(cos_theta+1)))
    colatitude = np.arccos(0.0)              # Colatitude = 90-latitude, 0 at equator 
    f = np.sin(colatitude)                   # Define to eliminate repetition
    ubar0 = np.outer(np.cos(longitude-phase_angle) , f) #ng by nt 

    for ig in range(ng): 
        for it in range(nt): 
            u0 = ubar0[ig,it]
            surf_reflect_weighted = surf_reflect*u0*F0PI
            albedo = albedo + surf_reflect_weighted * gweight[ig] * tweight[it]
            
    albedo = sym_fac * 0.5 * albedo /F0PI * (cos_theta + 1.0)

    # Since this is geometric albedo, we should divide it by pi (?)
    # (Equation 3.19 in Exoplanet Atmospheres, Sara Seager has A_g = (1/pi) * a complicated integral)
    # Maybe this 1/pi is baked into toon reflection
    albedo = albedo/np.pi
    
    return albedo