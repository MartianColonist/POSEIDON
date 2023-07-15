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
def emission_Toon(P, T, wl, dtau_tot, w_tot, g_tot, 
                  hard_surface = 0, tridiagonal = 0, 
                  Gauss_quad = 5, numt = 1):
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

    N_wl = len(wl)
    N_layer = len(P)
    N_level = N_layer + 1 # Number of levels

    #### No reflective surface (for now)
    surf_reflect = np.zeros(N_wl)

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
    if (Gauss_quad == 5):
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])

    # Only 5th order Gaussian quadrature is currently supported
    else:
        gangle = np.array([0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429])
        gweight = np.array([0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902])      


    # Calculate whatever ubar1 represents from the Gaussian angles
    cos_theta = 1.0
    longitude = np.arcsin((gangle-(cos_theta-1.0)/(cos_theta+1.0))/(2.0/(cos_theta+1)))
    colatitude = np.arccos(0.0)              # Colatitude = 90-latitude 
    f = np.sin(colatitude)                   # Define to eliminate repetition
    ubar1 = np.outer(np.cos(longitude), f) 

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
    g1 = 2.0 - (w_tot * (1 + g_tot))
    g2 = w_tot * (1 - g_tot)

    alpha = np.sqrt((1.0 - w_tot) / (1.0 - (w_tot * g_tot)) )
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
        b_surface = all_b[-1,:]*np.pi #for terrestrial, hard surface  
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
                int_plus[-1,:] = all_b[-1,:] *2*np.pi  # terrestrial flux /pi = intensity
            else:
                int_plus[-1,:] = ( all_b[-1,:] + b1[-1,:] * iubar)*2*np.pi #no hard surface   

            int_minus[0,:] =  (1 - np.exp(-tau_top / iubar)) * all_b[0,:] *2*np.pi
            
            exptrm_angle = np.exp( - dtau_tot / iubar)
            exptrm_angle_mdpt = np.exp( -0.5 * dtau_tot / iubar) 

            for itop in range(N_layer):

                #disbanning this for now because we dont need it in the thermal emission code
                #EQN 56,toon
                int_minus[itop+1,:]=(int_minus[itop,:]*exptrm_angle[itop,:]+
                                     (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive[itop,:]-exptrm_angle[itop,:])+
                                     (K[itop,:]/(lamda[itop,:]*iubar-1.0))*(exptrm_angle[itop,:]-exptrm_minus[itop,:])+
                                     sigma1[itop,:]*(1.-exptrm_angle[itop,:])+
                                     sigma2[itop,:]*(iubar*exptrm_angle[itop,:]+dtau_tot[itop,:]-iubar) )

                int_minus_mdpt[itop,:]=(int_minus[itop,:]*exptrm_angle_mdpt[itop,:]+
                                        (J[itop,:]/(lamda[itop,:]*iubar+1.0))*(exptrm_positive_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        (K[itop,:]/(-lamda[itop,:]*iubar+1.0))*(exptrm_minus_mdpt[itop,:]-exptrm_angle_mdpt[itop,:])+
                                        sigma1[itop,:]*(1.-exptrm_angle_mdpt[itop,:])+
                                        sigma2[itop,:]*(iubar*exptrm_angle_mdpt[itop,:]+0.5*dtau_tot[itop,:]-iubar))

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