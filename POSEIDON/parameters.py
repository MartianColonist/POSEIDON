# Functions related to processing parameters

import numpy as np
from numba.core.decorators import jit

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



def assign_free_params(param_species, object_type, PT_profile, X_profile, 
                       cloud_model, cloud_type, gravity_setting, stellar_contam, 
                       offsets_applied, error_inflation, PT_dim, X_dim, cloud_dim, 
                       TwoD_type, TwoD_param_scheme, species_EM_gradient, 
                       species_DN_gradient, species_vert_gradient,
                       Atmosphere_dimension, opaque_Iceberg):
    
    ''' Counts how many free parameters describe each chosen model feature.
    
        Also determines whether to enable clouds or hazes, and the appropriate priors.
    
    '''
    
    # Create lists storing names of free parameters
    params = []           # All parameters
    physical_params = []  # Physical parameters
    PT_params = []        # P-T profile parameters
    X_params = []         # Mixing ratio parameters
    cloud_params = []     # Cloud parameters
    geometry_params = []  # Geometry parameters
    stellar_params = []   # Stellar parameters

    #***** Physical property parameters *****#

    physical_params += ['R_p_ref']   # Reference radius parameter (R_J or R_E)

    if (gravity_setting == 'free'):
        physical_params += ['log_g']         # log_10 surface gravity (cm / s^2)

    if (object_type == 'directly_imaged'):
        physical_params += ['d']             # Distance to system (pc)

    N_physical_params = len(physical_params)   # Store number of physical parameters
    params += physical_params                  # Add physical parameter names to combined list

    #***** PT profile parameters *****#

    if (PT_profile not in ['isotherm', 'gradient', 'two-gradients', 'Madhu', 
                           'slope', 'file_read']):
        raise Exception("Error: unsupported P-T profile.")

    # Check profile settings are supported
    if ((PT_profile == 'isotherm') and (PT_dim > 1)):
        raise Exception("Cannot retrieve multiple PT profiles with an isothermal shape")
        
    if ((PT_profile == 'Madhu') and (PT_dim > 1)):
        raise Exception("Madhusudhan & Seager (2009) profile only supported for 1D models")

    if ((PT_profile == 'slope') and (PT_dim > 1)):
        raise Exception("Slope profile only supported for 1D models")
           
    # 1D model (global average)
    if (PT_dim == 1): 
 
        if (PT_profile == 'isotherm'):  
            PT_params += ['T']
        elif (PT_profile == 'gradient'):  
            PT_params += ['T_high', 'T_deep']
        elif (PT_profile == 'two-gradients'):  
            PT_params += ['T_high', 'T_mid', 'log_P_mid', 'T_deep']
        elif (PT_profile == 'Madhu'):     
            PT_params += ['a1', 'a2', 'log_P1', 'log_P2', 'log_P3', 'T_deep']
        elif (PT_profile == 'slope'):
            PT_params += ['T_phot', 'Delta_T_10-1mb', 'Delta_T_100-10mb', 
                          'Delta_T_1-0.1b', 'Delta_T_3.2-1b', 'Delta_T_10-3.2b', 
                          'Delta_T_32-10b', 'Delta_T_100-32b']
        
    # 2D model (asymmetric terminator or day-night transition)
    elif (PT_dim == 2):
        
        # Parametrisation with separate morning / evening or day / night profiles
        if (TwoD_param_scheme == 'absolute'):

            if (TwoD_type == 'E-M'):
                if (PT_profile == 'gradient'):            
                    PT_params += ['T_Even_high', 'T_Morn_high', 'T_deep']
                elif (PT_profile == 'two-gradients'):   
                    PT_params += ['T_Even_high', 'T_Even_mid', 'T_Morn_high',
                                  'T_Morn_mid', 'log_P_mid', 'T_deep']

            elif (TwoD_type == 'D-N'):
                if (PT_profile == 'gradient'):            
                    PT_params += ['T_Day_high', 'T_Night_high', 'T_deep']
                elif (PT_profile == 'two-gradients'):   
                    PT_params += ['T_Day_high', 'T_Day_mid', 'T_Night_high',
                                  'T_Night_mid', 'log_P_mid', 'T_deep']
   
        # Difference parameter prescription from MacDonald & Lewis (2022)
        elif (TwoD_param_scheme == 'difference'):

            if (TwoD_type == 'E-M'):
                if (PT_profile == 'gradient'):
                    PT_params += ['T_bar_term_high', 'Delta_T_term_high', 'T_deep']
                elif (PT_profile == 'two-gradients'):            
                    PT_params += ['T_bar_term_high', 'T_bar_term_mid', 'Delta_T_term_high', 
                                  'Delta_T_term_mid', 'log_P_mid', 'T_deep']

            elif (TwoD_type == 'D-N'):
                if (PT_profile == 'gradient'):            
                    PT_params += ['T_bar_DN_high', 'Delta_T_DN_high', 'T_deep']
                elif (PT_profile == 'two-gradients'):            
                    PT_params += ['T_bar_DN_high', 'T_bar_DN_mid', 'Delta_T_DN_high', 
                                  'Delta_T_DN_mid', 'log_P_mid', 'T_deep']
    
    # 3D model (asymmetric terminator + day-night transition)
    elif (PT_dim == 3):

        if (PT_profile == 'gradient'):            
            PT_params += ['T_bar_term_high', 'Delta_T_term_high', 'Delta_T_DN_high', 'T_deep']
        elif (PT_profile == 'two-gradients'):            
            PT_params += ['T_bar_term_high', 'T_bar_term_mid', 'Delta_T_term_high', 
                          'Delta_T_term_mid', 'Delta_T_DN_high', 'Delta_T_DN_mid', 
                          'log_P_mid', 'T_deep']
        
    N_PT_params = len(PT_params)   # Store number of P-T profile parameters
    params += PT_params            # Add P-T parameter names to combined list
    
    #***** Mixing ratio parameters *****#

    if (X_profile not in ['isochem', 'gradient', 'two-gradients', 'file_read']):
        raise Exception("Error: unsupported mixing ratio profile.")
        
    # Create list of mixing ratio free parameters
    for species in param_species:
        
        # If all species have uniform abundances (1D X_i)
        if (X_dim == 1):
            
            # Check if given species has an altitude profile or not
            if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                if (X_profile == 'gradient'):  
                    X_params += ['log_' + species + '_high', 'log_' + species + '_deep']
                elif (X_profile == 'two-gradients'):  
                    X_params += ['log_' + species + '_high', 'log_' + species + '_mid', 
                                 'log_P_' + species + '_mid', 'log_' + species + '_deep']      
            else:
                X_params += ['log_' + species]
            
        # If some species vary either around or across the terminator (2D X_i)
        elif (X_dim == 2):

            # Parametrisation with separate morning / evening or day / night profiles
            if (TwoD_param_scheme == 'absolute'):
            
                # Species with variation only around the terminator (2D Evening-Morning X_i)
                if (TwoD_type == 'E-M'):                
                    if ((species_EM_gradient != []) and (species in species_EM_gradient)):
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_Even_high', 'log_' + species + '_Morn_high', 
                                             'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_Even_high', 'log_' + species + '_Even_mid',
                                             'log_' + species + '_Morn_high', 'log_' + species + '_Morn_mid', 
                                             'log_P_' + species + '_mid', 'log_' + species + '_deep']
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species + '_Even', 'log_' + species + '_Morn']

                    else:       # No Evening-Morning variation for this species
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_mid', 
                                            'log_P_' + species + '_mid', 'log_' + species + '_deep']      
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species]
                            
                # Species with variation only across the terminator (2D Day-Night X_i)
                if (TwoD_type == 'D-N'):                
                    if ((species_DN_gradient != []) and (species in species_DN_gradient)):
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_Day_high', 'log_' + species + '_Night_high', 
                                             'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_Day_high', 'log_' + species + '_Day_mid',
                                             'log_' + species + '_Night_high', 'log_' + species + '_Night_mid', 
                                             'log_P_' + species + '_mid', 'log_' + species + '_deep']
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species + '_Day', 'log_' + species + '_Night']

                    else:       # No Day-Night variation for this species
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_mid', 
                                            'log_P_' + species + '_mid', 'log_' + species + '_deep']      
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species]
                                                
            # Difference parameter prescription from MacDonald & Lewis (2022)
            if (TwoD_param_scheme == 'difference'):

                # Species with variation only around the terminator (2D Evening-Morning X_i)
                if (TwoD_type == 'E-M'):                
                    if ((species_EM_gradient != []) and (species in species_EM_gradient)):
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_bar_term_high', 'Delta_log_' + species + '_term_high', 
                                             'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_bar_term_high', 'log_' + species + '_bar_term_mid',
                                             'Delta_log_' + species + '_term_high', 'Delta_log_' + species + '_term_mid', 
                                             'log_P_' + species + '_mid', 'log_' + species + '_deep']
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species + '_bar_term', 'Delta_log_' + species + '_term']

                    else:       # No Evening-Morning variation for this species
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_mid', 
                                            'log_P_' + species + '_mid', 'log_' + species + '_deep']      
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species]

                # Species with variation only across the terminator (2D Day-Night X_i)
                if (TwoD_type == 'D-N'):                
                    if ((species_DN_gradient != []) and (species in species_DN_gradient)):
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_bar_DN_high', 'Delta_log_' + species + '_DN_high', 
                                             'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_bar_DN_high', 'log_' + species + '_bar_DN_mid',
                                             'Delta_log_' + species + '_DN_high', 'Delta_log_' + species + '_DN_mid', 
                                             'log_P_' + species + '_mid', 'log_' + species + '_deep']
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species + '_bar_DN', 'Delta_log_' + species + '_DN']

                    else:       # No Day-Night variation for this species
                        if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                            if (X_profile == 'gradient'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_deep']
                            elif (X_profile == 'two-gradients'):  
                                X_params += ['log_' + species + '_high', 'log_' + species + '_mid', 
                                            'log_P_' + species + '_mid', 'log_' + species + '_deep']      
                        else:   # No altitude variation for this species
                            X_params += ['log_' + species]
        
        # If some species vary both around and across the terminator (3D X_i)
        elif (X_dim == 3):

            # Species with 3D mixing ratio field
            if ((species_EM_gradient != []) and (species in species_EM_gradient) and
                (species_DN_gradient != []) and (species in species_DN_gradient)):
                if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                    if (X_profile == 'gradient'):  
                        X_params += ['log_' + species + '_bar_term_high', 'Delta_log_' + species + '_term_high',
                                     'Delta_log_' + species + '_DN_high', 'log_' + species + '_deep']
                    elif (X_profile == 'two-gradients'):  
                        X_params += ['log_' + species + '_bar_term_high', 'log_' + species + '_bar_term_mid',
                                     'Delta_log_' + species + '_term_high', 'Delta_log_' + species + '_term_mid', 
                                     'Delta_log_' + species + '_DN_high', 'Delta_log_' + species + '_DN_mid', 
                                     'log_P_' + species + '_mid', 'log_' + species + '_deep']
                else:   # No altitude variation for this species
                    X_params += ['log_' + species + '_bar_term', 'Delta_log_' + species + '_term',
                                 'Delta_log_' + species + '_DN']
            
            # Species with only Evening-Morning variation
            elif ((species_EM_gradient != []) and (species in species_EM_gradient)):
                if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                    if (X_profile == 'gradient'):  
                        X_params += ['log_' + species + '_bar_term_high', 'Delta_log_' + species + '_term_high', 
                                     'log_' + species + '_deep']
                    elif (X_profile == 'two-gradients'):  
                        X_params += ['log_' + species + '_bar_term_high', 'log_' + species + '_bar_term_mid',
                                     'Delta_log_' + species + '_term_high', 'Delta_log_' + species + '_term_mid', 
                                     'log_P_' + species + '_mid', 'log_' + species + '_deep']
                else:   # No altitude variation for this species
                    X_params += ['log_' + species + '_bar_term', 'Delta_log_' + species + '_term']

            # Species with only Day-Night variation
            elif ((species_DN_gradient != []) and (species in species_DN_gradient)):
                if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                    if (X_profile == 'gradient'):  
                        X_params += ['log_' + species + '_bar_DN_high', 'Delta_log_' + species + '_DN_high', 
                                     'log_' + species + '_deep']
                    elif (X_profile == 'two-gradients'):  
                        X_params += ['log_' + species + '_bar_DN_high', 'log_' + species + '_bar_DN_mid',
                                     'Delta_log_' + species + '_DN_high', 'Delta_log_' + species + '_DN_mid', 
                                     'log_P_' + species + '_mid', 'log_' + species + '_deep']
                else:   # No altitude variation for this species
                    X_params += ['log_' + species + '_bar_DN', 'Delta_log_' + species + '_DN']

            # Species with 1D profile
            else:
                if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                    if (X_profile == 'gradient'):  
                        X_params += ['log_' + species + '_high', 'log_' + species + '_deep']
                    elif (X_profile == 'two-gradients'):  
                        X_params += ['log_' + species + '_high', 'log_' + species + '_mid', 
                                    'log_P_' + species + '_mid', 'log_' + species + '_deep']      
                else:
                    X_params += ['log_' + species]
                
    N_species_params = len(X_params)   # Store number of mixing ratio parameters
    params += X_params                 # Add mixing ratio parameter names to combined list
                   
    #***** Cloud parameters *****#
    
    # Cloud-free models need no extra free parameters
    if (cloud_model == 'cloud-free'):
        cloud_params = []

    # Patchy cloud model from MacDonald & Madhusudhan (2017)
    elif (cloud_model == 'MacMad17'):
        
        if ('haze' in cloud_type):
            cloud_params += ['log_a', 'gamma']
            
        if ('deck' in cloud_type):
            cloud_params += ['log_P_cloud']
            
        # If working with a 2D patchy cloud model
        if (cloud_dim == 2):
            cloud_params += ['phi_cloud']
            
        if (cloud_type not in ['deck', 'haze', 'deck_haze']):
            raise Exception("Error: unsupported cloud model.")

        if (cloud_dim not in [1, 2]):
            raise Exception("The MacDonald & Madhusudhan (2017) cloud model " +
                            "only supports 1D and 2D clouds")
        
    # 3D patchy cloud model from MacDonald & Lewis (2022)
    elif (cloud_model == 'Iceberg'):
        
        # Disable haze in radiative transfer (for now)
                
    #    if ('haze' in cloud_type):
    #        cloud_params = np.append(cloud_params, np.array(['log_a', 'gamma']))
            
        if ('deck' in cloud_type):
            if (opaque_Iceberg == True):
                cloud_params += ['log_P_cloud']
            else:
                cloud_params += ['log_kappa_cloud', 'log_P_cloud']
            
        # If working with a 2D patchy cloud model
        if (cloud_dim == 2):
            if (TwoD_type == 'E-M'):
                cloud_params += ['f_cloud', 'phi_0']
            elif (TwoD_type == 'D-N'):
                cloud_params += ['theta_0']
                
        # If using a full 3D patchy cloud model
        if (cloud_dim == 3):
            cloud_params += ['f_cloud', 'phi_0', 'theta_0']
            
        if ('haze' not in cloud_type) and ('deck' not in cloud_type):
            raise Exception("Error: unsupported cloud model.")
            
    else:
        raise Exception("Error: unsupported cloud model.")
        
    N_cloud_params = len(cloud_params)   # Store number of cloud parameters
    params += cloud_params               # Add cloud parameter names to combined list
        
    #***** Geometry parameters *****#
    
    if (Atmosphere_dimension == 3):
        geometry_params += ['alpha', 'beta']
    elif (Atmosphere_dimension == 2):
        if (TwoD_type == 'E-M'):
            geometry_params += ['alpha']
        elif (TwoD_type == 'D-N'):
            geometry_params += ['beta']

    N_geometry_params = len(geometry_params)   # Store number of geometry parameters
    params += geometry_params                  # Add geometry parameter names to combined list
    
    #***** Stellar contamination parameters *****#
    
    if (stellar_contam == 'one-spot'):
        stellar_params += ['f_het', 'T_het', 'T_phot']
    elif (stellar_contam == 'No'):
        stellar_params = []
    else:
        raise Exception("Error: unsupported stellar contamination model.")
        
    N_stellar_params = len(stellar_params)   # Store number of stellar parameters
    params += stellar_params                 # Add stellar parameter names to combined list
             
    #***** Offset parameters *****#
    
    if (offsets_applied == 'single-dataset'):
        params += ['delta_rel']
        N_offset_params = 1
    elif (offsets_applied == 'No'):
        N_offset_params = 0
    else:
        raise Exception("Error: unsupported offset prescription.")
     
    #***** Error adjustment parameters *****#

    if (error_inflation == 'Line15'): 
        params += ['log_b']                  # TBD: CHECK definition
        N_error_params = 1
    elif (error_inflation == 'No'):    
        N_error_params = 0
    else:
        raise Exception("Error: unsupported error adjustment prescription.")
    
    #***** Final recasting of parameter arrays *****#

    # Convert parameter lists to numpy arrays
    params = np.array(params)
    physical_params = np.array(physical_params)
    PT_params = np.array(PT_params)
    X_params = np.array(X_params)
    cloud_params = np.array(cloud_params)
    geometry_params = np.array(geometry_params)
    stellar_params = np.array(stellar_params)
    
    # The cumulative sum of the number of each type of parameter saves time indexing later 
    N_params_cumulative = np.cumsum([N_physical_params, N_PT_params, 
                                     N_species_params, N_cloud_params,
                                     N_geometry_params, N_stellar_params, 
                                     N_offset_params, N_error_params])
    
    return params, physical_params, PT_params, X_params, cloud_params, \
           geometry_params, stellar_params, N_params_cumulative
      
         
def reformat_log_X(log_X_state_in, param_species, X_profile, X_dim, TwoD_type,
                   TwoD_param_scheme, species_EM_gradient, species_DN_gradient, 
                   species_vert_gradient):
    ''' 
    DEPRECATED!

    Convert the user-provided log mixing ratio array into the 1D array 
    format expected by generate_state function.    
    '''
          
    # Load in the mixing ratio state defined in config.py for each region
    log_X_state = log_X_state_in.T   # Transpose puts species as first index
    
    log_X_params = []
    
    for q, species in enumerate(param_species):
    
        # If all species have uniform abundances, just load in their mixing ratios
        if (X_dim == 1):
            log_X_params += [log_X_state[q,0]]
            
        # For 2D mixing ratios (for at least one species)
        elif (X_dim == 2):
            
            if (TwoD_param_scheme == 'absolute'):

                # For Evening-Morning variation, parameters are the evening and morning mixing ratios
                if (TwoD_type == 'E-M'):
                    if (species in species_EM_gradient):
                        
                        # Extract abundance parameters for this species
                        log_X_E = log_X_state[q,0]
                        log_X_M = log_X_state[q,1]
                        
                        # Add parameters to list
                        log_X_params += [log_X_E, log_X_M]
                        
                    else:
                        
                        # Check if the user specified different abundances even if gradient not flagged
                        if (log_X_state[q,0] != log_X_state[q,1]):
                            raise Exception("Error: You disabled an E-M gradient for " + species +
                                            " but didn't set the evening and morning abundances equal!")
                            
                        # If the test passes (equal abundances in evening and morning), just use evening value
                        else:
                            log_X_params += [log_X_state[q,0]]   
    
                # For Day-Night variation, parameters are the terminator average and difference
                elif (TwoD_type == 'D-N'):
                    if (species in species_DN_gradient):
                        
                        # Extract abundance parameters for this species
                        log_X_D = log_X_state[q,0]
                        log_X_N = log_X_state[q,1]
                        
                        # Add parameters to list
                        log_X_params += [log_X_D, log_X_N]
                        
                    else:
                        
                        # Check if the user specified different abundances even if gradient not flagged
                        if (log_X_state[q,0] != log_X_state[q,1]):
                            raise Exception("Error: You disabled a D-N gradient for " + species +
                                            " but didn't set the day and night abundances equal!")
                            
                        # If the test passes (equal abundances in day and night), just use dayside value
                        else:
                            log_X_params += [log_X_state[q,0]]  
                            
            if (TwoD_param_scheme == 'difference'):
                
                # For Evening-Morning variation, parameters are the terminator average and difference 
                if (TwoD_type == 'E-M'):
                    if (species in species_EM_gradient):
                        
                        # Extract abundance parameters for this species
                        log_X_bar_term = log_X_state[q,0]
                        delta_log_X_term = log_X_state[q,1]
                        
                        # Add parameters to list
                        log_X_params += [log_X_bar_term, delta_log_X_term]
                        
                    else:
                        
                        # Check if the user specified different abundances even if gradient not flagged
                        if (log_X_state[q,1] != 0.0):
                            raise Exception("Error: You disabled an E-M gradient for " + species +
                                            " but didn't set the abundance difference to zero!")
                            
                        # If the test passes (equal abundances in evening and morning), just use average value
                        else:
                            log_X_params += [log_X_state[q,0]]   
    
                # For Day-Night variation, parameters are the terminator average and difference
                elif (TwoD_type == 'D-N'):
                    if (species in species_DN_gradient):
                        
                        # Extract abundance parameters for this species
                        log_X_bar_DN = log_X_state[q,0]
                        delta_log_X_DN = log_X_state[q,1]
                        
                        # Add parameters to list
                        log_X_params += [log_X_bar_DN, delta_log_X_DN]
                        
                    else:
                        
                        # Check if the user specified different abundances even if gradient not flagged
                        if (log_X_state[q,1] != 0.0):
                            raise Exception("Error: You disabled a D-N gradient for " + species +
                                            " but didn't set the day and night abundances equal!")
                            
                        # If the test passes (equal abundances in day and night), just use average value
                        else:
                            log_X_params += [log_X_state[q,0]]  

            
        # For 3D mixing ratios (for at least one species)
        elif (X_dim == 3):
            
            # Variation in both Evening-Morning and Day-Night
            if ((species in species_EM_gradient) and (species in species_DN_gradient)):
            
                # Extract abundance parameters for this species
                log_X_bar_term = log_X_state[q,0]   
                delta_log_X_term = log_X_state[q,1]
                delta_log_X_DN = log_X_state[q,2]
                
                # Add parameters to list
                log_X_params += [log_X_bar_term, delta_log_X_term, delta_log_X_DN]
               
            # Variation in Evening-Morning only (2D for this species)
            elif (species in species_EM_gradient):
                
                # Check if the user specified different abundances even if gradient not flagged
                if (log_X_state[q,2] != 0.0):
                    raise Exception("Error: You disabled a D-N gradient for " + species +
                                    " but didn't set the day and night abundances equal!")
                    
                # If the test passes (equal abundances in day and night), just use dayside value
                else:
              
                    # Extract abundance parameters for this species
                    log_X_bar_term = log_X_state[q,0]
                    delta_log_X_term = log_X_state[q,1]
                    
                    # Add parameters to list
                    log_X_params += [log_X_bar_term, delta_log_X_term]
                
            # Variation in Day-Night only (2D for this species)
            elif (species in species_DN_gradient):
                
                # Check if the user specified different abundances even if gradient not flagged
                if (log_X_state[q,1] != 0.0):
                    raise Exception("Error: You disabled an E-M gradient for " + species +
                                    " but didn't set the evening and morning abundances equal!")
                    
                # If the test passes (equal abundances in evening and morning), just use evening value
                else:
                
                    # Extract abundance parameters for this species
                    log_X_bar_DN = log_X_state[q,0]
                    delta_log_X_DN = log_X_state[q,2]
                    
                    # Add parameters to list
                    log_X_params += [log_X_bar_DN, delta_log_X_DN]
            
            else:
                
                # Check if the user specified different abundances even if gradient not flagged
                if ((log_X_state[q,1] != 0.0) or (log_X_state[q,2] != 0.0)):
                    raise Exception("Error: You disabled a D-N and E-M gradient for " + species +
                                    " but didn't set the day, night, and terminator abundances equal!")
                  
                # If the test passes (equal abundances in all regions), just use evening dayside value
                else:
                    log_X_params += [log_X_state[q,0]]  
           
        # Add deep abundance parameter for any species with a vertical gradient
        if ((X_profile == 'gradient') and (species in species_vert_gradient)):  
            log_X_params += [log_X_state[q,-1]]      
           
    # With all mixing ratio parameters loaded, convert to numpy array
    log_X_params = np.array(log_X_params)
        
    
    return log_X_params
    
    
def split_params(params, N_params_cumulative):
    ''' 
    Converts MultiNest parameter cube array, into physical, PT, mixing ratio, 
    cloud, geometry, stellar, offset, and error inflation parameters.
    
    '''
    
    # Extract physical property parameters
    physical_drawn = params[0:N_params_cumulative[0]]
    
    # Extract PT profile parameters
    PT_drawn = params[N_params_cumulative[0]:N_params_cumulative[1]]
    
    # Extract mixing ratio parameters
    log_X_drawn = params[N_params_cumulative[1]:N_params_cumulative[2]]
    
    # Extract cloud parameters    
    clouds_drawn = params[N_params_cumulative[2]:N_params_cumulative[3]]
        
    # Extract geometry parameters    
    geometry_drawn = params[N_params_cumulative[3]:N_params_cumulative[4]]
        
    # Extract stellar parameters    
    stellar_drawn = params[N_params_cumulative[4]:N_params_cumulative[5]]

    # Extract offset parameters
    offsets_drawn = params[N_params_cumulative[5]:N_params_cumulative[6]]
    
    # Extract error adjustment parameters      
    err_inflation_drawn = params[N_params_cumulative[6]:N_params_cumulative[7]]
        
    return physical_drawn, PT_drawn, log_X_drawn, clouds_drawn, geometry_drawn, \
           stellar_drawn, offsets_drawn, err_inflation_drawn    
 
    
def generate_state(PT_in, log_X_in, param_species, PT_dim, X_dim, PT_profile,
                   X_profile, TwoD_type, TwoD_param_scheme, species_EM_gradient,
                   species_DN_gradient, species_vert_gradient):
                   
    ''' 
    Converts P-T profile and mixing ratio parameter arrays into the state arrays
    format expected by the POSEIDON.atmosphere module.
        
    '''

    # Store length of each P-T profile state array
    if (PT_profile == 'isotherm'):
        len_PT = 1
    elif (PT_profile == 'gradient'):  # MacDonald & Lewis (2022) profile  
        len_PT = 4   # (T_bar_term_high, Delta_T_term_high, Delta_T_DN_high, T_deep)     
    elif (PT_profile == 'two-gradients'):
        len_PT = 8     
    elif (PT_profile == 'Madhu'):   # Madhusudhan & Seager (2009) profile
        len_PT = 6
    elif (PT_profile == 'slope'):   # Piette & Madhusudhan (2020) profile
        len_PT = 8
    
    # Store length of mixing ratio state arrays
    if (X_profile == 'gradient'):    # MacDonald & Lewis (2022) profile  
        len_X = 4      # (log_X_bar_term_high, Delta_log_X_term_high, Delta_log_X_DN_high, log_X_deep)    
    elif (X_profile == 'two-gradients'):
        len_X = 8
    elif (X_profile == 'isochem'):
        len_X = 4      # To cover multi-D cases, we use same log_X format as gradient profile
    
    # Store number of parametrised chemical species in model
    N_param_species = len(param_species)
    
    # Initialise state arrays
    PT_state = np.zeros(len_PT)   
    log_X_state = np.zeros(shape=(N_param_species, len_X))
       
    #***** Process PT profile parameters into PT state array *****#    
        
    # 1D atmosphere
    if (PT_dim == 1):
        if (PT_profile == 'isotherm'):
            PT_state = PT_in                # Assign isotherm to T_iso
        elif (PT_profile == 'gradient'):  
            PT_state[0] = PT_in[0]          # Assign T_high to T_bar_term_high
            PT_state[1] = 0.0               # No Evening-Morning gradient
            PT_state[2] = 0.0               # No Day-Night gradient
            PT_state[3] = PT_in[1]          # Assign T_deep
        elif (PT_profile == 'two-gradients'):  
            PT_state[0] = PT_in[0]          # Assign T_high to T_bar_term_high
            PT_state[1] = PT_in[1]          # Assign T_mid to T_bar_term_mid
            PT_state[2] = 0.0               # No Evening-Morning gradients
            PT_state[3] = 0.0               # No Evening-Morning gradients
            PT_state[4] = 0.0               # No Day-Night gradients
            PT_state[5] = 0.0               # No Day-Night gradients
            PT_state[6] = PT_in[2]          # Assign log_P_mid
            PT_state[7] = PT_in[3]          # Assign T_deep
        elif (PT_profile == 'Madhu'):  
            PT_state = PT_in                # Assign 6 parameters defining this profile
        elif (PT_profile == 'slope'):
            PT_state = PT_in                # Assign 8 parameters defining this profile
               
    # 2D atmosphere
    elif (PT_dim == 2):

        # Convert input parameters into average terminator temperature and difference 
        if (PT_profile == 'gradient'):
            if (TwoD_param_scheme == 'absolute'):
                T_bar = 0.5*(PT_in[0] + PT_in[1])       # T_bar_term = 0.5 * (T_E/D + T_M/N)
                Delta_T = (PT_in[0] - PT_in[1])         # Delta_T_term = T_E/D - T_M/N
            elif (TwoD_param_scheme == 'difference'):
                T_bar = PT_in[0]
                Delta_T = PT_in[1]
            T_deep = PT_in[2]

        # Convert input parameters into average terminator temperatures and differences (high and mid atmosphere) 
        elif (PT_profile == 'two-gradients'):
            if (TwoD_param_scheme == 'absolute'):
                T_bar_high = 0.5*(PT_in[0] + PT_in[1])
                T_bar_mid = 0.5*(PT_in[2] + PT_in[3])
                Delta_T_high = (PT_in[0] - PT_in[1])
                Delta_T_mid = (PT_in[2] - PT_in[3])
            elif (TwoD_param_scheme == 'difference'):
                T_bar_high = PT_in[0]
                T_bar_mid = PT_in[1]
                Delta_T_high = PT_in[2]
                Delta_T_mid = PT_in[3]
            log_P_mid = PT_in[4]
            T_deep = PT_in[5]

        # For Evening-Morning gradients
        if (TwoD_type == 'E-M'):
            if (PT_profile == 'gradient'):  
                PT_state[0] = T_bar      
                PT_state[1] = Delta_T      
                PT_state[2] = 0.0                  # No Day-Night gradient
                PT_state[3] = T_deep           
            elif (PT_profile == 'two-gradients'):  
                PT_state[0] = T_bar_high
                PT_state[1] = T_bar_mid
                PT_state[2] = Delta_T_high
                PT_state[3] = Delta_T_mid       
                PT_state[4] = 0.0                  # No Day-Night gradients
                PT_state[5] = 0.0                  # No Day-Night gradients
                PT_state[6] = log_P_mid                     
                PT_state[7] = T_deep

        # For Day-Night gradients
        elif (TwoD_type == 'D-N'):
            if (PT_profile == 'gradient'):  
                PT_state[0] = T_bar
                PT_state[1] = 0.0                  # No Evening-Morning gradient
                PT_state[2] = Delta_T
                PT_state[3] = T_deep
            elif (PT_profile == 'two-gradients'):  
                PT_state[0] = T_bar_high
                PT_state[1] = T_bar_mid
                PT_state[2] = 0.0                   # No Evening-Morning gradients
                PT_state[3] = 0.0                   # No Evening-Morning gradients
                PT_state[4] = Delta_T_high
                PT_state[5] = Delta_T_mid       
                PT_state[6] = log_P_mid                     
                PT_state[7] = T_deep

    # 3D atmosphere
    elif (PT_dim == 3):
        if (PT_profile == 'gradient'):
            PT_state[0] = PT_in[0]              # T_bar_term_high
            PT_state[1] = PT_in[1]              # Delta_T_term_high
            PT_state[2] = PT_in[2]              # Delta_T_DN_high
            PT_state[3] = PT_in[3]              # T_deep
        elif (PT_profile == 'two-gradients'):  
            PT_state[0] = PT_in[0]              # T_bar_term_high
            PT_state[1] = PT_in[1]              # T_bar_term_mid
            PT_state[2] = PT_in[2]              # Delta_T_term_high
            PT_state[3] = PT_in[3]              # Delta_T_term_mid
            PT_state[4] = PT_in[4]              # Delta_T_DN_high
            PT_state[5] = PT_in[5]              # Delta_T_DN_mid
            PT_state[6] = PT_in[6]              # log_P_mid
            PT_state[7] = PT_in[7]              # T_deep
            
    #***** Process mixing ratio parameters into mixing ratio state array *****#
                
    # 1D atmosphere
    if (X_dim == 1):

        if (X_profile == 'isochem'):

            log_X_state[:,0] = log_X_in     # Assign log_X_iso to log_X_bar_term_high
            log_X_state[:,1] = 0.0          # No Evening-Morning gradient
            log_X_state[:,2] = 0.0          # No Day-Night gradient
            log_X_state[:,3] = log_X_in     # Assign log_X_iso to log_X_deep
        
        elif (X_profile == 'gradient'):  
            
            count = 0  # Counter to make tracking location in log_X_in easier
            
            # Loop over parametrised chemical species
            for q, species in enumerate(param_species):
                if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                    log_X_state[q,0] = log_X_in[count]       # log_X_bar_term_high
                    log_X_state[q,1] = 0.0                   # No Evening-Morning gradient
                    log_X_state[q,2] = 0.0                   # No Day-Night gradient
                    log_X_state[q,3] = log_X_in[count+1]     # log_X_deep
                    count += 2
                else:   # No altitude variation for this species
                    log_X_state[q,0] = log_X_in[count]       # log_X_bar_term_high
                    log_X_state[q,1] = 0.0                   # No Evening-Morning gradient
                    log_X_state[q,2] = 0.0                   # No Day-Night gradient
                    log_X_state[q,3] = log_X_in[count]       # log_X_deep
                    count += 1
        
        elif (X_profile == 'two-gradients'):  
            
            count = 0  # Counter to make tracking location in log_X_in easier
            
            # Loop over parametrised chemical species
            for q, species in enumerate(param_species):
                if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                    log_X_state[q,0] = log_X_in[count]       # log_X_bar_term_high
                    log_X_state[q,1] = log_X_in[count+1]     # log_X_bar_term_mid
                    log_X_state[q,2] = 0.0                   # No Evening-Morning gradient
                    log_X_state[q,3] = 0.0                   # No Evening-Morning gradient
                    log_X_state[q,4] = 0.0                   # No Day-Night gradient
                    log_X_state[q,5] = 0.0                   # No Day-Night gradient
                    log_X_state[q,6] = log_X_in[count+2]     # log_P_X_mid
                    log_X_state[q,7] = log_X_in[count+3]     # log_X_deep
                    count += 4
                else:   # No altitude variation for this species
                    log_X_state[q,0] = log_X_in[count]       # log_X_bar_term_high
                    log_X_state[q,1] = log_X_in[count]       # log_X_bar_term_mid
                    log_X_state[q,2] = 0.0                   # No Evening-Morning gradient
                    log_X_state[q,3] = 0.0                   # No Evening-Morning gradient
                    log_X_state[q,4] = 0.0                   # No Day-Night gradient
                    log_X_state[q,5] = 0.0                   # No Day-Night gradient
                    log_X_state[q,6] = -2.0                  # Fix P (X_mid) to 10 mbar for isochem
                    log_X_state[q,7] = log_X_in[count]       # log_X_deep
                    count += 1

    # 2D atmosphere
    elif (X_dim == 2):

        count = 0  # Counter to make tracking location in log_X_in easier
    
        # Loop over parametrised chemical species
        for q, species in enumerate(param_species):

            # Convert input parameters into average terminator mixing ratio and difference
            if (X_profile == 'isochem'):
                if (((species_EM_gradient != []) and (species in species_EM_gradient)) or 
                    ((species_DN_gradient != []) and (species in species_DN_gradient))):
                    if (TwoD_param_scheme == 'absolute'):
                        log_X_bar = 0.5*(log_X_in[count] + log_X_in[count+1])
                        Delta_log_X = (log_X_in[count] - log_X_in[count+1])
                    elif (TwoD_param_scheme == 'difference'):
                        log_X_bar = log_X_in[count]
                        Delta_log_X = log_X_in[count+1]
                    count += 2
                else:
                    log_X_bar = log_X_in[count]
                    Delta_log_X = 0.0
                    count += 1
                log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case

            # Convert input parameters into average terminator mixing ratio and difference
            elif (X_profile == 'gradient'):
                if (((species_EM_gradient != []) and (species in species_EM_gradient)) or 
                    ((species_DN_gradient != []) and (species in species_DN_gradient))):
                    if (TwoD_param_scheme == 'absolute'):
                        log_X_bar = 0.5*(log_X_in[count] + log_X_in[count+1])
                        Delta_log_X = (log_X_in[count] - log_X_in[count+1])
                    elif (TwoD_param_scheme == 'difference'):
                        log_X_bar = log_X_in[count]
                        Delta_log_X = log_X_in[count+1]
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_deep = log_X_in[count+2]
                        count += 3
                    else:
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 2
                else:
                    log_X_bar = log_X_in[count]
                    Delta_log_X = 0.0
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_deep = log_X_in[count+1]
                        count += 2
                    else:
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 1

            # Convert input parameters into average terminator mixing ratios and differences (high and mid atmosphere)
            elif (X_profile == 'two-gradients'):
                if (((species_EM_gradient != []) and (species in species_EM_gradient)) or 
                    ((species_DN_gradient != []) and (species in species_DN_gradient))):
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        if (TwoD_param_scheme == 'absolute'):
                            log_X_bar_high = 0.5*(log_X_in[count] + log_X_in[count+1])
                            log_X_bar_mid = 0.5*(log_X_in[count+2] + log_X_in[count+3])
                            Delta_log_X_high = (log_X_in[count] - log_X_in[count+1])
                            Delta_log_X_mid = (log_X_in[count+2] - log_X_in[count+3])
                        elif (TwoD_param_scheme == 'difference'):
                            log_X_bar_high = log_X_in[count]
                            log_X_bar_mid = log_X_in[count+1]
                            Delta_log_X_high = log_X_in[count+2]
                            Delta_log_X_mid = log_X_in[count+3]
                        log_P_X_mid = log_X_in[count+4]
                        log_X_deep = log_X_in[count+5]
                        count += 6
                    else:
                        if (TwoD_param_scheme == 'absolute'):
                            log_X_bar_high = 0.5*(log_X_in[count] + log_X_in[count+1])
                            log_X_bar_mid = log_X_bar_high
                            Delta_log_X_high = (log_X_in[count] - log_X_in[count+1])
                            Delta_log_X_mid = Delta_log_X_high
                        elif (TwoD_param_scheme == 'difference'):
                            log_X_bar_high = log_X_in[count]
                            log_X_bar_mid = log_X_bar_high
                            Delta_log_X_high = log_X_in[count+1]
                            Delta_log_X_mid = Delta_log_X_high
                        log_P_X_mid = -2.0     # Fix P (X_mid) to 10 mbar for isochem
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 2
                else:
                    log_X_bar_high = log_X_in[count]
                    Delta_log_X_high = 0.0
                    Delta_log_X_mid = 0.0
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_bar_mid = log_X_in[count+1]
                        log_P_X_mid = log_X_in[count+2]
                        log_X_deep = log_X_in[count+3]
                        count += 4
                    else:
                        log_X_bar_mid = log_X_bar_high
                        log_P_X_mid = -2.0     # Fix P (X_mid) to 10 mbar for isochem
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 1

            # For Evening-Morning gradients
            if (TwoD_type == 'E-M'):
                if (X_profile in ['isochem', 'gradient']):
                    log_X_state[q,0] = log_X_bar
                    log_X_state[q,1] = Delta_log_X
                    log_X_state[q,2] = 0.0                  # No Day-Night gradient
                    log_X_state[q,3] = log_X_deep
                elif (X_profile == 'two-gradients'):
                    log_X_state[q,0] = log_X_bar_high
                    log_X_state[q,1] = log_X_bar_mid
                    log_X_state[q,2] = Delta_log_X_high
                    log_X_state[q,3] = Delta_log_X_mid
                    log_X_state[q,4] = 0.0                  # No Day-Night gradients
                    log_X_state[q,5] = 0.0                  # No Day-Night gradients
                    log_X_state[q,6] = log_P_X_mid
                    log_X_state[q,7] = log_X_deep

            # For Day-Night gradients
            if (TwoD_type == 'D-N'):
                if (X_profile in ['isochem', 'gradient']):
                    log_X_state[q,0] = log_X_bar
                    log_X_state[q,1] = 0.0                  # No Evening-Morning gradient
                    log_X_state[q,2] = Delta_log_X
                    log_X_state[q,3] = log_X_deep
                elif (X_profile == 'two-gradients'):
                    log_X_state[q,0] = log_X_bar_high
                    log_X_state[q,1] = log_X_bar_mid
                    log_X_state[q,2] = 0.0                   # No Evening-Morning gradients
                    log_X_state[q,3] = 0.0                   # No Evening-Morning gradients
                    log_X_state[q,4] = Delta_log_X_high
                    log_X_state[q,5] = Delta_log_X_mid
                    log_X_state[q,6] = log_P_X_mid
                    log_X_state[q,7] = log_X_deep

    # 3D atmosphere
    elif (X_dim == 3):

        count = 0  # Counter to make tracking location in log_X_in easier
    
        # Loop over parametrised chemical species
        for q, species in enumerate(param_species):

            if (X_profile == 'isochem'):
                if (((species_EM_gradient != []) and (species in species_EM_gradient)) and
                    ((species_DN_gradient != []) and (species in species_DN_gradient))):
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = log_X_in[count+1]
                    Delta_log_X_DN = log_X_in[count+2]
                    count += 3
                elif ((species_EM_gradient != []) and (species in species_EM_gradient)):
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = log_X_in[count+1]
                    Delta_log_X_DN = 0.0
                    count += 2
                elif ((species_DN_gradient != []) and (species in species_DN_gradient)):
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = 0.0
                    Delta_log_X_DN = log_X_in[count+1]
                    count += 2
                else:
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = 0.0
                    Delta_log_X_DN = 0.0
                    count += 1
                log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case

            elif (X_profile == 'gradient'):
                if (((species_EM_gradient != []) and (species in species_EM_gradient)) and
                    ((species_DN_gradient != []) and (species in species_DN_gradient))):
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = log_X_in[count+1]
                    Delta_log_X_DN = log_X_in[count+2]
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_deep = log_X_in[count+3]
                        count += 4
                    else:
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 3
                elif ((species_EM_gradient != []) and (species in species_EM_gradient)):
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = log_X_in[count+1]
                    Delta_log_X_DN = 0.0
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_deep = log_X_in[count+2]
                        count += 3
                    else:
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 2
                elif ((species_DN_gradient != []) and (species in species_DN_gradient)):
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = 0.0
                    Delta_log_X_DN = log_X_in[count+1]
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_deep = log_X_in[count+2]
                        count += 3
                    else:
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 2     
                else:
                    log_X_bar_term = log_X_in[count]
                    Delta_log_X_term = 0.0
                    Delta_log_X_DN = 0.0
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_deep = log_X_in[count+1]
                        count += 2
                    else:
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 1
                
            elif (X_profile == 'two-gradients'):
                if (((species_EM_gradient != []) and (species in species_EM_gradient)) and
                    ((species_DN_gradient != []) and (species in species_DN_gradient))):
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_bar_term_high = log_X_in[count]
                        log_X_bar_term_mid = log_X_in[count+1]
                        Delta_log_X_term_high = log_X_in[count+2]
                        Delta_log_X_term_mid = log_X_in[count+3]
                        Delta_log_X_DN_high = log_X_in[count+4]
                        Delta_log_X_DN_mid = log_X_in[count+5]
                        log_P_X_mid = log_X_in[count+6]
                        log_X_deep = log_X_in[count+7]
                        count += 8
                    else:
                        log_X_bar_term_high = log_X_in[count]
                        log_X_bar_term_mid = log_X_bar_term_high
                        Delta_log_X_term_high = log_X_in[count+1]
                        Delta_log_X_term_mid = Delta_log_X_term_high
                        Delta_log_X_DN_high = log_X_in[count+2]
                        Delta_log_X_DN_mid = Delta_log_X_DN_high
                        log_P_X_mid = -2.0     # Fix P (X_mid) to 10 mbar for isochem
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 3
                elif ((species_EM_gradient != []) and (species in species_EM_gradient)):
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_bar_term_high = log_X_in[count]
                        log_X_bar_term_mid = log_X_in[count+1]
                        Delta_log_X_term_high = log_X_in[count+2]
                        Delta_log_X_term_mid = log_X_in[count+3]
                        Delta_log_X_DN_high = 0.0
                        Delta_log_X_DN_high = 0.0
                        log_P_X_mid = log_X_in[count+4]
                        log_X_deep = log_X_in[count+5]
                        count += 6
                    else:
                        log_X_bar_term_high = log_X_in[count]
                        log_X_bar_term_mid = log_X_bar_term_high
                        Delta_log_X_term_high = log_X_in[count+1]
                        Delta_log_X_term_mid = Delta_log_X_term_high
                        Delta_log_X_DN_high = 0.0
                        Delta_log_X_DN_mid = 0.0
                        log_P_X_mid = -2.0     # Fix P (X_mid) to 10 mbar for isochem
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 2
                elif ((species_DN_gradient != []) and (species in species_DN_gradient)):
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_bar_term_high = log_X_in[count]
                        log_X_bar_term_mid = log_X_in[count+1]
                        Delta_log_X_term_high = 0.0
                        Delta_log_X_term_mid = 0.0
                        Delta_log_X_DN_high = log_X_in[count+2]
                        Delta_log_X_DN_high = log_X_in[count+3]
                        log_P_X_mid = log_X_in[count+4]
                        log_X_deep = log_X_in[count+5]
                        count += 6
                    else:
                        log_X_bar_term_high = log_X_in[count]
                        log_X_bar_term_mid = log_X_bar_term_high
                        Delta_log_X_term_high = 0.0
                        Delta_log_X_term_mid = 0.0
                        Delta_log_X_DN_high = log_X_in[count+1]
                        Delta_log_X_DN_mid = Delta_log_X_term_high
                        log_P_X_mid = -2.0     # Fix P (X_mid) to 10 mbar for isochem
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 2
                else:
                    log_X_bar_term_high = log_X_in[count]
                    Delta_log_X_term_high = 0.0
                    Delta_log_X_term_mid = 0.0
                    Delta_log_X_DN_high = 0.0
                    Delta_log_X_DN_mid = 0.0
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_bar_term_mid = log_X_in[count+1]
                        log_P_X_mid = log_X_in[count+2]
                        log_X_deep = log_X_in[count+3]
                        count += 4
                    else:
                        log_X_bar_term_mid = log_X_bar_term_high
                        log_P_X_mid = -2.0     # Fix P (X_mid) to 10 mbar for isochem
                        log_X_deep = -50.0     # Dummy value for log_X_deep, since not used in this case
                        count += 1

            if (X_profile in ['isochem', 'gradient']):
                log_X_state[q,0] = log_X_bar_term
                log_X_state[q,1] = Delta_log_X_term
                log_X_state[q,2] = Delta_log_X_DN
                log_X_state[q,3] = log_X_deep
            elif (X_profile == 'two-gradients'):
                log_X_state[q,0] = log_X_bar_term_high
                log_X_state[q,1] = log_X_bar_term_mid
                log_X_state[q,2] = Delta_log_X_term_high
                log_X_state[q,3] = Delta_log_X_term_mid
                log_X_state[q,4] = Delta_log_X_DN_high
                log_X_state[q,5] = Delta_log_X_DN_mid
                log_X_state[q,6] = log_P_X_mid
                log_X_state[q,7] = log_X_deep
                
    return PT_state, log_X_state


def unpack_cloud_params(param_names, clouds_state, cloud_model, cloud_dim,
                        N_params_cumulative, TwoD_type):
    
    # Unpack names of cloud parameters
    cloud_param_names = param_names[N_params_cumulative[2]:N_params_cumulative[3]]

    # Check if haze enabled in the cloud model
    if ('log_a' in cloud_param_names):
        enable_haze = 1
    else:
        enable_haze = 0

    # Check if a cloud deck is enabled in the cloud model
    if ('log_P_cloud' in cloud_param_names):
        enable_deck = 1
    else:
        enable_deck = 0

    # Clear atmosphere
    if (cloud_model == 'cloud-free'):
        
        # Set dummy parameter values, not used when cloud-free
        kappa_cloud_0 = 1.0e250
        P_cloud = 100.0
        a, gamma = 1.0, -4.0  
        f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0

    # Patchy cloud model from MacDonald & Madhusudhan (2017)
    if (cloud_model == 'MacMad17'):
        
        kappa_cloud_0 = 1.0e250    # Cloud deck assumed to have infinite opacity

        # If haze is enabled in this model
        if (enable_haze == 1):
            a = np.power(10.0, clouds_state[np.where(cloud_param_names == 'log_a')[0][0]])
            gamma = clouds_state[np.where(cloud_param_names == 'gamma')[0][0]]
        else:
            a, gamma = 1.0, -4.0   # Dummy values, not used for models without hazes
        
        # If cloud deck enabled
        if (enable_deck == 1):
            P_cloud = np.power(10.0, clouds_state[np.where(cloud_param_names == 'log_P_cloud')[0][0]])
        else:
            P_cloud = 100.0   # Set to 100 bar for models without a cloud deck
            
        # If cloud model has patchy gaps
        if (cloud_dim != 1):
            phi_c = clouds_state[np.where(cloud_param_names == 'phi_cloud')[0][0]]    
       #     phi_0 = clouds_state[np.where(cloud_param_names == 'phi_0')[0][0]]
            phi_0 = 0.0       # Cloud start position doesn't matter for MacMad17
            f_cloud = phi_c   # Rename for consistency with Iceberg cloud
            theta_0 = -90.0   # Uniform from day to night
        else:
            if (enable_deck == 1):
                f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0  # 1D uniform cloud
            else:
                f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0   # Dummy values, not used when cloud-free
         
    # 3D patchy cloud model from MacDonald & Lewis (2022)
    elif (cloud_model == 'Iceberg'):
        
        # No haze in this model
        a, gamma = 1.0, -4.0   # Dummy values, haze extinction disabled here
        
        # If cloud deck enabled
        if (enable_deck == 1):
            
            kappa_cloud_0 = np.power(10.0, clouds_state[np.where(cloud_param_names == 'log_kappa_cloud')[0][0]])
            P_cloud = np.power(10.0, clouds_state[np.where(cloud_param_names == 'log_P_cloud')[0][0]])
            
            if (cloud_dim == 1):
                f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0   # 1D uniform cloud

            elif (cloud_dim == 2):
                if (TwoD_type == 'E-M'):
                    f_cloud = clouds_state[np.where(cloud_param_names == 'f_cloud')[0][0]]    
                    phi_0 = clouds_state[np.where(cloud_param_names == 'phi_0')[0][0]]
                    theta_0 = -90.0                # Cloud spans full day to night zones
                if (TwoD_type == 'D-N'):
                    f_cloud, phi_0 = 1.0, -90.0    # Uniform axially, not-uniform along ray
                    theta_0 = clouds_state[np.where(cloud_param_names == 'theta_0')[0][0]]
            
            elif (cloud_dim == 3):
                f_cloud = clouds_state[np.where(cloud_param_names == 'f_cloud')[0][0]]    
                phi_0 = clouds_state[np.where(cloud_param_names == 'phi_0')[0][0]]
                theta_0 = clouds_state[np.where(cloud_param_names == 'theta_0')[0][0]]
             
        else:   # Set dummy parameter values, not used when cloud-free
            kappa_cloud_0 = 1.0e250
            P_cloud = 100.0   
            f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0
            
    return kappa_cloud_0, P_cloud, f_cloud, phi_0, theta_0, a, gamma


def unpack_geometry_params(param_names, N_params_cumulative, geometry_state):
    
    # Unpack names of geometry parameters
    geometry_param_names = param_names[N_params_cumulative[3]:N_params_cumulative[4]]
    
    if ('alpha' in geometry_param_names):
        alpha = geometry_state[np.where(geometry_param_names == 'alpha')[0][0]]
    else:
        alpha = 0.0
        
    if ('beta' in geometry_param_names):
        beta = geometry_state[np.where(geometry_param_names == 'beta')[0][0]]
    else:
        beta = 0.0
    
    return alpha, beta
