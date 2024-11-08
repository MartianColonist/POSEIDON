''' 
Functions related to the free parameters defining a POSEIDON model.

'''

import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def assign_free_params(param_species, object_type, PT_profile, X_profile, 
                       cloud_model, cloud_type, gravity_setting, mass_setting,
                       stellar_contam, offsets_applied, error_inflation, 
                       PT_dim, X_dim, cloud_dim, TwoD_type, TwoD_param_scheme, 
                       species_EM_gradient, species_DN_gradient, species_vert_gradient,
                       Atmosphere_dimension, opaque_Iceberg, surface,
                       sharp_DN_transition, reference_parameter, disable_atmosphere,
                       aerosol_species, log_P_slope_arr, number_P_knots, PT_penalty):
    '''
    From the user's chosen model settings, determine which free parameters 
    define this POSEIDON model. The different types of free parameters are
    returned as separate arrays.
    
    Args:
        param_species (list of str):
            Chemical species with parametrised mixing ratios (trace species).
        object_type (str):
            Type of planet / brown dwarf the user wishes to model
            (Options: transiting / directly_imaged).
        PT_profile (str):
            Chosen P-T profile parametrisation 
            (Options: isotherm / gradient / two-gradients / Madhu / slope / 
             file_read).
        X_profile (str):
            Chosen mixing ratio profile parametrisation
            (Options: isochem / gradient / two-gradients / lever / file_read).
        cloud_model (str):
            Chosen cloud parametrisation 
            (Options: cloud-free / MacMad17 / Iceberg / Mie).
        cloud_type (str):
            Cloud extinction type to consider 
            (Options: deck / haze / deck_haze).
        gravity_setting (str):
            Whether log_g is fixed or a free parameter.
            (Options: fixed / free).
        mass_setting (str):
            Whether the planetary mass is fixed or a free parameter.
            (Options: fixed / free).
        stellar_contam (str):
            Chosen prescription for modelling unocculted stellar contamination
            (Options: one_spot / one_spot_free_log_g / two_spots / 
             two_spots_free_log_g).
        offsets_applied (str):
            Whether a relative offset should be applied to a dataset 
            (Options: single_dataset / two_datasets / three_datasets).
        error_inflation (str):
            Whether to consider inflation of error bars in a retrieval
            (Options: Line15).
        PT_dim (int):
            Dimensionality of the pressure-temperature field (uniform -> 1, 
            a day-night or evening-morning gradient -> 2, both day-night and 
            evening-morning gradients -> 3)
            (Options: 1 / 2 / 3).
        X_dim (int):
            Max dimensionality of the mixing ratio field (not all species need
            have gradients, this just specifies the highest dimensionality of 
            chemical gradients -- see the species_XX_gradient arguments)
            (Options: 1 / 2 / 3).
        cloud_dim (int):
            Dimensionality of the cloud model prescription (only the Iceberg
            cloud model supports 3D clouds)
            (Options: 1 / 2 / 3). 
        TwoD_type (str):
            For 2D models, specifies whether the model considers day-night
            gradients or evening-morning gradients
            (Options: D-N / E-M).
        TwoD_param_scheme (str):
            For 2D models, specifies which quantities should be consider as
            free parameters (e.g. day & night vs. terminator & day-night diff.)
            (Options: absolute / difference).
        species_EM_gradient (list of str):
            List of chemical species with an evening-morning mixing ratio gradient.
        species_DN_gradient (list of str):
            List of chemical species with a day-night mixing ratio gradient.
        species_vert_gradient (list of str):
            List of chemical species with a vertical mixing ratio gradient.
        Atmosphere_dimension (int):
            The dimensionality of the model atmosphere
            (Options: 1 / 2 / 3).
        opaque_Iceberg (bool):
            If using the Iceberg cloud model, True disables the kappa parameter.
        surface (bool):
            If True, model a surface via an opaque cloud deck.
        sharp_DN_transition (bool):
            For 2D / 3D models, sets day-night transition width (beta) to 0.
        reference_parameter (str):
            For retrievals, whether R_p_ref or P_ref will be a free parameter
            (Options: R_p_ref / P_ref).
        disable_atmosphere (bool):
            If True, returns a flat planetary transmission spectrum @ (Rp/R*)^2
        aerosol_species (list of string):
            Either 'free' or specific aerosol(s).
        log_P_slope_array (np.array of float):
            Log pressures where the temperature difference parameters are 
            defined (Piette & Madhusudhan 2020 profile only).
        number_P_knots (float):
            Number of uniform knots in pressure space 
            (only for the Pelletier 2021 P-T profile).
        PT_penalty (bool):
            If True, introduces the sigma_smooth parameter for retrievals
            (only for the Pelletier 2021 P-T profile).

    Returns:
        params (np.array of str):
            Free parameters defining this POSEIDON model.
        physical_params (np.array of str):
            Physical parameters of the planet.
        PT_params (np.array of str):
            Pressure-temperature profile parameters.
        X_params (np.array of str):
            Mixing ratio parameters.
        cloud_params (np.array of str):
            Aerosol parameters.
        geometry_params (np.array of str):
            Multidimensional atmospheric geometry parameters.
        stellar_params (np.array of str):
            Stellar heterogeneity parameters.
        N_params_cumulative (np.array of int):
            Cumulative sum of number of parameters (used for indexing).

    '''
    
    # Create lists storing names of free parameters
    params = []           # All parameters
    physical_params = []  # Physical parameters
    PT_params = []        # P-T profile parameters
    X_params = []         # Mixing ratio parameters
    cloud_params = []     # Cloud parameters
    geometry_params = []  # Geometry parameters
    stellar_params = []   # Stellar parameters

    # Models with no atmosphere only have Rp as a free parameter
    if (disable_atmosphere == True):

        physical_params += ['R_p_ref']   # Reference radius parameter (in R_J or R_E)

        N_physical_params = 1
        N_PT_params = 0 
        N_species_params = 0
        N_cloud_params = 0
        N_geometry_params = 0

        params += physical_params         # Add physical parameter names to combined list

    # Models including atmospheres
    else:

        #***** Physical property parameters *****#

        if (reference_parameter == 'R_p_ref'):
            physical_params += ['R_p_ref']                  # Reference radius parameter (in R_J or R_E)
        elif (reference_parameter == 'P_ref'):
            physical_params += ['log_P_ref']                # Reference pressure
        elif (reference_parameter == 'R_p_ref+P_ref'):
            physical_params += ['R_p_ref', 'log_P_ref']     # Reference radius and pressure
        else:
            raise Exception("Error: Only R_p_ref or P_ref are supported reference parameters.")
        
        if (('log_P_ref' in reference_parameter) and (Atmosphere_dimension > 1)):
            raise Exception("Error: P_ref is not a valid parameter for multidimensional models.")

        if ((gravity_setting == 'free') and (mass_setting == 'free')):
            raise Exception("Error: only one of mass or gravity can be a free parameter.")

        if (gravity_setting == 'free'):
            physical_params += ['log_g']         # log_10 surface gravity (cm / s^2)

        if (mass_setting == 'free'):
            physical_params += ['M_p']           # Planetary mass

        if (object_type == 'directly_imaged'):
            physical_params += ['d']             # Distance to system (pc)

        if (surface == True):
            physical_params += ['log_P_surf']       # Rocky planet surface pressure (bar)

        N_physical_params = len(physical_params)   # Store number of physical parameters
        params += physical_params                  # Add physical parameter names to combined list

        #***** PT profile parameters *****#

        if (PT_profile not in ['isotherm', 'gradient', 'two-gradients', 'Madhu', 
                               'slope', 'Pelletier', 'Guillot', 'Guillot_dayside',
                               'Line', 'file_read']):
            raise Exception("Error: unsupported P-T profile.")

        # Check profile settings are supported
        if ((PT_profile == 'isotherm') and (PT_dim > 1)):
            raise Exception("Cannot retrieve multiple PT profiles with an isothermal shape")
            
        if ((PT_profile == 'Madhu') and (PT_dim > 1)):
            raise Exception("Madhusudhan & Seager (2009) profile only supported for 1D models")
        
        if ((PT_profile == 'Pelletier') and (PT_dim > 1)):
            raise Exception("Pelletier (2021) profile only supported for 1D models")
        
        if ((PT_profile == 'Guillot') and (PT_dim > 1)):
            raise Exception("Guillot (2010) (pRT implementation) profile only supported for 1D models")
        
        if ((PT_profile == 'Guillot_dayside') and (PT_dim > 1)):
            raise Exception("Guillot (dayside) (2010) (pRT implementation) profile only supported for 1D models")
        
        if ((PT_profile == 'Line') and (PT_dim > 1)):
            raise Exception("Line (2013) (platon implementation) profile only supported for 1D models")

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
                PT_params += ['a1', 'a2', 'log_P1', 'log_P2', 'log_P3', 'T_ref']
            elif (PT_profile == 'slope'):
                PT_params += ['T_phot_PT']
                for i in range(len(log_P_slope_arr)):
                    PT_params += ['Delta_T_' + str(i+1)]
            elif (PT_profile == 'Pelletier'):
                if number_P_knots < 3:
                    raise Exception('number_P_knots must be at least 3. (Captures top, bottom, middle pressures in log space)')
                for i in range(number_P_knots):
                    PT_params += ['T_' + str(i+1)]
                if PT_penalty == True:
                    PT_params += ['sigma_s']
            elif (PT_profile == 'Guillot'):
                PT_params += ['log_kappa_IR', 'log_gamma', 'T_int', 'T_equ']
            elif (PT_profile == 'Guillot_dayside'):
                PT_params += ['log_kappa_IR', 'log_gamma', 'T_int', 'T_equ']
            elif (PT_profile == 'Line'):
                PT_params += ['log_kappa_IR', 'log_gamma', 'log_gamma_2', 
                              'alpha_Line', 'beta_Line', 'T_int']
            
        # 2D model (asymmetric terminator or day-night transition)
        elif (PT_dim == 2):

            # Check that a 2D model type has been specified 
            if (TwoD_type not in ['D-N', 'E-M']):
                raise Exception("Error: 2D model type is not 'D-N' or 'E-M'.")
            
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

            # Gradient parameter prescription from MacDonald & Lewis (2023)
            elif (TwoD_param_scheme == 'gradient'):

                if (TwoD_type == 'D-N'):
                    if (PT_profile == 'gradient'):
                        PT_params += ['T_bar_DN_high', 'Grad_theta_T_high', 'T_deep']
                    elif (PT_profile == 'two-gradients'):            
                        PT_params += ['T_bar_DN_high', 'T_bar_DN_mid', 'Grad_theta_T_high', 
                                    'Grad_theta_T_mid', 'log_P_mid', 'T_deep']
        
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

        if (X_profile not in ['isochem', 'gradient', 'two-gradients', 'file_read', 'lever', 'chem_eq']):
            raise Exception("Error: unsupported mixing ratio profile.")
        
        if (X_profile == 'lever') and (X_dim != 1):
            raise Exception("Error: Level profile currently only implemented for 1D atmospheres.")
            
        if X_profile != 'chem_eq':
        
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
                        elif (X_profile == 'lever'):
                            X_params += ['log_' + species + '_iso', 'log_P_' + species, 'Upsilon_' + species]
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

                    # Gradient parameter prescription from MacDonald & Lewis (2023)
                    if (TwoD_param_scheme == 'gradient'):

                        # Species with variation only across the terminator (2D Day-Night X_i)
                        if (TwoD_type == 'D-N'):                
                            if ((species_DN_gradient != []) and (species in species_DN_gradient)):
                                if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                                    if (X_profile == 'gradient'):  
                                        X_params += ['log_' + species + '_bar_DN_high', 'Grad_theta_log_' + species + '_high', 
                                                     'log_' + species + '_deep']
                                    elif (X_profile == 'two-gradients'):  
                                        X_params += ['log_' + species + '_bar_DN_high', 'log_' + species + '_bar_DN_mid',
                                                     'Grad_theta_log_' + species + '_high', 'Grad_theta_log_' + species + '_mid', 
                                                     'log_P_' + species + '_mid', 'log_' + species + '_deep']
                                else:   # No altitude variation for this species
                                    X_params += ['log_' + species + '_bar_DN', 'Grad_theta_log_' + species + '_DN']

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
    
        # If the X profile is set to chem_eq, then we have only CO and log_met
        else:
            X_params = ['C_to_O','log_Met']
                
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

        # Mie scattering     
        elif (cloud_model == 'Mie'):

            # If working with a 2D patchy cloud model
            if (cloud_dim == 2):
                cloud_params += ['f_cloud']

            if (cloud_type =='fuzzy_deck'):

                if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                    cloud_params += ['log_P_top_deck']
                    cloud_params += ['log_r_m', 'log_n_max', 'f']
                    cloud_params += ['r_i_real', 'r_i_complex']
                else:
                    for aerosol in aerosol_species:
                        cloud_params += ['log_P_top_deck_' + aerosol]
                        cloud_params += ['log_r_m_' + aerosol]
                        cloud_params += ['log_n_max_' + aerosol]
                        cloud_params += ['f_' + aerosol]

            elif (cloud_type == 'slab'):

                if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                    print('Only one slab can be defined with free or file_read')
                    cloud_params += ['log_P_top_slab']
                    cloud_params += ['Delta_log_P']
                    cloud_params += ['log_r_m']
                    cloud_params += ['log_X_Mie','r_i_real', 'r_i_complex']
                else:
                    for aerosol in aerosol_species:
                        cloud_params += ['log_P_top_slab_' + aerosol]
                        cloud_params += ['Delta_log_P_' + aerosol]
                        cloud_params += ['log_r_m_' + aerosol]
                        cloud_params += ['log_X_' + aerosol]

            elif (cloud_type == 'fuzzy_deck_plus_slab'):
                if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                    raise Exception('Error : Cannot use more than one aerosol species for file_read or free. Add aerosol to database first')
                else:
                    print('This mode assumes that the first aerosol in the list is the deck species, rest are slab species')
                    for index in range(len(aerosol_species)):
                        aerosol = aerosol_species[index]
                        if index == 0:
                            cloud_params += ['log_P_top_deck_' + aerosol]
                            cloud_params += ['log_r_m_' + aerosol]
                            cloud_params += ['log_n_max_' + aerosol]
                            cloud_params += ['f_' + aerosol]
                        else:
                            cloud_params += ['log_P_top_slab_' + aerosol]
                            cloud_params += ['Delta_log_P_' + aerosol]
                            cloud_params += ['log_r_m_' + aerosol]
                            cloud_params += ['log_X_' + aerosol]

            elif (cloud_type == 'opaque_deck_plus_slab'):

                cloud_params += ['log_P_top_deck']

                if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                        print('Only one slab can be defined with free or file_read')
                        cloud_params += ['log_P_top_slab']
                        cloud_params += ['Delta_log_P']
                        cloud_params += ['log_r_m']
                        cloud_params += ['log_X_Mie','r_i_real', 'r_i_complex']
                else:
                    for aerosol in aerosol_species:
                        cloud_params += ['log_P_top_slab_' + aerosol]
                        cloud_params += ['Delta_log_P_' + aerosol]
                        cloud_params += ['log_r_m_' + aerosol]
                        cloud_params += ['log_X_' + aerosol]
 
            elif (cloud_type == 'uniform_X'):

                if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                    cloud_params += ['log_r_m']
                    cloud_params += ['log_X_Mie','r_i_real', 'r_i_complex']
                else:
                    for aerosol in aerosol_species:
                        cloud_params += ['log_r_m_' + aerosol]
                        cloud_params += ['log_X_' + aerosol]

            elif (cloud_type == 'opaque_deck_plus_uniform_X'):

                cloud_params += ['log_P_top_deck']

                if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                    cloud_params += ['log_r_m']
                    cloud_params += ['log_X_Mie','r_i_real', 'r_i_complex']
                else:
                    for aerosol in aerosol_species:
                        cloud_params += ['log_r_m_' + aerosol]
                        cloud_params += ['log_X_' + aerosol]
            
            # One slab defined with multiple species included
            elif (cloud_type == 'one_slab'):

                if (aerosol_species == ['free'] or aerosol_species == ['file_read']):
                    raise Exception('Only one cloud species can be defined when using free or file_read. Use slab instead')

                else:
                    # define slab top pressure and extent
                    cloud_params += ['log_P_top_slab']
                    cloud_params += ['Delta_log_P']
                    
                    # add aerosol specific parameters
                    for aerosol in aerosol_species: 
                        cloud_params += ['log_r_m_' + aerosol]
                        cloud_params += ['log_X_' + aerosol]

            # Uniaxial random, uniaxial seperate, biaxial random, biaxial seperate slabs 
            # Make error message for fuzzy deck and UX (sorry!)
            # Random will assume 2/3 and 1/3 for uniaxial and 1/3 1/3 1/3 for biaxial 
            # Seperate will allow free mixing ratios, but keep particle size, slab width the same. 
            elif ('uniaxial' in cloud_type):

                if len(aerosol_species) > 2:
                    raise Exception('Only can include two aerosols for uniaxial models in this order: Ordinary then Extraordinary')
                
                if (cloud_type == 'uniaxial_random_slab'):
                    # define slab top pressure and extent
                    cloud_params += ['log_P_top_slab']
                    cloud_params += ['Delta_log_P']

                    # For the random, we only assume one mixing ratio and then split it 
                    cloud_params += ['log_r_m_uniaxial']
                    cloud_params += ['log_X_uniaxial']

                    print('This model assumes the first aerosol is ordinary (2/3 weight), then extraordinary (1/3 weight)')

                elif(cloud_type == 'uniaxial_slab'):
                    # define slab top pressure and extent
                    cloud_params += ['log_P_top_slab']
                    cloud_params += ['Delta_log_P']

                    # Still assume the same radius
                    cloud_params += ['log_r_m_uniaxial']

                    # Have seperate mixing ratios
                    cloud_params += ['log_X_ordinary']
                    cloud_params += ['log_X_extraordinary']

                    print('This model assumes the first aerosol is ordinary, then extraordinary')
                

            elif ('biaxial' in cloud_type):

                if len(aerosol_species) > 3:
                    raise Exception('Only can include three aerosols for biaxial models in this order: E||c (B1U,Ex), E||b (B2U,Ey), E||a (B3U, Ez)')
                
                if (cloud_type == 'biaxial_random_slab'):
                    # define slab top pressure and extent
                    cloud_params += ['log_P_top_slab']
                    cloud_params += ['Delta_log_P']

                    # For the random, we only assume one mixing ratio and then split it 
                    cloud_params += ['log_r_m_biaxial']
                    cloud_params += ['log_X_biaxial']

                    print('This model assumes the aerosols are in this order:  E||c (B1U,Ex), E||b (B2U,Ey), E||a (B3U, Ez)')

                elif(cloud_type == 'biaxial_slab'):
                    # define slab top pressure and extent
                    cloud_params += ['log_P_top_slab']
                    cloud_params += ['Delta_log_P']

                    # Still assume the same radius
                    cloud_params += ['log_r_m_biaxial']

                    # Have seperate mixing ratios
                    cloud_params += ['log_X_Ec']
                    cloud_params += ['log_X_Eb']
                    cloud_params += ['log_X_Ea']

                    print('This model assumes the aerosols are in this order:  E||c (B1U,Ex), E||b (B2U,Ey), E||a (B3U, Ez)')

            elif (cloud_type not in ['fuzzy_deck', 'uniform_X', 'slab', 'fuzzy_deck_plus_slab', 'opaque_deck_plus_slab', 
                                     'opaque_deck_plus_uniform_X', 'one_slab',
                                     'uniaxial_random_slab', 'uniaxial_slab',
                                     'biaxial_random_slab', 'biaxial_slab']):
                raise Exception("Error: unsupported cloud type. Supported types : fuzzy_deck, uniform_X, slab, one_slab, fuzzy_deck_plus_slab, opaque_deck_plus_slab, opaque_deck_plus_uniform_X.")
        
        elif (cloud_model == 'eddysed'):
            # If working with a 2D patchy cloud model
            if (cloud_dim == 2):
                cloud_params += ['f_cloud']

            cloud_params += ['kappa_cloud_eddysed']
            cloud_params += ['g_cloud_eddysed']
            cloud_params += ['w_cloud_eddysed']


        else:
            raise Exception("Error: unsupported cloud model.")
            
        N_cloud_params = len(cloud_params)   # Store number of cloud parameters
        params += cloud_params               # Add cloud parameter names to combined list
            
        #***** Geometry parameters *****#
        
        if (Atmosphere_dimension == 3):
            if (sharp_DN_transition == False):
                geometry_params += ['alpha', 'beta']
            else:
                geometry_params += ['alpha']
        elif (Atmosphere_dimension == 2):
            if (TwoD_type == 'E-M'):
                geometry_params += ['alpha']
            elif ((TwoD_type == 'D-N') and (sharp_DN_transition == False)):
                geometry_params += ['beta']

        N_geometry_params = len(geometry_params)   # Store number of geometry parameters
        params += geometry_params                  # Add geometry parameter names to combined list
    
    #***** Stellar contamination parameters *****#
    
    if (stellar_contam == 'one_spot'):
        stellar_params += ['f_het', 'T_het', 'T_phot']
    elif (stellar_contam == 'one_spot_free_log_g'):
        stellar_params += ['f_het', 'T_het', 'T_phot', 'log_g_het', 'log_g_phot']
    elif (stellar_contam == 'two_spots'):
        stellar_params += ['f_spot', 'f_fac', 'T_spot', 'T_fac', 'T_phot']
    elif (stellar_contam == 'two_spots_free_log_g'):
        stellar_params += ['f_spot', 'f_fac', 'T_spot', 'T_fac', 'T_phot', 
                           'log_g_spot', 'log_g_fac', 'log_g_phot']
    elif (stellar_contam == None):
        stellar_params = []
    else:
        raise Exception("Error: unsupported stellar contamination model.")
        
    N_stellar_params = len(stellar_params)   # Store number of stellar parameters
    params += stellar_params                 # Add stellar parameter names to combined list
             
    #***** Offset parameters *****#
    
    if (offsets_applied == 'single_dataset'):
        params += ['delta_rel']
        N_offset_params = 1
    elif (offsets_applied == 'two_datasets'):
        params += ['delta_rel_1', 'delta_rel_2']
        N_offset_params = 2
    elif (offsets_applied == 'three_datasets'):
        params += ['delta_rel_1', 'delta_rel_2', 'delta_rel_3']
        N_offset_params = 3
    elif (offsets_applied == None):
        N_offset_params = 0
    else:
        raise Exception("Error: unsupported offset prescription.")
     
    #***** Error adjustment parameters *****#

    if (error_inflation == 'Line15'): 
        params += ['b']
        N_error_params = 1
    elif (error_inflation == None):    
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
    
    
def split_params(params_drawn, N_params_cumulative):
    '''
    Split the array of drawn parameters from a retrieval (e.g. the MultiNest 
    parameter cube) into the separate types of parameters used by POSEIDON.
    
    Args:
        params_drawn (list of float | np.array of float):
            Values of the free parameters drawn from the prior distribution.
        N_params_cumulative (np.array of int):
            Cumulative sum of number of parameters (used for indexing).

    Returns:
        physical_drawn (list of float | np.array of float):
            Drawn values of the physical parameters of the planet.
        PT_drawn (list of float | np.array of float):
            Drawn values of the pressure-temperature profile parameters.
        log_X_drawn (list of float | np.array of float):
            Drawn values of the mixing ratio parameters.
        clouds_drawn (list of float | np.array of float):
            Drawn values of the aerosol parameters.
        geometry_drawn (list of float | np.array of float):
            Drawn values of the multidimensional atmospheric geometry parameters.
        stellar_drawn (list of float | np.array of float):
            Drawn values of the stellar heterogeneity parameters.
        offsets_drawn (list of float | np.array of float):
            Drawn values of the data offset parameters.
        err_inflation_drawn (list of float | np.array of float):
            Drawn values of the error inflation parameters.

    '''
    
    # Extract physical property parameters
    physical_drawn = params_drawn[0:N_params_cumulative[0]]
    
    # Extract PT profile parameters
    PT_drawn = params_drawn[N_params_cumulative[0]:N_params_cumulative[1]]
    
    # Extract mixing ratio parameters
    log_X_drawn = params_drawn[N_params_cumulative[1]:N_params_cumulative[2]]
    
    # Extract cloud parameters    
    clouds_drawn = params_drawn[N_params_cumulative[2]:N_params_cumulative[3]]
        
    # Extract geometry parameters    
    geometry_drawn = params_drawn[N_params_cumulative[3]:N_params_cumulative[4]]
        
    # Extract stellar parameters    
    stellar_drawn = params_drawn[N_params_cumulative[4]:N_params_cumulative[5]]

    # Extract offset parameters
    offsets_drawn = params_drawn[N_params_cumulative[5]:N_params_cumulative[6]]
    
    # Extract error adjustment parameters      
    err_inflation_drawn = params_drawn[N_params_cumulative[6]:N_params_cumulative[7]]
        
    return physical_drawn, PT_drawn, log_X_drawn, clouds_drawn, geometry_drawn, \
           stellar_drawn, offsets_drawn, err_inflation_drawn
 
    
def generate_state(PT_in, log_X_in, param_species, PT_dim, X_dim, PT_profile,
                   X_profile, TwoD_type, TwoD_param_scheme, species_EM_gradient,
                   species_DN_gradient, species_vert_gradient, alpha, beta):
    '''
    Convert the P-T profile and mixing ratio parameters into the state array
    format expected by the POSEIDON.atmosphere module. This function is called
    by 'make_atmosphere' in core.py.
    
    Args:
        PT_in (list of float | np.array of float):
            Drawn values of the pressure-temperature profile parameters.
        log_X_in (list of float | np.array of float):
            Drawn values of the mixing ratio parameters.
        param_species (list of str):
            Chemical species with parametrised mixing ratios (trace species).
        PT_dim (int):
            Dimensionality of the pressure-temperature field (uniform -> 1, 
            a day-night or evening-morning gradient -> 2, both day-night and 
            evening-morning gradients -> 3)
            (Options: 1 / 2 / 3).
        X_dim (int):
            Max dimensionality of the mixing ratio field (not all species need
            have gradients, this just specifies the highest dimensionality of 
            chemical gradients -- see the species_XX_gradient arguments)
            (Options: 1 / 2 / 3).
        PT_profile (str):
            Chosen P-T profile parametrisation 
            (Options: isotherm / gradient / two-gradients / Madhu / slope / file_read).
        X_profile (str):
            Chosen mixing ratio profile parametrisation
            (Options: isochem / gradient / two-gradients / file_read / chem_eq).
        TwoD_type (str):
            For 2D models, specifies whether the model considers day-night
            gradients or evening-morning gradients
            (Options: D-N / E-M).
        TwoD_param_scheme (str):
            For 2D models, specifies which quantities should be consider as
            free parameters (e.g. day & night vs. terminator & day-night diff.)
            (Options: absolute / difference).
        species_EM_gradient (list of str):
            List of chemical species with an evening-morning mixing ratio gradient.
        species_DN_gradient (list of str):
            List of chemical species with a day-night mixing ratio gradient.
        species_vert_gradient (list of str):
            List of chemical species with a vertical mixing ratio gradient.

    Returns:
        PT_state (np.array of float):
            P-T profile state array for the POSEIDON.atmosphere module.
        log_X_state (2D np.array of float):
            Mixing ratio state array for the POSEIDON.atmosphere module.

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
    elif (PT_profile == 'Pelletier'): # Pelletier (2021)
        len_PT = len(PT_in)
    elif (PT_profile == 'Guillot'): # Guillot (2010)
        len_PT = 4
    elif (PT_profile == 'Guillot_dayside'): # Guillot with f = 0.5 (2010)
        len_PT = 4
    elif (PT_profile == 'Line'): # Line (2013)
        len_PT = 6
    elif (PT_profile == 'slope'):   # Piette & Madhusudhan (2020) profile
        len_PT = 8
    elif (PT_profile == 'file_read'):   # User provided file
        len_PT = 0
    
    # Store length of mixing ratio state arrays
    if (X_profile == 'gradient'):    # MacDonald & Lewis (2022) profile  
        len_X = 4      # (log_X_bar_term_high, Delta_log_X_term_high, Delta_log_X_DN_high, log_X_deep)    
    elif (X_profile == 'two-gradients'):
        len_X = 8
    elif (X_profile == 'lever'):
        len_X = 3                     # (log_X_iso, log_P_X, Upsilon_X)
    elif (X_profile == 'isochem'):
        len_X = 4      # To cover multi-D cases, we use same log_X format as gradient profile
    elif (X_profile == 'file_read'):   # User provided file
        len_X = 0
    elif (X_profile == 'chem_eq'):   # Chemical equilibrium 
        len_X = 0
    
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
        elif (PT_profile == 'Pelletier'):
            PT_state = PT_in
        elif (PT_profile == 'Guillot'):
            PT_state = PT_in
        elif (PT_profile == 'Guillot_dayside'):
            PT_state = PT_in
        elif (PT_profile == 'Line'):
            PT_state = PT_in
               
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
            elif (TwoD_param_scheme == 'gradient'):
                T_bar = PT_in[0]
                if (TwoD_type == 'D-N'):
                    Delta_T = -1.0 * (PT_in[1] * beta)    # Delta_T = Grad_T * beta
                elif (TwoD_type == 'E-M'):
                    Delta_T = -1.0 * (PT_in[1] * alpha)    # Delta_T = - Grad_T * beta
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
            elif (TwoD_param_scheme == 'gradient'):
                T_bar_high = PT_in[0]
                T_bar_mid = PT_in[1]
                if (TwoD_type == 'D-N'):
                    Delta_T_high = -1.0 * (PT_in[2] * beta)
                    Delta_T_mid = -1.0 * (PT_in[3] * beta)
                elif (TwoD_type == 'E-M'):
                    Delta_T_high = -1.0 * (PT_in[2] * alpha)
                    Delta_T_mid = -1.0 * (PT_in[3] * alpha)
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
    
    if X_profile != 'chem_eq':

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

            elif (X_profile == 'lever'):  
                
                count = 0  # Counter to make tracking location in log_X_in easier
                
                # Loop over parametrised chemical species
                for q, species in enumerate(param_species):
                    if ((species_vert_gradient != []) and (species in species_vert_gradient)):
                        log_X_state[q,0] = log_X_in[count]       # log_X_iso
                        log_X_state[q,1] = log_X_in[count+1]     # log_P_X
                        log_X_state[q,2] = log_X_in[count+2]     # Upsilon_X
                        count += 3
                    else:   # No altitude variation for this species
                        log_X_state[q,0] = log_X_in[count]       # log_X_iso
                        log_X_state[q,1] = -2.0                  # Fix P_X to 10 mbar for isochem
                        log_X_state[q,2] = 0.0                   # Upsilon_X = 0 for isochem
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
                        elif (TwoD_param_scheme == 'gradient'):
                            log_X_bar = log_X_in[count]
                            if (TwoD_type == 'D-N'):
                                Delta_log_X = -1.0 * (log_X_in[count+1] * beta)
                            elif (TwoD_type == 'E-M'):
                                Delta_log_X = -1.0 * (log_X_in[count+1] * alpha)
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
                        elif (TwoD_param_scheme == 'gradient'):
                            log_X_bar = log_X_in[count]
                            if (TwoD_type == 'D-N'):
                                Delta_log_X = -1.0 * (log_X_in[count+1] * beta)
                            elif (TwoD_type == 'E-M'):
                                Delta_log_X = -1.0 * (log_X_in[count+1] * alpha)
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
                            elif (TwoD_param_scheme == 'gradient'):
                                log_X_bar_high = log_X_in[count]
                                log_X_bar_mid = log_X_in[count+1]
                                if (TwoD_type == 'D-N'):
                                    Delta_log_X_high = -1.0 * (log_X_in[count+2] * beta)
                                    Delta_log_X_mid = -1.0 * (log_X_in[count+3] * beta)
                                elif (TwoD_type == 'E-M'):
                                    Delta_log_X_high = -1.0 * (log_X_in[count+2] * alpha)
                                    Delta_log_X_mid = -1.0 * (log_X_in[count+3] * alpha)
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
                            elif (TwoD_param_scheme == 'gradient'):
                                if (TwoD_type == 'D-N'):
                                    Delta_log_X_high = -1.0 * (log_X_in[count+1] * beta)
                                    Delta_log_X_mid = Delta_log_X_high
                                elif (TwoD_type == 'E-M'):
                                    Delta_log_X_high = -1.0 * (log_X_in[count+1] * alpha)
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

    # If it is chem_eq, then we need the 
    else:
        log_X_state = log_X_in
                
    return PT_state, log_X_state


def unpack_cloud_params(param_names, clouds_in, cloud_model, cloud_dim,
                        N_params_cumulative, TwoD_type):
    '''
    Extract the aerosol property values (e.g cloud pressure and scattering
    properties) from the drawn cloud parameters, according to the cloud model
    specified by the user.
    
    Args:
        param_names (np.array of str):
            Free parameters defining this POSEIDON model.
        clouds_in (list of float | np.array of float):
            Drawn values of the aerosol parameters.
        cloud_model (str):
            Chosen cloud parametrisation 
            (Options: cloud-free / MacMad17 / Iceberg).
        cloud_dim (int):
            Dimensionality of the cloud model prescription (only the Iceberg
            cloud model supports 3D clouds)
            (Options: 1 / 2 / 3).
        N_params_cumulative (np.array of int):
            Cumulative sum of number of parameters (used for indexing).
        TwoD_type (str):
            For 2D models, specifies whether the model considers day-night
            gradients or evening-morning gradients
            (Options: D-N / E-M).

    Returns:
        kappa_cloud_0 (float):
            Grey cloud extinction coefficient (m^-1).
        P_cloud (float):
            Cloud top pressure (bar).
        f_cloud (float):
            Terminator azimuthal cloud fraction for 2D/3D models.
        phi_0 (float):
            Azimuthal angle in terminator plane, measured clockwise from the 
            North pole, where the patchy cloud begins for 2D/3D models (degrees).
        theta_0 (float):
            Zenith angle from the terminator plane, measured towards the 
            nightside, where the patchy cloud begins for 2D/3D models (degrees).
        a (float):
            Haze 'Rayleigh enhancement factor', relative to H2 Rayleigh 
            scattering, as defined in MacDonald & Madhusudhan (2017).
        gamma (float):
            Haze power law exponent, as defined in MacDonald & Madhusudhan (2017).

    '''
    
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
        P_slab_bottom = 100.0
        a, gamma = 1.0, -4.0  
        f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0

        # Mie scattering parameters not needed
        r_m = []
        log_n_max = 0
        fractional_scale_height = 0
        r_i_real = 0
        r_i_complex = 0
        log_X_Mie = []

        # Set eddysed values to dummy values 
        kappa_cloud_eddysed = 0
        g_cloud_eddysed = 0
        w_cloud_eddysed = 0

    # Patchy cloud model from MacDonald & Madhusudhan (2017)
    if (cloud_model == 'MacMad17'):
        
        kappa_cloud_0 = 1.0e250    # Cloud deck assumed to have infinite opacity

        # If haze is enabled in this model
        if (enable_haze == 1):
            a = np.power(10.0, clouds_in[np.where(cloud_param_names == 'log_a')[0][0]])
            gamma = clouds_in[np.where(cloud_param_names == 'gamma')[0][0]]
        else:
            a, gamma = 1.0, -4.0   # Dummy values, not used for models without hazes
        
        # If cloud deck enabled
        if (enable_deck == 1):
            P_cloud = np.power(10.0, clouds_in[np.where(cloud_param_names == 'log_P_cloud')[0][0]])
        else:
            P_cloud = 100.0   # Set to 100 bar for models without a cloud deck

        P_slab_bottom = 100.0  # Not used for this model
            
        # If cloud model has patchy gaps
        if (cloud_dim != 1):
            phi_c = clouds_in[np.where(cloud_param_names == 'phi_cloud')[0][0]]    
            phi_0 = 0.0       # Cloud start position doesn't matter for MacMad17
            f_cloud = phi_c   # Rename for consistency with Iceberg cloud
            theta_0 = -90.0   # Uniform from day to night
        else:
            if (enable_deck == 1):
                f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0  # 1D uniform cloud
            else:
                f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0   # Dummy values, not used when cloud-free

        # Mie scattering parameters not needed
        r_m = []
        log_n_max = 0
        fractional_scale_height = 0
        r_i_real = 0
        r_i_complex = 0
        log_X_Mie = []

        # Set eddysed values to dummy values 
        kappa_cloud_eddysed = 0
        g_cloud_eddysed = 0
        w_cloud_eddysed = 0
         
    # 3D patchy cloud model from MacDonald & Lewis (2022)
    elif (cloud_model == 'Iceberg'):
        
        # No haze in this model
        a, gamma = 1.0, -4.0   # Dummy values, haze extinction disabled here
        
        # If cloud deck enabled
        if (enable_deck == 1):
            
            # Check if cloud is fixed to be opaque
            if ('log_kappa_cloud' in cloud_param_names):
                kappa_cloud_0 = np.power(10.0, clouds_in[np.where(cloud_param_names == 'log_kappa_cloud')[0][0]])
            else:
                kappa_cloud_0 = 1.0e250

            P_cloud = np.power(10.0, clouds_in[np.where(cloud_param_names == 'log_P_cloud')[0][0]])
            
            if (cloud_dim == 1):
                f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0   # 1D uniform cloud

            elif (cloud_dim == 2):
                if (TwoD_type == 'E-M'):
                    f_cloud = clouds_in[np.where(cloud_param_names == 'f_cloud')[0][0]]    
                    phi_0 = clouds_in[np.where(cloud_param_names == 'phi_0')[0][0]]
                    theta_0 = -90.0                # Cloud spans full day to night zones
                if (TwoD_type == 'D-N'):
                    f_cloud, phi_0 = 1.0, -90.0    # Uniform axially, not-uniform along ray
                    theta_0 = clouds_in[np.where(cloud_param_names == 'theta_0')[0][0]]
            
            elif (cloud_dim == 3):
                f_cloud = clouds_in[np.where(cloud_param_names == 'f_cloud')[0][0]]    
                phi_0 = clouds_in[np.where(cloud_param_names == 'phi_0')[0][0]]
                theta_0 = clouds_in[np.where(cloud_param_names == 'theta_0')[0][0]]
             
        else:   # Set dummy parameter values, not used when cloud-free
            kappa_cloud_0 = 1.0e250
            P_cloud = 100.0   
            f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0

        P_slab_bottom = 100.0  # Not used for this model

        # Mie scattering parameters not needed
        r_m = []
        log_n_max = 0
        fractional_scale_height = 0
        r_i_real = 0
        r_i_complex = 0
        log_X_Mie = []

        # Set eddysed values to dummy values 
        kappa_cloud_eddysed = 0
        g_cloud_eddysed = 0
        w_cloud_eddysed = 0

    # Mie clouds 
    elif (cloud_model == 'Mie'):

        # Turn clouds_in into an array 
        clouds_in = np.asarray(clouds_in)
        
        # No haze in this model
        a, gamma = 1.0, -4.0   # Dummy values, haze extinction disabled here

        # Cloud is opaque up to P_cloud, and then follows an exponential distribution for 
        # number density of aerosols. This is set by n_cloud 
        kappa_cloud_0 = 1.0e250

        # Set eddysed values to dummy values 
        kappa_cloud_eddysed = 0
        g_cloud_eddysed = 0
        w_cloud_eddysed = 0
        
        if (cloud_dim == 1):
            f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0   # 1D uniform cloud
        elif (cloud_dim == 2):
            try:
                f_cloud = clouds_in[np.where(cloud_param_names == 'f_cloud')[0][0]]    
            except:
                f_cloud = clouds_in[np.where(cloud_param_names == 'f_cloud')[0]] 

            phi_0 = -90
            theta_0 = -90.0       

        # Set the Mie parameters 
        # Sorry that below is a bit of a mess if anyone is looking at this in the future
        # There are a lot of different cloud models, and 
        # numpy gets mad depending on if the inputs are an array or a string (file read vs database aerosol)
        # Details below

        # If the model is a uniaxial or biaxial slab run this, if not then run the 'normal' param code
        # Note that this does not need a try/except clause since 'file_read' is not an option here. 
        # Also, I had to put this if statement first because I couldn't figure out how to get the 
        # if/else statement to work otherwise :(
        if ('axial' in s for s in cloud_param_names):
            # If it is uniaxial_random_slab or uniaxial_slab
            if any ('uniaxial' in s for s in cloud_param_names):

                # Shared parameters between the two models
                # Note that we pull the [0] out so that its not an array, so we can build the array in the next step
                r_m_uniaxial = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m_uniaxial')!= -1)[0]])[0]

                # Need to make r_m an array of 2 since the two species share the same radius
                r_m = np.array([r_m_uniaxial, r_m_uniaxial])

                P_cloud = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top_slab')!= -1)[0]])
                P_slab_bottom = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top_slab')!= -1)[0]] + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0]])

                # If it is uniaxial_slab
                if any ('ordinary' in s for s in cloud_param_names):
                    
                    # We pull the [0] here so we can build the log_X_Mie array 
                    log_X_ordinary = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X_ordinary')!= -1)[0]][0]
                    log_X_extraoardinary = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X_extraordinary')!= -1)[0]][0]
                    log_X_Mie = np.array([log_X_ordinary,log_X_extraoardinary])
                
                # else, its a uniaxial random slab (2/3 O + 1/3 E)
                else:
                    # We pull the [0] here so we can build the log_X_Mie array
                    log_X_uniaxial = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X_uniaxial')!= -1)[0]][0]
                    log_X_Mie = np.array([np.log10(10**log_X_uniaxial*(2/3)),np.log10(10**log_X_uniaxial*(1/3))])

            # Else, it is a biaxial_random_slab or a biaxial_slab
            else:
                # Shared parameters between the two models 
                # Note that we pull the [0] out so that its not an array, so we can build the array in the next step
                r_m_biaxial = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m_biaxial')!= -1)[0]])[0]

                # Need to make r_m an array of 3 since the three species share the same radius
                r_m = np.array([r_m_biaxial, r_m_biaxial, r_m_biaxial])

                P_cloud = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top_slab')!= -1)[0]])
                P_slab_bottom = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top_slab')!= -1)[0]] + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0]])

                # If it is biaxial_slab
                if any ('Ea' in s for s in cloud_param_names):
                    
                    # We pull the [0] here so we can build the log_X_Mie array 
                    log_X_Ec = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X_Ec')!= -1)[0]][0]
                    log_X_Eb = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X_Eb')!= -1)[0]][0]
                    log_X_Ea = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X_Ea')!= -1)[0]][0]
                    log_X_Mie = np.array([log_X_Ec,log_X_Eb,log_X_Ea])
                
                # else, its a biaxial random slab (1/3 weight to all three)
                else:
                    # We pull the [0] here so we can build the log_X_Mie array 
                    log_X_biaxial = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X_biaxial')!= -1)[0]][0]
                    log_X_Mie = np.array([np.log10(10**log_X_biaxial*(1/3)),np.log10(10**log_X_biaxial*(1/3)),np.log10(10**log_X_biaxial*(1/3))])

            # Need to set the deck parameters to dummy values to pass into the cloud object 
            log_n_max = 0
            fractional_scale_height = 0

        # Models that aren't uniaxial or biaxial
        else:
            
            # If the cloud is a fuzzy_deck, slab, fuzzy_deck_plus_slab, or opaque_deck_plus_slab model 
            # it will contain the string 'log_P_top' in its parameter names 
            if any("log_P_top" in s for s in cloud_param_names):

                # If the cloud is a slab model, a fuzzy_deck_plus_slab, or an opaque_deck_plus_slab model 
                # It will contain the string 'Delta_log_P' in it 
                if any('Delta_log_P' in s for s in cloud_param_names):

                    # If it is a fuzzy_deck_plus_slab or an opaque_deck_plus_slab 
                    # It will contain the string 'log_P_top_deck' in it 
                    if any ('log_P_top_deck' in s for s in cloud_param_names):

                        # If its a fuzzy_deck_plus_slab
                        # It will contain the string 'log_n_max' in it 
                        if any ('log_n_max' in s for s in cloud_param_names):

                            # The try except is here because file_read option makes everything into a numpy file
                            # So you have to add an extra [0] to get the indexing to work correctly 
                            # Specifically, the file_read option doesn't like the float power line in r_m
                            try:
                                # Note of whats going on here 
                                # np.char.find is finding strings in the cloud param names that have the string 
                                # And that either returns a list of arrays, or something else, which is why we have the try-except 

                                # Shared parameters, where the first index needs to be the deck species 
                                # This is why we keep the deck and slab seperate and then join them together
                                r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0]])
                                P_deck = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_deck')!= -1)[0]])
                                P_slab = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_slab')!= -1)[0]])
                                P_cloud = np.concatenate((P_deck,P_slab), axis = 0) 

                                # Deck specific parameters 
                                log_n_max = clouds_in[np.where(np.char.find(cloud_param_names, 'log_n_max')!= -1)[0]]
                                fractional_scale_height = clouds_in[np.where(np.char.find(cloud_param_names, 'f')!= -1)[0]]

                                # Slab specific parameters 
                                log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]]
                                P_slab_bottom = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top_slab')!= -1)[0]] + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0]])
                                
                            except:
                                r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0][0]])
                                P_deck = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_deck')!= -1)[0][0]])
                                P_slab = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_slab')!= -1)[0][0]])
                                P_cloud = np.concatenate((P_deck,P_slab), axis = 0) 

                                log_n_max = clouds_in[np.where(np.char.find(cloud_param_names, 'log_n_max')!= -1)[0]][0]
                                fractional_scale_height = clouds_in[np.where(np.char.find(cloud_param_names, 'f')!= -1)[0]][0]

                                log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0][0]]
                                P_slab_bottom = np.power(10.0, (P_slab + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0][0]]))
                                

                        # Otherwise, it is an opaque_deck_plus_slab
                        else:
                            try:
                                # Shared parameters, where the first index needs to be the deck species 
                                # This is why we keep the deck and slab seperate and then join them together
                                r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0]])
                                P_deck = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_deck')!= -1)[0]])
                                P_slab = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_slab')!= -1)[0]])
                                P_cloud = np.concatenate((P_deck,P_slab), axis = 0) 

                                # Opaque decks don't have the 'fuzziness' parameters 
                                log_n_max = 0
                                fractional_scale_height = 0

                                # Slab specific parameters 
                                log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]]
                                P_slab_bottom = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top_slab')!= -1)[0]] + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0]])
                                
                            except:
                                r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0][0]])
                                P_deck = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_deck')!= -1)[0][0]])
                                P_slab = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names, 'log_P_top_slab')!= -1)[0][0]])
                                try:
                                    P_cloud = np.concatenate((P_deck,P_slab), axis = 0) 
                                # Put in because the file_read option doesn't like concatenate for some reason
                                except:
                                    P_cloud = np.array([P_deck, P_slab])

                                log_n_max = 0
                                fractional_scale_height = 0

                                log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0][0]]
                                P_slab_bottom = np.power(10.0, (P_slab + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0][0]]))

                    
                    # Otherwise, it is just a slab model 
                    else:
                        try:
                            r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0]])
                            log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]]
                            P_cloud = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top')!= -1)[0]])
                            P_slab_bottom = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top')!= -1)[0]] + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0]])
                        except:
                            r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0][0]])
                            log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]][0]
                            P_cloud = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top')!= -1)[0][0]])
                            P_slab_bottom = np.power(10.0, (P_cloud + clouds_in[np.where(np.char.find(cloud_param_names, 'Delta_log_P') != -1)[0][0]]))

                        # Need to set the deck parameters to dummy values to pass into the cloud object 
                        log_n_max = 0
                        fractional_scale_height = 0
                
                # Otherwise, its just a fuzzy deck with no slabs or opaque_deck_plus_uniform_X
                else:

                    # Fuzzy deck 
                    if any ('log_n_max' in s for s in cloud_param_names):
                    
                        try:
                            r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0]])
                            P_cloud = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top')!= -1)[0]])
                            log_n_max = clouds_in[np.where(np.char.find(cloud_param_names, 'log_n_max')!= -1)[0]]
                            fractional_scale_height = clouds_in[np.where(np.char.find(cloud_param_names, 'f')!= -1)[0]]
                        except:
                            r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0][0]])
                            P_cloud = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top')!= -1)[0][0]])
                            log_n_max = clouds_in[np.where(np.char.find(cloud_param_names, 'log_n_max')!= -1)[0]][0]
                            fractional_scale_height = clouds_in[np.where(np.char.find(cloud_param_names, 'f')!= -1)[0]][0]

                        # Need to set the slab parameters to dummy values to pass into the cloud object 
                        log_X_Mie = 100
                        P_slab_bottom = 100.0
                    
                    # opaque_deck_plus_uniform_x
                    else:

                        try:
                            r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0]])
                            P_cloud = np.power(10.0, clouds_in[np.where(np.char.find(cloud_param_names,'log_P_top')!= -1)[0]])
                            log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]]
                            
                        except:
                            r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0][0]])
                            P_cloud = np.power(10.0, clouds_in[np.where(cloud_param_names == 'log_P_top')[0][0]])
                            log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]][0]

                        # If uniform_X, you set the slab and deck parameters to dummy values to pass into the cloud object
                        log_n_max = 0
                        fractional_scale_height = 0
                        P_slab_bottom = 100.0

            
            # If the aerosol is a uniform_X
            else:

                try:
                    r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0]])
                    log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]]
                except:
                    r_m = np.float_power(10.0,clouds_in[np.where(np.char.find(cloud_param_names, 'log_r_m')!= -1)[0][0]])
                    log_X_Mie = clouds_in[np.where(np.char.find(cloud_param_names, 'log_X')!= -1)[0]][0]

                # If uniform_X, you set the slab and deck parameters to dummy values to pass into the cloud object
                log_n_max = 0
                fractional_scale_height = 0
                P_cloud = 100.0
                P_slab_bottom = 100.0


        # See if its a free or file_read aerosol retrieval or not
        if ('r_i_real' in cloud_param_names):
            r_i_real = clouds_in[np.where(cloud_param_names == 'r_i_real')[0][0]]
            r_i_complex = clouds_in[np.where(cloud_param_names == 'r_i_complex')[0][0]]
        else:
            r_i_real = 0
            r_i_complex = 0

    # This cloud model is to specifically take in kappa_cloud, g_cloud, and w_cloud from eddysed calculations
    # i.e from PICASO, VIRGA
    elif (cloud_model =='eddysed'):

        kappa_cloud_eddysed = clouds_in[np.where(cloud_param_names == 'kappa_cloud_eddysed')[0][0]]
        g_cloud_eddysed = clouds_in[np.where(cloud_param_names == 'g_cloud_eddysed')[0][0]]
        w_cloud_eddysed = clouds_in[np.where(cloud_param_names == 'w_cloud_eddysed')[0][0]]
        
        # Set dummy parameter values, not used when cloud-free
        kappa_cloud_0 = 1.0e250
        P_cloud = 100.0
        P_slab_bottom = 100.0
        a, gamma = 1.0, -4.0  

        if (cloud_dim == 1):
            f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0   # 1D uniform cloud
        elif (cloud_dim == 2):
            try:
                f_cloud = clouds_in[np.where(cloud_param_names == 'f_cloud')[0][0]]    
            except:
                f_cloud = clouds_in[np.where(cloud_param_names == 'f_cloud')[0]] 
            phi_0 = -90
            theta_0 = -90.0   

        # Mie scattering parameters not needed
        r_m = []
        log_n_max = 0
        fractional_scale_height = 0
        r_i_real = 0
        r_i_complex = 0
        log_X_Mie = []
    
    return kappa_cloud_0, P_cloud, f_cloud, phi_0, theta_0, a, gamma, r_m, log_n_max, \
           fractional_scale_height, r_i_real, r_i_complex, log_X_Mie, P_slab_bottom, \
           kappa_cloud_eddysed, g_cloud_eddysed, w_cloud_eddysed


def unpack_geometry_params(param_names, geometry_in, N_params_cumulative):
    '''
    Extract the multidimensional geometry property values (i.e. terminator
    opening angles) from the drawn geometry parameters, according to the model
    dimensionality specified by the user.
    
    Args:
        param_names (np.array of str):
            Free parameters defining this POSEIDON model.
        geometry_in (list of float | np.array of float):
            Drawn values of the multidimensional atmospheric geometry parameters.
        N_params_cumulative (np.array of int):
            Cumulative sum of number of parameters (used for indexing).

    Returns:
        alpha (float):
            Terminator opening angle (degrees).
        beta (float):
            Day-night opening angle (degrees).

    '''
    
    # Unpack names of geometry parameters
    geometry_param_names = param_names[N_params_cumulative[3]:N_params_cumulative[4]]
    
    if ('alpha' in geometry_param_names):
        alpha = geometry_in[np.where(geometry_param_names == 'alpha')[0][0]]
    else:
        alpha = 0.0
        
    if ('beta' in geometry_param_names):
        beta = geometry_in[np.where(geometry_param_names == 'beta')[0][0]]
    else:
        beta = 0.0
    
    return alpha, beta


def unpack_stellar_params(param_names, star, stellar_in, stellar_contam, 
                          N_params_cumulative):
    '''
    Extract the stellar properties from the drawn stellar parameters, according 
    to the stellar contamination model specified by the user.
    
    Args:
        param_names (np.array of str):
            Free parameters defining this POSEIDON model.
        star (dict):
            Collection of stellar properties used by POSEIDON.
        stellar_in (list of float | np.array of float):
            Drawn values of the stellar parameters.
        stellar_contam (str):
            Chosen prescription for modelling unocculted stellar contamination
            (Options: one_spot / one_spot_free_log_g / two_spots / 
             two_spots_free_log_g).
        N_params_cumulative (np.array of int):
            Cumulative sum of number of parameters (used for indexing).

    Returns:
        f_het (float):
            For the 'one_spot' model, the fraction of stellar photosphere 
            covered by either spots or faculae.
        f_spot (float):
            For the 'two_spots' model, the fraction of stellar photosphere 
            covered by spots.
        f_fac (float):
            For the 'two_spots' model, the fraction of stellar photosphere 
            covered by faculae.
        T_het (float):
            For the 'one_spot' model, the temperature of the heterogeneity (K).
        T_spot (float):
            For the 'two_spots' model, the temperature of the spot (K).
        T_fac (float):
            For the 'two_spots' model, the temperature of the facula (K).
        T_phot (float):
            Stellar photosphere temperature (K).
        log_g_het (float):
            For the 'one_spot' model, the log g of the heterogeneity (log10(cm/s^2)).
        log_g_spot (float):
            For the 'two_spots' model, the log g of the spot (log10(cm/s^2)).
        log_g_fac (float):
            For the 'two_spots' model, the log g of the facula (log10(cm/s^2)).
        log_g_phot (float):
            Stellar photosphere log g (log10(cm/s^2)).

    '''
    
    # Unpack names of stellar parameters
    stellar_param_names = param_names[N_params_cumulative[4]:N_params_cumulative[5]]

    # Unpack stellar properties
    T_phot_obs = star['T_eff']
    Met_phot_obs = star['Met']
    log_g_phot_obs = star['log_g']

    # Extract parameters for a single stellar heterogeneity
    if ('one_spot' in stellar_contam):

        f_het = np.array(stellar_in[np.where(stellar_param_names == 'f_het')[0][0]])
        T_het = np.array(stellar_in[np.where(stellar_param_names == 'T_het')[0][0]])
        T_phot = np.array(stellar_in[np.where(stellar_param_names == 'T_phot')[0][0]])

        # Extract log g parameters 
        if ('free_log_g' in stellar_contam):
            log_g_het = np.array(stellar_in[np.where(stellar_param_names == 'log_g_het')[0][0]])
            log_g_phot = np.array(stellar_in[np.where(stellar_param_names == 'log_g_phot')[0][0]])
        else:
            log_g_het = log_g_phot_obs
            log_g_phot = log_g_phot_obs

        # The below parameters are not used for a one heterogeneity model
        f_spot = 0.0
        f_fac = 0.0
        T_spot = T_phot_obs
        T_fac = T_phot_obs
        log_g_spot = log_g_phot_obs
        log_g_fac = log_g_phot_obs

    # Extract parameters for two stellar heterogeneities
    elif ('two_spots' in stellar_contam):

        f_spot = np.array(stellar_in[np.where(stellar_param_names == 'f_spot')[0][0]])
        f_fac = np.array(stellar_in[np.where(stellar_param_names == 'f_fac')[0][0]])
        T_spot = np.array(stellar_in[np.where(stellar_param_names == 'T_spot')[0][0]])
        T_fac = np.array(stellar_in[np.where(stellar_param_names == 'T_fac')[0][0]])
        T_phot = np.array(stellar_in[np.where(stellar_param_names == 'T_phot')[0][0]])

        # Extract log g parameters 
        if ('free_log_g' in stellar_contam):
            log_g_spot = np.array(stellar_in[np.where(stellar_param_names == 'log_g_spot')[0][0]])
            log_g_fac = np.array(stellar_in[np.where(stellar_param_names == 'log_g_fac')[0][0]])
            log_g_phot = np.array(stellar_in[np.where(stellar_param_names == 'log_g_phot')[0][0]])
        else:
            log_g_spot = log_g_phot_obs
            log_g_fac = log_g_phot_obs
            log_g_phot = log_g_phot_obs

        # The below parameters are not used for a two heterogeneity model
        f_het = 0.0
        T_het = T_phot_obs
        log_g_het = log_g_phot_obs

    return f_het, f_spot, f_fac, T_het, T_spot, T_fac, T_phot, log_g_het, \
           log_g_spot, log_g_fac, log_g_phot

