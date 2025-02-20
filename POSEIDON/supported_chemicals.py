''' 
Supported chemical species in the current version of POSEIDON.

'''

import numpy as np

# Chemical species with cross sections included in opacity database
supported_species = np.array(['H2O', 'CO2', 'CH4', 'CO', 'Na', 'K', 'NH3', 'HCN',
                              'SO2', 'H2S', 'PH3', 'C2H2', 'OCS', 'TiO', 'VO',
                              'AlO', 'SiO', 'CaO', 'MgO', 'NaO', 'LaO', 'ZrO', 
                              'SO', 'NO', 'PO', 'TiH', 'CrH', 'FeH', 'ScH',
                              'AlH', 'SiH', 'BeH', 'CaH', 'MgH', 'LiH', 'NaH',
                              'OH', 'OH+', 'CH', 'NH', 'SH', 'PN', 'PS', 'CS',
                              'C2', 'CH3', 'H3+', 'N2O', 'NO2', 'C2H4', 'C2H6', 
                              'CH3CN', 'CH3OH', 'CH3Cl', 'GeH4', 'CS2', 'O2', 
                              'O3', 'C2H6S', 'Al', 'Ba', 'Ba+', 'Ca', 'Ca+', 
                              'Cr', 'Cs', 'Fe', 'Fe+', 'Li', 'Mg', 'Mg+', 'Mn', 
                              'Ni', 'O', 'Rb', 'Sc', 'Ti', 'Ti+', 'V', 'V+',
                              '12C-16O', '13C-16O', '12C-18O', '12C-17O', 
                              '13C-18O', '13C-17O'])

# Chemical species supported in the fastchem equilibrium grid
fastchem_supported_species = np.array(['H2O', 'CO2', 'OH', 'SO', 'C2H2', 
                                       'C2H4', 'H2S', 'O2', 'O3', 'HCN',
                                       'NH3', 'SiO', 'CH4', 'CO', 'C2', 
                                       'CaH', 'CrH', 'FeH', 'HCl', 'K',
                                       'MgH', 'N2', 'Na', 'NO', 'NO2',
                                       'OCS', 'PH3', 'SH', 'SiH', 'SO2',
                                       'TiH', 'TiO', 'VO'])

# Collision-Induced Absorption (CIA) pairs included in opacity database
supported_cia = np.array(['H2-H2', 'H2-He', 'H2-H', 'H2-CH4', 'CO2-H2', 'N2-H2', 
                          'CO2-CO2', 'CO2-CH4', 'N2-N2', 'N2-H2O', 'O2-O2',
                          'O2-CO2', 'O2-N2'])

# Species treated as spectrally inactive (in standard cross section treatment)
inactive_species = np.array(['H2', 'He', 'e-', 'H-', 'N2', 'ghost'])  # H- handled separately

# Aerosol Supported Species
aerosol_supported_species = np.array(['ADP', 'Al2O3', 'Al2O3_KH', 'C', 'CH4_liquid', 'CH4_solid', 'CaTiO3', 'CaTiO3_KH', 'Cr', 'ExoHaze_1000xSolar_300K', 
                                      'ExoHaze_1000xSolar_400K','Fe', 'Fe2O3', 'Fe2SiO4_KH', 'FeO', 'FeS',
                                      'FeSiO3', 'H2O', 'H2O_ice', 'H2SO4', 'Hexene', 'Hibonite', 'IceTholin', 'KCl', 'Mg2SiO4_amorph_sol_gel', 'Mg2SiO4_amorph',
                                      'Mg2SiO4_Fe_poor', 'Mg2SiO4_Fe_rich', 'Mg2SiO4_crystalline',
                                      'Mg4Fe6SiO3_amorph_glass', 'Mg5Fe5SiO3_amorph_glass', 'Mg8Fe12SiO4_amorph_glass',
                                      'Mg8Fe2SiO3_amorph_glass', 'MgAl2O4', 'MgFeSiO4_amorph_glass', 'MgO', 'MgSiO3_amorph', 'MgSiO3_crystalline',
                                      'MgSiO3', 'MgSiO3_amorph_glass', 'MgSiO3_sol_gel', 'MnS', 'MnS_KH', 'MnS_Mor', 'Na2S', 'NaCl', 'NanoDiamonds', 'NH3', 'NH4SH', 'S8',
                                      'Saturn-Phosphorus-Haze', 'SiC', 'SiO', 'SiO2', 'SiO2_amorph','SiO2_crystalline_2023', 'SiO2_alpha_palik', 'SiO2_glass_palik', 'Soot', 'Soot_6mm', 'Tholin', 'Tholin-CO-0625', 'Tholin-CO-1',
                                      'TiC', 'TiO2_anatase', 'TiO2_rutile', 'VO', 'ZnS',
                                      'MgSiO3_r_m_std_dev_01','MgSiO3_r_m_std_dev_1','MgSiO3_g_w_calc_mean','MgSiO3_g_w_calc_trap'])
