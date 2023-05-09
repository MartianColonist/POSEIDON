''' 
Supported opacity sources in the current version of POSEIDON.

'''


import numpy as np

# Chemical species with cross sections included in opacity database
supported_species = np.array(['AlH', 'AlO', 'BeH', 'C2H2', 'CH', 'CH4', 'CO', 
                              'CO2', 'CaH', 'CaO', 'CrH', 'Cs', 'Fe', 'Fe+', 
                              'FeH', 'H2', 'H2O', 'H2S', 'H3+', 'HCN', 'K', 
                              'Li', 'LiH', 'MgH', 'N2', 'N2O', 'NH', 'NH3', 
                              'NO', 'NO2', 'Na', 'NaH', 'O2', 'O3', 'OH', 
                              'PH3', 'PN', 'PO', 'PS', 'Rb', 'SH', 'SO2', 'ScH', 
                              'SiH', 'SiO', 'Ti', 'Ti+', 'TiH', 'TiO', 'VO',
                              'Mg', 'Mg+', 'Ca', 'Ca+', 'Mn'])

# Chemical species with both a POSEIDON opacity and inclusion in the eq grid
supported_chem_eq_species = np.array(['H2O', 'CO2', 'OH', 'C2H2', 'H2S', 'O2',
                                      'O3', 'HCN', 'NH3', 'SiO', 'CH4', 'CO', 
                                      'CaH', 'CrH', 'FeH', 'K', 'MgH', 'N2', 
                                      'Na', 'NO', 'NO2', 'PH3', 'SH', 'SiH',
                                      'SO2', 'TiH', 'TiO', 'VO'])

# Collision-Induced Absorption (CIA) pairs included in opacity database
supported_cia = np.array(['H2-H2', 'H2-He', 'H2-H', 'H2-CH4', 'CO2-H2', 'N2-H2', 
                          'CO2-CO2', 'CO2-CH4', 'N2-N2', 'N2-H2O', 'O2-O2',
                          'O2-CO2', 'O2-N2'])

# Species treated as spectrally inactive (in standard cross section treatment)
inactive_species = np.array(['H2', 'He', 'H', 'e-', 'H-'])  # H- handled separately

supported_aerosols = np.array(['SiO2', 'Al2O3', 'CaTiO3', 'CH4', 'Fe2O3', 'Fe2SiO4',
                          'H2O','Hexene','Hibonite','KCl','Mg2SiO4',
                          'Mg2SiO4poor','MgAl2O4','MgSiO3','MnS',
                          'Na2S','NaCl','SiO2','Tholin','TiO2','ZnS',
                          'SiO2_amorph','C','Cr','Fe', 'FeS', 'Mg2SiO4_amorph_sol-gel',
                          'Mg04Fe06SiO3_amorph_glass','Mg05Fe05SiO3_amorph_glass',
                          'Mg08Fe02SiO3_amorph_glass','Mg08Fe12SiO4_amorph_glass',
                          'MgFeSiO4_amorph_glass','MgO','MgSiO3_amorph_glass',
                          'MgSiO3_amorph_sol-gel_complex','SiC','SiO','TiC','TiO2_anatase'])