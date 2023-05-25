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

# Aerosol Supported Species
aerosol_supported_species = np.array(['H2O','KCl','MgO','Tholin','Fe','Mg2SiO4_amorph_sol_gel',
                                      'MgSiO3','Ch4','Fe2O3','SiO2_amorph','SiO2', 'Na2S', 'NaCl'])

