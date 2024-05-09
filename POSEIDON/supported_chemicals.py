''' 
Supported chemical species in the current version of POSEIDON.

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
                              'Mg', 'Mg+', 'Ca', 'Ca+', 'Mn' 'Cr', 'V', 'V+',
                              'Ba', 'Ba+', 'Al', 'Ni', 'O', 'Sc', 'H'])

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
inactive_species = np.array(['H2', 'He', 'e-', 'H-'])  # H- handled separately

# Aerosol Supported Species#
aerosol_supported_species = np.array(['Al2O3', 'Al2O3_KH', 'C', 'CH4', 'CaTiO3', 'CaTiO3_KH', 'Cr', 'ExoHaze_1000xSolar_300K', 
                                      'ExoHaze_1000xSolar_400K','Fe', 'Fe2O3', 'Fe2SiO4', 'Fe2SiO4_KH', 'FeO', 'FeS',
                                      'FeSiO3', 'H2O', 'Hexene', 'Hibonite', 'IceTholin', 'KCl', 'Mg2SiO4_amorph_sol_gel', 'Mg2SiO4_Fe_poor', 'Mg2SiO4_Fe_rich', 'Mg2SiO4_crystalline',
                                      'Mg4Fe6SiO3_amorph_glass', 'Mg5Fe5SiO3_amorph_glass', 'Mg8Fe12SiO4_amorph_glass',
                                      'Mg8Fe2SiO3_amorph_glass', 'MgAl2O4', 'MgFeSiO4_amorph_glass', 'MgO',
                                      'MgSiO3', 'MgSiO3_amorph_glass', 'MgSiO3_sol_gel', 'MnS', 'MnS_KH', 'MnS_Mor', 'Na2S', 'NaCl', 'NH3', 'S8',
                                      'Saturn-Phosphorus-Haze', 'SiC', 'SiO', 'SiO2', 'SiO2_amorph', 'soot', 'Soot_6mm', 'Tholin', 'Tholin-CO-0625', 'Tholin-CO-1',
                                      'TiC', 'TiO2', 'TiO2_anatase', 'TiO2_rutile', 'VO', 'ZnS', 'Ethane'])