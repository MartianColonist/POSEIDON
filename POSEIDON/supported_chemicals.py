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
                              'O3', 'C2H6S', 'C2H6S2', 'CH3SH', 'C3H4', 
                              'Al', 'Ba', 'Ba+', 'Ca', 'Ca+', 
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

# Aerosol Supported Species for normal aerosol database
aerosol_supported_species = np.array(['ADP', 'Al2O3', 'Al2O3_KH', 'C', 'CH4_liquid', 'CH4_solid', 'CaTiO3', 'CaTiO3_KH', 'Cr', 'ExoHaze_1000xSolar_300K', 
                                      'ExoHaze_1000xSolar_400K','Fe', 'Fe2O3', 'Fe2SiO4_KH', 'FeO', 'FeS',
                                      'FeSiO3', 'H2O', 'H2O_ice', 'H2SO4', 'Hexene', 'Hibonite', 'IceTholin', 'KCl', 'Mg2SiO4_amorph_sol_gel', 'Mg2SiO4_amorph',
                                      'Mg2SiO4_Fe_poor', 'Mg2SiO4_Fe_rich', 'Mg2SiO4_crystalline',
                                      'Mg4Fe6SiO3_amorph_glass', 'Mg5Fe5SiO3_amorph_glass', 'Mg8Fe12SiO4_amorph_glass',
                                      'Mg8Fe2SiO3_amorph_glass', 'MgAl2O4', 'MgFeSiO4_amorph_glass', 'MgO', 'MgSiO3_amorph', 'MgSiO3_crystalline',
                                      'MgSiO3', 'MgSiO3_amorph_glass', 'MgSiO3_sol_gel', 'MnS', 'MnS_KH', 'MnS_Mor', 'Na2S', 'NaCl', 'NanoDiamonds', 'NH3', 'NH4SH', 'S8',
                                      'Saturn-Phosphorus-Haze', 'SiC', 'SiO', 'SiO2', 'SiO2_amorph','SiO2_crystalline_2023', 'SiO2_alpha_palik', 'SiO2_glass_palik', 'Soot', 'Soot_6mm', 'Tholin', 'Tholin-CO-0625', 'Tholin-CO-1',
                                      'TiC', 'TiO2_anatase', 'TiO2_rutile', 'VO', 'ZnS',])

# Directional and temperature specific aerosols 
aerosol_directional_supported_species=  np.array(['SiO2_alpha_crystal_A2_295K', 'SiO2_alpha_crystal_E_295K', 'SiO2_alpha_crystal_E_346K', 'SiO2_alpha_crystal_E_480K', 'SiO2_alpha_crystal_E_600K', 'SiO2_alpha_crystal_E_705K', 'SiO2_alpha_crystal_E_790K',
                                    'SiO2_beta_crystal_E_1010K', 'SiO2_beta_crystal_E_1125K', 'SiO2_beta_crystal_E_1170K', 'SiO2_beta_crystal_E_1310K', 'SiO2_beta_crystal_E_1394K', 'SiO2_beta_crystal_E_1520K', 'SiO2_beta_crystal_E_1590K',
                                    'SiO2_beta_crystal_E_1646K',
                                    'SiO2_beta_cristobalite_E_1810K','SiO2_beta_cristobalite_E_1880K',
                                    'SiO2_alpha_crystal_300K_extraordinary','SiO2_alpha_crystal_300K_ordinary',
                                    'SiO2_alpha_crystal_551K_extraordinary','SiO2_alpha_crystal_551K_ordinary',
                                    'SiO2_alpha_crystal_738K_extraordinary','SiO2_alpha_crystal_738K_ordinary',
                                    'SiO2_alpha_crystal_833K_extraordinary','SiO2_alpha_crystal_833K_ordinary',
                                    'SiO2_beta_crystal_928K_extraordinary','SiO2_beta_crystal_928K_ordinary',
                                    'SiO2_beta_tridymite_295K','SiO2_beta_tridymite_500K',
                                    'Mg2SiO4_295K_B1U', 'Mg2SiO4_546K_B1U', 'Mg2SiO4_950K_B1U', 'Mg2SiO4_1102K_B1U', 'Mg2SiO4_1147K_B1U', 'Mg2SiO4_1431K_B1U', 'Mg2SiO4_1518K_B1U', 'Mg2SiO4_1648K_B1U', 'Mg2SiO4_1742K_B1U', 'Mg2SiO4_1809K_B1U',
                                    'Mg2SiO4_295K_B2U',  'Mg2SiO4_547K_B2U', 'Mg2SiO4_720K_B2U', 'Mg2SiO4_946K_B2U', 'Mg2SiO4_1122K_B2U', 'Mg2SiO4_1303K_B2U', 'Mg2SiO4_1417K_B2U', 'Mg2SiO4_1535K_B2U', 'Mg2SiO4_1617K_B2U', 'Mg2SiO4_1818K_B2U',
                                    'Mg2SiO4_295K_B3U', 'Mg2SiO4_602K_B3U', 'Mg2SiO4_757K_B3U', 'Mg2SiO4_918K_B3U', 'Mg2SiO4_1055K_B3U', 'Mg2SiO4_1131K_B3U', 'Mg2SiO4_1256K_B3U',  'Mg2SiO4_1503K_B3U', 'Mg2SiO4_1793K_B3U', 'Mg2SiO4_1948K_B3U',
                                    'Al2O3_alpha_crystal_300K_extraordinary', 'Al2O3_alpha_crystal_551K_extraordinary', 'Al2O3_alpha_crystal_738K_extraordinary', 'Al2O3_alpha_crystal_928K_extraordinary',
                                    'Al2O3_alpha_crystal_300K_ordinary', 'Al2O3_alpha_crystal_551K_ordinary', 'Al2O3_alpha_crystal_738K_ordinary', 'Al2O3_alpha_crystal_928K_ordinary',
                                    'Mg172Fe021SiO4_crystal_10K_Ex', 'Mg172Fe021SiO4_crystal_100K_Ex' , 'Mg172Fe021SiO4_crystal_200K_Ex', 'Mg172Fe021SiO4_crystal_300K_Ex', 'Mg172Fe021SiO4_crystal_551K_Ex','Mg172Fe021SiO4_crystal_738K_Ex','Mg172Fe021SiO4_crystal_928K_Ex',
                                    'Mg172Fe021SiO4_crystal_10K_Ey', 'Mg172Fe021SiO4_crystal_100K_Ey' , 'Mg172Fe021SiO4_crystal_200K_Ey', 'Mg172Fe021SiO4_crystal_300K_Ey', 'Mg172Fe021SiO4_crystal_551K_Ey','Mg172Fe021SiO4_crystal_738K_Ey','Mg172Fe021SiO4_crystal_928K_Ey',
                                    'Mg172Fe021SiO4_crystal_10K_Ez', 'Mg172Fe021SiO4_crystal_100K_Ez' , 'Mg172Fe021SiO4_crystal_200K_Ez', 'Mg172Fe021SiO4_crystal_300K_Ez', 'Mg172Fe021SiO4_crystal_551K_Ez','Mg172Fe021SiO4_crystal_738K_Ez','Mg172Fe021SiO4_crystal_928K_Ez',
                                    'Mg172Fe021SiO4_crystal_visnir_Ex','Mg172Fe021SiO4_crystal_visnir_Ey','Mg172Fe021SiO4_crystal_visnir_Ez',
                                    'MgAl2O4_crystalline_natural_annealed_1223K','MgAl2O4_crystalline_natural',
                                    'Mg092Fe009SiO3_crystal_10K_Ex','Mg092Fe009SiO3_crystal_100K_Ex','Mg092Fe009SiO3_crystal_200K_Ex','Mg092Fe009SiO3_crystal_300K_Ex','Mg092Fe009SiO3_crystal_551K_Ex','Mg092Fe009SiO3_crystal_738K_Ex','Mg092Fe009SiO3_crystal_928K_Ex',
                                    'Mg092Fe009SiO3_crystal_10K_Ey','Mg092Fe009SiO3_crystal_100K_Ey','Mg092Fe009SiO3_crystal_200K_Ey','Mg092Fe009SiO3_crystal_300K_Ey','Mg092Fe009SiO3_crystal_551K_Ey','Mg092Fe009SiO3_crystal_738K_Ey','Mg092Fe009SiO3_crystal_928K_Ey',
                                    'Mg092Fe009SiO3_crystal_10K_Ez','Mg092Fe009SiO3_crystal_100K_Ez','Mg092Fe009SiO3_crystal_200K_Ez','Mg092Fe009SiO3_crystal_300K_Ez','Mg092Fe009SiO3_crystal_551K_Ez','Mg092Fe009SiO3_crystal_738K_Ez','Mg092Fe009SiO3_crystal_928K_Ez',
                                    'MgAl2O4_synthetic_10K','MgAl2O4_synthetic_100K','MgAl2O4_synthetic_300K','MgAl2O4_synthetic_551K','MgAl2O4_synthetic_738K','MgAl2O4_synthetic_928K',
                                    'TiO2_anatase_extraordinary', 'TiO2_anatase_ordinary',
                                    'TiO2_brookite_Ex','TiO2_brookite_Ey','TiO2_brookite_Ez',
                                    'TiO2_rutile_extraordinary','TiO2_rutile_ordinary',
                                    'Mg19Fe01SiO4_crystal_natural_Ex','Mg19Fe01SiO4_crystal_natural_Ey','Mg19Fe01SiO4_crystal_natural_Ez',
                                    'Al2O3_amorph_compact','Al2O3_amorph_porous',
                                    'Fe2SiO4_crystal_synthetic_Ex','Fe2SiO4_crystal_synthetic_Ey','Fe2SiO4_crystal_synthetic_Ez',
                                    'CaAl12O19_crystal_natural_extraordinary','CaAl12O19_crystal_natural_ordinary',
                                    'SiO2_alpha_crystal_300K_averaged','SiO2_beta_crystal_928K_averaged',
                                    'Mg2SiO4_295K_averaged','Mg2SiO4_1000K_averaged',
                                    'Mg092Fe009SiO3_crystal_300K_averaged','Mg092Fe009SiO3_crystal_928K_averaged',
                                    'SiO2_alpha_cristobalite_295K',
                                    ])

# Diamonds added to POSEIDON 
diamond_supported_species = np.array(['NanoDiamonds','Diamond_palik',
                                    'Hydrogentated_Diamond_fH0_N0_irradiated',
                                    'Hydrogentated_Diamond_fH0_N0_not_irradiated',
                                    'Hydrogentated_Diamond_fH025_N0_irradiated',
                                    'Hydrogentated_Diamond_fH025_N0_not_irradiated',
                                    'Hydrogentated_Diamond_fH1_N0_irradiated',
                                    'Hydrogentated_Diamond_fH1_N0_not_irradiated'])

# Aerosols with precomputed grids that have free lognormal logwidths
aerosols_lognormal_logwidth_free = np.array(['SiO2'])
