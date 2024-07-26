''' 
Dictionaries with various properties of atoms and molecules used by POSEIDON.

'''


# List of masses in atomic mass units (multiply by 1u = 1.66053904e-27 to convert to kg)
masses = {'H2O':   18.010565, 'CH4':   16.031300, 'NH3':   17.026549, 'HCN':   27.010899,
          'CO':    27.994915, 'CO2':   43.989830, 'O2':    31.989830, 'O3':    47.984745, 
          'N2':    28.006148, 'PH3':   33.997238, 'SO2':   63.961901, 'SO3':   79.956820,
          'C2H2':  26.015650, 'C2H4':  28.031300, 'H2S':   33.987721, 'NO':    29.997989,
          'NO2':   45.992904, 'N2O':   44.001062, 'H2O2':  34.005479, 'HNO3':  62.995644,
          'Li':    7.0160040, 'Na':    22.989770, 'K':     38.963707, 'Rb':    84.911789,
          'Cs':    132.90545, 'O':     15.994915, 'Fe':    55.934942, 'H3+':   3.0234750,
          'TiO':   63.942861, 'VO':    66.938871, 'AlO':   42.976454, 'SiO':   43.971842,
          'CaO':   55.957506, 'TiH':   48.955771, 'FeH':   56.942762, 'LiH':   8.0238300,
          'ScH':   45.963737, 'MgH':   24.992867, 'NaH':   23.997594, 'AlH':   27.989480,
          'CrH':   52.948333, 'CaH':   40.970416, 'BeH':   10.020007, 'SiH':   28.984752,
          'NH':    15.010899, 'CH':    13.007825, 'OH':    17.002740, 'SH':    32.979896,
          'NaCl':  57.958622, 'KCl':   73.932560, 'HCl':   35.976678, 'NS':    45.975145,
          'PS':    62.945833, 'CS':    43.972071, 'PO':    46.968676, 'CP':    42.973762,
          'CN':    26.003074, 'PN':    44.976836, 'H2CO':  30.010565, 'C2':    21.000000,  
          'CH3F':  34.021878, 'SiH4':  32.008227, 'HF':    20.006229, 'HBr':   79.926160,
          'HI':    127.91230, 'ClO':   50.963768, 'C2H6':  30.046950, 'OCS':   59.966986,
          'SF6':   145.96249, 'HO2':   32.997655, 'HOCl':  51.971593, 'CH3Cl': 49.992328,
          'NO+':   29.997989, 'CF4':   87.993616, 'CH4O':  32.026215, 'COF2':  65.991722,
          'HOBr':  95.921076, 'C4H2':  50.015650, 'CH3Br': 93.941811, 'H2CO2': 46.005480,
          'HC3N':  51.010899, 'C2N2':  52.006148, 'C2H3N': 41.026549, 'ClO3N': 96.956672,  
          'COCl2': 97.932620, 'H2':    2.0156500, 'He':    4.0026030, 'Fe+':   55.934942,
          'Ti':    47.947946, 'Ti+':   47.947946, 'H':     1.0078250, 'H-':    1.0083740,  
          'e-':    5.4858e-4, 'Mg':    23.985042, 'Mg+':   23.985042, 'Mn':    54.938050,
          'Ca':    39.962591, 'Ca+':   39.962591, 'CH3':   15.023475, 'CS2':   75.944142,
          'C2H4':  28.031300, 'C2H6':  30.046950, 'CH3CN': 41.026549, 'CH3OH': 32.026215,
          'GeH4':  77.952478}

# Polarisabilities (cgs units, cm^3) used for computing refractive index and Rayleigh scattering - Mostly from CRC handbook
polarisabilities = {'H2':   0.80e-24, 'He':   0.21e-24, 'N2':    1.74e-24, 'O2':    1.58e-24, 
                    'O3':   3.21e-24, 'H2O':  1.45e-24, 'CH4':   2.59e-24, 'CO':    1.95e-24,
                    'CO2':  2.91e-24, 'NH3':  2.26e-24, 'HCN':   2.59e-24, 'PH3':   4.84e-24, 
                    'SO2':  3.72e-24, 'SO3':  4.84e-24, 'C2H2':  3.33e-24, 'H2S':   3.78e-24,
                    'NO':   1.70e-24, 'NO2':  3.02e-24, 'H3+':   0.39e-24, 'OH':    6.97e-24,  # H3+from Kawaoka & Borkman, 1971
                    'Na':   24.1e-24, 'K':    42.9e-24, 'Li':    24.3e-24, 'Rb':    47.4e-24,     
                    'Cs':   59.4e-24, 'TiO':  16.9e-24, 'VO':    14.4e-24, 'AlO':   8.22e-24,   # Without tabulated values for metal
                    'SiO':  5.53e-24, 'CaO':  23.8e-24, 'TiH':   16.9e-24, 'MgH':   10.5e-24,   # oxides and hydrides, these are taken
                    'NaH':  24.1e-24, 'AlH':  8.22e-24, 'CrH':   11.6e-24, 'FeH':   9.47e-24,   # to be metal atom polarisabilities
                    'CaH':  23.8e-24, 'BeH':  5.60e-24, 'ScH':   21.2e-24, 'LiH':   24.3e-24,
                    'SiH':  5.53e-24, 'CH':   2.59e-24, 'NH':    2.59e-24, 'SH':    2.59e-24,   # No literature for CH,NH,SH, so ~ to CH4
                    'PN':   3.69e-24, 'PO':   3.69e-24, 'PS':    3.69e-24, 'Fe':    9.47e-24,   # No values, so ~ polarisability of P
                    'Fe+':  9.47e-24, 'Ti':   14.8e-24, 'Ti+':   14.8e-24, 'H':     0.67e-24,
                    'H-':   30.5e-24, 'e-':   0.00e-24, 'Mg':    10.6e-24, 'Mg+':   10.6e-24,
                    'Mn':   9.40e-24, 'Ca':   22.8e-24, 'Ca+':   22.8e-24, 'CH3':   2.59e-24,   # No literature value for CH3, so ~ CH4
                    'CS2':  8.74e-24, 'C2H4': 4.25e-24, 'C2H6':  4.47e-24, 'CH3CN': 4.40e-24, 
                    'CH3OH': 3.29e-24, 'GeH4': 5.84e-24}                                        # No data for GeH4, so ~ Ge

# List of masses in atomic mass units (multiply by 1u = 1.66053904e-27 to convert to kg)
# Exohazes, Tholins, IceTholins, Phosphorus-Haze, Soot have the weight of one PAN (CH2CHCN)
# Not used in the code at the moment, just here as a reference for future code updates
aerosol_masses = {'ADP': 97.02, 'Al2O3' : 101.96, 'Al2O3_KH' : 101.96, 'C' : 12.011, 'CH4' : 16.031300, 'CaTiO3' : 135.94, 'CaTiO3_KH' : 135.94, 
                  'Cr' : 52, 'ExoHaze_1000xSolar_300K' : 53.06, 'ExoHaze_1000xSolar_400K' : 53.06, 'Fe' : 55.934942, 
                  'Fe2O3' : 159.69, 'Fe2SiO4_KH' : 203.77, 'FeO' : 71.84, 'FeS' : 87.91,
                  'FeSiO3' : 131.93, 'H2O' : 18.010565, 'H2SO4' : 98.08, 'Hexene' : 84.16, 'Hibonite': 668, 
                  'IceTholin' : 53.06, 'KCl': 74.55,
                  'Mg2SiO4_Fe_rich' : 140.69, 'Mg2SiO4_amorph_sol_gel' : 140.69, 'Mg2SiO4_Fe_poor' : 140.69,
                  'Mg2SiO4_amorph' : 140.69, 'Mg2SiO4_crystalline' : 140.69, 
                  'Mg4Fe6SiO3_amorph_glass' : 508.37, 'Mg5Fe5SiO3_amorph_glass' : 476.83, 
                  'Mg8Fe12SiO4_amorph_glass' : 956.66, 'Mg8Fe2SiO3_amorph_glass' : 382.21, 'MgAl2O4' : 142.27,
                  'MgFeSiO4_amorph_glass' : 172.23, 'MgO' : 40.3, 'MgSiO3' : 100.39, 'MgSiO3_amorph_glass' : 100.39, 
                  'MgSiO3_amorph' : 100.39, 'MgSiO3_crystalline' : 100.39, 
                  'MgSiO3_sol_gel' : 100.39, 'MnS' : 87, 'MnS_KH' : 87, 'MnS_Mor' : 87, 'Na2S' : 78.04, 'NaCl' : 58.44, 
                  'NanoDiamonds' : 12.01, 'NH4SH' : 42, 'S8' : 256, 'Saturn-Phosphorus-Haze' : 53.06,
                  'NH3' : 17.026549, 'SiC' : 40.10, 'SiO' : 44.08, 'SiO2': 60.08, 'SiO2_crystalline_2023': 60.08,
                  'SiO2_amorph' : 60.08, 'Soot' : 53.06, 'Soot_6mm' : 53.06,
                  'Tholin' : 53.06, 'Tholin-CO-0625' : 53.06, 'TiC' : 59.88, 
                  'TiO2_anatase' : 79.87, 'TiO2_rutile' : 79.87,'VO' : 67,'ZnS' : 97.46,}

