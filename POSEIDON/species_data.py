"""  
Dictionaries with various properties of atoms and molecules used by POSEIDON.

"""

import numpy as np

# List of masses in atomic mass units (multiply by 1u = 1.66053904e-27 to convert to kg)
masses = {'H2O':   18.010565, 'CO2':   43.989830, 'CH4':    16.031300, 
          'CO':    28.010100, 'Na':    22.989770, 'K':      39.098300, 
          'NH3':   17.026549, 'HCN':   27.010899, 'SO2':    63.961901, 
          'H2S':   33.987721, 'PH3':   33.997238, 'C2H2':   26.015650, 
          'OCS':   59.966986, 'TiO':   63.942861, 'VO':     66.938871, 
          'AlO':   42.976454, 'SiO':   43.971842, 'CaO':    55.957506, 
          'MgO':   39.979956, 'NaO':   38.989200, 'LaO':    154.90490, 
          'ZrO':   107.22300, 'SO':    48.064000, 'NO':     29.997989,
          'PO':    46.968676, 'TiH':   48.955771, 'CrH':    52.948333, 
          'FeH':   56.942762, 'ScH':   45.963737, 'AlH':    27.989480, 
          'SiH':   28.984752, 'BeH':   10.020007, 'CaH':    40.970416, 
          'MgH':   24.992867, 'LiH':   8.0238300, 'NaH':    23.997594, 
          'OH':    17.002740, 'OH+':   17.002740, 'CH':     13.007825, 
          'NH':    15.010899, 'SH':    32.979896, 'PN':     44.976836, 
          'PS':    62.945833, 'CS':    43.972071, 'C2':     24.000000, 
          'CH3':   15.023475, 'H3+':   3.0234750, 'N2O':    44.001062,
          'NO2':   45.992904, 'C2H4':  28.031300, 'C2H6':   30.046950, 
          'CH3CN': 41.026549, 'CH3OH': 32.026215, 'CH3Cl':  49.992328,
          'GeH4':  77.952478, 'CS2':   75.944142, 'O2':     31.989830,
          'O3':    47.984745, 'C2H6S': 62.019000, 'C2H6S2': 93.991093,
          'CH3SH': 48.003371, 'C3H4':  40.031300, 
          'Al':    26.981539, 'Ba':    137.32770, 'Ba+':    137.32770, 
          'Ca':    40.078400, 'Ca+':   40.078400, 'Cr':     51.996160, 
          'Cs':    132.90545, 'Fe':    55.845200, 'Fe+':    55.845200, 
          'Li':    6.9675000, 'Mg':    24.305500, 'Mg+':    24.305500,
          'Mn':    54.938044, 'Ni':    58.693440, 'O':      15.999400, 
          'Rb':    85.467830, 'Sc':    44.955908, 'Ti':     47.867100,
          'Ti+':   47.867100, 'V':     50.941510, 'V+':     50.941510,
          'H2':    2.0156500, 'He':    4.0026030, 'H':      1.0078250,
          'N2':    28.006148, 'H-':    1.0083740, 'e-':     5.4858e-4, 
          '12C-16O': 27.994915, '13C-16O': 28.998270, '12C-18O': 29.999161,
          '12C-17O': 28.999130, '13C-18O': 31.002516, '13C-17O': 30.002485,
          }

# Polarisabilities (cgs units, cm^3) used for computing refractive index and Rayleigh scattering - Mostly from CRC handbook
polarisabilities = {'H2O':   1.45e-24, 'CO2':   2.91e-24, 'CH4':    2.59e-24, 
                    'CO':    1.95e-24, 'Na':    24.1e-24, 'K':      42.9e-24, 
                    'NH3':   2.26e-24, 'HCN':   2.59e-24, 'SO2':    3.72e-24, 
                    'H2S':   3.78e-24, 'PH3':   4.84e-24, 'C2H2':   3.33e-24,
                    'OCS':   5.71e-24, 'TiO':   16.9e-24, 'VO':     14.4e-24, 
                    'AlO':   8.22e-24, 'SiO':   5.53e-24, 'CaO':    23.8e-24,      # Without tabulated values for metal oxides and
                    'MgO':   10.6e-24, 'NaO':   24.1e-24, 'LaO':    31.1e-24,      # hydrides, take as ~ metal atom polarisabilities
                    'ZrO':   17.9e-24, 'SO':    3.72e-24, 'NO':     1.70e-24,      # Estimate SO as ~ SO2
                    'PO':    3.69e-24, 'TiH':   16.9e-24, 'CrH':    11.6e-24, 
                    'FeH':   9.47e-24, 'ScH':   21.2e-24, 'AlH':    8.22e-24, 
                    'SiH':   5.53e-24, 'BeH':   5.60e-24, 'CaH':    23.8e-24, 
                    'MgH':   10.5e-24, 'LiH':   24.3e-24, 'NaH':    24.1e-24, 
                    'OH':    6.97e-24, 'OH+':   6.97e-24, 'CH':     2.59e-24, 
                    'NH':    2.59e-24, 'SH':    2.59e-24, 'PN':     3.69e-24,      # No literature for CH, NH, SH, so ~ to CH4
                    'PS':    3.69e-24, 'CS':    8.74e-24, 'C2':     1.67e-24,      # PN, PS ~ P | CS ~ CS2
                    'CH3':   2.59e-24, 'H3+':   0.39e-24, 'N2O':    3.03e-24,      # C2 ~ C | CH3 ~ CH4 | H3+ from Kawaoka & Borkman (1971)
                    'NO2':   3.02e-24, 'C2H4':  4.25e-24, 'C2H6':   4.47e-24, 
                    'CH3CN': 4.40e-24, 'CH3OH': 3.29e-24, 'CH3Cl':  5.35e-24,
                    'GeH4':  5.84e-24, 'CS2':   8.74e-24, 'O2':     1.58e-24,      # GeH4 ~ Ge
                    'O3':    3.21e-24, 'C2H6S': 7.41e-24, 'C2H6S2': 7.41e-24,
                    'CH3SH': 7.41e-24, 'C3H4':  6.18e-24,                          # CH3SH ~ C2H6S
                    'Al':    6.80e-24, 'Ba':    39.7e-24, 'Ba+':    39.7e-24, 
                    'Ca':    22.8e-24, 'Ca+':   22.8e-24, 'Cr':     11.6e-24, 
                    'Cs':    59.4e-24, 'Fe':    9.47e-24, 'Fe+':    9.47e-24, 
                    'Li':    24.3e-24, 'Mg':    10.6e-24, 'Mg+':    10.6e-24,
                    'Mn':    9.40e-24, 'Ni':    6.80e-24, 'O':      0.80e-24,
                    'Rb':    47.4e-24, 'Sc':    17.8e-24, 'Ti':     14.8e-24,
                    'Ti+':   14.8e-24, 'V':     12.4e-24, 'V+':     12.4e-24,
                    'H2':    0.80e-24, 'He':    0.21e-24, 'H':      0.67e-24,
                    'N2':    1.74e-24, 'H-':    30.5e-24, 'e-':     0.00e-24,
                    '12C-16O': 1.95e-24, '13C-16O': 1.95e-24, '12C-18O': 1.95e-24,
                    '12C-17O': 1.95e-24, '13C-18O': 1.95e-24, '13C-17O': 1.95e-24,
                    } 

# Solar elemental reference values (Asplund et al. 2021)
# The solar abundance ratio relative to H is given by 10**(value-12)
solar_abundances = {'H': 12.0, 'He': 10.914, 'Li': 0.96, 'Be': 1.38, 'B': 2.70,
                    'C': 8.46, 'N': 7.83, 'O': 8.69, 'F': 4.40, 'Ne': 8.06,
                    'Na': 6.22, 'Mg': 7.55, 'Al': 6.43, 'Si': 7.51, 'P': 5.41,
                    'S': 7.12, 'Cl': 5.31, 'Ar': 6.38, 'K': 5.07, 'Ca': 6.30,
                    'Sc': 3.14, 'Ti': 4.97, 'V': 3.90, 'Cr': 5.62, 'Mn': 5.42,
                    'Fe': 7.46, 'Co': 4.94, 'Ni': 6.20, 'Cu': 4.18, 'Zn': 4.56,
                    'Ga': 3.02, 'Ge': 3.62, 'As': 2.30, 'Se': 3.34, 'Br': 2.54,
                    'Kr': 3.12, 'Rb': 2.32, 'Sr': 2.83, 'Y': 2.21, 'Zr': 2.59, 
                    'Nb': 1.47, 'Mo': 1.88, 'Ru': 1.75, 'Rh': 0.78, 'Pd': 1.57,
                    'Ag': 0.96, 'Cd': 1.71, 'In': 0.80, 'Sn': 2.02, 'Sb': 1.01,
                    'Te': 2.18, 'I': 1.55, 'Xe': 2.22, 'Cs': 1.08, 'Ba': 2.27,
                    'La': 1.11, 'Ce': 1.58, 'Pr': 0.75, 'Nd': 1.42, 'Sm': 0.95,
                    'Eu': 0.52, 'Gd': 1.08, 'Tb': 0.31, 'Dy': 1.10, 'Ho': 0.48,
                    'Er': 0.93, 'Tm': 0.11, 'Yb': 0.85, 'Lu': 0.10, 'Hf': 0.85,
                    'Ta': -0.15, 'W': 0.79, 'Re': 0.26, 'Os': 1.35, 'Ir': 1.32,
                    'Pt': 1.61, 'Au': 0.91, 'Hg': 1.17, 'Tl': 0.92, 'Pb': 1.95,
                    'Bi': 0.65, 'Th': 0.03, 'U': -0.54,
                    }

# List of masses in atomic mass units (multiply by 1u = 1.66053904e-27 to convert to kg)
# Exohazes, Tholins, IceTholins, Phosphorus-Haze, Soot have the weight of one PAN (CH2CHCN)
# Not used in the code at the moment, just here as a reference for future code updates
aerosol_masses = {'ADP': 97.02, 'Al2O3': 101.96, 'Al2O3_KH': 101.96, 
                  'C': 12.011, 'CH4': 16.031300, 'CaTiO3': 135.94, 
                  'CaTiO3_KH': 135.94, 'Cr': 52, 'Fe': 55.934942, 'KCl': 74.55,
                  'ExoHaze_1000xSolar_300K': 53.06, 'H2O': 18.010565,
                  'ExoHaze_1000xSolar_400K': 53.06, 'H2SO4': 98.08,
                  'Fe2O3': 159.69, 'Fe2SiO4_KH': 203.77, 'FeO': 71.84, 
                  'FeS': 87.91, 'FeSiO3': 131.93, 'Hexene': 84.16, 
                  'Hibonite': 668, 'IceTholin': 53.06, 'NH3': 17.026549,
                  'Mg2SiO4_Fe_rich': 140.69, 'Mg2SiO4_amorph_sol_gel': 140.69, 
                  'Mg2SiO4_Fe_poor': 140.69, 'Mg2SiO4_amorph': 140.69,
                  'Mg2SiO4_crystalline': 140.69, 'MgSiO3_sol_gel': 100.39,
                  'Mg4Fe6SiO3_amorph_glass': 508.37, 'MgAl2O4': 142.27,
                  'Mg5Fe5SiO3_amorph_glass': 476.83, 'MgO': 40.3,
                  'Mg8Fe12SiO4_amorph_glass': 956.66, 'MgSiO3': 100.39,
                  'Mg8Fe2SiO3_amorph_glass': 382.21, 'MgSiO3_amorph': 100.39,
                  'MgFeSiO4_amorph_glass': 172.23, 'MgSiO3_crystalline': 100.39,
                  'MgSiO3_amorph_glass': 100.39, 'MnS': 87, 'MnS_KH': 87, 
                  'MnS_Mor': 87, 'Na2S': 78.04, 'NaCl': 58.44, 'S8': 256,
                  'NanoDiamonds': 12.01, 'NH4SH': 42, 'SiC': 40.10, 
                  'SiO': 44.08, 'SiO2': 60.08, 'SiO2_crystalline_2023': 60.08, 
                  'SiO2_amorph': 60.08, 'Saturn-Phosphorus-Haze': 53.06, 
                  'Soot': 53.06, 'Soot_6mm': 53.06, 'Tholin': 53.06, 
                  'Tholin-CO-0625': 53.06, 'TiC': 59.88, 'TiO2_anatase': 79.87, 
                  'TiO2_rutile': 79.87, 'VO': 67, 'ZnS': 97.46,
                  }

