import numpy as np 
from scipy.interpolate import RegularGridInterpolator

print('Making Interpolating Object')
# Set up x, y, and z axis 
T_array = np.arange(500,1350,50)
g_array = np.array([31,56,100,178,316,562,1000,1780,3162])
Kzz_array = np.array([2,4,7])

#LVL  P(BARS)     TEMP    MU      H2     HE       CH4      CO       CO2        NH3          N2        H2O        TiO        VO        FeH        HCN   H       Na       K       PH3       H2S

# There are two particular atmospheric pairs for quenched species that involve important molecules 
# in exoplanet and brown dwarf atmospheres: CO-CH4 and N2-NH3.
# So I will just interpolate P, T, CO, CH4, N2, NH3

matrix = np.empty((len(T_array),len(g_array),len(Kzz_array),7,68))

# Ok, now, we need to actually organize the data into a 3D matrix where each point is a list 
for T in T_array:
    for g in g_array:
        for Kzz in Kzz_array:
                
                directory = '/Users/elijahmullens/Desktop/Poseidon-temp/input/opacity/composition_files/'
                file_name = directory + str(T) + 'K_' + str(g) +'g_logkzz' + str(Kzz) + '.cmp' 
                file_as_numpy = np.loadtxt(file_name,comments='#').T

                pressure = file_as_numpy[1]
                temperature = file_as_numpy[2]

                CH4 = file_as_numpy[6]
                CO = file_as_numpy[7]
                NH3 = file_as_numpy[9]
                N2 = file_as_numpy[10]
                H2O = file_as_numpy[11]

                T_index = np.argwhere(T_array == T)[0][0]
                g_index = np.argwhere(g_array == g)[0][0]
                Kzz_index = np.argwhere(Kzz_array == Kzz)[0][0]

                matrix[T_index][g_index][Kzz_index][0][:] = pressure
                matrix[T_index][g_index][Kzz_index][1][:] = temperature
                matrix[T_index][g_index][Kzz_index][2][:] = CH4
                matrix[T_index][g_index][Kzz_index][3][:] = CO
                matrix[T_index][g_index][Kzz_index][4][:] = NH3
                matrix[T_index][g_index][Kzz_index][5][:] = N2
                matrix[T_index][g_index][Kzz_index][6][:] = H2O

interp = RegularGridInterpolator((T_array, g_array, Kzz_array), matrix)

def read_logX_cholla(T_eff,log_gs,log_Kzz):
     
    g_s = 10**(log_gs)
    best_fit = interp((T_eff,g_s,log_Kzz))

    pressure = best_fit[0]

    CH4 = best_fit[2][np.where(np.log10(pressure)<2)]  # Poseidon can only fit up to 100 bars
    CO = best_fit[3][np.where(np.log10(pressure)<2)] 
    NH3 = best_fit[4][np.where(np.log10(pressure)<2)] 
    N2 = best_fit[5][np.where(np.log10(pressure)<2)] 
    H2O = best_fit[6][np.where(np.log10(pressure)<2)] 

    return((CH4,CO,NH3,N2,H2O))

def read_PT_cholla(T_eff,log_gs,log_Kzz):
     
    g_s = 10**(log_gs)
    best_fit = interp((T_eff,g_s,log_Kzz))

    pressure = best_fit[0] # Poseidon can only fit up to 100 bars 

    temperature = best_fit[1][np.where(np.log10(pressure)<2)]
    pressure = best_fit[0][np.where(np.log10(pressure)<2)]


    return((pressure,temperature))
     