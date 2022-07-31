import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

database = h5py.File('./eq_database.hdf5', 'r')
T_low = 300
T_high = 4000
T_step = 100

P_low = -7
P_high = 2
P_step = 0.5

Met_low = -1
Met_high = 4
Met_step = 0.05

C_O_low = 0.2
C_O_high = 2
C_O_step = 0.05
# finer between 0.9 and 1.1
C_O_finer_low = 0.9
C_O_finer_high = 1.1
C_O_finer_step = 0.01

def get_num(low, high, step):
  return int((high-low)/step)+1

T_num = get_num(T_low, T_high, T_step)
P_num = get_num(P_low, P_high, P_step)
Met_num = get_num(Met_low, Met_high, Met_step)

temperature_gird = np.linspace(T_low, T_high, T_num)
pressure_grid = np.logspace(P_low, P_high, P_num)
metallicity_grid = np.logspace(Met_low, Met_high, Met_num)
c_o_grid = np.concatenate((np.linspace(C_O_low, C_O_finer_low, get_num(C_O_low, C_O_finer_low, C_O_step), endpoint=False), \
           np.linspace(C_O_finer_low, C_O_finer_high, get_num(C_O_finer_low, C_O_finer_high, C_O_finer_step), endpoint=False), np.linspace(C_O_finer_high, C_O_high, get_num(C_O_low, C_O_finer_low, C_O_step))))

C_O_num = len(c_o_grid)


### P, Met are in logarithmic scale; T, C_O are in linear scale
def getMR(P_arr, T_arr, C_O, Met, molecules):
    assert len(P_arr) == len(T_arr), "Pressure and temperature have different length."
    database = h5py.File('./eq_database.hdf5', 'r')
    size = len(P_arr)
    C_O = np.full(size, C_O)
    Met = np.full(size, Met)
    MR_dict = {}
    for _, molecule in enumerate(molecules):
        array = np.array(database[molecule+'/log(X)'])
        array = array.reshape(Met_num, C_O_num, T_num, P_num)
        grid = RegularGridInterpolator((np.log10(metallicity_grid), c_o_grid, temperature_gird, np.log10(pressure_grid)), np.log10(array))
        MR_dict[molecule] = grid(np.vstack((Met, C_O, T_arr, P_arr)).T)
    return MR_dict
