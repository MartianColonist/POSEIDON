######################################################
######################################################
#  Functions that are used to incorporate surfaces into POSEIDON
######################################################
######################################################

import os
import numpy as np
import pandas as pd
import pymultinest
from mpi4py import MPI
from numba import jit, cuda
from spectres import spectres
from scipy.interpolate import interp1d as Interp


def find_nearest_less_than(searchVal, array):
    diff = array - searchVal
    diff[diff>0] = -np.inf
    idx = diff.argmax()
    return idx

def load_surface_components(surface_components):

    '''
    Loads in the txt files for albedos to store into model object

    Args:
        surface_components (list of strings):
            List of surface components (if surface_model = 'Lab_data').
    '''

    # Empty surface_component_albedos array
    surface_component_albedos = []

    # Find the directory where the user downloaded the input grid
    input_file_path = os.environ.get("POSEIDON_input_data")

    if input_file_path == None:
        raise Exception("POSEIDON cannot locate the input folder.\n" +
                        "Please set the 'POSEIDON_input_data' variable in " +
                        "your .bashrc or .bash_profile to point to the " +
                        "POSEIDON input folder.")
    
    # Load in the aerosol species
    surface_components = np.array(surface_components)

    for component in surface_components:
        file_path = input_file_path + 'surface_reflectivities/' + component + '.txt'
        data = np.loadtxt(file_path).T
        surface_component_albedos.append(data)

    return surface_component_albedos

def interpolate_surface_components(wl,surface_components,surface_component_albedos,):
        
    '''
    Interpolates txt files onto wavelength grid. Will throw up an error
    if the wavelength extends beyond lab data txt file

    Args:
        wl (np.array of float):
            Model wavelength grid (μm).
        surface_components (list of strings):
            List of surface components (if surface_model = 'Lab_data').
        surface_component_albedos (array):
            Array of [wavelength,albedo] for each txt file loaded in
    '''

    surf_reflect_array = []

    for n in range(len(surface_component_albedos)):

        wavelength_txt_file = surface_component_albedos[n][0]
        albedo_txt_file = surface_component_albedos[n][1]

        if (np.min(wl) < np.min(wavelength_txt_file)) or (np.max(wl) > np.max(wavelength_txt_file)):
            exception_txt = ('The wl grid exceeds the wavelengths of the albedo file: ' + surface_components[n] + ' (' + 
                             str(np.min(wavelength_txt_file)) + ', ' + str(np.max(wavelength_txt_file)) + ')')
            raise Exception(exception_txt)
        
        # Create interpolate object
        f = Interp(wavelength_txt_file, albedo_txt_file)

        # interpolate onto the wavelength grid
        albedo_interpolated = f(wl)

        # append to surf_reflect_array
        surf_reflect_array.append(albedo_interpolated)

    return surf_reflect_array
