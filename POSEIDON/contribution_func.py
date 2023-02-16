from POSEIDON.core import compute_spectrum, compute_spectrum_c, compute_spectrum_p 
from POSEIDON.utility import bin_spectrum

import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np

def pressure_contribution_function(planet, star, model, atmosphere, opac, wl,P,
                                   spectrum_type = 'transmission',
                                   contribution_molecule_list = [],):


    '''
    Computes the pressure contribution function 

    Args:
        planet
        star
        model
        atmopshere
        opac
        wl 
        spectrum_type = 'transmission'
        
        contribution_molecule_list (np.array) 
            list of strings of molecules that user wants pressure contribution calculated for 
    
    Returns:
        Contribution (np.array)
            Array. [i,j,k] i = molecule number (or total if total = True), j = Pressure layer, k = Wavelength 
        Norm (np.array)
            Array [i,j] where i = molecule number and j = wavelength. If user wants to normalize them   
    '''

    # Generate our normal transmission spectrum
    spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                spectrum_type = 'transmission')

    # Define arrays where pressure contribution functions will live 
    Contribution = np.zeros(shape=(len(contribution_molecule_list)+1,len(P), len(spectrum)))
    # For denominator of contribution function 
    norm = np.zeros(shape=(len(contribution_molecule_list)+1,len(spectrum)))   # Running sum for contribution


    for i in range(len(contribution_molecule_list)+1):

        if i != len(contribution_molecule_list):

            for j in range(len(P)):

                new_spectrum = compute_spectrum_p(planet, star, model, atmosphere, opac, wl,
                                                    spectrum_type = 'transmission', save_spectrum = False,
                                                    disable_continuum = False, suppress_print = False,
                                                    Gauss_quad = 2, use_photosphere_radius = True,
                                                    device = 'cpu', y_p = np.array([0.0]),
                                                    contribution_molecule = contribution_molecule_list[i],
                                                    layer_to_ignore = j)

                # Find the difference between spectrums
                diff = spectrum - new_spectrum 

                # Add to contribution function (not yet normalized)
                Contribution[i,j,:] = diff

                # Increment normalization factor 
                norm[i,:] += diff

        # If its the last index it runs total, which is just total = True
        else:

            for j in range(len(P)):

                new_spectrum = compute_spectrum_p(planet, star, model, atmosphere, opac, wl,
                                                        spectrum_type = 'transmission', save_spectrum = False,
                                                        disable_continuum = False, suppress_print = False,
                                                        Gauss_quad = 2, use_photosphere_radius = True,
                                                        device = 'cpu', y_p = np.array([0.0]),
                                                        layer_to_ignore = j,
                                                        total = True)

                # Find the difference between spectrums
                diff = spectrum - new_spectrum 

                # Add to contribution function (not yet normalized)
                Contribution[i,j,:] = diff

                # Increment normalization factor 
                norm[i,:] += diff


    #for i in range(len(contribution_molecule_list)+1):
    #   for j in range(len(P)):
    #        Contribution[i,j,:] = np.divide(Contribution[i,j,:],norm[-1,:])

    return Contribution, norm

# Now normalize everything 
# Loop over each molecule + 1
#for i in range(len(contribution_molecule_list)+1):
#    # Loop over each layer 
#    for j in range(len(P)):
#        Contribution[i,j,:] = Contribution[i,j,:]/norm[i,:]




def plot_pressure_contribution(wl,P,
                               Contribution,
                               contribution_molecule_list = [], 
                               R = 100):

    # Plots out the pressure contribution functions. Only displays them, doesn't save them.
    
    for i in range(len(contribution_molecule_list)+1):

            fig, ax = plt.subplots(figsize=(10, 10))

            a = ax.contourf(wl, np.log10(P), Contribution[i,:,:],cmap='plasma')

            ax.set_ylabel('Log Pressure (bar)')
            ax.invert_yaxis()
            ax.set_xlabel('Wavelength ($\mu$m)')

            if i != len(contribution_molecule_list):
                    title = 'Contribution Function : ' + str(contribution_molecule_list[i])
            else:
                    title = 'Contribution Function : Total'
            
            ax.set_title(title)
            plt.colorbar(a, label='Transmission CF')
            plt.show()

            # Trying Ryan's Binning 

            fig = plt.figure()  
            fig.set_size_inches(14, 7)
            ax = plt.gca()

            ax.set_yscale("log")

            # Bin the wavelengths using the first pressure layer of the spectrum 
            # This is because bin_spectrum returns both a wl binned and spectrum grid and we want the wl binned for now 
            wl_binned, _ , _ = bin_spectrum(wl, Contribution[i,0,:], R)

            # Now to create the contribution function but binned 
            Contribution_binned = np.zeros(shape=(len(P), len(wl_binned)))

            # Now loop over all pressure layers 
            for j in range(len(P)):
                    _, Contribution_binned[j,:], _ = bin_spectrum(wl, Contribution[i,j,:], R)

            X_bin, Y_bin = np.meshgrid(wl_binned, P)
            
            # Plot binned contribution function
            contour_plot = plt.contourf(X_bin, Y_bin, Contribution_binned[:,:], 100, cmap=cmr.swamp_r)
            #contour_plot = plt.contourf(wl_binned, P, Contribution_binned[:,:], 100, cmap=cmr.swamp_r)

            ax.invert_yaxis()    

            ax.set_xlim([wl[0], wl[-1]])
            ax.set_ylim([P[0], P[-1]])        
            
            ax.set_ylabel(r'P (bar)', fontsize = 15, labelpad=0.5)
            ax.set_xlabel(r'Wavelength ' + r'(μm)', fontsize = 15)
            ax.set_title(title)

            plt.colorbar()
            plt.show()


def photometric_contribution_function(wl, P, Contribution, 
                                      contribution_molecule_list = [],
                                      ):

    wl_min = np.min(wl)
    wl_max = np.max(wl)

    # Bin Stuff from minimum wavelength to maximum wavelength by 0.1 
    bins = np.arange(wl_min,wl_max+0.1,0.1)

    for b in range(len(bins)):
        bins[b] = round(bins[b],1)

    # Make it so the last bin includes the max wavelength (if not it will be a seperate bin)
    bins[-1] += 0.1
    bin_indices = np.digitize(wl, bins)
    bins[-1] -= 0.1

    bincount = np.bincount(bin_indices)

    # Finds the indices to loop over in the wavelength ranges
    indices_for_loop = []
    for n in range(len(bincount)):
        if n == 0:
            indices_for_loop.append(n)
        else:
            indices_for_loop.append(np.sum(bincount[0:n+1])-1)


    # Now to find photometric contribution 

    # [molecule][photometric conitrbution for each bin]
    photometric_contribution = []

    # Loop over each molecule
    for i in range(len(contribution_molecule_list)+1):

        median_array_one_molecule = []
        # Loop over each wavelength range 
        for j in range(len(indices_for_loop)-1):
            # Loop over each pressure range to get the median 
            temp_row = []
            for p in range(len(P)):

                temp_row.append(np.nanmedian(Contribution[i,p,indices_for_loop[j]:indices_for_loop[j+1]]))

            median_array_one_molecule.append(temp_row)

        photometric_contribution.append(median_array_one_molecule)

    # Finding the total photometric contribution for each molecule by adding everything up    
    photometric_total = []
    for i in range(len(photometric_contribution)):
        temp_row = np.zeros(len(photometric_contribution[i][0]))
        for j in range(len(photometric_contribution[i])):
            temp_row += photometric_contribution[i][j]
            
        photometric_total.append(temp_row)


    return photometric_contribution, photometric_total


def plot_photometric_contribution(wl,P,
                                  photometric_contribution, photometric_total,
                                  contribution_molecule_list = []):

    # Loop over each molecule

    labels = []
    for i in contribution_molecule_list:
        labels.append(i)
    labels.append('Total')

    for i in range(len(contribution_molecule_list)+1):

        fig, ax = plt.subplots(figsize=(10, 10))

        for b in range(len(photometric_contribution[i])):
            ax.plot(photometric_contribution[i][b],np.log10(P))

        ax.set_ylabel('Log Pressure (bar)')
        ax.invert_yaxis()
        ax.set_xlabel('Contribution')
        title = 'Photometric Contribution Function : ' + str(labels[i])
        ax.set_title(title)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylabel('Log Pressure (bar)')
        ax.invert_yaxis()
        ax.set_xlabel('Contribution')
        title = 'Photometric Contribution Function All Wavelength : ' + str(labels[i])
        ax.set_title(title)
        ax.plot(photometric_total[i],np.log10(P))
        plt.show()

    # Plots all of them together, and the log version as well 
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylabel('Log Pressure (bar)')
    ax.invert_yaxis()
    ax.set_xlabel('Contribution')
    title = 'Photometric Contribution Function All Wavelength All Molecules:'
    ax.set_title(title)
    for i in range(len(contribution_molecule_list)):
        ax.plot(photometric_total[i],np.log10(P), label = labels[i])
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylabel('Log Pressure (bar)')
    ax.invert_yaxis()
    ax.set_xlabel('Log Contribution')
    title = 'Photometric Contribution Function All Wavelength All Molecules:'
    ax.set_title(title)
    for i in range(len(contribution_molecule_list)):
        ax.plot(np.log10(photometric_total[i]),np.log10(P), label = labels[i])
    ax.legend()
    plt.show()

