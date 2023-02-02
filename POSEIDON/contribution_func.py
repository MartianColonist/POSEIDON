
import numpy as np
import matplotlib.pyplot as plt
from POSEIDON.constants import R_Sun, R_J, M_J
from POSEIDON.core import make_atmosphere
from POSEIDON.core import compute_spectrum


# Current issues 
# CIA not generalized 
# Not sure how to make copy of atmosphere object without remaking every loop iteration 
# When we turn off mixing ratio, it just fills everything back up to 1 (need to work with opacities)

def pressure_contribution_function(planet,star,model,P,P_ref,best_fit_params,
                                   opac,wl,spectrum_type='transmission'):
    
    
        ''' Computes the pressure contribution function, total and per molecule
       
        Inputs:
        
        planet = from create_planet()
        star = from create_star()
        model = from define_model()
        P = pressure grid used to create forward model 
        P_ref = reference pressure 
        best_fit_params = from a retreival, same order as model['param_names']
        opac = opacity object 
        wl = wavelength grid 
        spectrum_type = 'transmission'

        Outputs:
           
        Pressure Contribution Function for each molecule and total. 
       
    '''
    
        # Create atmosphere object 
        # Note : I wasn't able to run this without having to recreate the atmosphere object each time 
        # It seems like whenever I made a copy it would overwrite the original object no matter what 
        
        R_p_ref = best_fit_params[0] * R_J
        
        PT_params = np.array([best_fit_params[1]])
        
        log_X_params = np.array([best_fit_params[2:]])
        
        # Make a normal spectrum just to subtract from the new spectrum
        atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                    PT_params, log_X_params)
        
        spectrum = compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                spectrum_type = 'transmission')
        
        # Create contribution Function 
        # Order of indices : 
        # [H2,He,active species,all species][Pressure layers][Difference in terms of wavelength]
        

        # Define arrays where pressure contribution functions will live 
        Contribution = np.zeros(shape=(len(model['chemical_species'])+1,len(P), len(spectrum)))
        # For denominator of contribution function 
        norm = np.zeros(shape=(len(model['chemical_species'])+1,len(spectrum)))   # Running sum for contribution

        # Loop over each molecule
        for i in range(len(model['chemical_species'])):
            # Loop over each layer 
            for j in range(len(P)):

                # Generate the atmosphere
                atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                    PT_params, log_X_params)

                # Bug : I can't just make a copy of the atmopshere object above without breaking something 
                new_atmosphere = atmosphere

                # 'X' array lines up with chemical species one-to-one
                new_atmosphere['X'][i,j,0,0] = 0.0

                # 'X_active' only lines up with the end of chemical species 
                if i > 1:
                    new_atmosphere['X_active'][i-2,j,0,0] = 0.0

                # 'X_CIA' only has He and H (for now)
                # First index = species Second index = Reaction

                # if we are on H2
                if i == 0: 
                    new_atmosphere['X_CIA'][0,0,j,0,0] = 0.0 # H2 in H2-H2
                    new_atmosphere['X_CIA'][1,0,j,0,0] = 0.0 # H2 in H2-H2
                    new_atmosphere['X_CIA'][0,1,j,0,0] = 0.0 # H2 in H2-He
                    new_atmosphere['X_CIA'][1,1,j,0,0] = 0.0 # He in H2-He
                    
                # if we are on He
                if i == 1:
                    new_atmosphere['X_CIA'][0,1,j,0,0] = 0.0 # H2 in H2-He
                    new_atmosphere['X_CIA'][1,1,j,0,0] = 0.0 # He in H2-H2

                # Calculate new spectrum without molecule i and layer j
                new_spectrum = compute_spectrum(planet, star, model, new_atmosphere, opac, wl,
                                                spectrum_type = 'transmission')

                # Find the difference between spectrums
                diff = spectrum - new_spectrum 

                # Add to contribution function (not yet normalized)
                Contribution[i,j,:] = diff

                # Increment normalization factor 
                norm[i,:] += diff

        # Total contribtuion function with whole layer turned off 
        index_end = len(model['chemical_species'])

        for j in range(len(P)):

            # Generate the atmosphere
            atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, 
                                    PT_params, log_X_params)

            new_atmosphere = atmosphere

            # Turn off all molecules in layer j
            for i in range(len(model['chemical_species'])):

                new_atmosphere['X'][i,j,0,0] = 0.0

                if i > 1:
                    new_atmosphere['X_active'][i-2,j,0,0] = 0.0

            # Turn off all X_CIA 
            for k in range(len(new_atmosphere['X_CIA'][0])):
                new_atmosphere['X_CIA'][0,k,j,0,0] = 0.0
                new_atmosphere['X_CIA'][1,k,j,0,0] = 0.0

            # Find new spectrum without layer j
            new_spectrum = compute_spectrum(planet, star, model, new_atmosphere, opac, wl,
                                            spectrum_type = 'transmission')

            # Find the difference
            diff = spectrum - new_spectrum 

            # Add to contribution function (not yet normalized)
            Contribution[index_end,j,:] = diff

            # Increment normalization factor 
            norm[index_end,:] += diff


        # Now normalize everything 
        # Loop over each molecule + 1
        for i in range(len(model['chemical_species'])+1):
            # Loop over each layer 
            for j in range(len(P)):
                Contribution[i,j,:] = Contribution[i,j,:]/norm[i,:]
                
        return Contribution


def photometric_contribution_function(model,wl,N_layers,Contribution):
    
    ''' Computes photometric contribution function (averaging over wavelengths)
        
            Inputs:
            
            model = from define_model()
            wl = wavelength grid 
            N_Layers = How many pressure layers there are 
            Contribution = from pressure_contribution_function()

            Outputs:
            
            bins,photometric_contribution, photometric_total (goes straight into graphing function) 
        
        '''

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

    labels = np.append(model['chemical_species'],'Total')

    # [molecule][photometric conitrbution for each bin]
    photometric_contribution = []

    # Loop over each molecule
    for i in range(len(labels)):

        median_array_one_molecule = []
        # Loop over each wavelength range 
        for j in range(len(indices_for_loop)-1):
            # Loop over each pressure range to get the median 
            temp_row = []
            for p in range(N_layers):

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

    return bins,photometric_contribution, photometric_total 

# Graphing Functions
def plot_pressure_contribution(model,P,wl,Contribution):

    labels = np.append(model['chemical_species'],'Total')

    for i in range(len(labels)):

        fig, ax = plt.subplots(figsize=(10, 10))

        levels = np.arange(-16,2,2)
        a = ax.contourf(wl, np.log10(P), np.log10(Contribution[i,:,:]),cmap='plasma', levels = levels)

        ax.set_ylabel('Log Pressure (bar)')
        ax.invert_yaxis()
        ax.set_xlabel('Wavelength ($\mu$m)')
        title = 'Contribution Function : ' + str(labels[i])
        ax.set_title(title)
        plt.colorbar(a, label='Log10 Transmission CF')
        
def plot_photometric_contribution(model,P,bins,photometric_contribution,photometric_total):
    
     # Now to find contribution 
    labels = np.append(model['chemical_species'],'Total')

    # Loop over each molecule
    for i in range(len(labels)):

        fig, ax = plt.subplots(figsize=(10, 10))

        for b in range(len(bins)-1):
            label = '[' + str(round(bins[b],1)) + ':' + str(round(bins[b+1],1)) + ')'
            if b == len(bins)-2:
                label = '[' + str(round(bins[b],1)) + ':' + str(round(bins[b+1],1)) + ']'
            ax.plot(photometric_contribution[i][b],np.log10(P), label = label)

        ax.set_ylabel('Log Pressure (bar)')
        ax.invert_yaxis()
        ax.set_xlabel('Contribution')
        title = 'Photometric Contribution Function : ' + str(labels[i])
        ax.set_title(title)
        ax.legend()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylabel('Log Pressure (bar)')
        ax.invert_yaxis()
        ax.set_xlabel('Contribution')
        title = 'Photometric Contribution Function All Wavelength : ' + str(labels[i])
        ax.set_title(title)
        ax.plot(photometric_total[i],np.log10(P))
        plt.show()