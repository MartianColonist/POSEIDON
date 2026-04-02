This readme details the references for various txt files in this folder. 

These are the surface albedos that come default with POSEIDON. 
However, it is easy to add you own text files to the 'surface_reflectivities' folder in 'inputs'

All reflectivities are directional-hemispherical reflectivities (r_dh), except for data from Goodis Gordon et al. 2025 

See 
Mullens et al. 2026 for full description Sources 1-6
Zelakiewicz et al. 2026 for Source 7 

Author: Elijah Mullens (eem85@cornell.edu)

Source 1 : Hu, Ehlmann, & Seager 2012 (H12)
- From Table 2
- https://ui.adsabs.harvard.edu/abs/2012ApJ...752....7H/abstract
- Publicly available from PLATON (https://github.com/ideasrule/platon)

Source 2 : Paragas 2025 (P25)
- From Figure 3
- https://ui.adsabs.harvard.edu/abs/2025ApJ...981..130P/abstract
- Publicly available from PLATON (https://github.com/ideasrule/platon)

Source 3 : Hammond 2025 (H25)
- Single scattering albedos of surface types in Table 1
- Converted to r_dh using Equation 4 in their paper 
- https://ui.adsabs.harvard.edu/abs/2025ApJ...978L..40H/abstract
- Used the names from Column 1 of the tale, accessed data through Zenodo 
- https://zenodo.org/records/14017134

Source 4 : First 2024 (F25)
- Figure 2
- https://ui.adsabs.harvard.edu/abs/2025NatAs...9..370F/abstract
- Accessed data through Zenodo, or GitHub 
- https://zenodo.org/records/12822668
- https://github.com/ishan-mishra/rocky_exo_jwst/tree/main

Source 5 : Hammond 2025 (H25)
- Figure 2
- https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4569F/abstract
- Accessed data through Zenodo 
- https://zenodo.org/records/6323322#.YufC0S-B2lF
- 100963-30 and 112774-10 were not glassy and therefore not reported


Source 6 : Goodis Gordon et al. 2025 (GG25)
- From Figures 1 and 3 
- https://ui.adsabs.harvard.edu/abs/2025ApJ...983..168G/abstract
- Sent from Goodis Gordon (pers comm)

Source 7 : Zelakiewicz et al. 2026 (Z26)
- Tables 2-3, Figure 1 (b)
- https://ui.adsabs.harvard.edu/abs/2026arXiv260325694Z/abstract


Misc (all from 0.01 to 100 microns):

Black: Albedo = 0 at all wavelengths
White: Albedo = 1 at all wavelengths 

Red: 	Albedo = 1 from 0.620-0.750 microns
Orange:	Albedo = 1 from 0.590-0.620 microns	
Yellow: Albedo = 1 from 0.570-0.590 microns
Green:	Albedo = 1 from 0.495-0.570 microns
Blue:	Albedo = 1 from 0.450-0.495 microns
Purple:	Albedo = 1 from 0.380-0.450 microns


 