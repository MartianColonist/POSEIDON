POSEIDON V1.2 Base Aerosol Database README 
Author: Elijah Mullens (eem85@cornell.edu)
Date : 8/12/2024

Thanks to Dr. Hannah Wakeford, Dr. Daniel Kitzmann, Dr. Elspeth Lee, Dr. Ben Burningham, Dr. Mark Marley, and Dr. Sarah Moran for help putting this together (Thanks y'all!).

This readme corresponds to Table 1 in Mullens et al. 2024
'Implementation of Aerosol Mie Scattering in POSEIDON with Application to the hot Jupiter HD 189733 b's Transmission, Emission, and Reflected Light Spectrum'

Refractive Indices for POSEIDON are found under the 'refractive_indices_txt_files' folder. Each txt file is split up into folders depending on which database they originate from. There is a separate folder where each txt file has their name exactly as it appears in POSEIDON supported_species.py. 

Images showing the precomputed aerosol properties, as well as the refractive indices from 0.2 to 30 um are shown in the aerosol_database.pdf and also the png previews in the opacity database tab in POSEIDON's readthedocs.

Please, please, please double check things if you end up using a reference! 

Sorry in advance for any typos

#################################
Databases
#################################

The refractive indices used to compute the radiative properties of the aerosols in Table 1 of Mullens et al. 2024 were compiled from 5 databases. 

---------------------------------
Wakeford & Sing (2015) - WS15

ADS   : https://ui.adsabs.harvard.edu/abs/2015A%26A...573A.122W/abstract
Table : 1

Also available on Dr. Wakeford's site:

https://stellarplanet.org/science/condensates/

Wakeford and Sing (2015) compiled aerosol refractive indices for aerosols predicted to form in the upper atmospheres of hot Jupiters. Their philosophy was to never interpolate or extrapolate indices, and stuck to lab data. Wakeford & Sing (2015) utilized both plot digitizers and tables.

Some refractive indices were updated for the follow-up paper:

Wakeford (2017)
ADS    : https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4247W/abstract
Table  : N/A

Updated indices from 2015 to 2017 will be noted below

---------------------------------
Kitzmann & Heng (2018) - KH18

ADS    : https://ui.adsabs.harvard.edu/abs/2018MNRAS.475...94K/abstract
Table  : 1

Also available on GitHub 

https://github.com/NewStrangeWorlds/LX-MIE/tree/master/compilation

Kitzmann & Heng (2018) compiled aerosol refractive indices for aerosols expected to form in extrasolar planets and brown dwarfs. Their philosophy was to utilize the Kramers-Kronig relation or spline interpolation to recover missing real and imaginary indices from datasets. Kitzmann & Heng (2018) utilized both a plot digitizer and tables.

Additionally, when an aerosol is considered to be anisotropic, they would weigh the dielectric functions in each direction by 1/3 and convert them to refractive indices. 

When wavelengths for datasets overlapped, a specific dataset was chosen in order to make the refractive indices look smooth without jumps. 

---------------------------------
gcmCRT 

ADS    : https://ui.adsabs.harvard.edu/abs/2022ApJ...929..180L/abstract

This database is available on GitHub and includes aerosols for an exoplanet GCM.

https://github.com/ELeeAstro/gCMCRT/tree/main/data/nk_tables

---------------------------------
Burningham (2021) - B21

ADS    : https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract
Table  : 3 

This database is available by emailing Dr. Ben Burningham.

Compiled aerosols for retrievals of brown dwarf 2M224-0158. Note that the refractive indices have been interpolated to match the wavelength grid found in the EGP Mie code (which means that all these entries have the same wavelength grid from 0.268 to 227.5 um). 

Sometimes, this means that things are interpolated to shorter to longer wavelengths to fit the grid, which we will note below. 

---------------------------------
OpacityTool - optool

ADS    : https://ui.adsabs.harvard.edu/abs/2021ascl.soft04010D/abstract

This database is available on GitHub and includes aerosols to compute properties of.

https://github.com/cdominik/optool/tree/master/lnk_data

---------------------------------

The remainder of the references are from Mullens et al 2024.

ADS : 

https://github.com/MartianColonist/POSEIDON/tree/Elijah_V12/refractive_indices_txt_files/Misc

Or Zenodo

#################################
Secondary Databases
#################################

There are four common sources of refractive indices that are useful tools to find data from.

---------------------------------
Handbook of Optical Constants of Solids
Edited by Edward D. Palik 

There are three versions of this book: 

Volume 1 : 1985
ADS      : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Volume 2 : 1991 
ADS      : https://ui.adsabs.harvard.edu/abs/1991hocs.book.....P/abstract

Volume 3 : 1998
ADS      : N/A

This is a great resource to find refractive indices in. The book can usually be accessed through a university log-in from the following websites: 

https://www.sciencedirect.com/book/9780125444156/handbook-of-optical-constants-of-solids

https://app.knovel.com/kn/resources/kpHOCS000M/toc

In their 'Critiques' sections, Palik and co-authors compile refractive index from multiple lab sources. If a Palik entry is used, be sure to dig up the chapter in the textbook in order to see where the data comes from.

Note that their tables will often list the scientific notation for the first and last entry that has that same scientific notation (I.e. it will start with 1e-2 and end with 1e-2 and it is implied that every value between has a 1e-2 value).

Citations for Palik entries include the author of the specific section of the textbook, and Palik (i.e. Ribarsky in Palik (1985)).

---------------------------------
Database of Optical Constants for Cosmic Dust (DOCCD)

Run by the Laboratory Group of the AIU

This is a great resource as it compiles refractive indices of many relevant astronomical species from more current papers. 

https://www.astro.uni-jena.de/Laboratory/OCDB/index.html

Note that in early 2025 the website was moved: 

https://www2.astro.uni-jena.de/Laboratory/OCDB/index.html

---------------------------------
The Cosmic Ice Laboratory

Run by the Astrochemistry Laboratory at NASA

This is a great resources for very cold ices, including many hydrocarbon ices. While none of these indices are in POSEIDON at the moment, feel free to pull constants from this website and add them to the aerosol database. (See 'Make-Aerosol-Database.ipynb' tutorial).

https://science.gsfc.nasa.gov/691/cosmicice/constants.html

---------------------------------
The Optical Constants Database

Run by the NASA AMES Astrophysics and Astrochemistry Laboratory.

Compiles many ices and organic refractory materials. While none are in POSEIDON at the moment, feel free to pull constants from this website and add them. 

https://ocdb.smce.nasa.gov/

---------------------------------

#################################
Definitions and Concepts 
#################################

These are things I had to google a lot while making this.

---------------------------------
Refractive Indices 

Solids and liquids have their optical properties defined by wavelength-dependent refractive indices. Real refractive indices describe the scattering properties of an aerosol, while the imaginary describe the absorptive properties.
---------------------------------
Dielectric Function

The dielectric function is the square of the refractive index. Many lab papers will cite this.
---------------------------------
Wavenumber

The bane of all exoplanet scientists. Wavenumber is 1/cm. 
100 cm-1     = 100 um ~ 0.012 eV
1000 cm-1    = 10 um  ~ 0.12 eV
10000 cm-1   = 1 um   ~ 1.2 eV
100000 cm -1 = 0.1 um ~ 12 eV
---------------------------------
Kramers-Kronig Relation

In theory, the real and imaginary refractive indices are related through the Kramers-Kronig relation. Kitzmann and Heng 2018 utilize the Kramers-Kronig relation to fill in gaps where real or refractive indices were missing, and many lab papers will measure either the real or imaginary index and get the other through this relation. 

A good open source pipeline for this is pyElli

https://github.com/PyEllips/pyElli
---------------------------------
Lab Techniques 

There are a multitude of lab techniques utilized to measure the refractive constants/dielectric functions of aerosols. 

- Thin Film 

Aerosols are deposited on a thin film and then reflectance measurements are made. The thickness of the film is important to measure. From these measurements, one can derive the real refractive indices (and get the imaginary through Kramers-Kronig). It is also possible to get the transmittance through a film and derive imaginary refractive indices (and get the real through Kramers Kronig).

- Ellipsometry 

A technique for deriving refractive indices of thin films. Measures change in polarization of incident radiation due to reflection, absorption, scattering, or transmittance. 

- KBr pellet 

Ground powder of aerosols can be deposited on KBr pellets, which are optically transparent from 0.22 to 30 um. Transmittance is measured (how light is attenuated). Together with the powder density, one can derive imaginary refractive indices (and get the real through Kramers-Kronig).

- Natural Crystal

Crystals from Earth. Can perform both reflective and transmittance measurements. Crystals are usually cut and polished to find crystallographic axes easier. 

- Single Crystal

Crystals are grown in a lab (usually defined as a 'single crystal' or 'synthetic').

- Airborne 

Powders are suspended in a gas chamber where their optical properties are measured. 

---------------------------------
Amorphous vs Crystalline 

Amorphous solids have no crystal structure. This is usually due to being solidified too fast, or due to being organic in nature. Due to this, amorphous solids are isotropic and have little fine structure in their spectra. Additionally amorphous features are less sensitive to temperature than crystalline.

Glasses are a special kind of amorphous solid where the liquid/molten properties are 'frozen' in. (Think why glass is see-through is because it is molten sand that was frozen in a liquid configuration). All glasses are amorphous, but not all amorphous are glasses. Glass is considered to be a supercooled, configurationally frozen liquid. Therefore, it has optical properties similar to its liquid/molten state. 

Sol-gel is a technique to make amorphous solids. (Involves an liquid solution that is mixed with solids to form a gel, after which the particles are removed and usually densified to remove porosity).

Crystalline solids do have crystal structure and therefore can be isotropic or anisotropic, and have fine structure in their spectra. 

---------------------------------
Crystal Systems and Directionality

There are seven main types of crystal systems. 

cubic, tetragonal, hexagonal, trigonal/rhombohedral, orthorhombic/rhombic, monoclinic, triclinic

Where sometimes trigonal/rhomohedral is considered a subset of hexagonal 

Cubic systems are isotropic 

Tetragonal, hexagonal, and trigonal/rhombohedral crystals are anisotropic and usually uniaxial, meaning that they only have one optical axis. 

Optical properties are defined as polarizations of light parallel to the c-axis (extraordinary, E||c) and polarizations perpendicular to the c-axis (ordinary, E⟂c).

Orthorhombic/rhombic, monoclinic, and triclinic are anisotropic and usually biaxial, meaning that they have two optical axes.

Optical properties are defined as polarizations of light parallel to the a-axis, b-axis, and c-axis (E||a, E||b, and E||c)

When a specific direction is used for a non-cubic crystal, it will be noted below. 
When directions are averaged, it will be noted below.
(Note that the ordinary index gets a 2/3 weighting and the extraordinary a 1/3 weighting for uniaxial materials) 

There are exceptions to the rules above, which will be noted below.
---------------------------------
Equilibrium vs Disequilibrium Cloud Formation 

Parmentier (2016), section 2.3 gives a great summary of equilibrium vs disequilibrium cloud formation in section. We will refer to both in the sections below.

https://ui.adsabs.harvard.edu/abs/2016ApJ...828...22P/abstract#:~:text=We%20suggest%20that%20a%20transition,T%20transition%20on%20brown%20dwarfs.

'In the equilibrium cloud approach it is assumed that the change of the physical properties of a parcel of gas advected around the planet is slow compared to the condensation timescale. When a parcel of gas is transported from a hot to a cold part of the atmosphere, the most refractory condensates, such as MgSiO3, form first, depleting the gas in cloud-forming elements such as Si. When the temperature drops low enough, more volatile compounds, such as SiO2, are unable to form since the surrounding gas is depleted in Si (e.g., Visscher et al. 2010). If the opposite is assumed, i.e., if the parcels of gas are supposed to move faster than the growing of the grains, then all condensates form at the same time, leading to a prevalence of more volatile compounds such as SiO2 (e.g., Helling et al. 2008).'

Visscher 2010 (Equilibrium)
https://iopscience.iop.org/article/10.1088/0004-637X/716/2/1060

Helling 2008 (Disequilibrium)
https://ui.adsabs.harvard.edu/abs/2008arXiv0809.3657H/abstract

Evidence for equilibrium cloud formation (from Visscher 2010): 

1. First, the presence of germane (GeH4) and the absence of silane (SiH4) in the upper atmospheres of Jupiter and Saturn (even though Si is expected to be much more abundant than Ge) can be explained by the removal of Si from the gas into silicate clouds deeper in the atmosphere, whereas Ge remains in the gas phase

2. Second, the detection of H2S in Jupiter’s troposphere by the Galileo entry probe indicates that Fe must be sequestered into a cloud layer at deep atmospheric levels, because the formation of FeS would otherwise remove H2S from the gas above the ∼700 K level

3. Third, absorption from monatomic K gas in the spectra of T dwarfs (Burrows et al. 2000; Geballe et al. 2001) requires the removal of Al and Si at deeper atmospheric levels, because K would otherwise be removed from the observable atmosphere by the condensation of orthoclase (KAlSi3O8; Lodders & Fegley 2006). 

4. The presence of monatomic Na gas in brown dwarfs (Kirkpatrick et al. 1999; Burgasser et al. 2003; McLean et al. 2003; Cushing et al. 2005) also suggests Al and Si removal, because albite (NaAlSi3O8) condensation would otherwise effectively remove Na from the observable atmosphere.

5. Furthermore, the removal of Na by Na2S cloud formation is consistent with the observed weakening of Na atomic lines throughout the L dwarf spectral sequence and their disappearance in early T dwarfs

Evidence for disequilibrium cloud formation has come from observations of quartz (SiO2) in sub-stellar atmospheres (Burningham 2021, https://ui.adsabs.harvard.edu/abs/2021arXiv210504268B/abstract, and Grant 2023, https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract)

---------------------------------
Particle size vs Refractive Indices 

The short wavelength, UVIS, radiative properties of aerosols mostly depends on particle size (weakly dependent on species) whereas absorption features depends on refractive indices (strongly dependent on species). 

Note that some species can have absorption features or non-linear scattering properties (good examples are C, MnS, etc) but that many species have scattering that depends mostly on particle size. 

For more information on on small-particle, medium-particle, and large-particle scattering and absorption theory, see the following textbooks:

Atmospheric Radiation: Theoretical Basis by Goody & Yung (1989), chapter 7 and 8
Giant Planets of Our Solar System by Irwin (2009), chapter 6
Exoplanet Atmospheres:Physical Processes by Seager (2010), chapter 8

---------------------------------
Structure of the aerosol entries 
---------------------------------

Aerosol Name      : 
Name in POSEIDON  : 
Database          : 
Wavelengths       :

Chemical Formula  :
Crystal or Amorph :  
Crystal Shape     : 

Refractive Index References:
--------------
ADS        :
Paper Info :
--------------

Exoplanet/Brown Dwarf Papers:

#################################
Super-Hot Aerosols
#################################

We start with aerosols that are expected to form on super/ultra-Hot Jupiters.

These are condensates found in the M-L transition space. Specifically, Ca, Ti, and Al bearing species have been found to condense out. It is expected that hibonite, cordunum, and perovskite will condense out. These species are expected to deplete atmosphere of aluminum. 

Wakeford (2017) (Application of these aerosols to exoplanets, see Figure 1)
ADS : https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4247W/abstract
---------------------------------
---------------------------------

Aerosol Name      : Hibonite
Name in POSEIDON  : Hibonite
Database          : W15 (Updated for W17)
Wavelengths       : 2-1000 um

Chemical Formula  : CaAl[12]O[19] (really, Ca[0.85]Al[11.37]Ti[0.26]Fe[0.38]O[19])
Crystal or Amorph : Crystalline 
Crystal Shape     : Hexagonal (Uniaxial)

Refractive Index References:
--------------
Mutschke 2002 

ADS        : https://ui.adsabs.harvard.edu/abs/2002A%26A...392.1047M/abstract

Paper Info :
 
Measured the optical constants via IR reflectance of crystalline, natural Hibonite crystals from Evisa and Antsirabe Madagascar. The refractive indices of the txt file are made up of the extraordinary refractive indices of the Antsirabe crystals. 

Also on DOCCD (Hibonite E||c)
https://www2.astro.uni-jena.de/Laboratory/OCDB/aloxides.html
--------------

Exoplanet/Brown Dwarf Papers:

Wakeford (2017)
ADS : https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4247W/abstract

---------------------------------
---------------------------------
Aerosol Name      : Gamma Cordundum
Name in POSEIDON  : Al2O3
Database          : WS15
Wavelengths       : 0.335-150 um

Chemical Formula  : Al[2]O[3]
Crystal or Amorph : Crystalline 
Crystal Shape     : Cubic (Isotropic)

Refractive Index References:
--------------
Koike et al (1995)

ADS        : https://ui.adsabs.harvard.edu/abs/1995Icar..114..203K/abstract
Paper Info :

Used two different kinds of gamma corundum (a commercially available Alumina and a combustion product ISAS). Used the KBr pellet method. Indices available on Table A1. 0.3-0.4 um utilized the ISAS column, 0.5,0.6,0.7,0.8,0.9,1.0,2.0,...,10.0,10.1,...150 um utilized the Alumina Column. 

Still a mystery where the other wavelength data comes from (the in between wavelengths, for example, between 1-2 um). It can be assumed that this was either an extrapolation, or a plot digitizer of Figure 7.
--------------

Exoplanet/Brown Dwarf Papers:

Can be considered a Mg-free spinel. Makes up rubies and sapphires. 

Wakeford (2017)
ADS : https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4247W/abstract

---------------------------------
---------------------------------

Aerosol Name      : Corundum
Name in POSEIDON  : Al2O3_KH
Database          : KH18
Wavelengths       : 0.2-500 um

Chemical Formula  : Al[2]O[3]
Crystal or Amorph : Mixed
Crystal Shape     : Gamma is Cubic, Amorphous is N/A

Note from Dr. Daniel Kitzmann: 

'For Al2O3, I used the porous data from Begemann that goes up to 500 microns and the ISAS data from Koike (their Table A1). The data sets were stitched together at about 12 microns. Presumably I did that because that formed a smooth transition between the two data sets.

The joined data was then interpolated onto a new wavelength grid with 200 values equidistantly on logarithmic wavelength space. This is why the data points in my compilation don't exactly match those from the two papers.'

Refractive Index References:
--------------
Begemann et al. (1997) (7.8-500 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1997ApJ...476..199B/abstract

Paper Info :

Amorphous alumina was produced using the sol-gel technique. It was found that amorphous alumina transforms to gamma-crystalline (cubic) around 723-873K, and alpha-crystalline (hexagonal) above 1273K. Paper has both compact and porous Al2O3 in Table 1. The porous indices were used since they go up to 500 um.

Also found in DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/aloxides.html
--------------
Koike (1995) (0.12-12 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1995Icar..114..203K/abstract

Paper Info :

Same as above (used in WS15 Al2O3 entry). 0.2-0.4 um in the txt file utilize the ISAS column. See note above. 
--------------

Exoplanet/Brown Dwarf Papers:

Can be considered a Mg-free spinel. Makes up rubies and sapphires. 

Wakeford (2017)
ADS : https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4247W/abstract

---------------------------------
---------------------------------

Aerosol Name      : Perovskite
Name in POSEIDON  : CaTiO3
Database          : WS15
Wavelengths       : 2-50 um

Chemical Formula  : CaTiO[3]
Crystal or Amorph : Crystal 
Crystal Shape     : Orthorhombic (near cubic structure)


Refractive Index References:
--------------
Posch (2003) 

ADS        : https://ui.adsabs.harvard.edu/abs/2003ApJS..149..437P/abstract

Paper Info :

Pseudocubic natural perovskite crystals for reflectance measurements. Measured two faces to check for anisotropy (Figure 9), but probably took an average of the two faces to get the refractive indices. 

Also on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/tioxides.html
--------------

Exoplanet/Brown Dwarf Papers:

Important since its predicted to show up in brown dwarfs and stars, condenses at 2000K, and depletes atmosphere of TiO gas.

Wakeford (2017)
ADS : https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4247W/abstract

---------------------------------
---------------------------------

Aerosol Name      : Perovskite
Name in POSEIDON  : CaTiO3_KH
Database          : KH18
Wavelengths       : 3.56e-2 - 5.84e3 um

Chemical Formula  : CaTiO[3]
Crystal or Amorph : Crystal 
Crystal Shape     : Orthorhombic (near cubic structure)


Refractive Index References:
--------------
Posch (2003) (2-5.8e5 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2003ApJS..149..437P/abstract

Paper Info : 

Same reference used in WS15 entry.

Also on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/tioxides.html
--------------
Ueda (1998) (0.02-2 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1998JPCM...10.3669U/abstract
Paper Info :

Indices look like they were taken from Figure 3. Reflection spectra in UV. Single crystal rods were cut into the shape of a plate. One side was polished with a diamond and corundum slurry, the other side was roughened with sandpaper to avoid back-scattering reflections. 
--------------

Exoplanet/Brown Dwarf Papers:

Important since its predicted to show up in brown dwarfs and stars, condenses at 2000K, and depletes atmosphere of TiO gas.

Wakeford (2017)
ADS : https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4247W/abstract

#################################
M-L Dwarfs
#################################

These are aerosols that are expected to form in temperatures similar to the atmospheres of M and L dwarfs. These aerosols are usually related to the condensation of Ti or V gas and serve as the 'seed' particles in many GCMs. (TiO and VO gas are very good UVIS absorbers and are though to cause thermal inversions, they can get condensed out and become cold-trapped or form lofted seed particles)

---------------------------------
---------------------------------

Aerosol Name      : Anatase
Name in POSEIDON  : TiO2_anatase
Database          : KH18
Wavelengths       : 1.2e-1 - 5.843e3 um

Chemical Formula  : TiO[2] (technically Ti[0.992]V[0.008]O[2] + TiO[2], see below)
Crystal or Amorph : Crystalline
Crystal Shape     : Tetragonal 

Refractive Index References:
--------------
Zeidler (2011) (0.4-10 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2011A%26A...526A..68Z/abstract

Paper Info :

Measured transmittance spectra of natural anatase for both the extraordinary and ordinary directions. KH18 took a weighted average of these directions (2/3-1/3).

Also on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/tioxides.html
--------------
Posch (2003) (10-6e3 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2003ApJS..149..437P/abstract

Paper Info :

Measured reflectance spectra of natural crystals that were un-annealed from Diamantina, Brazil at room temperature. Extraordinary and ordinary directions were measured. KH18 took a weighted average of these directions (2/3-1/3). Technically Ti[0.992]V[0.008]O[2].

In this paper, they also measured a partially annealed anatase from Hardangervidda Norway to demonstrate how annealed anatase starts to show features of rutile. Notes that anatase is a uniaxial, optically negative crystal (resulting in an oblate spheroid crystal).

Also on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/tioxides.html
--------------
Siefke (2016) (0.1-125 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2016arXiv160704866S/abstract

Paper Info :

Took UVIS measurements of TiO2 thin film, see Figure 5 which matches the txt file. Goal was to test how well TiO2 was as a wire grid polarizer. Due to the thin film method utilized, the TiO2 in this sample has no directionality. Additionally, it doesn't mention that the sample is specifically anatase. 

--------------

Exoplanet/Brown Dwarf Papers:

TiO2 is important as a 'seed' particle that other aerosols can condense onto, and it is expected to condense out alongside CaTiO3 first in brown dwarfs.

Anatase is a low temperature, low pressure modification of TiO2. Anatase transforms into rutile via annealing at around T = 1200K.

Helling (2006) and Helling & Woitke (2006)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Lee (2016), Lee (2017), and Lee 2018 
https://ui.adsabs.harvard.edu/abs/2016A%26A...594A..48L/abstract
https://ui.adsabs.harvard.edu/abs/2017A%26A...601A..22L/abstract
https://ui.adsabs.harvard.edu/abs/2018A%26A...614A.126L/abstract

---------------------------------
---------------------------------

Aerosol Name      : Rutile
Name in POSEIDON  : TiO2_rutile
Database          : gCMCRT
Wavelengths       : 0.47-36.2 um

Chemical Formula  : TiO[2] (technically Ti[0.984]V[0.008]Fe[0.008]O[2] + TiO[2], see below)
Crystal or Amorph : Crystalline 
Crystal Shape     : Tetragonal

Refractive Index References:
--------------
Ribarsky in Palik (1985) [Volume 1, Section 39]

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Zeidler (2011) 

ADS        : https://ui.adsabs.harvard.edu/abs/2011A%26A...526A..68Z/abstract

Paper Info:

Ribarsky in Palik (1985) compiled many different lab sources to tabulate the ordinary and extraordinary indices of rutile.

Zeidler (2011) measured the reflectance spectra of natural rutile (technically Ti[0.984]V[0.008]Fe[0.008]O[2]) to fill in gaps in Ribarsky in Palik (1985). They note that their natural rutile had water inclusions, which could cause extra OH modes around 3 um.

The indices for the rutile txt file are a mesh-mash of both references listed above. Specifically, Zeidler 2011 filled in a gap in imaginary indices from Ribarsky in Palik 1985 from 0.5 to 8 um region.

This text file is specifically composted of the ordinary direction (E⟂c or E||a,b).

Also on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/tioxides.html
--------------

Exoplanet/Brown Dwarf Papers:

TiO2 is important as a 'seed' particle that other aerosols can condense onto, and it is expected to condense out alongside CaTiO3 first in brown dwarfs.

Rutile is most abundant form of TiO2 on Earth, and expected at hotter temperatures.

Helling (2006) and Helling & Woitke (2006)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Lee (2016), Lee (2017), and Lee 2018 
https://ui.adsabs.harvard.edu/abs/2016A%26A...594A..48L/abstract
https://ui.adsabs.harvard.edu/abs/2017A%26A...601A..22L/abstract
https://ui.adsabs.harvard.edu/abs/2018A%26A...614A.126L/abstract

---------------------------------
---------------------------------

Aerosol Name      : Titanium Carbide
Name in POSEIDON  : TiC
Database          : KH18
Wavelengths       : 1.5e-2 - 207 um

Chemical Formula  : TiC
Crystal or Amorph : Crystalline
Crystal Shape     : Cubic (Face-centered)

Note from Dr. Daniel Kitzmann: 

'I do have have file that has the reflectance of the Henning & Mutschke (2001) paper in it. I guess I must have found it somewhere. I don't think that I have taken it directly from the figure because the x-axis strangely is given in angular frequency instead of wavelength. Furthermore, the reflectance in this file is not just the one from the Henning & Mutschke (2001) paper but includes extended data
towards much smaller wavelengths from, presumably, additional data sources. Unfortunately, I can't remember from where I got the file from, though. However, I only used the wavelength range from the Henning & Mutschke paper and converted the stated reflectance into the usual optical constants that were then joined with the data from Koide.'

Refractive Index References:
--------------
Koide (1990) (9e-3 - 0.9 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1990PhRvB..42.4979K/abstract

Paper Info :

Single crystals discs were cut from crystal rods prepared via the floating-zone technique (for reference of this lab technique, see Shigeki (1983); https://ui.adsabs.harvard.edu/abs/1983JCrGr..62..211O/abstract), after which reflectance spectra were made. Paper reports dielectric functions.
--------------
Henning & Dutschke 2001 (1e-4 - 207 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2001AcSpA..57..815H/abstract

Paper Info :

No table or figure of refractive indices found in paper. Reflectance spectroscopy of TiC bulk material, produced by melting commercial TiC powder in a furnace, after which solicited droplets were cut and polished into a plane surface.
--------------

Exoplanet/Brown Dwarf Papers:

Carbides, like TiC and SiC, are very stable (thermally and mechanically), and very hard (high mohs scale). These studies investigated TiC as a potential condensate for carbon-rich outflows of stars. 

---------------------------------
---------------------------------

Aerosol Name      : Vanadium Oxide
Name in POSEIDON  : VO
Database          : gCMCRT
Wavelengths       : 0.3-30 um

Chemical Formula  : VO[2]
Crystal or Amorph : See below 
Crystal Shape     : See below 

Refractive Index References:
--------------
Wan et al (2019)

ADS        : https://ui.adsabs.harvard.edu/abs/2019AnP...53100188W/abstract

Paper Info :

Selected four VO[2] thin film samples. Sample 2, insulating (measurements carried at 303.15K), was used to generate the txt file (Table in Section 6), which had a thickness of 130 nm on a Si+ native oxide substrate. 

VO[2] thin film sample was produced by sputtering V[2]O[5] material, which was then superheated to 973K to convert V[2]O[5] to VO[2].  

VO[2] is monoclinic (uniaxial) and insulating below 340K, and tetragonal (uniaxial) and metallic above. However, the thin film procedure ensured that the VO[2] was randomly oriented. 
--------------

Exoplanet/Brown Dwarf Papers:

Equilibrium calculations have VO be the first V bearing species to condense, however VO[2] and V[2]O[5] are more common on Earth. The above VO[2] indices can be potentially used as a VO proxy.

Though there seems to be debates on whether VO condenses as VO around 1800K, or as a solid solution with Ti-bearing condensates. Also seems like VO can get cold-trapped in hot Jupiter atmospheres.

Lodders (2002) (VO doesn't homogeneously condense)
https://ui.adsabs.harvard.edu/abs/2002ApJ...577..974L/abstract

Burrows & Sharp (1999) (VO does homogeneously condense)
https://ui.adsabs.harvard.edu/abs/1999ApJ...512..843B/abstract

Spiegel (2009) ('vertical' cold trap)
https://ui.adsabs.harvard.edu/abs/2009ApJ...699.1487S/abstract

Parmentier et al. (2013) ('day-night' cold trap)
https://ui.adsabs.harvard.edu/abs/2013A%26A...558A..91P/abstract

---------------------------------
---------------------------------

Aerosol Name      : Meteoritic Nano-Diamonds 
Name in POSEIDON  : NanoDiamonds
Database          : Mullens et al. 2024 
Wavelengths       : 0.02 - 110 um

Chemical Formula  : C
Crystal or Amorph : Crystalline
Crystal Shape     : Cubic (large diamonds can still be anisotropic due to growth patterns)

Refractive Index References:
--------------
Mutschke (2004) 

ADS        : https://ui.adsabs.harvard.edu/abs/2004A%26A...423..983M/abstract
Vizier 	   : https://ui.adsabs.harvard.edu/abs/2004yCat..34230983M/abstract

Paper Info :

Uses a multitude of transmission techniques to measure the optical properties of meteoritic nano-diamonds from 0.12 to 100 um. 

Short wavelength data is calculated from EELs data (electron energy loss spectroscopy) which produces an 'artificial' spectrum from 0.01 to 0.2 um.

Thin film is used from 0.12-0.5 um.

KBr pellets were used from 0.5-100 um.

Originally from the Allende meteorite. Braatz et al. (2000) describes how nano diamonds were taken from Allende sample (https://ui.adsabs.harvard.edu/abs/2000M%26PS...35...75B/abstract). 

Txt file generated from the Vizier page, click the FTP tab.
--------------

Exoplanet/Brown Dwarf Papers:

Diamonds account for 99% of premolar meteoritic material, and are expected to potential form first in stellar outflows (diamonds have been found in spectra of Herbig-A3Be pre main sequence stars). Could potentially form in carbon rich atmospheres. 

#################################
Iron
#################################

Iron is expected to condense out in a homogenous manner since the condensation temperature is around 1500-2300K (the Fe-deck). This is slightly proved by the existence of H2S in Jupiter's atmosphere (FeS would be dominate gas species, not H2S.

Pyroxenes silicate series follow the equation  Mg[x]Fe[1-x]SiO3 with a Mg rich end-member (MgSiO[3], Enstatite) and a Fe rich end-member [FeSiO[3], Ferrosilite].

Olivines follow the equation Mg[2y]Fe[2-2y]SiO4 with a Mg rich end-member (Mg[2]SiO[4], Forsterite) and a Fe rich end-member [Fe[2]SiO[4], Fayalite].

---------------------------------
---------------------------------

Aerosol Name      : Alpha Iron 
Name in POSEIDON  : Fe
Database          : KH18
Wavelengths       : 1.24e-4 - 286 um

Chemical Formula  : Fe
Crystal or Amorph : Crystalline
Crystal Shape     : Cubic (body-centered, depends on allotrope) (still anisotropic, see below)

Refractive Index References:
--------------
Lynch and Hunter in Palik (1991) [Volume 2, Chapter 15]

ADS        : https://ui.adsabs.harvard.edu/abs/1991hocs.book.....P/abstract

Paper Info :

Collection of room temperature lab measurements of alpha iron (presumed alpha since measurements were taken at room temperature). Even though iron is cubic, alpha iron is ferromagnetic below 1040K, which makes it anisotropic. This effect is minimal unless there is an applied magnetic field. 

Iron has many allotropes. Normal pressures and high temperatures forms sigma iron (body centered cubic) at 1811K, which then forms gamma iron (face centered cubic) at 1667K, and then alpha iron below 1185K. High pressure iron forms epsilon and beta iron. 

Above 1040K, alpha iron is paramagnetic. 

--------------

Exoplanet/Brown Dwarf Papers:

In many gas giants and brown dwarfs, iron is expected to homogeneously condense as a deep 'deck'. Acts as a mostly opaque deck since Fe has no distinguishable features, also depletes upper atmosphere of Fe. 

Visscher (2010) (Thermochemical Eq predictions of iron, magnesium, and silicon species)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

Fegley and Lodders (1996)(Thermochemical Eq Predictions of gases in a brown dwarf, shows Fe-deck)
https://ui.adsabs.harvard.edu/abs/1996ApJ...472L..37F/abstract

Burningham (2021) (Retrieval with an iron deck included)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

---------------------------------
---------------------------------

Aerosol Name      : Wustite
Name in POSEIDON  : FeO
Database          : WS15
Wavelengths       : 0.2-42 um

Chemical Formula  : FeO
Crystal or Amorph : Crystalline (assumed, see below)
Crystal Shape     : Cubic (face-centered cubic)

Refractive Index References:
--------------
Begemann (1995) 

ADS        : https://ui.adsabs.harvard.edu/abs/1995P%26SS...43.1257B/abstract

Paper Info :

Laboratory prepared magnesiowusite (Fe[x]Mg[1-x]O) samples. Formed by quenching a melt of FeO in an argon atmosphere to prevent oxidation of Fe from ferrous to ferric. Never is explicitly stated to be crystalline or amorphous, but can be assumed crystalline. Pellets of FeO were embedded in epoxy-resin and polished. Optical constants were derived from IR reflection analysis. 

Txt file is slightly updated from WS15. Txt file was remade solely using the indices from Table 2 of Begemann (1995). 
--------------

Exoplanet/Brown Dwarf Papers:

Could condense in oxygen-rich envelopes around M stars. Expected to form in oxygen rich, silicate poor, and iron rich environments. 

---------------------------------
---------------------------------

Aerosol Name      : Troilite
Name in POSEIDON  : FeS
Database          : KH18
Wavelengths       : 0.1-487 um

Chemical Formula  : FeS
Crystal or Amorph : Crystalline (assumed, see below)
Crystal Shape     : Hexagonal (possibly isotropic due to crystal growth as slabs)

This was a pain to back track references for...

Note from KH18 Paper to keep in mind: 
‘For troilite we combine the two data sets of Pollack et al. (1994) and Henning & Mutschke (1997). The compilation of Pollack et al. (1994) used a few experimental data points in combination with extrapolations to account for the missing data. Especially in the infrared region, this set is not consistent with the measured data provided by Henning & Mutschke (1997). We therefore replace the IR data with the latter one and use the Kramers-Kronig relations to consistently combine the IR part with the measured shortwave data provided in Pollack et al. (1994). This leads to a shift of the strong iron feature near 3 µm to slightly larger wavelengths compared to the original compilation provided by Pollack et al. (1994).’

Refractive Index References:
--------------
Pollack (1994) (0.1 to 1e4 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1994ApJ...421..615P/abstract

Paper Info :

ISM paper looked at potential compositions of molecular cloud grains. FeS forms between H2S and Fe at 680K, independent of pressure. The txt file matches Figure 1 in the paper. 

Do note that the authors compile their FeS indices from many sources, that included experimental data and extrapolations (see their Appendix B5 and table for references). More specifically:

1. In the visible FeS displays quasi-metallic behavior and therefore they extrapolated the imaginary index down to 0.1 um and up to 3 um by scaling visible values by square root of lambda. (Egan & Hilgeman, https://ui.adsabs.harvard.edu/abs/1977Icar...30..413E/abstract) who measured a troillite sample from Del Norte County, California (U. S. National Museum Sample No. 94472) (crystalline) (https://ui.adsabs.harvard.edu/abs/1975AJ.....80..587E/abstract).

2. Used KK extrapolation to get 1 to 15 um

3. The rest are microwave and radio. Note that 15 to 62.5 is interpolated. (radio data is actually pyrrhotite which is Fe[7]S[8], Beblo 1982 in the textbook Landolt-Bornstein) (radio and microwave references, https://ui.adsabs.harvard.edu/abs/1985ApJ...290L..41N/abstract and https://ui.adsabs.harvard.edu/abs/1982lbg6.conf.....A/abstract)

KH18 replaced their IR data with the one found in the next reference.
--------------
Henning & Mutschke (1997) (20-487 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1997A%26A...327..743H/abstract

Paper Info : 

Looked at temperature dependent data of ISM aerosols (300, 200, 100, and 10K). Due to overlapping wavelengths with the other file, and KH18's use of Kramers-Kronig, I am unsure of which temperature was used, but probably the hottest one (300K). 

FeS was prepared by melting in an arc furnace and quenching. Samples were then cut from a silica tube. These processes is described in both Begemann (1994) and Begemann (1993) (https://ui.adsabs.harvard.edu/abs/1994ApJ...423L..71B/abstract, https://ui.adsabs.harvard.edu/abs/1993AGAb....9..164M/abstract) where Begemann (1994) has refractive indices for many Mg-Fe sulfides. 

Reflection measurements were made of FeS embedded into an epoxy resin and polished, transmission measurements were made by grounding FeS into small balls and embedding them in KBr and PE pellets. 

This paper is unclear on whether it is amorphous or crystalline, but Figure 1b has a lot of fine structure, which leads me to believe that it is crystalline. 

Also on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/sulfides.html
--------------

Exoplanet/Brown Dwarf Papers:

FeS is the iron-rich end member of the pyrrhotite group (Fe[1-x]S) and is the only member that is not magnetic. 

Is a semiconductor and thought to possibly be a component of the rings of Saturn. 

Egan & Hilgeman (1977) (Saturn rings and FeS)
https://ui.adsabs.harvard.edu/abs/1977Icar...30..413E/abstract

The crystal growth of troillite is rapid and forms layers. These layers can make troillite isotropic. 

Levi (1994) (Troillite formation)
https://ui.adsabs.harvard.edu/abs/1994Metic..29R.490L/abstract

---------------------------------
---------------------------------

Aerosol Name      : Hematite (alpha Fe2O3) (assumed, see below)
Name in POSEIDON  : Fe2O3
Database          : WS15
Wavelengths       : 0.1 - 987 um

Chemical Formula  : Fe[2]O[3]
Crystal or Amorph : Crystalline (assumed, see below) 
Crystal Shape     : Rhombohedral (assumed, see below) 

Refractive Index References:
--------------
Unpublished, created by Amaury H.M.J. Triaud 

Found on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/mgfeoxides.html

The website only includes the ordinary ray (E||a,b), which matches the txt file.

No information on phase. However, alpha-Fe2O3 (hematite) is rhombohedral, gamma-Fe2O3 is cubic, and epsilon-Fe2O3 is orthorhombic. Alpha is the most common form of Fe2O3 and rhombohedral would form a uniaxial crystal, necessitating the distinction that the indices are ordinary. 

--------------

Exoplanet/Brown Dwarf Papers:

Fe2O3 is not expected to form in planetary atmospheres since the homogenous formation of the Fe deck is expected to deplete the atmosphere of iron. Fe2O3 is only expected to form in highly oxidizing atmospheres or atmospheres where Fe gas doesn't get depleted into a deck. 

---------------------------------
---------------------------------

Aerosol Name      : Ferrosilite (Iron-rich silicate pyroxene)
Name in POSEIDON  : FeSiO3
Database          : WS15
Wavelengths       : 8.2-33 um

Chemical Formula  : FeSiO[3]
Crystal or Amorph : Amorphous 
Crystal Shape     : N/A

Refractive Index References:
--------------
Day (1981)

ADS        : https://ui.adsabs.harvard.edu/abs/1981ApJ...246..110D/abstract

Paper Info :

Thin films produced by sputtering cathode of FeSi and Fe2Si onto KBR in a chamber of argon gas. Performed transmission measurements. No table in the paper, therefore indices were most likely digitized from Figure 2. 
--------------

Exoplanet/Brown Dwarf Papers:

Fe is expected to form homogeneously in a Fe deck, so will need to find a way to have silicate and iron gas exist together. 

---------------------------------
---------------------------------

Aerosol Name      : Fayalite (Iron-rich olivine)
Name in POSEIDON  : Fe2SiO4_KH
Database          : KH18
Wavelengths       : 0.4-1e4 um

Chemical Formula  : Fe[2]SiO[4]
Crystal or Amorph : Crystalline 
Crystal Shape     : Orthorhombic 

Refractive Index References:
--------------
Fabian 2001  (0.4-1e4 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2001A%26A...378..228F/abstract

Paper Info :

Paper is specifically looking at crystalline olivines. Looked at the reflection spectra of single, synthetic crystals. Single crystals are grown using the scull method where 1-2 kg of polycrystalline fayalite was inductively molten and slowly cooled under a defined oxygen partial pressure. 6mm crystals. Confirmed to be entirely fayalite with no inclusions. Figure 8 shows how opacity changes with different Mie assumptions (sphere, CDE1, CDE2, powder)

Hard to compare txt file directly with indices on DOCCD since KH18 average based on direction (each direction gets a 1/3 weighting). 

Found on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/crsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Fe is expected to form homogeneously in a Fe deck, so will need to find a way to have silicate and iron gas exist together. 

#################################
Magnesium 
#################################

Magnesium silicates are very common in brown dwarf and exoplanet literature. They are expected to condense out above the Fe-deck assuming equilibrium cloud formation as forms of forsterite and enstatite. 

Magnesium is the assumed limiting element. Forsterite will form first, and then enstatite. Pyroxenes and olivine with iron inclusions can form if Fe isn't rained out. Assuming a solar Mg/Si abundance, Mg2SiO4 formation consumes nearly half of the total Si abundance because the solar elemental abundances of Mg and Si are approximately equal.

Alumina-magnesium species, like spinel, will only form if aluminum gas isn't depleted by condensation of aluminum oxides like corundum. 

Expected to form in observable layers of L-dwarfs. As brown dwarfs continue to cool to 1400K (T-dwarf), the clouds start to dissipate (patchy clouds) and sink to deeper in the atmosphere. 

Pyroxenes silicate series follow the equation  Mg[x]Fe[1-x]SiO3 with a Mg rich end-member (MgSiO[3], Enstatite) and a Fe rich end-member [FeSiO[3], Ferrosilite].

Olivines follow the equation Mg[2y]Fe[2-2y]SiO4 with a Mg rich end-member (Mg[2]SiO[4], Forsterite) and a Fe rich end-member [Fe[2]SiO[4], Fayalite]. Think 'Four' for Forsterite! 

Sudarsky (2003) (See section 3.2 for good tea on clouds)
https://ui.adsabs.harvard.edu/abs/2003ApJ...588.1121S/abstract

Olivines follow the equation Mg[2y]Fe[2-2y]SiO4 with a Mg rich end-member (Mg[2]SiO[4], Forsterite) and a Fe rich end-member [Fe[2]SiO[4], Fayalite].

Visscher (2010) (Thermochemical Eq predictions of iron, magnesium, and silicon species)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

Burningham (2021) (Retrieval with magnesium species)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

---------------------------------
---------------------------------

Aerosol Name      : Periclase
Name in POSEIDON  : MgO
Database          : KH18
Wavelengths       : 1.65e-2 - 625 um

Chemical Formula  : MgO
Crystal or Amorph : Crystalline (assumed, see below)
Crystal Shape     : Cubic

Refractive Index References:
--------------
Roessler & Huffman in Palik (1991) [Volume 2, Section 46]

ADS        : https://ui.adsabs.harvard.edu/abs/1991hocs.book.....P/abstract

Paper Info :

Found in nature as periclase (old name is beta-magnesia). Cubic structure (similar to halite/rock salt, NaCl). In nature, readily forms a solution with FeO (wustite).

Palik (1991) collects data from many lab sources. Most of it seems to be crystalline, even though there is some thin film measurements. Txt file matches the table in section 46. Whenever the imaginary indices column was empty, KH18 used Kramers-Kronig to derive them from the real indices. 
--------------

Exoplanet/Brown Dwarf Papers:

Condenses via the reaction Mg + H2O = MgO(s) + H2

Only expected to condense at high pressures (see Figure 4 in following paper) 

Visscher (2010)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Forsterite (Iron-rich)
Name in POSEIDON  : Mg2SiO4_Fe_rich
Database          : WS15
Wavelengths       : 0.21 - 446 um

Chemical Formula  : Mg[0.8]Fe[1.2]SiO[4] (assumed, see below)
Crystal or Amorph : Amorphous (glass) (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Henning (2005)

ADS        : https://ui.adsabs.harvard.edu/abs/2005IAUS..231..457H/abstract

Paper Info :

Review article that was cited in WS15. Indices were most likely taken from Figure 5 with a plot digitizer. Note that this figure is in wavenumber. Comparing and contrasting the txt file with the ones found in the DOCCD, it seems like this entry is most similar to Mg[0.8]Fe[1.2]SiO[4] glass from Dorschner (1995). 

--------------
Dorschner (1995) (assumed reference)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

Found on DOCCD [under, Mg(0.8)Fe(1.2)SiO4]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Burningham (2021) (Retrieval with amorphous species)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

Forsterite is predicted to be the first silicate to form in brown dwarf atmospheres. 

Visscher (2010) (Thermochemical Eq predictions for forsterite condensation)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Forsterite (Iron-poor)
Name in POSEIDON  : Mg2SiO4_Fe_poor
Database          : WS15
Wavelengths       : 0.2 - 8190 um

Chemical Formula  : Mg[1.72]Fe[0.21]SiO[4]
Crystal or Amorph : Crystalline 
Crystal Shape     : Orthorhombic

Refractive Index References:
--------------
Zeidler (2011)

ADS        : https://ui.adsabs.harvard.edu/abs/2011A%26A...526A..68Z/abstract

Paper Info :

The paper compares Fe-poor natural olivine from San Carlos to Fe-rich natural olivine from Ratnapura Sri Lanka, both crystalline. Use both reflectance and absorption. The Fe-poor natural olivine from San Carlos was used to make this dataset. 

The San Carlos indices were made temperature dependent in Zeidler 2015 and updated on the website. Comparing the txt file from WS15 to the datasets on the website, it looks like E||c/x at room temperature matches the most. 

Also on DOCCD [See San Carlos Olivine in the VIS-NIR and San-Carlos at 300K E||c/x]
https://www2.astro.uni-jena.de/Laboratory/OCDB/crsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

It is expected that silicates condense in a crystalline state at high temperatures and amorphous states at lower temperatures, where amorphous particles can transition to a crystalline state by being exposed to high temperatures for a set period of time (i.e. annealing). 

Therefore, it can be reasonable to predict that crystalline forms of silicates are present in hot Jupiters. 

Tsuchiyama (1998) (silicates are crystalline at high temp)
https://ui.adsabs.harvard.edu/abs/1998MinJ...20...59T/abstract

Fabian (2000) (silicates are amorphous at lower temps, and crystalline after annealing)
https://ui.adsabs.harvard.edu/abs/2000A%26A...364..282F/abstract

Forsterite is predicted to be the first silicate to form in brown dwarf atmospheres. 

Visscher (2010) (Thermochemical Eq predictions for forsterite condensation)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Forsterite 
Name in POSEIDON  : Mg2SiO4_amorph
Database          : B21
Wavelengths       : 0.268 - 227.531

Chemical Formula  : Mg[2]SiO[4]
Crystal or Amorph : Amorphous
Crystal Shape     : N/A

Refractive Index References:
--------------
Scott & Duley (1996)

ADS        : https://ui.adsabs.harvard.edu/abs/1996ApJS..105..401S/abstract
Paper Info :

Thin films of amorphous magnesium silicates with compositions similar to enstatite and forsterite are deposited using excimer laser ablation of parent materials. Refractive indices are derived from optical transmission and reflection together with Kramers-Kronig analysis. Measured indices are reported for 0.12-17.5 um and extended to short and longer wavelengths by fitting data reported from other experiments and theoretical predictions. 

Sample for forsterite was made by 308 mm excimer laser ablation of geological, natural samples of polycrystalline forsterite. 

Measured imaginary indices and used Kramers-Kronig to derive the real indices, and then iterated using a numerical procedure to obtain consistent sets of n and k. 

Refractive indices are from Table 1, which is in eV. 

1. Scott & Duley measurements are reported from 0.07-0.43 eV (17.5 um - 2.88 um) and 4.7-10 eV (0.26 um - 0.12 um)

2. Astronomical silicate from Draine & Lee (1984) from 0-0.07 eV (61.72-20.6 um), and >20 eV (<0.06 um)

3. Reflection of crystalline forsterite (Nitsan & Shankland 1976) (10-20 eV, 0.12-0.06 um) 

Where Draine & Lee (1984) and Nitsan & Shankland (1976) were normalized to Scott & Daley imaginary indices at 0.07 and 10 eV.

Note that the table runs from 0.06 to 62 um, and the txt file runs from 0.268 to 227 um. Any wavelengths beyond 62 um were interpolated in order to work with the EGP wavelength grid. 
--------------
Draine & Lee (1984)

ADS        : https://ui.adsabs.harvard.edu/abs/1984ApJ...285...89D/abstract

Paper Info :

Created an astronomical, infrared signal for a silicate from observations alone ('astronomical silicate'). 

‘Although the optical constants of olivine (Huffman and Stapp 1973) are often used to represent those of interstellar silicates in current grain models (e.g., MRN; Hong and Greenberg 1980), the observed interstellar features at 9.7 and 18 um are substantially broader than the observed resonances in terrestrial and lunar silicates.’

‘In order to obtain a dielectric function for “astronomical silicate,” it has therefore been necessary to construct the infrared portion, using observational data to constrain the dielectric function insofar as possible.’

Data greater than 6 um, which was used in Scott & Duley, is given by a collection of 33 Lorentz oscillator functions fit to observational data. See section 2.c in their paper for more details.
--------------
Nitsan & Shankland (1976)

ADS        : https://ui.adsabs.harvard.edu/abs/1976GeoJ...45...59N/abstract

Paper Info :

Measurements made at 300K. This paper only has measurements from 4 to 15 eV, so I guess they extrapolated from 15 to 20 eV to keep it smooth. Reflectance spectrum on synthetic forsterite crystal on the 010 surface (E||c and E||a polarizations).
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Burningham (2021) (Retrieval with amorphous species)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

Forsterite is predicted to be the first silicate to form in brown dwarf atmospheres. 

Visscher (2010) (Thermochemical Eq predictions for forsterite condensation)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Forsterite
Name in POSEIDON  : Mg2SiO4_amorph_sol_gel
Database          : KH18
Wavelengths       : 0.19 - 948 um

Chemical Formula  : Mg[2]SiO[4]
Crystal or Amorph : Amorphous (sol-gel)
Crystal Shape     : N/A

Refractive Index References:
--------------
Jager (2003) 

ADS        : https://ui.adsabs.harvard.edu/abs/2003A%26A...408..193J/abstract
Paper Info :

Presents optical constants of pure, amorphous Mg-silicates. Used sol-gel, a chemical technique based on the condensation of Mg- and Si-hydroxides in a liquid phase (crystallize at lower temperatures for some reason). They use sol gel, remove Mg-Si particles, and then densify them to remove porosity. Amorphousness is determined via Xray. For an explanation, see section 2 of this paper.

Figure 5 shows a comparison of sol-gel and thin-films from Scott and Duley (1996), which was used to make amorphous Mg2SiO4 from B21. Also shows in Figure 9 what assuming an iron core to the sol-gel does to spectra.

Also in DOCCD (Mg(2)SiO(4) matches txt file, in wavenumber on website)
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Burningham (2021) (Retrieval with amorphous species)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

Forsterite is predicted to be the first silicate to form in brown dwarf atmospheres. 

Visscher (2010) (Thermochemical Eq predictions for forsterite condensation)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Forsterite 
Name in POSEIDON  : Mg2SiO4_crystalline
Database          : gCMCRT
Wavelengths       : 0.1 - 1000 um

Chemical Formula  : Mg[2]SiO[4]
Crystal or Amorph : Crystalline
Crystal Shape     : Orthorhombic

Refractive Index References:
--------------
Suto (2006)

ADS        : https://ui.adsabs.harvard.edu/abs/2006MNRAS.370.1599S/abstract

Paper Info :

Measured infrared reflectance of low temperature (50-295K) crystalline forsterite along each crystallographic axis. Paper focuses on fitting dielectric constants, which can be converted to refractive index. 

Forsterite synthetic, single crystals were grown using the Czochralski method (described in Takei (1978), https://ui.adsabs.harvard.edu/abs/1978JCrGr..43..463T/abstract). Resulted in centimeter sized crystals, polished on each face.

gCMCRT received the indices from the ProDiMo team (Woitke (2009)), which is a disc code. Comparing the txt file to directional data on DOCCD, it looks like E||a and E||b were used. 

--------------

Exoplanet/Brown Dwarf Papers:

It is expected that silicates condense in a crystalline state at high temperatures and amorphous states at lower temperatures, where amorphous particles can transition to a crystalline state by being exposed to high temperatures for a set period of time (i.e. annealing). 

Therefore, it can be reasonable to predict that crystalline forms of silicates are present in hot Jupiters. 

Tsuchiyama (1998) (silicates are crystalline at high temp)
https://ui.adsabs.harvard.edu/abs/1998MinJ...20...59T/abstract

Fabian (2000) (silicates are amorphous at lower temps, and crystalline after annealing)
https://ui.adsabs.harvard.edu/abs/2000A%26A...364..282F/abstract

Forsterite is predicted to be the first silicate to form in brown dwarf atmospheres. 

Visscher (2010) (Thermochemical Eq predictions for forsterite condensation)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Olivine
Name in POSEIDON  : MgFeSiO4_amorph_glass
Database          : KH18
Wavelengths       : 0.2 - 500 um

Chemical Formula  : MgFeSiO[4]
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Dorschner (1995)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

y = 0.5 entry in Table 5

Found on DOCCD [MgFeSiO[4] [3.71 g/ccm]]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Olivines with iron inclusions can form if iron isn't completely depleted via homogenous condensation and rain out.

Sudarsky (2003) (See section 3.2)
https://ui.adsabs.harvard.edu/abs/2003ApJ...588.1121S/abstract

---------------------------------
---------------------------------

Aerosol Name      : Olivine
Name in POSEIDON  : Mg8Fe12SiO4_amorph_glass
Database          : KH18
Wavelengths       : 0.2 - 500 um

Chemical Formula  : Mg[0.8]Fe[1.2]SiO[4]
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Dorschner (1995)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

y = 0.4 entry in Table 5

Found on DOCCD [Mg(0.8)Fe(1.2)SiO4]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Olivines with iron inclusions can form if iron isn't completely depleted via homogenous condensation and rain out.

Sudarsky (2003) (See section 3.2)
https://ui.adsabs.harvard.edu/abs/2003ApJ...588.1121S/abstract

---------------------------------
---------------------------------

Aerosol Name      : Enstatite
Name in POSEIDON  : MgSiO3
Database          : WS15
Wavelengths       : 0.185 - 80 um

Chemical Formula  : MgSiO[3]
Crystal or Amorph : Mixed (see below)
Crystal Shape     : Orthorhombic

Refractive Index References:
--------------
Egan & Hilgeman (1975) (0.1-0.4 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1975AJ.....80..587E/abstract

Paper Info :

Focused on UV measurements of many silicate species. Got a sample of enstatite from a supplier in India that was originally in Huffman & Stamp (1971) (https://ui.adsabs.harvard.edu/abs/1971NPhS..229...45H/abstract). Their samples was natural, brown enstatite which means it was probably crystalline, but no polarization was mentioned.

Measured the thickness, transmission, and reflection to get refractive indices. 

Txt file matches Table 3, enstatite column.
--------------
Dorschner et al 1995 (0.5-80 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

x = 1 entry in Table 4

Found on DOCCD [MgSiO(3) [2/71 g/ccm]]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

It is expected that silicates condense in a crystalline state at high temperatures and amorphous states at lower temperatures, where amorphous particles can transition to a crystalline state by being exposed to high temperatures for a set period of time (i.e. annealing). 

Tsuchiyama (1998) (silicates are crystalline at high temp)
https://ui.adsabs.harvard.edu/abs/1998MinJ...20...59T/abstract

Fabian (2000) (silicates are amorphous at lower temps, and crystalline after annealing)
https://ui.adsabs.harvard.edu/abs/2000A%26A...364..282F/abstract

Enstatite is expected to form after forsterite, but around the same temperature and pressure.

Visscher (2010) (Thermochemical Eq predictions of enstatite)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Enstatite
Name in POSEIDON  : MgSiO3_amorph
Database          : B21
Wavelengths       : 0.268 - 227.531

Chemical Formula  : MgSiO[3]
Crystal or Amorph : Amorphous
Crystal Shape     : N/A

Refractive Index References:
--------------
Scott & Duley (1996)

ADS        : https://ui.adsabs.harvard.edu/abs/1996ApJS..105..401S/abstract
Paper Info :

Thin films of amorphous magnesium silicates with compositions similar to enstatite and forsterite are deposited using excimer laser ablation of parent materials. Refractive indices are derived from optical transmission and reflection together with Kramers-Kronig analysis. Measured indices are reported for 0.12-17.5 um and extended to short and longer wavelengths by fitting data reported from other experiments and theoretical predictions. 

Sample for enstatite was made by 308 mm excimer laser ablation of geological, natural samples of polycrystalline forsterite. 

Measured imaginary indices and used Kramers-Kronig to derive the real indices, and then iterated using a numerical procedure to obtain consistent sets of n and k. 

Refractive indices are from Table 1, which is in eV. 

1. Scott & Duley measurements are reported from 0.07-0.43 eV (17.5 um - 2.88 um) and 4.7-10 eV (0.26 um - 0.12 um)

2. Astronomical silicate from Draine & Lee (1984) from 0-0.07 eV (61.72-20.6 um), and >20 eV (<0.06 um)

3. Reflection of crystalline forsterite (Nitsan & Shankland 1976) (10-20 eV, 0.12-0.06 um) 

Where Draine & Lee (1984) and Nitsan & Shankland (1976) were normalized to Scott & Daley imaginary indices at 0.07 and 10 eV.

Note that the table runs from 0.06 to 62 um, and the txt file runs from 0.268 to 227 um. Any wavelengths beyond 62 um were interpolated in order to work with the EGP wavelength grid. 
--------------
Draine & Lee (1984)

ADS        : https://ui.adsabs.harvard.edu/abs/1984ApJ...285...89D/abstract

Paper Info :

Created an astronomical, infrared signal for a silicate from observations alone ('astronomical silicate'). 

‘Although the optical constants of olivine (Huffman and Stapp 1973) are often used to represent those of interstellar silicates in current grain models (e.g., MRN; Hong and Greenberg 1980), the observed interstellar features at 9.7 and 18 um are substantially broader than the observed resonances in terrestrial and lunar silicates.’

‘In order to obtain a dielectric function for “astronomical silicate,” it has therefore been necessary to construct the infrared portion, using observational data to constrain the dielectric function insofar as possible.’

Data greater than 6 um, which was used in Scott & Duley, is given by a collection of 33 Lorentz oscillator functions fit to observational data. See section 2.c in their paper for more details.
--------------
Nitsan & Shankland (1976)

ADS        : https://ui.adsabs.harvard.edu/abs/1976GeoJ...45...59N/abstract

Paper Info :

Measurements made at 300K. This paper only has measurements from 4 to 15 eV, so I guess they extrapolated from 15 to 20 eV to keep it smooth. Reflectance spectrum on synthetic forsterite crystal on the 010 surface (E||c and E||a polarizations).
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Burningham (2021) (Retrieval with amorphous species)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

Enstatite is expected to form after forsterite, but around the same temperature and pressure.

Visscher (2010) (Thermochemical Eq predictions of enstatite)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Enstatite
Name in POSEIDON  : MgSiO3_amorph_glass
Database          : KH18
Wavelengths       : 0.2 - 500 um

Chemical Formula  : MgSiO[3]
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Dorschner (1995)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

x = 1 entry in Table 4

Found on DOCCD [MgSiO(3) [2/71 g/ccm]]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Burningham (2021) (Retrieval with amorphous species)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

Enstatite is expected to form after forsterite, but around the same temperature and pressure.

Visscher (2010) (Thermochemical Eq predictions of enstatite)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Enstatite
Name in POSEIDON  : MgSiO3_sol_gel
Database          : KH18
Wavelengths       : 0.196 - 1000 um

Chemical Formula  : MgSiO[3]
Crystal or Amorph : Amorphous (sol-gel) 
Crystal Shape     : N/A

Refractive Index References:
--------------
Jager (2003) 

ADS        : https://ui.adsabs.harvard.edu/abs/2003A%26A...408..193J/abstract
Paper Info :

Presents optical constants of pure, amorphous Mg-silicates. Used sol-gel, a chemical technique based on the condensation of Mg- and Si-hydroxides in a liquid phase (crystallize at lower temperatures for some reason). They use sol gel, remove Mg-Si particles, and then densify them to remove porosity. Amorphousness is determined via Xray. For an explanation, see section 2 of this paper.

Also in DOCCD (MgSiO(3) matches txt file, in wavenumber on website)
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Burningham (2021) (Retrieval with amorphous species)
https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1944B/abstract

Enstatite is expected to form after forsterite, but around the same temperature and pressure.

Visscher (2010) (Thermochemical Eq predictions of enstatite)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Enstatite (ortho-enstatite) 
Name in POSEIDON  : MgSiO3_crystalline
Database          : B21
Wavelengths       : 0.268 - 227.531 um

Chemical Formula  : MgSiO[3]
Crystal or Amorph : Crystalline 
Crystal Shape     : Orthorhombic

Refractive Index References:
--------------
Jager (1998)

ADS        : https://ui.adsabs.harvard.edu/abs/1998A%26A...339..904J/abstract

Paper Info :

The crystalline revolution OoO.

Mass absorption coefficients were derived from transmission spectra of small grains embedded in KBr and polyethylene pellets. Except in the case of enstatite. 

For enstatite, a natural monocrystalline samples was available. They found optical constants for each crystallographic direction. B21 took a mean over the three crystallographic directions. 

Because it was natural enstatite, they found inclusions of Al2O3 in the sample. They also note that this is ortho-enstatite, which is more common in nature (ortho refers to orthorhombic crystal shape). They also took measurements of synthetic clino-enstatite (monoclinic), which has similar spectra (only forms in extreme environments).

Also because it was natural they found the enstatite experienced weathering which forms talc on the surface that can induce new features. 

Refractive indices for all three directions (figures 4A-C).

Also on DOCCD (first entry, Enstatite (natural))
https://www2.astro.uni-jena.de/Laboratory/OCDB/crsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

It is expected that silicates condense in a crystalline state at high temperatures and amorphous states at lower temperatures, where amorphous particles can transition to a crystalline state by being exposed to high temperatures for a set period of time (i.e. annealing). 

Tsuchiyama (1998) (silicates are crystalline at high temp)
https://ui.adsabs.harvard.edu/abs/1998MinJ...20...59T/abstract

Fabian (2000) (silicates are amorphous at lower temps, and crystalline after annealing)
https://ui.adsabs.harvard.edu/abs/2000A%26A...364..282F/abstract

Enstatite is expected to form after forsterite, but around the same temperature and pressure.

Visscher (2010) (Thermochemical Eq predictions of enstatite)
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

---------------------------------
---------------------------------

Aerosol Name      : Pyroxene
Name in POSEIDON  : Mg4Fe6SiO3_amorph_glass
Database          : KH18
Wavelengths       : 0.2 - 500 um

Chemical Formula  : Mg[0.4]Fe[0.6]SiO3
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Dorschner (1995)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

x = 0.4 entry in Table 5

Found on DOCCD [Mg(0.4)Fe(0.6)SIO(3)]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Pyroxenes with iron inclusions can form if iron isn't completely depleted via homogenous condensation and rain out.

Sudarsky (2003) (See section 3.2)
https://ui.adsabs.harvard.edu/abs/2003ApJ...588.1121S/abstract

---------------------------------
---------------------------------

Aerosol Name      : Pyroxene
Name in POSEIDON  : Mg5Fe5SiO3_amorph_glass
Database          : KH18
Wavelengths       : 0.2 - 500 um

Chemical Formula  : Mg[0.5]Fe[0.5]SiO[3]
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Dorschner (1995)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

x = 0.5 entry in Table 5

Found on DOCCD [Mg(0.5)Fe(0.5)SIO(3) [3.2  g/ccm])]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Pyroxenes with iron inclusions can form if iron isn't completely depleted via homogenous condensation and rain out.

Sudarsky (2003) (See section 3.2)
https://ui.adsabs.harvard.edu/abs/2003ApJ...588.1121S/abstract

---------------------------------
---------------------------------

Aerosol Name      : Pyroxene
Name in POSEIDON  : Mg8Fe2SiO3_amorph_glass
Database          : KH18
Wavelengths       : 0.2 - 500 um

Chemical Formula  : Mg[0.8]Fe[0.2]SiO[3]
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Dorschner (1995)

ADS        : https://ui.adsabs.harvard.edu/abs/1995A%26A...300..503D/abstract

Paper Info :

Explored many different olivine and pyroxenes. Reflectance and ellipsometric measurements measurements on polished surfaces, transmittance measurements of thin slabs. Glasses were made by quenching a melt. Their glasses required microscopic homogeneity, and any samples showing crystallization (specifically, iron-rich members of olivine) were thrown out. Preparation of samples is more detailed in paper 1 (Jager 1994, https://ui.adsabs.harvard.edu/abs/1994A%26A...292..641J/abstract). 

x = 0.8 entry in Table 4

Found on DOCCD [Mg(0.8)Fe(0.2)SIO(3)]
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------

Exoplanet/Brown Dwarf Papers:

Amorphous grains have been predicted to form in brown dwarfs, shown to fit some brown dwarf silicate features better due to their broadness. 

Helling 2006 (models predict amorphous solids)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract

Pyroxenes with iron inclusions can form if iron isn't completely depleted via homogenous condensation and rain out.

Sudarsky (2003) (See section 3.2)
https://ui.adsabs.harvard.edu/abs/2003ApJ...588.1121S/abstract

---------------------------------
---------------------------------

Aerosol Name      : Spinel (Disordered, annealed)
Name in POSEIDON  : MgAl2O4
Database          : WS15
Wavelengths       : 1.69 - 267 um

Chemical Formula  : MgAl[2]O[4] (technically, Mg[1.02]Al[1.93]Si[0.03]Fe[0.01]Cr[0.01]O[4])
Crystal or Amorph : Crystalline (assumed, see below)
Crystal Shape     : Cubic 

Refractive Index References:
--------------
Fabian et al. (2001) (assumed)

ADS        : https://ui.adsabs.harvard.edu/abs/2001A%26A...373.1125F/abstract

Paper Info :

The original WS15 paper cited the DOCCD database as the source of the spinel indices. The website has undergone updates since 2015, however it looks like the indices are from the 'Natural and natural-thermal-processed Mg-spinels (crystalline) (Fabian et al. 2001)' entry, specifically the annealed at 1h, 1223K data, by comparing the plots there with Figure 1 in WS15. 

It is assumed that WS15 used a plot digitizer, since the txt file doesn't match any txt file on the website. 

Synthesized a variety of nonstoichiometric spinels, which are available on DOCCD, and investigated natural spinel before and after thermal annealing. 

The natural crystal is a dark red, octahedral crystal from Burma. Through EDX, found it was actually Mg[1.02]Al[1.93]Si[0.03]Fe[0.01]Cr[0.01]O[4]. The inclusions shouldn't affect the mid-IR spectra. 

Reflectance, mid-infrared spectra of all samples. Natural spinel was annealed at 1223K for one hour in order to induce a phase transition from ordered to disordered spinel. 

Also on DOCCD
https://www2.astro.uni-jena.de/Laboratory/OCDB/aloxides.html
--------------

Exoplanet/Brown Dwarf Papers:

Alumina-magnesium species, like spinel, will only form if aluminum gas isn't depleted by condensation of aluminum oxides like corundum. 

Sudarsky (2003) (See section 3.2)
https://ui.adsabs.harvard.edu/abs/2003ApJ...588.1121S/abstract

---------------------------------
---------------------------------

Just to recap, from Dorschner (1995) we have 

MgSiO3 glassy in both WS15 (which combines two datasets) and KH18
Mg(0.8)Fe(0.2)SiO(3) in KH18
Mg(0.5)Fe(0.5)SiO3 in KH18
Mg(0.4)Fe(0.6)SiO3 in KH18 
Mg(0.8)Fe(1.2)SiO4 in KH18 and possibly WS15 (MgSiO4_Fe_rich)
MgFeSiO(4) in KH18

We do not have 
Mg(0.95)Fe(0.05)SiO(3) (Table 4)
Mg(0.7)Fe(0.3)SiO(3) (Table 4)
Mg(0.6)Fe(0.4)SiO(3) (Table 5)
Mg(0.5)Fe(0.43)Ca(0.03)Al(0.04)SiO(3) which is actually from Jager 1995, sample 1S, Table 3
But all of these are easily available on the DOCCD database for future users to add

#################################
Silicates 
#################################

Condensates with silicates (no magnesium) are not usually predicted by equilibrium cloud formation since Si is depleted through the formation of forsterite and enstatite.

From Visscher 2010 (Equilibrium cloud formation): 
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

'Mg2SiO4 formation consumes nearly half of the total Si abundance because the solar elemental abundances of Mg and Si are approximately equal. Thus, quartz (Tcond ∼ 1550 K) can only form in the absence of enstatite (Tcond ∼ 1600 K), because MgSiO3 otherwise efficiently removes silicon from the gas phase. We therefore conclude that SiO2 will not condense within the silicate cloud in equilibrium chemistry.'

More recently, it was found that SiO2 could condense in equilibrium cloud formation if an atmosphere has surpassed enstatite formation (subsolar Mg/Si abundances, Si abundance larger than Mg; Burningham 2021, https://ui.adsabs.harvard.edu/abs/2021arXiv210504268B/abstract)

In the 'dirty dust grain' model where refractory-rich gas is circulated to the upper atmosphere and forms 'seed particles', which grow as the fall through the atmosphere, SiO2 is predicted to be the most abundance cloud species. See Figure 1 in Helling 2018 for a visual depiction. 

Helling 2006/Helling & Woitke 2006 (models that show SiO2 is predicted)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Helling 2018 (nice review)
https://ui.adsabs.harvard.edu/abs/2019AREPS..47..583H/abstract

Recently, SiO2 was detected with JWST MIRI LRS transmission data on WASP-17b

Grant (2023) 
https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract

---------------------------------
---------------------------------

Aerosol Name      : Alpha Carborundum (Moissanite for natural occurring alpha-SiC)
Name in POSEIDON  : SiC
Database          : KH18
Wavelengths       : 1e-3 - 1000 um

Chemical Formula  : SiC
Crystal or Amorph : Crystalline (assumed, see below)
Crystal Shape     : Cubic

Refractive Index References:
--------------
Laor & Draine (1993) 

ADS        : https://ui.adsabs.harvard.edu/abs/1993ApJ...402..441L/abstract

Paper Info :

Compute dielectric functions from first principle equations and lab data (sec 2.1), assuming a density of 3.22 g/cm-3.

1. At x-Ray wavelengths (hv > 12 eV, wavelengths less than 0.1 um), dielectric functions are estimated from atomic theory (not sure if this is the right word, but it's an equation)  (Equation 2-3)

2. For 4-10 eV (0.3 um to 0.123 um) from experimental data in Philipp & Taft (1969) (actually 1960, see below)

3. 10eV to 12eV was smoothly extrapolated

4. Infrared wavelengths were computed from contributions of N damped oscillators (equation 4) (single oscillator fit for SiC). Specifically, they take the oscillator fit from Bohren & Huffman (1983) (see below) and add continuum to align it with Pegourie (1998) (see below). 

Paper only contains dielectric functions, which are tough to compare directly to txt file. 
--------------
Philipp & Taft (1960) 

ADS        : https://apps.dtic.mil/sti/tr/pdf/AD0464777.pdf

Paper Info :

Labor & Draine (1993) says 1969 but it's actually 1960, and from a military compiled textbook 

'Philipp, H. R., & Taft, E. A. 1960, in Silicon Carbide, ed. J. R. O’Connor & J. Smiltens (New York: Pergamon), 366–370'

Originally published at a conference (see reference 56 in military textbook).

Philipp and Taft measure both cubic (alpha) and 6-H hexagonal crystals (beta). I’ll assume the cubic (alpha) was used since its more common.

--------------
Bohren & Huffman (1983) [Section 12.3.4 and Section 9.1]

ADS        : https://ui.adsabs.harvard.edu/abs/1983asls.book.....B/abstract

Paper Info :

Textbook. Made reflectivity measurements of alpha-SiC (section 12.3.4) and one oscillator model (section 9.1).

--------------
Pergourie (1988) 

ADS        : https://ui.adsabs.harvard.edu/abs/1988A%26A...194..335P/abstract

Paper Info :

Developed a synthetic dielectric function to reproduce alpha-SiC lab measurements.
--------------

Exoplanet/Brown Dwarf Papers:

Not particularly mentioned often. Would need a carbon-rich environment I would reckon.

Alternate sources:

Palik (Vol 2, sec 32) has cubic and beta SiC. Also this new paper (Hofmeister (2009),https://ui.adsabs.harvard.edu/abs/2009ApJ...696.1502H/abstract) compiles many sources.

---------------------------------
---------------------------------

Aerosol Name      : Silicon Monoxide
Name in POSEIDON  : SiO
Database          : KH18
Wavelengths       : 4.95e-2 - 100 um

Chemical Formula  : SiO
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Philipp in Palik (1985) (Voume 1, Section 36)

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Noncrystalline, glassy, amorphous SiO forms when SiO is cooled rapidly. The txt file matches the table in Palik for the most part, when imaginary indices were missing a spline interpolation was preformed to fill in the gap. Palik compiled the data from many different sources. Reports 0.04959 - 14 um.
--------------
Wetzel (2013)

ADS        : https://ui.adsabs.harvard.edu/abs/2013A%26A...553A..92W/abstract

Paper Info :

Around S-stars with equal carbon and oxygen, it is expected that oxygen is locked up in SiO and CO. SiO can condense into solid SiO. Thin films of amorphous SiO were formed by SiO vapor deposition on a Si(111) wafer and the dielectric function was fit with a Brendel oscillator model from 2 to 100 um at 300K. Figure 3 shows the dielectric functions, real and imaginary. 

The real indices in the txt file match Figure 3, but the imaginary don't. I assume that the real indices were found by taking the square root of the real dielectric function, and the imaginary were derived with Kramers-Kronig analysis or spline interpolation. Additionally, it looks like these filled in wavelengths from 14 to 100 um.
--------------

Exoplanet/Brown Dwarf Papers:

Not particularly mentioned often, but SiO gas is predicted to be important in hot Jupiters as a silicate cloud precursor, or to form onto TiO2 seed particles in specific situations. 

Lee (2018)
https://ui.adsabs.harvard.edu/abs/2018A%26A...614A.126L/abstract

Lothringer (2022)
https://ui.adsabs.harvard.edu/abs/2022Natur.604...49L/abstract

---------------------------------
---------------------------------

Aerosol Name      : Quartz (alpha + beta)
Name in POSEIDON  : SiO2
Database          : WS15
Wavelengths       : 0.046 - 1000 um (see below)

Chemical Formula  : SiO[2]
Crystal or Amorph : Crystalline
Crystal Shape     : Trigonal (alpha)/Hexagonal (beta)

Refractive Index References:
--------------
Philipp in Palik (1985) (Volume 1, Section 34)

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Alpha quartz from a variety of lab sources. Alpha quartz is room temperature crystalline SiO2. Reports both ordinary and extraordinary indices. The short wavelengths in the txt file (0.046-0.15 um) matches the real and imaginary indices of the ordinary-ray column (n_o,k_o).
--------------
Zeidler (2013) 

ADS        : https://ui.adsabs.harvard.edu/abs/2013A%26A...553A..81Z/abstract

Paper Info :

Explore the temperature and directional dependence of quartz's spectral properties by taking IR reflection measurements of a natural alpha quartz crystal from Brazil. Their measurements range from 300K to 928K, capturing the alpha to beta quartz phase transition that occurs at 846-847K.

The indices found from 6.66 um to 1000 um matches the indices found for the 928K beta quartz, extraordinary ray.

Also on DOCCD (SiO2 at 928, Extraordinary, E||c)
https://www2.astro.uni-jena.de/Laboratory/OCDB/crsilicates.html
--------------

Note on wavelength coverage: 

The wavelength coverage from 0.15 to 6.66 um is sparse. In the aerosol database, the points from 0.15 to 3.45 to 6.66 is assumed to be linear. This isn't too much of an issue since 1. The real indices are mostly flat in this wavelength region, 2. The imaginary indices are very low (1e-5) (no major absorption) [See Palik indices below], and 3. Optical properties in this region are dominated mostly by particle size, not composition. 

The 3.45 um point is a bit of a mystery, but is assumed to just be an 'in-between' point between the two datasets.

Note: 

The original WS15 paper also cites Anderson 2006 (https://ui.adsabs.harvard.edu/abs/2006JQSRT.100....4A/abstract), which utilized amorphous glass indices from Palik. Plotting up the WS15 against the Palik glass ones, they don't match, and its assumed that the txt file was updated later on. 

Exoplanet/Brown Dwarf Papers:

Equilibrium cloud formation doesn't predict SiO2 since the formation of enstatite and forsterite consumes all the Si gas. Disequilibrium cloud formation predicts amorphous SiO2 in brown dwarfs. 

From Visscher 2010 (Equilibrium cloud formation): 
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

Helling 2006/Helling & Woitke 2006 (models that show SiO2 is predicted)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Recently, crystalline SiO2 was detected with JWST MIRI LRS transmission data on WASP-17b. This text file was used for the POSEIDON retrievals in that paper. 

Grant (2023) 
https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract

---------------------------------
---------------------------------

Aerosol Name      : Quartz 
Name in POSEIDON  : SiO2_crystalline_2023
Database          : Mullens 2024
Wavelengths       : 0.25 - 15.38 um

Chemical Formula  : SiO[2]
Crystal or Amorph : Crystalline 
Crystal Shape     : Trigonal (alpha) (assumed, see below)

Refractive Index References:
--------------
Herve Herbin & Petitprez (2023)

ADS        : https://www.tandfonline.com/doi/full/10.1080/02786826.2023.2165899 (no ads entry)

Paper Info :

Measured crystalline quartz, airborne particles (15 nm to 20 um) from infrared to UV. Used the airborne method since they did not like previous KBr pellet methods. Quartz sample comes from Sigma-Aldrich (CAS number 14808-60-7, particle size < 63 um) and has a density provided by the manufacturer of 2.65 g/cm3). Suspended in rotating N2 gas with a measured lognormal particle distribution centered around 0.24 um (width of 2.24). 

No need to separate extraordinary and ordinary rays since random orientation was checked with a polarizer. 

The txt file comes from the excel sheet in supplementary material, which was converted from wavenumber to wavelength. 

Assumed to be alpha quartz, since measurements were presumably made at room temperature. 
--------------

Exoplanet/Brown Dwarf Papers:

Equilibrium cloud formation doesn't predict SiO2 since the formation of enstatite and forsterite consumes all the Si gas. Disequilibrium cloud formation predicts amorphous SiO2 in brown dwarfs. 

From Visscher 2010 (Equilibrium cloud formation): 
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

Helling 2006/Helling & Woitke 2006 (models that show SiO2 is predicted)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Recently, crystalline SiO2 was detected with JWST MIRI LRS transmission data on WASP-17b. This text file was used for the POSEIDON retrievals in that paper. 

Grant (2023) 
https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract

---------------------------------
---------------------------------

Aerosol Name      : Quartz (alpha quartz + silica glass)
Name in POSEIDON  : SiO2_amorph
Database          : KH18
Wavelengths       : 4.76e-2 - 487 um

Chemical Formula  : SiO[2]
Crystal or Amorph : Mixed (see below)
Crystal Shape     : Trigonal/ N/A

Refractive Index References:
--------------
Henning & Mutschke (1997)

ADS        : https://ui.adsabs.harvard.edu/abs/1997A%26A...327..743H/abstract

Paper Info :

Measured silicates, ferrous oxide, and iron sulfides for low temperature cosmic dust. 

Amorphous SiO2 sample for reflection is a commercial quartz glass window (Suprasil, a window brand) which was cut into wedge shapes. 

For transmission, commercially available powder of amorphous powder consisting of mono-sized spherical particles of 500 nm diameter (“Monosphere powder” M 500, MERCK, Darmstadt). 

On DOCCD (Amorphous SiO2 (Low-T data), SiO2 (300K))
https://www2.astro.uni-jena.de/Laboratory/OCDB/amsilicates.html
--------------
Philipp in Palik (1985) (Volume 1, Section 34)

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Alpha quartz from a variety of lab sources. Alpha quartz is room temperature crystalline SiO2. Reports both ordinary and extraordinary indices. The short wavelengths in the txt file matche the real and imaginary indices of the ordinary-ray column (n_o,k_o). 

Note that the imaginary indices in the wavelength region from 0.1494 um to 5.8 um were generated from the real indices utilizing the Kramers-Kronig relation. 
--------------

Note on wavelengths: 

The amorphous quartz is used from 7.952 um to the longest wavelength, alpha quartz is used from 0.05 to 7.95 um. This is why the cross section has a bit of a jump on the blue edge of the feature. 

Exoplanet/Brown Dwarf Papers:

Equilibrium cloud formation doesn't predict SiO2 since the formation of enstatite and forsterite consumes all the Si gas. Disequilibrium cloud formation predicts amorphous SiO2 in brown dwarfs. 

From Visscher 2010 (Equilibrium cloud formation): 
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

Helling 2006/Helling & Woitke 2006 (models that show SiO2 is predicted)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Recently, crystalline SiO2 was detected with JWST MIRI LRS transmission data on WASP-17b. This text file was used for the POSEIDON retrievals in that paper. 

Grant (2023) 
https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract

---------------------------------
---------------------------------

Aerosol Name      : Quartz (alpha)
Name in POSEIDON  : SiO2_alpha_palik
Database          : Mullens 2024
Wavelengths       : 0.185 - 31 um

Chemical Formula  : SiO[2]
Crystal or Amorph : Crystalline 
Crystal Shape     : Trigonal

Refractive Index References:
--------------
Philipp in Palik (1985) (Volume 1, Section 34)

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Alpha quartz from a variety of lab sources. Alpha quartz is room temperature crystalline SiO2. Reports both ordinary and extraordinary indices.

Note that the ordinary, imaginary indices in the wavelength region from 0.1494 um to 5.8 um were generated from the real indices utilizing the Kramers-Kronig relation from KH18. 

The extraordinary imaginary indices from 0.185 to 5.8 um were generated from real indices using the Kramers-Kronig relation in Mullens 2024. 

I specifically iterated the Kramers-Kronig relation twice using the pyElli package (and code from Dr. Sarah Moran) to get a self consistent set of real and imaginary indices, and offset them to match the original Palik data. 

The two were then averaged using the (2/3 Ordinary) + (1/3 Extraordinary) weighting of refractive indices. 
--------------

Note on wavelength coverage:

This Palik entry was included in the Mullens 2024 database to provide an alternative SiO2 to use. 

Exoplanet/Brown Dwarf Papers:

Equilibrium cloud formation doesn't predict SiO2 since the formation of enstatite and forsterite consumes all the Si gas. Disequilibrium cloud formation predicts amorphous SiO2 in brown dwarfs. 

From Visscher 2010 (Equilibrium cloud formation): 
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

Helling 2006/Helling & Woitke 2006 (models that show SiO2 is predicted)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Recently, crystalline SiO2 was detected with JWST MIRI LRS transmission data on WASP-17b. This text file was used for the POSEIDON retrievals in that paper. 

Grant (2023) 
https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract

---------------------------------
---------------------------------

Aerosol Name      : Silica Glass
Name in POSEIDON  : SiO2_glass_palik
Database          : Mullens 2024
Wavelengths       : 0.184 - 33.33 um

Chemical Formula  : SiO[2]
Crystal or Amorph : Amorphous (glass)
Crystal Shape     : N/A

Refractive Index References:
--------------
Philipp in Palik (1985) (Volume 1, Section 35)

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Compiled many glass references. Notes that getting the imaginary indices is difficult due to H2O's OH band messing up lab measurements. 

Kramers-Kronig was utilized to compute the imaginary indices from 0.184 to 3.55 um, 6.75 to 7.63 um, real indices in gaps from 3.63 to 6.9 um. 
 
Both k (taken from reflection) and k_abs (taken from transmission) were included in the txt file. When a row had both values listed, an average of the two was taken. 

I specifically iterated the Kramers-Kronig relation twice using the pyElli package (and code from Dr. Sarah Moran) to get a self consistent set of real and imaginary indices, and offset them to match the original Palik data.
--------------

Note on wavelength coverage:

This Palik entry was included in the Mullens 2024 database to provide an alternative SiO2 to use. 

Exoplanet/Brown Dwarf Papers:

Equilibrium cloud formation doesn't predict SiO2 since the formation of enstatite and forsterite consumes all the Si gas. Disequilibrium cloud formation predicts amorphous SiO2 in brown dwarfs. 

From Visscher 2010 (Equilibrium cloud formation): 
https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1060V/abstract

Helling 2006/Helling & Woitke 2006 (models that show SiO2 is predicted)
https://ui.adsabs.harvard.edu/abs/2006A%26A...451L...9H/abstract
https://ui.adsabs.harvard.edu/abs/2006A%26A...455..325H/abstract

Recently, crystalline SiO2 was detected with JWST MIRI LRS transmission data on WASP-17b. This text file was used for the POSEIDON retrievals in that paper. 

Grant (2023) 
https://ui.adsabs.harvard.edu/abs/2023ApJ...956L..32G/abstract

#################################
T-Y Dwarf
#################################

These are aerosols that are predicted to condense in the T-Y dwarf temperature regime. 
We treat ices that are expected to condense out in Y-dwarfs and solar system gas giants separately in the next section. 

In general: MgSiO3 -> Cr -> MnS -> Na2S -> ZnS -> KCl

Where Na2S will have the highest produced cloud mass

See Morley (2012) (Figure 3 and Table 3) and Morley (2014) for more details 
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract
https://ui.adsabs.harvard.edu/abs/2014ApJ...787...78M/abstract

Also potentially forms in sub-Neptunes like GJ1214b and Hot Jupiter terminators 

Morley (2013) (GJ 1214b)
https://ui.adsabs.harvard.edu/abs/2013ApJ...775...33M/abstract

Kataria (2016) (Hot Jupiter gcms)
https://ui.adsabs.harvard.edu/abs/2016ApJ...821....9K/abstract

---------------------------------
---------------------------------

Aerosol Name      : Chromium 
Name in POSEIDON  : Cr
Database          : KH18
Wavelengths       : 4.2e-2 - 500 um

Chemical Formula  : Cr
Crystal or Amorph : Crystalline 
Crystal Shape     : Cubic (body centered cubic) (bulk growth) but technically tetragonal

Note from Dr. Daniel Kitzmann:

'Up until 20.66 this should be the data from Palik. Beyond that Rakic is used.'

Refractive Index References:
--------------
Lynch & Hunter in Palik (1991) [Volume 2, Section 15.6] (0.04-31 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1991hocs.book.....P/abstract

Paper Info :

Compiles many sources. Cr is body centered cubic metal (bulk growth). Antiferromagnetic below 312K (room temp)(has properties depending on external magnetic field) and paramagnetic above. Its magnetic properties induce fine structure.

Crystal shape is technically tetragonal, but not by much (doesn't deviate much from cubic)

The txt file matches up to 15.50 um (note that 8 to 30 um is very sparse in Palik). 20.66 um to 31 does not match.

The imaginary index of Cr grows very large (up to 65 at 31 um)

--------------
Rakic (1998) (31-500 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1998ApOpt..37.5271R/abstract

Paper Info :

Computed dielectric functions from first principles and shown to fit lab data (see Figure 6). 
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after MgSiO3 with a cloud mass 30% of Na2S.

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

Can be a seed particle for ZnS, Na2S, and MnS 

Lee (2018)
https://ui.adsabs.harvard.edu/abs/2018A%26A...614A.126L/abstract

---------------------------------
---------------------------------

Aerosol Name      : Manganese Sulfide (alpha)
Name in POSEIDON  : MnS
Database          : WS15
Wavelengths       : 0.14 - 13 um

Chemical Formula  : MnS
Crystal or Amorph : Crystalline
Crystal Shape     : Cubic

Refractive Index References:
--------------
Huffman & Wild (1967)

ADS        : https://ui.adsabs.harvard.edu/abs/1967PhRv..156..989H/abstract
Paper Info :

Single crystal samples were grown from alpha MnS powder. Used three different techniques to get properties (depending on wavelength range). 

Transparent regime - transmission - 4.5 to 20 um
Shorter and longer wavelengths used UV and infrared reflectance

WS15 used a plot digitizer on Figure 5, which is in photon energy (eV) where 12 eV is 0.1 um, 6 eV is 0.2 um, and 1 eV is 1.2 um. 

Note that the real indices in the file do not match Table 1 in the paper. 

Note that alpha MnS is cubic, beta MnS (similar to zinc blende, ZnS) is cubic, and gamma MnS (similar to wurtzite, ZnS) is hexagonal.
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after Cr with a cloud mass 36% of Na2S.

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

GCMs predict is condensing on dayside of hot Jupiter HD 189733b

Kataria (2016) 
https://ui.adsabs.harvard.edu/abs/2016ApJ...821....9K/abstract

---------------------------------
---------------------------------

Aerosol Name      : Manganese Sulfide (alpha) (+ Na2S)
Name in POSEIDON  : MnS_KH
Database          : KH18
Wavelengths       : 9e-2 - 190 um

Chemical Formula  : MnS (+ Na2S)
Crystal or Amorph : Crystalline 
Crystal Shape     : Cubic

Note from KH18:

‘In case of MnS, no infrared data was available, which includes in particular the sulphur feature. Here, we follow the approach of Morley et al. (2012) and use extrapolations based on the other two sulphur-bearing species Na2S and ZnS to reconstruct the missing part.’

Note from Dr. Daniel Kitzmann: 

'I took the real index directly from the Table 1 in Huffman & Wild and extended it using their Fig. 5. The imaginary part has also been taken from Fig. 5, with the missing data reconstructed by using the Kramers-Kronig relations.'

'Since, as far I as I was aware when I made the compilation, no MnS data was available that covered the sulphur feature, I used the same approach as explained in one of Caroline Morley's papers. We both took the sulphur feature from NaS2 and transplanted it into the data for MnS. In my case, I only took the real part from NaS2 and then used the Kramers-Kronig relations to compute the imaginary part. My choice of using NaS2 is, therefore, mainly influenced by Caroline's paper. Technically, one could have also used ZnS just as well, I suppose. However, both choices are probably equally wrong in the end. Without adequate MnS data in the wavelength region of the expected sulphur feature, we therefore should be very careful in trusting our data set too much.'

Refractive Index References:
--------------
Huffman & Wild (1967) (0.05-13 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1967PhRv..156..989H/abstract

Paper Info :

Single crystal samples were grown from alpha MnS powder. Used three different techniques to get properties (depending on wavelength range). 

Transparent regime - transmission - 4.5 to 20 um
Shorter and longer wavelengths used UV and infrared reflectance

Note that alpha MnS is cubic, beta MnS (similar to zinc blende, ZnS) is cubic, and gamma MnS (similar to wurtzite, ZnS) is hexagonal.
--------------
Montaner (1979) (2.5-200 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1979PSSAR..52..597M/abstract

Paper Info :

KH18 took this data from a figure. Experimental infrared data of crystalline (face centered cubic) Na2S. Used low temperature infrared (reflectance) and Raman measurements. 

How they grew the crystals from powder is detailed in Moret & Bill (1977) (https://ui.adsabs.harvard.edu/abs/1977PSSAR..41..163M/abstract).

Two samples done at 15K. One in a H2 atmosphere and one in a Na vapor brought to vacuum. Unsure which indices were used since the paper reports dielectric functions.
--------------
Palik & Addamiano in Palik (1985) [Volume 1, Section 27]

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Compiled many different lab sources of cubic ZnS. Assumed reference (see below)
--------------

NOTE: 

KH18 says they used Na2S and ZnS, but only cite a Na2S paper. It is assumed that the ZnS in Palik was used, since it has a feature before 30 microns. Additionally, while their imaginary indices match WS15's, their real indices do not. It is assumed that the imaginary indices were taken from Huffman & Wild and Montaner, and the real indices were derived using the Kramers-Kronig relation. Because of this, the scattering slope shape is different than the one in WS15.

Exoplanet/Brown Dwarf Papers:

Expected to condense after Cr with a cloud mass 36% of Na2S.

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

GCMs predict is condensing on dayside of hot Jupiter HD 189733b

Kataria (2016) 
https://ui.adsabs.harvard.edu/abs/2016ApJ...821....9K/abstract

---------------------------------
---------------------------------

Aerosol Name      : Manganese Sulfide (alpha)
Name in POSEIDON  : MnS_Mor
Database          : WS15 + KH18 
Wavelengths       : 0.14 - 190 um

Chemical Formula  : MnS
Crystal or Amorph : Crystalline
Crystal Shape     : Cubic

From Mullens 2024 
'Morley 2012 introduced a hybrid treatment, where the measured refractive indices are used in conjunction with a sulfide extrapolation. For this entry, we combine the refractive indices from WS15 up to 13 um and the refractive indices from KH18 from 13-30 um (this entry has both the non-uniform scattering slope and sulfide absorption at longer wavelengths)/'

Refractive Index References:
--------------
Huffman & Wild (1967) (0.05-13 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1967PhRv..156..989H/abstract

Paper Info :

Single crystal samples were grown from alpha MnS powder. Used three different techniques to get properties (depending on wavelength range). 

Transparent regime - transmission - 4.5 to 20 um
Shorter and longer wavelengths used UV and infrared reflectance

WS15 used a plot digitizer on Figure 5. 

Note that alpha MnS is cubic, beta MnS (similar to zinc blende, ZnS) is cubic, and gamma MnS (similar to wurtzite, ZnS) is hexagonal.
--------------
Montaner (1979) (2.5-200 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1979PSSAR..52..597M/abstract

Paper Info :

KH18 took this data from a figure. Experimental infrared data of crystalline (face centered cubic) Na2S. Used low temperature infrared (reflectance) and Raman measurements. 

How they grew the crystals from powder is detailed in Moret & Bill (1977) (https://ui.adsabs.harvard.edu/abs/1977PSSAR..41..163M/abstract).

Two samples done at 15K. One in a H2 atmosphere and one in a Na vapor brought to vacuum. Unsure which indices were used since the paper reports dielectric functions.
--------------
Palik & Addamiano in Palik (1985) [Volume 1, Section 27]

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Compiled many different lab sources of cubic ZnS. Assumed reference (see below)
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after Cr with a cloud mass 36% of Na2S.

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

GCMs predict is condensing on dayside of hot Jupiter HD 189733b

Kataria (2016) 
https://ui.adsabs.harvard.edu/abs/2016ApJ...821....9K/abstract

---------------------------------
---------------------------------

Aerosol Name      : Sodium sulfide
Name in POSEIDON  : Na2S
Database          : WS15
Wavelengths       : 0.03 - 74 um

Chemical Formula  : Na[2]S
Crystal or Amorph : Crystalline 
Crystal Shape     : Cubic (face centered cubic)

Refractive Index References:
--------------
Morley (2012) 

ADS        : https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

Paper Info :

Used two different refractive indices, listed below. Specifically used the overlap region (25-91 um) of Montaner over the Khachai ones.
--------------
Montaner (1979) (25-198 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1979PSSAR..52..597M/abstract

Paper Info :

Experimental infrared data of crystalline (face centered cubic) Na2S. Used low temperature infrared (reflectance) and Raman measurements. 

How they grew the crystals from powder is detailed in Moret & Bill (1977) (https://ui.adsabs.harvard.edu/abs/1977PSSAR..41..163M/abstract).

Two samples done at 15K. One in a H2 atmosphere and one in a Na vapor brought to vacuum. Unsure which indices were used since the paper reports dielectric functions.
--------------
Khachai (2009) (0.03-91 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2009JPCM...21i5404K/abstract

Paper Info :

Computed optical properties from first principles (completely computational, no lab data). Computed optical properties (dielectric constants) of cubic anti-fluorite structure (the cations and anions reversed)
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after MnS with the largest cloud mass.

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

GCMs predict is condensing on terminator of hot Jupiter HD 189733b

Kataria (2016) 
https://ui.adsabs.harvard.edu/abs/2016ApJ...821....9K/abstract

---------------------------------
---------------------------------

Aerosol Name      : Zinc Sulfide (Zinc blende, sphalerite)
Name in POSEIDON  : ZnS
Database          : WS15
Wavelengths       : 0.22 - 167 um

Chemical Formula  : ZnS
Crystal or Amorph : Crystalline
Crystal Shape     : Cubic

Refractive Index References:
--------------
Querry (1987)

ADS        : https://apps.dtic.mil/sti/citations/ADA192210 (no ADS link)

Paper Info :

This is a US military textbook. ZnS reported in section 4.6, Table 6 matches the txt file. 

Reflectance measurements of high purity, optically isotropic sample. Because it was labeled isotropic, we know that this was specifically zinc blende (also known as sphalerite). Wurtzite has a hexagonal structure. 
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after Na2S with a cloud mass 5% of Na2S.

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

GCMs predict is condensing on terminator of hot Jupiter HD 189733b

Kataria (2016) 
https://ui.adsabs.harvard.edu/abs/2016ApJ...821....9K/abstract

---------------------------------
---------------------------------

Aerosol Name      : Halite (rock salt)
Name in POSEIDON  : NaCl
Database          : WS15
Wavelengths       : 0.047 - 1000 um

Chemical Formula  : NaCl
Crystal or Amorph : Crystalline 
Crystal Shape     : Cubic

Refractive Index References:
--------------
Eldridge & Palik in Palik (1985) (Volume 1, Section 38)

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Compiled many sources for crystalline NaCl at room temperature. 

In the Palik textbook, sometimes the imaginary index column is empty. When this happens, WS15 will take the value as constant until the next k-value is listed. This constant k causes a 'step-function-like' opacity in the database. 

There are two n and k columns in Palik. When a wavelength has two n values listed, the first two columns are the default. It is not clear to me what the difference between these columns are. 
--------------

Exoplanet/Brown Dwarf Papers:

Not mentioned very often. Na2S is assumed to be the primary condensate since Na is the limiting element in that condensation. Apparently predicted on Kepler-434b. 

Can be a seed particle for NH4H2PO4, H2O, NH4SH

Lee (2018)
https://ui.adsabs.harvard.edu/abs/2018A%26A...614A.126L/abstract

---------------------------------
---------------------------------

Aerosol Name      : Sylvite (assumed, see below)
Name in POSEIDON  : KCl
Database          : WS15
Wavelengths       : 0.028 - 200 um

Chemical Formula  : KCl
Crystal or Amorph : Crystalline 
Crystal Shape     : Cubic (face centered cubic)

Refractive Index References:
--------------
Palik in Palik (1985) (Volume 1, Section 33)

ADS        : https://ui.adsabs.harvard.edu/abs/1985hocs.book.....P/abstract

Paper Info :

Compiled many room temperature sources. Like NaCl:

In the Palik textbook, sometimes the imaginary index column is empty. When this happens, WS15 will take the value as constant until the next k-value is listed. This constant k causes a 'step-function-like' opacity in the database. 

I assume that this is Sylvite which is the most likely polymorph of KCl at room temperature. 
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after ZnS with a cloud mass 12% of Na2S.

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

Can be a seed particle for NH4H2PO4, H2O, NH4SH

Lee (2018)
https://ui.adsabs.harvard.edu/abs/2018A%26A...614A.126L/abstract


#################################
Ices
#################################

Ices form after the salt and sulphide clouds.

In order, ADP is expected to form, then H2O, NH3, NH4SH, and CH4.

These ices are expected to form in Y dwarfs and solar system gas giants. 

Morley (2014) for more details on Y dwarfs
https://ui.adsabs.harvard.edu/abs/2014ApJ...787...78M/abstract

Giant Planets of Our Solar System by Irwin (2009), chapter 4

---------------------------------
---------------------------------

Aerosol Name      : Ammonium Dihydrogen Phosphate
Name in POSEIDON  : ADP
Database          : Mullens 2024
Wavelengths       : 0.2 - 20 um

Chemical Formula  : NH[4]H[2]PO[4]
Crystal or Amorph : Crystalline (+Liquid)
Crystal Shape     : Tetragonal 

Refractive Index References:
--------------
Zernike (1965) (0.2 - 1.9 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1965JOSA...55..210Z/abstract

Paper Info :

The original paper was Zernike (1964) (https://ui.adsabs.harvard.edu/abs/1964JOSA...54.1215Z/abstract), which had methods but an error in the tables that was fixed in the follow up Zernike (1965). 

Used a prism with triangular faces of ADP and measured extraordinary and ordinary directions at room temperature. 

I used Table 7, absolute (calibrated) indices in 1965 paper.

I assumed that the short wavelengths have a refractive index of 0 due to being in the transparent regime up to 1.15 um. From 1.15 to 2um, the indices are extrapolated (see Fig 1 in 1964). 

Edwards and White in Palik Vol 2, Section 49 speak on why KDP and ADP make good crystals for nonlinear optical devices and laser, and why assuming k = 0 in the shortest wavelength is a decent assumption. 

I weighted the extraordinary by ⅓ and ordinary by ⅔, assuming the crystal is symmetric on two of its axes (its crystal shape is a prism, much like quartz, except it optically negative which means it assumes an oblate spheroid shape).

I excluded 2 um here in lieu of the next dataset.

ADP crystal is piezoelectric, which means it can generate electric signals when stressed (neat!).

--------------
Querry (1974) (2 - 19.99 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1974JOSA...64...39Q/abstract

Paper Info :

Reflectance measurements in 0.5M aqueous solution of ADP. Not much more information than that, unfortunately. 

--------------

Exoplanet/Brown Dwarf Papers:

Expected to be the first ice that condenses after KCl. Limiting element is PH3, and can remove phosphorus efficiently from the atmosphere. 

Morley (2012) (Figure 3 and Table 3)
https://ui.adsabs.harvard.edu/abs/2012ApJ...756..172M/abstract

Morley (2018) (Figure 21)
https://ui.adsabs.harvard.edu/abs/2018ApJ...858...97M/abstract

Edwards and White in Palik Vol 2, Section 49 have ADP real indices from 0.220 to 1.160 m and some infrared lattice modes, but more work needs to be done to get full wavelength coverage. 

Good link to learn more about ADP crystals 

Linear Electrooptic Modular Materials (Textbook), chapter on ADP
https://link.springer.com/chapter/10.1007/978-1-4684-6168-8_4#preview

---------------------------------
---------------------------------

Aerosol Name      : Water (liquid)
Name in POSEIDON  : H2O
Database          : WS15
Wavelengths       : 0.2 - 200 um

Chemical Formula  : H[2]O
Crystal or Amorph : N/A
Crystal Shape     : N/A

Refractive Index References:
--------------
Hale & Quarry (1973)

ADS        : https://ui.adsabs.harvard.edu/abs/1973ApOpt..12..555H/abstract

Paper Info :

25 Celsius (298.5K) liquid water. Compilation of 58 difference references. 
--------------

Exoplanet/Brown Dwarf Papers:

Water clouds are expected to form in Y dwarf atmospheres, and deep water clouds are expected in solar system gas giants. Papers predict that all water would be in the form of solid ice in Y dwarfs, but that solar system giants can have aqueous and solid water clouds (ice clouds form the upper layer of water clouds, aqueous clouds form the lower layer of water clouds and potentially mix with ammonia). 

Morley 2014 (water clouds in Y-dwarfs)
https://ui.adsabs.harvard.edu/abs/2014ApJ...787...78M/abstract

Giant Planets of Our Solar System by Irwin (2009), chapter 4 (water clouds in solar system giants)

---------------------------------
---------------------------------

Aerosol Name      : Water (Ice 1h)
Name in POSEIDON  : H2O_ice
Database          : WS15
Wavelengths       : 4.43e-2 - 2e6 um

Chemical Formula  : H[2]O
Crystal or Amorph : Crystalline
Crystal Shape     : Hexagonal

Refractive Index References:
--------------
Warren (1984)

ADS        : https://ui.adsabs.harvard.edu/abs/1984ApOpt..23.1206W/abstract

Paper Info :

This is a review article that compiles a lot of different sources (see their Figure 2), which included growing ice crystals with different structures and phases, to measured absorption through blocks of lake ice. 

The txt file was made from the -7 C, ice-1h, Table 1.

Virtually all ice on Earth is ice 1h. 
--------------

Note on ice polymorphs:

There are many ice polymorphs. At atmospheric pressures there is Ice 6 (low pressure) is orthorhombic, Ice 1c is cubic, Ice 1h is hexagonal. There is also amorphous ice. 

A potentially good resource for crystalline and amorphous temperature dependent ice is the following:

Mastrapa al (2009) 
https://ui.adsabs.harvard.edu/abs/2009ApJ...701.1347M/abstract

Kim 2024 (Figure 1, no ADS at time of writing)
https://www.arxiv.org/abs/2408.03278

Exoplanet/Brown Dwarf Papers:

Water clouds are expected to form in Y dwarf atmospheres, and deep water clouds are expected in solar system gas giants. Papers predict that all water would be in the form of solid ice in Y dwarfs, but that solar system giants can have aqueous and solid water clouds (ice clouds form the upper layer of water clouds, aqueous clouds form the lower layer of water clouds and potentially mix with ammonia). 

Morley 2014 (water clouds in Y-dwarfs)
https://ui.adsabs.harvard.edu/abs/2014ApJ...787...78M/abstract

Giant Planets of Our Solar System by Irwin (2009), chapter 4 (water clouds in solar system giants)

---------------------------------
---------------------------------

Aerosol Name      : Ammonium Hydrosulfide
Name in POSEIDON  : NH4SH
Database          : Mullens 2024
Wavelengths       : 0.5 - 1000 um

Chemical Formula  : NH[4]SH
Crystal or Amorph : Crystalline (assumed, see below)
Crystal Shape     : Rhombic

Refractive Index References:
--------------
Personal Communication from Dr. Carly Howett, unpublished NH4SH indices

According to Dr. Howett, these indices are mentioned in her paper: 

Howett (2007)

ADS        : https://ui.adsabs.harvard.edu/abs/2007JOSAB..24..126H/abstract

Paper Info :

NH4SH transitions from amorphous to crystalline at 160K. Annealing crystalline NH4SH results in absorption features becoming sharper and increasing in magnitude, with little change in wavelength position. 

The paper above mentions that there are a few unpublished NH4SH datasets floating around the planetary science community (set A and set B in Howett (2007)). 

Just comparing the txt file to Figure 10, my indices match set A better. I believe that set A and set B are mislabeled in this plot, and that set A is actually the polycrystalline set (set B), just by comparing the tables and the figures. 

Set B indices are supposedly based off the work of Ferraro (1980) (https://ui.adsabs.harvard.edu/abs/1980ApSpe..34..525F/abstract) where 0.5-2.6 um were extrapolated. 

While Carly Howett mentions an updated dataset in the paper, the website that hosted them has been taken down. 
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after H2O. Is predicted to exist between water and ammonia clouds in both Saturn and Jupiter. (Ignoring photochemical hazes)

Giant Planets of Our Solar System by Irwin (2009), chapter 4 (ammonia clouds in solar system giants)

---------------------------------
---------------------------------

Aerosol Name      : Ammonia 
Name in POSEIDON  : NH3
Database          : optool
Wavelengths       : 0.14 - 200 um

Chemical Formula  : NH[3]
Crystal or Amorph : Crystalline 
Crystal Shape     : Cubic 

Refractive Index References:
--------------
Martonchik (1984)

ADS        : https://ui.adsabs.harvard.edu/abs/1984ApOpt..23..541M/abstract

Paper Info :

Cubic phase (crystalline) of solid ammonia. Real indices were derived via Kramers-Kronig. Absorption coefficients used to determine the imaginary indices are from a variety of sources. 

1. Absorption coefficients from Sill et al. (1981) (https://ui.adsabs.harvard.edu/abs/1981JChPh..74..997S/abstract) form the bulk of the data set used to determine optical constants. 

Sill et al. (1981) used Fourier spectrometers in the 1.4 - 200 um range of an ammonia ice thin film sample at 88K. Absorption coefficients were obtained but no refractive indices were derived. 

2. UV wavelength mass absorption coefficients were taken from  77K Dressler & Schnepp (1960) (https://pubs.aip.org/aip/jcp/article-abstract/33/1/270/207591/Absorption-Spectra-of-Solid-Methane-Ammonia-and?redirectedFrom=fulltext), and Browell & Anderson (1975) (https://ui.adsabs.harvard.edu/abs/1975JOSA...65..919B/abstract). 

Dressler & Schnepp (1960) took measurements of thin films created my ammonia vapor deposition at both 77K and 175K. Since ammonia ice undergoes a phase transition from amorphous to cubic from 70 to 90K, Martonchik (1984) assumes that the 77K substrate was cubic in phase and took the absorption coefficients at 77K, reduced by a factor of 4.0 (recalibration). 

Browell & Anderson (1975) recalibrated the absorption coefficients of Dressler & Schnepp (1960).
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after NH4SH. Is the predicted top cloud layer in both Saturn and Jupiter. (Ignoring photochemical hazes)

Giant Planets of Our Solar System by Irwin (2009), chapter 4 (ammonia clouds in solar system giants)

---------------------------------
---------------------------------

Aerosol Name      : Methane (Liquid)
Name in POSEIDON  : CH4_liquid
Database          : WS15
Wavelengths       : 0.002 - 71.43 um

Chemical Formula  : CH[4]
Crystal or Amorph : N/A
Crystal Shape     : N/A

Refractive Index References:
--------------
Martonchik & Orton (1994)

ADS        : https://ui.adsabs.harvard.edu/abs/1994ApOpt..33.8306M/abstract
Paper Info :

Measured optical properties of both solid and liquid methane. Compiled a lot of extant lab data and synthesized it. Table 2 is liquid methane and Table 3 is solid methane. Note that the refractive indices in the table are sometimes and upper or lower limit, which we take as the true value in the refractive index txt file.

Txt file was generated using the 111K liquid methane in Table 2. 
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after NH3 and H2S (H2S is not in this version of the database) and is the topmost cloud of Uranus and Neptune. (Ignoring photochemical hazes). 

Liquid methane in particular is not expected on Uranus and Neptune, but has been found on the surface of Saturn's moon Titan. 

Giant Planets of Our Solar System by Irwin (2009), chapter 4 (methane clouds in solar system giants)

---------------------------------
---------------------------------

Aerosol Name      : Methane (solid)
Name in POSEIDON  : CH4_solid
Database          : WS15 (see below)
Wavelengths       : 0.002 - 100 um

Chemical Formula  : CH[4]
Crystal or Amorph : Crystalline
Crystal Shape     : Cubic

Refractive Index References:
--------------
Martonchik & Orton (1994)

ADS        : https://ui.adsabs.harvard.edu/abs/1994ApOpt..33.8306M/abstract
Paper Info :

Measured optical properties of both solid and liquid methane. Compiled a lot of extant lab data and synthesized it. Table 2 is liquid methane and Table 3 is solid methane. Note that the refractive indices in the table are sometimes and upper or lower limit, which we take as the true value in the refractive index txt file.

Txt file was generated using the 90K solid methane in Table 3. 

This entry is slightly new to Mullens 2024, since the indices I got from WS15 were solely the liquid methane. However, I decided to keep it grouped with WS15 since it's the same reference. 
--------------

Exoplanet/Brown Dwarf Papers:

Expected to condense after NH3 and H2S (H2S is not in this version of the database) and is the topmost cloud of Uranus and Neptune. (Ignoring photochemical hazes). 

Giant Planets of Our Solar System by Irwin (2009), chapter 4 (methane clouds in solar system giants)

---------------------------------
---------------------------------

Aerosol Name      : Ice Tholins
Name in POSEIDON  : IceTholin
Database          : Mullens 2024
Wavelengths       : 6e-2 - 40 um

Chemical Formula  : C2H6/H2O irradiation residue 
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Khare (1993) 

ADS        : https://ui.adsabs.harvard.edu/abs/1993Icar..103..290K/abstract

Paper Info :

Indices were generated from Table 1 and Table 2. Since the wavelengths for real and imaginary indices were not the same, the real and imaginary indices were interpolated onto a grid of shared wavelengths. 

Ice tholins were generated by 50 irradiations over a five month timeline of C2H6/H2O (ethane/water) ice (1:6 ratio) at 77K, which produced the ice tholins.  

Because it is a photochemical product, I assume it is amorphous. 
--------------

Exoplanet/Brown Dwarf Papers:

Used to explain photochemical products found on asteroids, as well as Neptune's atmosphere. 

Gwenael (2024) (application of Ice Tholins to Neptune)
https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.303M/abstract

#################################
Soots and Hazes
#################################

Soots and hazes are mostly photochemical products that can exist in planetary atmospheres. 

Technically, soots are a combustion product while hazes are photochemical (and usually organic)

These usually form in the upper atmosphere at temperatures less than 900K. (Sub-Neptunes, Venus-like planets, potentially the limb of Hot Jupiters). 

---------------------------------
---------------------------------

Aerosol Name      : Graphite
Name in POSEIDON  : C
Database          : KH18
Wavelengths       : 1.2e-4 - 1.2e5 um

Chemical Formula  : C
Crystal or Amorph : Crystalline 
Crystal Shape     : Hexagonal (crystal growth is a rhombic crystal)

Refractive Index References:
--------------
Draine (2003) & Draine (2003)

ADS        : https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1017D/abstract
             https://ui.adsabs.harvard.edu/abs/2003ApJ...598.1026D/abstract

Paper Info :

Graphite is known to be strongly anisotropic, so KH18 took each direction and averaged (after converting dielectric function to refractive index)

The two papers above are a series that look at scattering of interstellar dust grains in the X-ray, Optical, and UV.

Lab data was originally compiled in Draine & Lee (1984) (https://ui.adsabs.harvard.edu/abs/1984ApJ...285...89D/abstract), which across the board compiled room temperature graphite crystal data. See section 2b for more details on how lab data was compiled for graphite. They were refined for Draine 2003 an and b. 

--------------

Exoplanet/Brown Dwarf Papers:

There are a few C/O ratios and temperatures where graphite is the expected dominant form of carbon in a planetary atmospheres. 

Moses (2013) (Figure 7)
https://ui.adsabs.harvard.edu/abs/2013ApJ...777...34M/abstract

Carbon as a photochemical soot has been shown to be stable and potentially form on terminators of hot Jupiters like HD 189733 b. 

Lavvas & Koskinen (2017)
https://ui.adsabs.harvard.edu/abs/2017ApJ...847...32L/abstract

---------------------------------
---------------------------------

Aerosol Name      : ExoHaze (1000x solar, 300K)
Name in POSEIDON  : ExoHaze_1000xSolar_300K
Database          : Mullens 2024
Wavelengths       : 0.4 - 28.6 um

Chemical Formula  : H2O + CH4 + N2 + CO2 + He Photochemical Residue 
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
He et al. (2023) 

ADS        : https://ui.adsabs.harvard.edu/abs/2024NatAs...8..182H/abstract
Paper Info :

Used the PHAZER chamber (exposing gas mixtures to UV light) to produce haze analogs for super earth and sub-Neptunes. In particular, the hazes were made from a more 'realistic' photochemical haze from atmospheres with exoplanet-like compositions (in lieu of Tholins that assume a N2-CH4 atmosphere). 

Did both film reflectance and powder + KBr pellet transmittance to get real and imaginary indices. 

The 300K, 1000x solar gas mixture that was exposed to UV light was 66% H2O, 6.6% CH4, 6.5% N2, 4.9% CO2, and 16% He.

Assumed to be amorphous due to its organic nature. 

Note that this is a follow up paper to a study where they found the 1000x solar mixture produced the most haze output.

He (2018) (https://ui.adsabs.harvard.edu/abs/2018ApJ...856L...3H/abstract)
--------------

Exoplanet/Brown Dwarf Papers:

Sub-Neptunes are known to be hazy. The best paper for background on this is the refractive index paper cited above (He (2023)). 

---------------------------------
---------------------------------

Aerosol Name      : ExoHaze (1000x solar, 400K)
Name in POSEIDON  : ExoHaze_1000xSolar_400K
Database          : Mullens 2024
Wavelengths       : 0.4 - 28.6 um

Chemical Formula  : H2O + CH4 + N2 + CO2 + He Photochemical Residue 
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
He et al. (2023) 

ADS        : https://ui.adsabs.harvard.edu/abs/2024NatAs...8..182H/abstract
Paper Info :

Used the PHAZER chamber (exposing gas mixtures to UV light) to produce haze analogs for super earth and sub-Neptunes. In particular, the hazes were made from a more 'realistic' photochemical haze from atmospheres with exoplanet-like compositions (in lieu of Tholins that assume a N2-CH4 atmosphere). 

Did both film reflectance and powder + KBr pellet transmittance to get real and imaginary indices. 

The 300K, 1000x solar gas mixture that was exposed to UV light was 56% H2O, 11% CH4, 10% CO2, 6.4% N2, 1.9% H2, and 14.7% He.

Assumed to be amorphous due to its organic nature. 

Note that this is a follow up paper to a study where they found the 1000x solar mixture produced the most haze output.

He (2018) (https://ui.adsabs.harvard.edu/abs/2018ApJ...856L...3H/abstract)
--------------

Exoplanet/Brown Dwarf Papers:

Sub-Neptunes are known to be hazy. The best paper for background on this is the refractive index paper cited above (He (2023)). 

---------------------------------
---------------------------------

Aerosol Name      : Flame Soot
Name in POSEIDON  : Soot
Database          : gCMCRT
Wavelengths       : 0.03 - 60 um

Chemical Formula  : C
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Lavvas & Koskinen (2017)

ADS        : https://ui.adsabs.harvard.edu/abs/2017ApJ...847...32L/abstract

Paper Info :

Exoplanet specific paper looking at the stability and optical properties of flame soots in Hot Jupiters. 

From the paper: 

'Among the compounds investigated, the material with the highest resistance to extreme temperatures is soot. 

Although this term encompasses a large family of combustion or pyrolysis products of carbon chemistry, all characteristic examples we found have the lowest rates of evaporation at high temperatures. 

Soot is a common product of high temperature chemistry, but its formation in a planetary atmosphere will depend on the availability of carbon species.'

They compiled many laboratory investigations of the refractive index of soot particles (see their Figure 9). They checked the consistency between real and imaginary indices of the refractive indices by using the Kramers-Kronig relation by 1) deriving the average n and k spectra from all the lab sources (solid lines in Figure 9) and 2) calculating the corresponding n and k from Kramers Kronig (dashed lines in Figure 9), which were consistent with the average spectra.

The txt file is compiled from the Kramers-Kronig data (dotted lines in Figure 9).

Here is a list of the lab sources they compiled: 

1. Chang & Charalampopoulos (1990)
https://ui.adsabs.harvard.edu/abs/1990RSPSA.430..577C/abstract

2. Stagg & Charalampopoulos (1993)
https://ui.adsabs.harvard.edu/abs/1993CoFl...94..381S/abstract

Measured at three temperatures, all very similar though (Table 4)
Paper has this in Figure 9 but doesn’t mention it in Section 4. I would imagine it was used though. 

3. Lee & Tien (1981)
From the Combustion Institute :) 
Not on ADS
https://www.sciencedirect.com/science/article/pii/S0082078481801208

4. Gavilan et al (2016)
https://ui.adsabs.harvard.edu/abs/2016A%26A...586A.106G/abstract
Soot nanoparticles films were produced in an ethylene (C2H4) flame

--------------

Exoplanet/Brown Dwarf Papers:

There are a few C/O ratios and temperatures where graphite is the expected dominant form of carbon in a planetary atmospheres. 

Moses (2013) (Figure 7)
https://ui.adsabs.harvard.edu/abs/2013ApJ...777...34M/abstract

Carbon as a photochemical soot has been shown to be stable and potentially form on terminators of hot Jupiters like HD 189733 b. 

Lavvas & Koskinen (2017)
https://ui.adsabs.harvard.edu/abs/2017ApJ...847...32L/abstract

---------------------------------
---------------------------------

Aerosol Name      : 1-Hexene (Liquid)
Name in POSEIDON  : Hexene
Database          : WS15
Wavelengths       : 2 - 25 um

Chemical Formula  : C[6]H[12]
Crystal or Amorph : N/A
Crystal Shape     : N/A

Refractive Index References:
--------------
Anderson (2000)

ADS        : https://apps.dtic.mil/sti/citations/ADA379578 (not on ADS)

Paper Info :

A master's thesis that measured the indices of many organic fuels. They measured the refractive indices of 1-Hexene, which is C[6]H[12] (linear alpha olefin, which is used in industry a lot).

Indices are found in table 3 and match the txt file.

Liquid indices were measured.
--------------

Exoplanet/Brown Dwarf Papers:

The upper atmospheres (stratospheres) of solar system gas giants can have complex carbon chemistry (due to the photolysis of methane). 

While C2H2 (acetylene), C2H4 (ethene), and C2H6 (ethane) are expected to form initially from methane photolysis, as the photolyzed products 'rain' down they can grow more complex in structure (and potentially form complex hydrocarbons like hexene). 

Note that hexane is C6H14 while hexane is C6H12

Giant Planets of Our Solar System by Irwin (2009), chapter 4 (methane photolysis in solar system giants)

Serigano (2022) (Cassini Grand Finale, upper atmosphere of Saturn)
https://ui.adsabs.harvard.edu/abs/2022JGRE..12707238S/abstract

Fletcher (2023) (JWST MIRI MRS data of Saturn, High-res stratospheric hydrocarbon analysis)
https://ui.adsabs.harvard.edu/abs/2023JGRE..12807924F/abstract

---------------------------------
---------------------------------

Aerosol Name      : Sulfuric Acid (Liquid)
Name in POSEIDON  : H2SO4
Database          : Mullens 2024 
Wavelengths       : 0.36 - 25 um

Chemical Formula  : H[2]SO[4]
Crystal or Amorph : N/A
Crystal Shape     : N/A

Refractive Index References:
--------------
Palmer & Williams (1975) 

ADS        : https://ui.adsabs.harvard.edu/abs/1975ApOpt..14..208P/abstract

Paper Info :

Measurements made of 300K sulfuric acid solutions having different concentrations by weight ( 95.6, 84.5, 75, 50, 38, and 25%), with the other weight being water.

Used Kramers-Kronig analysis to obtain values of n and k from reflection measurements in the intermediate infrared (400 to 4000 1/cm).

I used the 84.5% columns of Table 1 and 2 since it had the most complete shortest wavelengths. 

I made the 0.702 to 0.360 imaginary indices were made with an exponential fit (scipy curve fit) since the table had no imaginary index data for those recorded real indices.

( made the 2.564 to 2.770 imaginary indices were made with exponential fit, and then offset down by 0.005 to fit the data (table had no imaginary index data for those recorded real indices).

--------------

I was pointed towards these indices by Michael Radke, who has a database of Venusian-like aerosols. 

Exoplanet/Brown Dwarf Papers:

Aqueous sulfuric acid forms a thick cloud deck on Venus by photolysis of carbon dioxide, sulfur dioxide, and water vapor. 

---------------------------------
---------------------------------

Aerosol Name      : Cyclo-Octasulfur (Orthorhombic Sulfur, alpha Sulfur)
Name in POSEIDON  : S8
Database          : gCMCRT
Wavelengths       : 0.4e-4 - 30.3 um

Chemical Formula  : S[8]
Crystal or Amorph : Crystalline 
Crystal Shape     : Orthorhombic

Refractive Index References:
--------------
Fuller, Downing, & Querry in Palik (1998) (Volume 3, Section 42)

ADS        : https://books.google.com/books/about/Handbook_of_Optical_Constants_of_Solids.html?id=nxoqxyoHfbIC (no ADS link)

Paper Info :

Orthorhombic sulfur (also known as octasulfur, or alpha-S), only has one stable molecule, which is cyclo-octasulfur (also known as lambda sulfur). From lambda sulfur, you get two different kind of crystal polymorphs: alpha-S (strongly anisotropic orthorhombic crystals) and beta-S (weakly anisotropic monoclinic crystals). Alpha-S is stable at room temperature and turns into beta-S above 368K.

Palik compiled many different sources and only records one polarization axis, specifically the polarizations along the an and b axes. They note that while orthorhombic crystals are usually biaxial, they average the an and b axes to make an effective 'ordinary ray'. 

Table 1, n_perp and k_perp matches the txt file. 
--------------

Exoplanet/Brown Dwarf Papers:

The photochemical destruction of H2S can result in the formation of S8 crystals in hot Jupiter atmospheres.

Gao et al. (2017)
https://ui.adsabs.harvard.edu/abs/2017AJ....153..139G/abstract

---------------------------------
---------------------------------

Aerosol Name      : Saturn Phosphorus Haze
Name in POSEIDON  : Saturn-Phosphorus-Haze
Database          : Mullens 2024
Wavelengths       : 0.25 - 20 um

Chemical Formula  : Phosphorus Photochemical Residue (Diphosphine proxy)
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Noy (1981) (0.25-0.6 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1981JGR....8611985N/abstract

Paper Info :

Simulated the Jovian atmosphere with H2 gas and 3% PH3. Irradiated the gas with a mercury lamp to simulate photolysis. Produced a yellow solid material that is different than red phosphorus one can buy, and different from yellow combustible phosphorus.

Preformed absorption measurements on a 1 um thick film to get imaginary indices. Did not record a real index. See Table 1. 
--------------
Fletcher (2023) (0.6-20 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2023JGRE..12807924F/abstract

Paper Info :

See section 4.3. Freely retrieved a constant imaginary index for both the lower cloud layer (which represents the highest cloud, NH3 or NH4SH) and the upper cloud layer (the lowest photochemical haze layer, PH3 photolysis into P2H4 perhaps).

They find an imaginary index for the upper cloud layer of 5e-3. 
--------------

How I made the indices:

I followed the suggestions of Sromovsky (2019) (https://ui.adsabs.harvard.edu/abs/2020Icar..34413398S/abstract), Section 3.4.2.

I used a real index of 1.82 for all wavelengths, which corresponds to white phosphorus (P4), which has no near-IR features. This value is close to the value of diphosphine (P2H4) at 195K recorded by Wohlfarth (2008). [Wohlfarth is a textbook that just reports the real index of P2H4 at different temperatures using white light, is 1.74 at 195K].

I hen divide the imaginary indices of Noy (1981) by 10, as suggested, and anything past 0.6 um utilized the retrieved imaginary index 5e-3 from Fletcher (2023).

Exoplanet/Brown Dwarf Papers:

Fletcher (2023) (JWST MIRI MRS data of Saturn, discusses the proposed P2H4 haze)
https://ui.adsabs.harvard.edu/abs/2023JGRE..12807924F/abstract

---------------------------------
---------------------------------

Aerosol Name      : Soot (6mm above a flame)
Name in POSEIDON  : Soot_6mm
Database          : Mullens 2024
Wavelengths       : 0.2 - 28.4 um

Chemical Formula  : C
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Chang & Charalampopoulos (1990)

ADS        : https://ui.adsabs.harvard.edu/abs/1990RSPSA.430..577C/abstract

Paper Info :

Indices were taken from Table 3. 

Measured refractive indices as a function of height above burner that was producing soot. This is because above a flame, the temperature is different as well as the particle size. Particle sizes grow with height above flame because its colder. The 6mm height data was taken.
--------------

Exoplanet/Brown Dwarf Papers:

There are a few C/O ratios and temperatures where graphite is the expected dominant form of carbon in a planetary atmospheres. 

Moses (2013) (Figure 7)
https://ui.adsabs.harvard.edu/abs/2013ApJ...777...34M/abstract

Carbon as a photochemical soot has been shown to be stable and potentially form on terminators of hot Jupiters like HD 189733 b. 

Lavvas & Koskinen (2017)
https://ui.adsabs.harvard.edu/abs/2017ApJ...847...32L/abstract

---------------------------------
---------------------------------

Aerosol Name      : Tholin ('Titan Tholins')
Name in POSEIDON  : Tholin
Database          : WS15
Wavelengths       : 0.0099 - 1000 um

Chemical Formula  : N2 + CH4 Photochemical Residual
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Khare (1984) (0.01 to 0.2, 1.1 to 1000 um)

ADS        : https://ui.adsabs.harvard.edu/abs/1984Icar...60..127K/abstract

Paper Info :

Exposed a 'Titan-like' atmosphere to electrical discharge for 4 months to produce a dark-red solid. They tried to take reflectivity measurements of optically thick, pressed pellets, which proved difficult. They instead took transmission measurements of optically thin films instead. 

Gas mixture of 90% N2 + 10% CH4 at 0.2 mbar. 

Txt file doesn't match Table 1 in table directly, so it's assumed Figure 4 generated the txt file with a plot digitizer.

The paper notes that the 0.2 to 0.4 um region of their measurements don't match their Kramers-Kronig analysis, and therefore is suspect to error (which is why the Ramirez indices were used).

Assumed amorphous since its an organic photochemical product.
--------------
Ramirez (2002) (0.2-1 um)

ADS        : https://ui.adsabs.harvard.edu/abs/2002Icar..156..515R/abstract

Paper Info :

Short wavelength (0.2 - 0.9 um) measurements using transmittance and reflectance. 

Used a 98% N2 + 2% CH4 mixture since the composition of Titan's atmosphere had been better constrained.

Produced thin films using cold plasma discharge at 2mbar at room temperature. 

I'm assuming Table 2 was used and interpolated over (since it doesn't match txt file directly).

Assumed amorphous since its an organic photochemical product.
--------------

Exoplanet/Brown Dwarf Papers:

Tholins are found to form high in Titan's atmosphere. Tholins are often used to simulate photochemical hazes in exoplanet papers, even though they are known to be very absorptive. 

Kempton (2023) (Tholins used to model phase curve, bad fit)
https://ui.adsabs.harvard.edu/abs/2023Natur.620...67K/abstract

He (2023) (Transmission spectra of tholins compared to other hazes)
https://ui.adsabs.harvard.edu/abs/2024NatAs...8..182H/abstract

---------------------------------
---------------------------------

Aerosol Name      : Tholin (C/O = 1)
Name in POSEIDON  : Tholin-CO-1
Database          : Mullens 2024
Wavelengths       : 0.13 - 9.99 um

Chemical Formula  : N2 + CO2 + CH4 Photochemical Residue
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Corrales (2023)

ADS        : https://ui.adsabs.harvard.edu/abs/2023ApJ...943L..26C/abstract

Paper Info :

Explores how oxygen influences Tholin properties (the Khare (1984) experiments had no oxygen in their apparatus) by exploring different C/O ratios.

Lab measurements for the four Tholin samples were originally made in Gavilan (2018) (https://iopscience.iop.org/article/10.3847/1538-4357/aac8df).

Complex index of refraction is derived from transmission measurements of Gavilan (2018) with interpolation from 1 to 1.05 um. Real indices were derived from the Kramers-Kronig relation. 

C/O = 1 results in a gas mixture of 90% N2, 5% CO2, 5% CH4. 

Assumed amorphous since its an organic photochemical product.
--------------

Exoplanet/Brown Dwarf Papers:

Tholins are found to form high in Titan's atmosphere. Tholins are often used to simulate photochemical hazes in exoplanet papers, even though they are known to be very absorptive. These oxygen-rich Tholins were made to better simulate sub-Neptune/super-Earth atmospheres. 

Kempton (2023) (Tholins used to model phase curve, bad fit)
https://ui.adsabs.harvard.edu/abs/2023Natur.620...67K/abstract

He (2023) (Transmission spectra of tholins compared to other hazes)
https://ui.adsabs.harvard.edu/abs/2024NatAs...8..182H/abstract

---------------------------------
---------------------------------

Aerosol Name      : Tholin (C/O = 0.625)
Name in POSEIDON  : Tholin-CO-0625
Database          : Mullens 2024
Wavelengths       : 0.13 - 9.99 um

Chemical Formula  : N2 + CO2 + CH4 Photochemical Residue
Crystal or Amorph : Amorphous (assumed, see below)
Crystal Shape     : N/A

Refractive Index References:
--------------
Corrales (2023)

ADS        : https://ui.adsabs.harvard.edu/abs/2023ApJ...943L..26C/abstract

Paper Info :

Explores how oxygen influences Tholin properties (the Khare (1984) experiments had no oxygen in their apparatus) by exploring different C/O ratios.

Lab measurements for the four Tholin samples were originally made in Gavilan (2018) (https://iopscience.iop.org/article/10.3847/1538-4357/aac8df).

Complex index of refraction is derived from transmission measurements of Gavilan (2018) with interpolation from 1 to 1.05 um. Real indices were derived from the Kramers-Kronig relation. 

C/O = 0.625 results in a gas mixture of 90% N2, 8% CO2, 2% CH4. (Near Solar)

Assumed amorphous since its an organic photochemical product.
--------------

Exoplanet/Brown Dwarf Papers:

Tholins are found to form high in Titan's atmosphere. Tholins are often used to simulate photochemical hazes in exoplanet papers, even though they are known to be very absorptive. These oxygen-rich Tholins were made to better simulate sub-Neptune/super-Earth atmospheres. 

Kempton (2023) (Tholins used to model phase curve, bad fit)
https://ui.adsabs.harvard.edu/abs/2023Natur.620...67K/abstract

He (2023) (Transmission spectra of tholins compared to other hazes)
https://ui.adsabs.harvard.edu/abs/2024NatAs...8..182H/abstract
