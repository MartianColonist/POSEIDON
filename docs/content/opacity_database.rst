Opacity Database
================


Molecular, atomic, and aerosol opacities are a required input to any radiative 
transfer codes. POSEIDON's opacity database includes high-resolution 
(:math:`R \sim 10^6`) line-by-line absorption cross sections and continuum 
absorption processes, alongside Mie scattering extinction cross sections, 
single scattering albedos, and asymmetry parameters.


Line-by-line Cross Sections
___________________________

POSEIDON v1.2 includes a comprehensive update to the opacity database to reflect 
recent theoretical and experimental advances:

* New line lists.
* Updated :math:`\mathrm{H_2}` + :math:`\mathrm{He}` broadening.
* Improved line position accuracy for high-resolution analyses (i.e. ExoMol's MARVEL procedure).
* UV-Visible wavelength coverage (down to :math:`0.2 \, \mathrm{\mu m}`, where available).

The current line-by-line opacity sources included in POSEIDON are summarised below:

.. list-table::
   :widths: 20 20 20 20 20 20
   :header-rows: 1

   * - Species
     - Line List
       
       (Version)
     - Reference
     - Broadening
     - Updates vs. 
     
       POSEIDON 1.0
     - Plot

       (Click)
   * - :math:`\mathrm{H_2O}`
     - `POKAZATEL <https://www.exomol.com/data/molecules/H2O/1H2-16O/POKAZATEL/>`_
       
       (2023-06-21)
     - `Polyansky et al. <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.2597P/abstract>`_

       `(2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.2597P/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - MARVELised 
     - 
       .. image:: ../_static/opacity_previews/H2O.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CO_2}`
     - `UCL-4000 <https://www.exomol.com/data/molecules/CO2/12C-16O2/UCL-4000/>`_
       
       (2020-06-30)
     - `Yurchenko et al. <https://ui.adsabs.harvard.edu/abs/2020MNRAS.496.5282Y/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.496.5282Y/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/CO2.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CH_4}`
     - `MM <https://www.exomol.com/data/molecules/CH4/12C-1H4/MM/>`_
       
       (2024-01-13)
     - `Yurchenko et al. <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.3719Y/abstract>`_

       `(2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.3719Y/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/CH4.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CO}`
     - `Li2015 <https://www.exomol.com/data/molecules/CO/12C-16O/Li2015/>`_
       
       (2017-01-31)
     - `Li et al. <https://ui.adsabs.harvard.edu/abs/2015ApJS..216...15L/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015ApJS..216...15L/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 6 Isotopes*
     - 
       .. image:: ../_static/opacity_previews/CO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Na}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Na.png
          :width: 400
          :align: center

   * - :math:`\mathrm{K}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/K.png
          :width: 400
          :align: center

   * - :math:`\mathrm{NH_3}`
     - `CoYuTe <https://www.exomol.com/data/molecules/NH3/14N-1H3/CoYuTe/>`_
       
       (2020-07-30)
     - `Coles et al. <https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.4638C/abstract>`_

       `(2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.4638C/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - MARVELised 
     - 
       .. image:: ../_static/opacity_previews/NH3.png
          :width: 400
          :align: center
  
   * - :math:`\mathrm{HCN}`
     - `Harris <https://www.exomol.com/data/molecules/HCN/1H-12C-14N/Harris/>`_
       
       (2016-12-05)
     - `Barber et al. <https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1828B/abstract>`_

       `(2014) <https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1828B/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - No Change
     - 
       .. image:: ../_static/opacity_previews/HCN.png
          :width: 400
          :align: center

   * - :math:`\mathrm{SO_2}`
     - `ExoAmes <https://www.exomol.com/data/molecules/SO2/32S-16O2/ExoAmes/>`_
       
       (2017-01-31)
     - `Underwood et al. <https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.3890U/abstract>`_

       `(2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.459.3890U/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - MARVELised
     - 
       .. image:: ../_static/opacity_previews/SO2.png
          :width: 400
          :align: center

   * - :math:`\mathrm{H_2 S}`
     - `AYT2 <https://www.exomol.com/data/molecules/H2S/1H2-32S/AYT2/>`_
       
       (2022-09-18)
     - `Azzam et al. <https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.4063A/abstract>`_

       `(2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.4063A/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/H2S.png
          :width: 400
          :align: center

   * - :math:`\mathrm{PH_3}`
     - `SAlTY <https://www.exomol.com/data/molecules/PH3/31P-1H3/SAlTY/>`_
       
       (2017-01-31)
     - `Sousa-Silva et al. <https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.2337S/abstract>`_

       `(2014) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.2337S/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - No Change
     - 
       .. image:: ../_static/opacity_previews/PH3.png
          :width: 400
          :align: center

   * - :math:`\mathrm{C_2 H_2}`
     - `aCeTY <https://www.exomol.com/data/molecules/C2H2/12C2-1H2/aCeTY/>`_
       
       (2022-09-18)
     - `Chubb et al. <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.1531C/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.1531C/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/C2H2.png
          :width: 400
          :align: center

   * - :math:`\mathrm{OCS}`
     - `OYT8 <https://www.exomol.com/data/molecules/OCS/16O-12C-32S/OYT8/>`_
       
       (2024-04-25)
     - `Owens et al. <https://ui.adsabs.harvard.edu/abs/2024MNRAS.530.4004O/abstract>`_

       `(2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.530.4004O/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/OCS.png
          :width: 400
          :align: center

   * - :math:`\mathrm{TiO}`
     - `Toto <https://www.exomol.com/data/molecules/TiO/49Ti-16O/Toto/>`_
       
       (2024-05-09)
     - `McKemmish et al. <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_

       `(2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - MARVELised 
      
       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/TiO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{VO}`
     - `VOMYT <https://www.exomol.com/data/molecules/VO/51V-16O/VOMYT/>`_
       
       (2016-07-26)
     - `McKemmish et al. <https://ui.adsabs.harvard.edu/abs/2016MNRAS.463..771M/abstract>`_

       `(2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.463..771M/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/VO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{AlO}`
     - `ATP <https://www.exomol.com/data/molecules/AlO/27Al-16O/ATP/>`_
       
       (2021-06-22)
     - `Patrascu et al. <https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3613P/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3613P/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - MARVELised 

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/AlO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{SiO}`
     - `SiOUVenIR <https://www.exomol.com/data/molecules/SiO/28Si-16O/SiOUVenIR/>`_
       
       (2021-11-05)
     - `Yurchenko et al. <https://ui.adsabs.harvard.edu/abs/2022MNRAS.510..903Y/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.510..903Y/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/SiO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CaO}`
     - `VBATHY <https://www.exomol.com/data/molecules/CaO/40Ca-16O/VBATHY/>`_
       
       (2023-02-20)
     - `Yurchenko et al. <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.4524Y/abstract>`_

       `(2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.4524Y/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/CaO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{MgO}`
     - `LiTY <https://www.exomol.com/data/molecules/MgO/24Mg-16O/LiTY/>`_
       
       (2019-04-01)
     - `Li et al. <https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2351L/abstract>`_

       `(2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.2351L/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/MgO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{NaO}`
     - `NaOUCMe <https://www.exomol.com/data/molecules/NaO/23Na-16O/NaOUCMe/>`_
       
       (2021-11-17)
     - `Mitev et al. <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2349M/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2349M/abstract>`_
     - `SB'07 <https://ui.adsabs.harvard.edu/abs/2007ApJS..168..140S/abstract>`_
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/NaO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{LaO}`
     - `BDL <https://www.exomol.com/data/molecules/LaO/139La-16O/BDL/>`_
       
       (2023-09-23)
     - `Bernath et al. <https://ui.adsabs.harvard.edu/abs/2023ApJ...953..181B/abstract>`_

       `(2023) <https://ui.adsabs.harvard.edu/abs/2023ApJ...953..181B/abstract>`_
     - `SB'07 <https://ui.adsabs.harvard.edu/abs/2007ApJS..168..140S/abstract>`_
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/LaO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{ZrO}`
     - `ZorrO <https://www.exomol.com/data/molecules/ZrO/90Zr-16O/ZorrO/>`_
       
       (2023-07-13)
     - `Perri et al. <https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.4631P/abstract>`_

       `(2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.4631P/abstract>`_
     - `SB'07 <https://ui.adsabs.harvard.edu/abs/2007ApJS..168..140S/abstract>`_
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/ZrO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{SO}`
     - `SOLIS <https://www.exomol.com/data/molecules/SO/32S-16O/SOLIS/>`_
       
       (2023-09-14)
     - `Brady et al. <https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6675B/abstract>`_

       `(2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.6675B/abstract>`_
     - Fixed 
     
       :math:`\gamma_L = 0.07`

       :math:`n_L = 0.5`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/SO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{NO}`
     - `XABC <https://www.exomol.com/data/molecules/NO/14N-16O/XABC/>`_
       
       (2021-04-22)
     - `Qu et al. <https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.5768Q/abstract>`_

       `(2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.5768Q/abstract>`_
     - Air 
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/NO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{PO}`
     - `POPS <https://www.exomol.com/data/molecules/PO/31P-16O/POPS/>`_
       
       (2017-09-10)
     - `Qu et al. <https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3648P/abstract>`_

       `(2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3648P/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/PO.png
          :width: 400
          :align: center

   * - :math:`\mathrm{TiH}`
     - `MoLLIST <https://www.exomol.com/data/molecules/TiH/48Ti-1H/MoLLIST/>`_
       
       (2016-07-26)
     - `Bernath <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List
      
       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/TiH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CrH}`
     - `MoLLIST <https://www.exomol.com/data/molecules/CrH/52Cr-1H/MoLLIST/>`_
       
       (2016-07-26)
     - `Bernath <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List
      
       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/CrH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{FeH}`
     - `MoLLIST <https://www.exomol.com/data/molecules/FeH/56Fe-1H/MoLLIST/>`_
       
       (2016-07-26)
     - `Bernath <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/FeH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{ScH}`
     - `MoLLIST <https://www.exomol.com/data/molecules/ScH/45Sc-1H/LYT/>`_
       
       (2016-07-26)
     - `Lodi et al. <https://ui.adsabs.harvard.edu/abs/2015MolPh.113.1998L/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015MolPh.113.1998L/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/ScH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{AlH}`
     - `AloHa <https://www.exomol.com/data/molecules/AlH/27Al-1H/AloHa/>`_
       
       (2016-07-26)
     - `Yurchenko et al. <https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.9736Y/abstract>`_

       `(2023) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.9736Y/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/AlH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{SiH}`
     - `SiGHTLY <https://www.exomol.com/data/molecules/SiH/28Si-1H/SiGHTLY/>`_
       
       (2017-11-01)
     - `Yurchenko et al. <https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.5324Y/abstract>`_

       `(2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.5324Y/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/SiH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{BeH}`
     - `Darby-Lewis <https://www.exomol.com/data/molecules/BeH/9Be-1H/Darby-Lewis/>`_
       
       (2018-02-12)
     - `Darby-Lewis et al. <https://ui.adsabs.harvard.edu/abs/2018JPhB...51r5701D/abstract>`_

       `(2018) <https://ui.adsabs.harvard.edu/abs/2018JPhB...51r5701D/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/BeH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CaH}`
     - `XAB <https://www.exomol.com/data/molecules/CaH/40Ca-1H/XAB/>`_
       
       (2022-02-11)
     - `Owens et al. <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5448O/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5448O/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/CaH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{MgH}`
     - `XAB <https://www.exomol.com/data/molecules/MgH/24Mg-1H/XAB/>`_
       
       (2022-02-11)
     - `Owens et al. <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5448O/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5448O/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/MgH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{LiH}`
     - `CLT <https://www.exomol.com/data/molecules/LiH/7Li-1H/CLT/>`_
       
       (2016-09-27)
     - `Coppola et al. <https://ui.adsabs.harvard.edu/abs/2011MNRAS.415..487C/abstract>`_

       `(2011) <https://ui.adsabs.harvard.edu/abs/2011MNRAS.415..487C/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/LiH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{NaH}`
     - `Rivlin <https://www.exomol.com/data/molecules/NaH/23Na-1H/Rivlin/>`_
       
       (2016-09-27)
     - `Rivlin et al. <https://ui.adsabs.harvard.edu/abs/2015MNRAS.451..634R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.451..634R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/NaH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{OH}`
     - `MoLLIST <https://www.exomol.com/data/molecules/OH/16O-1H/MoLLIST/>`_
       
       (2018-07-19)
     - `Bernath <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List
     
       Air Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/OH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{OH^{+}}`
     - `MoLLIST <https://www.exomol.com/data/molecules/OH_p/16O-1H_p/MoLLIST/>`_
       
       (2022-07-13)
     - `Bernath <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/OH+.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CH}`
     - `MoLLIST <https://www.exomol.com/data/molecules/CH/12C-1H/MoLLIST/>`_
       
       (2019-02-14)
     - `Bernath <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_

       `(2020) <https://ui.adsabs.harvard.edu/abs/2020JQSRT.24006687B/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/CH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{NH}`
     - `kNigHt <https://www.exomol.com/data/molecules/NH/14N-1H/kNigHt/>`_
       
       (2024-03-01)
     - `Perri et al. <https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.3023P/abstract>`_

       `(2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.3023P/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/NH.png
          :width: 400
          :align: center
  
   * - :math:`\mathrm{SH}`
     - `GYT <https://www.exomol.com/data/molecules/SH/32S-1H/GYT/>`_
       
       (2019-08-01)
     - `Gorman et al. <https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.1652G/abstract>`_

       `(2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.1652G/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List
     
       Air Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/SH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{PN}`
     - `PaiN <https://www.exomol.com/data/molecules/PN/31P-14N/PaiN/>`_
       
       (2024-05-05)
     - `Semenov et al.`

       `(2024)`
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - New Line List

       SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/PN.png
          :width: 400
          :align: center

   * - :math:`\mathrm{PS}`
     - `POPS <https://www.exomol.com/data/molecules/PS/31P-32S/POPS/>`_
       
       (2017-09-10)
     - `Prajapat et al. <https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3648P/abstract>`_

       `(2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3648P/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - SB'07 Broad to 

       :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - 
       .. image:: ../_static/opacity_previews/PS.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CS}`
     - `JnK <https://www.exomol.com/data/molecules/CS/12C-32S/JnK/>`_
       
       (2016-07-26)
     - `Paulose et al. <https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.1931P/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.1931P/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/CS.png
          :width: 400
          :align: center

   * - :math:`\mathrm{C_2}`
     - `8states <https://www.exomol.com/data/molecules/C2/12C2/8states/>`_
       
       (2020-06-28)
     - `Yurchenko et al. <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.3397Y/abstract>`_

       `(2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.3397Y/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/C2.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CH_3}`
     - `AYYJ <https://www.exomol.com/data/molecules/CH3/12C-1H3/AYYJ/>`_
       
       (2019-05-01)
     - `Adam et al. <https://ui.adsabs.harvard.edu/abs/2019JPCA..123.4755A/abstract>`_

       `(2019) <https://ui.adsabs.harvard.edu/abs/2019JPCA..123.4755A/abstract>`_
     - Fixed 
     
       :math:`\gamma_L = 0.053`

       :math:`n_L = 0.5`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/CH3.png
          :width: 400
          :align: center

   * - :math:`\mathrm{H_{3}^{+}}`
     - `MiZATeP <https://www.exomol.com/data/molecules/OH_p/16O-1H_p/MoLLIST/>`_
       
       (2017-03-30)
     - `Mizus et al. <https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.1717M/abstract>`_

       `(2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.1717M/abstract>`_
     - Fixed 
     
       :math:`\gamma_L = 0.07`

       :math:`n_L = 0.5`
     - No Change
     - 
       .. image:: ../_static/opacity_previews/H3+.png
          :width: 400
          :align: center

   * - :math:`\mathrm{N_2 O}`
     - `HITEMP-2020 <https://hitran.org/hitemp/>`_
     - `Hargreaves et al. <https://ui.adsabs.harvard.edu/abs/2019JQSRT.232...35H/abstract>`_

       `(2019) <https://ui.adsabs.harvard.edu/abs/2019JQSRT.232...35H/abstract>`_
     - Air
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/N2O.png
          :width: 400
          :align: center

   * - :math:`\mathrm{NO_2}`
     - `HITEMP-2020 <https://hitran.org/hitemp/>`_
     - `Hargreaves et al. <https://ui.adsabs.harvard.edu/abs/2019JQSRT.232...35H/abstract>`_

       `(2019) <https://ui.adsabs.harvard.edu/abs/2019JQSRT.232...35H/abstract>`_
     - Air
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/NO2.png
          :width: 400
          :align: center

   * - :math:`\mathrm{C_2 H_4}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?90=on>`_
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_
     - Air
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/C2H4.png
          :width: 400
          :align: center

   * - :math:`\mathrm{C_2 H_6}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?78=on>`_
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_
     - Air
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/C2H6.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CH_3 CN}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?95=on>`_
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_
     - Air
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/CH3CN.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CH_3 OH}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?92=onn>`_
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_
     - Air
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/CH3OH.png
          :width: 400
          :align: center

   * - :math:`\mathrm{GeH_4}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?139=on>`_
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_
     - Air
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/GeH4.png
          :width: 400
          :align: center

   * - :math:`\mathrm{CS_2}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?131=on>`_
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_
     - Air
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/CS2.png
          :width: 400
          :align: center

   * - :math:`\mathrm{O_2}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?36=on>`_
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_
     - Air
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/O2.png
          :width: 400
          :align: center

   * - :math:`\mathrm{O_3}`
     - `HITRAN-2020 <https://hitran.org/lbl/3?16=on>`_

       Laboratory
     - `Gordon et al. <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `(2022) <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_

       `Serdyuchenko et al. <https://ui.adsabs.harvard.edu/abs/2014AMT.....7..625S/abstract>`_

       `(2014) <https://ui.adsabs.harvard.edu/abs/2014AMT.....7..625S/abstract>`_
     - Air
     - New Line List
     - 
       .. image:: ../_static/opacity_previews/O3.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Al}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/Al.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Ba}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/Ba.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Ba^{+}}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/Ba+.png
          :width: 400
          :align: center
  
   * - :math:`\mathrm{Ca}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Ca.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Ca^{+}}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Ca+.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Cr}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added Species**
     - 
       .. image:: ../_static/opacity_previews/Cr.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Cs}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Cs.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Fe}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Cs.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Fe^{+}}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Fe+.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Li}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Li.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Mg}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Mg.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Mg^{+}}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Mg+.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Mn}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Mn.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Ni}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added species**
     - 
       .. image:: ../_static/opacity_previews/Ni.png
          :width: 400
          :align: center

   * - :math:`\mathrm{O}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added species**
     - 
       .. image:: ../_static/opacity_previews/O.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Rb}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Rb.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Sc}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - **Added species**
     - 
       .. image:: ../_static/opacity_previews/Sc.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Ti}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Ti.png
          :width: 400
          :align: center

   * - :math:`\mathrm{Ti^{+}}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/Ti+.png
          :width: 400
          :align: center

   * - :math:`\mathrm{V}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/V.png
          :width: 400
          :align: center

   * - :math:`\mathrm{V^{+}}`
     - `VALD3 <https://vald.astro.uu.se/~vald/>`_
     - `Ryabchikova et al. <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_

       `(2015) <https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract>`_
     - :math:`\mathrm{H_2}` + :math:`\mathrm{He}`
     - Air Wave. to

       Vacuum
     - 
       .. image:: ../_static/opacity_previews/V+.png
          :width: 400
          :align: center



`*` For CO, POSEIDON defaults to a weighted average using terrestrial isotope ratios.
Users can also treat each CO isotopologue as separate species (e.g. `12C-16O`, 
`13C-16O`, `12C-17O`, etc.) for modelling and retrieval purposes. All other 
chemical species use cross sections for the principal isotopologue only.

:math:`\mathrm{H_2 + He}` broadening data are mostly sourced from ExoMol's 
H2.broad and He.broad files and we include the J dependence (a0). Where these 
data are not available, we use the estimated :math:`\mathrm{H_2 + He}` pressure 
broadening parameters from `Chubb et al. (2022) <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_.

The continuum opacity sources, including collision-induced absorption (CIA) and
Rayleigh scattering cross sections, are unchanged from POSEIDON v1.0.

Is your favourite molecule missing? Has a revolutionary new line list just been
released? Please address any request for new opacities to: ryanjmac@umich.edu.


Aerosol Database
________________

@ Elijah







A description of the original public release POSEIDON opacity database can be found in 
`MacDonald & Lewis (2022) <https://ui.adsabs.harvard.edu/abs/2021arXiv211105862M/abstract>`_
(Appendix C).


