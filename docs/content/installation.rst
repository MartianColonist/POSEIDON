Installation
============

Step 1: Download opacity database and stellar grids
___________________________________________________

Two of the key inputs for POSEIDON, stellar models and chemical opacity data,
are stored separately from the GitHub repository (due to file size limitations).
Before installing POSEIDON, you will need to download these input files 
(which amount to around 35 GB):

* `Opacity_database_0.01cm-1.hdf5 <https://drive.google.com/file/d/1Rk_6sbIYC8c9La0fWHWpMPve6Jik7a3h/view?usp=sharing>`_
* `Opacity_database_cia.hdf5 <https://drive.google.com/file/d/1HA3gZUTmDIzZGFLTtuiPe6VDUxstxjZ_/view?usp=sharing>`_
* `Stellar_grids.zip <https://drive.google.com/file/d/1xZzbVserwHZx0jmmhhEeQzk5RnxjFf2C/view?usp=sharing>`_

Place the two opacity files in a directory of your choice and unzip the stellar 
grid folder.

.. attention::
   For Windows users, we recommend installing `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/about>`_
   before proceeding. WSL provides a Linux environment you can use on Windows.
   
   One of the main modules used by POSEIDON, PyMultiNest, does not natively 
   support Windows. However, it will work fine if you use WSL to install and 
   run POSEIDON.

Next, you need to create two new environment variables: one pointing to the 
directory containing the opacity files and a second to the (unzipped) stellar
grid folder. You can do this on Linux by entering into a terminal 

.. code-block:: bash

   echo 'export POSEIDON_input_data="/PATH/TO/YOUR/OPACITY/DIRECTORY/"' >>~/.bashrc
   echo 'export PYSYN_CDBS="/PATH/TO/stellar_grids/"' >>~/.bashrc

On Mac OS, instead use
   
.. code-block:: bash

   echo 'export POSEIDON_input_data="/PATH/TO/YOUR/OPACITY/DIRECTORY/"' >>~/.bash_profile
   echo 'export PYSYN_CDBS="/PATH/TO/stellar_grids/"' >>~/.bash_profile

Now POSEIDON will know where to find the opacity database and stellar models.


Step 2: Install PyMultiNest
___________________________

MultiNest is the main sampling algorithm used for parameter space exploration
in POSEIDON retrievals. MultiNest has a convenient Python wrapper, PyMultiNest.

You can install *both* MultiNest and PyMultiNest in a single line via 
conda-forge (don't follow the complicated instructions you might see elsewhere).

.. code-block:: bash

    conda install -c conda-forge pymultinest
   

Step 3: Install POSEIDON from GitHub
____________________________________

Now all that is left is the clone POSEIDON from GitHub and install the module.

.. code-block:: bash
		
   git clone https://github.com/MartianColonist/POSEIDON_public.git
   cd POSEIDON_public
   python setup.py install

And that, splendidly, is all there is to it. Onwards, to the tutorials!