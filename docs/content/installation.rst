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
* `Opacity_database_0.01cm-1_Temperate.hdf5 <https://drive.google.com/file/d/1hYLTzIy7cVicqGU8LHmLnq-3WQuyKISX/view?usp=sharing>`_
* `Stellar_grids.zip <https://drive.google.com/file/d/1xZzbVserwHZx0jmmhhEeQzk5RnxjFf2C/view?usp=sharing>`_

Place the three opacity files in a directory called 'opacity' in a location of 
your choice and unzip the stellar grid folder. I usually place these in an
'input' folder with the structure 'input/opacity' and 'input/stellar_grids'.

.. note::
   For Windows users, we recommend installing `Windows Subsystem for Linux (WSL) 
   <https://docs.microsoft.com/en-us/windows/wsl/about>`_
   before proceeding. WSL provides a Linux environment you can use on Windows.
   
   One of the main modules used by POSEIDON, PyMultiNest, does not natively 
   support Windows. However, it will work fine if you use WSL to install and 
   run POSEIDON.

Next, you need to create two new environment variables: one pointing to the 
directory containing the opacity files and a second to the (unzipped) stellar
grid folder. 

* Linux environment variables:
  
Enter the following lines into a terminal:

.. code-block:: bash

   echo 'export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/opacity/"' >>~/.bashrc
   echo 'export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/stellar_grids/"' >>~/.bashrc

You should replace the bold text above with the location of your input directory.

Alternatively, you can just open your .bashrc file (a hidden file in your home 
directory) in a text editor and add the following two lines at the bottom:

.. code-block:: bash

   export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/opacity/"
   export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/stellar_grids/"

* Mac OS environment variables:

Check whether your terminals use bash (older OS) or zsh (newer OS).

For bash terminals, enter the following lines:
   
.. code-block:: bash

   echo 'export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/opacity/"' >>~/.bash_profile
   echo 'export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/stellar_grids/"' >>~/.bash_profile

For zsh terminals, instead enter the following lines:
   
.. code-block:: bash

   echo export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/opacity/" >>~/.zshrc
   echo export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗜𝗡𝗣𝗨𝗧/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/stellar_grids/" >>~/.zshrc

Now POSEIDON will know where to find the opacity database and stellar models.


Step 2: Install mpi4py and PyMultiNest
______________________________________

.. attention::
   We recommend installing POSEIDON in a fresh Anaconda environment. You can
   create a new Python 3.9 environment via :

   .. code-block:: bash

    conda create --name 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘 python=3.9

MultiNest is the main sampling algorithm used for parameter space exploration
in POSEIDON retrievals. MultiNest has a convenient Python wrapper, PyMultiNest.

You first need to install mpi4py, which is used by PyMultiNest for parallel
computations on multiple cores.

.. code-block:: bash

    conda install -c conda-forge mpi4py

Then you can install *both* MultiNest and PyMultiNest in a single line via 
conda-forge (you might see way more complicated instructions elsewhere, this
is the simplest way!).

.. code-block:: bash

    conda install -c conda-forge pymultinest
   

Step 3: Install POSEIDON from GitHub
____________________________________

Now all that is left is to obtain POSEIDON from GitHub and install the module.
You can download `POSEIDON from GitHub <https://github.com/MartianColonist/POSEIDON_rev>`_
or clone the repository:

.. code-block:: bash
		
   git clone https://github.com/MartianColonist/POSEIDON_dev.git

Then navigate into the 'POSEIDON_dev' directory and install the package via:

.. code-block:: bash
		
   pip install .

And that, splendidly, is all there is to it. Onwards to the tutorials!
