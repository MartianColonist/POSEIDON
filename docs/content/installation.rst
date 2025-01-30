Installation
============

Step 1: Download input files
____________________________

POSEIDON requires various input files (opacity data, stellar models, etc.) that
are stored separately from the GitHub repository (due to file size limitations).
Before installing POSEIDON, you will need to download these input files, which
have been packaged for convenience as a single .zip file (67 GB):

* `inputs.zip <https://drive.google.com/file/d/1D4w1GdM2cYpwzLhVg_k26iC3xHilibfY/view?usp=sharing>`_

You can also find all the required input files on `Zenodo 
<https://zenodo.org/record/7927435#.ZF22itLMJhE>`_.


Step 2: Install mpi4py and PyMultiNest
______________________________________

While waiting for the input files to download, the next step is the install
the dependencies for POSEIDON: PyMultiNest and mpi4py.

.. note::
   For Windows users, we recommend installing `Windows Subsystem for Linux (WSL) 
   <https://docs.microsoft.com/en-us/windows/wsl/about>`_ before proceeding.
   WSL provides a Linux environment you can use in Windows.
   
   We recommend using WSL because PyMultiNest does not natively support Windows.
   However, it will work fine if you use WSL to install and run POSEIDON.

We recommend installing POSEIDON in a fresh `Anaconda <https://www.anaconda.com/>`_ 
environment. The following instructions show you how to create the conda environment
on either Linux (e.g. Ubuntu) or Mac OS.

Linux conda environment setup
-----------------------------

POSEIDON currently supports Python versions up to 3.11.9. You can create a new 
anaconda environment with, say, the latest version of Python 3.10 via:

.. code-block:: bash

   conda create --name 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘 python=3.10

Once the basic Python packages are installed in this fresh environment, you
can activate the environment where POSEIDON will dwell:

.. code-block:: bash

   conda activate 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘

Mac OS conda environment setup
------------------------------

For Mac OS, you need to set the CONDA_SUBDIR environment variable to osx-64 
(because PyMultiNest does not currently support ARM chips). You can do this via:

.. code-block:: bash

   CONDA_SUBDIR=osx-64 conda create -n 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘 python=3.10
   conda activate 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘
   conda env config vars set CONDA_SUBDIR=osx-64

Installing PyMultiNest
----------------------

`MultiNest <https://academic.oup.com/mnras/article/398/4/1601/981502>`_ is the 
main sampling algorithm used for parameter space exploration in POSEIDON retrievals. 
MultiNest has a convenient Python wrapper, `PyMultiNest 
<https://johannesbuchner.github.io/PyMultiNest/>`_.

You first need to install mpi4py, which is used by PyMultiNest for parallel
computations on multiple cores. With your conda environment activated, call:

.. code-block:: bash

    conda install -c conda-forge mpi4py

Then you can install *both* MultiNest and PyMultiNest in a single line via 
conda-forge (you might see way more complicated instructions elsewhere, this
is the simplest way!).

.. code-block:: bash

    conda install -c conda-forge pymultinest
   

Step 3: Install POSEIDON from GitHub
____________________________________

Now you are ready to download and install POSEIDON. You can download 
`POSEIDON from GitHub <https://github.com/MartianColonist/POSEIDON>`_ 
or clone the repository:

.. code-block:: bash
		
   git clone https://github.com/MartianColonist/POSEIDON.git

Then navigate into the top-level :code:`POSEIDON` directory and install the 
package via:

.. code-block:: bash
		
   cd POSEIDON
   pip install -e .


Step 4: Set input file environment variables
____________________________________________

By this point, the input files should have hopefully finished downloading. 

Place :code:`inputs.zip` in your top-level :code:`POSEIDON` folder (the one 
containing :code:`setup.py`, :code:`README`, etc.) and unzip it:

.. code-block:: bash

   unzip inputs.zip

You should now have an :code:`inputs` folder with three subdirectories: 
:code:`inputs/opacity`, :code:`inputs/stellar_grids`, and :code:`inputs/chemistry_grids` 

Now all that is left to do it to create environment variables telling POSEIDON
where to find the input files.

Linux environment variables
---------------------------
  
If you are using Linux, enter the following lines into a terminal:

.. code-block:: bash

   echo 'export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/"' >>~/.bashrc
   echo 'export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/"' >>~/.bashrc

You should replace the bold text above with the location of your POSEIDON directory.

Alternatively, you can just open your .bashrc file (a hidden file in your Home 
directory) with a text editor and add the following two lines at the bottom:

.. code-block:: bash

   export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/"
   export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/"

Mac OS environment variables
----------------------------

Setting environment variables on macOS differs depending on your OS version
(thanks to Apple's infinite wisdom). 

If you are using macOS >= 10.15 your default terminal will be zsh, for which
you can set the environment variables like so:

.. code-block:: bash

   echo export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/" >>~/.zshrc
   echo export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/" >>~/.zshrc

Alternatively, for earlier macOS versions, the default terminal is bash:
   
.. code-block:: bash

   echo 'export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/"' >>~/.bash_profile
   echo 'export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/"' >>~/.bash_profile


Now POSEIDON will know where to find the input files.

And that, splendidly, is all there is to it. Onwards to the tutorials!

The best place to begin is the quick start guide 
`"Generating Transmission Spectra" <notebooks/transmission_basic.html>`_
