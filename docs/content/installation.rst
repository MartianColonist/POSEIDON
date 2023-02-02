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

You can also find all the required input files on `Zenodo 
<https://zenodo.org/record/7500292#.Y7YLwdLMJhE>`_.


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

.. attention::
   We recommend installing POSEIDON in a fresh `Anaconda <https://www.anaconda.com/>`_ 
   environment. You can create a new Python 3.9 environment via:

   .. code-block:: bash

    conda create --name 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘 python=3.9

   Once the basic Python packages are installed in this fresh environment, you
   can activate the environment where POSEIDON will dwell:

   .. code-block:: bash

    conda activate 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘

   Note that POSEIDON currently supports Python up to 3.10.

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

By this point, the input files should have finished downloading. Let's make
an input directory to hold the files:

.. code-block:: bash

   mkdir -p inputs/opacity

Place the three opacity files you downloaded into this :code:`inputs/opacity` 
directory.

Next, place :code:`stellar_grids.zip` inside the :code:`inputs` directory and 
unzip it:

.. code-block:: bash

   unzip inputs/stellar_grids.zip -d inputs

You should now have an :code:`inputs` folder with two subdirectories: :code:`inputs/opacity` 
and :code:`inputs/stellar_grids`

Now all that is left to do it to create environment variables telling POSEIDON
where to find these input files.

Linux environment variables
---------------------------
  
If you are using Linux, enter the following lines into a terminal:

.. code-block:: bash

   echo 'export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/opacity/"' >>~/.bashrc
   echo 'export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/"' >>~/.bashrc

You should replace the bold text above with the location of your POSEIDON directory.

Alternatively, you can just open your .bashrc file (a hidden file in your home 
directory) with a text editor and add the following two lines at the bottom:

.. code-block:: bash

   export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/opacity/"
   export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/"

Mac OS environment variables
----------------------------

Setting environment variables on macOS differs depending on your OS version
(thanks to Apple's infinite wisdom). 

If you are using macOS >= 10.15 your default terminal will be zsh, for which
you can set the environment variables like so:

.. code-block:: bash

   echo export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/opacity/" >>~/.zshrc
   echo export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/" >>~/.zshrc

Alternatively, for earlier macOS versions, the default terminal is bash:
   
.. code-block:: bash

   echo 'export POSEIDON_input_data="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/opacity/"' >>~/.bash_profile
   echo 'export PYSYN_CDBS="/𝗣𝗔𝗧𝗛/𝗧𝗢/𝗬𝗢𝗨𝗥/𝗣𝗢𝗦𝗘𝗜𝗗𝗢𝗡/𝗗𝗜𝗥𝗘𝗖𝗧𝗢𝗥𝗬/inputs/stellar_grids/"' >>~/.bash_profile


Now POSEIDON will know where to find the opacity database and stellar models.

And that, splendidly, is all there is to it. Onwards to the tutorials!

The best place to begin is the quick start guide 
`"Generating Transmission Spectra" <notebooks/transmission_basic.html>`_
