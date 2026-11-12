Installation
============

Step 1: Download input files
____________________________

POSEIDON requires various input files (opacity data, stellar models, etc.) that
are stored separately from the GitHub repository (due to file size limitations).
Before installing POSEIDON, you will need to download these input files, which
have been packaged for convenience as a single .zip file (72 GB):

* `inputs.zip <https://drive.google.com/file/d/1ZSMoW8R8swYVTCiRAQZgIkyGmVDQQX2N/view?usp=sharing>`_

You can also find all the required input files on `Zenodo 
<https://zenodo.org/records/19595136>`_.

.. note::
   For POSEIDON v1.4, there is a surface albedo database that is now included in the inputs folder. 
   If you already have downloaded the v1.3 version of the inputs, and don't want to redownload them, you can 
   access the surface_reflectivities folder `here <https://github.com/MartianColonist/POSEIDON/tree/dev_bugfix/POSEIDON/reference_data/surface_reflectivities_txt_files/albedo_database_for_v1.4>`_ to add it manually.

   Download the surface_reflectivities.zip folder, unzip it, and add it to your inputs folder. As of 1.4 you should have four top-level folders in inputs: stellar_grids, opacity, chemistry_grids, and surface_reflectivities. 

Step 2: Create a conda environment
__________________________________

While waiting for the input files to download, you can install POSEIDON and
its dependencies.

.. note::
   For Windows users, we recommend installing `Windows Subsystem for Linux (WSL) 
   <https://docs.microsoft.com/en-us/windows/wsl/about>`_ before proceeding.
   WSL provides a Linux environment you can use in Windows.
   
   We recommend using WSL because PyMultiNest does not natively support Windows.
   However, it will work fine if you use WSL to install and run POSEIDON.

We recommend installing POSEIDON in a fresh conda environment (e.g. via
`Miniforge <https://github.com/conda-forge/miniforge>`_ or
`Anaconda <https://www.anaconda.com/>`_). POSEIDON currently supports Python
versions up to 3.12. On either Linux (e.g. Ubuntu) or macOS, create and activate a new environment via:

.. code-block:: bash

   conda create --name 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘 python=3.12
   conda activate 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘

Step 3: Install MultiNest
______________________________________

`MultiNest <https://academic.oup.com/mnras/article/398/4/1601/981502>`_ is the 
main sampling algorithm used for parameter space exploration in POSEIDON
retrievals, which POSEIDON calls via its Python wrapper `PyMultiNest 
<https://johannesbuchner.github.io/PyMultiNest/>`_. Retrievals run in parallel
on multiple cores using MPI (via mpi4py).

First, download `POSEIDON from GitHub <https://github.com/MartianColonist/POSEIDON>`_ 
or clone the repository:

.. code-block:: bash
		
   git clone https://github.com/MartianColonist/POSEIDON.git
   cd POSEIDON

With your conda environment activated, install MultiNest, mpi4py, and
PyMultiNest using POSEIDON's installation script.

.. code-block:: bash

   bash install_multinest.sh

This script builds the latest version of MultiNest (v3.12) against
the same MPI library used by mpi4py (Open MPI by default). If you are
using macOS, make sure to reactivate your environment after the script finishes.

.. tip::
   On a computing cluster, first load the cluster's MPI library (e.g.
   :code:`module load openmpi`) and then run
   :code:`bash install_multinest.sh --mpi system`, so retrievals use the
   cluster's MPI. To use MPICH from conda-forge instead of Open MPI, use
   :code:`--mpi mpich`.

.. note::
   If the script does not work on your machine, you can instead install an
   older version of MultiNest (v3.10) and PyMultiNest directly from
   conda-forge, then install POSEIDON as above:

   .. code-block:: bash

      conda install -c conda-forge mpi4py pymultinest

Step 4: Install POSEIDON
______________________________________

Now the time has come to install POSEIDON! 🔱

.. code-block:: bash

   pip install -e .

Step 5: Configure input files
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

Optional: Install PyMSG
_______________________

`PyMSG <https://msg.readthedocs.io/en/stable/>`_ is an optional stellar grid
interpolation package, which is *strongly* recommended for models and
retrievals including unocculted stellar contamination (spots / faculae).

MSG normally requires the MESA SDK. POSEIDON instead provides a script that
builds MSG using only conda-forge packages, then installs PyMSG into your
POSEIDON environment. With your POSEIDON conda environment activated, run
the following from the top-level :code:`POSEIDON` folder:

.. code-block:: bash

   bash install_pymsg.sh --download-grid newera

This installs MSG inside your conda environment, sets the :code:`MSG_DIR`
environment variable, and downloads the NewEra JWST stellar grid (3.7 GB),
which covers 0.6–28.5 μm. The SPHINX M dwarf grid (0.1 GB) and the PHOENIX
grid (Goettingen-HiRes, 34 GB) can be downloaded via
:code:`--download-grid sphinx` and :code:`--download-grid phoenix`, and the
flag can be given more than once. If you have already downloaded MSG grid
files (:code:`sg-*.h5`), you can reuse them via
:code:`--grid-dir /𝗣𝗔𝗧𝗛/𝗧𝗢/𝗚𝗥𝗜𝗗𝗦`.

Once the script finishes, re-activate your environment:

.. code-block:: bash

   conda deactivate
   conda activate 𝗬𝗢𝗨𝗥_𝗘𝗡𝗩_𝗡𝗔𝗠𝗘_𝗛𝗘𝗥𝗘

You can then use PyMSG by setting :code:`interp_backend = 'pymsg'` when calling
:code:`create_star()`, together with :code:`stellar_grid = 'newera'`,
:code:`'sphinx'`, or :code:`'phoenix'`.

.. note::
   Please do not run :code:`pip install pymsg`. The PyPI package with that
   name is unrelated to MSG.

And that, splendidly, is all there is to it. Onwards to the tutorials!

The best place to begin is the quick start guide 
`"Generating Transmission Spectra" <notebooks/transmission_basic.html>`_
