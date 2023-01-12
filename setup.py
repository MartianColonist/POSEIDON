from setuptools import setup, find_packages
import os

setup(
    name = 'POSEIDON',
    version = '1.0.0',
    description = 'Exoplanet atmospheric retrieval package',
    long_description = open(os.path.join(
                            os.path.dirname(__file__), 'README.rst')).read(),
    long_description_content_type = 'text/x-rst',
    author = 'Ryan J. MacDonald',
    author_email = 'ryanjmac@umich.edu',
    license = 'BSD 3-Clause License',
    packages = ['POSEIDON'],
    include_package_data = True,
    python_requires = '<3.11',
    install_requires = ['numpy',
                        'scipy',
                        'matplotlib<=3.5.1',
                        'astropy',
                        'h5py',
                        'numba',
                        'pandas',
                        'mpi4py',
                        'pysynphot',
                        'pymultinest',
                        'spectres',
                        'jupyter',
                        'pytest'],
    zip_safe = False,
)
