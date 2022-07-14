from setuptools import setup, find_packages
import os

setup(
    name = 'POSEIDON',
    version = '0.8.5',
    description = 'Exoplanet atmospheric retrieval package',
    long_description = open(os.path.join(
                            os.path.dirname(__file__), 'README.rst')).read(),
    long_description_content_type = 'text/x-rst',
    author = 'Ryan J. MacDonald',
    author_email = 'rmacdonald@astro.cornell.edu',
    license = 'BSD 3-Clause License',
    packages = ['POSEIDON'],
    include_package_data = True,
    python_requires = '<3.10',
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
                        'corner',
                        'spectres',
                        'jupyter'],
    zip_safe = False,
)
