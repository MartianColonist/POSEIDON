from setuptools import setup, find_packages
import os

setup(
    name = 'POSEIDON',
    version = '0.6.0',
    description = 'Exoplanet atmospheric retrieval package',
    long_description = open(os.path.join(
                            os.path.dirname(__file__), 'README.rst')).read(),
    long_description_content_type='text/x-rst',
    author = 'Ryan J. MacDonald',
    author_email = 'rmacdonald@astro.cornell.edu',
    license = 'BSD 3-Clause License',
    packages = find_packages(),
    include_package_data = True,
    package_data={'': ['reference_data/*']},
    python_requires='<=3.8.12',
    install_requires = ['numpy<=1.21',
                        'scipy',
                        'matplotlib>=3.3',
                        'astropy',
                        'h5py',
                        'numba',
                        'pandas',
                        'pysynphot',
                        'mpi4py',
                        'pymultinest',
                        'corner',
                        'spectres'],
    zip_safe = False,
)
