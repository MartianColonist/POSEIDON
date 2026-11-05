'''
Test of the POSEIDON container: MPI, MultiNest, POSEIDON, and PyMSG.

Run with several MPI ranks via SLURM (see poseidon.slurm), or as a single
process via:

    apptainer exec --bind /path/to/inputs:/inputs poseidon.sif \
        python test_container.py

'''

import os
import shutil
import socket
import tempfile
import numpy as np
from mpi4py import MPI
import pymultinest

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#***** Check MPI *****#

print('host = ' + socket.gethostname() + ', rank = ' + str(rank) +
      ', size = ' + str(size), flush = True)

comm.Barrier()

if (rank == 0):
    print('\nMPI library: ' + MPI.Get_library_version().splitlines()[0],
          flush = True)


#***** Check MultiNest *****#

# 2D Gaussian likelihood with uniform priors on [-5, 5] in each dimension.
# The analytic evidence is Z = 2π / 100 (Gaussian tails beyond ±5 are
# negligible), so ln(Z) = -2.7674.
def prior(cube, ndim, nparams):
    cube[0] = 10.0 * cube[0] - 5.0
    cube[1] = 10.0 * cube[1] - 5.0


def log_like(cube, ndim, nparams):
    return -0.5 * (cube[0]**2 + cube[1]**2)


# Write MultiNest output to a temporary folder shared by all ranks
if (rank == 0):
    output_dir = tempfile.mkdtemp(prefix = 'poseidon_container_test_',
                                  dir = os.getcwd())
else:
    output_dir = None

output_dir = comm.bcast(output_dir, root = 0)

pymultinest.run(LogLikelihood = log_like, Prior = prior, n_dims = 2,
                outputfiles_basename = output_dir + '/test-',
                n_live_points = 400, resume = False, verbose = False)

comm.Barrier()

if (rank == 0):

    analyser = pymultinest.Analyzer(n_params = 2, verbose = False,
                                    outputfiles_basename = output_dir + '/test-')
    stats = analyser.get_stats()

    ln_Z = stats['nested sampling global log-evidence']
    ln_Z_err = stats['nested sampling global log-evidence error']
    ln_Z_true = np.log(2.0 * np.pi / 100.0)

    print('\nMultiNest ln(Z) = ' + str(np.round(ln_Z, 3)) + ' ± ' +
          str(np.round(ln_Z_err, 3)) + ' (analytic: ' +
          str(np.round(ln_Z_true, 3)) + ')')

    shutil.rmtree(output_dir)

    if (abs(ln_Z - ln_Z_true) > max(5.0 * ln_Z_err, 0.2)):
        raise Exception("Error: MultiNest evidence does not match the " +
                        "analytic value.")

    print('MultiNest: OK (' + str(size) + ' MPI rank(s))')


#***** Check POSEIDON and its input files *****#

if (rank == 0):

    import POSEIDON
    import pymsg

    print('\nPOSEIDON imported from: ' + os.path.dirname(POSEIDON.__file__))

    # Check the input files bind mounted at POSEIDON_input_data
    input_dir = os.environ.get('POSEIDON_input_data')

    for folder in ['opacity', 'chemistry_grids', 'stellar_grids',
                   'surface_reflectivities']:
        found = os.path.isdir(os.path.join(input_dir, folder))
        print('  ' + os.path.join(input_dir, folder) + ': ' +
              ('found' if found else 'NOT FOUND'))

    # Check which MSG stellar grids are available
    grid_dir = os.path.join(os.environ['MSG_DIR'], 'data', 'grids')

    for grid, name in [('SPHINX', 'sphinx'), ('Goettingen-HiRes', 'phoenix')]:
        found = os.path.exists(os.path.join(grid_dir, 'sg-' + grid + '.h5'))
        print("  PyMSG stellar_grid = '" + name + "': " +
              ('available' if found else 'not found'))

    print('\nContainer test complete.')
