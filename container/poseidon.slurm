#!/bin/bash
#SBATCH --job-name=poseidon
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=poseidon-%j.out
#SBATCH --error=poseidon-%j.err
#
# Run a POSEIDON script inside the container on multiple MPI ranks.
#
# Usage:
#
#   sbatch poseidon.slurm [script.py]     (default: test_container.py)
#
# Edit the paths below before submitting.

set -euo pipefail

#***** User settings *****#

# Container image
SIF=/path/to/poseidon.sif

# POSEIDON input files (folder containing opacity/, stellar_grids/, etc.)
INPUTS=/path/to/POSEIDON/inputs

# Optional: folder holding extra MSG grids (e.g. sg-Goettingen-HiRes.h5 for
# stellar_grid = 'phoenix'). Leave empty if not needed.
MSG_GRIDS=""

SCRIPT=${1:-test_container.py}

#***** Run *****#

module load apptainer

unset PYTHONHOME
unset PYTHONPATH

# Work around PMIx gds_shmem2 issue (see README.md)
export PMIX_MCA_gds=hash

BINDS="--bind ${INPUTS}:/inputs:ro --bind ${PWD}"
if [[ -n "$MSG_GRIDS" ]]; then
    BINDS="$BINDS --bind ${MSG_GRIDS}:/msg_grids:ro"
fi

echo "HOSTNAME=$(hostname)  JOBID=${SLURM_JOB_ID}  NTASKS=${SLURM_NTASKS}"

# Process launch is handled by SLURM's PMIx (rather than mpirun inside the
# container); the container's OpenMPI does the communication
srun --mpi=pmix \
    apptainer exec ${BINDS} --pwd "${PWD}" "${SIF}" \
    python "${SCRIPT}"
