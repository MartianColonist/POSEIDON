#!/bin/bash
#SBATCH --job-name=poseidon
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G 
#SBATCH --time=00:15:00
#SBATCH --output=poseidon-%j.out
#SBATCH --error=poseidon-%j.err
#
# Run a POSEIDON script inside the container on multiple MPI ranks.
#
# Usage:
#
#   sbatch poseidon.slurm [script.py]     (default: check_container.py)
#
# Edit the paths below before submitting.

set -euo pipefail

#***** User settings *****#

# Container image
SIF=/path/to/poseidon.sif

# POSEIDON input files (folder containing opacity/, stellar_grids/, etc.).
# PyMSG grids not built into the image are read from stellar_grids/pymsg/.
INPUTS=/path/to/POSEIDON/inputs

# Optional: your own POSEIDON checkout (the top-level folder containing
# setup.py), used instead of the POSEIDON built into the container.
# Leave empty to use the container's POSEIDON.
POSEIDON_SRC=""

SCRIPT=${1:-check_container.py}

#***** Run *****#

module load apptainer

unset PYTHONHOME
unset PYTHONPATH

# Work around PMIx gds_shmem2 issue (see README.md)
export PMIX_MCA_gds=hash

BINDS="--bind ${INPUTS}:/inputs:ro --bind ${PWD}"

# Mount your POSEIDON checkout over the container's (installed in editable
# mode from /opt/POSEIDON, so Python then imports your code)
if [[ -n "$POSEIDON_SRC" ]]; then
    if [[ ! -f "${POSEIDON_SRC}/setup.py" ]] || [[ ! -d "${POSEIDON_SRC}/POSEIDON" ]]; then
        echo "Error: POSEIDON_SRC (${POSEIDON_SRC}) must be the top-level" \
             "POSEIDON folder containing setup.py and POSEIDON/." >&2
        exit 1
    fi
    BINDS="$BINDS --bind ${POSEIDON_SRC}:/opt/POSEIDON"
    echo "POSEIDON source: ${POSEIDON_SRC}" \
         "($(git -C "${POSEIDON_SRC}" log -1 --format='%h %s' 2> /dev/null || echo 'not a git repository'))"
else
    echo "POSEIDON source: built into the container"
fi

echo "HOSTNAME=$(hostname)  JOBID=${SLURM_JOB_ID}  NTASKS=${SLURM_NTASKS}"

# Process launch is handled by SLURM's PMIx (rather than mpirun inside the
# container); the container's OpenMPI does the communication
srun --mpi=pmix \
    apptainer exec ${BINDS} --pwd "${PWD}" "${SIF}" \
    python "${SCRIPT}"
