# POSEIDON container

An [Apptainer](https://apptainer.org/) container for running POSEIDON
retrievals on a SLURM-managed Linux cluster. The container includes:

- **OpenMPI 5.0.7**, built from source to match the cluster's OpenMPI version
  (the Rocky Linux 9 repos only provide 4.1.x).
- **MultiNest 3.12** from [farhanferoz/MultiNest](https://github.com/farhanferoz/MultiNest)
  (`MultiNest_v3.12_CMake`), built against this OpenMPI. Both the serial and
  MPI libraries are installed in `/opt/multinest/lib`.
- A Miniforge conda environment (`poseidon`, Python 3.12) containing:
  - `mpi4py`, compiled from source against the container's OpenMPI
  - `pymultinest` and POSEIDON (`pip install -e`, source at `/opt/POSEIDON`)
  - PyMSG, installed with POSEIDON's `install_pymsg.sh` (MSG at `/opt/msg`),
    with the SPHINX M dwarf grid included

The POSEIDON input files (~72 GB) are **not** in the container. You bind them
in when you run it.

| Path in container | Contents |
| --- | --- |
| `/inputs` | Mount point for the POSEIDON `inputs` folder (`POSEIDON_input_data` points here) |
| `/msg_grids` | Optional mount point for extra MSG grids (e.g. PHOENIX) |
| `/opt/POSEIDON` | POSEIDON source (cloned from GitHub at build time) |
| `/opt/msg` | MSG installation (`MSG_DIR`) |

## Building

You need Apptainer 1.2 or later, because the definition file uses build
arguments. Building compiles OpenMPI, MultiNest, and MSG from source, which
takes around 20–30 minutes. Run it on a compute node (e.g. in an interactive
job), not on the login node:

```bash
cd container
apptainer build poseidon.sif poseidon.def
```

POSEIDON is cloned from GitHub during the build, so the build uses whatever
is on the chosen branch at that moment. To pick up later POSEIDON changes,
rebuild the image. You can set these build arguments with `--build-arg NAME=value`:

| Argument | Default | Purpose |
| --- | --- | --- |
| `POSEIDON_BRANCH` | `dev` | POSEIDON branch (or tag) to install |
| `POSEIDON_REPO` | MartianColonist/POSEIDON | Repository to clone (e.g. your fork) |
| `PYTHON_VERSION` | `3.12` | Python version (POSEIDON supports up to 3.12) |
| `OPENMPI_VERSION` | `5.0.7` | Set this to the OpenMPI version on your cluster |
| `MULTINEST_COMMIT` | `cc616c1…` | farhanferoz/MultiNest commit (pinned for reproducibility) |
| `MSG_GRIDS` | `sphinx` | MSG grids to include in the image (`"sphinx phoenix"` adds 34 GB) |

For example:

```bash
apptainer build --build-arg POSEIDON_BRANCH=main poseidon.sif poseidon.def
```

At the end of the build, the `%test` section checks that the MultiNest
libraries load, that mpi4py links to the container's OpenMPI, and that
POSEIDON, PyMultiNest, and PyMSG import. If the build host can't initialise
MPI, skip this check with `--notest` and run `apptainer test poseidon.sif` on
a compute node.

## Usage

Always bind your POSEIDON `inputs` folder to `/inputs`. A single process
(e.g. computing a forward model) runs like this:

```bash
apptainer exec --bind /path/to/POSEIDON/inputs:/inputs poseidon.sif \
    python my_forward_model.py
```

You can also open an interactive shell:

```bash
apptainer shell --bind /path/to/POSEIDON/inputs:/inputs poseidon.sif
```

POSEIDON writes its output (`POSEIDON_output/`) to the current working
directory, so run from a folder that is visible inside the container. Apptainer
binds `$HOME` and the current directory by default.

### Retrievals with SLURM

Edit the `SIF` and `INPUTS` paths at the top of `poseidon.slurm`. Then, from
the folder that holds your script, submit:

```bash
sbatch poseidon.slurm                   # runs check_container.py
sbatch poseidon.slurm my_retrieval.py   # runs your retrieval script
```

`check_container.py` runs a 2D Gaussian through MultiNest on all MPI ranks and
compares the Bayesian evidence to the analytic value. It then reports which
input folders and MSG grids the container can see. Copy it next to your job
script before submitting it.

### PHOENIX grid for PyMSG

By default, only the SPHINX grid (`stellar_grid = 'sphinx'`) is inside the
image. To use `stellar_grid = 'phoenix'` with `interp_backend = 'pymsg'`,
download `sg-Goettingen-HiRes.h5` (34 GB) once on the cluster:

```bash
curl -fL -o /path/to/msg_grids/sg-Goettingen-HiRes.h5 \
    http://user.astro.wisc.edu/~townsend/resource/download/msg/grids/Goettingen/HiRes/v3/sg-Goettingen-HiRes.h5
```

Then bind that folder to `/msg_grids` (in `poseidon.slurm`, set `MSG_GRIDS`).
The image already links `$MSG_DIR/data/grids/sg-Goettingen-HiRes.h5` to
`/msg_grids/`, so no other set-up is needed.

### Running the POSEIDON test suite

```bash
apptainer exec --bind /path/to/POSEIDON/inputs:/inputs poseidon.sif \
    pytest -rA -p no:cacheprovider /opt/POSEIDON/tests
```

## The PMIx `gds/shmem2` issue

`poseidon.slurm` uses SLURM's native PMIx integration (`srun --mpi=pmix`) to
launch the container's MPI processes, instead of `mpirun` inside the
container. The host's SLURM PMIx server handles process launch and rendezvous,
and the container's OpenMPI 5.0.7 handles communication. The two only need to
agree on the PMIx wire protocol.

By default, PMIx uses its `shmem2` GDS (global data store) component, which
shares job information between host and container through shared memory
segments. In this container/host combination, that component fails: MPI ranks
fail to start, hang, or crash during PMIx initialisation. The workaround
forces the older hash-table GDS component:

```bash
export PMIX_MCA_gds=hash
```

`poseidon.slurm` already sets this. If you write your own SLURM script with
`--mpi=pmix`, set it there too. When you rebuild against a newer OpenMPI, test
without it. A newer OpenMPI/PMIx/UCX combination may fix the problem.

## Notes

- `PYTHONNOUSERSITE=1` is set inside the container, so packages in your
  host's `~/.local` don't shadow the container's packages.
- The container is read-only. Numba therefore caches compiled functions in
  your home directory (`~/.cache`) rather than next to the POSEIDON source.
- Don't `pip install pymsg` inside or outside the container. The PyPI package
  with that name is unrelated to MSG.
