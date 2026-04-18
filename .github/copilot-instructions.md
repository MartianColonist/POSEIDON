# POSEIDON AI coding instructions

## Big-picture architecture
- Core orchestration lives in [POSEIDON/core.py](POSEIDON/core.py): most public entry points (e.g., `create_star()`, `create_planet()`, `define_model()`, `make_atmosphere()`, `read_opacities()`, `compute_spectrum()`) wire together atmospheric profiles, opacities, geometry, and radiative transfer.
- Atmospheric state/profiles are defined in [POSEIDON/atmosphere.py](POSEIDON/atmosphere.py) (P–T profiles, mixing ratio profiles, profile parametrizations).
- Opacity sampling / cross-section interpolation and extinction calculations are in [POSEIDON/absorption.py](POSEIDON/absorption.py); this is central to memory/runtime tradeoffs.
- Emission RT is in [POSEIDON/emission.py](POSEIDON/emission.py); transmission RT is in TRIDENT (see [POSEIDON/transmission.py](POSEIDON/transmission.py)).
- Retrieval workflows are orchestrated in [POSEIDON/retrieval.py](POSEIDON/retrieval.py) and exercised in tests (see [tests/test_retrieval.py](tests/test_retrieval.py)).

## Data + environment dependencies (critical)
- Large input datasets live under inputs/ (opacity, chemistry grids, stellar grids). Code expects these at runtime; see [docs/content/installation.rst](docs/content/installation.rst).
- Environment variables are required:
  - `POSEIDON_input_data` → inputs/ root
  - `PYSYN_CDBS` → inputs/stellar_grids/
  These are set in CI (see [pytest_testing.yml](.github/workflows/pytest_testing.yml)).
- Optional GPU paths use `cupy`; CPU remains default (see [POSEIDON/absorption.py](POSEIDON/absorption.py), [POSEIDON/emission.py](POSEIDON/emission.py)).

## Developer workflows
- Install with editable pip after creating a conda env (see [setup.py](setup.py) and [docs/content/installation.rst](docs/content/installation.rst)).
- Tests run via `pytest -rA`; CI installs `mpi4py` + `pymultinest` and downloads minimal CIA opacity data (see [pytest_testing.yml](.github/workflows/pytest_testing.yml)).

## Project-specific conventions & patterns
- The public user flow is “define star → define planet → define model → make atmosphere → read opacities → compute spectrum”, used in tutorials and tests (see [docs/content/notebooks/transmission_basic.ipynb](docs/content/notebooks/transmission_basic.ipynb) and [tests/test_TRIDENT.py](tests/test_TRIDENT.py)).
- Many heavy numerical kernels are `numba`-jitted; avoid changing signatures without updating the dependent call sites (core orchestrates these through `atmosphere`, `absorption`, `emission`).
- Opacity sampling vs. line-by-line is a first-class switch (`opacity_treatment`), with different memory/runtime implications; preserve this split (see [POSEIDON/absorption.py](POSEIDON/absorption.py) and [docs/content/notebooks/transmission_basic.ipynb](docs/content/notebooks/transmission_basic.ipynb)).
- Tests may create/clean local output and data dirs (e.g., `./data`, `./POSEIDON_output` in [tests/test_retrieval.py](tests/test_retrieval.py)); avoid hard-coding paths outside the repo.

## Integration points
- MultiNest / PyMultiNest and MPI (`mpi4py`) are core to retrievals; changes to retrieval code should be tested with those installed (see [tests/test_retrieval.py](tests/test_retrieval.py)).
- Opacity and chemistry HDF5 files are read directly by core functions; ensure any new readers respect the same grid conventions and metadata used in [POSEIDON/absorption.py](POSEIDON/absorption.py).

## Style
- Do not use bold text and avoid em-dashes in markdown text.