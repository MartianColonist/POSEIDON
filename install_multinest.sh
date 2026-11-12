#!/usr/bin/env bash
#
# Install MultiNest 3.12, mpi4py, and PyMultiNest into the active POSEIDON
# conda environment.
#
# MultiNest (Feroz et al. 2009) is the nested sampling algorithm used for
# POSEIDON retrievals, and PyMultiNest is its Python interface. This script
# builds MultiNest 3.12 from source (https://github.com/farhanferoz/MultiNest)
# against a single MPI library, which mpi4py also uses. Only conda-forge
# packages are needed (no root access or system compilers).
#
# Usage (with your POSEIDON conda environment activated):
#
#   bash install_multinest.sh [options]
#
# Options:
#   --mpi NAME       MPI library to build against:
#                      openmpi  (Open MPI from conda-forge, default)
#                      mpich    (MPICH from conda-forge)
#                      system   (the MPI already on your PATH, e.g. a
#                                cluster's MPI loaded via 'module load')
#   --commit SHA     MultiNest commit to build
#                    (default: the same commit used by POSEIDON's CI).
#   -h, --help       Show this message.
#
# Then install POSEIDON itself from the top-level POSEIDON folder:
#
#   pip install -e .
#

set -euo pipefail

MULTINEST_URL="https://github.com/farhanferoz/MultiNest.git"
MULTINEST_COMMIT="cc616c17ed440fbff8d35937997c86256ed4d744"

#***** Parse command line options *****#

MPI_CHOICE="openmpi"

print_usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mpi)     MPI_CHOICE=$2; shift 2 ;;
        --commit)  MULTINEST_COMMIT=$2; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Error: unknown option '$1' (see --help)."; exit 1 ;;
    esac
done

info() { echo -e "\n\033[1m==> $*\033[0m"; }
fail() { echo -e "\n\033[1;31mError:\033[0m $*" >&2; exit 1; }

case $MPI_CHOICE in
    openmpi|mpich|system) ;;
    *) fail "unknown MPI '$MPI_CHOICE' (options: openmpi / mpich / system)." ;;
esac

#***** Check the conda environment *****#

if [[ -z "${CONDA_PREFIX:-}" ]] || [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    fail "please activate your POSEIDON conda environment (not 'base') " \
         "before running this script."
fi

if ! command -v conda > /dev/null; then
    fail "conda command not found."
fi

PYTHON="$CONDA_PREFIX/bin/python"
PY_VERSION=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')

OS=$(uname -s)
if [[ "$OS" != "Linux" ]] && [[ "$OS" != "Darwin" ]]; then
    fail "unsupported operating system '$OS' (Linux or macOS only)."
fi

echo "Installing MultiNest 3.12 + PyMultiNest into conda environment:"
echo "  $CONDA_PREFIX (Python $PY_VERSION)"
echo "MPI library: $MPI_CHOICE"

# Ensure no other compiler settings leak into the build
unset FFLAGS FCFLAGS CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

if [[ "$MPI_CHOICE" == "system" ]]; then

    # The system MPI must come first on the PATH, ahead of the conda env
    export PATH="$PATH:$CONDA_PREFIX/bin"

    for wrapper in mpicc mpifort; do
        wrapper_path=$(command -v $wrapper || true)
        if [[ -z "$wrapper_path" ]]; then
            fail "'$wrapper' not found. For --mpi system, first load your " \
                 "MPI library (e.g. 'module load openmpi')."
        fi
        if [[ "$wrapper_path" == "$CONDA_PREFIX"/* ]]; then
            fail "'$wrapper' comes from this conda environment " \
                 "($wrapper_path), not the system. Remove the conda MPI " \
                 "('conda remove openmpi mpich') or use --mpi openmpi."
        fi
    done
else
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

#***** Remove conda-forge MultiNest (version 3.10) if present *****#

# Packages installed via pip also show up in 'conda list' (channel pypi)
CONDA_MULTINEST=$(conda list --prefix "$CONDA_PREFIX" \
                  '^(pymultinest|multinest)$' 2> /dev/null |
                  grep -v '^#' | grep -v pypi | awk '{print $1}' || true)

if [[ -n "$CONDA_MULTINEST" ]]; then
    info "Removing conda-forge MultiNest packages: $(echo $CONDA_MULTINEST)"
    conda remove -y --prefix "$CONDA_PREFIX" --force $CONDA_MULTINEST
fi

#***** Remove a different conda MPI if present *****#

# Asking conda to swap one MPI library for another (e.g. MPICH for Open MPI)
# can take a very long time to solve, so remove the old MPI stack first
CONDA_MPI=$(conda list --prefix "$CONDA_PREFIX" '^(openmpi|mpich)$' \
            2> /dev/null | grep -v '^#' | grep -v pypi | awk '{print $1}' ||
            true)

for old_mpi in $CONDA_MPI; do
    if [[ "$old_mpi" != "$MPI_CHOICE" ]] && [[ "$MPI_CHOICE" != "system" ]]; then
        info "Removing conda-forge $old_mpi (replacing it with $MPI_CHOICE)"
        conda remove -y --prefix "$CONDA_PREFIX" --force \
            "$old_mpi" mpi mpi4py > /dev/null 2>&1 || true
    fi
done

# Use conda's fast libmamba solver when available (the default since conda
# 23.10, and optional in earlier versions)
SOLVER_ARGS=()
if conda install --help 2> /dev/null | grep -q -- '--solver' &&
   "${CONDA_PYTHON_EXE:-python}" -c 'import conda_libmamba_solver' \
       > /dev/null 2>&1; then
    SOLVER_ARGS=("--solver=libmamba")
fi

# Conda versions before 23.10 default to the 'classic' solver, which is slow
# and can fail on environments containing compilers and MPI
CONDA_VERSION=$(conda --version 2> /dev/null | awk '{print $2}')
CONDA_MAJOR=${CONDA_VERSION%%.*}
CONDA_MINOR=$(echo "$CONDA_VERSION" | cut -d. -f2)
if [[ ${#SOLVER_ARGS[@]} -eq 0 ]] && [[ "$CONDA_MAJOR" =~ ^[0-9]+$ ]] &&
   { [[ "$CONDA_MAJOR" -lt 23 ]] ||
     { [[ "$CONDA_MAJOR" -eq 23 ]] && [[ "${CONDA_MINOR:-0}" -lt 10 ]]; }; }; then
    echo -e "\n\033[1;33mWarning:\033[0m conda $CONDA_VERSION uses the slow " \
            "'classic' solver, so this may take a while. If installing " \
            "packages later fails with a solver error, update conda first: " \
            "conda update -n base conda"
fi

#***** Install build dependencies from conda-forge *****#

info "Installing build dependencies from conda-forge"

# Same compiler versions as install_pymsg.sh, so both scripts can share an
# environment
CONDA_PKGS=("python=$PY_VERSION" "gfortran=15" "cmake" "make" "git"
            "liblapack" "libblas")
if [[ "$OS" == "Linux" ]]; then

    # conda-forge's MPI libraries need glibc >= 2.28 (the conda-forge
    # baseline) when linking, so ensure the compilers use a recent sysroot
    CONDA_PKGS+=("gcc=15" "gxx=15" "sysroot_linux-64>=2.28")
else
    CONDA_PKGS+=("clang" "clangxx")
fi

# A conda-forge MPI, plus mpi4py built against it
if [[ "$MPI_CHOICE" != "system" ]]; then
    CONDA_PKGS+=("$MPI_CHOICE" "mpi4py" "mpi=*=$MPI_CHOICE")
fi

# Only use conda-forge: compilers from Anaconda's 'defaults' channel ship an
# older C library than conda-forge's MPI libraries need, so linking fails
conda install -y --prefix "$CONDA_PREFIX" --override-channels -c conda-forge \
    "${SOLVER_ARGS[@]+"${SOLVER_ARGS[@]}"}" "${CONDA_PKGS[@]}"

# conda-forge's Open MPI tries UCX for MPI shared memory windows, which prints
# an error for every window on machines without UCX hardware (e.g. laptops).
# Open MPI's own shared memory component works everywhere, so use that.
if [[ "$MPI_CHOICE" == "openmpi" ]]; then
    conda env config vars set --prefix "$CONDA_PREFIX" OMPI_MCA_osc="^ucx" \
        > /dev/null
    export OMPI_MCA_osc="^ucx"
fi

# Point the conda MPI compiler wrappers at the conda compilers
if [[ "$MPI_CHOICE" != "system" ]]; then
    if [[ "$OS" == "Linux" ]]; then
        CC_NAME=gcc; CXX_NAME=g++
    else
        CC_NAME=clang; CXX_NAME=clang++
    fi
    export OMPI_CC=$CC_NAME OMPI_CXX=$CXX_NAME OMPI_FC=gfortran
    export MPICH_CC=$CC_NAME MPICH_CXX=$CXX_NAME MPICH_FC=gfortran
fi

#***** Build MultiNest 3.12 *****#

info "Downloading MultiNest (commit ${MULTINEST_COMMIT:0:7})"

BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

LOG_FILE="$CONDA_PREFIX/share/multinest_build.log"
mkdir -p "$CONDA_PREFIX/share"

git clone -q "$MULTINEST_URL" "$BUILD_DIR/MultiNest"
(cd "$BUILD_DIR/MultiNest" && git checkout -q "$MULTINEST_COMMIT")

info "Building MultiNest (this takes about a minute)"

# The CMake build of MultiNest 3.12 lives in MultiNest_v3.12_CMake/multinest
# (its CMakeLists.txt still labels the library version as 3.11, but the
# Fortran sources are those of v3.12)
MULTINEST_SRC="$BUILD_DIR/MultiNest/MultiNest_v3.12_CMake/multinest"
mkdir -p "$MULTINEST_SRC/build"

# CMAKE_POLICY_VERSION_MINIMUM lets newer CMake versions accept MultiNest's
# old cmake_minimum_required(VERSION 2.8). MultiNest's library variants share
# Fortran .mod files, so parallel make has a race condition: build
# single-threaded, and only the serial and MPI shared libraries (the bundled
# examples are not needed, and some do not compile with recent compilers).
(cd "$MULTINEST_SRC/build" &&
 FC=mpifort CC=mpicc CXX=mpicxx \
 cmake -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
       -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
       .. > "$LOG_FILE" 2>&1 &&
 make -j1 multinest_shared multinest_mpi_shared >> "$LOG_FILE" 2>&1) ||
    { tail -30 "$LOG_FILE";
      fail "MultiNest build failed (see $LOG_FILE)."; }

#***** Install the MultiNest libraries *****#

info "Installing MultiNest libraries into $CONDA_PREFIX/lib"

if [[ "$OS" == "Darwin" ]]; then
    LIB_EXT="dylib"
    cp -P "$MULTINEST_SRC"/lib/libmultinest*.dylib "$CONDA_PREFIX/lib/"
else
    LIB_EXT="so"
    cp -P "$MULTINEST_SRC"/lib/libmultinest*.so* "$CONDA_PREFIX/lib/"
fi

for lib in libmultinest libmultinest_mpi; do
    if [[ ! -e "$CONDA_PREFIX/lib/$lib.$LIB_EXT" ]]; then
        fail "$lib.$LIB_EXT was not built (see $LOG_FILE)."
    fi
done

# On Linux, conda's Python finds libraries in $CONDA_PREFIX/lib by itself.
# On macOS, PyMultiNest searches the default library folders, so add the
# conda environment's lib folder to them whenever the environment is active.
if [[ "$OS" == "Darwin" ]]; then
    DYLD_PATH="$CONDA_PREFIX/lib:$HOME/lib:/usr/local/lib:/usr/lib"
    conda env config vars set --prefix "$CONDA_PREFIX" \
        DYLD_FALLBACK_LIBRARY_PATH="$DYLD_PATH" > /dev/null
    export DYLD_FALLBACK_LIBRARY_PATH="$DYLD_PATH"
fi

#***** Install mpi4py (system MPI only) and PyMultiNest *****#

if [[ "$MPI_CHOICE" == "system" ]]; then

    info "Building mpi4py against $(command -v mpicc)"

    # A binary mpi4py would bring its own MPI, which cannot use the cluster's
    # interconnect or scheduler
    conda remove -y --prefix "$CONDA_PREFIX" --force mpi4py \
        > /dev/null 2>&1 || true
    MPICC=$(command -v mpicc) \
    "$PYTHON" -m pip install --no-cache-dir --no-binary=mpi4py \
                             --force-reinstall mpi4py
fi

info "Installing PyMultiNest"

# PyMultiNest is pure Python, so the pip version uses the MultiNest built
# here (PyMultiNest imports NumPy, SciPy, and Matplotlib, but does not install
# them itself)
"$PYTHON" -m pip install pymultinest numpy scipy matplotlib

#***** Verify the installation *****#

info "Testing MPI and MultiNest"

"$PYTHON" - << 'EOF'
import sys
import ctypes
import ctypes.util
from mpi4py import MPI

print('MPI library: ' + MPI.Get_library_version().splitlines()[0])

# Load both MultiNest libraries the same way PyMultiNest does
for lib in ['multinest', 'multinest_mpi']:
    if (sys.platform == 'darwin'):
        lib_path = ctypes.util.find_library(lib)
        if (lib_path is None):
            raise Exception("Error: lib" + lib + ".dylib not found.")
    else:
        lib_path = 'lib' + lib + '.so'
    ctypes.CDLL(lib_path)
    print('Loaded ' + lib_path)

import pymultinest
print('PyMultiNest works: MultiNest 3.12 libraries loaded.')
EOF

info "Done!"
if [[ "$OS" == "Darwin" ]]; then
    echo "Re-activate your environment (so PyMultiNest can find MultiNest):"
    echo "  conda deactivate && conda activate ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
fi
echo "Next, install POSEIDON from the top-level POSEIDON folder:"
echo "  pip install -e ."
if [[ "$MPI_CHOICE" == "openmpi" ]]; then
    echo "Note: Open MPI's 'mpirun -n N' needs N <= your number of CPU cores"
    echo "(or add --oversubscribe)."
fi
