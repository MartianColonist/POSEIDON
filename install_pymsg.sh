#!/usr/bin/env bash
#
# Install PyMSG into the active POSEIDON conda environment.
#
# PyMSG is the Python interface to MSG (Townsend & Lopez 2023), which POSEIDON
# can use to interpolate stellar model grids (interp_backend = 'pymsg').
#
# MSG officially requires the MESA SDK. This script instead builds MSG using
# only conda-forge packages (GCC 15, HDF5, LAPACK), so no MESA SDK is needed.
#
# Usage (with your POSEIDON conda environment activated):
#
#   bash install_pymsg.sh [options]
#
# Options:
#   --msg-dir DIR         Where to install MSG
#                         (default: $CONDA_PREFIX/share/msg-<version>).
#   --grid-dir DIR        Folder holding (or to download) MSG grid files
#                         (sg-*.h5). Grids found here are linked into MSG.
#                         Use this to reuse grids you have already downloaded.
#   --download-grid NAME  Download a stellar grid used by POSEIDON:
#                           newera   (NewEra-JWST grid, 0.6-28.5 μm, 3.7 GB)
#                           sphinx   (SPHINX M dwarf grid, 0.1 GB)
#                           phoenix  (Goettingen-HiRes PHOENIX grid, 34 GB)
#                         May be given more than once.
#   --jobs N              Number of parallel compile jobs (default: all cores).
#   -h, --help            Show this message.
#
# Once finished, re-activate the environment so that MSG_DIR is set:
#
#   conda deactivate && conda activate <your_env>
#

set -euo pipefail

MSG_VERSION="2.2.1"
MSG_URL="https://github.com/rhdtownsend/msg/releases/download/v${MSG_VERSION}/msg-${MSG_VERSION}.tar.gz"
LAPACK95_URL="https://www.netlib.org/lapack95/lapack95.tgz"
GRID_URL="http://user.astro.wisc.edu/~townsend/resource/download/msg/grids"

#***** Parse command line options *****#

MSG_DIR_INSTALL=""
GRID_DIR=""
DOWNLOAD_GRIDS=()
N_JOBS=$(getconf _NPROCESSORS_ONLN 2> /dev/null || echo 4)

print_usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --msg-dir)       MSG_DIR_INSTALL=$2; shift 2 ;;
        --grid-dir)      GRID_DIR=$2; shift 2 ;;
        --download-grid) DOWNLOAD_GRIDS+=("$2"); shift 2 ;;
        --jobs)          N_JOBS=$2; shift 2 ;;
        -h|--help)       print_usage; exit 0 ;;
        *) echo "Error: unknown option '$1' (see --help)."; exit 1 ;;
    esac
done

info() { echo -e "\n\033[1m==> $*\033[0m"; }
fail() { echo -e "\n\033[1;31mError:\033[0m $*" >&2; exit 1; }

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

if [[ -z "$MSG_DIR_INSTALL" ]]; then
    MSG_DIR_INSTALL="$CONDA_PREFIX/share/msg-${MSG_VERSION}"
fi
mkdir -p "$MSG_DIR_INSTALL"
MSG_DIR_INSTALL=$(cd "$MSG_DIR_INSTALL" && pwd)

echo "Installing MSG ${MSG_VERSION} + PyMSG into conda environment:"
echo "  $CONDA_PREFIX (Python $PY_VERSION)"
echo "MSG directory: $MSG_DIR_INSTALL"

# Ensure no MESA SDK (or other compiler settings) leaks into the build
unset MESASDK_ROOT MESASDK_VERSION FFLAGS FCFLAGS CFLAGS CPPFLAGS LDFLAGS
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -i mesasdk | paste -sd ':' -)
export PATH="$CONDA_PREFIX/bin:$PATH"

#***** Install build dependencies from conda-forge *****#

info "Installing build dependencies from conda-forge"

# GCC 15 matches the MESA SDK compiler (GCC 16 hides some MSG C symbols).
# fypp 3.3 clashes with MSG's fypp_deps wrapper, so pin fypp 3.2.
CONDA_PKGS=("python=$PY_VERSION" "gfortran=15" "fypp=3.2" "cython" "make"
            "pkgconf" "hdf5" "liblapack" "libblas" "curl")
if [[ "$OS" == "Linux" ]]; then
    CONDA_PKGS+=("gcc=15")
fi

conda install -y --prefix "$CONDA_PREFIX" -c conda-forge "${CONDA_PKGS[@]}"

#***** Download MSG and LAPACK95 sources *****#

info "Downloading MSG ${MSG_VERSION}"

curl -fL "$MSG_URL" | tar xzf - -C "$MSG_DIR_INSTALL" --strip-components=1

DEPS_DIR="$MSG_DIR_INSTALL/deps"
rm -rf "$DEPS_DIR"
mkdir -p "$DEPS_DIR/src" "$DEPS_DIR/lib/pkgconfig" "$DEPS_DIR/include"

info "Downloading LAPACK95"

curl -fL "$LAPACK95_URL" | tar xzf - -C "$DEPS_DIR/src"

#***** Build LAPACK95 (Fortran 95 interface to LAPACK, used by MSG) *****#

info "Building LAPACK95"

# The LAPACK95 makefile has no module dependencies, so build serially
(cd "$DEPS_DIR/src/LAPACK95/SRC" &&
 make double FC="gfortran -ffree-form" FC1="gfortran -ffixed-form" \
             OPTS0="-O2 -fPIC" MODLIB="-I./../lapack95_modules" \
             > "$DEPS_DIR/lapack95_build.log" 2>&1) ||
    { tail -30 "$DEPS_DIR/lapack95_build.log";
      fail "LAPACK95 build failed (see $DEPS_DIR/lapack95_build.log)."; }

cp "$DEPS_DIR/src/LAPACK95/lapack95.a" "$DEPS_DIR/lib/liblapack95.a"
cp "$DEPS_DIR"/src/LAPACK95/lapack95_modules/*.mod "$DEPS_DIR/include/"

# conda-forge does not ship pkg-config files for LAPACK, so MSG cannot find it
cat > "$DEPS_DIR/lib/pkgconfig/lapack.pc" << EOF
Name: lapack
Description: LAPACK (conda-forge)
Version: 3
Libs: -Wl,-rpath,$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib -llapack -lblas
EOF

cat > "$DEPS_DIR/lib/pkgconfig/lapack95.pc" << EOF
Name: lapack95
Description: Fortran 95 interface to LAPACK
Version: 3
Requires: lapack
Cflags: -I$DEPS_DIR/include
Libs: -L$DEPS_DIR/lib -llapack95
EOF

#***** Build MSG *****#

info "Building MSG (this takes about a minute)"

export MSG_DIR="$MSG_DIR_INSTALL"
export PKG_CONFIG_PATH="$DEPS_DIR/lib/pkgconfig:$CONDA_PREFIX/lib/pkgconfig"

# conda's pkgconf otherwise drops -I flags for the environment's include
# folder, which gfortran needs to find the HDF5 Fortran modules
export PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1
export PKG_CONFIG_ALLOW_SYSTEM_LIBS=1

# MSG's fypp_deps wrapper reads MESASDK_ROOT, but fypp comes from conda here.
# VERSION_CHECK=passed skips the MESA SDK version check. PORTABLE=yes avoids
# -march=native, so the build also runs on other nodes of a cluster.
(cd "$MSG_DIR_INSTALL" &&
 make clean > /dev/null 2>&1 || true)
(cd "$MSG_DIR_INSTALL" &&
 MESASDK_ROOT="$CONDA_PREFIX" make -j "$N_JOBS" VERSION_CHECK=passed \
                              PORTABLE=yes TESTS=no TOOLS=no \
                              > "$DEPS_DIR/msg_build.log" 2>&1) ||
    { tail -30 "$DEPS_DIR/msg_build.log";
      fail "MSG build failed (see $DEPS_DIR/msg_build.log)."; }

#***** Install the PyMSG Python package *****#

info "Installing PyMSG"

# Note: 'pip install pymsg' from PyPI installs an unrelated package, so PyMSG
# must be installed from the MSG source folder
"$PYTHON" -m pip uninstall -y pymsg > /dev/null 2>&1 || true
"$PYTHON" -m pip install "$MSG_DIR_INSTALL/python"

# Set MSG_DIR whenever this conda environment is activated
conda env config vars set --prefix "$CONDA_PREFIX" MSG_DIR="$MSG_DIR_INSTALL" \
    > /dev/null

#***** Stellar grids *****#

GRIDS_DIR_MSG="$MSG_DIR_INSTALL/data/grids"
mkdir -p "$GRIDS_DIR_MSG"

if [[ -z "$GRID_DIR" ]]; then
    GRID_DIR="$GRIDS_DIR_MSG"
fi
mkdir -p "$GRID_DIR"
GRID_DIR=$(cd "$GRID_DIR" && pwd)

for grid in "${DOWNLOAD_GRIDS[@]+"${DOWNLOAD_GRIDS[@]}"}"; do
    case $(echo "$grid" | tr '[:upper:]' '[:lower:]') in
        newera)  grid_path="NewEra/JWST/sg-NewEra-JWST.h5" ;;
        sphinx)  grid_path="SPHINX/sg-SPHINX.h5" ;;
        phoenix) grid_path="Goettingen/HiRes/v3/sg-Goettingen-HiRes.h5" ;;
        *) fail "unknown grid '$grid' (options: newera / sphinx / phoenix)." ;;
    esac
    info "Downloading $(basename "$grid_path") into $GRID_DIR"
    curl -fL -C - -o "$GRID_DIR/$(basename "$grid_path")" "$GRID_URL/$grid_path"
done

# Link grids from a separate grid folder into MSG's grid folder
if [[ "$GRID_DIR" != "$GRIDS_DIR_MSG" ]]; then
    for grid_file in "$GRID_DIR"/sg-*.h5; do
        [[ -e "$grid_file" ]] || continue
        ln -sf "$grid_file" "$GRIDS_DIR_MSG/$(basename "$grid_file")"
    done
fi

#***** Verify the installation *****#

info "Testing PyMSG"

MSG_DIR="$MSG_DIR_INSTALL" "$PYTHON" - << 'EOF'
import os
import numpy as np
import pymsg

grid_dir = os.path.join(os.environ['MSG_DIR'], 'data', 'grids')

# Interpolate a spectrum from the demo grid shipped with MSG
specgrid = pymsg.SpecGrid(os.path.join(grid_dir, 'sg-demo.h5'))
flux = specgrid.flux({'Teff': 5000.0, 'log(g)': 4.0}, 0,
                     np.array([4000.0, 5000.0]))
assert (flux[0] > 0.0)
print('PyMSG works: demo grid interpolation successful.')

for grid, name in [('NewEra-JWST', 'newera'), ('Goettingen-HiRes', 'phoenix'),
                   ('SPHINX', 'sphinx')]:
    found = os.path.exists(os.path.join(grid_dir, 'sg-' + grid + '.h5'))
    print("  stellar_grid = '" + name + "': " +
          ('available' if found else 'grid not found (sg-' + grid + '.h5)'))
EOF

info "Done!"
echo "Re-activate your environment to set MSG_DIR:"
echo "  conda deactivate && conda activate ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
echo "Then use interp_backend = 'pymsg' in POSEIDON's create_star()."
