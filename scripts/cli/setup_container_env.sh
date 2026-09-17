#!/usr/bin/env bash
# =============================================================================
# ColliderML Container Environment Setup
# =============================================================================
# Sets up the environment for running ColliderML pipeline stages inside the
# OpenDataDetector software container:
#   ghcr.io/opendatadetector/sw:0.2.2_linux-ubuntu24.04_gcc-13.3.0
#
# Usage:
#   source scripts/cli/setup_container_env.sh
#
# This script configures PATH, LD_LIBRARY_PATH, PYTHONPATH, and other
# environment variables needed to use the full HEP software stack
# (ACTS, Pythia8, ROOT, DD4hep, Geant4, HepMC3, etc.) installed via Spack.
#
# For stages beyond Pythia generation (simulation, digitization, conversion),
# it also handles:
#   - DD4hep environment (ddsim binary, DD4hep library paths)
#   - Geant4 environment (physics dataset paths)
#   - ROOT_INCLUDE_PATH (for podio/edm4hep Python bindings)
#   - ODD detector geometry (cloned from GitHub if not present)
#   - Geant4 physics datasets (downloaded if not present)
#   - Python packages for postprocessing (pip install if missing)
# =============================================================================

SPACK_BASE="/spack/opt/spack/linux-x86_64"
CACHE_DIR="${COLLIDERML_CACHE:-/cache}"

if [ ! -d "$SPACK_BASE" ]; then
    echo "ERROR: Spack installation not found at $SPACK_BASE"
    echo "This script must be run inside the ODD software container."
    return 1 2>/dev/null || exit 1
fi

# --- 0a. Install bc (required by MadGraph for shower parameter calculation) ---
if [ "${SKIP_GENERATION_SETUP:-}" != "1" ]; then
if ! command -v bc &>/dev/null; then
    if [ -f "$CACHE_DIR/bc_"*.deb ]; then
        dpkg -i "$CACHE_DIR"/bc_*.deb 2>/dev/null \
            && echo "Installed bc from cached .deb." \
            || echo "WARNING: Failed to install bc from cache."
    else
        apt-get update -qq && apt-get install -y -qq bc &>/dev/null \
            && echo "Installed bc via apt-get." \
            || echo "WARNING: Failed to install bc. MadGraph shower will not work."
    fi
fi

# --- 0b. Install MG5aMC_PY8_interface (MadGraph needs this to run Pythia8 shower) ---
# Compiles the C++ driver that steers Pythia8 showering from MadGraph.
# Uses HepMC2 (dynamic linking) + Pythia 8.313. Pre-compiled binary cached.
_mg_dir=$(ls -d "$SPACK_BASE"/madgraph5amc-*/bin 2>/dev/null | head -1)
_mg_dir="${_mg_dir%/bin}"
if [ -n "$_mg_dir" ] && [ ! -f "$_mg_dir/HEPTools/MG5aMC_PY8_interface/MG5aMC_PY8_interface" ]; then
    _hepmc2_dir=$(ls -d "$SPACK_BASE"/hepmc-*/include 2>/dev/null | head -1)
    _hepmc2_dir="${_hepmc2_dir%/include}"
    _py8_dir=$(ls -d "$SPACK_BASE"/pythia8-*/include 2>/dev/null | head -1)
    _py8_dir="${_py8_dir%/include}"
    _zlib_dir=$(ls -d "$SPACK_BASE"/zlib-ng-*/include 2>/dev/null | head -1)
    _zlib_dir="${_zlib_dir%/include}"

    if [ -n "$_hepmc2_dir" ] && [ -n "$_py8_dir" ]; then
        mkdir -p "$_mg_dir/HEPTools/MG5aMC_PY8_interface"

        # Use cached source + binary if available
        if [ -f "$CACHE_DIR/MG5aMC_PY8_interface/MG5aMC_PY8_interface" ]; then
            cp -r "$CACHE_DIR/MG5aMC_PY8_interface/"* "$_mg_dir/HEPTools/MG5aMC_PY8_interface/"
            echo "MG5aMC_PY8_interface installed from cache."
        elif [ -f "$CACHE_DIR/MG5aMC_PY8_interface/MG5aMC_PY8_interface.cc" ]; then
            cp -r "$CACHE_DIR/MG5aMC_PY8_interface/"* "$_mg_dir/HEPTools/MG5aMC_PY8_interface/"
            cd "$_mg_dir/HEPTools/MG5aMC_PY8_interface"
            echo "Compiling MG5aMC_PY8_interface against Pythia8 + HepMC2..."
            g++ MG5aMC_PY8_interface.cc -o MG5aMC_PY8_interface \
                -I"$_py8_dir/include" -I"$_hepmc2_dir/include" ${_zlib_dir:+-I"$_zlib_dir/include"} \
                -O2 -std=c++20 -fPIC -pthread -DGZIP \
                -L"$_py8_dir/lib" -Wl,-rpath,"$_py8_dir/lib" -lpythia8 \
                -L"$_hepmc2_dir/lib" -Wl,-rpath,"$_hepmc2_dir/lib" -lHepMC \
                ${_zlib_dir:+-L"$_zlib_dir/lib" -Wl,-rpath,"$_zlib_dir/lib"} -lz -ldl 2>/dev/null \
                && echo "MG5aMC_PY8_interface compiled successfully." \
                || echo "WARNING: MG5aMC_PY8_interface compilation failed."
            # Create version marker files MG5 expects
            echo "3.5.9" > MG5AMC_VERSION_ON_INSTALL
            echo "8.313" > PYTHIA8_VERSION_ON_INSTALL
            # Cache the compiled binary + markers
            [ -f MG5aMC_PY8_interface ] && cp MG5aMC_PY8_interface MG5AMC_VERSION_ON_INSTALL PYTHIA8_VERSION_ON_INSTALL "$CACHE_DIR/MG5aMC_PY8_interface/"
            cd - >/dev/null
        else
            echo "WARNING: MG5aMC_PY8_interface source not found in cache. MadGraph shower disabled."
            echo "  Fix: git clone https://github.com/mg5amcnlo/MG5aMC_PY8_interface.git $CACHE_DIR/MG5aMC_PY8_interface"
        fi

        # Configure MG5 to use the interface
        if [ -f "$_mg_dir/HEPTools/MG5aMC_PY8_interface/MG5aMC_PY8_interface" ]; then
            sed -i "s|# mg5amc_py8_interface_path = ./HEPTools/MG5aMC_PY8_interface|mg5amc_py8_interface_path = $_mg_dir/HEPTools/MG5aMC_PY8_interface|" \
                "$_mg_dir/input/mg5_configuration.txt" 2>/dev/null
        fi
    fi
    unset _hepmc2_dir _py8_dir _zlib_dir
fi
unset _mg_dir
else
    echo "Skipping MadGraph/Pythia generation setup."
fi

# --- 1. Source .bashrc for PATH and CMAKE_PREFIX_PATH ---
# The container's .bashrc sets up PATH and CMAKE_PREFIX_PATH for all spack
# packages, but guards behind an interactive-shell check. We bypass it.
if [ -z "${PS1:-}" ]; then
    export PS1="colliderml"
fi
# shellcheck disable=SC1090
source /root/.bashrc 2>/dev/null || true

# --- 2. LD_LIBRARY_PATH: all spack lib/ directories ---
_ld_paths=""
for dir in "$SPACK_BASE"/*/lib; do
    [ -d "$dir" ] && _ld_paths="$dir:$_ld_paths"
done
export LD_LIBRARY_PATH="${_ld_paths}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- 3. PYTHONPATH: Python site-packages + ROOT + ACTS ---

# 3a. Collect all site-packages directories
_py_paths=""
for dir in "$SPACK_BASE"/*/lib/python*/site-packages; do
    [ -d "$dir" ] && _py_paths="$dir:$_py_paths"
done

# 3b. ROOT Python bindings (installed under lib/root/)
for dir in "$SPACK_BASE"/root-*/lib/root; do
    [ -d "$dir" ] && _py_paths="$dir:$_py_paths"
done

# 3c. ACTS Python bindings
# ACTS installs its Python package under <prefix>/python/ with __init__.py
# directly in that directory, but it must be importable as "acts". We create
# a symlink so Python can find it as a package named "acts".
_acts_python_dir=$(find "$SPACK_BASE"/acts-*/python -maxdepth 0 -type d 2>/dev/null | head -1)
if [ -n "$_acts_python_dir" ]; then
    if [ -f "$_acts_python_dir/acts/__init__.py" ]; then
        # Standard layout: <prefix>/python/acts/ is the package. The native
        # Arrow spack image (acts +python) installs this way, so just put
        # <prefix>/python on PYTHONPATH.
        _py_paths="$_acts_python_dir:$_py_paths"
    elif [ -f "$_acts_python_dir/__init__.py" ]; then
        # Legacy layout (sw:0.2.2): <prefix>/python/ IS the acts package, with
        # __init__.py directly inside. Symlink so it imports as "acts".
        ln -sf "$_acts_python_dir" /tmp/acts
        _py_paths="/tmp:$_py_paths"
    else
        echo "WARNING: ACTS python dir found but no recognizable package layout"
    fi
else
    echo "WARNING: ACTS Python bindings not found"
fi

export PYTHONPATH="${_py_paths}${PYTHONPATH:+:$PYTHONPATH}"

# --- 4. PYTHIA8DATA: Pythia8 XML data directory ---
_pythia8_data=$(find "$SPACK_BASE"/pythia8-*/share/Pythia8/xmldoc -maxdepth 0 -type d 2>/dev/null | head -1)
if [ -n "$_pythia8_data" ]; then
    export PYTHIA8DATA="$_pythia8_data"
fi

# --- 4b. LHAPDF: PDF sets for MadGraph+Pythia8 showering ---
_lhapdf_config=$(ls "$SPACK_BASE"/lhapdf-*/bin/lhapdf-config 2>/dev/null | head -1)
_lhapdf_datadir=$(ls -d "$SPACK_BASE"/lhapdf-*/share/LHAPDF 2>/dev/null | head -1)
_lhapdfsets_dir=$(ls -d "$SPACK_BASE"/lhapdfsets-*/share/lhapdfsets 2>/dev/null | head -1)
if [ -n "$_lhapdf_config" ]; then
    export PATH="$(dirname "$_lhapdf_config"):$PATH"
    # Symlink PDF set data into LHAPDF's expected data directory
    if [ -n "$_lhapdf_datadir" ] && [ -n "$_lhapdfsets_dir" ]; then
        for _setdir in "$_lhapdfsets_dir"/*/; do
            [ -d "$_setdir" ] || continue
            _setname=$(basename "$_setdir")
            [ -e "$_lhapdf_datadir/$_setname" ] || ln -sf "$_setdir" "$_lhapdf_datadir/$_setname" 2>/dev/null
        done
    fi
    # Configure MG5 to use LHAPDF
    _mg_config=$(ls "$SPACK_BASE"/madgraph5amc-*/input/mg5_configuration.txt 2>/dev/null | head -1)
    if [ -n "$_mg_config" ]; then
        sed -i "s|# lhapdf = /PATH/TO/lhapdf-config|lhapdf = $_lhapdf_config|" "$_mg_config" 2>/dev/null
    fi
fi
unset _lhapdf_config _lhapdf_datadir _lhapdfsets_dir _setdir _setname _mg_config

# --- 5. DD4hep environment (ddsim binary, DD4hep paths) ---
_dd4hep_dir=$(ls "$SPACK_BASE"/dd4hep-*/bin/thisdd4hep.sh 2>/dev/null | head -1)
if [ -n "$_dd4hep_dir" ]; then
    # shellcheck disable=SC1090
    source "$_dd4hep_dir" 2>/dev/null || true
fi

# --- 6. Geant4 environment (physics dataset paths) ---
_g4_setup=$(ls "$SPACK_BASE"/geant4-*/bin/geant4.sh 2>/dev/null | head -1)
if [ -n "$_g4_setup" ]; then
    # shellcheck disable=SC1090
    source "$_g4_setup" 2>/dev/null || true
fi

# --- 7. ROOT_INCLUDE_PATH for podio/edm4hep/dd4hep headers ---
# Required for podio Python bindings (loads Frame.h via cppyy) and edm4hep
_include_paths=""
for dir in "$SPACK_BASE"/podio-*/include "$SPACK_BASE"/edm4hep-*/include "$SPACK_BASE"/dd4hep-*/include; do
    [ -d "$dir" ] && _include_paths="${dir}${_include_paths:+:$_include_paths}"
done
if [ -n "$_include_paths" ]; then
    export ROOT_INCLUDE_PATH="${_include_paths}${ROOT_INCLUDE_PATH:+:$ROOT_INCLUDE_PATH}"
fi

# --- 8. OpenDataDetector (ODD) geometry + factory library ---
# The ODD detector geometry is required for simulation and digitization stages.
# Uses ODD v4.0.4 from CERN GitLab (matches the ACTS version in this container).
# The factory library (libOpenDataDetector.so) provides DD4hep geometry plugins
# (ODDCylinder, ODDPixelBarrel, etc.) needed by ddsim to construct the detector.
_odd_src="${CACHE_DIR}/odd-v4"
_odd_install="${CACHE_DIR}/odd-v4-install"

# Clone ODD source if not present
if [ ! -f "$_odd_src/xml/OpenDataDetector.xml" ]; then
    echo "ODD geometry not found. Cloning ODD v4.0.4 from CERN GitLab..."
    mkdir -p "$(dirname "$_odd_src")"
    if git clone --depth 1 --branch v4.0.4 \
        https://gitlab.cern.ch/acts/OpenDataDetector.git "$_odd_src" 2>/dev/null; then
        echo "ODD v4.0.4 cloned successfully to $_odd_src"
    else
        echo "WARNING: Failed to clone ODD. Simulation/digitization stages will fail."
        echo "  Manual fix: git clone --branch v4.0.4 https://gitlab.cern.ch/acts/OpenDataDetector.git $_odd_src"
    fi
fi

# Build ODD factory library if not present
if [ -f "$_odd_src/CMakeLists.txt" ] && [ ! -f "$_odd_install/lib/libOpenDataDetector.so" ]; then
    echo "Building ODD factory library..."
    _odd_build="/tmp/odd-build"
    rm -rf "$_odd_build" && mkdir -p "$_odd_build"
    if (cd "$_odd_build" && \
        cmake "$_odd_src" -DCMAKE_INSTALL_PREFIX="$_odd_install" 2>/dev/null && \
        make -j"${ODD_BUILD_JOBS:-2}" 2>/dev/null && \
        make install 2>/dev/null); then
        echo "ODD factory library built successfully."
    else
        echo "WARNING: Failed to build ODD factory library. Simulation will fail."
    fi
    rm -rf "$_odd_build"
fi

# Set ODD_PATH and add factory library to LD_LIBRARY_PATH
export ODD_PATH="${ODD_PATH:-$_odd_src}"
if [ -d "$_odd_install/lib" ]; then
    export LD_LIBRARY_PATH="$_odd_install/lib:$LD_LIBRARY_PATH"
fi

# --- 9. Geant4 physics datasets ---
# Required for detector simulation. ~2 GB download, cached in CACHE_DIR.
_g4_data_dir=$(ls -d "$SPACK_BASE"/geant4-*/share/Geant4/data 2>/dev/null | head -1)
if [ "${SKIP_G4_DOWNLOAD:-}" = "1" ]; then
    echo "Skipping Geant4 datasets (not needed for EDM4hep digitization)."
elif [ -n "$_g4_data_dir" ] && [ -z "$(ls -A "$_g4_data_dir" 2>/dev/null)" ]; then
    # Data directory exists but is empty — need to download
    if [ -d "$CACHE_DIR/g4data" ] && [ -n "$(ls -A "$CACHE_DIR/g4data" 2>/dev/null)" ]; then
        # Cached data exists — symlink each dataset
        echo "Linking cached Geant4 datasets from $CACHE_DIR/g4data..."
        for dataset in "$CACHE_DIR/g4data"/*/; do
            [ -d "$dataset" ] || continue
            _name=$(basename "$dataset")
            [ -e "$_g4_data_dir/$_name" ] || ln -sf "$dataset" "$_g4_data_dir/$_name" 2>/dev/null
        done
    else
        echo "Geant4 datasets not found. Downloading (~2 GB)..."
        mkdir -p "$CACHE_DIR/g4data"
        if timeout 600 download_geant4_datasets.sh 2>/dev/null; then
            # Move downloaded data to cache and symlink
            for dataset in "$_g4_data_dir"/*/; do
                _name=$(basename "$dataset")
                if [ -d "$dataset" ] && [ "$_name" != "." ]; then
                    mv "$dataset" "$CACHE_DIR/g4data/$_name" 2>/dev/null
                    ln -sf "$CACHE_DIR/g4data/$_name" "$_g4_data_dir/$_name" 2>/dev/null
                fi
            done
            echo "Geant4 datasets downloaded and cached."
        else
            echo "WARNING: Failed to download Geant4 datasets. Simulation will fail."
            echo "  Manual fix: run 'download_geant4_datasets.sh' inside the container"
        fi
    fi
fi

# --- 10. Python packages for postprocessing ---
# Install packages needed by convert_all.py and other postprocessing scripts.
_pip_target="$CACHE_DIR/pip"
if [ "${SKIP_POSTPROCESSING_DEPS:-}" = "1" ]; then
    echo "Skipping postprocessing Python packages."
elif [ -d "$CACHE_DIR" ]; then
    # Always add pip target to PYTHONPATH first (may already be populated from cache)
    if [ -d "$_pip_target" ]; then
        export PYTHONPATH="$_pip_target:$PYTHONPATH"
    fi
    # Install if not yet available
    if ! python3 -c "import pyarrow" 2>/dev/null; then
        echo "Installing Python packages for postprocessing..."
        mkdir -p "$_pip_target"
        timeout 180 python3 -m pip install --quiet --timeout 15 \
            --trusted-host pypi.org --trusted-host files.pythonhosted.org \
            --target="$_pip_target" \
            pyarrow uproot pandas awkward h5py tqdm pyhepmc psutil pyedm4hep \
            polars huggingface_hub 2>/dev/null \
            && echo "Python packages installed." \
            || echo "WARNING: pip install failed. Postprocessing stages may fail."
        # Update PYTHONPATH if target was just created
        if [ -d "$_pip_target" ]; then
            export PYTHONPATH="$_pip_target:$PYTHONPATH"
        fi
    fi
fi

# --- 11. Clean up temporary variables ---
unset _ld_paths _py_paths _acts_python_dir _pythia8_data _dd4hep_dir _g4_setup
unset _include_paths _g4_data_dir _pip_target _name

echo "ColliderML container environment configured successfully."
echo "  Python:  $(which python3) ($(python3 --version 2>&1 | cut -d' ' -f2))"
echo "  ACTS:    $(python3 -c 'import acts; print(acts.__version__)' 2>/dev/null || echo 'not available')"
echo "  ddsim:   $(which ddsim 2>/dev/null || echo 'not available')"
echo "  ODD:     ${ODD_PATH:-not set}"
