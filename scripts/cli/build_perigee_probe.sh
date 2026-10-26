#!/usr/bin/env bash
# Run inside the pinned image after sourcing /opt/acts-arrow/setup.sh.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT=${1:-/cache/perigee-probe}
mkdir -p "$OUTPUT"
PYTHON_INCLUDE=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
EXTENSION=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
includes=()
for path in /spack/opt/spack/linux-x86_64/py-pybind11-*/include \
            /spack/opt/spack/linux-x86_64/eigen-*/include/eigen3 \
            /spack/opt/spack/linux-x86_64/boost-*/include; do
    includes+=("-I$path")
done
"${CXX:-c++}" -O1 -shared -fPIC -std=c++20 -I"$PYTHON_INCLUDE" \
    -I"$REPO/docker/acts-perigee/include" -I/opt/acts-arrow/include "${includes[@]}" \
    "$REPO/notebooks/validation/perigee_probe.cpp" \
    -L/opt/acts-arrow/lib -lActsCore -Wl,-rpath,/opt/acts-arrow/lib \
    -o "$OUTPUT/perigee_probe$EXTENSION"
echo "Built $OUTPUT/perigee_probe$EXTENSION"
