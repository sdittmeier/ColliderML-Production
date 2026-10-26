#!/usr/bin/env bash
# Host-side entry point. No sudo is used for writes to the home-directory inputs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CLUSTER_ROOT=${CLUSTER_ROOT:-$(dirname "$REPO")}
CLUSTER_SCRATCH=${CLUSTER_SCRATCH:-/tmp/colliderml-paired-$(id -u)}
IMAGE=${COLLIDERML_IMAGE:-colliderml-acts:7cba36b17}

docker image inspect "$IMAGE" >/dev/null

mkdir -p "$CLUSTER_ROOT/data/higgs_portal/v2" "$CLUSTER_SCRATCH/cache" "$CLUSTER_SCRATCH/output"
if [ ! -f "$CLUSTER_SCRATCH/cache/odd-v4/xml/OpenDataDetector.xml" ]; then
    if [ -e "$CLUSTER_SCRATCH/cache/odd-v4" ]; then
        printf '%s\n' "Incomplete ODD cache exists. Retain/rename it before retrying: $CLUSTER_SCRATCH/cache/odd-v4" >&2
        exit 1
    fi
    git clone --depth 1 --branch v4.0.4 https://gitlab.cern.ch/acts/OpenDataDetector.git \
        "$CLUSTER_SCRATCH/cache/odd-v4"
fi

python3 "$REPO/scripts/cli/prepare_higgs_portal_inputs.py" \
    --output "$CLUSTER_ROOT/data/higgs_portal/v2"

if [ "$#" -eq 0 ]; then
    set -- --events 1 --threads 1 --run 0 --signal-start 0 \
        --output /output/higgs-portal-pilot-run0-n1
fi

printf '%s\n' "Host outputs: $CLUSTER_SCRATCH/output" \
    "Geant4 datasets are required now; they will be downloaded/cached if missing."

docker run --rm -i --entrypoint /bin/bash \
    -v "$REPO":/workspace:ro \
    -v "$CLUSTER_ROOT/data":/data:ro \
    -v "$CLUSTER_SCRATCH/cache":/cache \
    -v "$CLUSTER_SCRATCH/output":/output \
    -e COLLIDERML_CACHE=/cache \
    -e SKIP_GENERATION_SETUP=1 \
    -e SKIP_G4_DOWNLOAD=0 \
    "$IMAGE" --noprofile --norc -c '
        set -eo pipefail
        source /opt/acts-arrow/setup.sh
        source /workspace/scripts/cli/setup_container_env.sh
        source /opt/acts-arrow/setup.sh
        for g4_setup in /spack/opt/spack/linux-x86_64/geant4-*/bin/geant4.sh; do
            if [ -f "$g4_setup" ]; then
                source "$g4_setup"
                break
            fi
        done
        cd /workspace
        python3 -c "import acts.examples.pythia8, acts.examples.hepmc3, DDSim, pyhepmc, pyedm4hep, pyarrow, uproot, yaml; print(\"Pilot imports OK\")"
        python3 -m unittest tests.test_higgs_portal_pilot tests.test_ddsim_llp_settings \
            tests.test_recover_duplicate_events tests.test_run_paired_samples \
            tests.test_export_paired_clusters tests.test_production_pixel_quantization
        exec python3 scripts/simulation/higgs_portal_pilot.py "$@"
    ' bash "$@"
