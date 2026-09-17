# Paired ttbar cluster samples

One command builds hits, constituent cells, and particles from the first N
events of each hard-scatter and full-pileup EDM4hep file. N defaults to 1.
ACTS digitization and the existing production particle/tracker-hit converter
run on the same input events. The exporter checks their alignment before
adding particle truth to each paired hit. It retains all `simhit_ids` and
resolves them to the distinct `particle_ids` contributing
to each measurement. The scalar `particle_id` is filled only when that list
contains exactly one particle; a merged measurement with multiple particles
has a null scalar label. An unresolved SimHit fails export rather than
becoming particle ID zero.

There is no cell-level particle label or track reconstruction.
This run enables `simhits.root`, which the previous `paired-samples-n1` run did
not write; use a new output directory, but reuse the downloaded EDM4hep files.

## One-time setup

Start with the two downloaded EDM4hep files at `data/pu0/edm4hep.root` and
`data/pu200/edm4hep.root`. The image uses ACTS commit `7cba36b17` and ODD
v4.0.4 (`b0992c148305224899a16d21fcd56406408bd393`). The pipeline
checks those revisions before running. No Geant4 datasets or MadGraph shower
are needed. On this host,
`/home` is NFS-backed and container root cannot write there, so Docker cache
and output mounts must be under `/tmp`.

From the host:

```bash
CLUSTER_ROOT=/home/atlas/dittmeier/git/cluster_extraction
CLUSTER_REPO="$CLUSTER_ROOT/ColliderML-Production"
CLUSTER_SCRATCH=/tmp/colliderml-paired-$(id -u)
mkdir -p "$CLUSTER_SCRATCH/cache" "$CLUSTER_SCRATCH/output"
test -s "$CLUSTER_ROOT/data/pu0/edm4hep.root"
test -s "$CLUSTER_ROOT/data/pu200/edm4hep.root"
if [ ! -f "$CLUSTER_SCRATCH/cache/odd-v4/xml/OpenDataDetector.xml" ]; then
  git clone --depth 1 --branch v4.0.4 \
    https://gitlab.cern.ch/acts/OpenDataDetector.git \
    "$CLUSTER_SCRATCH/cache/odd-v4"
fi
git -C "$CLUSTER_SCRATCH/cache/odd-v4" rev-parse HEAD
```

If the image is not already installed, build it once from the repository:

```bash
cd "$CLUSTER_REPO"
sudo docker build -f docker/acts-arrow/Dockerfile \
  -t colliderml-acts:7cba36b17 .
```

## Run in Docker

Start the container with the prepared cache and inputs:

```bash
docker run --rm -it --entrypoint /bin/bash \
  -v "$CLUSTER_REPO":/workspace:ro \
  -v "$CLUSTER_ROOT/data":/data:ro \
  -v "$CLUSTER_SCRATCH/cache":/cache \
  -v "$CLUSTER_SCRATCH/output":/output \
  -e COLLIDERML_CACHE=/cache \
  -e SKIP_GENERATION_SETUP=1 \
  -e SKIP_G4_DOWNLOAD=1 \
  colliderml-acts:7cba36b17
```

Inside the container, the setup script builds and installs the ODD factory
library into `/cache/odd-v4-install` if it is not already present. It uses
two build jobs by default (`ODD_BUILD_JOBS=2`). Source ACTS and verify the
library and Python dependencies before starting the processing command:

```bash
set -e
source /workspace/scripts/cli/setup_container_env.sh
source /opt/acts-arrow/setup.sh
test -f /cache/odd-v4-install/lib/libOpenDataDetector.so
git -c safe.directory=/cache/odd-v4 -C /cache/odd-v4 rev-parse HEAD
python3 -c 'import acts, pyedm4hep, pandas, pyarrow, uproot, yaml; print("Runtime imports OK")'
cd /workspace
python3 -m unittest tests.test_paired_nullable_particle_id \
  tests.test_export_paired_clusters tests.test_run_paired_samples
python3 /workspace/scripts/postprocessing/run_paired_samples.py \
  --output /output/paired-samples-truth-n1
```

For more events, use `--events N` and a **new** output directory, for example
`--events 2 --output /output/paired-samples-n2`. The command refuses to
overwrite an existing directory and fails before ACTS if either input has
fewer than N events. It processes both samples with one thread. If it stops,
the partial directory remains for diagnosis but has no `complete.json`.

If the import check fails because the image already has `pyarrow` but not the
other postprocessing packages, install them into the writable cache, export
`PYTHONPATH`, and retry:

```bash
python3 -m pip install --target /cache/pip \
  pyedm4hep pandas pyarrow uproot h5py tqdm pyyaml awkward psutil
export PYTHONPATH="/cache/pip:$PYTHONPATH"
```

## Validate and copy

Within the container, the presence of `complete.json` means both samples
passed ACTS/CSV, converted-hit alignment, particle-ID, and event-count checks:

```bash
python3 - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/output/paired-samples-truth-n1')
summary = json.loads((root / 'complete.json').read_text())
for sample in ('pu0', 'pu200'):
    folder = root / sample
    report = json.loads((folder / 'report.json').read_text())
    hits = pq.read_metadata(folder / 'hits.parquet').num_rows
    cells = pq.read_metadata(folder / 'cells.parquet').num_rows
    particles = pq.read_metadata(folder / 'particles.parquet').num_rows
    assert report['events'] == summary['events_per_sample'] == particles
    assert (hits, cells) == (report['measurements'], report['cells'])
    print(sample, 'events', particles, 'hits', hits, 'cells', cells,
          'multi-particle hits', report['multi_particle_measurements'],
          'null scalar labels', report['null_particle_labels'])
PY
```

After exiting Docker, copy only the completed products to the home directory:

```bash
mkdir -p "$CLUSTER_ROOT/exports/paired-samples-truth-n1"
cp -r "$CLUSTER_SCRATCH/output/paired-samples-truth-n1/pu0" \
      "$CLUSTER_SCRATCH/output/paired-samples-truth-n1/pu200" \
      "$CLUSTER_ROOT/exports/paired-samples-truth-n1/"
cp "$CLUSTER_SCRATCH/output/paired-samples-truth-n1/complete.json" \
   "$CLUSTER_ROOT/exports/paired-samples-truth-n1/"
```

Each sample directory contains `hits.parquet`, `cells.parquet`,
`particles.parquet`, `report.json`, and the ACTS ROOT/CSV outputs under
`acts/`. Hits and cells join on `(campaign, dataset, version, run,
local_event, measurement_id)`; hit `particle_id` joins to the particle ID
within the same event, and every ID in `particle_ids` does likewise. The
`simhit_ids` list retains the ACTS association evidence. Particle Parquet
retains the production converter's
one-row-per-event, list-column layout. Reports include source checksums and
ACTS, ODD, and ColliderML-Production revisions.
The particle table comes directly from EDM4hep; optional ACTS-derived fields
such as `perigee_d0`, `perigee_z0`, and `vertex_primary` may be absent because
this pipeline does not request `particles.root`.
