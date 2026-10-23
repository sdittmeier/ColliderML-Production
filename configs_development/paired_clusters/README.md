# Paired ttbar cluster samples

One command builds hits, constituent cells, and particles from the first N
events of each hard-scatter and full-pileup EDM4hep file. N defaults to 1.
ACTS digitization and the existing production particle/tracker-hit converter
run on the same input events. The exporter checks their alignment before
adding particle truth to each paired hit. It retains all `simhit_ids` and
resolves them to distinct `particle_ids` contributing to each measurement.
`particle_ids_complete` is true only when every associated SimHit resolves;
unresolved indices remain in `unresolved_simhit_ids`. The scalar `particle_id`
is filled only for complete, single-particle measurements. A merged or
incompletely resolved measurement has a null scalar label. The production
coordinate-match label is retained separately as `converted_particle_id`;
disagreements with resolved SimHit truth are counted in `report.json`, not
silently substituted for it.

There is no cell-level particle label or track reconstruction.
Every cell retains its native activation in `value`. By default, pixel cells
also receive a **P99.9 quantized `pixel_code`**, with frozen calibration, while
strip codes are null. See [Pixel quantization](#pixel-quantization) below.
This run enables both `simhits.root` and `particles.root`. Existing paired
outputs without `particles.root` cannot acquire ACTS perigee/vertex fields
through `--resume-export`; use a new output directory for a fresh ACTS run,
but reuse the downloaded EDM4hep files.

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
  tests.test_export_paired_clusters tests.test_run_paired_samples \
  tests.test_production_pixel_quantization
python3 /workspace/scripts/postprocessing/run_paired_samples.py \
  --output /output/paired-samples-features-n1
```

For more events, use `--events N` and a **new** output directory, for example
`--events 2 --output /output/paired-samples-features-n2`. The command refuses to
overwrite an existing directory and fails before ACTS if either input has
fewer than N events. It processes both samples with one thread. If it stops,
the partial directory remains for diagnosis but has no `complete.json`.

If digitization and conversion finished but export failed, update the mounted
repository and resume only the export in the same container environment:

```bash
python3 /workspace/scripts/postprocessing/run_paired_samples.py \
  --events 2 --output /output/paired-samples-features-n2 --resume-export
```

The resume command checks the existing input links, event count, ACTS files,
and conversion files. It does not rerun ACTS or the production converter.
It uses the requested quantization option (default P99.9), even if the
original attempt used a different option; the final reports record it.
It cannot add perigee/vertex fields to an older run without `particles.root`.
`complete.json` is written only after both exports pass. Unresolved SimHits
are reported and leave their hit's truth incomplete; they do not stop export.

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

root = Path('/output/paired-samples-features-n1')
summary = json.loads((root / 'complete.json').read_text())
for sample in ('pu0', 'pu200'):
    folder = root / sample
    report = json.loads((folder / 'report.json').read_text())
    hits = pq.read_metadata(folder / 'hits.parquet').num_rows
    cells = pq.read_metadata(folder / 'cells.parquet').num_rows
    particles = pq.read_metadata(folder / 'particles.parquet').num_rows
    assert report['events'] == summary['events_per_sample'] == particles
    assert (hits, cells) == (report['measurements'], report['cells'])
    assert all(report['particle_feature_coverage'][name]['present']
               for name in ('perigee_d0', 'perigee_z0', 'vertex_primary'))
    print(sample, 'events', particles, 'hits', hits, 'cells', cells,
          'multi-particle hits', report['multi_particle_measurements'],
          'null scalar labels', report['null_particle_labels'],
          'unresolved SimHit links', report['unresolved_simhit_links'],
          'incomplete truth hits', report['incomplete_truth_measurements'],
          'converted-label disagreements', report['converted_label_disagreements'])
PY
```

After exiting Docker, copy only the completed products to the home directory:

```bash
mkdir -p "$CLUSTER_ROOT/exports/paired-samples-features-n1"
cp -r "$CLUSTER_SCRATCH/output/paired-samples-features-n1/pu0" \
      "$CLUSTER_SCRATCH/output/paired-samples-features-n1/pu200" \
      "$CLUSTER_ROOT/exports/paired-samples-features-n1/"
cp "$CLUSTER_SCRATCH/output/paired-samples-features-n1/complete.json" \
   "$CLUSTER_ROOT/exports/paired-samples-features-n1/"
```

Each sample directory contains `hits.parquet`, `cells.parquet`,
`particles.parquet`, `report.json`, and the ACTS ROOT/CSV outputs under
`acts/`. Hits and cells join on `(campaign, dataset, version, run,
local_event, measurement_id)`; hit `particle_id` joins to the particle ID
within the same event, and every ID in `particle_ids` does likewise. The
`simhit_ids` list retains the ACTS association evidence. `particle_ids` can be
partial when `particle_ids_complete` is false; never interpret a partial list
as the full set of contributors. Particle Parquet
retains the production converter's
one-row-per-event, list-column layout. Reports include source checksums and
ACTS, ODD, and ColliderML-Production revisions.
The particle table comes from EDM4hep, enriched with ACTS-derived
`perigee_d0`, `perigee_z0`, and `vertex_primary` from `particles.root`.
Unmatched EDM particles are retained with null enrichment fields, and the
report gives per-field coverage.

## Pixel quantization

Quantization is an optional feature calculation at export time, not a change
to ACTS digitization or reconstruction. All new cell files have two columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `value` | float64 | Original ACTS activation, unchanged for every cell |
| `pixel_code` | nullable uint8 | Positive code 1-15 for pixels; null for all other volumes or when disabled |

Pixel volumes are **16, 17 and 18 only**. Short strips (23-25), long strips
(28-30), and other volumes are not quantized. A null code means "not encoded",
not an active zero-valued channel. Retaining native `value` is useful for
validation, recalibration and the future strip study. No redundant native
copy is added to each row. The code range could fit four bits, but the Parquet
column deliberately uses the standard nullable **uint8** type.

Choose a preset with `--pixel-quantization`; **p99.9 is the default**:

```bash
# Default P99.9; creates both pu0 and pu200, each with native values and codes.
python3 /workspace/scripts/postprocessing/run_paired_samples.py \
  --events 50 --output /output/paired-samples-p99_9-n50

# Other fixed presets (use a new output directory for each run).
python3 /workspace/scripts/postprocessing/run_paired_samples.py \
  --events 50 --pixel-quantization p99.5 --output /output/paired-samples-p99_5-n50
python3 /workspace/scripts/postprocessing/run_paired_samples.py \
  --events 50 --pixel-quantization p99 --output /output/paired-samples-p99-n50

# Disable the extra encoding; value is still native and pixel_code is all null.
python3 /workspace/scripts/postprocessing/run_paired_samples.py \
  --events 50 --pixel-quantization none --output /output/paired-samples-native-n50
```

These are alternative full-run commands, not a requirement to rerun ACTS for
each preset. The standalone `export_paired_clusters.py` accepts the same
option and can use existing ACTS CSV/ROOT and converted-particle outputs.
For an **incomplete** pipeline run, `--resume-export` also reuses these outputs.
Existing completed exports are never automatically upgraded or changed.

The authoritative definitions are in
[`pixel_quantization_v1.json`](pixel_quantization_v1.json), version
`odd-pixel-linear-v1`. They contain all 14 fixed boundaries per preset and the
calibration provenance. Do not change these boundaries under the same version;
a later calibration should have a new version and file.

| Preset | Code-15 edge (native activation) | Nominal bin width |
| --- | --- | --- |
| `p99.9` | 0.16547740542600492 | 0.011105528959000351 |
| `p99.5` | 0.14173012115 | 0.009409294367857143 |
| `p99` | 0.13679915850000002 | 0.00905708275 |

The calibration used pooled pixels from **ttbar pu200 run 0, events 0-24**.
The same boundaries are used for every event, pu0 and pu200. Names refer to
that original calibration population: **no percentile is fitted to a new
event, file or physics sample**. These are path-length-based codes, not
calibrated ATLAS ToT. Portability to other samples still needs validation.

Encoding is exactly the notebook's linear scan: 14 equal-width bins start at
activation 0.01, and code 15 starts at the selected edge. A value equal to a
boundary goes into the upper bin. Values above the last boundary stay at 15;
every retained pixel gets at least 1. Nonfinite/nonpositive values, or pixel
values below 0.01 when encoding is enabled, cause an error, not suppression.
There is no extra noise, threshold, random rounding, or charge decoding.

Reports distinguish all cells assigned 15 from **genuine clipping** beyond
the end of its nominal-width bin (`code15_edge + bin_width`). All cell counts,
addresses, cluster membership, hit truth and native hit positions are unchanged;
positions are **not** recomputed from codes. The existing validation notebook
can continue using `value` as native activation.

`report.json` contains `cell_quantization` with the scheme, preset version and
checksum, exact boundaries, calibration inputs, pixel/nonpixel counts, code
histogram, and clipping counts. The same object is embedded in the **cell
Parquet schema metadata** as `colliderml.cell_quantization` so the encoding
travels with the file. `complete.json` records the selected scheme as well.
Quantized presets require the exact geometric digitization configuration used
in calibration; a different config fails before ACTS. Use `none` to disable
codes when studying a different setup rather than applying an unvalidated mapping.

For an additional bounded-memory encoding check after the run, inside Docker:

```bash
cd /workspace
python3 - <<'PY'
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.postprocessing.utils.pixel_quantization import METADATA_KEY, PixelQuantizer

root = Path('/output/paired-samples-p99_9-n50')
for sample in ('pu0', 'pu200'):
    folder = root / sample
    report = json.loads((folder / 'report.json').read_text())
    source = pq.ParquetFile(folder / 'cells.parquet')
    encoding = json.loads(source.schema_arrow.metadata[METADATA_KEY])
    assert encoding == report['cell_quantization']
    assert source.schema_arrow.field('pixel_code').type == pa.uint8()
    quantizer = PixelQuantizer(encoding['scheme'])
    for batch in source.iter_batches(columns=['geometry_id', 'value', 'pixel_code']):
        expected = quantizer.encode(batch.column('geometry_id').to_numpy(),
                                    batch.column('value').to_numpy())
        codes = batch.column('pixel_code')
        assert np.array_equal(codes.is_valid().to_numpy(), expected != 0)
        assert np.array_equal(codes.fill_null(0).to_numpy(), expected)
    assert quantizer.report() == encoding
    print(sample, encoding['scheme'], 'pixel cells', encoding['pixel_cells'], 'OK')
PY
```
