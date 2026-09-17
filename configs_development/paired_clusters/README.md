# One-event paired cluster samples

## Docker on this host

Digitization starts from EDM4hep. It needs ODD geometry, but no Geant4
datasets or MadGraph shower. The home directory is NFS-backed, so container
root cannot write to cache or output bind mounts there. Use local `/tmp` for
those mounts and copy the finished one-event files back afterward.

From a host shell:

```bash
CLUSTER_ROOT=/home/atlas/dittmeier/git/cluster_extraction
CLUSTER_REPO="$CLUSTER_ROOT/ColliderML-Production"
CLUSTER_SCRATCH=/tmp/colliderml-paired-$(id -u)
mkdir -p "$CLUSTER_SCRATCH/cache" "$CLUSTER_SCRATCH/output" "$CLUSTER_ROOT/output" "$CLUSTER_ROOT/exports"
git clone --depth 1 --branch v4.0.4 \
  https://gitlab.cern.ch/acts/OpenDataDetector.git \
  "$CLUSTER_SCRATCH/cache/odd-v4"
sudo docker run --rm -it --entrypoint /bin/bash \
  -v "$CLUSTER_REPO":/workspace:ro \
  -v "$CLUSTER_ROOT/data":/data:ro \
  -v "$CLUSTER_SCRATCH/cache":/cache \
  -v "$CLUSTER_SCRATCH/output":/output \
  -e COLLIDERML_CACHE=/cache \
  -e SKIP_GENERATION_SETUP=1 \
  -e SKIP_G4_DOWNLOAD=1 \
  -e SKIP_POSTPROCESSING_DEPS=1 \
  colliderml-acts:7cba36b17
```

Clone ODD only if it is not already at the indicated cache path. Inside the
container, run:

```bash
set -e
source /workspace/scripts/cli/setup_container_env.sh
source /opt/acts-arrow/setup.sh
test -f /cache/odd-v4-install/lib/libOpenDataDetector.so
cd /workspace
python3 -c 'import acts, acts.examples.edm4hep; print(acts.__version__)'
python3 scripts/simulation/digi_and_reco.py \
  --config configs_development/paired_clusters/one_event.yaml \
  --events 1 --threads 1 \
  --input-file /data/pu0/edm4hep.root --output /output/pu0
python3 scripts/simulation/digi_and_reco.py \
  --config configs_development/paired_clusters/one_event.yaml \
  --events 1 --threads 1 \
  --input-file /data/pu200/edm4hep.root --output /output/pu200
```

After both runs, exit the container and copy the results to the home directory:

```bash
cp -r "$CLUSTER_SCRATCH/output/pu0" "$CLUSTER_ROOT/output/"
cp -r "$CLUSTER_SCRATCH/output/pu200" "$CLUSTER_ROOT/output/"
```

Export on the host with the `paired-clusters` conda environment. The exporter
records the ACTS and ODD revisions and SHA-256 hashes of the EDM4hep input,
digitization config, and measurements ROOT file.

```bash
ACTS_REVISION=7cba36b173dd73a14342cc65043a43179a4b1dbd
ODD_REVISION=$(git -C "$CLUSTER_SCRATCH/cache/odd-v4" rev-parse HEAD)
for cluster_sample in pu0 pu200; do
  if [ "$cluster_sample" = pu0 ]; then
    cluster_campaign=hard_scatter
  else
    cluster_campaign=full_pileup
  fi
  PYTHONNOUSERSITE=1 \
  /home/atlas/dittmeier/.conda/envs/paired-clusters/bin/python \
    "$CLUSTER_REPO/scripts/postprocessing/export_paired_clusters.py" \
    --campaign "$cluster_campaign" --dataset ttbar --version v1 --run 0 \
    --acts-revision "$ACTS_REVISION" --odd-revision "$ODD_REVISION" \
    --edm4hep "$CLUSTER_ROOT/data/$cluster_sample/edm4hep.root" \
    --digi-config "$CLUSTER_REPO/scripts/simulation/odd-full-geo-digi-config.json" \
    --measurements-root "$CLUSTER_ROOT/output/$cluster_sample/measurements.root" \
    --csv-dir "$CLUSTER_ROOT/output/$cluster_sample/csv" \
    --output "$CLUSTER_ROOT/exports/$cluster_sample"
done
```

Each export contains `hits.parquet`, `cells.parquet`, and `report.json`.
`simhit_ids` on each hit retains all ACTS measurement-to-SimHit links. A
pileup-200 event has a Poisson-distributed number of additional interactions;
it need not contain exactly 200. Do not combine these new measurements with
the published tracker-hit Parquet files.

## Convert the same events to particles and single-label tracker hits

The existing production converter can read the event-0 EDM4hep file and the
new `measurements.root` together. It assigns one `particle_id` per measurement
by matching the measurement's true position to an EDM4hep tracker SimHit.
This is a single matched label, not a complete list of pileup contributors.
The ACTS `simhit_ids` in the paired hits remain available for a later
multi-contributor product.

Start the same Docker image with the mounts above, but set
`SKIP_POSTPROCESSING_DEPS=0` (or omit that environment variable). Then run
the following **inside** the container. This uses `/output` scratch space;
the source EDM4hep files and original paired outputs are not modified.

```bash
set -e
source /workspace/scripts/cli/setup_container_env.sh
source /opt/acts-arrow/setup.sh
python3 -c 'import pyedm4hep, pandas, pyarrow, uproot; print("Postprocessing imports OK")'
for cluster_sample in pu0 pu200; do
  mkdir -p "/output/paired-conversion-input/$cluster_sample/runs/0"
  ln -sfn "/data/$cluster_sample/edm4hep.root" \
    "/output/paired-conversion-input/$cluster_sample/runs/0/edm4hep.root"
  ln -sfn "/output/$cluster_sample/measurements.root" \
    "/output/paired-conversion-input/$cluster_sample/runs/0/measurements.root"
  python3 /workspace/scripts/postprocessing/convert_all.py \
    --config "/workspace/configs_development/paired_clusters/convert_$cluster_sample.yaml" \
    --chunk-index 0
done
```

If the import check fails because the image already has `pyarrow` but not the
other postprocessing packages, install them into the writable cache and retry:

```bash
python3 -m pip install --target /cache/pip \
  pyedm4hep pandas pyarrow uproot h5py tqdm pyyaml awkward psutil
export PYTHONPATH="/cache/pip:$PYTHONPATH"
```

For each setting, expect one particle file and one tracker-hit file under
`/output/paired-conversion/<campaign>/ttbar/v1/parquet/`. Check them in the
container before copying anything back:

```bash
python3 - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

base = Path('/output/paired-conversion')
for campaign in ('hard_scatter', 'full_pileup'):
    root = base / campaign / 'ttbar' / 'v1' / 'parquet'
    for kind, subdir in (('particles', 'truth/particles'),
                         ('tracker_hits', 'reco/tracker_hits')):
        files = list((root / subdir).glob('*.events0-0.parquet'))
        assert len(files) == 1, (campaign, kind, files)
        table = pq.read_table(files[0])
        assert table.num_rows == 1, (files[0], table.num_rows)
        assert table['event_id'][0].as_py() == 0
        print(campaign, kind, len(table[table.column_names[1]][0].as_py()), files[0])
PY
```

On the host, copy each newly converted particle file into its paired export:

```bash
cp "$CLUSTER_SCRATCH/output/paired-conversion/hard_scatter/ttbar/v1/parquet/truth/particles/"*.events0-0.parquet \
  "$CLUSTER_ROOT/exports/pu0/particles.parquet"
cp "$CLUSTER_SCRATCH/output/paired-conversion/full_pileup/ttbar/v1/parquet/truth/particles/"*.events0-0.parquet \
  "$CLUSTER_ROOT/exports/pu200/particles.parquet"
```

Do not replace `exports/*/hits.parquet` with the converted tracker-hit files:
their schemas differ and the converter does not add `particle_id` to the paired
hits. Attaching the single labels to those hits still needs a checked
row-alignment step after this conversion succeeds.

Before that step, check the converted event against the paired hits on the
host (requires the `paired-clusters` environment, which includes PyArrow and
NumPy):

```bash
export CLUSTER_ROOT CLUSTER_SCRATCH
PYTHONNOUSERSITE=1 /home/atlas/dittmeier/.conda/envs/paired-clusters/bin/python - <<'PY'
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root = Path(os.environ['CLUSTER_ROOT'])
scratch = Path(os.environ['CLUSTER_SCRATCH']) / 'output' / 'paired-conversion'
for sample, campaign in (('pu0', 'hard_scatter'), ('pu200', 'full_pileup')):
    paired = pq.read_table(root / 'exports' / sample / 'hits.parquet')
    converted_dir = scratch / campaign / 'ttbar' / 'v1' / 'parquet' / 'reco' / 'tracker_hits'
    files = list(converted_dir.glob('*.events0-0.parquet'))
    assert len(files) == 1, files
    converted = pq.read_table(files[0])
    def event_list(table, name):
        return table[name][0].as_py()
    count = paired.num_rows
    assert converted.num_rows == 1 and len(event_list(converted, 'x')) == count
    assert paired['measurement_id'].to_pylist() == list(range(count))
    for paired_name, converted_name in (('global_x', 'x'), ('global_y', 'y'), ('global_z', 'z')):
        assert np.allclose(paired[paired_name].to_numpy(), event_list(converted, converted_name),
                           rtol=0, atol=2e-4), (sample, paired_name)
    geometry = np.asarray(paired['geometry_id'].to_numpy(), dtype=np.uint64)
    for name, shift, mask in (('volume_id', 56, 0xff), ('layer_id', 36, 0xfff),
                              ('surface_id', 8, 0xfffff)):
        assert np.array_equal((geometry >> shift) & mask, event_list(converted, name)), (sample, name)
    particles = pq.read_table(root / 'exports' / sample / 'particles.parquet')
    assert particles.num_rows == 1 and particles['event_id'][0].as_py() == 0
    known_ids = set(event_list(particles, 'particle_id'))
    labels = event_list(converted, 'particle_id')
    missing = sum(label is None for label in labels)
    unknown = {label for label in labels if label is not None and label not in known_ids}
    assert not unknown, (sample, len(unknown))
    print(sample, 'hits', count, 'particles', len(known_ids), 'null labels', missing)
PY
```
