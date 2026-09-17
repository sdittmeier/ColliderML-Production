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
