# Corrected Higgs-portal + pu200 pilot

This pilot reuses corrected **pu0 v2 run 0** and its signal HepMC, generates fresh
SoftQCD minimum-bias events, merges at Poisson mean 200, then runs corrected
DD4hep/Geant4 simulation and the existing paired Parquet flow. The output retains
native activation and adds frozen P99.9 pixel codes; strip codes remain null.
No tracks are reconstructed by the paired export.

The physics point is h(125 GeV) -> aa -> 4b, m_a = 55 GeV and c*tau = 100 mm.
The source URLs, expected sizes, signal MD5 and upstream references are pinned
in [input_sources.json](input_sources.json). The private upstream fix commit
mentioned in the production log is not publicly retrievable. Our small port is
therefore based on the published settings and the exact DD4hep API/source, and
must pass the runtime gates; it is not claimed to be a bitwise upstream reproduction.

## Current export interface

New productions write `v2_cells_compat1` using the nested CERN-compatible hit
schema and dataset-wide event IDs. Source physics remains Higgs-portal v2. Existing
simulation, perigee recovery and P99.9 encoding are unchanged. For a subsequent
shard, supply **both** `--signal-start` (input selection) and `--event-id-offset`
(output identity); these are separate concepts.

For the next 100-event shard after migrated runs 3/4, choose an unused run ID and
use `--signal-start 200 --event-id-offset 200`. Inspect the live pileup registry
before selecting the run. Old completed productions must use the standalone
Parquet migration, not a resume stage with modified source checksums.

See [schema, migration and validation](../../docs/paired_schema_compatibility.md).
The historical commands below describe earlier runs; add the explicit event-ID
offset when invoking the current launcher with a nonzero run.

## From a fresh host bash shell

The previously built `colliderml-acts:7cba36b17` Docker image and existing cache
are reused. You need host `python3`, `git`, working `sudo docker`, network access
and disk space for approximately 3.34 GB of source inputs, Geant4 data (~2 GB if
uncached), plus generated events and ROOT/CSV/Parquet outputs.

```bash
CLUSTER_ROOT=/home/atlas/dittmeier/git/cluster_extraction
cd "$CLUSTER_ROOT/ColliderML-Production"

bash scripts/cli/run_higgs_portal_pilot.sh
```

This default command produces **one event**, run 0, with one simulation thread,
in `/tmp/colliderml-paired-$(id -u)/output/higgs-portal-pilot-run0-n1` on the host.
It runs the stages below and stops on any failure. It does not open an interactive
container and does not need a conda environment on the host. The input downloader
uses only the Python standard library. Downloads resume via `.part` files; existing
files are not overwritten. The ROOT input's size is checked (no published checksum
is available); the signal is checked against the published MD5. Local SHA256
checksums are recorded in `data/higgs_portal/v2/download_provenance.json`.

If the Docker image or cache has another location, set `COLLIDERML_IMAGE` or
`CLUSTER_SCRATCH`. Unlike redigitizing ttbar, this really needs Geant4 datasets:
the wrapper explicitly sets **SKIP_G4_DOWNLOAD=0**. Missing/incomplete physics
datasets cause a preflight failure, not an attempted simulation with missing data.
The preflight resolves individual dataset paths from `geant4-config --datasets`
and `GEANT4_DATA_DIR`: this image's `geant4.sh` does not export the individual
`G4...DATA` variables. Resolved paths are validated and exported to subprocesses.
If an older pilot stopped with `G4ENSDFSTATEDATA=/missing-geant4-dataset`, rerun
the same wrapper after updating the code. That failure preceded generation and
seed reservation; the downloaded inputs and Geant4 datasets are reused.
`SKIP_GENERATION_SETUP=1` skips the unnecessary MadGraph interface; the ACTS Pythia8
plugin is still used to generate minimum-bias events.

## Stages and gates

1. Check pinned ACTS/ODD, ODD factory library, Geant4 physics datasets, imports,
   DD4hep LLP option support, corrected source sizes/MD5 and pu0 collections.
2. Select signal entries and record their original HepMC event numbers. Renumber
   the selected file for ACTS. Generate a new exclusive minimum-bias file, with a
   five-sigma Poisson buffer, and merge using the pinned native ACTS reader.
3. Simulate pu200 with the local LLP fix. **Only merge applies vertex smearing**;
   Geant4's additional vertex offset/sigma are zero, avoiding double smearing.
4. Require exactly N ROOT events, all required collections, unique ROOT-to-HepMC
   event mapping, two matched LLPs/event and endpoints within 1 mm of HepMC. For
   charged stable LLP descendants produced beyond radius 10 mm, reject placement
   at the collision point (ROOT radius below 1 mm). Match by generator particle
   PDG/momentum, not by assuming array order. A one-event sample might not exercise
   the displaced-descendant gate; the report and console explicitly flag that.
5. Only after simulation passes, redigitize both inputs and export hits, cells and
   particles using the existing pairing/particle/code validation. The output is
   labelled `higgs_portal/v2_cells_pilot`, not misrepresented as public production v2.

The simulation fix sets `physics.alternativeDecayStatuses={22}` and adds PDG 35
to `rejectPDGs` without removing the default rejection set. The extra-particle
table defines PDG 36 as unstable with mass 55 GeV and c*tau 100 mm. In the pinned
DD4hep implementation, a **zero input width marks the particle stable even with
a positive lifetime**. The table uses width flag -1, causing DD4hep to mark it
unstable and derive the width from the supplied lifetime. Preparation preserves
the **full existing production particle table**, changing only the PDG 36 row;
the small checked-in table is an override specification, not a replacement for
all other particle definitions. The generated full table and its checksum stay
with the pilot outputs.

These are pilot sanity checks, not a complete LLP detector/physics validation.
In particular they do not replace the larger decay-length/vertex comparison
performed for upstream v2, or the deferred quantization/resolution studies.
LLP PDG/momentum matching remains strict. Stable decay daughters may use a
unique bounded fallback within `0.001 * max(|p|, 0.02 GeV)` in vector momentum
distance, because the first real event showed small changes between preassigned
HepMC daughter momenta and stored Geant4 momenta. Ambiguous or larger differences
still fail. The validation report records fallback counts and maximum relative
adjustment; vertex and endpoint checks are not relaxed. For run 1 event 0 the
largest observed adjustment was 0.0128%, with 26 charged descendants checked and
22 exercising the displacement gate.

## Sequential writer fix and recovery

The pinned DDG4 helper creates a shared EDM4hep event action even in sequential
mode. A one-muon runtime test reproduced two ROOT entries for one event; using a
non-shared action produced exactly one. `ddsim_run.py` now uses a non-shared
EDM4hep output plugin for single-thread ROOT output, retaining the same event
parameters, run header and numbering offsets. Multi-thread output is unchanged.
The writer sources are pinned to DD4hep revision
`a3fdbc4302494ee6dad3e3da460ec42b2b61c99f`:
[DDG4 helper](https://github.com/AIDASoft/DD4hep/blob/a3fdbc4302494ee6dad3e3da460ec42b2b61c99f/DDG4/python/DDG4.py)
and [EDM4hep writer](https://github.com/AIDASoft/DD4hep/blob/a3fdbc4302494ee6dad3e3da460ec42b2b61c99f/DDG4/edm4hep/Geant4Output2EDM4hep.cpp).

For the existing run-1 pilot that stopped with two entries instead of one, do
not regenerate pileup or rerun Geant4. First run the explicit recovery stage:

```bash
bash scripts/cli/run_higgs_portal_pilot.sh \
  --events 1 --threads 1 --run 1 --signal-start 0 \
  --stage recover --output /output/higgs-portal-pilot-run1-n1
```

Recovery requires exactly two copies of every logical event and exact equality
of all event branches except `EventHeader.timeStamp`. It copies the first entry
of each pair through podio, retaining particle relations and all frame categories,
then compares every copied branch including metadata and runs the LLP gates.
Only after validation passes does it archive the original as
`simulation/pu200/edm4hep.duplicate-writer.root` and install the recovered
`edm4hep.root`. `writer_recovery.json` records entry selection, checksums and
pre/post-fix code provenance. Unrelated code/input changes are rejected.

After recovery succeeds, export without simulation:

```bash
bash scripts/cli/run_higgs_portal_pilot.sh \
  --events 1 --threads 1 --run 1 --signal-start 0 \
  --stage export --output /output/higgs-portal-pilot-run1-n1
```

Recovery is explicit, not automatic deduplication in normal production. Different
physics contents, unexpected multiplicities or failed LLP gates remain errors.
The original Geant4 timing remains in `simulation/pu200/timing_summary.txt`;
recovery does not invent a new simulation runtime. No Docker image rebuild is
required: the wrapper uses the updated repository through its bind mount.

## Pileup non-reuse

The pilot never uses the old archived pu200 background. Each new pilot run has
a fresh generated file and an exclusive seed reservation under
`$CLUSTER_SCRATCH/cache/higgs-portal-pilot/pileup-seeds.json`. A file lock protects
concurrent reservations. Colliding seeds/run keys fail; failed reservations are
not recycled. The **entire** generated file, including unused buffer events, is
reserved for this one merge and is not reused for another pilot run.

The pinned ACTS reader consumes physical events sequentially and aborts at EOF;
it does not randomly resample or wrap the background file. Generation/merging
uses one thread for deterministic consumption. Simulation can use more threads.
The pilot records file hashes, seed, count, source-ID range, selected signal
entries and code/config hashes in `pilot.json`.

This prevents file-event reuse among runs using this persistent registry; it is
not proof of global uniqueness against unrelated productions with untracked
seeds. Do not delete the registry or copy a reserved background file to another
production. Signal reuse is a separate issue. This pilot's pu0 ROOT entries and
pu200 signal selections are **not assumed event-by-event matched**: upstream
multithreaded simulation reordered events. Both products are from the same
physics point; the original event mapping is retained explicitly.

## Logs and results

From a second host shell, follow the Geant4 log once it exists:

```bash
tail -f /tmp/colliderml-paired-$(id -u)/output/higgs-portal-pilot-run0-n1/simulation.log
```

The pilot directory contains `generation.log`, `simulation.log`, `export.log`,
`pilot.json`, generated inputs under `generation/`, Geant4 output under
`simulation/pu200/`, and `simulation_validation.json`. Final products are under
`paired/{pu0,pu200}/`; `paired/complete.json` is the export completion marker and
`pilot.json` must additionally contain `"complete": true`.

`pilot.json` reports Geant4 wall time and seconds/event **with the recorded thread
count**; single-event latency is not eight-thread batch throughput. It also records
generation/merge and full digitization/conversion/export times separately.

Failed attempts remain for diagnosis. Do not delete completion markers or change
run labels to disguise a retry. If the run fails, bring back the relevant log.
`--stage prepare`, `simulate` and `export` permit a deliberate later-stage start
using an already prepared pilot; they check the same code/run/events metadata.
For example, after successful `--stage prepare`, rerun with `--stage simulate`
and the same run/events/signal-start/output arguments. Existing simulation or
paired output directories are never silently replaced. A failed simulation
directory needs explicit diagnosis before retrying.

## Timing pilot, after the one-event check

Use a **new run ID, fresh output directory and fresh background reservation**:

```bash
bash scripts/cli/run_higgs_portal_pilot.sh \
  --events 16 --threads 8 --run 1 --signal-start 1 \
  --output /output/higgs-portal-pilot-run1-n16
```

For new runs, `--signal-start` advances both the selected signal HepMC entries
and the pu0 ROOT-entry range. Historical runs made before the `pu0_selection`
manifest field was added always used the first requested pu0 ROOT entries,
regardless of `--signal-start`; inspect their provenance before concatenating
pilot outputs with production shards. The background is fresh.
Before a 1,000-event campaign, review timing/RAM/disk use, require the displaced
gate to be exercised, and plan production sharding/source-event bookkeeping.

After success, copy the whole pilot directory from `/tmp` to a new directory
under `exports/` to preserve its logs, provenance and generation/simulation inputs.
Copying only `paired/` is sufficient for analysis but loses the pilot-level
simulation evidence unless that is preserved separately.

## Resume after a post-simulation validation failure

If Geant4 completed successfully but descendant matching failed, retain the run
and its ROOT file. The validator keeps the original strict LLP matching and the
0.1% daughter-momentum allowance. A further fallback handles physical decay
boosts: independently identify the parent, verify its relation graph reaches the
same signal LLP, require a unique daughter with the correct PDG and parent link,
and require its production vertex within `1e-4 mm` of that parent's simulated
endpoint. Boost the original daughter through the original parent's rest frame
into the frame defined by the parent's recorded momentum at decay. The predicted
and stored daughter momenta must agree within the existing 0.1% convention
(`max(|p|, 0.02 GeV)`). Neutral descendants remain included. No simulation values
are modified. Unexplained and ambiguous matches still fail.

The report records each boost-assisted match, its original discrepancy and its
prediction residual. Descendant matching failures are collected across all
events into `simulation_validation.json` before failing the validation gate.
LLP endpoint and structural failures still stop immediately.

For the existing 100-event run, from a host shell in the same tmux/Slurm session:

```bash
cd /home/atlas/dittmeier/git/cluster_extraction/ColliderML-Production

bash scripts/cli/run_higgs_portal_pilot.sh \
  --events 100 --threads 25 --run 3 --signal-start 0 --event-id-offset 0 \
  --stage validate --output /output/higgs-portal-run3-n100 && \
bash scripts/cli/run_higgs_portal_pilot.sh \
  --events 100 --threads 25 --run 3 --signal-start 0 --event-id-offset 0 \
  --stage export --output /output/higgs-portal-run3-n100
```

`validate` checks prepared-input checksums and saved event/thread settings, checks
that the ROOT checksum is unchanged during validation, and records the successful
validation and its source checksums. It permits an updated validator script while
preserving the original producer source checksums; changes to other production
sources are rejected. `export` accepts that recorded validator version and starts
only after validation succeeds. Neither command generates pileup or reruns
Geant4. No image rebuild is required. The `25` thread argument identifies the
saved simulation settings; validation/export remain serial. Reading the full
merged HepMC reference can take tens of minutes; progress is printed every ten
events. Paired products are written under the existing run's `paired/` directory.


## Next disjoint 100-event shard, using persistent storage

Run 3 contains source entries 0–99. Use run 4 and source entries 100–199 next.
The two source files remain independent event selections: identical positions in
pu0 ROOT and signal HepMC do not establish event-by-event pairing.

For a nonzero `--signal-start`, preparation writes a selected pu0 PODIO file at
`inputs/pu0/edm4hep.root`, used consistently by ACTS, the production converter,
and the paired exporter. `pu0_selection.json` and `pilot.json` record its source
entry range, checksum and verification. All physical fields and metadata frames
are checked exactly against the original entries. Current EDM4hep schema evolution
removes the two obsolete MCParticle `colorFlow` fields and changes EventHeader
`eventNumber`/`runNumber` storage from signed 32-bit to unsigned 64/32-bit. Only
those documented transformations are permitted and recorded; the invalid run
sentinel -1 becomes unsigned 4294967295. Event numbers are not renumbered.
The original downloaded file remains untouched. Any other schema/value change
fails before Geant4 starts.

From a fresh login shell inside tmux, request a **new** Slurm allocation; an old
allocation may have too little remaining time or RAM:

```bash
srun --partition=debug --ntasks=1 --cpus-per-task=25 --mem=192G \
  --time=08:00:00 --pty bash
```

Inside that allocation:

```bash
cd /home/atlas/dittmeier/git/cluster_extraction/ColliderML-Production
export CLUSTER_SCRATCH=/mnt/data2/data/collider_ml/production

test -s "$CLUSTER_SCRATCH/cache/higgs-portal-pilot/pileup-seeds.json"
test ! -e "$CLUSTER_SCRATCH/output/higgs-portal-run4-n100"

bash scripts/cli/run_higgs_portal_pilot.sh \
  --events 100 --threads 25 --run 4 --signal-start 100 --event-id-offset 100 \
  --output /output/higgs-portal-run4-n100
```

The launcher applies `SLURM_CPUS_PER_TASK` as Docker's CPU quota and
`SLURM_MEM_PER_NODE` (MiB) as its memory limit, since Docker otherwise runs outside
the Slurm task cgroup. The preceding run used approximately 130 GiB during Geant4;
192 GiB provides headroom. No image rebuild is needed. Allow roughly 4–6 hours;
export remains serial after the multithreaded Geant4 step.

All outputs stay under
`/mnt/data2/data/collider_ml/production/output/higgs-portal-run4-n100/`.
The persistent registry reserves a fresh run-4 pileup seed; the namespace must
remain unchanged. No Geant4 event or pileup generation is started by the unit and
input-selection smoke tests. After completion, point `PAIRED_ROOT` in the
validation notebook at this run's `paired` directory. Publishing additional
homebrew shards is a separate step; do not overwrite the existing shard.
