# Recovery-only truth-perigee writer

This rebuild patches `RootParticleWriter` at ACTS revision
`7cba36b173dd73a14342cc65043a43179a4b1dbd`. It updates the ROOT I/O library
and its Python binding. No Geant4 rerun or Parquet schema change is involved.

## Policy: writer-recovery-v2

1. Run the original charged-particle propagation with its original settings and
   unchanged field. Return an originally finite d0/z0 pair directly, including
   its other original perigee components. Do not apply new domain or minimum
   checks to successful legacy results.
2. Retry only failures/non-finite d0/z0, first using `ForcedSurfaceReached` at
   1e-6 mm surface tolerance, then damped target corrections. Loop protection
   and the existing 1,000-step limit remain enabled.
3. Restrict retries to a configured cylinder. The paired ODD configuration uses
   `perigee_recovery_max_r: 1160.0` and
   `perigee_recovery_max_abs_z: 3000.0`, in mm. These match `sol_rmin` and
   `sol_hlength` in the pinned ODD `xml/OpenDataDetectorEnvelopes.xml`.
   Field queries outside that cylinder fail the retry, including trial-step
   queries; production positions and recovered endpoints must also lie inside.
4. Require finite recovered parameters/path and a transverse-distance minimum.
   Otherwise retain NaNs. Never fill failures with zero or remove particles.

**There is no field-boundary splitting or field extension.** The recovery limits
apply only to retries. Legacy finite values outside the central field remain
unchanged and are not newly certified as accurate. In particular, the two
previously identified finite pu200 boundary discrepancies remain unchanged by
this deliberately narrower policy. The boundary-dependent pu0 recovery remains
NaN, as does the very-low-pT pu200 pion.

The generic C++ configuration exposes `perigeeRecoveryMaxR` and
`perigeeRecoveryMaxAbsZ`, defaulting to the largest finite double. The paired
YAML explicitly supplies the ODD limits above. Older images fail clearly when
those fields are requested. The former `perigeeFieldBoundaryZ` option is removed.

`particles.root` stores a `TNamed` object `colliderml_perigee_writer`, whose JSON
records the algorithm, ACTS base revision, patch checksum, recovery limits,
`preserve_legacy_finite: true` and `boundary_splitting: false`. The exporter reads
this identity from the actual ROOT input into `report.json`; it never infers it
from the current runtime. The installed manifest is
`/opt/acts-arrow/perigee_patch.json`. The patch checksum is SHA256 of the patch
file's SHA256, a colon, and the helper header's SHA256.

## Build and validate

From the repository root:

```bash
docker build --network=none --build-arg JOBS=1 \
  -t colliderml-acts:recovery-candidate docker/acts-perigee
```

This reuses existing image layers. Only the three affected tracked ACTS source
files are restored to the pinned commit inside the build image before applying
the patch, so updating an already patched image is supported. Host files are not
reset. One compiler job is the default. A temporary candidate tag adds no full
image copy.

Inside a configured candidate container, validate against unpatched-writer
outputs replayed from the **same stored input**:

```bash
python3 /workspace/scripts/cli/validate_perigee_writer.py \
  --paired-root /pilot/paired \
  --reference /cache/perigee-writer-build/helper-full \
  --baseline-root /cache/perigee-recovery-v2/baseline \
  --output /cache/perigee-recovery-v2/root-validation
```

The baseline must come from the original unpatched image, using the same
`RootParticleReader` and input files. Stored float32 phase space is a replay,
so direct equality to the original float64 simulation run is not the appropriate
writer-only comparison. Every previously finite perigee component must match
this baseline exactly. Newly recovered selected generator particles are checked
against the saved independent Lorentz-integration reference at 0.001 mm d0 /
0.01 mm z0 tolerance. The two legacy finite discrepancies are reported separately,
not silently corrected or counted as newly validated recoveries.

After validation, update the existing image tag and remove the candidate tag:

```bash
docker tag colliderml-acts:recovery-candidate colliderml-acts:7cba36b17
docker image rm colliderml-acts:recovery-candidate
```

Start a **new container** to use the updated tag. Existing containers retain their
original runtime. Existing source exports and saved notebooks remain unchanged.
`--resume-export` does not recalculate perigees; use a fresh ACTS output directory.

## Run the saved Higgs-portal input from a new host shell

```bash
CLUSTER_ROOT=/home/atlas/dittmeier/git/cluster_extraction
CLUSTER_REPO="$CLUSTER_ROOT/ColliderML-Production"
CLUSTER_SCRATCH=/tmp/colliderml-paired-$(id -u)
docker run --rm -i --entrypoint /bin/bash \
  -v "$CLUSTER_REPO":/workspace:ro \
  -v "$CLUSTER_ROOT/data":/data:ro \
  -v "$CLUSTER_ROOT/exports/higgs-portal-pilot-run1-n1":/pilot:ro \
  -v "$CLUSTER_SCRATCH/cache":/cache \
  -v "$CLUSTER_SCRATCH/output":/output \
  -e COLLIDERML_CACHE=/cache -e SKIP_GENERATION_SETUP=1 -e SKIP_G4_DOWNLOAD=1 \
  colliderml-acts:7cba36b17 -c '
    set -e
    source /opt/acts-arrow/setup.sh
    source /workspace/scripts/cli/setup_container_env.sh
    source /opt/acts-arrow/setup.sh
    python3 /workspace/scripts/postprocessing/run_paired_samples.py \
      --events 1 --dataset higgs_portal --version v2_cells_pilot --run 1 \
      --input-pu0 /data/higgs_portal/v2/pu0/run0/edm4hep.root \
      --input-pu200 /pilot/simulation/pu200/edm4hep.root \
      --output /output/higgs-portal-pilot-run1-perigee-recovery
  '
```

The host output is
`/tmp/colliderml-paired-$(id -u)/output/higgs-portal-pilot-run1-perigee-recovery`.
Choose another new directory if it already exists. The normal paired command
still produces both pu0 and pu200, native cells and frozen pixel codes. No event
generation or Geant4 simulation is needed for these downloaded/saved inputs.

The superseded `writer-v1` boundary-correcting results remain historical artifacts
under `exports/higgs-portal-pilot-run1-n1/validation/perigee_writer/`. They describe
that older policy, not recovery-only v2. Its old smoke sample remains intact.

## Validated local result

The local `colliderml-acts:7cba36b17` tag now uses image
`sha256:81af69aad5b1b00ac54aa684d3b47f0acfdaf24279d610eaea1ae554097761b8`,
with patch checksum
`9423561225b2057cd349743baea30686428bfe9075b89c5c2fb342d153b50b0d`.
It shares the original base layers; the new layer is approximately 4.9 MB.

- All 71 relevant Python regression tests pass; the actual writer rejects
  three invalid recovery-limit configurations.
- The actual ROOT writer preserves every component of every legacy finite
  perigee exactly against the unpatched replay from identical inputs.
- A complete fresh paired run preserves all 1,412 pu0 and 126,359 pu200
  originally finite Parquet d0/z0 pairs exactly. This includes the 220 / 8,864
  secondary pairs affected by the superseded boundary correction.
- Selected generator-origin charged particles with tracker hits gain 20 pu0
  and 249 pu200 finite perigees, independently validated at the stated tolerances.
  Their maximum recovered d0 error is 0.00001297 mm and z0 error 0.00012846 mm.
- Selected finite coverage is now 124/125 at pu0 and 7,256/7,257 at pu200.
  Pu0 particle 713 (boundary-dependent) and pu200 particle 22453 (3.60 MeV/c
  pion) remain unresolved. The two known legacy finite pu200 mismatches,
  particles 7633 and 22398, are preserved intentionally.
- Hits, cells, memberships, native values, pixel codes and every non-perigee
  particle feature match the original paired tables exactly. No source files
  or saved notebook outputs were changed.

Fresh test products are at
`/tmp/colliderml-paired-40167/output/perigee-recovery-v2-smoke/{pu0,pu200}`.
Completion and comparison reports are `complete.json` and
`perigee-validation.json` in that output directory. Persistent validation reports
and replay scripts are under the original pilot's
`validation/perigee_recovery_v2/` directory.
