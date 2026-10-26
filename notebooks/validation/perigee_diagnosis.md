# Higgs-portal truth-perigee diagnosis

## Current policy: recovery only (writer-recovery-v2)

The current [writer patch](../../docker/acts-perigee/README.md) preserves every
successful legacy d0/z0 pair unchanged. Only originally failed/non-finite
results are retried. Retries are restricted to the pinned ODD central-field
cylinder (r < 1160 mm, |z| < 3000 mm), with finite-output and minimum checks.
No field-boundary splitting or field extension remains. Existing finite
outside-field secondary parameters are retained as legacy output, not newly
certified as accurate. The two finite pu200 boundary discrepancies are also
retained under the agreed recovery-only policy.

The helper regression recovers 20 selected pu0 and 249 selected pu200 perigees,
leaving pu0 particle 713 (boundary-dependent recovery) and pu200 particle 22453
(the 3.60 MeV/c pion) unresolved. These are generator-origin charged particles
with tracker hits; broader secondary perigee coverage is not a reference-accuracy
guarantee. See the patch README for the actual writer and paired-run checks.

## Superseded boundary-correcting experiment (writer-v1)

The first writer patch also recalculated finite results crossing z = ±3000 mm.
It recovered 270 selected perigees and corrected three boundary discrepancies,
but replaced 220 pu0 / 8,864 pu200 previously finite secondary pairs with NaNs.
That broader policy has been removed at the user's request. Its reports remain
historical artifacts in `validation/perigee_writer/` under the saved pilot.
They do not describe recovery-only v2. Original samples and notebook outputs
remain unchanged. The following sections document the original diagnosis.

## Independent validation follow-up

The independent reference is now implemented in `perigee_reference.py` and
`validate_perigee_reference.py`. It uses SciPy DOP853 to integrate Lorentz
motion in field values queried from the actual ODD provider. Its event roots
find local minima of transverse distance, with a positive second derivative.
It searches both directions and selects the minimum nearest to production in
absolute path; equal-distance ties use negative path. Searches have an explicit
finite bound and do not claim a global minimum over unlimited later turns.

The reference passes 20 exact uniform-field helix cases covering both charges,
2/4 Tesla, prompt, displaced, forward and curling trajectories. ACTS candidate
handling passes 16/20; the four extreme 5.83 MeV/c cases remain non-convergent.
All 7,382 charged generator particles with tracker hits in the two event samples
were evaluated. Candidate acceptance tolerances were fixed before inspection
at 0.001 mm for d0 and 0.01 mm for z0, with additional path-branch checks where
ACTS candidate path lengths are available. All recovered cases, the first ten
stored-success controls per sample, and stored mismatches receive tighter
integration/convergence checks. Tolerances were not loosened after failures.

| Cohort | Evaluated | Pass | Mismatch | No ACTS candidate |
|---|---:|---:|---:|---:|
| pu0 formerly NaN | 21 | 20 | 1 | 0 |
| pu200 formerly NaN | 250 | 249 | 0 | 1 |
| pu0 stored finite | 104 | 104 | 0 | 0 |
| pu200 stored finite | 7,007 | 7,005 | 2 | 0 |

The mismatches are pu0 particle 713 (candidate z0 error +0.8510 mm), pu200
particle 7633 (stored z0 error +0.1443 mm) and pu200 particle 22398 (stored z0
error -0.1915 mm). The last also has a d0 error of +0.00211 mm. All three cross
the sharp solenoid end boundary at |z|=3000 mm. Field queries confirm 2 Tesla
inside and zero outside in their radial range. An additional exact piecewise
helix/straight-line calculation, valid because these trajectories cannot reach
the radial return-field region, agrees with the independent numerical
reference within approximately 1e-6 mm. Thus these are not merely disagreements
between two unverified numerical methods.

The diagnostic finite-result count of 270 is **not** the number independently
validated: 269 pass, one fails, and the remaining 3.60 MeV/c pion has no ACTS
candidate. No existing source data have been corrected. All six source Parquet
checksums stayed unchanged; 7,109 of 7,111 stored finite perigees pass the stated
reference checks. These are truth-parameter numerical consistency results, not
detector resolution or tracking-efficiency measurements.

Persistent results are in
`exports/higgs-portal-pilot-run1-n1/validation/perigee_reference/`: `summary.json`,
`uniform_field_checks.csv`, `odd_field_comparisons.csv` and
`solenoid_boundary_checks.csv`. The notebook has appended tables, failure
listings and residual/survival plots; existing outputs were preserved.

To repeat inside the configured container, use a new output directory:

```bash
python3 /workspace/notebooks/validation/validate_perigee_reference.py \
  --paired-root /output/higgs-portal-pilot-run1-n1/paired \
  --probe-report /cache/perigee-probe/report.json \
  --output /output/higgs-portal-pilot-run1-n1/validation/perigee-reference-new
```

Production work must address both target convergence and integration across
field discontinuities, reject non-finite/incorrect-branch results, and retain
explicit failure information. Simply switching the target aborter is not a
complete fix. No Geant4 rerun is required for that particle-output work.

## Where values are lost

The 21 pu0 and 250 pu200 non-finite perigees among generator-origin charged
particles with tracker SimHits are not zero impact parameters or failed Parquet
matches. Every selected particle matches an ACTS ROOT record, and all these
records already contain NaN perigees. Geant4 positions, momenta, particles and
hits are available. The failing stage is the optional truth extrapolation in
the pinned ACTS `RootParticleWriter`, invoked by `add_output_writers` in
`scripts/simulation/digi_and_reco.py`. The reference perigee is at (0, 0, 0).

The chain is: EDM4hep truth -> ACTS particle conversion -> ROOT writer perigee
extrapolation -> particles ROOT/Parquet join -> notebook. The failure is in
the ROOT writer extrapolation, not Geant4, digitization, or the Parquet join.

## Runtime evidence

A standalone C++ diagnostic uses the same pinned ACTS and ODD field, with the
stored float32 particle phase space. It reproduced all 271 failures and gave
finite results for ten stored-success controls per pileup setting. This is
an independent replay from stored precision, not a trace of the original process.

The default `SurfaceReached` actor rejects intersections more than 0.1 mm behind
the propagation direction. On a curved trajectory, the straight-line step
estimate can overshoot the perigee by more than this. The required correction
is rejected, and propagation continues to the loop-protection path limit. The
final conversion to bound perigee coordinates fails because the endpoint is
not on the target surface. The writer then deliberately stores NaNs.

For pu0 particle 456, the trace shows an initial step of -45.4214 mm, followed
by a correction intersection at -0.710405 mm. The -0.1 mm near limit rejects it.
Propagation subsequently reaches its -3666.25 mm loop limit, and bound-state
creation returns `SurfaceError: position not on surface`. Forced-target
handling accepts the correction and converges.

| Diagnostic variant | Finite pu0 results / 21 | Finite pu200 results / 250 |
|---|---:|---:|
| Writer defaults | 0 | 0 |
| Forced target, loop protection retained | 20 | 248 |
| Plus tighter tolerance for the near-parallel case | 21 | 248 |
| Plus damped target correction for oscillation cases | 21 | 249 |

Both SympyStepper and EigenStepper agree on the finite forced-target/damped
results to numerical precision. Finite output alone is not an independent
validation of a correct closest-approach solution.

Three additional edge cases:

- **pu0 particle 713:** its direction has sin(theta)^2 = 7.16276e-5. The line
  intersection uses the default 1e-4 tolerance to reject a near-parallel track,
  yielding an invalid/infinite initial intersection. A 1e-6 tolerance yields
  finite parameters. Its production vertex is approximately
  (-58.03, -26.14, -1454.20) mm.
- **pu200 particle 33914:** forced target steps oscillate around the target
  and reach the 1000-step cap, also at 10000 steps. Halving the target correction
  resolves the oscillation in the diagnostic. Both steppers yield approximately
  d0=152.878744 mm and z0=134.225962 mm.
- **pu200 particle 22453:** a pion with pT approximately 3.60 MeV and |p| about
  204 MeV remains a convergence failure under the tested variants. It must
  retain a failure status; no zero or straight-line replacement is justified.

Changing the nominal stepper or merely increasing maxSteps did not fix the
271 default failures. Disabling loop protection is specifically **not a fix**:
many calls then report success with NaN endpoint parameters and unphysical
path lengths. The diagnostic explicitly distinguishes finite results from
successful return codes with non-finite parameters.

## Reproduction

The diagnostic is not imported by production. It does not modify sample files,
rebuild ACTS or run Geant4. Inside the existing pinned container, after the
normal environment setup and ODD library setup:

```bash
source /opt/acts-arrow/setup.sh
bash /workspace/scripts/cli/build_perigee_probe.sh
python3 /workspace/notebooks/validation/probe_perigee.py \
  --paired-root /output/higgs-portal-pilot-run1-n1/paired \
  --report /output/higgs-portal-pilot-run1-n1/perigee_diagnostic.json
```

The build helper uses Python's own sysconfig for headers/extension suffix;
the generic python3-config in this image can select Python 3.12 instead of
the ACTS runtime's Python 3.13. The compiled extension lives in the cache.
The ODD detector object must remain alive while its field is used.

The investigation report and verbose traces were preserved under
`exports/higgs-portal-pilot-run1-n1/validation/` on the host. Non-finite numbers
are JSON nulls in the saved report; finite-result flags/error messages retain
their meaning. These are analysis artifacts, not replacement physics data.

## Consequences

Do not fill missing perigees with zero or exclude these particles silently in
LRT studies. A first-parent-chain diagnostic links 13 affected pu0 particles
and six affected pu200 particles to LLPs; this is not a complete ancestry map.

A production correction should use robust target handling, explicit finite
result checks and per-particle failure diagnostics, with tests for prompt,
displaced, forward and low-pT trajectories and an independent closest-approach
reference. It should preserve currently correct results. This investigation
does not change the pinned production writer or existing ROOT/Parquet files.
Updating perigees would need only particle extrapolation/conversion, not a
new Geant4 simulation or a change to hit/cell memberships.

Pinned source references:
[RootParticleWriter](https://github.com/murnanedaniel/acts/blob/7cba36b173dd73a14342cc65043a43179a4b1dbd/Examples/Io/Root/src/RootParticleWriter.cpp),
[SurfaceReached](https://github.com/murnanedaniel/acts/blob/7cba36b173dd73a14342cc65043a43179a4b1dbd/Core/include/Acts/Propagator/StandardAborters.hpp),
[LineSurface](https://github.com/murnanedaniel/acts/blob/7cba36b173dd73a14342cc65043a43179a4b1dbd/Core/src/Surfaces/LineSurface.cpp).
