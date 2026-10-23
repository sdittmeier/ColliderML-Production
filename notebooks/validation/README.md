# Paired vs published event notebook

`paired_vs_published.ipynb` compares one new ttbar event with one independent
published event. It defaults to new pu0 event 25 (held out from empirical
calibration) and published event 0. Change `SAMPLE` to
`'pu200'` in the first code cell for pileup 200, or edit the event IDs and
paths there. The completed 50-event run contains events 0-49 per sample.

The notebook needs Python with `numpy`, `pyarrow`, `pandas`, `matplotlib`, `uproot`, `scipy`,
`ipykernel`, and `jupyterlab`. A separate regular conda environment keeps these
plotting dependencies out of the ACTS runtime:

```bash
conda create -n paired-validation -c conda-forge \
  python=3.10 numpy pyarrow pandas matplotlib uproot scipy ipykernel jupyterlab
conda activate paired-validation
cd /home/atlas/dittmeier/git/cluster_extraction/ColliderML-Production
python -m jupyter lab notebooks/validation/paired_vs_published.ipynb
```

If `paired-validation` already exists, install any missing packages into it
instead of creating it again. The default new-data path points to persistent
storage at `/home/atlas/dittmeier/git/cluster_extraction/exports/paired-samples-features-n50`;
adjust it if your completed output was copied elsewhere.

Blue always represents the new paired sample and orange the published release.
The two events are independent, so compare distributions and detector coverage,
not equality of individual values. All displayed histograms share bins and are
normalized within each source. Spatial views sample at most 10,000 hits per
source for readability; integrity counts use the full selected events.
Panels with a dominant bin and sparse tails use a logarithmic y-axis; the
feature values and shared x-bins remain unchanged.

The cluster-image section shows six distinct sizes per detector type: pixels
(volumes 16-18), short strips (23-25), and long strips (28-30). Set
`CLUSTER_TARGET_SIZES` and `CLUSTER_COLOR_SCALE` (`'linear'` or `'log'`) in the
configuration cell. Each detector gallery shares a color scale. Images use
actual channel addresses and mask unoccupied cells; square image pixels do not
imply square physical cells. The accompanying table reports actual sizes and
activation sums, and the full-range value table exposes histogram tails.

Cell activation is the geometric path-length proxy after configured smearing,
thresholding, and merging, without an upper limit of 0.5. The notebook includes
the precise settings and links to the pinned ACTS implementation. Restart the
kernel and run all cells after updating the companion plotting module.

The pileup/contributor section compares **only `NEW_EVENT`** from each of pu0
and pu200, independently of `SAMPLE`. It joins cells on the full measurement
key and reports cell-level and cluster-level tail fractions separately.
`CELL_TAIL_THRESHOLD` defaults to 0.5 and is diagnostic only. Incomplete truth
is a separate cohort; the displayed particle count always belongs to the
parent cluster and cannot establish the contributors to an individual cell.
Pu0 is blue and pu200 green in these new plots. Full-range survival plots and
tables expose extremes beyond the original percentile-limited histograms.

The ATLAS pixel section reads `ATLAS_FILE`, by default event 193 in the supplied
feature store. `ATLAS_PYTHON` defaults to the existing
`/local/anaconda3/envs/acorn_seb/bin/python`, which has PyTorch and PyG. The
reader runs in that interpreter and returns arrays in memory, so the notebook
kernel need not install PyTorch. Adjust this interpreter path on another host.
The selected file is a trusted local PyG graph loaded using `torch.load`.

Pixel regions are identified using the graph's configuration. Empty pixel
images remain counted, while strip images are excluded. Shape overlays use
ATLAS nonzero image entries (orange) against the complete ColliderML pixel
cell lists (blue/green). The ATLAS image is an 11x11 window: edge contact is
reported, and lost cells or active zero-code pixels cannot be reconstructed.
Native channel axes are labeled without asserting alignment across detectors.
Amplitude panels keep ColliderML path-length units and ATLAS ToT codes
separate; code 15 is counted without assuming an electronics saturation model.
All derived quantities stay in memory. No Parquet production, source dataset,
quantization, or stored cluster position is changed by those sections.

## Pixel quantization sensitivity

The final notebook section adds in-memory, occupancy-preserving pixel mappings.
No source Parquet or stored position is rewritten, and no cells are removed or
clusters rebuilt. Existing notebook outputs are retained; execute the new cells
after restarting the kernel to load the updated helper module.

`ATLAS_REFERENCE_DIR` defaults to the feature store's `valset` directory.
`ATLAS_REFERENCE_COUNT=20` chooses evenly spaced sorted files; alternating files
form calibration and held-out cohorts. Set `ATLAS_REFERENCE_FILES` for an explicit
ordered list instead. The selected event-193 file is excluded. Files are read one
at a time through `ATLAS_PYTHON`; only event summaries remain in memory. Tables
include filenames, pooled/per-event positive-code percentiles, code-15 cell and
cluster fractions, and whole-event bootstrap intervals (1,000 resamples, seed 7).
Windowed, zero-suppressed images cannot identify active zero-code cells or the
unrecorded below-threshold population. Bootstrap intervals describe these
selected events and may be zero-width when no rare codes are observed.

Some reference events contain native code 16. Following the agreed overflow
policy, only the empirical fit/comparison histogram folds native codes >=15
into bin 15. Native histograms, maximum codes, above-15 counts, exact-code-15
fractions, and >=15 overflow fractions are reported separately. No native
values are changed, and native percentiles are calculated before folding.

The empirical mapping is calibrated on pu200 events 0-24 and frozen for both
pileup settings. `NEW_EVENT` controls single-event plots (default 25).
The Athena-inspired gain reference remains pu200 event 0, keeping that existing
exploratory comparison separate from the empirical campaign study.
`SHOW_CONTINUOUS_DIAGNOSTICS=False` hides continuous-response distributions and
draws integer Athena response steps by default. Continuous weights remain
internal diagnostics for separating response changes from integer truncation.
The T12/T13/T13.5 gain choices need not populate low codes: their responses at
the retained minimum 0.01 are approximately 3.34/6.25/8.99, respectively.

- **Athena-inspired:** `Q=gain*activation`, `T=14*(Q-threshold)/(Q+C)` without
  output renormalization. The default parameters are 600/4500 electrons, with
  900/4000 exposed as an alternative preset, not an ODD layer assignment.
  `ATHENA_TARGET_TOTS=(12,13,13.5)` fixes gains via the calibration event's 99.5th
  activation percentile. Gains that would map the ACTS retained minimum 0.01
  below ToT 1 are rejected. Integer conversion truncates to codes 1-14. The
  extra ToT smearing is **off by default** (`ADD_TOT_SMEARING=False`), because
  native ACTS activations already contain smearing. Opting in adds a separately
  seeded variation with sigma `0.4+0.02*T`; tables count values needing
  the lower floor or upper clamp after truncation. Clamping smeared codes to 1
  is an explicit occupancy-preserving adaptation, not exact Athena digitization.
- **Empirical:** activation quantiles match the calibration ATLAS frequencies
  of codes 1-15. Boundary ties go to the upper bin without random splitting.
  Held-out frequency deviations expose ties and independent-event variation.
  This is distribution matching, not electronics calibration.

## Run the 50-event empirical study

Restart the notebook kernel and run all cells, then save the notebook. Existing
outputs have been preserved but are historical until rerun; do not interpret
them as results of the new configuration. No ACTS/Docker run is needed.

- `EMPIRICAL_CALIBRATION_EVENTS=tuple(range(25))` fits only pu200 events 0-24.
  `EMPIRICAL_HELD_OUT_EVENTS=tuple(range(25, 50))` must be disjoint. Both cohorts
  are evaluated at pu0 and pu200 using the same final boundaries.
- `EMPIRICAL_CHECKPOINTS=(1,2,5,10,25)` exposes boundary convergence and held-out
  frequency deviations as calibration grows. The final 25-event mapping is
  selected in advance; held-out results do not automatically select or refit it.
- Calibration pools exact, count-compressed activation populations sequentially
  and uses linear-interpolated quantiles, equivalent to concatenating raw
  calibration activations. It does not retain event cell tables. Validation
  reads pixels from one event at a time and retains only code histograms,
  event summaries, and cluster-level displacement vectors for exact pooled
  quantiles. No all-50-event cell DataFrame is constructed. Progress is printed
  for each event; this section may take several minutes.
- Pooled and per-event tables distinguish cell code-15 fractions from fractions
  of clusters containing code 15. Frequency differences are compared against
  independent held-out ATLAS events. Total variation is half the sum of absolute
  code-frequency deviations, not a position-reconstruction metric.
- Displacements include median, 95th/99th percentile and maximum, exact cluster
  sizes, and complete/incomplete contributor groups. Pooled statistics weight
  individual clusters, not event quantiles. Full-range survival plots display
  at most 2,000 exact ECDF support points including extrema; statistics use all
  clusters. Largest held-out examples are reloaded deterministically and show
  identical channel addresses before/after mapping.
- The final check compares SHA256 checksums of the six source Parquets. All
  mappings and derived results remain in memory. Save the notebook's outputs
  for review; no source dataset or production code is modified.

For review, the key outputs are `empirical_calibration.stability`,
`empirical_validation.convergence`, `.pooled`, `.code_comparison`, and pooled
held-out rows of `.shifts`. Blue is pu0, green pu200, orange ATLAS. Incomplete
truth remains separate; contributor counts refer to parent clusters, not cells.

## Linear Mapping And Truth Residuals

The appended **Linear codes with ATLAS-targeted overflow and actual truth
residuals** section compares native, empirical, and linear-code pixel positions.
It can run using the existing kernel after empirical calibration completes:
reload the notebook from disk to see the appended cells, without overwriting
them from an older open browser copy. The first appended code cell reloads the
helper module. No need to rerun the previous 50-event campaign if its variables
are still in memory. After a kernel restart, run the preceding setup/calibration
cells first (the empirical campaign validation itself is not a prerequisite).

Linear codes use 14 equal-width activation bins from 0.01 to the calibration
overflow edge, with code 15 at or above that edge. The edge is the empirical
14-to-15 boundary; it uses the ATLAS **calibration** overflow fraction, not
99.5% or a held-out target. Both mappings therefore have identical overflow
membership, although their interior code frequencies differ. The calibration
table reports the target fraction/percentile and bin width. Every retained
cell remains positive; ties go to the upper bin. This is distribution-guided
quantization, not an ATLAS electronics simulation.

`RESIDUAL_ACTS_ROOT` points to the scratch run containing each sample's
`acts/measurements.root`. These ROOT files were not included in the persistent
Parquet copy; preserve them to rerun truth comparisons. `RESIDUAL_DIGI_CONFIG`
selects the geometric digitization JSON used in production. The reader indexes
event spans once per ROOT file, preserves original per-event measurement row
positions, and loads one event at a time. It requires `uproot` in the notebook
environment, without ACTS, Docker, or a geometry factory installation.

`RESIDUAL_EVENTS` defaults to held-out events 25-49 at both pileup settings;
use `(NEW_EVENT,)` for a shorter held-out pilot. Calibration events are rejected.
The residual subset requires complete truth, exactly one SimHit, and finite
`true_loc0/1` and stored residuals. Multi-SimHit single-particle clusters are
excluded too. Coverage tables report all exclusions. Native activation-weighted
channel centers are converted to millimetres using JSON bin centers and pitch,
then verified against both Parquet local positions and ROOT reconstructed local
positions before subtraction. ROOT residuals verify the reconstructed-minus-truth
sign convention. Broken event/row/geometry alignment fails the analysis.

`pixel_residual_study.summary` contains per-event and pooled residual metrics in
**micrometres**, grouped by coordinate, method, pixel volume, and exact cluster
size. Bias is signed; RMS is about zero; standard deviation is about the mean;
central-68% half-width is `(q84-q16)/2`, not a Gaussian fit. Absolute 95th/99th
percentiles and extrema expose tails. The default signed histogram range is
-100 to +100 micrometres; density uses the entire selected population, with
outside-range counts shown separately. Full-range absolute-residual survival
curves include extrema (at most 2,000 exact ECDF support points for display).
Sparse size groups can have zero width; inspect their cluster counts.

Code-frequency comparisons use **all pixel cells** in the selected held-out
events, not only the truth-residual subset. Separate cell and cluster overflow
fractions are retained. All native positions and cluster memberships stay
unchanged; code-weighted positions are analysis-only, without charge decoding
or reclustering. The final cell verifies the six Parquet source checksums.

The Athena source is pinned to
[`46f1900866d629c4241b75f4f178d3038abd5b1c`](https://gitlab.cern.ch/atlas/athena/-/tree/46f1900866d629c4241b75f4f178d3038abd5b1c):
[ITk calibration defaults](https://gitlab.cern.ch/atlas/athena/-/blob/46f1900866d629c4241b75f4f178d3038abd5b1c/InnerDetector/InDetConditions/PixelConditionsAlgorithms/src/ITkPixChargeCalibAlg.cxx),
[RD53 digitizer](https://gitlab.cern.ch/atlas/athena/-/blob/46f1900866d629c4241b75f4f178d3038abd5b1c/InnerDetector/InDetDigitization/PixelDigitization/src/RD53SimTool.cxx),
and [code conversion](https://gitlab.cern.ch/atlas/athena/-/blob/46f1900866d629c4241b75f4f178d3038abd5b1c/InnerDetector/InDetDigitization/PixelDigitization/src/PixelNoiseFunctions.cxx).
Gain is exploratory, not a calibrated charge yield. Sensor response, threshold
fluctuations, and timing are not reproduced; current Athena's maximum code 14
must not be confused with code 15 observed in the historical ATLAS sample.

Centroids use channel addresses weighted directly by native activations,
continuous response, or codes, with no decoding. Displacements are in channel
units, not micrometres or reconstructed-track residuals. Tables include exact
cluster sizes and parent-contributor groups; incomplete truth stays separate.
Full-range survival curves and largest-shift images expose tails. Choose
`QUANTIZATION_EXAMPLE_VARIANT`, e.g. `empirical` or `athena_T13_integer`, to inspect
different mappings. The last cell verifies SHA256 checksums of all six source
Parquets. These are sensitivity tests, not Athena cluster-position reconstruction.

## Linear Overflow-Threshold Scan

The final **Linear overflow-threshold scan: changes from native positions**
section adds three lower overflow edges, fitted at pooled calibration activation
percentiles 99.9, 99.5 and 99. These nominally overflow 0.1%, 0.5% and 1% of
calibration cells. They deliberately do not match the much rarer ATLAS overflow.
The current ATLAS-targeted linear mapping and empirical mapping remain baselines.
All variants retain 14 regular bins from 0.01 and code 15 as overflow, without
new noise, suppression, decoding, or changes to stored data.

Reload the updated notebook from disk, keeping the existing kernel, and execute
only this appended section. It reloads the helper and reuses
`empirical_calibration` and `atlas_code_reference`. No ROOT files, ACTS runtime,
surface projection, prior campaign validation, or truth-residual run is needed.
After restarting the kernel, run preceding setup and empirical calibration first.

`LINEAR_SCAN_PERCENTILES=(99.9,99.5,99.0)` controls the scan. Exact pooled activation
counts are read once from the same pu200 calibration events (normally 0-24),
not three times. Edges freeze for `LINEAR_SCAN_EVENTS`, normally 25-49 at both
pileup settings; use `(NEW_EVENT,)` for a held-out pilot. Overlapping event
cohorts are rejected. Calibration tables show target and achieved overflow
fractions; ties always enter the upper bin and are not split randomly.

`linear_scan_study.summary` compares Euclidean centroid displacement from native
and signed coordinate changes in **channel units**, not truth resolution.
It includes fractions moving strictly more than 0.1, 0.5 and 1 channel,
cluster-size and contributor groups, plus overflow/non-overflow clusters.
Overflow-group membership is variant-specific. `.overflow` separates cell and
cluster denominators and shows the fraction of cells coded 1 or 2. All pixel
clusters are included, including merged and incomplete truth.

Plots overlay variants using line styles, with pu0 blue, pu200 green and ATLAS
orange. Signed histograms report clipped-tail counts; full-range survival plots
display up to 2,000 exact ECDF support points, with all observations used for
statistics. Largest-shift images are deterministic per variant and compare
identical cell addresses against native weights. The final source-checksum
assertion confirms all six Parquets are unchanged. The unresolved truth-surface
reference issue does not affect these native-to-quantized shift comparisons.

## Gaussian Fits To Residual Distributions

The appended **Gaussian fits to cluster-position residual distributions** section
fits a Gaussian to position residuals and reports its **sigma as a core-width
resolution estimate**, plus the fitted mean as bias. It does not fit positions,
tracks or quantization mappings. Native, empirical and all four linear variants
share the same held-out, complete, single-SimHit cluster selection. Reload the
notebook from disk, keep the kernel, and run this new section; it reuses the
frozen empirical and linear-scan calibrations. A new residual pass is needed for
the three percentile-linear cases, but not another calibration or production run.

`RESOLUTION_FIT_WINDOW_UM=30` fixes a common `[-30,30)` micrometre range before
comparing methods. The unbinned likelihood includes finite-window normalization;
the fitted sigma is not just a trimmed sample standard deviation. Per-event
sufficient statistics avoid retaining cell frames for the fits. Fit failures,
insufficient entries and parameter-boundary solutions are flagged. Tables also
retain core coverage, full-range robust metrics and a Pearson chi-square-per-ndof
diagnostic; no Gaussian assumption is made for the tails. A single Gaussian may
describe mixed cluster populations poorly, so inspect the actual data/fit panels.

`RESOLUTION_BOOTSTRAP_RESAMPLES=200`, seed 7, resamples whole events with identical
draws across methods. Pooled sigma and sigma/native-sigma ratios have paired
95% bootstrap intervals. These intervals do not include truth-reference
systematics or Gaussian-model inadequacy. Volume fits have point estimates only;
their uncertainty columns are empty. Set resamples to 0 for a pilot, and
`RESOLUTION_EVENTS=(25,49)` to check two events before the full held-out scan.

**Do not interpret these provisional ROOT-reference core widths as validated
detector resolution yet.** The pinned ACTS
[truth helper](https://github.com/murnanedaniel/acts/blob/7cba36b173dd73a14342cc65043a43179a4b1dbd/Examples/Framework/include/ActsExamples/EventData/AverageSimHits.hpp)
transforms the recorded SimHit position with thickness tolerance, without an
explicit plane intersection. Earlier along-track outliers motivate a separate
surface-reference check. This section does not implement geometry fitting,
surface projection, or a correction to truth or reconstructed positions.

Fit plots are small multiples: one method per panel, colored residual data,
black dashed fitted Gaussian, shaded fit range and numeric sigma. Method colors
are consistent across plots, with distinct markers as a second cue. Other
residual/shift comparisons split pu0 and pu200 instead of using similar colors
for every method. The default detailed fit panels show pu200; add pu0 to
`RESOLUTION_PLOT_SAMPLES` for its panels. Sigma comparison plots include both.
Rerun old plotting cells after helper reload to refresh their saved figures;
all existing notebook outputs are preserved until rerun.

SciPy is needed only for fitting. It was installed as `scipy==1.15.3` in the
existing `paired-clusters` notebook environment, without upgrading NumPy or
changing the ACTS runtime. On another existing notebook environment, install
SciPy there before running the fit cells.

## Stored Pixel-Code Validation

After producing and copying `exports/paired-samples-p99_9-n50`, reload the notebook
from disk and find **Validate stored pixel codes against the native export** at
the end. Run only its two code cells, even in a fresh `paired-clusters` kernel.
There is no need to rerun the previous calibration/ATLAS/resolution sections.

`STORED_CODE_ROOT` selects the new export; `NATIVE_CODE_REFERENCE_ROOT` selects the
original `paired-samples-features-n50`. `STORED_CODE_SCHEME` defaults to `p99.9`.
This validation checks all events and both samples, not just `NEW_EVENT`.

Bounded-memory batches check all original cell columns row for row, code bounds,
exact frozen boundary/tie behavior, null strip/other codes, counts and metadata.
Hit and particle files must be byte-identical to the reference. All twelve source
Parquets are hashed before and after. It does not write or refit any dataset.
The section displays per-sample checks, a full code histogram and shared-axis
pu0/pu200 frequency overlays in both linear and logarithmic views. Files with
different row-group sizes are compared with aligned row slices.

For collaborator discussion, keep an information-preserving native cell product
separate from optional quantized views. Do not overwrite raw activations,
membership, native positions or truth links based solely on a core fit. Any
derived quantizer should identify its calibration population, frozen boundaries,
overflow convention and version. Resolution/reference validation and downstream
physics/training tests remain prerequisites for stronger performance claims.

Run the focused checks in the existing notebook environment:

```bash
conda run -n paired-clusters python -m unittest discover -s tests -p 'test_*cell*.py'
conda run -n paired-clusters python -m unittest discover -s tests -p 'test_pixel_quantization.py'
conda run -n paired-clusters python -m unittest discover -s tests -p 'test_pixel_residuals.py'
conda run -n paired-clusters python -m unittest discover -s tests -p 'test_linear_threshold_scan.py'
conda run -n paired-clusters python -m unittest discover -s tests -p 'test_resolution_fits.py'
```
