# CERN-compatible paired Parquet interface

Schema identifier: `paired-cern-compatible-v1`. This is a public export-interface change, not a simulation/reconstruction change. The inherited baseline is production commit `351ae166fa663d43b67b862fa95311bc7c0a4577`, with ACTS `7cba36b17`, ODD v4.0.4 and the existing geometric digitization JSON. The optional upstream ACTS-native per-SimHit export path is untouched; this product remains one **measurement** per hit-list entry.

## Layout and identity

- Hits: one row per `event_id`; **all other columns are aligned lists** ordered by event-local `measurement_id`.
- Particles: one row per `event_id`, with unchanged aligned particle-feature lists.
- Cells: one flat row per cell, with `event_id`, `measurement_id`, existing context/geometry and unchanged addresses/signals/codes.
- There are no on-disk `global_x/y/z` or `local_event` aliases. Dataset/campaign/version/run context for nested hits is in `colliderml.paired_context` Parquet metadata and the collection manifest. The reader injects that context into its in-memory flat frame.

Global IDs are unique **within one sample/dataset/version**. IDs are 0–99 for run3 and 100–199 for run4. Equal IDs across independent pu0/pu200 samples do not imply matched physics events. Future producers supply `--event-id-offset`; nonzero runs cannot silently default it. Never derive it from run × requested event count. Manifest ranges are half-open and must not overlap.

Metadata records `source_local_event = event_id - event_id_start`. Use that mapping for ROOT entry/CSV lookup. Particle, parent, measurement and SimHit indices stay unchanged; only the event identity is remapped. Particle IDs remain event-local.

## Meanings and provenance

`x/y/z`, geometry components, `true_x/y/z`, `time` and `detector` are copied from the existing production converter after strict row/count/geometry/position alignment. Geometry component numbers retain the current ACTS/ODD values; they are not remapped to numbers from an older CERN release. Float32 `x/y/z` match the ROOT-derived converter values; the previous CSV float64 representation may differ by round-trip precision, checked with the existing 0.0002 mm alignment tolerance and reported explicitly. No cluster position is recalculated.

`true_*` and hit `time` describe the legacy coordinate-matched SimHit, not every contributor to a merged measurement. Missing values remain NaN and unknown detector code remains 255. A complete contributor list can coexist with NaN legacy truth fields when a merged centroid has no single coordinate match. Use `particle_ids_complete`, not legacy NaNs, to identify incomplete contributor truth. Hit time comes from EDM4hep in ns; it is not a measured timing signal. Particle `time` is production time in ns. We do not reinterpret ACTS ROOT `tt` or `true_time` units to populate these fields.

`particle_id` retains the intentionally stricter convention: one known particle only if truth is complete; otherwise null. `converted_particle_id` is the inherited coordinate-matcher's diagnostic label. `particle_ids` contains every resolved distinct contributor; if `particle_ids_complete` is false the list is a lower bound. `unresolved_simhit_ids` retains the unresolved evidence. Multiple contributors and incomplete truth are distinct conditions. Do not turn an unresolved cluster into a noise/negative label.

Native `value` remains a path-length-based signal proxy in mm units. `pixel_code` remains the frozen P99.9/P99.5/P99 feature, with no refitting, new noise or suppression. Strips remain unencoded (null code). Local coordinates and geometry identifiers remain native ACTS information. There is no complete cell-level particle truth or reconstructed-track table.

Particle counts and perigee/vertex fields were supported by production before our branch. We retain the validated NaN-only perigee recovery and particle-preservation option; finite particle values are not recomputed by migration. `primary` includes pileup generator particles and is not a signal label. `num_tracker_hits` counts simulated hits, not measurements. `parent_id` is the first parent, not complete ancestry.

## Reading and joining

```python
from pathlib import Path
from scripts.postprocessing.utils import paired_schema as paired
import pyarrow.parquet as pq

root = Path('/mnt/data2/data/collider_ml/homebrew/higgs_portal_v2_cells_compat1')
paths, report = paired.resolve_sample(root, 'pu200', event=100)
hits = paired.read_event(paths['hits'], 100).to_pandas()
cells = paired.read_event(paths['cells'], 100).to_pandas()
key = paired.KEY  # context + event_id + measurement_id
joined = cells.merge(hits[key + ['particle_ids', 'particle_ids_complete']],
                     on=key, how='left', validate='many_to_one', indicator=True)
assert joined['_merge'].eq('both').all()
source_entry = paired.source_event(paths['hits'], 100)  # 0
```

For a conventional CERN-style loader, read one event row, remove scalar `event_id`, and create a DataFrame from the remaining lists. The shared reader accepts old flat paired files too and normalizes their names in memory.

## Migration and verification

`migrate_paired_schema.py` reads final source Parquets plus retained `_conversion_results` tracker hits. It never invokes generation, ACTS, Geant4, a truth matcher or a simulation-resume stage. It processes events/batches without collecting the full dataset (PyArrow may buffer larger input row groups), validates every copied value, rereads outputs, and rechecks all input hashes. Original reports are copied verbatim under each provenance directory's `source/`; new reports state that they are migration reports. Their inherited `inputs`, `revisions` and `source_sha256` trace the original producer; `migration_validation.json` records the migration inputs/code separately. Exact executed migration code is retained under `provenance/migration_code/`.

The current collection is not overwritten. Canonical raw outputs are retained under `production/output/higgs-portal-parquet-compat1`; the separate collection is `homebrew/higgs_portal_v2_cells_compat1`. Product version is `v2_cells_compat1`; source physics remains v2. All 12 files and per-run event ranges are recorded in the manifest. The backup uses `parquet_source_relative` to archive new raw Parquets once and link their published names remotely. Changed schemas are genuinely new file content; incremental backup cannot skip those new bytes.

First test with fresh trial paths and `--limit-events 1`, then run the full migration:

```bash
cd /home/atlas/dittmeier/git/cluster_extraction/ColliderML-Production
~/.conda/envs/paired-clusters/bin/python scripts/postprocessing/migrate_paired_schema.py \
  --work-root /mnt/data2/data/collider_ml/production/output/higgs-portal-parquet-compat1 \
  --output /mnt/data2/data/collider_ml/production/output/higgs-portal-v2-cells-compat1-release
```

Existing destinations are refused. Install the validated release into the new homebrew directory only after it passes; never rerun the command into completed work. The new backup selection is:

```bash
CLUSTER_PUBLISHED=/mnt/data2/data/collider_ml/homebrew/higgs_portal_v2_cells_compat1 \
  bash ~/backup-colliderml.sh
```

The old backup selection continues to work; no backup or production is started automatically. For the notebook, use the new collection, `EVENT=100`, and `PREVIEW_EVENTS=(0,100,199)`; restart the kernel. Saved outputs are historical and intentionally preserved.

## Complete storage types

### Hits

Only `event_id` is scalar. Two list levels in contributor fields mean event → measurement → contributing IDs.

| Column | Arrow type |
|---|---|
| `event_id` | `uint32` |
| `x` | `list<item: float>` |
| `y` | `list<item: float>` |
| `z` | `list<item: float>` |
| `volume_id` | `list<item: uint8>` |
| `layer_id` | `list<item: uint16>` |
| `surface_id` | `list<item: uint32>` |
| `true_x` | `list<item: float>` |
| `true_y` | `list<item: float>` |
| `true_z` | `list<item: float>` |
| `time` | `list<item: float>` |
| `detector` | `list<item: uint8>` |
| `measurement_id` | `list<item: uint64>` |
| `geometry_id` | `list<item: uint64>` |
| `local0` | `list<item: double>` |
| `local1` | `list<item: double>` |
| `particle_id` | `list<item: uint64>` |
| `converted_particle_id` | `list<item: uint64>` |
| `particle_ids` | `list<item: list<item: uint64>>` |
| `particle_ids_complete` | `list<item: bool>` |
| `unresolved_simhit_ids` | `list<item: list<item: uint64>>` |
| `simhit_ids` | `list<item: list<item: uint64>>` |

### Particles

All existing fields and values are retained, with only event IDs changed. Positions/perigees are mm, momentum/mass/energy GeV, time ns, and charge elementary-charge units.

| Column | Arrow type |
|---|---|
| `event_id` | `uint32` |
| `particle_id` | `list<item: uint64>` |
| `pdg_id` | `list<item: int64>` |
| `mass` | `list<item: float>` |
| `energy` | `list<item: float>` |
| `charge` | `list<item: float>` |
| `vx` | `list<item: float>` |
| `vy` | `list<item: float>` |
| `vz` | `list<item: float>` |
| `time` | `list<item: float>` |
| `px` | `list<item: float>` |
| `py` | `list<item: float>` |
| `pz` | `list<item: float>` |
| `perigee_d0` | `list<item: float>` |
| `perigee_z0` | `list<item: float>` |
| `num_tracker_hits` | `list<item: uint16>` |
| `num_calo_hits` | `list<item: uint16>` |
| `primary` | `list<item: bool>` |
| `vertex_primary` | `list<item: uint16>` |
| `parent_id` | `list<item: int64>` |

### Cells

| Column | Arrow type / meaning |
|---|---|
| `campaign`, `dataset`, `version` | string context |
| `run`, `event_id` | uint32 |
| `measurement_id`, `geometry_id` | uint64 |
| `channel0`, `channel1` | int32 channel addresses |
| `value` | float64 native activation |
| `pixel_code` | nullable uint8, codes 1–15 for encoded pixels |
