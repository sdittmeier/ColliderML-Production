"""Read-only Higgs-portal notebook calculations and configuration."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from notebooks.validation import higgs_portal_validation as validation


class HiggsPortalNotebookTest(unittest.TestCase):
    def test_generator_cohort_excludes_material_secondaries(self):
        frame = pd.DataFrame(dict(charge=[1, 1, 0, 1], num_tracker_hits=[1, 2, 3, 0],
                                  primary=[True, False, True, True], production_r=[20, 100, 0, 0]))
        self.assertEqual(len(validation.particle_selection(frame, "generator_charged_with_hits")), 1)
        self.assertEqual(len(validation.particle_selection(frame, "charged_with_hits")), 2)
        self.assertEqual(len(validation.particle_selection(frame, "displaced_charged_with_hits")), 2)

    def test_integrity_detects_missing_particle_and_duplicate_measurement(self):
        key = dict(campaign="hard_scatter", dataset="higgs_portal", version="test", run=1,
                   local_event=0, measurement_id=0)
        hits = pa.Table.from_pylist([{**key, "particle_id": 10, "converted_particle_id": 10,
                                     "particle_ids": [10], "simhit_ids": [7],
                                     "unresolved_simhit_ids": [], "particle_ids_complete": True}])
        report = dict(campaign="hard_scatter", dataset="higgs_portal", version="test", run=1,
                      events=1, measurements=1, cells=1, particles=1, simhit_links=1,
                      null_particle_labels=0, multi_particle_measurements=0,
                      unresolved_simhit_links=0, incomplete_truth_measurements=0,
                      converted_label_disagreements=0)
        sample = SimpleNamespace(hits=hits, particles=pd.DataFrame(dict(particle_id=[10], parent_id=[-1])),
                                 cells=SimpleNamespace(num_rows=1), report=report, event=0)
        self.assertEqual(validation.integrity({"pu0": sample}).loc["pu0", "association_checks"], "PASS")
        incomplete = sample.hits.to_pylist()[0]
        incomplete.update(particle_id=None, particle_ids=[], particle_ids_complete=False,
                          unresolved_simhit_ids=[7])
        incomplete_sample = SimpleNamespace(**{**sample.__dict__, "hits": pa.Table.from_pylist([incomplete]),
            "report": dict(report, null_particle_labels=1, unresolved_simhit_links=1,
                           incomplete_truth_measurements=1)})
        incomplete_check = validation.integrity({"pu0": incomplete_sample})
        self.assertEqual(incomplete_check.loc["pu0", "converted_label_disagreements"], 0)
        sample.particles.loc[0, "particle_id"] = 11
        with self.assertRaisesRegex(ValueError, "missing contributor"):
            validation.integrity({"pu0": sample})
        sample.hits = pa.concat_tables([hits, hits])
        with self.assertRaisesRegex(ValueError, "duplicate full measurement"):
            validation.integrity({"pu0": sample})

    def test_particle_alignment_and_zero_pt(self):
        row = dict(event_id=0, particle_id=[1, 2], px=[0., 3.], py=[0., 4.], pz=[0., 5.],
                   vx=[0., 6.], vy=[0., 8.])
        frame = validation.particle_frame(row)
        np.testing.assert_array_equal(frame.pt.to_numpy(), [0, 5])
        np.testing.assert_array_equal(frame.production_r.to_numpy(), [0, 10])
        self.assertTrue(np.isnan(frame.eta.iloc[0]))
        row["particle_id"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "unique"):
            validation.particle_frame(row)
        row["px"] = [1]
        with self.assertRaisesRegex(ValueError, "aligned"):
            validation.particle_frame(row)

    def sample(self):
        encoded = dict(pixel_volumes=[16, 17, 18], activation_min=.01,
                       boundaries=list(np.arange(2, 16) / 100), code15_bin_end=.16,
                       code15_cells=1, genuinely_clipped_cells=1, scheme="p99.9")
        cells = pd.DataFrame(dict(volume_id=[16, 18, 23], pixel_code=[2., 15., np.nan],
                                  geometry_id=np.array([16, 18, 23], dtype=np.uint64) << 56,
                                  value=[.02, .18, .1]))
        return SimpleNamespace(report=dict(events=1, cell_quantization=encoded),
                               paths=dict(cells=Path("cells.parquet")), cells=pa.Table.from_pandas(cells),
                               analysis=SimpleNamespace(cells=cells))

    def test_stored_codes_and_upper_bin_ties(self):
        sample = self.sample()
        schema = SimpleNamespace(metadata={b"colliderml.cell_quantization":
                                           json.dumps(sample.report["cell_quantization"]).encode()})
        with patch.object(validation.pq, "read_schema", return_value=schema):
            report = validation.quantization({"pu0": sample})
            self.assertEqual(report.loc["pu0", "pixel_cells"], 2)
            self.assertEqual(report.loc["pu0", "code15_fraction"], .5)
            sample.analysis.cells.loc[0, "pixel_code"] = 1
            sample.cells = pa.Table.from_pandas(sample.analysis.cells)
            with self.assertRaisesRegex(ValueError, "wrong pixel codes"):
                validation.quantization({"pu0": sample})
            sample.analysis.cells.loc[0, "pixel_code"] = 2
            sample.analysis.cells.loc[2, "pixel_code"] = 1
            sample.cells = pa.Table.from_pandas(sample.analysis.cells)
            with self.assertRaisesRegex(ValueError, "encoded strip cells"):
                validation.quantization({"pu0": sample})

    def test_notebook_code_compiles(self):
        path = Path(validation.__file__).with_name("higgs_portal_pilot.ipynb")
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")

    def shard_fixture(self, root, sample="pu0"):
        folder = root / sample
        folder.mkdir()
        encoded = dict(pixel_volumes=[16, 17, 18], activation_min=.01,
                       boundaries=list(np.arange(2, 16) / 100), code15_bin_end=.16,
                       code15_cells=0, genuinely_clipped_cells=0, scheme="p99.9")
        hits, cells, particles = [], [], []
        for event in range(2):
            key = dict(campaign="hard_scatter" if sample == "pu0" else "full_pileup",
                       dataset="higgs_portal", version="test", run=3, local_event=event, measurement_id=0)
            geometry = 16 << 56
            hits.append(dict(**key, geometry_id=geometry, particle_id=10, converted_particle_id=10,
                             particle_ids=[10], simhit_ids=[7], unresolved_simhit_ids=[], particle_ids_complete=True))
            cells.append(dict(**key, geometry_id=geometry, channel0=0, channel1=0, value=.02, pixel_code=2))
            particles.append(dict(event_id=event, particle_id=[10], parent_id=[-1], pdg_id=[211],
                                  px=[1.], py=[0.], pz=[.1], vx=[20.], vy=[0.], vz=[0.],
                                  primary=[True], charge=[1.], num_tracker_hits=[1], perigee_d0=[.2], perigee_z0=[.1]))
        for kind, rows in (("hits", hits), ("cells", cells), ("particles", particles)):
            table = pa.Table.from_pylist(rows)
            if kind == "cells":
                table = table.replace_schema_metadata({b"colliderml.cell_quantization": json.dumps(encoded).encode()})
            pq.write_table(table, folder / f"{kind}.parquet", row_group_size=1)
        report = dict(campaign=hits[0]["campaign"], dataset="higgs_portal", version="test", run=3,
                      events=2, measurements=2, cells=2, particles=2, simhit_links=2,
                      null_particle_labels=0, multi_particle_measurements=0, unresolved_simhit_links=0,
                      incomplete_truth_measurements=0, converted_label_disagreements=0, cell_quantization=encoded)
        (folder / "report.json").write_text(json.dumps(report))
        return folder

    def test_full_scan_and_event_boundary_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sample in ("pu0", "pu200"):
                folder = self.shard_fixture(root, sample)
                events = list(validation.iter_flat_events(folder / "cells.parquet", batch_size=1))
                self.assertEqual([event for event, _ in events], [0, 1])
            before = validation.file_hashes(root)
            scan = validation.scan_campaign(root, progress=False)
            self.assertEqual(before, validation.file_hashes(root))
            self.assertEqual(len(scan["counts"]), 4)
            self.assertEqual(scan["counts"].cells.sum(), 4)
            self.assertTrue(scan["issues"].empty)
            self.assertEqual(scan["perigees"].nonfinite.sum(), 0)
            report_path = root / "pu200/report.json"
            report = json.loads(report_path.read_text())
            report["measurements"] = 3
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "all-event measurements"):
                validation.scan_campaign(root, progress=False)

    def test_full_key_cell_join_rejects_cross_event_geometry_and_missing_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.shard_fixture(Path(temporary))
            hits = pq.read_table(folder / "hits.parquet")
            cells = pq.read_table(folder / "cells.parquet")
            validation.validate_cell_links(hits, cells)
            broken = cells.to_pylist()
            broken[0]["local_event"] = 9
            with self.assertRaisesRegex(ValueError, "Orphan cells"):
                validation.validate_cell_links(hits, pa.Table.from_pylist(broken))
            broken = cells.to_pylist()
            broken[0]["geometry_id"] += 1
            with self.assertRaisesRegex(ValueError, "geometry"):
                validation.validate_cell_links(hits, pa.Table.from_pylist(broken))
            with self.assertRaisesRegex(ValueError, "without cells"):
                validation.validate_cell_links(hits, cells.slice(0, 1))

    def test_single_event_fast_join_still_checks_all_context_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = self.shard_fixture(Path(temporary))
            hits = pq.read_table(folder / "hits.parquet").slice(0, 1)
            cells = pq.read_table(folder / "cells.parquet").slice(0, 1)
            validation.validate_cell_links(hits, cells)
            for field in ("campaign", "dataset", "version", "run", "local_event"):
                row = cells.to_pylist()[0]
                row[field] = "wrong" if isinstance(row[field], str) else row[field] + 1
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "full event context"):
                    validation.validate_cell_links(hits, pa.Table.from_pylist([row]))

    def test_numeric_particle_loading_preserves_large_ids_and_nan(self):
        identifier = 2**63 + 11
        batch = pa.record_batch([
            pa.array([0]), pa.array([[identifier]], type=pa.list_(pa.uint64())),
            pa.array([[float("nan")]])], names=["event_id", "particle_id", "perigee_d0"])
        row = validation.particle_batch_row(batch)
        self.assertEqual(row["event_id"], 0)
        self.assertEqual(int(row["particle_id"][0]), identifier)
        self.assertEqual(row["particle_id"].dtype, np.uint64)
        self.assertTrue(np.isnan(row["perigee_d0"][0]))

    def test_flat_event_iterator_rejects_out_of_order_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unordered.parquet"
            pq.write_table(pa.table(dict(local_event=[1, 0], value=[.1, .2])), path)
            with self.assertRaisesRegex(ValueError, "ordered"):
                list(validation.iter_flat_events(path))

    def test_published_layout_resolution_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ("hits", "cells", "particles"):
                name = f"higgs_portal_pu0_{kind}"
                folder = root / name / "data" / name
                folder.mkdir(parents=True)
                (folder / "train-00000-of-00001.parquet").touch()
            report = root / "provenance/run-00003/pu0_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}")
            paths, resolved = validation.sample_files(root, "pu0")
            self.assertEqual(resolved, report)
            self.assertEqual(len(paths), 3)
            paths["hits"].with_name("train-00001-of-00002.parquet").touch()
            with self.assertRaisesRegex(ValueError, "Expected one"):
                validation.sample_files(root, "pu0")

    def test_full_scan_detects_mismatched_event_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sample in ("pu0", "pu200"):
                self.shard_fixture(root, sample)
            path = root / "pu0/particles.parquet"
            table = pq.read_table(path).slice(0, 1)
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "event sets differ"):
                validation.scan_campaign(root, progress=False)

    def perigee_frames(self):
        baseline = pd.DataFrame(dict(particle_id=[1, 2, 3], primary=[True]*3,
            charge=[1.]*3, num_tracker_hits=[1]*3,
            perigee_d0=[.3, np.nan, np.nan], perigee_z0=[.4, np.nan, np.nan]))
        current = baseline.copy()
        current.loc[1, ["perigee_d0", "perigee_z0"]] = [.2, .2]
        reference = pd.DataFrame(dict(particle_id=[1, 2, 3], reference_status=["ok"]*3,
            reference_d0=[.5, .2, .1], reference_z0=[.4, .2, .1],
            candidate_d0=[99.]*3, candidate_path_length_mm=[99.]*3,
            delta_path_length_mm=[99.]*3))
        return current, baseline, reference

    def test_current_perigees_replace_stale_candidate_values(self):
        current, baseline, reference = self.perigee_frames()
        result, counts = validation.compare_current_perigees(
            current, baseline, reference, "pu0", dict(d0=.001, z0=.01))
        self.assertEqual(counts["recovered_validated"], 1)
        self.assertEqual(counts["legacy_finite_preserved"], 1)
        self.assertEqual(result.status.tolist(), ["legacy_mismatch_preserved", "pass", "unresolved"])
        self.assertEqual(result.candidate_d0.iloc[0], .3)
        self.assertNotIn("candidate_path_length_mm", result.columns)
        self.assertNotIn("delta_path_length_mm", result.columns)

    def test_current_perigees_reject_changed_legacy_and_bad_recovery(self):
        current, baseline, reference = self.perigee_frames()
        current.loc[0, "perigee_d0"] = .6
        with self.assertRaisesRegex(ValueError, "legacy finite"):
            validation.compare_current_perigees(current, baseline, reference, "pu0", dict(d0=.001, z0=.01))
        current.loc[0, "perigee_d0"] = .3
        current.loc[1, "perigee_z0"] = 9.
        with self.assertRaisesRegex(ValueError, "recovered perigees disagree"):
            validation.compare_current_perigees(current, baseline, reference, "pu0", dict(d0=.001, z0=.01))

    def test_current_perigees_reject_changed_features_and_reference_coverage(self):
        current, baseline, reference = self.perigee_frames()
        current.loc[0, "charge"] = -1.
        with self.assertRaisesRegex(ValueError, "non-perigee"):
            validation.compare_current_perigees(current, baseline, reference, "pu0", dict(d0=.001, z0=.01))
        current.loc[0, "charge"] = 1.
        with self.assertRaisesRegex(ValueError, "does not cover"):
            validation.compare_current_perigees(current, baseline, reference.iloc[:2], "pu0", dict(d0=.001, z0=.01))


if __name__ == "__main__":
    unittest.main()
