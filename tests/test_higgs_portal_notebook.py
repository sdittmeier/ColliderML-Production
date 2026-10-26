"""Read-only Higgs-portal notebook calculations and configuration."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa

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
                                  value=[.02, .18, .1]))
        return SimpleNamespace(report=dict(events=1, cell_quantization=encoded),
                               paths=dict(cells=Path("cells.parquet")),
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
            with self.assertRaisesRegex(ValueError, "wrong pixel codes"):
                validation.quantization({"pu0": sample})
            sample.analysis.cells.loc[0, "pixel_code"] = 2
            sample.analysis.cells.loc[2, "pixel_code"] = 1
            with self.assertRaisesRegex(ValueError, "encoded strip cells"):
                validation.quantization({"pu0": sample})

    def test_notebook_code_compiles(self):
        path = Path(validation.__file__).with_name("higgs_portal_pilot.ipynb")
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")

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
