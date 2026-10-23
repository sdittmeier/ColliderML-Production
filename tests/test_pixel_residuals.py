"""Linear overflow calibration and local truth residuals without production changes."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pyarrow", "pandas", "matplotlib"))


@unittest.skipUnless(AVAILABLE, "requires notebook plotting dependencies")
class PixelResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        from notebooks.validation import paired_vs_published as plots
        cls.np, cls.pd, cls.pa, cls.p = np, pd, pa, plots

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = Path(self.temporary.name) / "digi.json"
        self.config.write_text(json.dumps({"entries": [
            {"volume": volume, "value": {"geometric": {"digital": False, "indices": [0, 1],
                "segmentation": {"binningdata": [
                    {"min": -.5, "max": .5, "bins": 10, "value": axis, "type": "equidistant"}
                    for axis in ("binX", "binY")]}}}}
            for volume in (16, 17, 18)]}))
        histogram = self.np.r_[0, self.np.arange(15, 0, -1)]
        self.reference = {"histograms": {"calibration": histogram, "held_out": histogram.copy()}}
        self.empirical = self.p.EmpiricalCalibration((0, 1), self.np.linspace(.015, .12, 14),
                                                    self.pd.DataFrame(), histogram)
        self.linear = self.p.fit_linear_calibration(self.empirical)

    def tearDown(self):
        self.p.plt.close("all")

    def fixture(self, sample, event):
        p, np, pa = self.p, self.np, self.pa
        hits, cells, truth = [], [], []
        for mid, values, simhits, complete in ((0, [.02], [0], True),
                (1, [.02, .08], [1], True), (2, [.03, .05], [2, 3], True),
                (3, [.03], [4], False), (4, [.04], [5], True)):
            key = dict(campaign=sample, dataset="ttbar", version="v1", run=0,
                       local_event=event, measurement_id=mid)
            volume = 16 + mid % 3
            geometry = (volume << 56) | (2 << 36) | (3 << 8)
            addresses = np.column_stack([np.arange(len(values)) + 4, np.full(len(values), 4)])
            native = np.average(-.5 + (addresses + .5) * .1, axis=0, weights=values)
            hits.append(dict(**key, geometry_id=geometry, particle_ids=[mid] if complete else [],
                             particle_ids_complete=complete, simhit_ids=simhits,
                             local0=native[0], local1=native[1]))
            for address, value in zip(addresses, values):
                cells.append(dict(**key, geometry_id=geometry, channel0=int(address[0]),
                                  channel1=int(address[1]), value=value))
            true = native + [.01, -.005] if mid < 3 else np.array([np.nan, np.nan])
            truth.append(dict(local_event=event, measurement_id=mid, volume_id=volume,
                              layer_id=2, surface_id=3, rec_loc0=native[0], rec_loc1=native[1],
                              true_loc0=true[0], true_loc1=true[1],
                              residual_loc0=native[0] - true[0], residual_loc1=native[1] - true[1]))
        hit_table = pa.Table.from_pylist(hits)
        analysis = p.build_cluster_analysis(hit_table, pa.Table.from_pylist(cells), sample, event)
        analysis.clusters = analysis.clusters.merge(
            hit_table.select(p.MEASUREMENT_KEY + ["local0", "local1"]).to_pandas(),
            on=p.MEASUREMENT_KEY, validate="one_to_one")
        return analysis, self.pd.DataFrame(truth)

    def run_study(self, **kwargs):
        return self.p.run_pixel_residual_study("unused", self.empirical, self.linear, self.reference,
            self.config, {}, events=(25, 49), progress=None,
            analysis_loader=lambda root, sample, event: self.fixture(sample, event)[0],
            truth_loader=kwargs.pop("truth_loader", lambda sample, event: self.fixture(sample, event)[1]),
            **kwargs)

    def test_linear_bins_overflow_ties_and_positive_occupancy(self):
        np, p = self.np, self.p
        edges = self.linear.boundaries
        np.testing.assert_allclose(np.diff(np.r_[.01, edges]), (.12 - .01) / 14)
        self.assertEqual(self.linear.target_overflow_fraction, 1 / 120)
        codes = p.linear_codes([.01, *edges, .12, 100.], self.linear)
        np.testing.assert_array_equal(codes[:15], np.arange(1, 16))
        self.assertEqual(codes[-1], 15)
        values = np.linspace(.01, .5, 1000)
        np.testing.assert_array_equal(p.linear_codes(values, self.linear) == 15,
                                      p.empirical_codes(values, self.empirical.boundaries) == 15)
        with self.assertRaisesRegex(ValueError, "below the declared"):
            p.linear_codes([.009], self.linear)
        with self.assertRaisesRegex(ValueError, "Linear range"):
            p.fit_linear_calibration(self.empirical, .13)

    def test_true_residuals_units_selection_and_source_frames_unchanged(self):
        np, p = self.np, self.p
        analysis, truth = self.fixture("pu0", 25)
        original = analysis.cells.copy(deep=True)
        with patch.object(p, "load_pixel_analysis", return_value=analysis):
            result = p.run_pixel_residual_study("unused", self.empirical, self.linear, self.reference,
                self.config, {}, events=(25,), progress=None, truth_loader=lambda sample, event: truth.copy())
        self.pd.testing.assert_frame_equal(analysis.cells, original)
        self.assertTrue((result.coverage.pixel_clusters == 5).all())
        self.assertTrue((result.coverage.residual_clusters == 2).all())
        self.assertTrue((result.coverage.complete_multiple_simhits == 1).all())
        self.assertTrue((result.coverage.incomplete_truth == 1).all())
        self.assertTrue((result.coverage.nonfinite_truth == 1).all())
        native = result.summary.query("event == 'pooled' and grouping == 'all' and variant == 'native'")
        np.testing.assert_allclose(native.query("axis == 'local0'").bias, -10.)
        np.testing.assert_allclose(native.query("axis == 'local1'").bias, 5.)
        np.testing.assert_allclose(native.central68_half_width, 0., atol=1e-12)
        all_codes = result.codes.query("atlas_cohort == 'held_out'")
        self.assertTrue((all_codes.cells == 7).all())  # all cells, not only truth-selected clusters
        self.assertTrue((all_codes.clusters == 5).all())

    def test_alignment_geometry_sign_and_separation_fail_closed(self):
        for column, message in (("surface_id", "geometry differs"), ("rec_loc0", "Native centroid differs"),
                                 ("residual_loc0", "convention")):
            def bad_truth(sample, event):
                frame = self.fixture(sample, event)[1]
                frame.loc[0, column] += 1
                return frame
            with self.assertRaisesRegex(ValueError, message):
                self.run_study(truth_loader=bad_truth)
        def missing_truth(sample, event):
            return self.fixture(sample, event)[1].iloc[1:]
        with self.assertRaisesRegex(ValueError, "missing ROOT measurement"):
            self.run_study(truth_loader=missing_truth)
        with self.assertRaisesRegex(ValueError, "held out"):
            self.p.run_pixel_residual_study("unused", self.empirical, self.linear, self.reference,
                self.config, {}, events=(0,), progress=None)

    def test_pooled_statistics_tail_accounting_and_plots(self):
        p, np = self.p, self.np
        result = self.run_study(histogram_edges=[-1, 0, 1])
        native = result.histograms.query("variant == 'native'")
        self.assertTrue((native.total_clusters == 4).all())
        self.assertTrue(((native.below_range + native.above_range) == 4).all())
        self.assertTrue((native.density == 0).all())
        self.assertTrue((result.survival.fraction > 0).all())
        with patch.object(p.plt, "show"):
            p.plot_pixel_residuals(result)
            p.plot_linear_comparison(self.empirical, self.linear,
                {sample: self.fixture(sample, 25)[0] for sample in p.PILEUP_COLORS}, self.reference)

    def test_named_linear_variants_have_common_truth_selection_and_fit_statistics(self):
        p, np = self.p, self.np
        models = {"linear_ATLAS": self.linear,
                  "linear_P99.9": p.LinearCalibration(.01, .09, .001, np.linspace(.01, .09, 15)[1:])}
        result = self.run_study(linear_variants=models)
        self.assertEqual(set(result.summary.variant), {"native", "empirical", *models})
        all_rows = result.summary.query("event == 'pooled' and grouping == 'all'")
        self.assertTrue((all_rows.clusters == 4).all())
        self.assertEqual(set(result.fit_statistics.window_um), {30.})
        self.assertTrue((result.fit_statistics.query("grouping == 'all'").n_full == 2).all())
        with patch.object(p.plt, "show"):
            p.plot_pixel_residuals(result)

    @unittest.skipUnless(importlib.util.find_spec("uproot"), "requires uproot")
    def test_root_measurement_positions_preserved_for_interleaved_events(self):
        import uproot
        np, p = self.np, self.p
        path = Path(self.temporary.name) / "measurements.root"
        arrays = {name: np.arange(4, dtype=float) for name in p.RootMeasurementTruth.fields}
        arrays["event_nr"] = np.array([1, 0, 1, 0], dtype=np.uint32)
        with uproot.recreate(path) as root:
            root["measurements"] = arrays
        with p.RootMeasurementTruth(path) as reader:
            event = reader.read(0)
            np.testing.assert_array_equal(event.measurement_id, [0, 1])
            np.testing.assert_array_equal(event.rec_loc0, [1., 3.])
            np.testing.assert_array_equal(reader.read(1).rec_loc0, [0., 2.])
            with self.assertRaisesRegex(ValueError, "missing ROOT event"):
                reader.read(2)


if __name__ == "__main__":
    unittest.main()
