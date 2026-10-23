"""Frozen percentile-overflow scans and position changes, without ROOT truth."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pyarrow", "pandas", "matplotlib"))


@unittest.skipUnless(AVAILABLE, "requires notebook plotting dependencies")
class LinearThresholdScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        from notebooks.validation import paired_vs_published as plots
        cls.np, cls.pd, cls.pa, cls.p = np, pd, pa, plots

    def tearDown(self):
        self.p.plt.close("all")

    def calibration(self):
        np, p = self.np, self.p
        self.values = {0: np.linspace(.01, .15, 101), 1: np.linspace(.01, .2, 301)}
        hist = np.r_[0, np.full(14, 1000), 1]
        self.reference = {"histograms": {"calibration": hist, "held_out": hist.copy()}}
        bounds = p.fit_empirical_boundaries(np.concatenate(list(self.values.values())), hist)
        self.empirical = p.EmpiricalCalibration((0, 1), bounds, self.pd.DataFrame(), hist)
        visited = []
        def reader(root, event):
            visited.append(event)
            return self.values[event]
        with patch.object(p, "pixel_calibration_values", side_effect=reader):
            self.scan = p.fit_linear_threshold_scan("unused", self.empirical, progress=None)
        return visited

    def analysis(self, sample, event):
        hits, cells = [], []
        for run, mid, values, complete in ((0, 0, [.02], True),
                (1, 0, [.02, .07, .24], True), (0, 1, [.03, .1], False)):
            key = dict(campaign=sample, dataset="ttbar", version="v1", run=run,
                       local_event=event, measurement_id=mid)
            hits.append(dict(**key, geometry_id=16 << 56, particle_ids=[1] if complete else [],
                             particle_ids_complete=complete, simhit_ids=[mid]))
            for index, value in enumerate(values):
                cells.append(dict(**key, geometry_id=16 << 56, channel0=10 + index,
                                  channel1=20 + (index % 2), value=value))
        return self.p.build_cluster_analysis(self.pa.Table.from_pylist(hits),
                                             self.pa.Table.from_pylist(cells), sample, event)

    def test_pooled_percentile_edges_and_no_heldout_calibration(self):
        np, p = self.np, self.p
        self.assertEqual(self.calibration(), [0, 1])
        expected = np.percentile(np.concatenate(list(self.values.values())), [99.9, 99.5, 99.])
        observed = [self.scan.variants[f"linear_P{value:g}"].overflow_edge for value in (99.9, 99.5, 99.)]
        np.testing.assert_allclose(observed, expected)
        self.assertTrue(np.all(np.diff(observed) < 0))
        self.assertEqual(self.scan.events, (0, 1))
        self.assertTrue((self.scan.calibration.cells == 402).all())
        for name, model in self.scan.variants.items():
            np.testing.assert_allclose(np.diff(np.r_[.01, model.boundaries]), (model.overflow_edge - .01) / 14)
            np.testing.assert_array_equal(p.linear_codes([.01, model.overflow_edge, 1e9], model), [1, 15, 15])
        for percentiles in ((99, 99), (100,), (self.np.nan,), ()):
            with self.assertRaisesRegex(ValueError, "distinct finite"):
                p.fit_linear_threshold_scan("unused", self.empirical, percentiles, progress=None)

    def test_ties_report_achieved_overflow_instead_of_random_splitting(self):
        self.calibration()
        with patch.object(self.p, "pixel_calibration_values", return_value=self.np.full(10, .02)):
            scan = self.p.fit_linear_threshold_scan("unused", self.empirical, progress=None)
        variants = scan.calibration.query("variant != 'linear_ATLAS'")
        self.assertTrue((variants.actual_calibration_overflow_fraction == 1.).all())
        self.assertTrue((variants.target_overflow_fraction < .02).all())

    def test_full_key_counts_grouping_displacements_and_plots(self):
        np, p = self.np, self.p
        self.calibration()
        visited = []
        def loader(root, sample, event):
            visited.append((sample, event))
            return self.analysis(sample, event)
        originals = self.analysis("pu0", 25).cells.copy(deep=True)
        result = p.run_linear_threshold_scan("unused", self.empirical, self.scan,
                                             events=(25, 49), loader=loader, progress=None)
        self.assertEqual(visited, [(s, e) for s in p.PILEUP_COLORS for e in (25, 49)])
        self.assertEqual(len(result.events), 20)
        self.assertTrue((result.events.cells == 6).all())
        self.assertTrue((result.events.clusters == 3).all())
        self.assertTrue((result.overflow.cells == 12).all())
        self.assertTrue((result.overflow.clusters == 6).all())
        self.assertIn("Unknown/incomplete", set(result.summary.group))
        self.assertIn("contains_overflow", set(result.summary.grouping))
        self.pd.testing.assert_frame_equal(self.analysis("pu0", 25).cells, originals)
        analysis = self.analysis("pu0", 25)
        for name, model in self.scan.variants.items():
            delta = p.weighted_channel_centroids(analysis.cells, p.linear_codes(analysis.cells.value, model))
            delta -= p.weighted_channel_centroids(analysis.cells, analysis.cells.value)
            distances = np.linalg.norm(delta, axis=1)
            row = result.summary.query("sample == 'pu0' and variant == @name and grouping == 'all'").iloc[0]
            self.assertAlmostEqual(row.q99, np.quantile(np.tile(distances, 2), .99))
            self.assertAlmostEqual(row.fraction_gt_0_1, np.mean(distances > .1))
            self.assertAlmostEqual(row.mean_delta1, delta[:, 1].mean())
        with patch.object(p.plt, "show"):
            p.plot_linear_threshold_scan(self.empirical, self.scan, result, self.reference)
            with patch.object(p, "load_pixel_analysis", side_effect=loader):
                examples = p.plot_linear_scan_examples("unused", self.empirical, self.scan, result,
                                                        variants=("linear_P99.5",), per_sample=1)
        self.assertEqual(len(examples), 2)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            p.run_linear_threshold_scan("unused", self.empirical, self.scan, events=(0,), progress=None)

    def test_fifty_event_scan_does_not_retain_event_analyses(self):
        import weakref
        p = self.p
        self.calibration()
        previous = []
        def loader(root, sample, event):
            if previous:
                self.assertIsNone(previous[-1]())
            analysis = self.analysis(sample, event)
            previous.append(weakref.ref(analysis))
            return analysis
        result = p.run_linear_threshold_scan("unused", self.empirical, self.scan,
                                             events=range(25, 50), loader=loader, progress=None)
        self.assertEqual(len(result.events), 250)  # 25 events x two samples x five maps
        self.assertTrue(all(ref() is None for ref in previous))
        self.assertTrue((result.overflow.events == 25).all())


if __name__ == "__main__":
    unittest.main()
