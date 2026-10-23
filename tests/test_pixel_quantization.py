"""Occupancy-preserving response, event reference, and centroid checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pyarrow", "pandas", "matplotlib"))


@unittest.skipUnless(AVAILABLE, "requires notebook plotting dependencies")
class PixelQuantizationTest(unittest.TestCase):
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

    def reference(self):
        histogram = self.np.r_[0, self.np.arange(15, 0, -1)]
        return {"histograms": {"calibration": histogram, "held_out": histogram[::-1][::-1]}}

    def analysis(self, sample="pu200", event=0):
        hits, cells = [], []
        for run, mid, values, complete, particle_ids in (
                (0, 0, [.02], True, [1]),
                (1, 0, [.02, .02], True, [2]),
                (0, 1, [.015, .025, .06], False, [3, 4])):
            key = dict(campaign="test", dataset="ttbar", version="v1", run=run,
                       local_event=event, measurement_id=mid)
            hits.append(dict(**key, geometry_id=16 << 56, particle_ids=particle_ids,
                             particle_ids_complete=complete, simhit_ids=[mid]))
            for i, value in enumerate(values):
                cells.append(dict(**key, geometry_id=16 << 56, channel0=10 + i,
                                  channel1=20 + (i % 2), value=value))
        return self.p.build_cluster_analysis(self.pa.Table.from_pylist(hits),
                                             self.pa.Table.from_pylist(cells), sample, event)

    def test_equation_inverse_and_minimum_gain(self):
        np, p = self.np, self.p
        charge = p.athena_charge_for_tot(1)
        self.assertAlmostEqual(charge, 992.3076923076923)
        gain = charge / .01
        response = p.athena_response([.01, .02, .2, 1e100], gain)
        self.assertAlmostEqual(response[0], 1.)
        self.assertTrue(np.all(np.diff(response) >= 0))
        self.assertTrue(np.all(response <= 14))
        codes, _ = p.athena_integer_codes(response)
        self.assertTrue(np.all((codes >= 1) & (codes <= 14)))
        with self.assertRaisesRegex(ValueError, "below ToT 1"):
            p.athena_response([.02], gain * .99)
        with self.assertRaisesRegex(ValueError, "below the declared"):
            p.athena_response([.001], gain)
        for tot in (1, 12, 13, 13.5):
            q = p.athena_charge_for_tot(tot)
            self.assertAlmostEqual(14 * (q - 600) / (q + 4500), tot)
        with self.assertRaises(ValueError):
            p.athena_charge_for_tot(14)

    def test_gain_fit_and_invalid_inputs(self):
        p, np = self.p, self.np
        values = np.linspace(.01, .15, 100)
        gains = p.fit_athena_gains(values)
        for row in gains.itertuples():
            self.assertAlmostEqual(p.athena_response([row.reference_activation], row.gain)[0], row.target_tot)
        with self.assertRaisesRegex(ValueError, "below ToT 1"):
            p.fit_athena_gains([10, 20])
        for values in ([], [0], [-1], [np.nan]):
            with self.assertRaises(ValueError):
                p.fit_athena_gains(values)

    def test_integer_truncation_and_seeded_smearing(self):
        np, p = self.np, self.p
        response = np.array([1., 1.999, 5.5, 13.999, 14.])
        codes, floors = p.athena_integer_codes(response)
        np.testing.assert_array_equal(codes, [1, 1, 5, 13, 14])
        self.assertEqual(floors, {"lower_floor_cells": 0, "upper_clamp_cells": 0})
        values = np.r_[np.ones(1000), np.full(1000, 14.)]
        a, counts = p.athena_integer_codes(values, np.random.default_rng(7))
        b, other = p.athena_integer_codes(values, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(counts, other)
        self.assertGreater(counts["lower_floor_cells"], 0)
        self.assertGreater(counts["upper_clamp_cells"], 0)
        self.assertTrue(np.all((a >= 1) & (a <= 14)))

    def test_empirical_quantiles_ties_and_outliers(self):
        np, p = self.np, self.p
        values = np.linspace(.01, .2, 1001)
        hist = np.r_[0, np.ones(15)]
        boundaries = p.fit_empirical_boundaries(values, hist)
        np.testing.assert_allclose(boundaries, np.quantile(values, np.arange(1, 15) / 15, method="linear"))
        self.assertEqual(p.empirical_codes([boundaries[4]], boundaries)[0], 6)
        codes = p.empirical_codes([.001, *boundaries, 1000], boundaries)
        self.assertEqual(codes[0], 1)
        self.assertEqual(codes[-1], 15)
        self.assertTrue(np.all(np.diff(codes.astype(int)) >= 0))
        tied = p.fit_empirical_boundaries(np.full(20, .02), hist)
        np.testing.assert_array_equal(p.empirical_codes(np.full(20, .02), tied), 15)
        with self.assertRaises(ValueError):
            p.fit_empirical_boundaries(values, np.zeros(16))

    def test_centroids_full_key_invariance_and_asymmetry(self):
        np, p = self.np, self.p
        cells = self.analysis().cells
        before = cells.copy(deep=True)
        native = p.weighted_channel_centroids(cells, cells.value)
        self.assertEqual(len(native), 3)  # repeated measurement ID in different runs
        np.testing.assert_allclose(native[0], [10, 20])
        np.testing.assert_allclose(native[1], [10.5, 20.5])
        np.testing.assert_allclose(native, p.weighted_channel_centroids(cells, 100 * cells.value))
        binary = p.weighted_channel_centroids(cells, np.ones(len(cells)))
        self.assertGreater(np.linalg.norm(native[2] - binary[2]), 0)
        self.pd.testing.assert_frame_equal(cells, before)
        with self.assertRaisesRegex(ValueError, "positive weight"):
            p.weighted_channel_centroids(cells, np.zeros(len(cells)))

    def test_event_reference_and_bootstrap(self):
        np, p = self.np, self.p
        files = [Path(f"event{i}.pyg") for i in range(6)]
        visited = []
        def reader(path):
            visited.append(path)
            i = files.index(path)
            return dict(values=np.array([1, 2, 15] if i % 2 else [1, 1, 2]),
                        event_id=np.asarray(str(i)), n_cells=np.array([2, 1]),
                        contains_code15=np.array([False, bool(i % 2)]),
                        contains_overflow15=np.array([False, bool(i % 2)]),
                        touches_edge=np.array([False, True]))
        reference = p.collect_atlas_code_reference(files, reader=reader)
        self.assertEqual(visited, files)
        self.assertEqual(reference["events"].cohort.tolist(), ["calibration", "held_out"] * 3)
        self.assertEqual(reference["histograms"]["calibration"][15], 0)
        self.assertEqual(reference["histograms"]["held_out"][15], 3)
        pooled = reference["pooled"].set_index("cohort")
        self.assertAlmostEqual(pooled.loc["held_out", "code15_cell_fraction"], 1 / 3)
        self.assertAlmostEqual(pooled.loc["held_out", "code15_cluster_fraction"], 1 / 2)
        again = p.collect_atlas_code_reference(files, reader=reader)
        self.pd.testing.assert_frame_equal(reference["intervals"], again["intervals"])
        self.assertFalse(any(isinstance(v, np.ndarray) for v in reference["events"].iloc[0]))
        with self.assertRaisesRegex(ValueError, "distinct files"):
            p.collect_atlas_code_reference([files[0]] * 4, reader=reader)

    def test_native_codes_above15_are_reported_and_folded_only_for_matching(self):
        np, p = self.np, self.p
        def reader(path):
            return dict(values=np.array([1, 15, 16]), event_id=np.asarray(str(path)),
                        n_cells=np.array([2, 1]), contains_code15=np.array([True, False]),
                        contains_overflow15=np.array([True, True]), touches_edge=np.array([False, False]))
        reference = p.collect_atlas_code_reference([Path(f"event{i}") for i in range(4)], reader=reader)
        self.assertEqual(reference["histograms"]["calibration"][15], 4)
        self.assertEqual(reference["native_histograms"]["calibration"][16], 2)
        row = reference["pooled"].iloc[0]
        self.assertEqual(row.native_max_code, 16)
        self.assertEqual(row.cells_above15, 2)
        self.assertAlmostEqual(row.code15_cell_fraction, 1 / 3)
        self.assertAlmostEqual(row.overflow15_cell_fraction, 2 / 3)
        self.assertAlmostEqual(row.code15_cluster_fraction, 1 / 2)
        self.assertAlmostEqual(row.overflow15_cluster_fraction, 1.)

    def test_file_selection_and_histogram_percentile(self):
        p, np = self.p, self.np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(25):
                (root / f"event{i:03}.pyg").touch()
            files = p.select_atlas_reference_files(root, 20)
            self.assertEqual(len(set(files)), 20)
            self.assertEqual(files[0].name, "event000.pyg")
            self.assertEqual(files[-1].name, "event024.pyg")
            with self.assertRaises(ValueError):
                p.select_atlas_reference_files(root, 30)
        values = np.array([1, 1, 2, 15])
        hist = np.bincount(values, minlength=16)
        for percentile in p.CODE_PERCENTILES:
            self.assertAlmostEqual(p._histogram_percentile(hist, percentile), np.percentile(values, percentile))

    def test_study_freezes_fit_and_preserves_sources(self):
        p, np = self.p, self.np
        analyses = {sample: self.analysis(sample, 1) for sample in p.PILEUP_COLORS}
        originals = {sample: a.cells.copy(deep=True) for sample, a in analyses.items()}
        calibration = np.linspace(.01, .15, 300)
        reference = self.reference()
        study = p.run_pixel_quantization(analyses, calibration, reference)
        self.assertEqual(len(study.weights), 14)
        self.assertEqual(len(study.shifts), 14)
        self.assertFalse(any(variant.endswith("_smeared") for _, variant in study.weights))
        for (sample, variant), weights in study.weights.items():
            self.assertEqual(len(weights), len(originals[sample]))
            self.assertTrue(np.all(weights >= 1))
            self.assertTrue(np.all(weights <= (15 if variant == "empirical" else 14)))
        for sample in analyses:
            self.pd.testing.assert_frame_equal(originals[sample], analyses[sample].cells)
        self.assertTrue((study.summary.event == 1).all())
        self.assertIn("Unknown/incomplete", set(study.summary.group))
        self.assertTrue((study.code_comparison.query("variant != 'empirical' and code == 15").fraction == 0).all())
        alternative = self.reference()
        alternative["histograms"]["held_out"] = np.r_[0, np.ones(15) * 100]
        other = p.run_pixel_quantization(analyses, calibration, alternative)
        np.testing.assert_array_equal(study.boundaries, other.boundaries)
        for key in study.weights:
            np.testing.assert_array_equal(study.weights[key], other.weights[key])
        with patch.object(p.plt, "show"):
            p.plot_pixel_quantization(study, reference)
            examples = p.plot_quantization_examples(study, per_sample=1)
        self.assertEqual(len(examples), 2)

    def test_study_smearing_is_opt_in(self):
        p, np = self.p, self.np
        analyses = {sample: self.analysis(sample, 1) for sample in p.PILEUP_COLORS}
        calibration = np.linspace(.01, .15, 300)
        baseline = p.run_pixel_quantization(analyses, calibration, self.reference())
        noisy = p.run_pixel_quantization(analyses, calibration, self.reference(), include_smearing=True)
        repeated = p.run_pixel_quantization(analyses, calibration, self.reference(), include_smearing=True)
        self.assertEqual(len(noisy.weights), 20)
        self.assertEqual(sum(v.endswith("_smeared") for _, v in noisy.weights), 6)
        for key, weights in baseline.weights.items():
            np.testing.assert_array_equal(weights, noisy.weights[key])
        for key, weights in noisy.weights.items():
            np.testing.assert_array_equal(weights, repeated.weights[key])

    def test_multievent_exact_pooled_fit_and_calibration_separation(self):
        p, np = self.p, self.np
        populations = {0: np.array([.01, .02, .02]),
                       1: np.array([.03, .04, .08, .1]), 2: np.array([99.])}
        visited = []
        def reader(root, event):
            visited.append(event)
            return populations[event]
        with patch.object(p, "pixel_calibration_values", side_effect=reader):
            calibration = p.fit_multievent_empirical(Path("unused"), self.reference(),
                                                     events=(0, 1), checkpoints=(1, 2), progress=None)
        self.assertEqual(visited, [0, 1])
        expected = p.fit_empirical_boundaries(np.concatenate([populations[0], populations[1]]),
                                              self.reference()["histograms"]["calibration"])
        np.testing.assert_allclose(calibration.boundaries, expected)
        self.assertEqual(calibration.events, (0, 1))
        final = calibration.stability.query("calibration_events == 2")
        self.assertTrue((final.cells == 7).all())
        np.testing.assert_array_equal(final.relative_change_to_final, 0)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            p.validate_multievent_empirical("unused", calibration, self.reference(), (1, 2), progress=None)
        for events in ((0, 0), (), (-1,), (0.5,)):
            with self.assertRaisesRegex(ValueError, "distinct nonnegative"):
                p.fit_multievent_empirical("unused", self.reference(), events, checkpoints=(1,), progress=None)

    def test_multievent_validation_counts_pooling_and_plotting(self):
        p, np = self.p, self.np
        with patch.object(p, "pixel_calibration_values", return_value=np.linspace(.01, .15, 101)):
            calibration = p.fit_multievent_empirical("unused", self.reference(), (0, 1), (1, 2), progress=None)
        visited = []
        def loader(root, sample, event):
            visited.append((sample, event))
            return self.analysis(sample, event)
        result = p.validate_multievent_empirical("unused", calibration, self.reference(), (2, 3),
                                                 loader=loader, progress=None)
        self.assertEqual(visited, [(s, e) for s in ("pu0", "pu200") for e in range(4)])
        self.assertEqual(len(result.events), 8)
        self.assertTrue((result.events.cells == 6).all())
        self.assertTrue((result.events.clusters == 3).all())
        self.assertTrue((result.pooled.events == 2).all())
        self.assertTrue((result.pooled.cells == 12).all())
        self.assertTrue((result.pooled.clusters == 6).all())
        self.assertEqual(len(result.convergence), 4)
        pooled = result.shifts.query("event == 'pooled' and grouping == 'all'")
        analysis = self.analysis()
        codes = p.empirical_codes(analysis.cells.value, calibration.boundaries)
        distances = np.linalg.norm(p.weighted_channel_centroids(analysis.cells, codes)
                                    - p.weighted_channel_centroids(analysis.cells, analysis.cells.value), axis=1)
        for row in pooled.itertuples():
            self.assertAlmostEqual(row.q99, np.quantile(np.tile(distances, 2), .99))
            self.assertAlmostEqual(row.maximum, distances.max())
        self.assertIn("Unknown/incomplete", set(result.shifts.group))
        self.assertTrue((result.survival.fraction > 0).all())
        bounds_before = calibration.boundaries.copy()
        with patch.object(p.plt, "show"):
            p.plot_multievent_empirical(calibration, result, self.reference())
            with patch.object(p, "load_pixel_analysis", side_effect=loader):
                examples = p.plot_campaign_examples("unused", calibration, result, per_sample=1)
        self.assertEqual(len(examples), 2)
        np.testing.assert_array_equal(bounds_before, calibration.boundaries)
        changed = self.reference()
        changed["histograms"]["calibration"] = np.r_[0, np.ones(15)]
        with self.assertRaisesRegex(ValueError, "reference changed"):
            p.validate_multievent_empirical("unused", calibration, changed, (2, 3), progress=None)

    def test_pixel_event_projection_and_source_checksums(self):
        import pyarrow.parquet as pq
        p, np, pa = self.p, self.np, self.pa
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample in p.PILEUP_COLORS:
                folder = root / sample
                folder.mkdir()
                analysis = self.analysis(sample, 0)
                # Include a strip in the same event and pixels in a different event.
                cells = analysis.cells[[*p.MEASUREMENT_KEY, "geometry_id", "channel0", "channel1", "value"]].to_dict("records")
                cells += [{**cells[0], "geometry_id": 23 << 56, "value": 10.},
                          {**cells[0], "local_event": 1, "value": 20.}]
                pq.write_table(pa.Table.from_pylist(cells), folder / "cells.parquet")
                for name in ("hits", "particles"):
                    pq.write_table(pa.table({"marker": [1]}), folder / f"{name}.parquet")
            before = p.quantization_source_checksums(root)
            values = p.pixel_calibration_values(root, 0)
            np.testing.assert_array_equal(values, analysis.cells.value)
            self.assertEqual(p._pixel_event_table(root, "pu200", "cells", 0, ["value"]).column_names, ["value"])
            with self.assertRaisesRegex(ValueError, "no pixel"):
                p.pixel_calibration_values(root, 2)
            self.assertEqual(before, p.quantization_source_checksums(root))

    def test_selected_event_uses_external_empirical_boundaries(self):
        p, np = self.p, self.np
        values = np.linspace(.01, .15, 200)
        boundaries = np.linspace(.02, .1, 14)
        analyses = {s: self.analysis(s, 25) for s in p.PILEUP_COLORS}
        study = p.run_pixel_quantization(analyses, values, self.reference(), empirical_boundaries=boundaries)
        np.testing.assert_array_equal(study.boundaries, boundaries)
        for sample, analysis in analyses.items():
            np.testing.assert_array_equal(study.weights[(sample, "empirical")],
                                          p.empirical_codes(analysis.cells.value, boundaries))

    def test_fifty_event_campaign_releases_each_event_analysis(self):
        import weakref
        p, np = self.p, self.np
        read_events = []
        def values(root, event):
            read_events.append(event)
            return np.linspace(.01, .1 + event / 1000, 40)
        with patch.object(p, "pixel_calibration_values", side_effect=values):
            calibration = p.fit_multievent_empirical("unused", self.reference(), progress=None)
        self.assertEqual(read_events, list(range(25)))
        previous, visited = [], []
        def loader(root, sample, event):
            if previous:
                self.assertIsNone(previous[-1](), "An entire previous event was retained")
            analysis = self.analysis(sample, event)
            previous.append(weakref.ref(analysis))
            visited.append((sample, event))
            return analysis
        result = p.validate_multievent_empirical("unused", calibration, self.reference(),
                                                 loader=loader, progress=None)
        self.assertEqual(visited, [(s, e) for s in p.PILEUP_COLORS for e in range(50)])
        self.assertEqual(len(result.events), 100)
        self.assertTrue((result.pooled.events == 25).all())
        self.assertTrue((result.pooled.clusters == 75).all())
        self.assertTrue((result.pooled.cells == 150).all())
        self.assertEqual(len(result.convergence), 10)
        self.assertTrue(all(ref() is None for ref in previous))


if __name__ == "__main__":
    unittest.main()
