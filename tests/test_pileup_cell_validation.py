"""Read-only contributor analysis and ATLAS window-summary checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pyarrow", "pandas", "matplotlib"))


@unittest.skipUnless(AVAILABLE, "requires notebook plotting dependencies")
class PileupCellValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from notebooks.validation import paired_vs_published as plots
        cls.np, cls.pa, cls.pq, cls.plots = np, pa, pq, plots

    def tearDown(self):
        self.plots.plt.close("all")

    def tables(self, event=0):
        hits, cells = [], []
        specs = [(0, 0, 28, [1], True, [11.7]),
                 (1, 0, 16, [1, 2], True, [0.6, 0.2, 0.1]),
                 (0, 1, 16, [1, 2], True, [0.2, 0.1]),
                 (0, 2, 16, [1, 2], False, [0.9])]
        for run, mid, volume, particles, complete, values in specs:
            key = dict(campaign="test", dataset="ttbar", version="v1", run=run,
                       local_event=event, measurement_id=mid)
            geometry = (volume << 56) | 256
            hits.append({**key, "geometry_id": geometry, "particle_ids": particles,
                         "particle_ids_complete": complete, "simhit_ids": list(range(len(particles)))})
            for channel, value in enumerate(values):
                cells.append({**key, "geometry_id": geometry, "channel0": 10 + channel,
                              "channel1": 5, "value": value})
        return self.pa.Table.from_pylist(hits), self.pa.Table.from_pylist(cells)

    def test_full_key_join_and_single_particle_large_cell(self):
        hits, cells = self.tables()
        before = cells.to_pylist()
        analysis = self.plots.build_cluster_analysis(hits, cells, "pu200", 0)
        self.assertEqual(len(analysis.clusters), 4)
        self.assertEqual(len(analysis.cells), 7)
        maximum = analysis.cells.loc[analysis.cells["value"].idxmax()]
        self.assertEqual(maximum["known_particles"], 1)
        self.assertEqual(maximum["simhit_count"], 1)
        self.assertEqual(maximum["contributor_group"], "1 particle")
        self.assertEqual(cells.to_pylist(), before)
        repeated = analysis.clusters[analysis.clusters["measurement_id"] == 0]
        self.assertEqual(set(repeated["run"]), {0, 1})
        self.assertEqual(set(repeated["n_cells"]), {1, 3})
        report = self.plots.pileup_event_summary({"pu200": analysis})
        row = report[report["detector_type"] == "Long strips"].iloc[0]
        self.assertEqual(row["max_cell_parent_particles"], 1)
        self.assertEqual(row["max_cell_parent_simhits"], 1)

    def test_incomplete_truth_and_denominators_are_separate(self):
        analysis = self.plots.build_cluster_analysis(*self.tables(), "pu200", 0)
        incomplete = analysis.cells[~analysis.cells["truth_complete"]]
        self.assertEqual(incomplete["contributor_group"].tolist(), ["Unknown/incomplete"])
        stats = self.plots.contributor_statistics({"pu200": analysis})
        complete = stats[(stats["detector_type"] == "Pixels") & stats["truth_complete"]].iloc[0]
        self.assertEqual(complete["known_particles"], 2)
        self.assertEqual(complete["cells"], 5)
        self.assertEqual(complete["clusters"], 2)
        self.assertAlmostEqual(complete["cell_tail_fraction"], 1 / 5)
        self.assertAlmostEqual(complete["cluster_tail_fraction"], 1 / 2)
        self.assertEqual(len(stats[(stats["detector_type"] == "Pixels") & ~stats["truth_complete"]]), 1)
        largest = self.plots.largest_cells({"pu200": analysis}, 2)
        self.assertEqual(largest["value"].tolist(), [11.7, 0.9])
        self.assertEqual(largest["n_cells"].tolist(), [1, 1])

    def test_join_failures_and_invalid_event_are_rejected(self):
        hits, cells = self.tables()
        with self.assertRaisesRegex(ValueError, "missing parent"):
            self.plots.build_cluster_analysis(hits.slice(1), cells, "pu0", 0)
        with self.assertRaisesRegex(ValueError, "selected local event"):
            self.plots.build_cluster_analysis(hits, cells, "pu0", 1)
        with self.assertRaisesRegex(ValueError, "threshold"):
            self.plots.build_cluster_analysis(hits, cells, "pu0", 0, self.np.nan)

    def test_loader_selects_only_requested_event(self):
        with tempfile.TemporaryDirectory() as folder:
            for sample in ("pu0", "pu200"):
                target = Path(folder) / sample
                target.mkdir()
                pairs = [self.tables(event) for event in (0, 1)]
                for index, name in enumerate(("hits", "cells")):
                    self.pq.write_table(self.pa.concat_tables([pair[index] for pair in pairs]), target / f"{name}.parquet")
            analyses = self.plots.load_pileup_comparison(Path(folder), event=1)
            for analysis in analyses.values():
                self.assertEqual(set(analysis.cells["local_event"]), {1})
                self.assertEqual(len(analysis.clusters), 4)

    def atlas_summary(self):
        images = self.np.zeros((5, 11, 11), dtype=self.np.uint8)
        images[1, 0, 5] = 15
        images[1, 2, 7] = 2
        images[2, 5, 5] = 1
        images[3, 5, 5] = 9  # A nonempty strip must still be excluded.
        return self.plots.summarize_atlas_pixel_images(images, [1, 1, 3, 2, 6], {1, 3, 5}, "193")

    def test_atlas_pixel_selection_and_window_shapes(self):
        atlas = self.atlas_summary()
        self.assertEqual(atlas["n_cells"].tolist(), [0, 2, 1])
        self.assertEqual(atlas["values"].tolist(), [15, 2, 1])
        self.assertEqual(atlas["width0"].tolist(), [0, 3, 1])
        self.assertEqual(atlas["width1"].tolist(), [0, 3, 1])
        self.assertTrue(self.np.isnan(atlas["occupancy"][0]))
        self.assertAlmostEqual(atlas["occupancy"][1], 2 / 9)
        self.assertEqual(atlas["touches_edge"].tolist(), [False, True, False])
        report = self.plots.atlas_pixel_report(atlas).iloc[0]
        self.assertEqual(report["pixel_images"], 3)
        self.assertEqual(report["empty_pixel_images"], 1)
        self.assertEqual(report["excluded_nonpixel_hits"], 2)
        self.assertEqual(report["excluded_nonempty_images"], 1)
        self.assertAlmostEqual(report["window_edge_fraction"], 1 / 3)
        self.assertAlmostEqual(report["window_edge_fraction_nonempty"], 1 / 2)
        self.assertAlmostEqual(report["fraction_at_code_15"], 1 / 3)

    def test_comparison_plots_use_expected_sources(self):
        analysis = self.plots.build_cluster_analysis(*self.tables(), "pu0", 0)
        atlas = self.atlas_summary()
        with patch.object(self.plots.plt, "show"):
            self.plots.plot_atlas_pixel_comparison({"pu0": analysis}, atlas)
        figures = [self.plots.plt.figure(number) for number in self.plots.plt.get_fignums()]
        self.assertEqual(len(figures), 2)
        self.assertIn("ColliderML native pixel activation", [ax.get_title() for ax in figures[1].axes])
        self.assertIn("ATLAS native pixel codes (different units)", [ax.get_title() for ax in figures[1].axes])
        self.assertEqual(figures[1].axes[1].get_yscale(), "log")
        self.assertEqual(atlas["values"].tolist(), [15, 2, 1])

    def test_survival_keeps_full_range_and_duplicate_weights(self):
        _, ax = self.plots.plt.subplots()
        self.plots._survival(ax, {"pu0": self.np.array([0.1, 0.1, 11.7])}, self.plots.PILEUP_COLORS)
        self.np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.1, 11.7])
        self.np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 1 / 3])
        self.assertEqual(ax.lines[0].get_drawstyle(), "steps-pre")


if __name__ == "__main__":
    unittest.main()
