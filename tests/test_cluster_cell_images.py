"""Cluster galleries must preserve channel placement and activation values."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pyarrow", "pandas", "matplotlib"))


@unittest.skipUnless(AVAILABLE, "requires the notebook plotting dependencies")
class ClusterCellImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import pyarrow as pa
        from notebooks.validation import paired_vs_published as plots
        cls.np, cls.pa, cls.plots = np, pa, plots

    def tearDown(self):
        self.plots.plt.close("all")

    def event(self, sizes=(1, 2, 3, 4, 8, 16, 32), volumes=(16, 23, 28)):
        cells, hits = [], []
        for family, volume in enumerate(volumes):
            geometry = (volume << 56) | 256
            for offset, count in enumerate(sizes):
                mid = family * 100 + offset
                hits.append({"measurement_id": mid, "geometry_id": geometry,
                             "particle_ids": [4], "particle_ids_complete": True})
                for index in range(count):
                    cells.append({"measurement_id": mid, "geometry_id": geometry,
                                  "channel0": 100 + index, "channel1": 0 if volume == 28 else 30,
                                  "value": 0.02 * (index + 1)})
        return SimpleNamespace(cells=self.pa.Table.from_pylist(cells),
                               new_hits=self.pa.Table.from_pylist(hits),
                               sample="pu0", new_event=0)

    def test_distinct_sizes_and_deterministic_selection(self):
        event = self.event()
        selected = self.plots.select_cluster_examples(event)
        self.assertEqual(len(selected), 18)
        for _, group in selected.groupby("Detector type", sort=False):
            self.assertEqual(group["Cells"].tolist(), [1, 2, 4, 8, 16, 32])
            self.assertEqual(group["Cells"].nunique(), 6)
        self.assertEqual(selected.iloc[0]["Measurement ID"], 0)
        self.assertEqual(selected.iloc[2]["Measurement ID"], 3)
        self.assertTrue(selected.equals(self.plots.select_cluster_examples(event)))

    def test_nearest_sizes_and_insufficient_family(self):
        event = self.event(sizes=(1, 3, 5), volumes=(16,))
        selected = self.plots.select_cluster_examples(event, (2, 4, 8, 16, 32, 64))
        self.assertEqual(selected["Cells"].tolist(), [1, 3, 5])
        stream = io.StringIO()
        with redirect_stdout(stream), patch.object(self.plots.plt, "show"):
            summary = self.plots.plot_cluster_galleries(event)
        self.assertEqual(len(summary), 3)
        self.assertIn("Pixels: only 3 distinct cluster sizes", stream.getvalue())
        self.assertIn("Long strips: only 0", stream.getvalue())

    def test_pixel_addresses_padding_and_duplicate_sums(self):
        image, extent = self.plots._cluster_cell_image([10, 11, 11], [20, 22, 22], [0.1, 0.2, 0.3])
        self.assertEqual(extent, (8.5, 12.5, 18.5, 23.5))
        self.assertEqual(image.shape, (5, 4))
        self.assertAlmostEqual(float(image[1, 1]), 0.1)
        self.assertAlmostEqual(float(image[3, 2]), 0.5)
        self.assertTrue(image.mask[2, 1])
        self.assertTrue(image.mask[0].all())
        self.assertAlmostEqual(float(image.sum()), 0.6)

    def test_single_cell_and_strip_row(self):
        image, extent = self.plots._cluster_cell_image([327], [59], [0.117])
        self.assertEqual(image.shape, (3, 3))
        self.assertEqual(extent, (325.5, 328.5, 57.5, 60.5))
        self.assertEqual(image.count(), 1)
        image, _ = self.plots._cluster_cell_image([2, 3, 4], [0, 0, 0], [0.1, 0.2, 0.3])
        self.assertEqual(image.shape, (3, 5))
        self.assertTrue(image.mask[0].all() and image.mask[2].all())
        self.assertEqual(image.count(), 3)

    def test_gallery_norms_and_cell_value_sums(self):
        event = self.event()
        with patch.object(self.plots.plt, "show"):
            summary = self.plots.plot_cluster_galleries(event)
        figures = [self.plots.plt.figure(number) for number in self.plots.plt.get_fignums()]
        self.assertEqual(len(figures), 3)
        for figure in figures:
            artists = [ax.images[0] for ax in figure.axes if ax.images]
            self.assertEqual(len(artists), 6)
            self.assertTrue(all(artist.norm is artists[0].norm for artist in artists))
            self.assertEqual(artists[0].norm.vmin, 0)
            self.assertTrue(all(artist.origin == "lower" for artist in artists))
        for row in summary.to_dict("records"):
            n = row["Cells"]
            self.assertAlmostEqual(row["Value sum"], 0.02 * n * (n + 1) / 2)
        self.plots.plt.close("all")
        with patch.object(self.plots.plt, "show"):
            self.plots.plot_cluster_galleries(event, color_scale="log")
        for number in self.plots.plt.get_fignums():
            artists = [ax.images[0] for ax in self.plots.plt.figure(number).axes if ax.images]
            self.assertTrue(all(isinstance(artist.norm, self.plots.LogNorm) for artist in artists))
            self.assertGreater(artists[0].norm.vmin, 0)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct positive integers"):
            self.plots.select_cluster_examples(self.event(), [1, 1])
        with self.assertRaisesRegex(ValueError, "color_scale"):
            self.plots.plot_cluster_galleries(self.event(), color_scale="other")
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            self.plots._cluster_cell_image([1], [0], [self.np.nan])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.plots._cluster_cell_image([], [], [])


if __name__ == "__main__":
    unittest.main()
