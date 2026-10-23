"""Frozen production presets, boundaries, masks, provenance and native preservation."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.postprocessing.utils.pixel_quantization import (
    DEFAULT_SCHEME, PRESETS_PATH, SCHEMES, PixelQuantizer, add_quantization_argument)


DIGI_CONFIG = Path(__file__).resolve().parents[1] / "scripts/simulation/odd-full-geo-digi-config.json"
EDGES = {"p99.9": .16547740542600492, "p99.5": .14173012115,
         "p99": .13679915850000002}


class ProductionPixelQuantizationTest(unittest.TestCase):
    def test_frozen_presets_and_notebook_equivalent_codes(self):
        for scheme, edge in EDGES.items():
            with self.subTest(scheme=scheme):
                quantizer = PixelQuantizer(scheme)
                expected = np.linspace(.01, edge, 15)[1:]
                np.testing.assert_array_equal(quantizer.boundaries, expected)
                values = np.r_[np.linspace(.01, 1, 5000), expected, 11.6826]
                np.testing.assert_array_equal(quantizer.codes(values),
                    1 + np.searchsorted(expected, values, side="right"))
                self.assertEqual(quantizer.codes(values).dtype, np.uint8)
                self.assertEqual(quantizer.boundaries[-1], edge)
                self.assertEqual(quantizer.config["calibration"]["local_events"], list(range(25)))

    def test_exact_boundaries_tie_up_and_overflow_clamps(self):
        for scheme in EDGES:
            q = PixelQuantizer(scheme)
            for index, boundary in enumerate(q.boundaries):
                np.testing.assert_array_equal(q.codes([
                    np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)]),
                    [index + 1, index + 2, index + 2])
            np.testing.assert_array_equal(q.codes([.01, q.boundaries[-1], q.code15_bin_end, 1e300]),
                                          [1, 15, 15, 15])

    def test_pixel_only_and_no_native_changes(self):
        volumes = np.array([16, 17, 18, 23, 24, 25, 28, 29, 30, 15, 19], dtype=np.uint64)
        geometry = (volumes << 56) | 123
        values = np.linspace(.01, .2, len(volumes))
        original_values, original_geometry = values.copy(), geometry.copy()
        q = PixelQuantizer()
        codes = q.encode(geometry, values)
        np.testing.assert_array_equal(codes[:3], q.codes(values[:3]))
        np.testing.assert_array_equal(codes[3:], 0)
        np.testing.assert_array_equal(values, original_values)
        np.testing.assert_array_equal(geometry, original_geometry)
        report = q.report()
        self.assertEqual(report["pixel_cells"], 3)
        self.assertEqual(report["unchanged_nonpixel_cells"], 8)
        self.assertEqual(sum(report["pixel_code_counts"].values()), 3)
        self.assertTrue(report["native_values_preserved"])

    def test_diagnostics_accumulate_and_distinguish_code15_from_clipping(self):
        q = PixelQuantizer()
        values = [.01, q.boundaries[-1], np.nextafter(q.code15_bin_end, -np.inf),
                  q.code15_bin_end, 100]
        for _ in range(2):
            q.encode(np.full(5, 16 << 56, dtype=np.uint64), values)
        report = json.loads(json.dumps(q.report()))
        self.assertEqual(report["pixel_cells"], 10)
        self.assertEqual(report["code15_cells"], 8)
        self.assertEqual(report["genuinely_clipped_cells"], 4)
        self.assertEqual(report["preset_sha256"], hashlib.sha256(PRESETS_PATH.read_bytes()).hexdigest())

    def test_native_mode_and_empty_events(self):
        q = PixelQuantizer("none")
        np.testing.assert_array_equal(q.encode([16 << 56, 23 << 56], [.001, 3.2]), [0, 0])
        self.assertFalse(q.report()["enabled"])
        self.assertEqual(q.report()["scheme"], "none")
        self.assertEqual(len(q.encode([], [])), 0)
        self.assertEqual(len(PixelQuantizer().encode([], [])), 0)
        with self.assertRaisesRegex(ValueError, "Native mode"):
            q.codes([.01])

    def test_invalid_inputs_and_scheme_fail(self):
        q = PixelQuantizer()
        for values in ([np.nan], [np.inf], [-1], [0], [.009999], [[.1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                q.codes(values)
        with self.assertRaisesRegex(ValueError, "aligned"):
            q.encode([16 << 56], [.1, .2])
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            q.encode([23 << 56], [np.nan])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            PixelQuantizer("P99.9")

    def test_geometry_config_gate_and_cli(self):
        PixelQuantizer().validate_digi_config(DIGI_CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrong.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "calibrated geometric"):
                PixelQuantizer().validate_digi_config(path)
            PixelQuantizer("none").validate_digi_config(path)
        parser = argparse.ArgumentParser()
        add_quantization_argument(parser)
        self.assertEqual(parser.parse_args([]).pixel_quantization, DEFAULT_SCHEME)
        for scheme in SCHEMES:
            self.assertEqual(parser.parse_args(["--pixel-quantization", scheme]).pixel_quantization, scheme)


if __name__ == "__main__":
    unittest.main()
