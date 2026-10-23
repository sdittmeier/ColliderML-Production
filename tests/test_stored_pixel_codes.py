"""Independent checks of stored codes, unchanged data, reports and streaming alignment."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from notebooks.validation import paired_vs_published as plots
from scripts.postprocessing.utils.pixel_quantization import METADATA_KEY, PixelQuantizer


class StoredPixelCodesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(plots.plt.close, "all")
        self.old = Path(self.temp.name) / "native"
        self.new = Path(self.temp.name) / "encoded"
        self.make_fixture()

    def make_fixture(self, scheme="p99.9"):
        native_reports, encoded_reports = {}, {}
        for sample in plots.PILEUP_COLORS:
            for root in (self.old, self.new):
                (root / sample).mkdir(parents=True, exist_ok=True)
            original = pa.table({
                "campaign": [sample] * 6, "dataset": ["ttbar"] * 6, "version": ["v1"] * 6,
                "run": [0] * 6, "local_event": [0, 0, 0, 1, 1, 1],
                "measurement_id": [0, 0, 1, 0, 0, 1],
                "geometry_id": pa.array([16 << 56, 17 << 56, 23 << 56,
                                         18 << 56, 16 << 56, 30 << 56], type=pa.uint64()),
                "channel0": [1, 2, 10, 1, 2, 30], "channel1": [3, 3, 5, 4, 4, 0],
                "value": [.01, .03, .15, .12, .9, .25],
            })
            q = PixelQuantizer(scheme)
            encoded = q.encode(original["geometry_id"].to_numpy(), original["value"].to_numpy())
            codes = pa.array(encoded, mask=encoded == 0, type=pa.uint8())
            derived = original.append_column("pixel_code", codes).replace_schema_metadata(
                {METADATA_KEY: json.dumps(q.report()).encode()})
            pq.write_table(original, self.old / sample / "cells.parquet", row_group_size=2)
            pq.write_table(derived, self.new / sample / "cells.parquet", row_group_size=3)
            report = dict(events=2, local_events=[0, 1], measurements=4, cells=6, particles=2)
            for kind in ("hits", "particles"):
                pq.write_table(pa.table({"id": [1, 2]}), self.old / sample / f"{kind}.parquet")
                shutil.copyfile(self.old / sample / f"{kind}.parquet",
                                self.new / sample / f"{kind}.parquet")
            (self.old / sample / "report.json").write_text(json.dumps(report))
            native_reports[sample] = report
            report = {**report, "cell_quantization": q.report()}
            (self.new / sample / "report.json").write_text(json.dumps(report))
            encoded_reports[sample] = report
        (self.old / "complete.json").write_text(json.dumps(
            {"events_per_sample": 2, "samples": native_reports}))
        (self.new / "complete.json").write_text(json.dumps(
            {"events_per_sample": 2, "pixel_quantization": scheme, "samples": encoded_reports}))

    def run_check(self, **kwargs):
        return plots.validate_stored_pixel_codes(self.new, self.old, batch_size=2,
                                                 progress=None, **kwargs)

    def edit_cells(self, name, values):
        path = self.new / "pu0/cells.parquet"
        table = pq.read_table(path)
        column = table.schema.get_field_index(name)
        replacement = pa.array(values, type=table.schema.field(name).type)
        table = table.set_column(column, table.schema.field(name), replacement)
        pq.write_table(table, path)

    def test_all_presets_and_native_checksums_preserved(self):
        for scheme in ("p99.9", "p99.5", "p99"):
            self.make_fixture(scheme)
            before = plots.quantization_source_checksums(self.old)
            result = self.run_check(expected_scheme=scheme)
            self.assertEqual(result.summary.events.tolist(), [2, 2])
            self.assertEqual(result.summary.pixel_cells.tolist(), [4, 4])
            self.assertEqual(result.summary.null_nonpixel_codes.tolist(), [2, 2])
            self.assertTrue(result.summary.all_codes_match.all())
            self.assertEqual(len(result.code_counts), 30)
            self.assertEqual(len(result.source_checksums), 12)
            self.assertEqual(before, plots.quantization_source_checksums(self.old))
            with patch.object(plots.plt, "show"):
                plots.plot_stored_pixel_codes(result)

    def test_rejects_native_value_or_full_key_change(self):
        self.edit_cells("value", [.01, .031, .15, .12, .9, .25])
        with self.assertRaisesRegex(ValueError, "native cell keys"):
            self.run_check()
        self.make_fixture()
        self.edit_cells("measurement_id", [0, 99, 1, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "native cell keys"):
            self.run_check()

    def test_rejects_wrong_code_and_nonpixel_code(self):
        self.edit_cells("pixel_code", [2, 2, None, 10, 15, None])
        with self.assertRaisesRegex(ValueError, "disagree with frozen"):
            self.run_check()
        self.make_fixture()
        self.edit_cells("pixel_code", [1, 2, 1, 10, 15, None])
        with self.assertRaisesRegex(ValueError, "non-null strip"):
            self.run_check()

    def test_rejects_missing_pixel_code(self):
        self.edit_cells("pixel_code", [None, 2, None, 10, 15, None])
        with self.assertRaisesRegex(ValueError, "missing pixel"):
            self.run_check()

    def test_rejects_changed_hit_file(self):
        pq.write_table(pa.table({"id": [2, 1]}), self.new / "pu0/hits.parquet")
        with self.assertRaisesRegex(ValueError, "hits file changed"):
            self.run_check()

    def test_rejects_wrong_manifest_and_metadata(self):
        path = self.new / "complete.json"
        manifest = json.loads(path.read_text())
        manifest["pixel_quantization"] = "p99"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "wrong quantization"):
            self.run_check()
        self.make_fixture()
        cell_path = self.new / "pu0/cells.parquet"
        pq.write_table(pq.read_table(cell_path).replace_schema_metadata({}), cell_path)
        with self.assertRaisesRegex(ValueError, "metadata and report"):
            self.run_check()

    def test_aligns_different_row_groups_and_batch_boundaries(self):
        for size in (1, 2, 3, 5, 100):
            a = pq.ParquetFile(self.old / "pu0/cells.parquet")
            b = pq.ParquetFile(self.new / "pu0/cells.parquet")
            pairs = list(plots._paired_parquet_batches(a, b, ["value", "geometry_id"], size))
            self.assertEqual(sum(old.num_rows for old, _ in pairs), 6)
            self.assertTrue(all(old.equals(new) for old, new in pairs))


if __name__ == "__main__":
    unittest.main()
