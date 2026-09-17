"""Checks for ACTS measurement, cell, and truth-link export."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import uproot


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/postprocessing/export_paired_clusters.py"
SPEC = importlib.util.spec_from_file_location("export_paired_clusters", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
GEOMETRY_ID = (16 << 56) | (2 << 36) | (3 << 8)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PairedClusterExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.csv_dir = self.base / "csv"
        self.csv_dir.mkdir()
        self.prefix = "event000000000-"
        self.measurements = [
            dict(measurement_id=0, geometry_id=GEOMETRY_ID, global_x=1.0,
                 global_y=2.0, global_z=3.0, local0=0.1, local1=0.2),
            dict(measurement_id=1, geometry_id=GEOMETRY_ID, global_x=4.0,
                 global_y=5.0, global_z=6.0, local0=0.3, local1=0.4),
        ]
        self.cells = [
            dict(measurement_id=0, geometry_id=GEOMETRY_ID, channel0=10, channel1=20, value=0.4),
            dict(measurement_id=0, geometry_id=GEOMETRY_ID, channel0=11, channel1=20, value=0.6),
            dict(measurement_id=1, geometry_id=GEOMETRY_ID, channel0=21, channel1=22, value=0.8),
        ]
        self.links = [dict(measurement_id=0, hit_id=7), dict(measurement_id=0, hit_id=8),
                      dict(measurement_id=1, hit_id=9)]
        self.args = argparse.Namespace(
            campaign="full_pileup", dataset="ttbar", version="v1", run=0,
            acts_revision="test-acts", odd_revision="test-odd", csv_dir=self.csv_dir,
            measurements_root=self.base / "measurements.root",
            edm4hep=self.base / "edm4hep.root", digi_config=self.base / "digi.json",
            output=self.base / "export",
        )
        self.args.edm4hep.write_bytes(b"test input")
        self.args.digi_config.write_text(json.dumps({"entries": [{"volume": 16}]}))
        with uproot.recreate(self.args.measurements_root) as root:
            root["measurements"] = {
                "event_nr": np.array([0, 0], dtype=np.int32),
                "volume_id": np.array([16, 16], dtype=np.int32),
                "layer_id": np.array([2, 2], dtype=np.int32),
                "surface_id": np.array([3, 3], dtype=np.int32),
                "rec_gx": np.array([1, 4], dtype=np.float32),
                "rec_gy": np.array([2, 5], dtype=np.float32),
                "rec_gz": np.array([3, 6], dtype=np.float32),
                "clus_size": np.array([2, 1], dtype=np.int32),
            }
        self.write_inputs()

    def write_inputs(self) -> None:
        write_csv(self.csv_dir / f"{self.prefix}measurements.csv",
                  list(self.measurements[0]), self.measurements)
        write_csv(self.csv_dir / f"{self.prefix}cells.csv", list(self.cells[0]), self.cells)
        write_csv(self.csv_dir / f"{self.prefix}measurement-simhit-map.csv",
                  list(self.links[0]), self.links)

    def test_preserves_multiple_contributors_and_cells(self) -> None:
        report = exporter.export(self.args)
        hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
        cells = pq.read_table(self.args.output / "cells.parquet").to_pylist()
        self.assertEqual([hit["simhit_ids"] for hit in hits], [[7, 8], [9]])
        self.assertEqual([cell["channel0"] for cell in cells], [10, 11, 21])
        self.assertEqual(report["multi_contributor_measurements"], 1)
        self.assertEqual((report["events"], report["measurements"], report["cells"]), (1, 2, 3))
        self.assertEqual(len(report["inputs"]), 3)

    def test_rejects_orphan_cell(self) -> None:
        self.cells[0]["measurement_id"] = 99
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "orphan cell"):
            exporter.export(self.args)
        self.assertFalse(self.args.output.exists())

    def test_rejects_different_cluster_size(self) -> None:
        self.cells.pop()
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "cluster size mismatch"):
            exporter.export(self.args)


if __name__ == "__main__":
    unittest.main()
