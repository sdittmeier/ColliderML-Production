"""Checks for ACTS measurement, cell, and truth-link export."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
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
            pixel_quantization="none",
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

    def write_converted(self, labels=(4, 5)) -> None:
        self.args.converted_hits = self.base / "converted_hits.parquet"
        self.args.particles = self.base / "particles.parquet"
        pq.write_table(pa.table({
            "event_id": [0], "x": [[1.0, 4.0]], "y": [[2.0, 5.0]],
            "z": [[3.0, 6.0]], "volume_id": [[16, 16]],
            "layer_id": [[2, 2]], "surface_id": [[3, 3]],
            "particle_id": [list(labels)],
        }), self.args.converted_hits)
        pq.write_table(pa.table({"event_id": [0], "particle_id": [[4, 5, 6]]}),
                       self.args.particles)

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

    def test_adds_pixel_codes_without_changing_native_features(self) -> None:
        self.args.digi_config = MODULE_PATH.parents[2] / "scripts/simulation/odd-full-geo-digi-config.json"
        strip_geometry = (23 << 56) | (2 << 36) | (3 << 8)
        self.measurements[1]["geometry_id"] = strip_geometry
        self.cells[2]["geometry_id"] = strip_geometry
        self.cells[0]["value"] = .01
        self.cells[1]["value"] = .12
        self.write_inputs()
        roots = exporter.root_measurements(self.args.measurements_root)
        roots[0][1]["volume_id"] = 23
        self.write_converted(labels=(4, None))
        # Converted geometry must match the same pixel/strip event, too.
        converted = pq.read_table(self.args.converted_hits).to_pydict()
        converted["volume_id"] = [[16, 23]]
        pq.write_table(pa.table(converted), self.args.converted_hits)
        reference_hits = None
        for scheme in ("none", "p99.9", "p99.5", "p99"):
            self.args.pixel_quantization = scheme
            self.args.output = self.base / scheme
            with patch.object(exporter, "root_measurements", return_value=roots):
                report = exporter.export(self.args)
            table = pq.read_table(self.args.output / "cells.parquet")
            cells = table.to_pylist()
            self.assertEqual(table.schema.field("pixel_code").type, pa.uint8())
            for output, source in zip(cells, self.cells):
                for name in source:
                    self.assertEqual(output[name], source[name])
            self.assertIsNone(cells[2]["pixel_code"])
            if scheme == "none":
                self.assertTrue(all(cell["pixel_code"] is None for cell in cells))
            else:
                expected = exporter.PixelQuantizer(scheme).codes([.01, .12]).tolist()
                self.assertEqual([cell["pixel_code"] for cell in cells[:2]], expected)
                self.assertEqual(sum(report["cell_quantization"]["pixel_code_counts"].values()), 2)
            encoding = json.loads(table.schema.metadata[exporter.METADATA_KEY])
            self.assertEqual(encoding, report["cell_quantization"])
            self.assertEqual(encoding["scheme"], scheme)
            self.assertEqual(encoding["pixel_cells"], 2)
            self.assertEqual(encoding["unchanged_nonpixel_cells"], 1)
            hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
            if reference_hits is None:
                reference_hits = hits
            self.assertEqual(hits, reference_hits)
            self.assertEqual((self.args.output / "particles.parquet").read_bytes(),
                             self.args.particles.read_bytes())

    def test_default_codes_and_config_gate_before_output(self) -> None:
        del self.args.pixel_quantization
        with self.assertRaisesRegex(ValueError, "calibrated geometric"):
            exporter.export(self.args)
        self.assertFalse(self.args.output.exists())
        self.args.digi_config = MODULE_PATH.parents[2] / "scripts/simulation/odd-full-geo-digi-config.json"
        report = exporter.export(self.args)
        self.assertEqual(report["cell_quantization"]["scheme"], "p99.9")
        self.assertEqual(pq.read_table(self.args.output / "cells.parquet")["pixel_code"].to_pylist(),
                         [15, 15, 15])

    def test_rejects_different_cluster_size(self) -> None:
        self.cells.pop()
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "cluster size mismatch"):
            exporter.export(self.args)

    def test_adds_single_labels_and_copies_particles(self) -> None:
        self.write_converted(labels=(4, None))
        report = exporter.export(self.args)
        hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
        self.assertEqual([hit["particle_id"] for hit in hits], [4, None])
        self.assertEqual([hit["simhit_ids"] for hit in hits], [[7, 8], [9]])
        self.assertEqual(report["null_particle_labels"], 1)
        self.assertEqual(report["particles"], 3)
        self.assertEqual((self.args.output / "particles.parquet").read_bytes(),
                         self.args.particles.read_bytes())

    def test_rejects_shuffled_converted_hit(self) -> None:
        self.write_converted()
        table = pq.read_table(self.args.converted_hits).to_pydict()
        table["x"] = [[4.0, 1.0]]
        pq.write_table(pa.table(table), self.args.converted_hits)
        with self.assertRaisesRegex(ValueError, "converted position differs"):
            exporter.export(self.args)
        self.assertFalse(self.args.output.exists())

    def test_nonzero_source_runs_preserve_local_particle_and_measurement_ids(self) -> None:
        self.args.dataset, self.args.version = "higgs_portal", "v2"
        self.write_converted()
        keys = []
        for run in (17, 18):
            self.args.run = run
            self.args.output = self.base / f"run{run}"
            report = exporter.export(self.args)
            self.assertEqual((report["dataset"], report["version"], report["run"]),
                             ("higgs_portal", "v2", run))
            hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
            cells = pq.read_table(self.args.output / "cells.parquet").to_pylist()
            self.assertEqual([hit["local_event"] for hit in hits], [0, 0])
            self.assertEqual([hit["measurement_id"] for hit in hits], [0, 1])
            self.assertTrue(all(cell["run"] == run for cell in cells))
            self.assertEqual(pq.read_table(self.args.output / "particles.parquet")["event_id"].to_pylist(), [0])
            keys.append({tuple(hit[name] for name in exporter.KEY_FIELDS) for hit in hits})
        self.assertTrue(keys[0].isdisjoint(keys[1]))

    def test_rejects_unknown_particle_label(self) -> None:
        self.write_converted(labels=(4, 99))
        with self.assertRaisesRegex(ValueError, "absent from particles"):
            exporter.export(self.args)
        self.assertFalse(self.args.output.exists())

    def test_rejects_missing_converted_event(self) -> None:
        self.write_converted()
        pq.write_table(pa.table({"event_id": [1], "particle_id": [[4, 5]]}),
                       self.args.particles)
        with self.assertRaisesRegex(ValueError, "missing converted hits or particles"):
            exporter.export(self.args)

    def test_two_events_keep_event_local_keys(self) -> None:
        second = "event000000001-"
        write_csv(self.csv_dir / f"{second}measurements.csv",
                  list(self.measurements[0]), self.measurements)
        write_csv(self.csv_dir / f"{second}cells.csv", list(self.cells[0]), self.cells)
        write_csv(self.csv_dir / f"{second}measurement-simhit-map.csv",
                  list(self.links[0]), self.links)
        with uproot.recreate(self.args.measurements_root) as root:
            root["measurements"] = {
                "event_nr": np.array([0, 0, 1, 1], dtype=np.int32),
                "volume_id": np.full(4, 16, dtype=np.int32),
                "layer_id": np.full(4, 2, dtype=np.int32),
                "surface_id": np.full(4, 3, dtype=np.int32),
                "rec_gx": np.array([1, 4, 1, 4], dtype=np.float32),
                "rec_gy": np.array([2, 5, 2, 5], dtype=np.float32),
                "rec_gz": np.array([3, 6, 3, 6], dtype=np.float32),
                "clus_size": np.array([2, 1, 2, 1], dtype=np.int32),
            }
        self.write_converted()
        hits = pq.read_table(self.args.converted_hits).to_pydict()
        particles = pq.read_table(self.args.particles).to_pydict()
        for table in (hits, particles):
            for name, values in table.items():
                table[name] = values * 2 if name != "event_id" else [0, 1]
        pq.write_table(pa.table(hits), self.args.converted_hits)
        pq.write_table(pa.table(particles), self.args.particles)
        report = exporter.export(self.args)
        output_hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
        self.assertEqual(report["events"], 2)
        self.assertEqual((report["measurements"], report["cells"]), (4, 6))
        self.assertEqual([(hit["local_event"], hit["measurement_id"]) for hit in output_hits],
                         [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_resolves_multiple_particles_per_measurement(self) -> None:
        self.links = [dict(measurement_id=0, hit_id=0),
                      dict(measurement_id=0, hit_id=1),
                      dict(measurement_id=1, hit_id=2)]
        self.write_inputs()
        self.write_converted(labels=(None, 5))
        self.args.simhits_root = self.base / "simhits.root"
        with uproot.recreate(self.args.simhits_root) as root:
            root["hits"] = {
                "event_id": np.zeros(3, dtype=np.int32),
                "tx": np.array([11, 12, 13], dtype=np.float32),
                "ty": np.array([21, 22, 23], dtype=np.float32),
                "tz": np.array([31, 32, 33], dtype=np.float32),
                "barcode_vertex_primary": np.zeros(3, dtype=np.uint32),
                "barcode_vertex_secondary": np.zeros(3, dtype=np.uint32),
                "barcode_particle": np.array([4, 5, 5], dtype=np.uint32),
                "barcode_generation": np.zeros(3, dtype=np.uint32),
                "barcode_sub_particle": np.zeros(3, dtype=np.uint32),
            }
        edm_hits = [(11, 21, 31, 4), (12, 22, 32, 5), (13, 23, 33, 5)]
        with patch.object(exporter, "edm_tracker_hits", return_value=edm_hits):
            report = exporter.export(self.args)
        hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
        self.assertEqual([hit["particle_ids"] for hit in hits], [[4, 5], [5]])
        self.assertEqual([hit["particle_id"] for hit in hits], [None, 5])
        self.assertEqual([hit["particle_ids_complete"] for hit in hits], [True, True])
        self.assertEqual(report["multi_particle_measurements"], 1)
        self.assertEqual(report["null_particle_labels"], 1)

        with patch.object(exporter, "edm_tracker_hits", return_value=edm_hits[1:]):
            report = exporter.export(self.args)
        hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
        self.assertEqual([hit["particle_ids"] for hit in hits], [[5], [5]])
        self.assertEqual([hit["particle_id"] for hit in hits], [None, 5])
        self.assertEqual([hit["particle_ids_complete"] for hit in hits], [False, True])
        self.assertEqual([hit["unresolved_simhit_ids"] for hit in hits], [[0], []])
        self.assertEqual(report["unresolved_simhit_links"], 1)
        self.assertEqual(report["incomplete_truth_measurements"], 1)

        self.write_converted(labels=(None, 4))
        with patch.object(exporter, "edm_tracker_hits", return_value=edm_hits):
            report = exporter.export(self.args)
        hits = pq.read_table(self.args.output / "hits.parquet").to_pylist()
        self.assertEqual(hits[1]["converted_particle_id"], 4)
        self.assertEqual(hits[1]["particle_id"], 5)
        self.assertEqual(report["converted_label_disagreements"], 1)
        self.assertEqual(report["converted_label_disagreement_examples"][0]["measurement_id"], 1)

    def test_barcode_fallback_resolves_missing_coordinate(self) -> None:
        rows = [
            {"tx": 1, "ty": 2, "tz": 3, "barcode_particle": 7},
            {"tx": 4, "ty": 5, "tz": 6, "barcode_particle": 7},
        ]
        for row in rows:
            row.update(barcode_vertex_primary=0, barcode_vertex_secondary=0,
                       barcode_generation=0, barcode_sub_particle=0)
        mapping, unresolved, fallbacks = exporter.simhit_particle_map(0, rows, [(1, 2, 3, 42)], {0, 1})
        self.assertEqual(mapping, {0: 42, 1: 42})
        self.assertEqual(unresolved, set())
        self.assertEqual(fallbacks, 1)

    def test_rejects_conflicting_barcode_truth(self) -> None:
        rows = [
            {"tx": 1, "ty": 2, "tz": 3, "barcode_particle": 7},
            {"tx": 4, "ty": 5, "tz": 6, "barcode_particle": 7},
        ]
        for row in rows:
            row.update(barcode_vertex_primary=0, barcode_vertex_secondary=0,
                       barcode_generation=0, barcode_sub_particle=0)
        with self.assertRaisesRegex(ValueError, "conflicting EDM4hep particles"):
            exporter.simhit_particle_map(0, rows, [(1, 2, 3, 42), (4, 5, 6, 43)], {0, 1})

    def test_marks_unresolved_simhit(self) -> None:
        row = {"tx": 1, "ty": 2, "tz": 3, "barcode_particle": 7,
               "barcode_vertex_primary": 0, "barcode_vertex_secondary": 0,
               "barcode_generation": 0, "barcode_sub_particle": 0}
        mapping, unresolved, fallbacks = exporter.simhit_particle_map(0, [row], [], {0})
        self.assertEqual((mapping, unresolved, fallbacks), ({}, {0}, 0))


if __name__ == "__main__":
    unittest.main()
