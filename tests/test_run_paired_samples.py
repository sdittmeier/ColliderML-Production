"""Preflight and orchestration checks for the paired sample command."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import uproot
import pyarrow as pa
import pyarrow.parquet as pq


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/postprocessing/run_paired_samples.py"
SPEC = importlib.util.spec_from_file_location("run_paired_samples", MODULE_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class PairedPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        inputs = []
        for sample in ("pu0", "pu200"):
            path = self.base / f"{sample}.root"
            with uproot.recreate(path) as root:
                root["events"] = {"index": np.arange(2, dtype=np.int32)}
            inputs.append(path)
        odd_install = self.base / "odd-install"
        (odd_install / "lib").mkdir(parents=True)
        (odd_install / "lib/libOpenDataDetector.so").touch()
        self.args = argparse.Namespace(
            input_pu0=inputs[0], input_pu200=inputs[1],
            output=self.base / "result", events=1,
            digi_config=pipeline.DIGI_CONFIG,
            acts_source=self.base, odd_source=self.base, odd_install=odd_install,
        )

    def test_rejects_too_many_events_before_creating_output(self) -> None:
        self.args.events = 3
        with self.assertRaisesRegex(ValueError, "requested 3"):
            pipeline.run(self.args)
        self.assertFalse(self.args.output.exists())

    def write_export_report(self, parts, **overrides):
        target = Path(parts[parts.index("--output") + 1])
        target.mkdir(parents=True, exist_ok=True)
        report = {name: parts[parts.index(f"--{name}") + 1]
                  for name in ("campaign", "dataset", "version", "run")}
        report["run"] = int(report["run"])
        report.update(events=self.args.events, local_events=list(range(self.args.events)))
        report.update(overrides)
        (target / "report.json").write_text(json.dumps(report))

    def prepare_resume(self, legacy=False):
        self.args.resume_export = True
        self.args.output.mkdir()
        for sample, campaign in pipeline.SAMPLES:
            acts = self.args.output / sample / "acts"
            (acts / "csv").mkdir(parents=True)
            for name in ("measurements", "simhits", "particles"):
                (acts / f"{name}.root").touch()
            source = self.args.output / "_conversion_input" / sample / "runs" / "0"
            source.mkdir(parents=True)
            (source / "edm4hep.root").symlink_to(getattr(self.args, f"input_{sample}"))
            for name in ("measurements", "particles"):
                (source / f"{name}.root").symlink_to(acts / f"{name}.root")
            config = {
                "campaign": campaign, "dataset": getattr(self.args, "dataset", "ttbar"),
                "version": getattr(self.args, "version", "v1"),
                "run_size": self.args.events, "chunk_size": self.args.events,
            }
            if not legacy:
                config.update(paired_run=getattr(self.args, "run", 0),
                              paired_digi_config_sha256=pipeline.hashlib.sha256(
                                  self.args.digi_config.read_bytes()).hexdigest())
            (self.args.output / f"_convert_{sample}.json").write_text(json.dumps(config))

    def test_rejects_existing_output(self) -> None:
        self.args.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            pipeline.run(self.args)

    def test_rejects_wrong_geometry_revision_before_creating_output(self) -> None:
        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, "wrong-odd"]):
            with self.assertRaisesRegex(ValueError, "ODD revision"):
                pipeline.run(self.args)
        self.assertFalse(self.args.output.exists())

    def test_builds_both_samples_with_requested_event_count(self) -> None:
        self.args.events = 2
        commands = []

        def fake_command(*parts):
            commands.append(parts)
            if parts[1].endswith("export_paired_clusters.py"):
                self.write_export_report(parts)

        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command", side_effect=fake_command), \
             patch.object(pipeline, "one_parquet", return_value=self.base / "converted.parquet"), \
             patch.object(pipeline, "validate_particle_features"):
            result = pipeline.run(self.args)

        self.assertEqual(result["events_per_sample"], 2)
        self.assertEqual(result["pixel_quantization"], "p99.9")
        self.assertEqual((result["dataset"], result["version"], result["run"]), ("ttbar", "v1", 0))
        self.assertEqual(len(commands), 6)
        self.assertEqual(sum("--simhits-root" in command for command in commands), 2)
        for command in commands:
            if command[1].endswith("export_paired_clusters.py"):
                self.assertEqual(command[command.index("--pixel-quantization") + 1], "p99.9")
        for sample in ("pu0", "pu200"):
            config_path = self.args.output / f"_convert_{sample}.json"
            config = json.loads(config_path.read_text())
            self.assertEqual((config["run_size"], config["chunk_size"]), (2, 2))
            self.assertEqual((config["dataset"], config["version"], config["paired_run"]),
                             ("ttbar", "v1", 0))
            self.assertEqual(Path(config["input_base_dir"]),
                             self.args.output / "_conversion_input" / sample)
            self.assertTrue((self.args.output / "_conversion_input" / sample / "runs/0/edm4hep.root").is_symlink())
            self.assertTrue((self.args.output / "_conversion_input" / sample / "runs/0/particles.root").is_symlink())
        self.assertTrue((self.args.output / "complete.json").is_file())

    def test_resume_reuses_digitization_and_conversion(self) -> None:
        self.args.events = 2
        self.args.pixel_quantization = "p99.5"
        self.args.dataset, self.args.version, self.args.run = "higgs_portal", "v2", 17
        self.prepare_resume()
        commands = []

        def fake_command(*parts):
            commands.append(parts)
            self.write_export_report(parts)

        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command", side_effect=fake_command), \
             patch.object(pipeline, "one_parquet", return_value=self.base / "converted.parquet"), \
             patch.object(pipeline, "validate_particle_features"):
            pipeline.run(self.args)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(command[1].endswith("export_paired_clusters.py") for command in commands))
        self.assertTrue(all(command[command.index("--pixel-quantization") + 1] == "p99.5"
                            for command in commands))
        self.assertTrue((self.args.output / "complete.json").exists())

    def test_higgs_portal_paths_and_nonzero_run_keep_conversion_local(self):
        self.args.dataset, self.args.version, self.args.run = "higgs_portal", "v2", 17
        self.args.events = 2
        commands = []

        def fake_command(*parts):
            commands.append(parts)
            if parts[1].endswith("export_paired_clusters.py"):
                self.write_export_report(parts)

        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command", side_effect=fake_command), \
             patch.object(pipeline, "one_parquet", return_value=self.base / "converted.parquet") as find, \
             patch.object(pipeline, "validate_particle_features"):
            result = pipeline.run(self.args)
        self.assertEqual((result["dataset"], result["version"], result["run"]),
                         ("higgs_portal", "v2", 17))
        self.assertEqual(len(commands), 6)
        expected_paths = []
        for sample, campaign in pipeline.SAMPLES:
            config = json.loads((self.args.output / f"_convert_{sample}.json").read_text())
            self.assertEqual((config["dataset"], config["version"], config["paired_run"]),
                             ("higgs_portal", "v2", 17))
            self.assertTrue((self.args.output / "_conversion_input" / sample / "runs/0/edm4hep.root").is_symlink())
            self.assertFalse((self.args.output / "_conversion_input" / sample / "runs/17").exists())
            base = self.args.output / "_conversion_results" / campaign / "higgs_portal/v2/parquet"
            expected_paths.extend([base / "reco/tracker_hits", base / "truth/particles"])
        self.assertEqual([call.args[0] for call in find.call_args_list], expected_paths)
        for parts in commands:
            if parts[1].endswith("export_paired_clusters.py"):
                for name, expected in (("dataset", "higgs_portal"), ("version", "v2"), ("run", "17")):
                    self.assertEqual(parts[parts.index(f"--{name}") + 1], expected)

    def test_resume_rejects_identity_or_config_mismatch_before_any_export(self):
        self.prepare_resume()
        # A mismatch in pu200 must also prevent overwriting the pu0 export.
        path = self.args.output / "_convert_pu200.json"
        original = json.loads(path.read_text())
        for name, value in (("dataset", "higgs_portal"), ("version", "v2"),
                            ("paired_run", 17), ("campaign", "other"),
                            ("paired_digi_config_sha256", "wrong"), ("run_size", 2)):
            with self.subTest(name=name):
                path.write_text(json.dumps({**original, name: value}))
                with patch.object(pipeline, "revision", side_effect=[
                         pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
                     patch.object(pipeline, "run_command") as command:
                    with self.assertRaisesRegex(ValueError, "do not match"):
                        pipeline.run(self.args)
                    command.assert_not_called()
                self.assertFalse((self.args.output / "complete.json").exists())

    def test_resume_rejects_changed_input_path(self):
        self.prepare_resume()
        self.args.input_pu200 = self.args.input_pu0
        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command") as command:
            with self.assertRaisesRegex(ValueError, "do not match"):
                pipeline.run(self.args)
            command.assert_not_called()

    def test_legacy_resume_is_run_zero_only(self):
        self.prepare_resume(legacy=True)
        self.args.run = 17
        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command") as command:
            with self.assertRaisesRegex(ValueError, "do not match"):
                pipeline.run(self.args)
            command.assert_not_called()
        self.args.run = 0
        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command", side_effect=lambda *parts: self.write_export_report(parts)), \
             patch.object(pipeline, "one_parquet", return_value=self.base / "converted.parquet"), \
             patch.object(pipeline, "validate_particle_features"):
            self.assertEqual(pipeline.run(self.args)["run"], 0)

    def test_missing_particles_root_blocks_resume_before_any_export(self):
        self.prepare_resume()
        (self.args.output / "pu200/acts/particles.root").unlink()
        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command") as command:
            with self.assertRaisesRegex(ValueError, "missing ACTS output"):
                pipeline.run(self.args)
            command.assert_not_called()

    def test_invalid_identity_labels_and_run(self):
        for name, value in (("dataset", "../bad"), ("dataset", ""),
                            ("version", "v 1"), ("version", "/tmp/escape"),
                            ("run", -1), ("run", 2**32)):
            with self.subTest(name=name, value=value), \
                 patch.object(pipeline, "run_command") as command:
                setattr(self.args, name, value)
                with self.assertRaises(ValueError):
                    pipeline.run(self.args)
                command.assert_not_called()
                delattr(self.args, name)
                self.assertFalse(self.args.output.exists())

    def test_non_ttbar_requires_explicit_both_inputs(self):
        self.args.dataset = "higgs_portal"
        self.args.input_pu0 = None
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            pipeline.run(self.args)
        self.assertFalse(self.args.output.exists())

    def test_ttbar_default_input_paths_remain_unchanged(self):
        self.args.input_pu0 = self.args.input_pu200 = None
        commands = []

        def fake_command(*parts):
            commands.append(parts)
            if parts[1].endswith("export_paired_clusters.py"):
                self.write_export_report(parts)

        with patch.object(pipeline, "input_events", return_value=2) as available, \
             patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command", side_effect=fake_command), \
             patch.object(pipeline, "one_parquet", return_value=self.base / "converted.parquet"), \
             patch.object(pipeline, "validate_particle_features"):
            pipeline.run(self.args)
        self.assertEqual([call.args[0] for call in available.call_args_list],
                         [Path("/data/pu0/edm4hep.root"), Path("/data/pu200/edm4hep.root")])

    def test_wrong_export_identity_does_not_mark_run_complete(self):
        def fake_command(*parts):
            if parts[1].endswith("export_paired_clusters.py"):
                self.write_export_report(parts, run=99)
        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command", side_effect=fake_command), \
             patch.object(pipeline, "one_parquet", return_value=self.base / "converted.parquet"), \
             patch.object(pipeline, "validate_particle_features"):
            with self.assertRaisesRegex(ValueError, "identity does not match"):
                pipeline.run(self.args)
        self.assertFalse((self.args.output / "complete.json").exists())

    def test_invalid_quantization_fails_before_running(self) -> None:
        self.args.pixel_quantization = "unknown"
        with patch.object(pipeline, "run_command") as command:
            with self.assertRaisesRegex(ValueError, "Unknown pixel"):
                pipeline.run(self.args)
        command.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_quantized_config_gate_precedes_simulation(self) -> None:
        self.args.digi_config = self.base / "other-digi.json"
        self.args.digi_config.write_text('{"entries": [{"volume": 16}]}')
        with patch.object(pipeline, "run_command") as command:
            with self.assertRaisesRegex(ValueError, "calibrated geometric"):
                pipeline.run(self.args)
        command.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_particle_feature_gate(self) -> None:
        path = self.base / "particles.parquet"
        pq.write_table(pa.table({"particle_id": [[1]]}), path)
        with self.assertRaisesRegex(ValueError, "perigee_d0"):
            pipeline.validate_particle_features(path)
        pq.write_table(pa.table({"particle_id": [[1]], "perigee_d0": [[0.1]],
                                 "perigee_z0": [[0.2]], "vertex_primary": [[0]]}), path)
        pipeline.validate_particle_features(path)
        pq.write_table(pa.table({"particle_id": [[1]], "perigee_d0": [[None]],
                                 "perigee_z0": [[0.2]], "vertex_primary": [[0]]}), path)
        with self.assertRaisesRegex(ValueError, "perigee_d0 has no non-null"):
            pipeline.validate_particle_features(path)


if __name__ == "__main__":
    unittest.main()
