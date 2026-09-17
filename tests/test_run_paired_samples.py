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
                target = Path(parts[parts.index("--output") + 1])
                target.mkdir(parents=True)
                (target / "report.json").write_text(json.dumps({"events": 2, "local_events": [0, 1]}))

        with patch.object(pipeline, "revision", side_effect=[
                 pipeline.PINNED_ACTS_PREFIX, pipeline.PINNED_ODD_REVISION]), \
             patch.object(pipeline, "run_command", side_effect=fake_command), \
             patch.object(pipeline, "one_parquet", return_value=self.base / "converted.parquet"):
            result = pipeline.run(self.args)

        self.assertEqual(result["events_per_sample"], 2)
        self.assertEqual(len(commands), 6)
        for sample in ("pu0", "pu200"):
            config_path = self.args.output / f"_convert_{sample}.json"
            config = json.loads(config_path.read_text())
            self.assertEqual((config["run_size"], config["chunk_size"]), (2, 2))
            self.assertEqual(Path(config["input_base_dir"]),
                             self.args.output / "_conversion_input" / sample)
            self.assertTrue((self.args.output / "_conversion_input" / sample / "runs/0/edm4hep.root").is_symlink())
        self.assertTrue((self.args.output / "complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
