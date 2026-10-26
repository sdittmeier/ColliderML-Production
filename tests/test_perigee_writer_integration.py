"""Configuration and provenance checks without requiring the ACTS runtime."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.postprocessing.export_paired_clusters import perigee_writer_provenance


class PerigeeWriterIntegrationTest(unittest.TestCase):
    def add_writers(self, patched=True):
        source = Path(__file__).resolve().parents[1] / "scripts/simulation/digi_and_reco.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "add_root_writers")
        writer = MagicMock()
        writer.Config.side_effect = lambda **kwargs: SimpleNamespace(**dict(
            {"perigeeRecoveryMaxR": 1e300, "perigeeRecoveryMaxAbsZ": 1e300} if patched else {}, **kwargs))
        namespace = {"acts": SimpleNamespace(examples=SimpleNamespace(root=SimpleNamespace(
            RootParticleWriter=writer)), Vector3=lambda *v: v,
            logging=SimpleNamespace(WARNING=2)), "LOG_LEVEL": 2}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["add_root_writers"], writer

    def test_recovery_limits_reach_writer(self):
        add, writer = self.add_writers()
        add(MagicMock(), Path("/output"), object(), SimpleNamespace(
            output_particles_root=True, perigee_recovery_max_r=1160., perigee_recovery_max_abs_z=3000.))
        self.assertEqual(writer.call_args.kwargs["config"].perigeeRecoveryMaxR, 1160.)
        self.assertEqual(writer.call_args.kwargs["config"].perigeeRecoveryMaxAbsZ, 3000.)

    def test_old_runtime_fails_clearly_when_recovery_limits_requested(self):
        add, _ = self.add_writers(patched=False)
        with self.assertRaisesRegex(RuntimeError, "recovery-only ACTS RootParticleWriter"):
            add(MagicMock(), Path("/output"), object(), SimpleNamespace(
                output_particles_root=True, perigee_recovery_max_r=1160., perigee_recovery_max_abs_z=3000.))

    def test_provenance_comes_from_input_root(self):
        metadata = {"algorithm": "writer-recovery-v2", "patch_sha256": "file-patch"}
        source = MagicMock()
        source.__contains__.return_value = True
        source.__getitem__.return_value.member.return_value = json.dumps(metadata)
        with patch("scripts.postprocessing.export_paired_clusters.uproot.open") as opened:
            opened.return_value.__enter__.return_value = source
            self.assertEqual(perigee_writer_provenance(Path("/old/particles.root")), metadata)
            source.__getitem__.return_value.member.assert_called_once_with("fTitle")
            source.__contains__.return_value = False
            self.assertIsNone(perigee_writer_provenance(Path("/older/particles.root")))


if __name__ == "__main__":
    unittest.main()
