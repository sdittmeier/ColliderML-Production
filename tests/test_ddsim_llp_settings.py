"""Test DDSim configuration without importing the container-only physics runtime."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import traceback
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import sys


SOURCE = Path(__file__).resolve().parents[1] / "scripts/simulation/ddsim_run.py"


def function(name, scope):
    node = next(node for node in ast.parse(SOURCE.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[name]


class DDSimLLPSettingsTest(unittest.TestCase):
    def test_single_thread_writer_is_not_shared_and_keeps_metadata(self):
        kernel, action = Mock(), Mock()
        ddg4 = SimpleNamespace(Kernel=Mock(), EventAction=Mock(return_value=action))
        ddg4.Kernel.return_value.worker.return_value = kernel
        simulation = SimpleNamespace(outputFile="event.root", meta=Mock())
        simulation.meta.runNumberOffset = -1
        simulation.meta.eventNumberOffset = 3
        simulation.meta.parseEventParameters.return_value = ({"a": "x"}, {"b": "1"}, {"c": "2"})
        with patch.dict(sys.modules, {"DDG4": ddg4}):
            function("single_thread_edm4hep_output", {})(simulation)
        ddg4.EventAction.assert_called_once_with(kernel, "Geant4Output2EDM4hep/EDM4hepOutput", False)
        kernel.eventAction.return_value.add.assert_called_once_with(action)
        self.assertEqual(action.Output, "event.root")
        self.assertEqual(action.EventParametersString, {"a": "x"})
        self.assertEqual(action.RunNumberOffset, 0)
        self.assertEqual(action.EventNumberOffset, 3)
        self.assertIs(action.RunHeader, simulation.meta.addParametersToRunHeader.return_value)

    def test_alternative_status_and_union_with_existing_rejections(self):
        configure = function("configure_physics", {"GeV": 1})
        physics = SimpleNamespace(rejectPDGs={1, 2, 21, 25}, alternativeDecayStatuses=set())
        simulation = SimpleNamespace(physics=physics, part=SimpleNamespace())
        config = SimpleNamespace(alternative_decay_statuses=[22], extra_reject_pdgs=[35],
                                 pdg_file="test.tbl", minimalKineticEnergy=.1,
                                 truthParticleHandler="Geant4FullTruthParticleHandler")
        configure(simulation, config, Mock())
        self.assertEqual(physics.alternativeDecayStatuses, {22})
        self.assertEqual(physics.rejectPDGs, {1, 2, 21, 25, 35})
        self.assertEqual(physics.pdgfile, "test.tbl")
        self.assertEqual(simulation.part.minimalKineticEnergy, .1)

    def test_existing_configuration_defaults_do_not_change(self):
        configure = function("configure_physics", {"GeV": 1})
        physics = SimpleNamespace(rejectPDGs={21}, alternativeDecayStatuses={7})
        configure(SimpleNamespace(physics=physics, part=SimpleNamespace()), SimpleNamespace(), Mock())
        self.assertEqual(physics.rejectPDGs, {21})
        self.assertEqual(physics.alternativeDecayStatuses, {7})

    def test_simulation_failure_propagates_and_timer_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output=Path(temp), input_file=Path("input.hepmc3"))
            timer = Mock()
            timer.record.return_value = nullcontext()
            scope = dict(parse_args=lambda: args, load_config=lambda _: SimpleNamespace(),
                         Path=Path, setup_logging=Mock(), TimingRecorder=lambda _: timer,
                         run_ddsim=Mock(side_effect=RuntimeError("simulation failed")), traceback=traceback)
            main = function("main", scope)
            with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                main()
            timer.write_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
