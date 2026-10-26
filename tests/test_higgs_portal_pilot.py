"""Pilot provenance, fresh-background reservations and displaced-decay gates."""

import importlib.util
import ast
import json
import math
import numpy as np
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import awkward as ak
import uproot

from scripts.cli.prepare_higgs_portal_inputs import verify
from scripts.simulation import higgs_portal_pilot as pilot


HAS_HEPMC = importlib.util.find_spec("pyhepmc") is not None


class GenerationFileLookupTest(unittest.TestCase):
    def test_explicit_paths_from_json_and_cli(self):
        source = pilot.REPO / "scripts/simulation/pythia_gen.py"
        nodes = ast.parse(source.read_text()).body
        scope = {"Path": Path}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.hepmc3"
            path.touch()
            for name in ("find_hard_scatter_file", "find_pileup_file"):
                node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
                for explicit in (str(path), path):
                    self.assertEqual(scope[name](Path(folder), SimpleNamespace(), explicit), path)


class Geant4DataPathsTest(unittest.TestCase):
    def test_common_directory_resolves_and_exports_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            names = ("G4LEDATA", "G4LEVELGAMMADATA", "G4PARTICLEXSDATA", "G4ENSDFSTATEDATA")
            for name in names:
                dataset = Path(folder) / name
                dataset.mkdir()
                (dataset / "data").write_text("test")
            manifest = "\n".join(f"dataset {name} /installed/{name}" for name in names)
            with patch.dict(os.environ, {"GEANT4_DATA_DIR": folder}, clear=True), \
                 patch.object(pilot.shutil, "which", return_value="/bin/geant4-config"), \
                 patch.object(pilot.subprocess, "check_output", return_value=manifest):
                paths = pilot.geant4_data_paths()
                self.assertEqual(len(paths), 4)
                for name in names:
                    self.assertEqual(os.environ[name], str(Path(folder) / name))
                with patch.dict(os.environ, {"G4ENSDFSTATEDATA": "/invalid"}):
                    with self.assertRaisesRegex(ValueError, "G4ENSDFSTATEDATA=/invalid"):
                        pilot.geant4_data_paths()
                (Path(folder) / "G4LEDATA" / "data").unlink()
                with self.assertRaisesRegex(ValueError, "G4LEDATA"):
                    pilot.geant4_data_paths()

    def test_missing_manifest_is_rejected(self):
        with patch.object(pilot.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "geant4-config is missing"):
                pilot.geant4_data_paths()
        with patch.object(pilot.shutil, "which", return_value="/bin/geant4-config"), \
             patch.object(pilot.subprocess, "check_output", return_value=""):
            with self.assertRaisesRegex(ValueError, "lacks required"):
                pilot.geant4_data_paths()


@unittest.skipUnless(HAS_HEPMC, "requires pyhepmc")
class HiggsPortalPilotTest(unittest.TestCase):
    def setUp(self):
        import pyhepmc
        self.hep = pyhepmc
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def signal(self, path, numbers=(0, 1)):
        h = self.hep
        with h.open(str(path), "w") as writer:
            for number in numbers:
                event = h.GenEvent(h.Units.GEV, h.Units.MM)
                event.event_number = number
                higgs = h.GenParticle(h.FourVector(0, 0, 0, 125), 35, 22)
                production = h.GenVertex(h.FourVector(0, 0, 0, 0))
                production.add_particle_in(h.GenParticle(h.FourVector(0, 0, 6500, 6500), 2212, 4))
                production.add_particle_in(h.GenParticle(h.FourVector(0, 0, -6500, 6500), 2212, 4))
                production.add_particle_out(higgs)
                event.add_vertex(production)
                vertex = h.GenVertex(h.FourVector(0, 0, 0, 0))
                vertex.add_particle_in(higgs)
                event.add_vertex(vertex)
                for sign in (1, -1):
                    llp = h.GenParticle(h.FourVector(sign * 30, 0, 0, (30**2 + 55**2)**.5), 36, 22)
                    llp.generated_mass = 55
                    vertex.add_particle_out(llp)
                    decay = h.GenVertex(h.FourVector(sign * (20 + number), 0, 4, 25))
                    decay.add_particle_in(llp)
                    child = h.GenParticle(h.FourVector(sign, .2, .3, 1.1), sign * 211, 1)
                    decay.add_particle_out(child)
                    event.add_vertex(decay)
                writer.write(event)

    def edm(self, path, numbers=(1, 0), collapsed=False, bad_endpoint=False, daughter_adjustment=0.):
        fields = {"PDG": [36, 36, 211, -211], "generatorStatus": [22, 22, 1, 1],
                  "charge": [0., 0., 1., -1.], "momentum.x": [30., -30., 1., -1.],
                  "momentum.y": [0., 0., .2, .2], "momentum.z": [0., 0., .3, .3]}
        fields["momentum.x"][2] += daughter_adjustment
        data = {"EventHeader/EventHeader.eventNumber": ak.Array([[n] for n in numbers])}
        for name, values in fields.items():
            data["MCParticles/MCParticles." + name] = ak.Array([values for _ in numbers])
        for axis in "xyz":
            vertices, endpoints = [], []
            for number in numbers:
                v = [0., 0., 20. + number, -20. - number] if axis == "x" else [0., 0., 4., 4.] if axis == "z" else [0.] * 4
                if collapsed:
                    v = [0.] * 4
                e = [20. + number, -20. - number, 0., 0.] if axis == "x" else [4., 4., 0., 0.] if axis == "z" else [0.] * 4
                if bad_endpoint:
                    e = [0.] * 4
                vertices.append(v)
                endpoints.append(e)
            data[f"MCParticles/MCParticles.vertex.{axis}"] = ak.Array(vertices)
            data[f"MCParticles/MCParticles.endpoint.{axis}"] = ak.Array(endpoints)
        for name in pilot.TRACKER_COLLECTIONS:
            data[f"{name}/{name}.cellID"] = ak.Array([[1] for _ in numbers])
        with uproot.recreate(path) as root:
            root["events"] = data

    def test_decay_daughter_matching_allows_only_small_reported_adjustments(self):
        source, root = self.root / "source.hepmc3", self.root / "event.root"
        self.signal(source, numbers=(0,))
        self.edm(root, numbers=(0,), daughter_adjustment=.0003)
        report = pilot.validate_simulation(root, source, 1)
        self.assertEqual(report["decay_momentum_adjusted_matches"], 1)
        self.assertLess(report["maximum_decay_momentum_relative_adjustment"], .001)
        self.edm(root, numbers=(0,), daughter_adjustment=.003)
        with self.assertRaisesRegex(ValueError, "no unique generator MCParticle"):
            pilot.validate_simulation(root, source, 1)

    def decay_boost_fixture(self, wrong_parent=False, vertex_offset=0., corrupt_momentum=0., ambiguous=False):
        source, root = self.root / "boost.hepmc3", self.root / "boost.root"
        self.signal(source, numbers=(0,))
        self.edm(root, numbers=(0,))
        with uproot.open(root) as handle:
            arrays = {name: ak.to_list(value) for name, value in handle["events"].arrays(library="ak", how=dict).items()}
        prefix = "MCParticles/MCParticles."
        endpoint = np.array([31., 0., 0.])
        predicted = pilot.boost_four_momentum(
            pilot.boost_four_momentum([1., .2, .3, 1.1], [-30 / math.hypot(30, 55), 0, 0]),
            endpoint / math.sqrt(float(endpoint @ endpoint) + 55**2))
        for axis, value in zip("xyz", predicted):
            arrays[prefix + "momentum." + axis][0][2] = float(value) + (corrupt_momentum if axis == "x" else 0.)
            arrays[prefix + "momentumAtEndpoint." + axis] = [[
                float(endpoint["xyz".index(axis)]), -30. if axis == "x" else 0., 0., 0.]]
        arrays[prefix + "vertex.x"][0][2] += vertex_offset
        arrays[prefix + "parents_begin"] = [[0, 0, 0, 1]]
        arrays[prefix + "parents_end"] = [[0, 0, 1, 2]]
        arrays[prefix + "mass"] = [[55., 55., .13957, .13957]]
        arrays["_MCParticles_parents/_MCParticles_parents.index"] = [[1 if wrong_parent else 0, 1]]
        if ambiguous:
            for name, values in arrays.items():
                if name.startswith(prefix):
                    values[0].append(values[0][2])
        with uproot.recreate(root) as handle:
            handle["events"] = {name: ak.Array(value) for name, value in arrays.items()}
        return source, root

    def test_decay_boost_fallback_uses_saved_parent_and_vertex(self):
        source, root = self.decay_boost_fixture()
        report = pilot.validate_simulation(root, source, 1)
        self.assertTrue(report["passed"])
        self.assertEqual(report["decay_boost_match_count"], 1)
        row = report["decay_boost_matches"][0]
        self.assertGreater(row["original_relative_momentum_difference"], .001)
        self.assertLess(row["predicted_relative_momentum_residual"], 1e-12)

    def test_decay_boost_rejects_wrong_parent_vertex_momentum_and_ambiguity(self):
        for options in ({"wrong_parent": True}, {"vertex_offset": .001},
                        {"corrupt_momentum": .01}, {"ambiguous": True}):
            with self.subTest(options=options):
                source, root = self.decay_boost_fixture(**options)
                report_path = self.root / "failed.json"
                with self.assertRaisesRegex(ValueError, "Descendant matching failed"):
                    pilot.validate_simulation(root, source, 1, report_path)
                report = json.loads(report_path.read_text())
                self.assertFalse(report["passed"])
                self.assertEqual(len(report["descendant_matching_failures"]), 1)

    def test_neutral_descendants_are_still_validated(self):
        source, root = self.decay_boost_fixture()
        with self.hep.open(str(source)) as reader:
            event = reader.read()
        next(p for p in event.particles if p.pid == 211).pid = 22
        with self.hep.open(str(source), "w") as writer:
            writer.write(event)
        with uproot.open(root) as handle:
            arrays = {name: ak.to_list(value) for name, value in handle["events"].arrays(library="ak", how=dict).items()}
        arrays["MCParticles/MCParticles.PDG"][0][2] = 22
        arrays["MCParticles/MCParticles.charge"][0][2] = 0.
        with uproot.recreate(root) as handle:
            handle["events"] = {name: ak.Array(value) for name, value in arrays.items()}
        report = pilot.validate_simulation(root, source, 1)
        self.assertEqual(report["decay_boost_matches"][0]["pdg"], 22)
        self.assertEqual(report["charged_descendants_checked"], 1)

    def test_lorentz_boost_identity_inverse_and_invalid_beta(self):
        vector = np.array([1., .2, -.3, 2.])
        np.testing.assert_array_equal(pilot.boost_four_momentum(vector, [0, 0, 0]), vector)
        boosted = pilot.boost_four_momentum(vector, [.2, -.1, .3])
        np.testing.assert_allclose(pilot.boost_four_momentum(boosted, [-.2, .1, -.3]), vector)
        self.assertAlmostEqual(boosted[3]**2 - boosted[:3] @ boosted[:3],
                               vector[3]**2 - vector[:3] @ vector[:3])
        with self.assertRaisesRegex(ValueError, "Invalid decay boost"):
            pilot.boost_four_momentum(vector, [1, 0, 0])

    def test_recovery_preserves_original_and_records_old_code_provenance(self):
        from scripts.simulation import recover_duplicate_events
        output = self.root / "recover"
        simulation = output / "simulation/pu200/edm4hep.root"
        simulation.parent.mkdir(parents=True)
        simulation.write_bytes(b"original duplicated output")
        merged = output / "generation/merged_events.hepmc3"
        merged.parent.mkdir()
        self.signal(merged, numbers=(0,))
        pilot.write_particle_table(output / "particle_llpfix.tbl")
        pilot.write_json(output / "simulation.json", dict(events=1, threads=1))
        checksums = {str(p.relative_to(pilot.REPO)): pilot.checksum(p) for p in pilot.SOURCE_FILES}
        old_checksums = dict(checksums, **{"scripts/simulation/ddsim_run.py": "old-writer-code"})
        manifest = dict(prepared=True, source_checksums=old_checksums, run=1, events=1,
                        signal_start=0, pileup_reservation=dict(namespace="test"),
                        merged_sha256=pilot.checksum(merged),
                        particle_table_sha256=pilot.checksum(output / "particle_llpfix.tbl"))
        pilot.write_json(output / "pilot.json", manifest)
        args = SimpleNamespace(output=output, events=1, threads=1, run=1, signal_start=0,
                               stage="recover", seed_namespace="test")

        def fake_recover(source, destination, events):
            self.assertEqual(source.read_bytes(), b"original duplicated output")
            self.edm(destination, numbers=(0,))
            return dict(recovered_events=events)

        with patch.object(pilot, "runtime_preflight", return_value={}), \
             patch.object(recover_duplicate_events, "recover", side_effect=fake_recover), \
             patch.object(pilot, "command") as command:
            result = pilot.run(args)
            command.assert_not_called()
        self.assertTrue(result["simulation_validated"])
        self.assertEqual(result["simulation_source_checksums"], old_checksums)
        self.assertEqual(result["source_checksums"], checksums)
        self.assertEqual(simulation.with_name("edm4hep.duplicate-writer.root").read_bytes(),
                         b"original duplicated output")

    def test_select_signal_records_original_ids_and_renumbers(self):
        source, selected = self.root / "source.hepmc3", self.root / "selected.hepmc3"
        self.signal(source, numbers=(5, 2, 8))
        records = pilot.select_signal(source, selected, 1, 2)
        self.assertEqual(records, [dict(source_entry=1, source_event_number=2, logical_event=0),
                                   dict(source_entry=2, source_event_number=8, logical_event=1)])
        self.assertEqual(pilot.hepmc_events(selected, 2), [0, 1])
        with self.assertRaisesRegex(ValueError, "too few"):
            pilot.select_signal(source, self.root / "short.hepmc3", 2, 2)

    def test_duplicate_generator_event_ids_fail(self):
        path = self.root / "duplicate.hepmc3"
        self.signal(path, numbers=(0, 0))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pilot.hepmc_events(path)

    def test_root_event_reordering_and_displaced_truth_gate(self):
        hepmc, edm = self.root / "input.hepmc3", self.root / "edm4hep.root"
        self.signal(hepmc)
        self.edm(edm)
        report = pilot.validate_simulation(edm, hepmc, 2)
        self.assertEqual(report["root_to_hepmc"], [dict(root_entry=0, hepmc_event=1),
                                                  dict(root_entry=1, hepmc_event=0)])
        self.assertEqual(report["llps_checked"], 4)
        self.assertEqual(report["charged_descendants_checked"], 4)
        self.assertEqual(report["displaced_charged_descendants_checked"], 4)
        self.assertTrue(report["displacement_gate_exercised"])
        self.assertLess(report["maximum_llp_endpoint_error_mm"], 1e-9)

    def test_collapsed_displaced_descendants_are_rejected(self):
        hepmc, edm = self.root / "input.hepmc3", self.root / "edm4hep.root"
        self.signal(hepmc)
        self.edm(edm, collapsed=True)
        with self.assertRaisesRegex(ValueError, "placed at collision"):
            pilot.validate_simulation(edm, hepmc, 2)

    def test_wrong_endpoints_and_event_count_are_rejected(self):
        hepmc, edm = self.root / "input.hepmc3", self.root / "edm4hep.root"
        self.signal(hepmc)
        self.edm(edm, bad_endpoint=True)
        with self.assertRaisesRegex(ValueError, "endpoint differs"):
            pilot.validate_simulation(edm, hepmc, 2)
        self.edm(edm, numbers=(0,))
        with self.assertRaisesRegex(ValueError, "Geant4 wrote 1"):
            pilot.validate_simulation(edm, hepmc, 2)

    def test_fresh_seed_reservations_cannot_be_recycled(self):
        path = self.root / "registry.json"
        seed = pilot.production_seed("test", 0, "generation")
        self.assertEqual(seed, pilot.production_seed("test", 0, "generation"))
        self.assertNotEqual(seed, pilot.production_seed("test", 1, "generation"))
        self.assertTrue(1 <= seed + 1000 <= 900000000)
        pilot.reserve_pileup(path, "test", 0, seed + 1000, self.root / "output0")
        with self.assertRaisesRegex(ValueError, "already reserved"):
            pilot.reserve_pileup(path, "test", 0, seed + 1000, self.root / "retry")
        with self.assertRaisesRegex(ValueError, "already reserved"):
            pilot.reserve_pileup(path, "other", 1, seed + 1000, self.root / "collision")
        new = pilot.production_seed("test", 1, "generation")
        pilot.reserve_pileup(path, "test", 1, new + 1000, self.root / "output1")
        self.assertEqual(len(json.loads(path.read_text())), 2)

    def test_particle_table_marks_scalar_unstable_and_correct_lifetime(self):
        records = [line.split() for line in (pilot.CONFIG / "particle_llpfix.tbl").read_text().splitlines()
                   if line and not line.startswith("//")]
        scalar = next(row for row in records if row[0] == "36")
        self.assertEqual(float(scalar[3]), 55.)
        self.assertLess(float(scalar[4]), -.5)
        self.assertEqual(float(scalar[5]), 100.)
        config = json.loads((pilot.CONFIG / "simulation.json").read_text())
        self.assertEqual(config["vertexSigma"], [0.] * 4)
        self.assertEqual(config["alternative_decay_statuses"], [22])
        self.assertEqual(config["extra_reject_pdgs"], [35])
        target = self.root / "full_particle_table.tbl"
        pilot.write_particle_table(target)
        def unchanged_lines(path):
            return [line for line in path.read_text().splitlines()
                    if not line.split() or line.split()[0] != "36"]
        self.assertEqual(unchanged_lines(target), unchanged_lines(pilot.BASE_PARTICLE_TABLE))

    def test_download_size_and_published_checksum_gate(self):
        path = self.root / "input"
        path.write_bytes(b"verified source")
        spec = dict(bytes=path.stat().st_size, md5=pilot.checksum(path, "md5"))
        verify(path, spec)
        with self.assertRaisesRegex(ValueError, "Wrong file size"):
            verify(path, {**spec, "bytes": 1})
        with self.assertRaisesRegex(ValueError, "MD5"):
            verify(path, {**spec, "md5": "wrong"})

    def test_full_pilot_orchestration_and_staged_reuse_without_running_physics(self):
        config = self.root / "config"
        shutil.copytree(pilot.CONFIG, config)
        signal, pu0 = self.root / "signal.hepmc3", self.root / "pu0.root"
        self.signal(signal)
        self.edm(pu0)
        sources = json.loads((config / "input_sources.json").read_text())
        sources["source_events"] = 2
        sources["inputs"]["signal"].update(bytes=signal.stat().st_size, md5=pilot.checksum(signal, "md5"))
        sources["inputs"]["pu0"]["bytes"] = pu0.stat().st_size
        (config / "input_sources.json").write_text(json.dumps(sources))
        args = SimpleNamespace(output=self.root / "pilot", events=2, threads=8, run=17,
                               signal_start=0, stage="prepare", seed_namespace="unit-test",
                               registry=self.root / "registry.json", signal=signal, input_pu0=pu0)
        calls = []

        def fake_command(parts, log):
            calls.append(parts)
            Path(log).write_text("Mocked physics stage; no actual simulation.\n")
            script = str(parts[1])
            destination = Path(parts[parts.index("--output") + 1])
            if script.endswith("pythia_gen.py"):
                count = math.ceil(2 * 200 + 5 * math.sqrt(2 * 200))
                self.signal(destination / "events_pileup.hepmc3", numbers=range(count))
                shutil.copyfile(destination / "selected_signal.hepmc3", destination / "merged_events.hepmc3")
            elif script.endswith("ddsim_run.py"):
                destination.mkdir(parents=True)
                self.edm(destination / "edm4hep.root")
            elif script.endswith("run_paired_samples.py"):
                destination.mkdir()
                complete = dict(dataset="higgs_portal", version="v2_cells_pilot", run=17,
                                events_per_sample=2, samples={"pu0": {}, "pu200": {}})
                (destination / "complete.json").write_text(json.dumps(complete))
            else:
                raise AssertionError(script)
            return .125

        with patch.object(pilot, "CONFIG", config), \
             patch.object(pilot, "runtime_preflight", return_value={"mocked": True}), \
             patch.object(pilot, "command", side_effect=fake_command):
            prepared = pilot.run(args)
            self.assertTrue(prepared["prepared"])
            self.assertEqual(prepared["generated_pileup_events"], 500)
            self.assertEqual(prepared["reserved_pileup_source_id_range"], [0, 499])
            self.assertEqual(len(calls), 1)
            args.stage = "simulate"
            simulated = pilot.run(args)
            self.assertTrue(simulated["simulation_validated"])
            self.assertEqual(simulated["simulation_threads"], 8)
            original_sources = dict(simulated["source_checksums"])
            old_manifest = json.loads((args.output / "pilot.json").read_text())
            old_manifest["source_checksums"]["scripts/simulation/higgs_portal_pilot.py"] = "old-validator"
            old_manifest.pop("simulation_validated")
            (args.output / "pilot.json").write_text(json.dumps(old_manifest))
            args.stage = "validate"
            validated = pilot.run(args)
            self.assertTrue(validated["simulation_validated"])
            self.assertEqual(len(calls), 2)  # validation never invokes a physics command
            self.assertEqual(validated["source_checksums"], old_manifest["source_checksums"])
            self.assertEqual(validated["validation_source_checksums"], original_sources)
            args.threads = 7
            with self.assertRaisesRegex(ValueError, "events/threads"):
                pilot.run(args)
            args.threads = 8
            args.stage = "export"
            finished = pilot.run(args)
            self.assertTrue(finished["complete"])
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(json.loads(args.registry.read_text())), 1)
            args.seed_namespace = "different"
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                pilot.run(args)
        simulation_config = json.loads((args.output / "simulation.json").read_text())
        self.assertEqual(simulation_config["vertexSigma"], [0.] * 4)
        self.assertEqual(simulation_config["alternative_decay_statuses"], [22])


if __name__ == "__main__":
    unittest.main()
