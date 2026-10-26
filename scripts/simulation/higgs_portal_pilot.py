#!/usr/bin/env python3
"""Prepare, simulate, validate and export a small corrected Higgs-portal pu200 pilot."""

from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import uproot


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs_development/higgs_portal_pilot"
BASE_PARTICLE_TABLE = REPO / "configs_production/hard_scatter/higgs_portal/particle.tbl"
TRACKER_COLLECTIONS = ("PixelBarrelReadout", "PixelEndcapReadout", "ShortStripBarrelReadout",
                       "ShortStripEndcapReadout", "LongStripBarrelReadout", "LongStripEndcapReadout")
SOURCE_FILES = (Path(__file__), REPO / "scripts/simulation/pythia_gen.py",
                REPO / "scripts/simulation/ddsim_run.py",
                REPO / "scripts/simulation/recover_duplicate_events.py", CONFIG / "simulation.json",
                CONFIG / "particle_llpfix.tbl", CONFIG / "input_sources.json", BASE_PARTICLE_TABLE)


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def production_seed(namespace, run, stage):
    value = int.from_bytes(hashlib.sha256(f"{namespace}:run{run}:{stage}".encode()).digest()[:8], "big")
    return 1 + value % 899_998_000  # Leave room for generate_pileup's +1000 offset.


def write_particle_table(destination):
    """Preserve the production particle table, replacing only the scalar record."""
    overrides = [line.split() for line in (CONFIG / "particle_llpfix.tbl").read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith(("//", "#"))]
    if (len(overrides) != 1 or len(overrides[0]) != 6 or overrides[0][0] != "36"
            or float(overrides[0][3]) != 55 or float(overrides[0][4]) >= -.5
            or float(overrides[0][5]) != 100):
        raise ValueError("LLP override must define only unstable PDG 36, mass 55 GeV, c*tau 100 mm")
    lines, replaced = [], 0
    for line in BASE_PARTICLE_TABLE.read_text().splitlines(keepends=True):
        fields = line.split()
        if fields and fields[0] == "36":
            if len(fields) != 6:
                raise ValueError("Unexpected production PDG 36 particle-table record")
            line = " ".join(overrides[0]) + "\n"
            replaced += 1
        lines.append(line)
    if replaced != 1:
        raise ValueError("Expected exactly one PDG 36 record in the production table")
    Path(destination).write_text("".join(lines))


def reserve_pileup(registry, namespace, run, seed, output):
    """Reserve the entire fresh background file, including its unused Poisson buffer."""
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(registry, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(descriptor, "r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        records = json.loads(stream.read() or "[]")
        key = f"{namespace}:run{run}"
        if any(row["key"] == key or row["pileup_generator_seed"] == seed for row in records):
            raise ValueError("Pileup seed/run already reserved; use a new --run. Failed reservations are not recycled.")
        record = dict(key=key, namespace=namespace, run=run, pileup_generator_seed=seed,
                      output=str(output), policy="exclusive fresh file; unused buffer also reserved")
        records.append(record)
        stream.seek(0)
        stream.write(json.dumps(records, indent=2) + "\n")
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    return record


def hepmc_events(path, expected=None):
    import pyhepmc
    numbers = []
    with pyhepmc.open(str(path)) as reader:
        for event in reader:
            if event.momentum_unit != pyhepmc.Units.GEV or event.length_unit != pyhepmc.Units.MM:
                raise ValueError(f"Unexpected HepMC units in {path}")
            numbers.append(event.event_number)
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"Duplicate HepMC event numbers in {path}")
    if expected is not None and len(numbers) != expected:
        raise ValueError(f"{path}: expected {expected} events, got {len(numbers)}")
    return numbers


def select_signal(source, output, start, count):
    """Select by file entry, retain original IDs in provenance, renumber for ACTS."""
    import pyhepmc
    selected = []
    with pyhepmc.open(str(source)) as reader, pyhepmc.open(str(output), "w") as writer:
        for entry, event in enumerate(reader):
            if entry < start:
                continue
            if len(selected) == count:
                break
            if event.momentum_unit != pyhepmc.Units.GEV or event.length_unit != pyhepmc.Units.MM:
                raise ValueError("Signal must use GeV and mm units")
            scalars = [p for p in event.particles if p.pid == 36 and p.end_vertex is not None]
            if len(scalars) != 2 or any(abs(p.generated_mass - 55) > .5 for p in scalars):
                raise ValueError(f"Signal entry {entry} is not the expected two-LLP, 55 GeV sample")
            if any(p.status not in (2, 22) for p in scalars):
                raise ValueError("Unexpected LLP generator status; do not simulate without reviewing its handling")
            selected.append(dict(source_entry=entry, source_event_number=event.event_number,
                                 logical_event=len(selected)))
            event.event_number = len(selected) - 1
            writer.write(event)
    if len(selected) != count:
        raise ValueError(f"Signal has too few events after entry {start}: {len(selected)} < {count}")
    return selected


def branch(tree, name):
    matches = [key for key in tree.keys() if key == name or key.endswith("/" + name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one ROOT branch for {name}, found {matches}")
    return matches[0]


def check_collections(tree):
    keys = tree.keys()
    missing = [name for name in (*TRACKER_COLLECTIONS, "MCParticles")
               if not any(key == name or key.startswith(name + "/") or key.startswith(name + ".") for key in keys)]
    if missing:
        raise ValueError(f"EDM4hep input lacks required collections: {missing}")


def validate_simulation(edm4hep, merged_hepmc, events):
    """Check source-event mapping and catastrophic displaced-decay failure modes."""
    import awkward as ak
    import pyhepmc
    reference = {}
    with pyhepmc.open(str(merged_hepmc)) as reader:
        for event in reader:
            if event.event_number in reference:
                raise ValueError("Duplicate merged event number")
            reference[event.event_number] = event
    if set(reference) != set(range(events)):
        raise ValueError("Merged HepMC event set differs from the requested logical events")
    endpoint_errors, checked_descendants, displaced = [], 0, 0
    adjusted_matches, maximum_adjustment = 0, 0.0
    mapping = []
    with uproot.open(edm4hep) as root:
        tree = root["events"]
        if tree.num_entries != events:
            raise ValueError(f"Geant4 wrote {tree.num_entries} events; expected {events}")
        check_collections(tree)
        fields = ("PDG", "generatorStatus", "charge", "vertex.x", "vertex.y", "vertex.z",
                  "endpoint.x", "endpoint.y", "endpoint.z", "momentum.x", "momentum.y", "momentum.z")
        names = {field: branch(tree, "MCParticles." + field) for field in fields}
        header = branch(tree, "EventHeader.eventNumber")
        for entry in range(events):
            batch = tree.arrays([header, *names.values()], entry_start=entry, entry_stop=entry+1, library="ak")
            event_numbers = np.asarray(ak.to_list(batch[header][0])).reshape(-1)
            if len(event_numbers) != 1:
                raise ValueError("An EDM4hep entry must have exactly one EventHeader")
            number = int(event_numbers[0])
            if number not in reference or number in {row["hepmc_event"] for row in mapping}:
                raise ValueError("Invalid or duplicate ROOT-to-HepMC event mapping")
            mapping.append(dict(root_entry=entry, hepmc_event=number))
            data = {field: ak.to_numpy(batch[name][0]) for field, name in names.items()}
            momentum = np.column_stack([data[f"momentum.{axis}"] for axis in "xyz"])

            def match(particle, allow_decay_adjustment=False):
                nonlocal adjusted_matches, maximum_adjustment
                expected = np.array([particle.momentum.px, particle.momentum.py, particle.momentum.pz])
                mask = ((data["PDG"] == particle.pid) & (data["generatorStatus"] > 0)
                        & np.isclose(momentum, expected, rtol=2e-6, atol=2e-5).all(axis=1))
                indices = np.flatnonzero(mask)
                # Preassigned decay daughters can have small momentum changes in Geant4.
                # Keep LLP matching strict; require a unique, bounded daughter match.
                if len(indices) == 0 and allow_decay_adjustment:
                    scale = max(float(np.linalg.norm(expected)), 0.02)
                    distances = np.linalg.norm(momentum - expected, axis=1)
                    candidates = np.flatnonzero((data["PDG"] == particle.pid)
                                               & (data["generatorStatus"] > 0)
                                               & (distances <= 1e-3 * scale))
                    if len(candidates) == 1:
                        indices = candidates
                        adjusted_matches += 1
                        maximum_adjustment = max(maximum_adjustment, float(distances[indices[0]] / scale))
                if len(indices) != 1:
                    raise ValueError(f"Event {number}: no unique generator MCParticle match for PDG {particle.pid}")
                return int(indices[0])

            signal = reference[number]
            scalars = [p for p in signal.particles if p.pid == 36 and p.end_vertex is not None]
            if len(scalars) != 2:
                raise ValueError(f"Event {number}: merged event lacks two decaying signal LLPs")
            descendants = {}
            for scalar in scalars:
                index = match(scalar)
                expected = scalar.end_vertex.position
                endpoint = np.array([data[f"endpoint.{axis}"][index] for axis in "xyz"])
                error = float(np.linalg.norm(endpoint - [expected.x, expected.y, expected.z]))
                if not math.isfinite(error) or error > 1.0:
                    raise ValueError(f"Event {number}: LLP endpoint differs from HepMC by {error} mm")
                endpoint_errors.append(error)
                todo = list(scalar.end_vertex.particles_out)
                visited = set()
                while todo:
                    particle = todo.pop()
                    if particle.id in visited:
                        continue
                    visited.add(particle.id)
                    if particle.status == 1 and particle.production_vertex is not None:
                        descendants[particle.id] = particle
                    if particle.end_vertex is not None:
                        todo.extend(particle.end_vertex.particles_out)
            for particle in descendants.values():
                index = match(particle, allow_decay_adjustment=True)
                if data["charge"][index] == 0:
                    continue
                checked_descendants += 1
                position = particle.production_vertex.position
                if math.hypot(position.x, position.y) > 10:
                    displaced += 1
                    radius = math.hypot(data["vertex.x"][index], data["vertex.y"][index])
                    if not math.isfinite(radius) or radius < 1:
                        raise ValueError(f"Event {number}: displaced LLP descendant placed at collision point")
    if len(endpoint_errors) != 2 * events:
        raise ValueError("LLP endpoint coverage is incomplete")
    return dict(events=events, root_to_hepmc=mapping, llps_checked=len(endpoint_errors),
                maximum_llp_endpoint_error_mm=max(endpoint_errors),
                charged_descendants_checked=checked_descendants,
                displaced_charged_descendants_checked=displaced,
                decay_momentum_adjusted_matches=adjusted_matches,
                maximum_decay_momentum_relative_adjustment=maximum_adjustment,
                decay_momentum_matching_relative_tolerance=1e-3,
                displacement_gate_exercised=displaced > 0,
                interpretation="Pilot sanity gate, not a complete LLP physics validation")


def geant4_data_paths():
    config = shutil.which("geant4-config")
    if config is None:
        raise ValueError("geant4-config is missing; source the installed geant4.sh")
    manifest = subprocess.check_output([config, "--datasets"], text=True)
    paths = {}
    for line in manifest.splitlines():
        _, name, installed = line.split(maxsplit=2)
        # Modern geant4.sh exports only the common data directory, not each variable.
        default = Path(os.environ["GEANT4_DATA_DIR"]) / Path(installed).name if os.environ.get("GEANT4_DATA_DIR") else Path(installed)
        paths[name] = os.environ.get(name, str(default))
    required = {"G4LEDATA", "G4LEVELGAMMADATA", "G4PARTICLEXSDATA", "G4ENSDFSTATEDATA"}
    if not required <= paths.keys():
        raise ValueError("Geant4 dataset manifest lacks required physics datasets")
    paths.update({name: value for name, value in os.environ.items()
                  if name.startswith("G4") and name.endswith("DATA")})
    for name, value in sorted(paths.items()):
        path = Path(value)
        if not path.is_dir() or not any(path.iterdir()):
            raise ValueError(f"Missing Geant4 physics data: {name}={path}; check the cached datasets")
    os.environ.update(paths)
    return dict(sorted(paths.items()))


def runtime_preflight():
    sys.path.insert(0, str(REPO))
    from scripts.postprocessing.run_paired_samples import revision, PINNED_ACTS_PREFIX, PINNED_ODD_REVISION
    import acts
    import acts.examples.pythia8
    import acts.examples.hepmc3
    from DDSim.DD4hepSimulation import DD4hepSimulation
    acts_revision = revision(Path("/opt/acts-arrow-src"))
    odd_revision = revision(Path("/cache/odd-v4"))
    if not acts_revision.startswith(PINNED_ACTS_PREFIX) or odd_revision != PINNED_ODD_REVISION:
        raise ValueError("Pilot requires the same pinned ACTS and ODD as paired-sample production")
    if not Path("/cache/odd-v4-install/lib/libOpenDataDetector.so").is_file():
        raise ValueError("ODD factory library is missing")
    data_paths = geant4_data_paths()
    simulation = DD4hepSimulation()
    simulation.physics.alternativeDecayStatuses = {22}
    simulation.physics.rejectPDGs = set(simulation.physics.rejectPDGs) | {35}
    simulation.physics.pdgfile = str(CONFIG / "particle_llpfix.tbl")
    return dict(acts=acts_revision, odd=odd_revision, python=sys.version,
                geant4_data=data_paths)


def command(parts, logfile):
    print("Running:", " ".join(map(str, parts)), flush=True)
    print("Log:", logfile, flush=True)
    start = time.perf_counter()
    with logfile.open("w") as log:
        result = subprocess.run(list(map(str, parts)), cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        with logfile.open() as log:
            print("".join(deque(log, maxlen=30)), file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, parts)
    return time.perf_counter() - start


def run(args):
    if args.events < 1 or args.threads < 1 or not 0 <= args.run < 2**32 or args.signal_start < 0:
        raise ValueError("Require positive events/threads, nonnegative signal start, and a uint32 run ID")
    sources = json.loads((CONFIG / "input_sources.json").read_text())
    if args.signal_start + args.events > sources["source_events"]:
        raise ValueError("Requested signal entries exceed the 1,000-event source run")
    runtime = runtime_preflight()
    output = args.output.resolve()
    manifest_path = output / "pilot.json"
    source_checksums = {str(path.relative_to(REPO)): checksum(path) for path in SOURCE_FILES}
    if args.stage in ("all", "prepare"):
        if output.exists():
            raise ValueError("Output exists; choose a new directory/run or use a specific later --stage")
        for kind, path in (("signal", args.signal), ("pu0", args.input_pu0)):
            if not path.is_file() or path.stat().st_size != sources["inputs"][kind]["bytes"]:
                raise ValueError(f"Missing/incomplete corrected {kind} input: {path}")
        if checksum(args.signal, "md5") != sources["inputs"]["signal"]["md5"]:
            raise ValueError("Signal MD5 differs from the published corrected-source manifest")
        with uproot.open(args.input_pu0) as root:
            check_collections(root["events"])
            if root["events"].num_entries != sources["source_events"]:
                raise ValueError("Corrected pu0 source should contain 1,000 events")
        seed = production_seed(args.seed_namespace, args.run, "generation")
        reservation = reserve_pileup(args.registry, args.seed_namespace, args.run, seed + 1000, output)
        output.mkdir(parents=True)
        write_particle_table(output / "particle_llpfix.tbl")
        generation = output / "generation"
        generation.mkdir()
        selected = generation / "selected_signal.hepmc3"
        selected_events = select_signal(args.signal, selected, args.signal_start, args.events)
        generation_config = dict(events=args.events, threads=1, seed=seed, hard_process=None,
                                 pileup=200, merge=True, poisson_sample=True, poisson_buffer_sigma=5.0,
                                 hard_scatter_file=str(selected), vertex_sigma_xy=.0125,
                                 vertex_sigma_z=55.5, vertex_sigma_t=.185)
        write_json(output / "generation.json", generation_config)
        manifest = dict(dataset="higgs_portal", version="v2_cells_pilot", run=args.run,
                        events=args.events, signal_start=args.signal_start, runtime=runtime,
                        source_checksums=source_checksums, source_dataset=sources["source_dataset"],
                        physics=sources["physics"], selected_signal_events=selected_events,
                        input_sha256={str(args.signal): checksum(args.signal), str(args.input_pu0): checksum(args.input_pu0)},
                        pileup_reservation=reservation,
                        particle_table_sha256=checksum(output / "particle_llpfix.tbl"),
                        pileup_policy="Fresh exclusive file; pinned ACTS reads sequentially and aborts at EOF; no wraparound. All buffer entries reserved.",
                        comparison_note="pu0 ROOT entries and pu200 signal selections are not assumed event-by-event paired",
                        timings_seconds={})
        write_json(manifest_path, manifest)
        elapsed = command([sys.executable, REPO / "scripts/simulation/pythia_gen.py", "--config",
                           output / "generation.json", "--output", generation], output / "generation.log")
        generated = math.ceil(args.events * 200 + 5 * math.sqrt(args.events * 200))
        numbers = hepmc_events(generation / "events_pileup.hepmc3", generated)
        if sorted(numbers) != list(range(generated)):
            raise ValueError("Generated pileup source IDs must be unique and contiguous from zero")
        if sorted(hepmc_events(generation / "merged_events.hepmc3", args.events)) != list(range(args.events)):
            raise ValueError("Merged events are not numbered by the selected signal's logical IDs")
        manifest.update(prepared=True, generated_pileup_events=generated,
                        reserved_pileup_source_id_range=[0, generated-1],
                        pileup_sha256=checksum(generation / "events_pileup.hepmc3"),
                        merged_sha256=checksum(generation / "merged_events.hepmc3"))
        manifest["timings_seconds"]["generation_and_merge"] = elapsed
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text())
        changed_sources = {name for name in manifest["source_checksums"].keys() | source_checksums.keys()
                           if manifest["source_checksums"].get(name) != source_checksums.get(name)}
        recovery_changes = {"scripts/simulation/higgs_portal_pilot.py",
                            "scripts/simulation/ddsim_run.py",
                            "scripts/simulation/recover_duplicate_events.py"}
        code_matches = (not changed_sources or
                        (args.stage == "recover" and changed_sources <= recovery_changes))
        if (not manifest.get("prepared") or not code_matches
                or manifest["run"] != args.run or manifest["events"] != args.events
                or manifest["signal_start"] != args.signal_start
                or manifest["pileup_reservation"]["namespace"] != args.seed_namespace):
            raise ValueError("Existing pilot metadata does not match the requested run/code/events")
    if args.stage == "recover":
        from scripts.simulation.recover_duplicate_events import recover
        if manifest.get("simulation_validated") or (output / "paired").exists():
            raise ValueError("Recovery is only for an unvalidated pilot before paired export")
        simulation_config = json.loads((output / "simulation.json").read_text())
        if (args.threads != 1 or simulation_config.get("threads") != 1
                or simulation_config.get("events") != args.events):
            raise ValueError("Writer recovery is restricted to the sequential duplicate-output case")
        simulation = output / "simulation/pu200/edm4hep.root"
        candidate = simulation.with_name("edm4hep.recovered.root")
        backup = simulation.with_name("edm4hep.duplicate-writer.root")
        if backup.exists():
            raise ValueError("An archived duplicate output already exists; inspect recovery state")
        merged = output / "generation/merged_events.hepmc3"
        if (checksum(merged) != manifest["merged_sha256"] or
                checksum(output / "particle_llpfix.tbl") != manifest["particle_table_sha256"]):
            raise ValueError("Prepared simulation inputs changed")
        original_sha256 = checksum(simulation)
        recovery = recover(simulation, candidate, args.events)
        validation = validate_simulation(candidate, merged, args.events)
        recovery.update(original_sha256=original_sha256, recovered_sha256=checksum(candidate),
                        original_path=str(backup), source_checksums_before=manifest["source_checksums"],
                        source_checksums_after=source_checksums)
        # A hard link retains the original bytes before atomically installing the verified copy.
        os.link(simulation, backup)
        os.replace(candidate, simulation)
        write_json(output / "writer_recovery.json", recovery)
        write_json(output / "simulation_validation.json", validation)
        manifest.update(simulation_validated=True, simulation_sha256=recovery["recovered_sha256"],
                        simulation_source_checksums=manifest["source_checksums"],
                        source_checksums=source_checksums, writer_recovery=recovery,
                        simulation_threads=simulation_config["threads"],
                        original_simulation_timing_report=str(simulation.parent / "timing_summary.txt"))
        write_json(manifest_path, manifest)
    if args.stage in ("all", "simulate"):
        merged = output / "generation/merged_events.hepmc3"
        if checksum(merged) != manifest["merged_sha256"]:
            raise ValueError("Merged input changed after preparation")
        simulation = output / "simulation/pu200"
        if simulation.exists():
            raise ValueError("Simulation output already exists; retain failed attempts for diagnosis")
        simulation_config = json.loads((CONFIG / "simulation.json").read_text())
        particle_table = output / "particle_llpfix.tbl"
        if checksum(particle_table) != manifest["particle_table_sha256"]:
            raise ValueError("Prepared LLP particle table changed")
        simulation_config.update(events=args.events, threads=args.threads,
                                 seed=production_seed(args.seed_namespace, args.run, "geant4"),
                                 pdg_file=str(particle_table))
        write_json(output / "simulation.json", simulation_config)
        elapsed = command([sys.executable, REPO / "scripts/simulation/ddsim_run.py", "--config",
                           output / "simulation.json", "--input-file", merged, "--output", simulation],
                          output / "simulation.log")
        validation = validate_simulation(simulation / "edm4hep.root", merged, args.events)
        write_json(output / "simulation_validation.json", validation)
        if not validation["displacement_gate_exercised"]:
            print("WARNING: no charged descendant beyond 10 mm in this small sample; exercise that gate in the larger pilot.")
        manifest.update(simulation_validated=True, simulation_sha256=checksum(simulation / "edm4hep.root"))
        manifest["timings_seconds"]["geant4"] = elapsed
        manifest["geant4_wall_seconds_per_event"] = elapsed / args.events
        manifest["simulation_threads"] = args.threads
        write_json(manifest_path, manifest)
    if args.stage in ("all", "export"):
        simulated = output / "simulation/pu200/edm4hep.root"
        if not manifest.get("simulation_validated") or checksum(simulated) != manifest["simulation_sha256"]:
            raise ValueError("Simulation validation/checksum gate must pass before export")
        if checksum(args.input_pu0) != manifest["input_sha256"].get(str(args.input_pu0)):
            raise ValueError("Corrected pu0 reference changed after preparation")
        elapsed = command([sys.executable, REPO / "scripts/postprocessing/run_paired_samples.py",
                           "--dataset", manifest["dataset"], "--version", manifest["version"],
                           "--run", args.run, "--events", args.events, "--input-pu0", args.input_pu0,
                           "--input-pu200", simulated, "--pixel-quantization", "p99.9",
                           "--output", output / "paired"], output / "export.log")
        completed = json.loads((output / "paired/complete.json").read_text())
        if (completed.get("dataset") != manifest["dataset"]
                or completed.get("version") != manifest["version"]
                or completed.get("run") != args.run
                or completed.get("events_per_sample") != args.events
                or set(completed.get("samples", {})) != {"pu0", "pu200"}):
            raise ValueError("Paired export completion marker does not match this pilot")
        manifest["timings_seconds"]["digitization_conversion_export"] = elapsed
        manifest["complete"] = True
        write_json(manifest_path, manifest)
        print("Completed paired products:", output / "paired", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--signal-start", type=int, default=0)
    parser.add_argument("--stage", choices=("all", "prepare", "simulate", "recover", "export"), default="all")
    parser.add_argument("--seed-namespace", default="colliderml-higgs-portal-cells-pilot-2026-10")
    parser.add_argument("--registry", type=Path, default=Path("/cache/higgs-portal-pilot/pileup-seeds.json"))
    parser.add_argument("--signal", type=Path, default=Path("/data/higgs_portal/v2/signal/run0/events_signal.hepmc3"))
    parser.add_argument("--input-pu0", type=Path, default=Path("/data/higgs_portal/v2/pu0/run0/edm4hep.root"))
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
