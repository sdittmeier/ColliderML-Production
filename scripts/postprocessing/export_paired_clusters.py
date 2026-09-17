#!/usr/bin/env python3
"""Export ACTS measurements, cells, and optional single-particle labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import uproot


KEY_FIELDS = ["campaign", "dataset", "version", "run", "local_event", "measurement_id"]
KEY_SCHEMA = [
    pa.field("campaign", pa.string()),
    pa.field("dataset", pa.string()),
    pa.field("version", pa.string()),
    pa.field("run", pa.uint32()),
    pa.field("local_event", pa.uint32()),
    pa.field("measurement_id", pa.uint64()),
]
HIT_SCHEMA = pa.schema(KEY_SCHEMA + [
    pa.field("geometry_id", pa.uint64()),
    pa.field("global_x", pa.float64()),
    pa.field("global_y", pa.float64()),
    pa.field("global_z", pa.float64()),
    pa.field("local0", pa.float64()),
    pa.field("local1", pa.float64()),
    pa.field("particle_id", pa.uint64()),
    pa.field("particle_ids", pa.list_(pa.uint64())),
    pa.field("simhit_ids", pa.list_(pa.uint64())),
])
CELL_SCHEMA = pa.schema(KEY_SCHEMA + [
    pa.field("geometry_id", pa.uint64()),
    pa.field("channel0", pa.int32()),
    pa.field("channel1", pa.int32()),
    pa.field("value", pa.float64()),
])
CSV_FIELDS = {
    "measurements": {"measurement_id", "geometry_id", "global_x", "global_y", "global_z", "local0", "local1"},
    "cells": {"measurement_id", "geometry_id", "channel0", "channel1", "value"},
    "measurement-simhit-map": {"measurement_id", "hit_id"},
}
ROOT_FIELDS = ["event_nr", "volume_id", "layer_id", "surface_id", "rec_gx", "rec_gy", "rec_gz", "clus_size"]
EVENT_RE = re.compile(r"event(\d+)-measurements\.csv$")
SIMHIT_FIELDS = ("event_id", "tx", "ty", "tz", "barcode_vertex_primary",
                 "barcode_vertex_secondary", "barcode_particle",
                 "barcode_generation", "barcode_sub_particle")
BARCODE_FIELDS = SIMHIT_FIELDS[4:]


def read_csv(path: Path, kind: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing ACTS {kind} file: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = CSV_FIELDS[kind] - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_revision() -> str:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-c", f"safe.directory={repo}",
                                    "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def root_measurements(path: Path) -> dict[int, list[dict]]:
    if not path.is_file():
        raise ValueError(f"Missing ROOT measurements: {path}")
    by_event = defaultdict(list)
    with uproot.open(path) as root:
        tree = root["measurements"]
        missing = set(ROOT_FIELDS) - set(tree.keys())
        if missing:
            raise ValueError(f"ROOT measurements are missing branches: {sorted(missing)}")
        for batch in tree.iterate(ROOT_FIELDS, library="np", step_size="50 MB"):
            for values in zip(*(batch[name] for name in ROOT_FIELDS)):
                row = dict(zip(ROOT_FIELDS, values))
                by_event[int(row["event_nr"])].append(row)
    return by_event


def root_simhits(path: Path) -> dict[int, list[dict]]:
    if not path.is_file():
        raise ValueError(f"Missing ACTS SimHits ROOT file: {path}")
    by_event = defaultdict(list)
    with uproot.open(path) as root:
        tree = root["hits"]
        missing = set(SIMHIT_FIELDS) - set(tree.keys())
        if missing:
            raise ValueError(f"ROOT SimHits are missing branches: {sorted(missing)}")
        for batch in tree.iterate(SIMHIT_FIELDS, library="np", step_size="50 MB"):
            for values in zip(*(batch[name] for name in SIMHIT_FIELDS)):
                row = dict(zip(SIMHIT_FIELDS, values))
                by_event[int(row["event_id"])].append(row)
    return by_event


def edm_tracker_hits(path: Path, event: int) -> list[tuple]:
    from pyedm4hep import EDM4hepEventBatch

    batch = EDM4hepEventBatch(str(path), events=(event, event + 1))
    hits = batch.get_tracker_hits_df()
    required = {"event_id", "x", "y", "z", "particle_id"}
    if not required.issubset(hits.columns):
        raise ValueError(f"Event {event}: EDM4hep tracker hits lack {sorted(required - set(hits.columns))}")
    hits = hits[hits["event_id"] == event]
    return list(hits[["x", "y", "z", "particle_id"]].itertuples(index=False, name=None))


def simhit_particle_map(event: int, simhits: list[dict], edm_hits: list[tuple],
                        requested: set[int]) -> tuple[dict[int, int], int]:
    def position(values):
        return tuple(float(np.float32(value)) for value in values)

    if any(index < 0 or index >= len(simhits) for index in requested):
        raise ValueError(f"Event {event}: SimHit association references an absent SimHit")
    by_position = defaultdict(set)
    for x, y, z, particle_id in edm_hits:
        by_position[position((x, y, z))].add(int(particle_id))

    requested_barcodes = {
        tuple(int(simhits[index][name]) for name in BARCODE_FIELDS) for index in requested
    }
    by_barcode = {}
    for row in simhits:
        barcode = tuple(int(row[name]) for name in BARCODE_FIELDS)
        if barcode not in requested_barcodes:
            continue
        candidates = by_position.get(position((row["tx"], row["ty"], row["tz"])), set())
        if len(candidates) != 1:
            continue
        particle_id = next(iter(candidates))
        previous = by_barcode.setdefault(barcode, particle_id)
        if previous != particle_id:
            raise ValueError(f"Event {event}: ACTS barcode maps to conflicting EDM4hep particles")

    mapping = {}
    fallback_count = 0
    for index in requested:
        row = simhits[index]
        candidates = by_position.get(position((row["tx"], row["ty"], row["tz"])), set())
        barcode = tuple(int(row[name]) for name in BARCODE_FIELDS)
        barcode_particle = by_barcode.get(barcode)
        if len(candidates) == 1:
            particle_id = next(iter(candidates))
            if barcode_particle != particle_id:
                raise ValueError(f"Event {event}: SimHit {index} has inconsistent barcode truth")
        elif barcode_particle is not None and (not candidates or barcode_particle in candidates):
            particle_id = barcode_particle
            fallback_count += 1
        else:
            raise ValueError(f"Event {event}: SimHit {index} has no unambiguous EDM4hep particle")
        mapping[index] = particle_id
    return mapping, fallback_count


def converted_events(path: Path, required: set[str]) -> dict[int, dict]:
    if not path.is_file():
        raise ValueError(f"Missing converted Parquet file: {path}")
    table = pq.read_table(path)
    missing = (required | {"event_id"}) - set(table.column_names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    events = {}
    for row in table.to_pylist():
        event = int(row["event_id"])
        if event in events:
            raise ValueError(f"Duplicate converted event {event} in {path}")
        events[event] = row
    return events


def aligned_particle_ids(event: int, hits: list[dict], converted: dict,
                         particles: dict) -> tuple[list[int | None], int]:
    fields = ("x", "y", "z", "volume_id", "layer_id", "surface_id", "particle_id")
    if any(not isinstance(converted[name], list) or len(converted[name]) != len(hits)
           for name in fields):
        raise ValueError(f"Event {event}: converted tracker-hit counts differ")
    particle_ids = particles["particle_id"]
    if not isinstance(particle_ids, list):
        raise ValueError(f"Event {event}: converted particle IDs are not a list")
    known = set(particle_ids)
    if len(known) != len(particle_ids):
        raise ValueError(f"Event {event}: duplicate particle IDs")
    labels = []
    for index, hit in enumerate(hits):
        geometry = int(hit["geometry_id"])
        expected = ((geometry >> 56) & 0xff, (geometry >> 36) & 0xfff,
                    (geometry >> 8) & 0xfffff)
        actual = tuple(int(converted[name][index]) for name in fields[3:6])
        if actual != expected:
            raise ValueError(f"Event {event}: converted geometry differs at measurement {index}")
        for hit_name, converted_name in (("global_x", "x"), ("global_y", "y"),
                                         ("global_z", "z")):
            if not math.isclose(hit[hit_name], float(converted[converted_name][index]),
                                rel_tol=0, abs_tol=2e-4):
                raise ValueError(f"Event {event}: converted position differs at measurement {index}")
        label = converted["particle_id"][index]
        if label is not None:
            label = int(label)
            if label not in known:
                raise ValueError(f"Event {event}: particle ID {label} is absent from particles")
        labels.append(label)
    return labels, len(known)


def export(args: argparse.Namespace) -> dict:
    if args.run < 0:
        raise ValueError("run must be nonnegative")
    if not args.edm4hep.is_file() or not args.digi_config.is_file():
        raise ValueError("EDM4hep input and digitization config must exist")
    with args.digi_config.open() as stream:
        digi = json.load(stream)
    configured_volumes = {int(entry["volume"]) for entry in digi["entries"]}
    if not configured_volumes:
        raise ValueError("Digitization config has no geometry volumes")

    converted_hits_path = getattr(args, "converted_hits", None)
    particles_path = getattr(args, "particles", None)
    simhits_root_path = getattr(args, "simhits_root", None)
    if bool(converted_hits_path) != bool(particles_path):
        raise ValueError("converted hits and particles must be supplied together")
    if simhits_root_path and not particles_path:
        raise ValueError("SimHit truth requires converted particles")
    converted_hits = (converted_events(converted_hits_path,
                      {"x", "y", "z", "volume_id", "layer_id", "surface_id", "particle_id"})
                      if converted_hits_path else None)
    particles = (converted_events(particles_path, {"particle_id"})
                 if particles_path else None)
    simhits_by_event = root_simhits(simhits_root_path) if simhits_root_path else None

    root_events = root_measurements(args.measurements_root)
    measurement_paths = sorted(args.csv_dir.glob("event*-measurements.csv"))
    if not measurement_paths:
        raise ValueError(f"No ACTS measurement CSV files in {args.csv_dir}")

    hits, cells = [], []
    null_labels = 0
    particle_count = 0
    multi_particle_measurements = 0
    simhit_truth_fallbacks = 0
    seen_events = set()
    for path in measurement_paths:
        match = EVENT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected measurement CSV filename: {path.name}")
        event = int(match.group(1))
        if event in seen_events:
            raise ValueError(f"Duplicate ACTS event {event}")
        seen_events.add(event)
        prefix = path.name.removesuffix("measurements.csv")
        measurements = read_csv(path, "measurements")
        event_cells = read_csv(args.csv_dir / f"{prefix}cells.csv", "cells")
        links = read_csv(args.csv_dir / f"{prefix}measurement-simhit-map.csv", "measurement-simhit-map")
        root_rows = root_events.get(event)
        if root_rows is None or len(root_rows) != len(measurements):
            raise ValueError(f"Event {event}: ROOT and CSV measurement counts differ")

        ids = [int(row["measurement_id"]) for row in measurements]
        if ids != list(range(len(ids))):
            raise ValueError(f"Event {event}: measurement IDs must follow ROOT row order")
        by_id = {int(row["measurement_id"]): row for row in measurements}
        cell_counts = Counter()
        simhit_ids = defaultdict(list)
        for link in links:
            measurement_id = int(link["measurement_id"])
            if measurement_id not in by_id:
                raise ValueError(f"Event {event}: orphan SimHit link {measurement_id}")
            simhit_ids[measurement_id].append(int(link["hit_id"]))
        truth_map = None
        if simhits_by_event is not None:
            requested = {simhit_id for ids in simhit_ids.values() for simhit_id in ids}
            truth_map, fallbacks = simhit_particle_map(
                event, simhits_by_event.get(event, []),
                edm_tracker_hits(args.edm4hep, event) if requested else [], requested)
            simhit_truth_fallbacks += fallbacks

        key_base = dict(campaign=args.campaign, dataset=args.dataset, version=args.version,
                        run=args.run, local_event=event)
        for row in event_cells:
            measurement_id = int(row["measurement_id"])
            if measurement_id not in by_id:
                raise ValueError(f"Event {event}: orphan cell {measurement_id}")
            geometry_id = int(row["geometry_id"])
            if geometry_id != int(by_id[measurement_id]["geometry_id"]):
                raise ValueError(f"Event {event}: cell geometry differs from measurement {measurement_id}")
            cells.append({**key_base, "measurement_id": measurement_id,
                          "geometry_id": geometry_id, "channel0": int(row["channel0"]),
                          "channel1": int(row["channel1"]), "value": float(row["value"])})
            cell_counts[measurement_id] += 1

        event_hits = []
        for measurement_id, (row, root_row) in enumerate(zip(measurements, root_rows)):
            geometry_id = int(row["geometry_id"])
            root_geometry = (int(root_row["volume_id"]), int(root_row["layer_id"]),
                             int(root_row["surface_id"]))
            csv_geometry = ((geometry_id >> 56) & 0xff, (geometry_id >> 36) & 0xfff,
                            (geometry_id >> 8) & 0xfffff)
            if root_geometry != csv_geometry:
                raise ValueError(f"Event {event}: geometry mismatch at measurement {measurement_id}")
            if cell_counts[measurement_id] != int(root_row["clus_size"]):
                raise ValueError(f"Event {event}: cluster size mismatch at measurement {measurement_id}")
            for csv_name, root_name in (("global_x", "rec_gx"), ("global_y", "rec_gy"),
                                        ("global_z", "rec_gz")):
                if not math.isclose(float(row[csv_name]), float(root_row[root_name]),
                                    rel_tol=0, abs_tol=1e-4):
                    raise ValueError(f"Event {event}: position mismatch at measurement {measurement_id}")
            event_hits.append({**key_base, "measurement_id": measurement_id,
                         "geometry_id": geometry_id,
                         **{name: float(row[name]) for name in ("global_x", "global_y", "global_z", "local0", "local1")},
                         "simhit_ids": simhit_ids[measurement_id]})
        if converted_hits is not None:
            if event not in converted_hits or event not in particles:
                raise ValueError(f"Event {event}: missing converted hits or particles")
            labels, count = aligned_particle_ids(event, event_hits, converted_hits[event],
                                                 particles[event])
            for hit, label in zip(event_hits, labels):
                hit["particle_id"] = label
            particle_count += count
        if truth_map is not None:
            known_particles = set(particles[event]["particle_id"])
            for hit in event_hits:
                particle_ids = list(dict.fromkeys(truth_map[index] for index in hit["simhit_ids"]))
                if not set(particle_ids).issubset(known_particles):
                    raise ValueError(f"Event {event}: SimHit truth references an absent particle")
                prior_label = hit.get("particle_id")
                if prior_label is not None and prior_label not in particle_ids:
                    raise ValueError(f"Event {event}: converted label disagrees with SimHit truth")
                hit["particle_ids"] = particle_ids
                hit["particle_id"] = particle_ids[0] if len(particle_ids) == 1 else None
                multi_particle_measurements += len(particle_ids) > 1
        null_labels += sum(hit.get("particle_id") is None for hit in event_hits)
        hits.extend(event_hits)

    if set(root_events) != seen_events:
        raise ValueError("ROOT and CSV event sets differ")
    if converted_hits is not None and (set(converted_hits) != seen_events or set(particles) != seen_events):
        raise ValueError("Converted hit, particle, and ACTS event sets differ")
    observed_volumes = {(hit["geometry_id"] >> 56) & 0xff for hit in hits}
    if not observed_volumes.issubset(configured_volumes):
        raise ValueError(f"Digitization config lacks observed volumes: {sorted(observed_volumes - configured_volumes)}")
    if not hits or not cells:
        raise ValueError("Digitization produced no hits or no cells")

    report = {
        "campaign": args.campaign, "dataset": args.dataset, "version": args.version,
        "run": args.run, "events": len(seen_events), "local_events": sorted(seen_events),
        "measurements": len(hits),
        "cells": len(cells), "simhit_links": sum(len(hit["simhit_ids"]) for hit in hits),
        "multi_contributor_measurements": sum(len(hit["simhit_ids"]) > 1 for hit in hits),
        "particles": particle_count if particles is not None else None,
        "null_particle_labels": null_labels if particles is not None else None,
        "multi_particle_measurements": multi_particle_measurements if simhits_root_path else None,
        "simhit_truth_barcode_fallbacks": simhit_truth_fallbacks if simhits_root_path else None,
        "revisions": {"colliderml_production": repo_revision(), "acts": args.acts_revision,
                      "odd": args.odd_revision},
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2])): sha256(path)
                          for path in (Path(__file__).resolve(),
                                       Path(__file__).resolve().parent / "run_paired_samples.py",
                                       Path(__file__).resolve().parent / "convert_all.py",
                                       Path(__file__).resolve().parent / "convert_particles.py",
                                       Path(__file__).resolve().parent / "convert_digihits.py",
                                       Path(__file__).resolve().parent / "utils/parquet_utils.py",
                                       Path(__file__).resolve().parents[1] / "simulation/digi_and_reco.py",
                                       Path(__file__).resolve().parents[2] /
                                       "configs_development/paired_clusters/digitization.yaml")},
        "inputs": {str(path): sha256(path) for path in
                   (args.edm4hep, args.digi_config, args.measurements_root,
                    *([converted_hits_path, particles_path] if converted_hits_path else []),
                    *([simhits_root_path] if simhits_root_path else []))},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(hits, schema=HIT_SCHEMA), args.output / "hits.parquet")
    pq.write_table(pa.Table.from_pylist(cells, schema=CELL_SCHEMA), args.output / "cells.parquet")
    if particles_path:
        shutil.copyfile(particles_path, args.output / "particles.parquet")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "dataset", "version", "acts-revision", "odd-revision"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--run", type=int, required=True)
    for name in ("csv-dir", "measurements-root", "edm4hep", "digi-config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--converted-hits", type=Path)
    parser.add_argument("--particles", type=Path)
    parser.add_argument("--simhits-root", type=Path)
    print(json.dumps(export(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
