#!/usr/bin/env python3
"""Build paired ttbar hit, cell, and particle samples from two EDM4hep inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import uproot
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
PAIRED_CONFIG = REPO / "configs_development/paired_clusters/digitization.yaml"
DIGI_CONFIG = REPO / "scripts/simulation/odd-full-geo-digi-config.json"
SAMPLES = (("pu0", "hard_scatter"), ("pu200", "full_pileup"))
PINNED_ACTS_PREFIX = "7cba36b17"
PINNED_ODD_REVISION = "b0992c148305224899a16d21fcd56406408bd393"


def revision(path: Path) -> str:
    return subprocess.check_output(["git", "-c", f"safe.directory={path}",
                                    "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def input_events(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"Missing EDM4hep input: {path}")
    with uproot.open(path) as root:
        return int(root["events"].num_entries)


def one_parquet(directory: Path) -> Path:
    files = list(directory.glob("*.parquet"))
    if len(files) != 1:
        raise ValueError(f"Expected one Parquet file in {directory}, found {len(files)}")
    return files[0]


def validate_particle_features(path: Path) -> None:
    required = {"perigee_d0", "perigee_z0", "vertex_primary"}
    schema = pq.read_schema(path)
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"Converted particles lack ACTS fields: {sorted(missing)}")
    table = pq.read_table(path, columns=["particle_id", *sorted(required)])
    counts = pc.list_value_length(table["particle_id"]).to_pylist()
    for name in sorted(required):
        values = table[name]
        if pc.list_value_length(values).to_pylist() != counts:
            raise ValueError(f"Converted particle field {name} is not aligned with particle_id")
        flat = pc.list_flatten(values)
        if len(flat) == flat.null_count:
            raise ValueError(f"Converted particle field {name} has no non-null values")


def run_command(*parts: str) -> None:
    print("Running:", " ".join(parts), flush=True)
    subprocess.run(parts, check=True, cwd=REPO)


def run(args: argparse.Namespace) -> dict:
    if args.events < 1:
        raise ValueError("--events must be positive")
    output = args.output.resolve()
    resume_export = getattr(args, "resume_export", False)
    if output.exists() and not resume_export:
        raise ValueError(f"Output already exists: {output}; choose a new directory")
    if resume_export and (not output.is_dir() or (output / "complete.json").exists()):
        raise ValueError("--resume-export requires an incomplete existing output directory")
    if not PAIRED_CONFIG.is_file() or not args.digi_config.is_file():
        raise ValueError("Paired ACTS config and geometric digitization config must exist")
    with args.digi_config.open() as stream:
        digi = json.load(stream)
    if not digi.get("entries"):
        raise ValueError("Geometric digitization config has no entries")

    inputs = {"pu0": args.input_pu0.resolve(), "pu200": args.input_pu200.resolve()}
    for sample, path in inputs.items():
        available = input_events(path)
        if available < args.events:
            raise ValueError(f"{sample} has {available} events; requested {args.events}")

    acts_revision = revision(args.acts_source)
    odd_revision = revision(args.odd_source)
    if not acts_revision.startswith(PINNED_ACTS_PREFIX):
        raise ValueError(f"ACTS revision {acts_revision} does not match {PINNED_ACTS_PREFIX}")
    if odd_revision != PINNED_ODD_REVISION:
        raise ValueError(f"ODD revision {odd_revision} does not match {PINNED_ODD_REVISION}")
    if not (args.odd_install / "lib/libOpenDataDetector.so").is_file():
        raise ValueError(f"ODD installation not found at {args.odd_install}")

    if not resume_export:
        output.mkdir(parents=True)
    reports = {}
    for sample, campaign in SAMPLES:
        sample_dir = output / sample
        acts_dir = sample_dir / "acts"
        conversion_input = output / "_conversion_input" / sample / "runs" / "0"
        conversion_output = output / "_conversion_results"
        config_path = output / f"_convert_{sample}.json"
        if resume_export:
            config = json.loads(config_path.read_text())
            if (config["run_size"] != args.events or config["chunk_size"] != args.events
                    or config["campaign"] != campaign
                    or (conversion_input / "edm4hep.root").resolve() != inputs[sample]
                    or (conversion_input / "measurements.root").resolve() != acts_dir / "measurements.root"):
                raise ValueError(f"{sample}: existing conversion inputs do not match this run")
            for path in (acts_dir / "measurements.root", acts_dir / "simhits.root",
                         acts_dir / "csv"):
                if not path.exists():
                    raise ValueError(f"{sample}: missing ACTS output {path}")
        else:
            run_command(sys.executable, str(REPO / "scripts/simulation/digi_and_reco.py"),
                        "--config", str(PAIRED_CONFIG), "--events", str(args.events),
                        "--threads", "1", "--digi-config", str(args.digi_config),
                        "--input-file", str(inputs[sample]), "--output", str(acts_dir))
            conversion_input.mkdir(parents=True)
            (conversion_input / "edm4hep.root").symlink_to(inputs[sample])
            (conversion_input / "measurements.root").symlink_to(acts_dir / "measurements.root")
            (conversion_input / "particles.root").symlink_to(acts_dir / "particles.root")
            config = {
                "campaign": campaign, "dataset": "ttbar", "version": "v1",
                "common": {"output_base_dir": str(conversion_output)},
                "input_base_dir": str(conversion_input.parents[1]),
                "h5_output_dir": str(conversion_output),
                "objects": ["particles", "tracker_hits"], "output_format": "parquet",
                "preserve_unmatched_particle_id": True,
                "preserve_particles_without_acts_match": True,
                "run_size": args.events, "chunk_size": args.events, "max_chunks": 1,
                "log_level": "INFO",
            }
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            run_command(sys.executable, str(REPO / "scripts/postprocessing/convert_all.py"),
                        "--config", str(config_path), "--chunk-index", "0")

        parquet_root = conversion_output / campaign / "ttbar/v1/parquet"
        converted_hits = one_parquet(parquet_root / "reco/tracker_hits")
        particles = one_parquet(parquet_root / "truth/particles")
        validate_particle_features(particles)
        run_command(sys.executable, str(REPO / "scripts/postprocessing/export_paired_clusters.py"),
                    "--campaign", campaign, "--dataset", "ttbar", "--version", "v1",
                    "--run", "0", "--acts-revision", acts_revision, "--odd-revision", odd_revision,
                    "--edm4hep", str(inputs[sample]), "--digi-config", str(args.digi_config),
                    "--measurements-root", str(acts_dir / "measurements.root"),
                    "--simhits-root", str(acts_dir / "simhits.root"),
                    "--csv-dir", str(acts_dir / "csv"), "--converted-hits", str(converted_hits),
                    "--particles", str(particles), "--output", str(sample_dir))
        report = json.loads((sample_dir / "report.json").read_text())
        if report["events"] != args.events or report["local_events"] != list(range(args.events)):
            raise ValueError(f"{sample}: exported events {report['local_events']}, "
                             f"expected {list(range(args.events))}")
        reports[sample] = report

    summary = {"events_per_sample": args.events, "samples": reports}
    (output / "complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pu0", type=Path, default=Path("/data/pu0/edm4hep.root"))
    parser.add_argument("--input-pu200", type=Path, default=Path("/data/pu200/edm4hep.root"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--resume-export", action="store_true",
                        help="Reuse ACTS and converted outputs from an incomplete run")
    parser.add_argument("--digi-config", type=Path, default=DIGI_CONFIG)
    parser.add_argument("--acts-source", type=Path, default=Path("/opt/acts-arrow-src"))
    parser.add_argument("--odd-source", type=Path, default=Path("/cache/odd-v4"))
    parser.add_argument("--odd-install", type=Path, default=Path("/cache/odd-v4-install"))
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
