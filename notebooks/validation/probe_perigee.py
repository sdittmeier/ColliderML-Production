"""Diagnose writer perigee failures inside the pinned ACTS container; no simulation."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot


def propagation_status(result):
    if not result.get("ok", False):
        return result.get("error", "propagation_error")
    if not result.get("has_end_parameters", False):
        return "no_end_parameters"
    values = [result.get(name, np.nan) for name in ("d0", "z0", "path_length_mm")]
    if not result.get("finite_parameters", False) or not np.isfinite(values).all():
        return "nonfinite_end_parameters"
    return "finite_result"


def load_particle_cohort(root, sample, event):
    row = next(batch.to_pylist()[0] for batch in
               pq.ParquetFile(root / sample / "particles.parquet").iter_batches(batch_size=1)
               if batch.column("event_id")[0].as_py() == event)
    particles = pd.DataFrame({name: values for name, values in row.items() if name != "event_id"})
    selected = particles[particles.primary & particles.charge.ne(0) & particles.num_tracker_hits.gt(0)].copy()
    columns = ["vx", "vy", "vz", "px", "py", "pz"]
    with uproot.open(root / sample / "acts/particles.root") as source:
        tree = source["particles"]
        entries = np.flatnonzero(tree["event_id"].array(library="np") == event)
        if len(entries) != 1:
            raise ValueError("Expected one ACTS ROOT entry per selected event")
        entry = int(entries[0])
        arrays = tree.arrays(columns + ["q", "particle_hash", "perigee_d0", "perigee_z0"],
                             entry_start=entry, entry_stop=entry + 1)
    raw = pd.DataFrame({name: ak.to_numpy(arrays[name][0]) for name in arrays.fields})
    for name in columns:
        selected[name] = selected[name].astype("float32")
        raw[name] = raw[name].astype("float32")
    raw = raw.merge(selected[columns].drop_duplicates(), on=columns, how="inner")
    joined = selected.merge(raw, on=columns, how="left", suffixes=("_parquet", "_root"),
                            validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Selected particles do not all match ACTS ROOT")
    return joined


def run(root, event, module_dir):
    import acts
    from acts.examples.odd import getOpenDataDetector
    sys.path.insert(0, str(module_dir))
    import perigee_probe

    detector = getOpenDataDetector(odd_dir=Path("/cache/odd-v4"))
    field = detector.field
    result = {}
    for sample in ("pu0", "pu200"):
        joined = load_particle_cohort(root, sample, event)
        missing = ~np.isfinite(joined.perigee_d0_parquet)
        targets = pd.concat([joined[missing], joined[~missing].head(10)])
        rows = []
        for _, particle in targets.iterrows():
            position = np.array([particle[name] for name in ("vx", "vy", "vz")], dtype=float)
            momentum = np.array([particle[name] for name in ("px", "py", "pz")], dtype=float)
            charge = float(particle.q)
            tests = {"nominal": dict(perigee_probe.sympy(field, position, momentum, charge))}
            if not np.isfinite(particle.perigee_d0_parquet):
                for direction in (-1, 1):
                    tests[f"direction_{direction}"] = dict(perigee_probe.sympy(field, position, momentum, charge, direction))
                tests["no_loop_protection"] = dict(perigee_probe.sympy(field, position, momentum, charge, 0, False))
                tests["more_steps"] = dict(perigee_probe.sympy(field, position, momentum, charge, 0, True, 10000))
                tests["eigen"] = dict(perigee_probe.eigen(field, position, momentum, charge))
                tests["forced_target"] = dict(perigee_probe.sympy(field, position, momentum, charge, 0, True, 1000, True))
                tests["forced_target_eigen"] = dict(perigee_probe.eigen(field, position, momentum, charge, 0, True, 1000, True))
                if not tests["forced_target"].get("finite_parameters", False):
                    opposite = 1 if tests["nominal"]["linear_intersection_path_mm"] < 0 else -1
                    tests["forced_target_reverse"] = dict(perigee_probe.sympy(field, position, momentum, charge, opposite, True, 1000, True))
                    tests["forced_target_more_steps"] = dict(perigee_probe.sympy(field, position, momentum, charge, 0, True, 10000, True))
                    tests["forced_target_tighter_tolerance"] = dict(perigee_probe.sympy(field, position, momentum, charge, 0, True, 1000, True, 1e-6))
                    tests["forced_target_short_steps"] = dict(perigee_probe.sympy(field, position, momentum, charge, 0, True, 1000, True, 1e-6, 10.))
                    tests["damped_target"] = dict(perigee_probe.sympy(field, position, momentum, charge, damped_target=True, surface_tolerance=1e-6))
                    tests["damped_target_eigen"] = dict(perigee_probe.eigen(field, position, momentum, charge, damped_target=True, surface_tolerance=1e-6))
            rows.append(dict(particle_id=int(particle.particle_id), pdg_id=int(particle.pdg_id),
                             particle_hash=int(particle.particle_hash),
                             stored_finite=bool(np.isfinite(particle.perigee_d0_parquet)),
                             production_position_mm=position.tolist(), momentum_GeV=momentum.tolist(),
                             charge=charge, tests=tests))
        variants = sorted({variant for row in rows for variant in row["tests"]})
        summary = {variant: dict(Counter(
            propagation_status(row["tests"][variant])
            for row in rows if not row["stored_finite"] and variant in row["tests"]))
                   for variant in variants}
        result[sample] = dict(event=event, charged_generator_particles_with_hits=len(joined),
                              stored_missing=int(missing.sum()), summary=summary, particles=rows)
        print(sample, json.dumps(summary), flush=True)
    return dict(acts_version=str(acts.__version__),
                input_precision="Stored float32 phase space replayed in the same ODD field",
                source_files_modified=False, samples=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-root", required=True, type=Path)
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--module-dir", type=Path, default=Path("/cache/perigee-probe"))
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.paired_root, args.event, args.module_dir)
    # Preserve invalid numerical results as JSON null rather than nonstandard NaN/Infinity tokens.
    report = json.loads(json.dumps(report), parse_constant=lambda _: None)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Diagnostic report:", args.report)
