"""Compare stored/candidate perigees with independent analytic and ODD references."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from perigee_reference import analytic_perigee, reference_perigee, slab_solenoid_perigee
from probe_perigee import load_particle_cohort, propagation_status


D0_TOLERANCE_MM = 1e-3
Z0_TOLERANCE_MM = 1e-2
REFERENCE_D0_TOLERANCE_MM = 2e-4
REFERENCE_Z0_TOLERANCE_MM = 2e-3


def refine_reference(position, momentum, charge, field, first):
    previous = first
    last_d0, last_z0 = np.nan, np.nan
    for level, (rtol, atol, scale) in enumerate(((2e-11, 1e-10, .5),
                                               (2e-13, 1e-12, .25),
                                               (3e-14, 1e-13, .125)), start=1):
        current = reference_perigee(position, momentum, charge, field, rtol=rtol, atol=atol, step_scale=scale)
        if current["status"] != "ok":
            return previous, False, level, last_d0, last_z0
        last_d0, last_z0 = current["d0"] - previous["d0"], current["z0"] - previous["z0"]
        if (abs(last_d0) < REFERENCE_D0_TOLERANCE_MM and
                abs(last_z0) < REFERENCE_Z0_TOLERANCE_MM and
                abs(current["path_length_mm"] - previous["path_length_mm"]) < .005 + 1e-6 * abs(current["path_length_mm"])):
            return current, True, level, last_d0, last_z0
        previous = current
    return previous, False, level, last_d0, last_z0


def source_hashes(root):
    result = {}
    for sample in ("pu0", "pu200"):
        for kind in ("hits", "cells", "particles"):
            path = root / sample / f"{kind}.parquet"
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            result[str(path)] = digest.hexdigest()
    return result


def candidate_from_probe(particle):
    for name in ("forced_target", "forced_target_tighter_tolerance", "damped_target"):
        candidate = particle.get("tests", {}).get(name, {})
        if propagation_status(candidate) == "finite_result":
            return name, candidate
    return "no_candidate", {}


def uniform_checks(acts, probe, writer_patch=False):
    cases = [([.01, -.02, 10.], [1., .4, 2.]),
             ([100., -40., 200.], [.3, .4, 2.]),
             ([-58., -26., -1454.], [-.02, .1, -12.]),
             ([40., 20., 100.], [.005, .003, .2]),
             ([150., -250., 50.], [-.6, -.8, .2])]
    rows = []
    for index, (position, momentum) in enumerate(cases):
        for bz in (2., 4.):
            field = acts.ConstantBField(acts.Vector3(0., 0., bz * acts.UnitConstants.T))
            for charge in (-1., 1.):
                exact = analytic_perigee(position, momentum, charge, bz)
                reference = reference_perigee(position, momentum, charge, lambda _: np.array([0., 0., bz]))
                candidate = dict(probe.sympy(field, position, momentum, charge,
                                            forced_target=True, surface_tolerance=1e-6))
                method = "forced_target"
                if propagation_status(candidate) != "finite_result":
                    candidate = dict(probe.sympy(field, position, momentum, charge,
                                                damped_target=True, surface_tolerance=1e-6))
                    method = "damped_target"
                if writer_patch:
                    candidate = dict(probe.writer_perigee(field, position, momentum, charge))
                    method = "patched_writer"
                row = dict(case=index, bz_T=bz, charge=charge, method=method,
                           reference_status=reference["status"], candidate_status=propagation_status(candidate))
                for key in ("d0", "z0", "path_length_mm"):
                    row[f"analytic_{key}"] = exact[key]
                    row[f"reference_delta_{key}"] = reference.get(key, np.nan) - exact[key]
                    row[f"candidate_delta_{key}"] = candidate.get(key, np.nan) - exact[key]
                row["reference_pass"] = (reference["status"] == "ok" and
                                         abs(row["reference_delta_d0"]) < REFERENCE_D0_TOLERANCE_MM and
                                         abs(row["reference_delta_z0"]) < REFERENCE_Z0_TOLERANCE_MM)
                row["candidate_pass"] = (propagation_status(candidate) == "finite_result" and
                                         abs(row["candidate_delta_d0"]) < D0_TOLERANCE_MM and
                                         abs(row["candidate_delta_z0"]) < Z0_TOLERANCE_MM and
                                         abs(row["candidate_delta_path_length_mm"]) < .01 + 1e-5 * abs(exact["path_length_mm"]))
                rows.append(row)
    return pd.DataFrame(rows)


def boundary_cross_checks(comparisons, root, event):
    rows = []
    for sample in ("pu0", "pu200"):
        failures = comparisons[(comparisons["sample"] == sample) & comparisons.status.eq("parameter_mismatch")]
        if failures.empty:
            continue
        particles = load_particle_cohort(root, sample, event).set_index("particle_id")
        for _, item in failures.iterrows():
            particle = particles.loc[int(item.particle_id)]
            position = np.array([particle[key] for key in ("vx", "vy", "vz")], float)
            momentum = np.array([particle[key] for key in ("px", "py", "pz")], float)
            if abs(item.reference_z0) <= 3000.:
                continue
            exact = slab_solenoid_perigee(position, momentum, float(particle.q))
            delta_d0 = item.reference_d0 - exact["d0"]
            delta_z0 = item.reference_z0 - exact["z0"]
            rows.append(dict(sample=sample, particle_id=int(item.particle_id), region=exact["region"],
                             analytic_d0=exact["d0"], analytic_z0=exact["z0"],
                             numerical_minus_analytic_d0=delta_d0,
                             numerical_minus_analytic_z0=delta_z0,
                             reference_pass=abs(delta_d0) < REFERENCE_D0_TOLERANCE_MM and abs(delta_z0) < REFERENCE_Z0_TOLERANCE_MM))
    return pd.DataFrame(rows)


def run(args):
    import acts
    from acts.examples.odd import getOpenDataDetector
    sys.path.insert(0, str(args.module_dir))
    import perigee_probe

    args.output.mkdir(parents=True, exist_ok=False)
    before = source_hashes(args.paired_root)
    uniform = uniform_checks(acts, perigee_probe, getattr(args, "writer_patch", False))
    uniform.to_csv(args.output / "uniform_field_checks.csv", index=False)
    print("Uniform reference:", int(uniform.reference_pass.sum()), "/", len(uniform),
          "ACTS candidates:", int(uniform.candidate_pass.sum()), "/", len(uniform), flush=True)
    if not uniform.reference_pass.all():
        raise ValueError("Independent numerical reference failed its analytic tests")

    detector = getOpenDataDetector(odd_dir=Path("/cache/odd-v4"))
    field = detector.field
    lookup = lambda position: perigee_probe.field_tesla(field, position)
    probe_report = json.loads(args.probe_report.read_text())
    results = []
    for sample in ("pu0", "pu200"):
        particles = load_particle_cohort(args.paired_root, sample, args.event)
        if args.limit_per_sample:
            particles = particles.sort_values("perigee_d0_parquet", na_position="first").head(args.limit_per_sample)
        probe_by_id = {item["particle_id"]: item for item in probe_report["samples"][sample]["particles"]}
        finite_controls = 0
        for index, (_, particle) in enumerate(particles.iterrows()):
            position = np.array([particle[key] for key in ("vx", "vy", "vz")], float)
            momentum = np.array([particle[key] for key in ("px", "py", "pz")], float)
            stored_finite = bool(np.isfinite(particle.perigee_d0_parquet) and np.isfinite(particle.perigee_z0_parquet))
            if stored_finite:
                method = "stored"
                candidate = dict(d0=float(particle.perigee_d0_parquet), z0=float(particle.perigee_z0_parquet))
            else:
                method, candidate = candidate_from_probe(probe_by_id.get(int(particle.particle_id), {}))
            if getattr(args, "writer_patch", False):
                candidate = dict(perigee_probe.writer_perigee(field, position, momentum, float(particle.q), 1160., 3000.))
                method = "patched_writer" if propagation_status(candidate) == "finite_result" else "no_candidate"
            reference = reference_perigee(position, momentum, float(particle.q), lookup)
            first_z0 = reference.get("z0", np.nan)
            potential_mismatch = (abs(candidate.get("d0", np.nan) - reference.get("d0", np.nan)) > D0_TOLERANCE_MM or
                                  abs(candidate.get("z0", np.nan) - reference.get("z0", np.nan)) > Z0_TOLERANCE_MM)
            check_convergence = not stored_finite or potential_mismatch or finite_controls < 10
            if stored_finite and finite_controls < 10:
                finite_controls += 1
            convergence = None
            if check_convergence and reference["status"] == "ok":
                reference, converged, levels, change_d0, change_z0 = refine_reference(
                    position, momentum, float(particle.q), lookup, reference)
                convergence = dict(reference_converged=converged, refinement_levels=levels,
                                   reference_delta_tighter_d0=change_d0,
                                   reference_delta_tighter_z0=change_z0,
                                   reference_first_to_final_z0=reference["z0"] - first_z0)
            row = dict(sample=sample, event=args.event, particle_id=int(particle.particle_id),
                       pdg_id=int(particle.pdg_id), pt_GeV=float(np.hypot(*momentum[:2])),
                       production_r_mm=float(np.hypot(*position[:2])),
                       originally_finite=stored_finite, method=method,
                       writer_error=candidate.get("error", ""),
                       writer_boundary_crossings=candidate.get("boundaries", 0),
                       stored_d0=float(particle.perigee_d0_parquet), stored_z0=float(particle.perigee_z0_parquet),
                       reference_status=reference["status"], momentum_drift=reference["momentum_drift"],
                       reference_field_calls=reference["field_calls"])
            for key in ("d0", "z0", "path_length_mm"):
                row[f"candidate_{key}"] = candidate.get(key, np.nan)
                row[f"reference_{key}"] = reference.get(key, np.nan)
                row[f"delta_{key}"] = candidate.get(key, np.nan) - reference.get(key, np.nan)
            row["reference_curvature"] = reference.get("curvature", np.nan)
            row["status"] = ("reference_failure" if reference["status"] != "ok" else
                             "no_candidate" if method == "no_candidate" else
                             "parameter_mismatch" if abs(row["delta_d0"]) > D0_TOLERANCE_MM or abs(row["delta_z0"]) > Z0_TOLERANCE_MM else
                             "branch_mismatch" if "path_length_mm" in candidate and
                             abs(row["delta_path_length_mm"]) > .01 + 1e-5 * abs(reference["path_length_mm"]) else "pass")
            if convergence is not None:
                row.update(convergence)
                if not row["reference_converged"]:
                    row["status"] = "reference_not_converged"
            results.append(row)
            if index % 250 == 0:
                print(sample, index + 1, "/", len(particles), "last status", row["status"], flush=True)
        print(sample, pd.DataFrame(results).query("sample == @sample").status.value_counts().to_dict(), flush=True)
    results = pd.DataFrame(results)
    results.to_csv(args.output / "odd_field_comparisons.csv", index=False)
    # Confirm the pinned slab model before applying the additional exact check.
    for z, expected in ((0., 2.), (2999.999, 2.), (3000.001, 0.), (-3000.001, 0.)):
        if not np.allclose(lookup(np.array([50., 0., z])), [0., 0., expected], atol=1e-12, rtol=0):
            raise ValueError("ODD field differs from the pinned solenoid boundary model")
    boundary = boundary_cross_checks(results, args.paired_root, args.event)
    boundary.to_csv(args.output / "solenoid_boundary_checks.csv", index=False)
    after = source_hashes(args.paired_root)
    if before != after:
        raise ValueError("Source Parquet contents changed")
    summary = dict(branch_convention="Nearest local transverse-distance minimum in absolute path; negative-path tie break; finite search bound",
                   reference="SciPy DOP853 Lorentz integration and event-root search in queried ODD field; no ACTS stepper",
                   input_precision="Stored float32 phase space",
                   tolerances_mm=dict(d0=D0_TOLERANCE_MM, z0=Z0_TOLERANCE_MM,
                                      reference_d0=REFERENCE_D0_TOLERANCE_MM, reference_z0=REFERENCE_Z0_TOLERANCE_MM),
                   uniform_cases=len(uniform), uniform_reference_pass=int(uniform.reference_pass.sum()),
                   uniform_candidate_pass=int(uniform.candidate_pass.sum()), partial=bool(args.limit_per_sample),
                   solenoid_boundary_checks=boundary.to_dict("records"),
                   source_sha256=before, source_files_unchanged=True,
                   counts=results.groupby(["sample", "originally_finite", "status"]).size().rename("count").reset_index().to_dict("records"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Validation report:", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, default=Path("/cache/perigee-probe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event", type=int, default=0)
    parser.add_argument("--limit-per-sample", type=int)
    parser.add_argument("--writer-patch", action="store_true", help="Validate the production patch helper on every selected particle")
    run(parser.parse_args())
