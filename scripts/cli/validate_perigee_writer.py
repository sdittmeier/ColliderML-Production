#!/usr/bin/env python3
"""Exercise the installed ROOT writer on existing truth, without digitization.

Outputs are validation artifacts, not replacements for the source samples.
Run inside the configured patched ACTS container.
"""
import argparse
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pandas as pd
import uproot

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "notebooks/validation"))
sys.path.insert(0, str(REPO))
from probe_perigee import load_particle_cohort
from scripts.postprocessing.export_paired_clusters import sha256, perigee_writer_provenance


def run(args):
    import acts
    import acts.examples
    import acts.examples.root as root
    from acts.examples.odd import getOpenDataDetector

    args.output.mkdir(parents=True, exist_ok=False)
    detector = getOpenDataDetector(odd_dir=Path("/cache/odd-v4"))
    reference = pd.read_csv(args.reference / "odd_field_comparisons.csv")
    reference_summary = json.loads((args.reference / "summary.json").read_text())
    for saved_path, expected in reference_summary["source_sha256"].items():
        relative = Path(*Path(saved_path).parts[-2:])
        if sha256(args.paired_root / relative) != expected:
            raise AssertionError(f"Reference input checksum mismatch: {relative}")
    manifest = json.loads(Path("/opt/acts-arrow/perigee_patch.json").read_text())
    result = {"writer": manifest, "samples": {}}
    for invalid in (0., -1., float("nan")):
        try:
            root.RootParticleWriter(inputParticles="particles",
                filePath=str(args.output / "invalid.root"), bField=detector.field,
                writeHelixParameters=True, perigeeRecoveryMaxR=invalid,
                level=acts.logging.WARNING)
        except ValueError:
            continue
        raise AssertionError(f"Writer accepted invalid recovery radius: {invalid}")
    result["invalid_configuration_checks"] = 3
    for sample in ("pu0", "pu200"):
        source = args.paired_root / sample / "acts/particles.root"
        before = sha256(source)
        target = args.output / f"{sample}-particles.root"
        sequencer = acts.examples.Sequencer(events=1, numThreads=1,
                                             logLevel=acts.logging.WARNING,
                                             trackFpes=False)
        sequencer.addReader(root.RootParticleReader(
            filePath=str(source), outputParticles="particles", level=acts.logging.WARNING))
        sequencer.addWriter(root.RootParticleWriter(
            inputParticles="particles", filePath=str(target), bField=detector.field,
            writeHelixParameters=True, perigeeRecoveryMaxR=1160., perigeeRecoveryMaxAbsZ=3000.,
            level=acts.logging.WARNING))
        sequencer.run()
        provenance = perigee_writer_provenance(target)
        if provenance["patch_sha256"] != manifest["patch_sha256"]:
            raise AssertionError("ROOT provenance differs from installed patch")
        with uproot.open(source) as old, uproot.open(target) as new:
            if set(old["particles"].keys()) != set(new["particles"].keys()):
                raise AssertionError("ROOT branch schema changed")
            arrays = new["particles"].arrays(["particle_hash", "perigee_d0", "perigee_z0"])
            old_parameters = old["particles"].arrays(["perigee_d0", "perigee_z0"])
            perigee_names = [name for name in new["particles"].keys() if name.startswith("perigee_")]
            all_new = new["particles"].arrays(perigee_names)
            if len(arrays) != 1:
                raise AssertionError("Expected exactly one output event")
            if not np.array_equal(ak.to_numpy(old["particles"]["particle_hash"].array()[0]),
                                  ak.to_numpy(arrays.particle_hash[0])):
                raise AssertionError("Particle membership or order changed")
        with uproot.open(args.baseline_root / f"{sample}-particles.root") as baseline:
            baseline_values = baseline["particles"].arrays(["particle_hash", *perigee_names])
        if not np.array_equal(ak.to_numpy(baseline_values.particle_hash[0]),
                              ak.to_numpy(arrays.particle_hash[0])):
            raise AssertionError("Baseline particle order differs")
        baseline_finite = np.isfinite(ak.to_numpy(baseline_values.perigee_d0[0])) & np.isfinite(
            ak.to_numpy(baseline_values.perigee_z0[0]))
        for name in perigee_names:
            if not np.array_equal(ak.to_numpy(baseline_values[name][0])[baseline_finite],
                                  ak.to_numpy(all_new[name][0])[baseline_finite], equal_nan=True):
                raise AssertionError(f"{sample}: legacy finite {name} changed")
        old_finite = np.isfinite(ak.to_numpy(old_parameters.perigee_d0[0])) & np.isfinite(
            ak.to_numpy(old_parameters.perigee_z0[0]))
        new_finite = np.isfinite(ak.to_numpy(arrays.perigee_d0[0])) & np.isfinite(
            ak.to_numpy(arrays.perigee_z0[0]))
        written = pd.DataFrame({name: ak.to_numpy(arrays[name][0]) for name in arrays.fields})
        written["legacy_finite"] = baseline_finite
        cohort = load_particle_cohort(args.paired_root, sample, 0)
        joined = cohort[["particle_id", "particle_hash"]].merge(
            written, on="particle_hash", validate="one_to_one")
        joined = joined.merge(reference[reference["sample"].eq(sample)],
                               on="particle_id", validate="one_to_one")
        if len(joined) != len(cohort):
            raise AssertionError("Selected particle coverage changed")
        finite = np.isfinite(joined.perigee_d0) & np.isfinite(joined.perigee_z0)
        d0_error = (joined.perigee_d0 - joined.reference_d0).abs()
        z0_error = (joined.perigee_z0 - joined.reference_z0).abs()
        recovered = finite & ~joined.legacy_finite
        failures = joined[recovered & ((d0_error > .001) | (z0_error > .01))]
        if len(failures):
            raise AssertionError(f"{sample}: wrong perigees: {failures.particle_id.tolist()}")
        missing = joined.loc[~finite, "particle_id"].astype(int).tolist()
        if missing != ([22453] if sample == "pu200" else [713]):
            raise AssertionError(f"{sample}: unexpected missing perigees: {missing}")
        if sha256(source) != before:
            raise AssertionError("Source ROOT changed")
        result["samples"][sample] = dict(
            selected=len(joined), finite=int(finite.sum()), recovered=int(recovered.sum()),
            missing_particle_ids=missing,
            preserved_legacy_mismatch_ids=joined.loc[
                finite & joined.legacy_finite & ((d0_error > .001) | (z0_error > .01)),
                "particle_id"].astype(int).tolist(),
            max_recovered_d0_error_mm=float(d0_error[recovered].max()),
            max_recovered_z0_error_mm=float(z0_error[recovered].max()),
            exact_legacy_finite_preserved=int(baseline_finite.sum()),
            all_particles=dict(total=len(old_finite), old_finite=int(old_finite.sum()),
                new_finite=int(new_finite.sum()), gained_finite=int((~old_finite & new_finite).sum()),
                lost_finite=int((old_finite & ~new_finite).sum())),
            source_sha256=before, output_sha256=sha256(target), provenance=provenance)
        print(sample, result["samples"][sample], flush=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-root", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path,
                        help="Unpatched writer outputs replayed from the same inputs")
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
