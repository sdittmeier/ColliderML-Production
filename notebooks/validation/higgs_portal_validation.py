"""Read-only integrity checks and plots for the Higgs-portal paired pilot."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot

from . import paired_vs_published as paired


@dataclass
class Sample:
    name: str
    event: int
    paths: dict
    report: dict
    hits: object
    cells: object
    particles: pd.DataFrame
    analysis: object


def file_hashes(paired_root):
    result = {}
    for sample in paired.PILEUP_COLORS:
        for kind in ("hits", "cells", "particles"):
            path = Path(paired_root) / sample / f"{kind}.parquet"
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            result[str(path)] = digest.hexdigest()
    return result


def particle_frame(row):
    fields = {name: values for name, values in row.items() if name != "event_id"}
    count = len(fields["particle_id"])
    if any(len(values) != count for values in fields.values()):
        raise ValueError("Particle columns are not aligned")
    frame = pd.DataFrame(fields)
    if frame.particle_id.isna().any() or frame.particle_id.duplicated().any():
        raise ValueError("Particle IDs must be present and unique within the event")
    frame["pt"] = np.hypot(frame.px, frame.py)
    frame["production_r"] = np.hypot(frame.vx, frame.vy)
    frame["phi"] = np.arctan2(frame.py, frame.px)
    frame["eta"] = np.arcsinh(np.divide(frame.pz, frame.pt,
                                       out=np.full(count, np.nan), where=frame.pt.to_numpy() > 0))
    return frame


def load_samples(paired_root, event=0):
    result = {}
    for name in paired.PILEUP_COLORS:
        folder = Path(paired_root) / name
        paths = {kind: folder / f"{kind}.parquet" for kind in ("hits", "cells", "particles")}
        report = json.loads((folder / "report.json").read_text())
        for kind, expected in (("hits", "measurements"), ("cells", "cells"), ("particles", "events")):
            if pq.read_metadata(paths[kind]).num_rows != report[expected]:
                raise ValueError(f"{name}: {kind} row count disagrees with report.json")
        hits, cells = (paired._flat_event(paths[kind], event) for kind in ("hits", "cells"))
        if hits.num_rows == 0 or cells.num_rows == 0:
            raise ValueError(f"{name}: event {event} not present or empty")
        particles = particle_frame(paired._event_row(paths["particles"], event))
        analysis = paired.build_cluster_analysis(hits, cells, name, event)
        result[name] = Sample(name, event, paths, report, hits, cells, particles, analysis)
    return result


def integrity(samples):
    rows = []
    for name, data in samples.items():
        hits = data.hits.to_pandas()
        if hits.duplicated(paired.MEASUREMENT_KEY).any():
            raise ValueError(f"{name}: duplicate full measurement key")
        context = hits[paired.MEASUREMENT_KEY[:-1]].drop_duplicates()
        if len(context) != 1 or context.iloc[0].dataset != "higgs_portal":
            raise ValueError(f"{name}: mixed event context or wrong dataset")
        for field in ("campaign", "dataset", "version", "run"):
            if context.iloc[0][field] != data.report[field]:
                raise ValueError(f"{name}: {field} differs from report")
        known = set(data.particles.particle_id)
        lists = hits.particle_ids.tolist()
        if any(len(ids) != len(set(ids)) or not set(ids) <= known for ids in lists):
            raise ValueError(f"{name}: duplicate or missing contributor particle ID")
        scalar = hits.particle_id
        if not set(scalar.dropna().astype(np.uint64)) <= known:
            raise ValueError(f"{name}: missing scalar particle ID")
        if any(pd.notna(label) and int(label) not in ids for label, ids in zip(scalar, lists)):
            raise ValueError(f"{name}: scalar label absent from contributor list")
        if any(bool(done) != (len(ids) == 0) for done, ids in
               zip(hits.particle_ids_complete, hits.unresolved_simhit_ids)):
            raise ValueError(f"{name}: inconsistent truth-completeness flag")
        unresolved = sum(map(len, hits.unresolved_simhit_ids))
        incomplete = int((~hits.particle_ids_complete).sum())
        multi = sum(len(ids) > 1 for ids in lists)
        disagreements = sum(pd.notna(label) and int(label) not in ids
                            for label, ids in zip(hits.converted_particle_id, lists))
        measured = dict(measurements=len(hits), cells=data.cells.num_rows,
                        particles=len(data.particles), simhit_links=sum(map(len, hits.simhit_ids)),
                        null_particle_labels=int(scalar.isna().sum()),
                        multi_particle_measurements=multi, unresolved_simhit_links=unresolved,
                        incomplete_truth_measurements=incomplete,
                        converted_label_disagreements=disagreements)
        if data.report["events"] == 1:
            for key, value in measured.items():
                if data.report[key] != value:
                    raise ValueError(f"{name}: {key} differs from report ({value} vs {data.report[key]})")
        missing_parents = sum(parent >= 0 and parent not in known for parent in data.particles.parent_id)
        rows.append(dict(sample=name, event=data.event, **measured,
                         orphan_cells=0, hits_without_cells=0, missing_parent_ids=missing_parents,
                         association_checks="PASS"))
    return pd.DataFrame(rows).set_index("sample")


def quantization(samples):
    rows = []
    for name, data in samples.items():
        encoded = data.report["cell_quantization"]
        metadata = json.loads(pq.read_schema(data.paths["cells"]).metadata[b"colliderml.cell_quantization"])
        if metadata != encoded:
            raise ValueError(f"{name}: quantization metadata disagrees with report")
        cells = data.analysis.cells
        pixels = cells.volume_id.isin(encoded["pixel_volumes"])
        codes = cells.loc[pixels, "pixel_code"].to_numpy(dtype=float)
        values = cells.loc[pixels, "value"].to_numpy()
        if not np.isfinite(values).all() or (values < encoded["activation_min"]).any():
            raise ValueError(f"{name}: invalid native pixel activations")
        expected = 1 + np.searchsorted(encoded["boundaries"], values, side="right")
        if not np.array_equal(codes, expected) or cells.loc[~pixels, "pixel_code"].notna().any():
            raise ValueError(f"{name}: wrong pixel codes or encoded strip cells")
        count15 = int((codes == 15).sum())
        clipped = int((values >= encoded["code15_bin_end"]).sum())
        if data.report["events"] == 1:
            if count15 != encoded["code15_cells"] or clipped != encoded["genuinely_clipped_cells"]:
                raise ValueError(f"{name}: encoding counts differ from report")
        rows.append(dict(sample=name, scheme=encoded["scheme"], pixel_cells=len(codes),
                         code15_cells=count15, code15_fraction=count15 / len(codes),
                         genuinely_clipped_cells=clipped, clipped_fraction=clipped / len(codes),
                         native_max=float(values.max()), checks="PASS"))
    return pd.DataFrame(rows).set_index("sample")


def coverage(samples):
    rows = []
    for sample, data in samples.items():
        for kind, path in data.paths.items():
            for field in pq.read_schema(path):
                role = ("identifier / integrity" if field.name in paired.MEASUREMENT_KEY + ["event_id", "geometry_id", "parent_id"]
                        else "contributors / integrity" if field.name in ("particle_id", "particle_ids", "converted_particle_id", "simhit_ids", "unresolved_simhit_ids", "particle_ids_complete")
                        else "distribution / missingness")
                rows.append(dict(sample=sample, table=kind, feature=field.name, dtype=str(field.type), role=role))
    return pd.DataFrame(rows)


def nonfinite_rates(samples):
    rows = []
    for sample, data in samples.items():
        for kind, frame in (("particles", data.particles), ("hits", data.hits.to_pandas()),
                            ("cells", data.analysis.cells[["value", "pixel_code"]])):
            for name in frame.select_dtypes(include=["number", "bool"]).columns:
                values = frame[name].to_numpy(dtype=float)
                rows.append(dict(sample=sample, table=kind, feature=name, entries=len(values),
                                 nonfinite=int((~np.isfinite(values)).sum()),
                                 fraction=float((~np.isfinite(values)).mean())))
    return pd.DataFrame(rows)


def particle_selection(frame, cohort="charged_with_hits"):
    if cohort == "all":
        return frame
    selected = frame[(frame.charge != 0) & (frame.num_tracker_hits > 0)]
    if cohort == "generator_charged_with_hits":
        return selected[selected.primary.astype(bool)]
    if cohort == "displaced_charged_with_hits":
        return selected[selected.production_r > 10]
    if cohort != "charged_with_hits":
        raise ValueError("Unknown particle cohort")
    return selected


def overlay_grid(series, fields, heading):
    columns = 3
    fig, axes = plt.subplots((len(fields) + columns - 1) // columns, columns,
                             figsize=(15, 3.5 * ((len(fields) + columns - 1) // columns)),
                             squeeze=False, constrained_layout=True)
    for ax, field in zip(axes.flat, fields):
        values = {name: pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
                  for name, frame in series.items()}
        paired._comparison_hist(ax, values, paired.PILEUP_COLORS, field, field, log_y=True)
    for ax in list(axes.flat)[len(fields):]:
        ax.axis("off")
    fig.suptitle(heading)
    plt.show()


def plot_particles(samples, cohort="charged_with_hits"):
    frames = {name: particle_selection(data.particles, cohort) for name, data in samples.items()}
    fields = ["pt", "eta", "phi", "energy", "mass", "px", "py", "pz", "vx", "vy", "vz",
              "production_r", "perigee_d0", "perigee_z0", "time", "num_tracker_hits", "num_calo_hits"]
    overlay_grid(frames, fields, f"Particles: {cohort} (GeV, mm, ns; normalized overlays)")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, field in zip(axes.flat, ("pdg_id", "charge", "vertex_primary", "primary")):
        counts = {name: frame[field].value_counts(normalize=True) for name, frame in frames.items()}
        categories = pd.concat(counts, axis=1).fillna(0).sum(axis=1).nlargest(16).index
        x = np.arange(len(categories))
        for name, frequencies in counts.items():
            ax.plot(x, frequencies.reindex(categories, fill_value=0), "o-", color=paired.PILEUP_COLORS[name], label=name)
        ax.set_xticks(x, [str(value) for value in categories], rotation=70)
        ax.set(title=f"{field} (up to 16 most frequent categories)", ylabel="Fraction of selected particles")
        ax.legend()
    plt.show()


def plot_spatial(samples, limit=15000, seed=7, cohort="generator_charged_with_hits"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for name, data in reversed(list(samples.items())):
        color = paired.PILEUP_COLORS[name]
        particles = particle_selection(data.particles, cohort)
        subset = particles.sample(min(limit, len(particles)), random_state=seed)
        style = dict(s=9, facecolors="none", edgecolors=color, linewidths=.6, alpha=.65) if name == "pu0" else dict(s=3, color=color, alpha=.3)
        axes[0, 0].scatter(subset.vx, subset.vy, label=name, **style)
        axes[0, 1].scatter(subset.vz, subset.production_r, label=name, **style)
        hits = data.hits.select(["global_x", "global_y", "global_z"]).to_pandas()
        subset = hits.sample(min(limit, len(hits)), random_state=seed)
        axes[1, 0].scatter(subset.global_x, subset.global_y, label=name, **style)
        axes[1, 1].scatter(subset.global_z, np.hypot(subset.global_x, subset.global_y), label=name, **style)
    settings = [("Particle production x-y", "vx [mm]", "vy [mm]"),
                ("Particle production r-z", "vz [mm]", "r [mm]"),
                ("Measurements x-y", "x [mm]", "y [mm]"),
                ("Measurements r-z", "z [mm]", "r [mm]")]
    for ax, (title, x, y) in zip(axes.flat, settings):
        ax.set(title=title, xlabel=x, ylabel=y)
        ax.legend()
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[1, 0].set_aspect("equal", adjustable="datalim")
    plt.show()


def plot_hits(samples):
    frames = {}
    for name, data in samples.items():
        frame = data.hits.to_pandas()
        geom = frame.geometry_id.to_numpy(dtype=np.uint64)
        frame["volume_id"] = geom >> 56
        frame["layer_id"] = (geom >> 36) & 0xfff
        frame["surface_id"] = (geom >> 8) & 0xfffff
        frame["radius"] = np.hypot(frame.global_x, frame.global_y)
        frame["known_particles"] = frame.particle_ids.map(len)
        frame["simhit_count"] = frame.simhit_ids.map(len)
        frames[name] = frame
    overlay_grid(frames, ["global_x", "global_y", "global_z", "radius", "local0", "local1",
                          "volume_id", "layer_id", "surface_id", "known_particles", "simhit_count"],
                 "Measurements: native positions in mm; normalized overlays")


def plot_codes(samples):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for name, data in samples.items():
        cells = data.analysis.cells
        pixel = cells[cells.detector_type == "Pixels"]
        counts = pixel.pixel_code.value_counts().reindex(range(1, 16), fill_value=0)
        axes[0].step(range(1, 16), counts / len(pixel), where="mid", color=paired.PILEUP_COLORS[name], label=name)
        axes[1].scatter(pixel.value.iloc[::max(1, len(pixel) // 10000)],
                        pixel.pixel_code.iloc[::max(1, len(pixel) // 10000)],
                        s=5, alpha=.3, color=paired.PILEUP_COLORS[name], label=name)
    axes[0].set(title="Stored pixel codes", xlabel="pixel_code", ylabel="Fraction of pixel cells", xticks=range(1, 16), yscale="log")
    axes[1].set(title="Native activation to frozen code", xlabel="Activation [mm proxy]", ylabel="pixel_code", xscale="log", yticks=range(1, 16))
    for ax in axes:
        ax.legend()
    plt.show()


def cluster_gallery(data, target_sizes=(1, 2, 4, 8, 16, 32)):
    event = SimpleNamespace(cells=data.cells, new_hits=data.hits, sample=data.name, new_event=data.event)
    return paired.plot_cluster_galleries(event, target_sizes=target_sizes)


def load_perigee_validation(folder, paired_root):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    current = {"/".join(Path(path).parts[-2:]): value for path, value in file_hashes(paired_root).items()}
    expected = {"/".join(Path(path).parts[-2:]): value for path, value in summary["source_sha256"].items()}
    if current != expected:
        raise ValueError("Perigee validation report belongs to different source Parquets")
    return summary, pd.read_csv(folder / "odd_field_comparisons.csv"), pd.read_csv(folder / "uniform_field_checks.csv")


def compare_current_perigees(current, baseline, reference, sample, tolerances):
    """Compare current stored parameters, preserving the independent reference."""
    if set(current.columns) != set(baseline.columns):
        raise ValueError(f"{sample}: particle features changed")
    old = baseline.set_index("particle_id", verify_integrity=True)
    new = current.set_index("particle_id", verify_integrity=True)
    if set(old.index) != set(new.index):
        raise ValueError(f"{sample}: particle membership changed")
    new = new.reindex(old.index)
    unchanged = [column for column in old.columns if column not in ("perigee_d0", "perigee_z0")]
    if not old[unchanged].equals(new[unchanged]):
        raise ValueError(f"{sample}: non-perigee particle features changed")
    old_finite = np.isfinite(old.perigee_d0) & np.isfinite(old.perigee_z0)
    new_finite = np.isfinite(new.perigee_d0) & np.isfinite(new.perigee_z0)
    for field in ("perigee_d0", "perigee_z0"):
        if not np.array_equal(old.loc[old_finite, field], new.loc[old_finite, field]):
            raise ValueError(f"{sample}: legacy finite {field} changed")

    if reference.particle_id.duplicated().any():
        raise ValueError(f"{sample}: duplicate independent reference particle")
    selected_ids = set(particle_selection(current, "generator_charged_with_hits").particle_id)
    if set(reference.particle_id) != selected_ids:
        raise ValueError(f"{sample}: independent reference does not cover this selected event")
    # Discard old ACTS candidate/path values. The ROOT file stores no path length.
    drop = [name for name in reference.columns
            if name.startswith(("candidate_", "delta_", "writer_")) or name == "method"]
    result = reference.drop(columns=drop).copy()
    result["originally_finite"] = result.particle_id.map(old_finite).astype(bool)
    for field in ("d0", "z0"):
        result[f"candidate_{field}"] = result.particle_id.map(new[f"perigee_{field}"])
        result[f"delta_{field}"] = result[f"candidate_{field}"] - result[f"reference_{field}"]
    finite = result.particle_id.map(new_finite).astype(bool)
    recovered = finite & ~result.originally_finite
    agrees = (result.reference_status.eq("ok") &
              result.delta_d0.abs().le(tolerances["d0"]) &
              result.delta_z0.abs().le(tolerances["z0"]))
    if (recovered & ~agrees).any():
        bad = result.loc[recovered & ~agrees, "particle_id"].tolist()
        raise ValueError(f"{sample}: recovered perigees disagree with reference: {bad}")
    result["method"] = "current stored Parquet"
    result["status"] = np.select(
        [~finite, finite & agrees], ["unresolved", "pass"], default="legacy_mismatch_preserved")
    counts = dict(sample=sample, particles=len(current),
                  legacy_finite_preserved=int(old_finite.sum()),
                  selected_generator_particles=len(result),
                  selected_finite=int(finite.sum()), recovered_validated=int(recovered.sum()),
                  unresolved=int((~finite).sum()),
                  legacy_reference_mismatches=int((finite & ~agrees).sum()))
    return result, counts


def load_current_perigee_validation(samples, baseline_root, reference_root):
    """Validate current samples against the hash-verified original-event reference."""
    original, reference, uniform = load_perigee_validation(reference_root, baseline_root)
    if original.get("partial") or not uniform.reference_pass.all():
        raise ValueError("Independent reference validation is incomplete")
    roots = {data.paths["particles"].parent.parent for data in samples.values()}
    if len(roots) != 1:
        raise ValueError("Current samples must come from one paired directory")
    paired_root = roots.pop()
    before = file_hashes(paired_root)
    comparisons, counts = [], []
    for name, data in samples.items():
        provenance = data.report.get("perigee_writer") or {}
        if (provenance.get("algorithm") != "writer-recovery-v2" or
                provenance.get("preserve_legacy_finite") is not True or
                provenance.get("boundary_splitting") is not False):
            raise ValueError(f"{name}: expected recovery-only writer provenance")
        root_path = data.paths["particles"].parent / "acts/particles.root"
        with uproot.open(root_path) as source:
            metadata = json.loads(source["colliderml_perigee_writer"].member("fTitle"))
        if metadata != provenance:
            raise ValueError(f"{name}: ROOT writer provenance differs from report")
        digest = hashlib.sha256()
        with root_path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        recorded = [value for path, value in data.report["inputs"].items()
                    if Path(path).name == "particles.root"]
        if recorded != [digest.hexdigest()]:
            raise ValueError(f"{name}: ROOT particle checksum differs from report")
        baseline = particle_frame(paired._event_row(
            Path(baseline_root) / name / "particles.parquet", data.event))
        event_reference = reference[reference["sample"].eq(name) & reference.event.eq(data.event)]
        result, row = compare_current_perigees(
            data.particles, baseline, event_reference, name, original["tolerances_mm"])
        comparisons.append(result)
        row["writer"] = provenance["algorithm"]
        row["particle_file"] = str(data.paths["particles"])
        counts.append(row)
    if before != file_hashes(paired_root):
        raise ValueError("Current source Parquets changed during perigee validation")
    summary = dict(original, mode="current stored recovery-only perigees",
                   reference_source_sha256=original["source_sha256"],
                   source_sha256=before, source_files_unchanged=True)
    return summary, pd.concat(comparisons, ignore_index=True), uniform, pd.DataFrame(counts)


def perigee_validation_statistics(comparisons):
    rows = []
    for (sample, originally_finite), group in comparisons.groupby(["sample", "originally_finite"]):
        row = dict(sample=sample, cohort="Stored finite" if originally_finite else "Formerly NaN",
                   particles=len(group), passed=int(group.status.eq("pass").sum()),
                   status_counts=group.status.value_counts().to_dict())
        for field in ("d0", "z0"):
            values = np.abs(group[f"delta_{field}"].to_numpy(dtype=float)) * 1000
            values = values[np.isfinite(values)]
            for name, quantile in (("median", .5), ("p95", .95), ("max", 1.)):
                row[f"{field}_{name}_um"] = np.quantile(values, quantile) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_perigee_validation(comparisons, summary, value_label="Candidate"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for column, field in enumerate(("d0", "z0")):
        tolerance = summary["tolerances_mm"][field] * 1000
        bins = np.linspace(-3 * tolerance, 3 * tolerance, 81)
        for (sample, stored), group in comparisons.groupby(["sample", "originally_finite"]):
            residual = group[f"delta_{field}"].to_numpy(dtype=float) * 1000
            residual = residual[np.isfinite(residual)]
            if not len(residual):
                continue
            label = f"{sample}, {'stored' if stored else 'formerly NaN'} (n={len(residual)})"
            style = "-" if stored else "--"
            color = paired.PILEUP_COLORS[sample]
            axes[0, column].hist(residual, bins=bins, weights=np.ones(len(residual)) / len(residual),
                                 histtype="step", color=color, linestyle=style, linewidth=1.8, label=label)
            absolute, counts = np.unique(np.abs(residual), return_counts=True)
            survival = np.cumsum(counts[::-1])[::-1] / counts.sum()
            axes[1, column].step(absolute, survival, where="pre", color=color,
                                 linestyle=style, linewidth=1.8, label=label)
        for boundary in (-tolerance, tolerance):
            axes[0, column].axvline(boundary, color="black", linewidth=1, linestyle=":")
        axes[1, column].axvline(tolerance, color="black", linewidth=1, linestyle=":")
        axes[0, column].set(title=f"{field}: core residual (tails outside window remain in table)",
                            xlabel=f"{value_label} - independent reference [um]", ylabel="Fraction per bin", yscale="log")
        axes[1, column].set(title=f"{field}: full-range absolute residual survival",
                            xlabel="Absolute residual [um]", ylabel="Fraction at or above", yscale="log")
        axes[1, column].set_xscale("symlog", linthresh=1e-3)
        axes[1, column].set_xlim(left=0)
        for ax in axes[:, column]:
            ax.legend(fontsize=8)
    plt.show()
