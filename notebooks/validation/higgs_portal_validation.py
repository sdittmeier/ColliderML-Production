"""Read-only integrity checks and plots for the Higgs-portal paired pilot."""

from dataclasses import dataclass
from itertools import zip_longest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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


def sample_files(root, sample):
    """Resolve the original paired layout or the published homebrew layout."""
    root = Path(root)
    if (root / sample / "report.json").is_file():
        return ({kind: root / sample / f"{kind}.parquet" for kind in ("hits", "cells", "particles")},
                root / sample / "report.json")
    paths = {}
    for kind in ("hits", "cells", "particles"):
        name = f"higgs_portal_{sample}_{kind}"
        files = sorted((root / name / "data" / name).glob("*.parquet"))
        if len(files) != 1:
            raise ValueError(f"Expected one {name} shard under {root}, found {len(files)}")
        paths[kind] = files[0]
    reports = sorted(root.glob(f"provenance/run-*/{sample}_report.json"))
    if len(reports) != 1:
        raise ValueError("Select a release with exactly one run provenance directory")
    return paths, reports[0]


def production_evidence(root):
    root = Path(root)
    if (root / "complete.json").is_file():
        parent, complete = root.parent, root / "complete.json"
    else:
        files = sorted(root.glob("provenance/run-*/paired_complete.json"))
        if len(files) != 1:
            raise ValueError("Expected one paired completion report")
        complete, parent = files[0], files[0].parent
    pilot = json.loads((parent / "pilot.json").read_text())
    simulation = json.loads((parent / "simulation_validation.json").read_text())
    completion = json.loads(complete.read_text())
    if not pilot.get("complete") or not pilot.get("simulation_validated") or simulation.get("passed") is False:
        raise ValueError("Production is not complete and validated")
    if simulation.get("descendant_matching_failures"):
        raise ValueError("Simulation has unresolved descendant matching failures")
    if pilot["events"] != completion["events_per_sample"] or pilot["events"] != simulation["events"]:
        raise ValueError("Production event counts disagree")
    if set(completion["samples"]) != {"pu0", "pu200"}:
        raise ValueError("Production lacks a pileup setting")
    return pilot, completion, simulation


def file_hashes(paired_root):
    result = {}
    for sample in paired.PILEUP_COLORS:
        paths, _ = sample_files(paired_root, sample)
        for kind, path in paths.items():
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


def particle_batch_row(batch):
    """Keep nested numeric columns in NumPy, avoiding millions of Python scalars."""
    return {field.name: (batch.column(i)[0].as_py() if field.name == "event_id"
                         else batch.column(i)[0].values.to_numpy(zero_copy_only=False))
            for i, field in enumerate(batch.schema)}


def particle_event(path, event):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
        if int(batch.column("event_id")[0].as_py()) == event:
            return particle_frame(particle_batch_row(batch))
    raise ValueError(f"Particle event {event} not found in {path}")


def load_samples(paired_root, event=0, include_cells=True):
    result = {}
    for name in paired.PILEUP_COLORS:
        paths, report_path = sample_files(paired_root, name)
        report = json.loads(report_path.read_text())
        for kind, expected in (("hits", "measurements"), ("cells", "cells"), ("particles", "events")):
            if pq.read_metadata(paths[kind]).num_rows != report[expected]:
                raise ValueError(f"{name}: {kind} row count disagrees with report.json")
        hits = paired._flat_event(paths["hits"], event)
        cells = paired._flat_event(paths["cells"], event) if include_cells else None
        if hits.num_rows == 0 or (include_cells and cells.num_rows == 0):
            raise ValueError(f"{name}: event {event} not present or empty")
        particles = particle_event(paths["particles"], event)
        analysis = paired.build_cluster_analysis(hits, cells, name, event) if include_cells else None
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
        if isinstance(data.cells, pa.Table):
            validate_cell_links(data.hits, data.cells)
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
        disagreements = sum(complete and pd.notna(label) and int(label) not in ids
                            for complete, label, ids in
                            zip(hits.particle_ids_complete, hits.converted_particle_id, lists))
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
        cells = data.cells.select(["geometry_id", "value", "pixel_code"]).to_pandas()
        cells["volume_id"] = cells.geometry_id.to_numpy(dtype=np.uint64) >> 56
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
                         code15_cells=count15, code15_fraction=count15 / len(codes) if len(codes) else np.nan,
                         genuinely_clipped_cells=clipped, clipped_fraction=clipped / len(codes) if len(codes) else np.nan,
                         native_max=float(values.max()) if len(values) else np.nan, checks="PASS"))
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


def validate_cell_links(hits, cells):
    """Full-key join without building the plot-oriented per-cell truth frame."""
    keys = paired.MEASUREMENT_KEY
    contexts = hits.select(keys[:-1]).to_pandas().drop_duplicates()
    if len(contexts) == 1:
        # Same full-key check, avoiding repeated string keys on a million cells.
        # Only use numeric measurement IDs after proving every context field.
        for field in keys[:-1]:
            if not pc.all(pc.fill_null(pc.equal(cells[field], hits[field][0]), False)).as_py():
                raise ValueError("Orphan cells: full event context differs from parent hits")
        hit_index = pd.Index(hits["measurement_id"].to_numpy())
        addresses = cells.select(["measurement_id", "channel0", "channel1"]).to_pandas()
        cell_index = pd.Index(addresses.measurement_id)
        address_keys = ["measurement_id", "channel0", "channel1"]
    else:
        hit_index = pd.MultiIndex.from_frame(hits.select(keys).to_pandas())
        addresses = cells.select(keys + ["channel0", "channel1"]).to_pandas()
        cell_index = pd.MultiIndex.from_frame(addresses[keys])
        address_keys = keys + ["channel0", "channel1"]
    if not hit_index.is_unique:
        raise ValueError("Duplicate full measurement key")
    parents = hit_index.get_indexer(cell_index)
    if (parents < 0).any():
        raise ValueError("Orphan cells: missing full-key parent hit")
    if len(np.unique(parents)) != len(hits):
        raise ValueError("Hits without cells")
    if addresses.duplicated(address_keys).any():
        raise ValueError("Duplicate channel address within a cluster")
    if not np.array_equal(hits["geometry_id"].to_numpy()[parents], cells["geometry_id"].to_numpy()):
        raise ValueError("Cell geometry differs from parent hit")
    values = cells["value"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid cell activation")


def iter_flat_events(path, batch_size=65536):
    """Read each flat shard once, retaining at most one event plus one batch."""
    current, pieces = None, []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        ids = table["local_event"].to_numpy()
        if len(ids) == 0:
            continue
        if np.any(ids[1:] < ids[:-1]) or (current is not None and ids[0] < current):
            raise ValueError("Flat shard must be ordered by local_event")
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            event = int(ids[start])
            if current is not None and event != current:
                yield current, pa.concat_tables(pieces)
                pieces = []
            current = event
            pieces.append(table.slice(int(start), int(stop - start)))
    if current is not None:
        yield current, pa.concat_tables(pieces)


def perigee_coverage(particles, sample, event):
    rows = []
    for cohort in ("all", "generator_charged_with_hits", "charged_with_hits", "displaced_charged_with_hits"):
        frame = particle_selection(particles, cohort)
        for feature in ("perigee_d0", "perigee_z0"):
            finite = np.isfinite(frame[feature].to_numpy())
            rows.append(dict(sample=sample, event=event, cohort=cohort, feature=feature,
                             particles=len(frame), finite=int(finite.sum()), nonfinite=int((~finite).sum())))
    return rows


def truth_review(sample, limit=None):
    frame = sample.hits.to_pandas()
    incomplete = ~frame.particle_ids_complete
    disagreement = pd.Series([complete and pd.notna(label) and int(label) not in ids
                              for complete, label, ids in zip(frame.particle_ids_complete,
                                                             frame.converted_particle_id, frame.particle_ids)], index=frame.index)
    mask = incomplete | disagreement
    columns = paired.MEASUREMENT_KEY + ["particle_id", "converted_particle_id", "particle_ids",
                                       "particle_ids_complete", "unresolved_simhit_ids", "simhit_ids"]
    result = frame.loc[mask, columns].copy()
    result["issue"] = np.where(incomplete[mask], "incomplete contributor truth", "legacy label disagreement")
    result["sample"] = sample.name
    return result if limit is None else result.head(limit)


def scan_campaign(root, progress=True):
    """All events, all cells; sequential scans, no collection of large event frames."""
    counts, codes, coverage_rows, issues = [], [], [], []
    for name in paired.PILEUP_COLORS:
        paths, report_path = sample_files(root, name)
        report = json.loads(report_path.read_text())
        particle_rows = pq.ParquetFile(paths["particles"]).iter_batches(batch_size=1)
        seen = []
        for hit_event, cell_event, particle_batch in zip_longest(
                iter_flat_events(paths["hits"]), iter_flat_events(paths["cells"]), particle_rows):
            if hit_event is None or cell_event is None or particle_batch is None:
                raise ValueError("Hit/cell/particle event sets differ")
            row = particle_batch_row(particle_batch)
            event = int(row["event_id"])
            if hit_event[0] != event or cell_event[0] != event or event in seen:
                raise ValueError("Hit/cell/particle event IDs differ or repeat")
            seen.append(event)
            sample = Sample(name, event, paths, report, hit_event[1], cell_event[1], particle_frame(row), None)
            check = integrity({name: sample}).reset_index().iloc[0].to_dict()
            check["llps"] = int((sample.particles.pdg_id.abs() == 36).sum())
            counts.append(check)
            codes.append(dict(event=event, **quantization({name: sample}).reset_index().iloc[0].to_dict()))
            coverage_rows.extend(perigee_coverage(sample.particles, name, event))
            issues.extend(truth_review(sample).to_dict("records"))
            if progress and len(seen) % 10 == 0:
                print(f"{name}: checked {len(seen)}/{report['events']} complete events", flush=True)
            del sample, hit_event, cell_event, row, particle_batch
        if seen != list(range(report["events"])):
            raise ValueError("Shard does not contain exactly local events 0..N-1")
        measured = pd.DataFrame([r for r in counts if r["sample"] == name])
        for key in ("measurements", "cells", "particles", "simhit_links", "null_particle_labels",
                    "multi_particle_measurements", "unresolved_simhit_links", "incomplete_truth_measurements",
                    "converted_label_disagreements"):
            if int(measured[key].sum()) != report[key]:
                raise ValueError(f"{name}: all-event {key} differs from report")
        code_rows = [r for r in codes if r["sample"] == name]
        for field, expected in (("code15_cells", "code15_cells"), ("genuinely_clipped_cells", "genuinely_clipped_cells")):
            if sum(r[field] for r in code_rows) != report["cell_quantization"][expected]:
                raise ValueError(f"{name}: all-event quantization counts differ from report")
    return dict(counts=pd.DataFrame(counts), codes=pd.DataFrame(codes),
                perigees=pd.DataFrame(coverage_rows), issues=pd.DataFrame(issues))


def plot_campaign(scan):
    counts, codes, perigees = (scan[k] for k in ("counts", "codes", "perigees"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for name, color in paired.PILEUP_COLORS.items():
        frame = counts[counts["sample"] == name]
        code = codes[codes["sample"] == name]
        for ax, column in zip(axes.flat[:3], ("measurements", "cells", "particles")):
            ax.plot(frame.event, frame[column], color=color, marker=".", label=name)
            ax.set(title=column, xlabel="Local event", ylabel="Count", yscale="log")
        axes[1, 0].plot(frame.event, frame.incomplete_truth_measurements / frame.measurements,
                        color=color, marker=".", label=name)
        axes[1, 1].plot(code.event, code.code15_fraction, color=color, marker=".", label=name)
        coverage = perigees[(perigees["sample"] == name) & (perigees.cohort == "generator_charged_with_hits")
                           & (perigees.feature == "perigee_d0")]
        axes[1, 2].plot(coverage.event, coverage.nonfinite / coverage.particles.replace(0, np.nan),
                        color=color, marker=".", label=name)
    for ax, title, ylabel in zip(axes[1], ("Incomplete hit truth", "Pixel code 15", "Generator charged-with-hits d0 coverage"),
                                ("Fraction of hits", "Fraction of pixel cells", "Fraction non-finite")):
        ax.set(title=title, xlabel="Local event", ylabel=ylabel)
    for ax in axes.flat:
        ax.legend()
    plt.show()


def plot_linked_hits(samples, limit=15000, seed=7):
    """Display truth-labelled measurements; no reconstructed tracks are present."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for column, (name, sample) in enumerate(samples.items()):
        selected = particle_selection(sample.particles, "generator_charged_with_hits")
        selected = selected[selected.production_r > 10]
        ids = set(selected.particle_id)
        hits = sample.hits.to_pandas()
        hits = hits[[bool(ids.intersection(p)) for p in hits.particle_ids]]
        view = hits.sample(min(limit, len(hits)), random_state=seed)
        color = paired.PILEUP_COLORS[name]
        axes[0, column].scatter(view.global_x, view.global_y, s=3, color=color, alpha=.35)
        axes[1, column].scatter(view.global_z, np.hypot(view.global_x, view.global_y), s=3, color=color, alpha=.35)
        axes[0, column].set(title=f"{name}: hits linked to displaced generator particles", xlabel="x [mm]", ylabel="y [mm]")
        axes[1, column].set(xlabel="z [mm]", ylabel="r [mm]")
    plt.show()
