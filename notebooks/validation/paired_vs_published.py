"""Load and visualize one paired event against one independent public event."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


NEW_COLOR = "#1672b8"
PUBLIC_COLOR = "#df7926"
PARTICLE_FIELDS = ("pdg_id", "mass", "energy", "charge", "vx", "vy", "vz",
                   "time", "px", "py", "pz", "perigee_d0", "perigee_z0",
                   "vertex_primary", "parent_id", "primary")
HIT_FIELDS = ("x", "y", "z", "volume_id", "layer_id", "surface_id", "particle_id")


@dataclass
class EventData:
    sample: str
    new_event: int
    published_event: int
    new_particles: dict
    public_particles: dict
    new_hits: pa.Table
    public_hits: dict
    cells: pa.Table
    new_schemas: dict[str, pa.Schema]
    public_schemas: dict[str, pa.Schema]
    report: dict
    new_paths: dict[str, Path]
    public_paths: dict[str, Path]


def _event_row(path: Path, event: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    if "event_id" not in parquet.schema_arrow.names:
        raise ValueError(f"No event_id in {path}")
    for batch in parquet.iter_batches(batch_size=1):
        if int(batch.column("event_id")[0].as_py()) == event:
            return batch.to_pylist()[0]
    raise ValueError(f"Event {event} not found in {path}")


def _published_shard(root: Path, sample: str, kind: str, event: int) -> Path:
    name = f"ttbar_{sample}_{kind}"
    files = sorted((root / name / "data" / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No published {name} shards under {root}")
    for path in files:
        event_ids = pq.read_table(path, columns=["event_id"])["event_id"].to_pylist()
        if event in event_ids:
            return path
    raise ValueError(f"Published {name} event {event} not found in available shards")


def _flat_event(path: Path, event: int) -> pa.Table:
    if not path.is_file():
        raise FileNotFoundError(path)
    return ds.dataset(path, format="parquet").to_table(filter=ds.field("local_event") == event)


def load_event(new_root: Path, published_root: Path, sample: str = "pu0",
               new_event: int = 0, published_event: int = 0) -> EventData:
    if sample not in {"pu0", "pu200"}:
        raise ValueError("sample must be 'pu0' or 'pu200'")
    folder = Path(new_root) / sample
    paths = {kind: folder / f"{kind}.parquet" for kind in ("particles", "hits", "cells")}
    public_paths = {kind: _published_shard(Path(published_root), sample, kind, published_event)
                    for kind in ("particles", "tracker_hits")}
    particles = _event_row(paths["particles"], new_event)
    public_particles = _event_row(public_paths["particles"], published_event)
    new_hits = _flat_event(paths["hits"], new_event)
    cells = _flat_event(paths["cells"], new_event)
    public_hits = _event_row(public_paths["tracker_hits"], published_event)
    if new_hits.num_rows == 0 or cells.num_rows == 0:
        raise ValueError(f"New {sample} event {new_event} has no hits or cells")
    import json
    report = json.loads((folder / "report.json").read_text())
    expected = {"particles": report["events"], "hits": report["measurements"],
                "cells": report["cells"]}
    for kind, path in paths.items():
        rows = pq.read_metadata(path).num_rows
        if rows != expected[kind]:
            raise ValueError(f"{kind}: {rows} Parquet rows, {expected[kind]} in report.json")
    return EventData(sample, new_event, published_event, particles, public_particles,
                     new_hits, public_hits, cells,
                     {kind: pq.read_schema(path) for kind, path in paths.items()},
                     {kind: pq.read_schema(path) for kind, path in public_paths.items()},
                     report, paths, public_paths)


def _arrow_values(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist())


def _numeric(values) -> np.ndarray:
    result = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return result[np.isfinite(result)]


def _geom(table: pa.Table, shift: int, mask: int) -> np.ndarray:
    return (_arrow_values(table, "geometry_id").astype(np.uint64) >> shift) & mask


def _new_hit_feature(data: EventData, name: str) -> np.ndarray:
    if name in {"x", "y", "z"}:
        return _arrow_values(data.new_hits, f"global_{name}")
    if name == "volume_id":
        return _geom(data.new_hits, 56, 0xff)
    if name == "layer_id":
        return _geom(data.new_hits, 36, 0xfff)
    if name == "surface_id":
        return _geom(data.new_hits, 8, 0xfffff)
    return _arrow_values(data.new_hits, name)


def validate_event(data: EventData) -> pd.DataFrame:
    particle_ids = data.new_particles["particle_id"]
    known = set(particle_ids)
    if len(known) != len(particle_ids):
        raise AssertionError("Duplicate new particle IDs")
    for name, values in data.new_particles.items():
        if name != "event_id" and len(values) != len(particle_ids):
            raise AssertionError(f"Particle field {name} is not aligned")
    for name, values in data.public_particles.items():
        if name != "event_id" and len(values) != len(data.public_particles["particle_id"]):
            raise AssertionError(f"Published particle field {name} is not aligned")
    public_hit_count = len(data.public_hits["x"])
    for name, values in data.public_hits.items():
        if name != "event_id" and len(values) != public_hit_count:
            raise AssertionError(f"Published hit field {name} is not aligned")
    hit_ids = _arrow_values(data.new_hits, "measurement_id")
    cell_ids = _arrow_values(data.cells, "measurement_id")
    if len(np.unique(hit_ids)) != len(hit_ids):
        raise AssertionError("Duplicate measurement IDs")
    orphan_cells = int(np.count_nonzero(~np.isin(cell_ids, hit_ids)))
    hit_geometry = dict(zip(hit_ids.tolist(), _arrow_values(data.new_hits, "geometry_id").tolist()))
    cell_geometry = _arrow_values(data.cells, "geometry_id")
    wrong_cell_geometry = sum(hit_geometry.get(mid) != gid for mid, gid in zip(cell_ids, cell_geometry))
    labels = data.new_hits["particle_id"].to_pylist()
    contributors = data.new_hits["particle_ids"].to_pylist()
    unmatched_labels = sum(label is not None and label not in known for label in labels)
    unmatched_contributors = sum(pid not in known for ids in contributors for pid in (ids or []))
    if orphan_cells or wrong_cell_geometry or unmatched_labels or unmatched_contributors:
        raise AssertionError("Broken cell or particle association")
    simhit_ids = data.new_hits["simhit_ids"].to_pylist()
    unresolved = data.new_hits["unresolved_simhit_ids"].to_pylist()
    complete = data.new_hits["particle_ids_complete"].to_pylist()
    converted_labels = data.new_hits["converted_particle_id"].to_pylist()
    converted_disagreements = sum(label is not None and done and label not in (ids or [])
                                  for label, done, ids in zip(converted_labels, complete, contributors))
    rows = [
        ("Events in report", data.report["events"], None),
        ("Particles", len(particle_ids), len(data.public_particles["particle_id"])),
        ("Hits", len(hit_ids), public_hit_count),
        ("Cells", len(cell_ids), None),
        ("Orphan cells", orphan_cells, None),
        ("Wrong cell geometry", wrong_cell_geometry, None),
        ("Unmatched hit labels", unmatched_labels, None),
        ("Unmatched contributor IDs", unmatched_contributors, None),
        ("Null scalar hit labels", sum(label is None for label in labels),
         sum(label is None for label in data.public_hits["particle_id"])),
        ("Multi-SimHit measurements", sum(len(ids or []) > 1 for ids in simhit_ids), None),
        ("Multi-particle measurements", sum(len(ids or []) > 1 for ids in contributors), None),
        ("Unresolved SimHit links", sum(len(ids or []) for ids in unresolved), None),
        ("Incomplete truth measurements", sum(not value for value in complete), None),
        ("Converted-label disagreements", converted_disagreements, None),
    ]
    return pd.DataFrame(rows, columns=["Metric", "New", "Published"])


def feature_coverage(data: EventData) -> pd.DataFrame:
    rows = []
    particle_new = set(data.new_schemas["particles"].names)
    particle_old = set(data.public_schemas["particles"].names)
    hit_new = set(data.new_schemas["hits"].names)
    hit_old = set(data.public_schemas["tracker_hits"].names)
    mapped = {"x": "global_x", "y": "global_y", "z": "global_z",
              "volume_id": "geometry_id", "layer_id": "geometry_id",
              "surface_id": "geometry_id"}
    for name in sorted(particle_new | particle_old):
        mode = "ID/integrity" if name in {"event_id", "particle_id", "parent_id"} else (
            "overlay" if name in particle_new & particle_old else "new only" if name in particle_new else "published only")
        rows.append(("particles", name, name if name in particle_new else "", name if name in particle_old else "", mode))
    for name in sorted(hit_old):
        new_name = mapped.get(name, name)
        mode = "ID/integrity" if name in {"event_id", "particle_id"} else (
            "overlay" if new_name in hit_new else "published only")
        rows.append(("hits", name, new_name if new_name in hit_new else "", name, mode))
    covered = {mapped.get(name, name) for name in hit_old}
    for name in sorted(hit_new - covered):
        mode = "ID/integrity" if name in {"campaign", "dataset", "version", "run", "local_event",
                                           "measurement_id", "geometry_id", "converted_particle_id", "simhit_ids",
                                           "unresolved_simhit_ids", "particle_ids"} else "new only"
        rows.append(("hits", name, name, "", mode))
    for name in data.new_schemas["cells"].names:
        rows.append(("cells", name, name, "", "ID/integrity" if name in {
            "campaign", "dataset", "version", "run", "local_event", "measurement_id", "geometry_id"} else "new only"))
    return pd.DataFrame(rows, columns=["Table", "Feature", "New column", "Published column", "Treatment"])


def _overlay(ax, new, old, title: str, categorical: bool = False,
             log_y: bool = False) -> None:
    new = _numeric(new)
    old = _numeric(old)
    if categorical:
        counts = pd.Series(np.concatenate([new, old])).value_counts().head(15)
        categories = np.sort(counts.index.to_numpy())
        x = np.arange(len(categories))
        for values, color, label in ((new, NEW_COLOR, "New"), (old, PUBLIC_COLOR, "Published")):
            series = pd.Series(values).value_counts(normalize=True)
            ax.plot(x, [series.get(item, 0) for item in categories], "o-", color=color, label=label)
        ax.set_xticks(x, [f"{item:g}" for item in categories], rotation=45, ha="right")
        ax.set_ylabel("Fraction")
    else:
        joined = np.concatenate([new, old])
        if len(joined):
            lo, hi = np.quantile(joined, [0.005, 0.995])
            if lo == hi:
                lo, hi = float(np.min(joined)), float(np.max(joined))
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            bins = np.linspace(lo, hi, 51)
            for values, color, label in ((new, NEW_COLOR, "New"), (old, PUBLIC_COLOR, "Published")):
                within = values[(values >= lo) & (values <= hi)]
                if len(within):
                    ax.hist(within, bins=bins, density=True, histtype="step", linewidth=1.7,
                            color=color, label=label)
            ax.set_ylabel("Density")
            ax.text(0.98, 0.96, "0.5-99.5% range", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ax.get_ylabel()} (log scale)")
    ax.legend(fontsize=8)


def _grid(items, plot, heading: str, columns: int = 3) -> None:
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.4 * rows), squeeze=False)
    for ax, item in zip(axes.flat, items):
        plot(ax, item)
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    fig.suptitle(heading, fontsize=15)
    fig.tight_layout()
    plt.show()


def plot_particles(data: EventData) -> None:
    continuous = [name for name in PARTICLE_FIELDS if name not in {"pdg_id", "charge", "vertex_primary", "parent_id", "primary"}]
    def values(row, name):
        if name == "pt":
            px = pd.to_numeric(pd.Series(row["px"]), errors="coerce").to_numpy(dtype=float)
            py = pd.to_numeric(pd.Series(row["py"]), errors="coerce").to_numpy(dtype=float)
            return np.hypot(px, py)
        return row[name]
    _grid(continuous + ["pt"], lambda ax, name: _overlay(
        ax, values(data.new_particles, name), values(data.public_particles, name), name,
        log_y=name in {"mass", "energy", "time", "px", "py", "pz", "pt"}),
        "Particles: shared continuous features")
    _grid(["pdg_id", "charge", "vertex_primary", "primary"], lambda ax, name: _overlay(
        ax, data.new_particles[name], data.public_particles[name], name, categorical=True),
        "Particles: shared categorical features", columns=2)
    _overlay_parent(data)
    extra = [name for name in ("num_tracker_hits", "num_calo_hits") if name in data.new_particles]
    if extra:
        _grid(extra, lambda ax, name: _single(ax, data.new_particles[name], name, NEW_COLOR,
                                             log_y=name == "num_tracker_hits"),
              "New particles only", columns=2)


def _overlay_parent(data: EventData) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for row, color, label in ((data.new_particles, NEW_COLOR, "New"),
                              (data.public_particles, PUBLIC_COLOR, "Published")):
        ids = pd.to_numeric(pd.Series(row["parent_id"]), errors="coerce")
        ax.plot([0, 1, 2], [(ids < 0).mean(), (ids >= 0).mean(), ids.isna().mean()],
                "o-", color=color, label=label)
    ax.set_xticks([0, 1, 2], ["No parent", "Has parent", "Missing"])
    ax.set_ylabel("Fraction")
    ax.set_title("Particle parent association")
    ax.legend()
    plt.show()


def _single(ax, values, title: str, color: str, categorical: bool = False,
            log_y: bool = False) -> None:
    values = _numeric(values)
    if categorical:
        counts = pd.Series(values).value_counts(normalize=True).head(20)
        ax.bar(np.arange(len(counts)), counts.values, color=color)
        ax.set_xticks(np.arange(len(counts)), [f"{item:g}" for item in counts.index],
                      rotation=45, ha="right")
    elif len(values):
        lo, hi = np.quantile(values, [0.005, 0.995])
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        ax.hist(values[(values >= lo) & (values <= hi)], bins=np.linspace(lo, hi, 51),
                color=color, alpha=0.75)
    ax.set_title(title)
    ax.set_ylabel("Count" if not categorical else "Fraction")
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ax.get_ylabel()} (log scale)")


def plot_hits(data: EventData, scatter_limit: int = 10000, seed: int = 7) -> None:
    continuous = ["x", "y", "z"]
    categorical = ["volume_id", "layer_id", "surface_id"]
    _grid(continuous, lambda ax, name: _overlay(ax, _new_hit_feature(data, name),
          data.public_hits[name], f"Hit {name}"), "Hits: reconstructed position")
    _grid(categorical, lambda ax, name: _overlay(ax, _new_hit_feature(data, name),
          data.public_hits[name], f"Hit {name}", categorical=True),
          "Hits: geometry categories")
    _overlay_hit_labels(data)
    _plot_spatial(data, scatter_limit, seed)
    _grid(["true_x", "true_y", "true_z", "time", "detector"],
          lambda ax, name: _single(ax, data.public_hits[name],
                                   f"Published only: {name}", PUBLIC_COLOR,
                                   categorical=name == "detector", log_y=name == "time"),
          "Published hit-only fields")
    _grid(["local0", "local1", "simhit_count", "particle_count", "unresolved_count",
           "truth_complete", "converted_label_present"],
          lambda ax, name: _single(ax, _new_hit_only(data, name),
                                   f"New only: {name}", NEW_COLOR,
                                   categorical=name.endswith("count") or name.endswith("complete")
                                   or name.endswith("present"),
                                   log_y=name in {"simhit_count", "particle_count"}),
          "New hit-only fields")


def _new_hit_only(data: EventData, name: str):
    if name in {"local0", "local1"}:
        return _arrow_values(data.new_hits, name)
    if name == "truth_complete":
        return _arrow_values(data.new_hits, "particle_ids_complete")
    if name == "converted_label_present":
        return [value is not None for value in data.new_hits["converted_particle_id"].to_pylist()]
    source = {"simhit_count": "simhit_ids", "particle_count": "particle_ids",
              "unresolved_count": "unresolved_simhit_ids"}[name]
    return [len(values or []) for values in data.new_hits[source].to_pylist()]


def _overlay_hit_labels(data: EventData) -> None:
    new_count = _new_hit_only(data, "particle_count")
    old_count = [int(value is not None) for value in data.public_hits["particle_id"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _overlay(ax, new_count, old_count, "Known particle IDs per hit", categorical=True)
    fig.tight_layout()
    plt.show()


def _plot_spatial(data: EventData, limit: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    def coordinates(source):
        values = {name: pd.to_numeric(pd.Series(source[name]), errors="coerce").to_numpy(dtype=float)
                  for name in ("x", "y", "z")}
        valid = np.logical_and.reduce([np.isfinite(value) for value in values.values()])
        return {name: value[valid] for name, value in values.items()}
    new = coordinates({name: _new_hit_feature(data, name) for name in ("x", "y", "z")})
    old = coordinates({name: data.public_hits[name] for name in ("x", "y", "z")})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for values, color, label in ((new, NEW_COLOR, "New"), (old, PUBLIC_COLOR, "Published")):
        n = min(map(len, values.values()))
        idx = rng.choice(n, size=min(n, limit), replace=False)
        axes[0].scatter(values["x"][idx], values["y"][idx], s=2, alpha=0.2,
                        color=color, label=label)
        axes[1].scatter(values["z"][idx], np.hypot(values["x"][idx], values["y"][idx]),
                        s=2, alpha=0.2, color=color, label=label)
    axes[0].set(xlabel="x", ylabel="y", title="Hit x-y")
    axes[1].set(xlabel="z", ylabel="r", title="Hit r-z")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"Spatial overlay (up to {limit:,} hits per source)")
    fig.tight_layout()
    plt.show()


def plot_cells(data: EventData) -> None:
    _grid(["channel0", "channel1", "value"],
          lambda ax, name: _single(ax, _arrow_values(data.cells, name),
                                   f"New cells only: {name}", NEW_COLOR,
                                   log_y=name == "channel1"),
          "New cell-only features")
    ids = _arrow_values(data.cells, "measurement_id")
    _, occupancy = np.unique(ids, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _single(ax, occupancy, "Cells per measurement", NEW_COLOR,
            categorical=True, log_y=True)
    plt.show()


def missingness(data: EventData) -> pd.DataFrame:
    rows = []
    def add(table, name, label, values):
        series = pd.Series(values)
        rows.append((table, name, label, len(series), int(series.isna().sum()),
                     float(series.isna().mean())))
    for table, new, old in (("particles", data.new_particles, data.public_particles),
                            ("hits", {name: _new_hit_feature(data, name) for name in HIT_FIELDS},
                             {name: data.public_hits[name] for name in HIT_FIELDS})):
        names = PARTICLE_FIELDS if table == "particles" else HIT_FIELDS
        for name in names:
            for label, values in (("New", new[name]), ("Published", old[name])):
                add(table, name, label, values)
    for name in ("num_tracker_hits", "num_calo_hits"):
        if name in data.new_particles:
            add("particles", name, "New", data.new_particles[name])
    for name in ("true_x", "true_y", "true_z", "time", "detector"):
        add("hits", name, "Published", data.public_hits[name])
    for name in ("local0", "local1", "converted_particle_id", "particle_ids_complete"):
        add("hits", name, "New", data.new_hits[name].to_pylist())
    for name in ("channel0", "channel1", "value"):
        add("cells", name, "New", data.cells[name].to_pylist())
    return pd.DataFrame(rows, columns=["Table", "Feature", "Source", "Count", "Missing", "Missing fraction"])


def plot_missingness(data: EventData) -> None:
    frame = missingness(data)
    for table, part in frame.groupby("Table"):
        pivot = part.pivot(index="Feature", columns="Source", values="Missing fraction")
        fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.6), 3.5))
        x = np.arange(len(pivot))
        if "New" in pivot:
            ax.bar(x - 0.18, pivot["New"], width=0.36, color=NEW_COLOR, label="New")
        if "Published" in pivot:
            ax.bar(x + 0.18, pivot["Published"], width=0.36, color=PUBLIC_COLOR, label="Published")
        ax.set_xticks(x, pivot.index, rotation=45, ha="right")
        ax.set(ylabel="Missing fraction", ylim=(0, 1), title=f"{table.title()} missingness")
        ax.legend()
        fig.tight_layout()
        plt.show()
