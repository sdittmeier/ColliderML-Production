"""Load and visualize one paired event against one independent public event."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


NEW_COLOR = "#1672b8"
PUBLIC_COLOR = "#df7926"
PILEUP_COLORS = {"pu0": NEW_COLOR, "pu200": "#26966b"}
METHOD_COLORS = {"native": "#222222", "empirical": "#CC79A7", "linear_ATLAS": "#D55E00",
                 "linear_P99.9": "#0072B2", "linear_P99.5": "#009E73", "linear_P99": "#E69F00"}
METHOD_LABELS = {"native": "Native activation", "empirical": "Empirical ATLAS match",
                 "linear_ATLAS": "Linear ATLAS-tail", "linear_P99.9": "Linear P99.9",
                 "linear_P99.5": "Linear P99.5", "linear_P99": "Linear P99"}
METHOD_MARKERS = dict(zip(METHOD_COLORS, ("o", "s", "^", "D", "v", "P")))
MEASUREMENT_KEY = ["campaign", "dataset", "version", "run", "local_event", "measurement_id"]
PARTICLE_FIELDS = ("pdg_id", "mass", "energy", "charge", "vx", "vy", "vz",
                   "time", "px", "py", "pz", "perigee_d0", "perigee_z0",
                   "vertex_primary", "parent_id", "primary")
HIT_FIELDS = ("x", "y", "z", "volume_id", "layer_id", "surface_id", "particle_id")
DETECTOR_VOLUMES = {
    "Pixels": (16, 17, 18),
    "Short strips": (23, 24, 25),
    "Long strips": (28, 29, 30),
}


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


def cell_value_summary(data: EventData) -> pd.DataFrame:
    """Summarize the full event, including the tail hidden by histogram limits."""
    values = _arrow_values(data.cells, "value").astype(float)
    volumes = _geom(data.cells, 56, 0xff)
    groups = {"All": np.ones(len(values), dtype=bool)}
    groups.update({name: np.isin(volumes, ids) for name, ids in DETECTOR_VOLUMES.items()})
    rows = []
    for name, mask in groups.items():
        selected = values[mask]
        if len(selected):
            rows.append({"Detector type": name, "Cells": len(selected),
                         "Minimum": selected.min(), "Median": np.median(selected),
                         "99.5th percentile": np.quantile(selected, 0.995),
                         "Maximum": selected.max(), "Cells above 0.5": int(np.sum(selected > 0.5))})
    return pd.DataFrame(rows)


def select_cluster_examples(data: EventData, target_sizes=(1, 2, 4, 8, 16, 32)) -> pd.DataFrame:
    """Pick distinct nearest sizes per detector type; ties use the smaller size."""
    targets = tuple(target_sizes)
    if not targets or any(isinstance(size, bool) or not isinstance(size, (int, np.integer))
                          or size < 1 for size in targets) or len(set(targets)) != len(targets):
        raise ValueError("target_sizes must contain distinct positive integers")
    ids = _arrow_values(data.cells, "measurement_id")
    measurement_ids, first, counts = np.unique(ids, return_index=True, return_counts=True)
    volumes = _geom(data.cells, 56, 0xff)[first]
    rows = []
    columns = ["Detector type", "Target cells", "Measurement ID", "Volume", "Cells",
               "Known particles", "Truth complete"]
    for name, detector_volumes in DETECTOR_VOLUMES.items():
        family = np.isin(volumes, detector_volumes)
        remaining = set(counts[family].tolist())
        for target in targets:
            if not remaining:
                break
            size = min(remaining, key=lambda value: (abs(value - target), value))
            remaining.remove(size)
            candidates = np.flatnonzero(family & (counts == size))
            chosen = candidates[np.argmin(measurement_ids[candidates])]
            mid = int(measurement_ids[chosen])
            hit = data.new_hits.filter(pc.equal(data.new_hits["measurement_id"], mid))
            if hit.num_rows != 1:
                raise ValueError(f"Measurement {mid}: expected one parent hit")
            row = hit.select(["geometry_id", "particle_ids", "particle_ids_complete"]).to_pylist()[0]
            if int(row["geometry_id"]) >> 56 != int(volumes[chosen]):
                raise ValueError(f"Measurement {mid}: cell and hit detector volumes differ")
            rows.append({"Detector type": name, "Target cells": target,
                         "Measurement ID": mid, "Volume": int(volumes[chosen]), "Cells": int(size),
                         "Known particles": len(row["particle_ids"] or []),
                         "Truth complete": bool(row["particle_ids_complete"])})
    return pd.DataFrame(rows, columns=columns)


def _cluster_cell_image(channel0, channel1, values):
    """Build a channel-address image with one masked cell of padding per side."""
    x = np.asarray(channel0)
    y = np.asarray(channel1)
    values = np.asarray(values, dtype=float)
    if not len(values) or x.shape != y.shape or x.shape != values.shape or x.ndim != 1:
        raise ValueError("A cluster needs equally sized nonempty channel/value arrays")
    if (not np.all(np.isfinite(values)) or np.any(values < 0)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(x != np.floor(x)) or np.any(y != np.floor(y))):
        raise ValueError("Cell addresses must be integers and values finite and nonnegative")
    x = x.astype(int)
    y = y.astype(int)
    xmin, xmax = int(x.min()) - 1, int(x.max()) + 1
    ymin, ymax = int(y.min()) - 1, int(y.max()) + 1
    image = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=float)
    occupied = np.zeros_like(image, dtype=bool)
    np.add.at(image, (y - ymin, x - xmin), values)
    occupied[y - ymin, x - xmin] = True
    extent = (xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5)
    return np.ma.array(image, mask=~occupied), extent


def plot_cluster_galleries(data: EventData, target_sizes=(1, 2, 4, 8, 16, 32),
                           color_scale: str = "linear") -> pd.DataFrame:
    """Show one gallery per detector type with a shared raw-value color scale."""
    if color_scale not in {"linear", "log"}:
        raise ValueError("color_scale must be 'linear' or 'log'")
    targets = tuple(target_sizes)
    examples = select_cluster_examples(data, targets)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    summaries = []
    for name in DETECTOR_VOLUMES:
        selected = examples[examples["Detector type"] == name]
        if len(selected) < len(targets):
            print(f"{name}: only {len(selected)} distinct cluster sizes available; "
                  f"requested {len(targets)}.")
        if selected.empty:
            continue
        images = []
        for example in selected.to_dict("records"):
            mid = example["Measurement ID"]
            cells = data.cells.filter(pc.equal(data.cells["measurement_id"], mid))
            values = _arrow_values(cells, "value").astype(float)
            image, extent = _cluster_cell_image(_arrow_values(cells, "channel0"),
                                                _arrow_values(cells, "channel1"), values)
            if not np.isclose(float(image.sum()), float(values.sum())):
                raise AssertionError(f"Measurement {mid}: image lost cell activation")
            if color_scale == "log" and np.any(image.compressed() <= 0):
                raise ValueError("Logarithmic color scaling requires positive occupied cells")
            images.append((example, image, extent))
            summaries.append({**example, "Value sum": values.sum(),
                              "Minimum value": values.min(), "Maximum value": values.max()})
        maximum = max(float(image.max()) for _, image, _ in images)
        if color_scale == "log":
            minimum = min(float(image.min()) for _, image, _ in images)
            norm = LogNorm(vmin=minimum, vmax=max(maximum, minimum * 1.01))
        else:
            norm = Normalize(vmin=0, vmax=maximum if maximum > 0 else 1)
        columns = min(3, len(images))
        rows = (len(images) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.4 * rows),
                                 squeeze=False, constrained_layout=True)
        active_axes = []
        for ax, (example, image, extent) in zip(axes.flat, images):
            artist = ax.imshow(image, origin="lower", interpolation="nearest", aspect="equal",
                               extent=extent, cmap=cmap, norm=norm)
            ax.set(xlabel="channel0", ylabel="channel1")
            partial = " (partial)" if not example["Truth complete"] else ""
            cell_label = "cell" if example["Cells"] == 1 else "cells"
            particle_label = "particle" if example["Known particles"] == 1 else "particles"
            ax.set_title(f"Measurement {example['Measurement ID']}\n"
                         f"Volume {example['Volume']} | {example['Cells']} {cell_label}\n"
                         f"{example['Known particles']} known {particle_label}{partial}",
                         fontsize=10)
            if image.shape[0] > image.shape[1] * 2:
                first_channel = int(extent[0] + 1.5)
                last_channel = int(extent[1] - 1.5)
                ax.set_xticks(sorted({first_channel, last_channel}))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.set_xticks(np.arange(extent[0], extent[1] + 0.5), minor=True)
            ax.set_yticks(np.arange(extent[2], extent[3] + 0.5), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.5)
            ax.tick_params(which="minor", length=0)
            active_axes.append(ax)
        for ax in list(axes.flat)[len(images):]:
            ax.axis("off")
        fig.colorbar(artist, ax=active_axes, shrink=0.8,
                     label="Cell activation (path-length proxy, mm)")
        fig.suptitle(f"{name}: {data.sample}, event {data.new_event} | {color_scale} color scale")
        plt.show()
    return pd.DataFrame(summaries)


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


@dataclass
class ClusterAnalysis:
    sample: str
    event: int
    threshold: float
    cells: pd.DataFrame
    clusters: pd.DataFrame


def build_cluster_analysis(hits: pa.Table, cells: pa.Table, sample: str,
                           event: int, threshold: float = 0.5) -> ClusterAnalysis:
    """Attach parent-cluster truth to cells without inferring cell-level truth."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("The diagnostic tail threshold must be finite and nonnegative")
    for table in (hits, cells):
        if set(table["local_event"].to_pylist()) != {event}:
            raise ValueError("Analysis inputs must contain only the selected local event")
    metadata = hits.select(MEASUREMENT_KEY + ["geometry_id", "particle_ids_complete"]).to_pandas()
    metadata = metadata.rename(columns={"geometry_id": "hit_geometry_id",
                                        "particle_ids_complete": "truth_complete"})
    metadata["known_particles"] = pc.list_value_length(hits["particle_ids"]).to_numpy()
    metadata["simhit_count"] = pc.list_value_length(hits["simhit_ids"]).to_numpy()
    if metadata["truth_complete"].isna().any() or metadata["known_particles"].isna().any():
        raise ValueError("Missing contributor metadata")
    joined = cells.to_pandas().merge(metadata, on=MEASUREMENT_KEY, how="left",
                                     validate="many_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("Cells reference a missing parent cluster")
    joined = joined.drop(columns="_merge")
    if not (joined["geometry_id"] == joined["hit_geometry_id"]).all():
        raise ValueError("Cell geometry differs from its parent cluster")
    if joined.duplicated(MEASUREMENT_KEY + ["channel0", "channel1"]).any():
        raise ValueError("Duplicate channel address within a cluster")
    if not np.isfinite(joined["value"]).all() or (joined["value"] < 0).any():
        raise ValueError("Cell activations must be finite and nonnegative")
    joined["volume_id"] = joined["geometry_id"].to_numpy(dtype=np.uint64) >> 56
    volume_names = {volume: name for name, volumes in DETECTOR_VOLUMES.items() for volume in volumes}
    joined["detector_type"] = joined["volume_id"].map(volume_names)
    if joined["detector_type"].isna().any():
        raise ValueError("Unrecognized detector volume in cell analysis")
    joined["contributor_group"] = np.select(
        [joined["truth_complete"] & (joined["known_particles"] == 1),
         joined["truth_complete"] & (joined["known_particles"] == 2),
         joined["truth_complete"] & (joined["known_particles"] >= 3)],
        ["1 particle", "2 particles", "3+ particles"], default="Unknown/incomplete")
    joined["above_tail"] = joined["value"] > threshold
    groups = joined.groupby(MEASUREMENT_KEY, sort=False, observed=True)
    clusters = groups.agg(n_cells=("value", "size"), activation_sum=("value", "sum"),
                          activation_max=("value", "max"),
                          tail_cells=("above_tail", "sum"),
                          min_channel0=("channel0", "min"), max_channel0=("channel0", "max"),
                          min_channel1=("channel1", "min"), max_channel1=("channel1", "max"),
                          detector_type=("detector_type", "first"),
                          volume_id=("volume_id", "first"),
                          contributor_group=("contributor_group", "first")).reset_index()
    clusters = clusters.merge(metadata, on=MEASUREMENT_KEY, validate="one_to_one")
    if len(clusters) != len(metadata):
        raise ValueError("Some parent clusters have no cells")
    clusters["width0"] = clusters["max_channel0"] - clusters["min_channel0"] + 1
    clusters["width1"] = clusters["max_channel1"] - clusters["min_channel1"] + 1
    clusters["occupancy"] = clusters["n_cells"] / (clusters["width0"] * clusters["width1"])
    clusters["aspect_ratio"] = (clusters[["width0", "width1"]].max(axis=1)
                                  / clusters[["width0", "width1"]].min(axis=1))
    return ClusterAnalysis(sample, event, threshold, joined, clusters)


def load_pileup_comparison(new_root: Path, event: int = 0,
                           threshold: float = 0.5) -> dict[str, ClusterAnalysis]:
    """Read only one selected event at each pileup setting, without public files."""
    result = {}
    for sample in PILEUP_COLORS:
        folder = Path(new_root) / sample
        hits = _flat_event(folder / "hits.parquet", event)
        cells = _flat_event(folder / "cells.parquet", event)
        if not hits.num_rows or not cells.num_rows:
            raise ValueError(f"{sample}: no hits/cells for selected event {event}")
        result[sample] = build_cluster_analysis(hits, cells, sample, event, threshold)
    return result


def contributor_statistics(analyses: dict[str, ClusterAnalysis]) -> pd.DataFrame:
    """Keep exact known counts and incomplete-truth cohorts distinct."""
    tables = []
    keys = ["detector_type", "truth_complete", "known_particles"]
    for sample, analysis in analyses.items():
        cells = analysis.cells.groupby(keys, observed=True).agg(
            cells=("value", "size"), activation_median=("value", "median"),
            activation_q95=("value", lambda x: x.quantile(0.95)),
            activation_q995=("value", lambda x: x.quantile(0.995)),
            activation_max=("value", "max"),
            tail_cells=("above_tail", "sum"), cell_tail_fraction=("above_tail", "mean"))
        clusters = analysis.clusters.assign(
            above_tail=analysis.clusters["activation_max"] > analysis.threshold).groupby(keys, observed=True).agg(
                clusters=("n_cells", "size"), tail_clusters=("above_tail", "sum"),
                cluster_tail_fraction=("above_tail", "mean"))
        table = cells.join(clusters, validate="one_to_one").reset_index()
        table.insert(0, "sample", sample)
        table.insert(1, "local_event", analysis.event)
        table["diagnostic_threshold"] = analysis.threshold
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def pileup_event_summary(analyses: dict[str, ClusterAnalysis]) -> pd.DataFrame:
    rows = []
    for sample, analysis in analyses.items():
        for detector, cells in analysis.cells.groupby("detector_type", observed=True):
            clusters = analysis.clusters[analysis.clusters["detector_type"] == detector]
            tail = cells[cells["above_tail"]]
            complete_tail = tail[tail["truth_complete"]]
            maximum = cells.loc[cells["value"].idxmax()]
            rows.append({"sample": sample, "local_event": analysis.event, "detector_type": detector,
                         "clusters": len(clusters), "cells": len(cells),
                         "incomplete_clusters": int((~clusters["truth_complete"]).sum()),
                         "maximum_activation": cells["value"].max(),
                         "max_cell_parent_particles": int(maximum["known_particles"]),
                         "max_cell_parent_simhits": int(maximum["simhit_count"]),
                         "max_cell_truth_complete": bool(maximum["truth_complete"]),
                         "cell_tail_fraction": cells["above_tail"].mean(),
                         "cluster_tail_fraction": (clusters["activation_max"] > analysis.threshold).mean(),
                         "complete_truth_tail_cells": len(complete_tail),
                         "multi_particle_parent_fraction_of_complete_tail":
                             (complete_tail["known_particles"] > 1).mean() if len(complete_tail) else np.nan})
    return pd.DataFrame(rows)


def largest_cells(analyses: dict[str, ClusterAnalysis], per_sample: int = 10) -> pd.DataFrame:
    rows = []
    for sample, analysis in analyses.items():
        top = analysis.cells.nlargest(per_sample, "value").merge(
            analysis.clusters[MEASUREMENT_KEY + ["n_cells", "activation_sum"]],
            on=MEASUREMENT_KEY, validate="many_to_one")
        top.insert(0, "sample", sample)
        rows.append(top[["sample", *MEASUREMENT_KEY, "detector_type", "volume_id", "channel0", "channel1",
                         "value", "n_cells", "activation_sum", "known_particles", "simhit_count", "truth_complete"]])
    return pd.concat(rows, ignore_index=True)


def _comparison_hist(ax, series: dict[str, np.ndarray], colors: dict[str, str], title: str,
                      xlabel: str, log_y: bool = True) -> None:
    clean = {name: np.asarray(values, dtype=float) for name, values in series.items()}
    clean = {name: values[np.isfinite(values)] for name, values in clean.items()}
    present = [values for values in clean.values() if len(values)]
    ax.set(title=title, xlabel=xlabel, ylabel="Density")
    if not present:
        ax.text(0.5, 0.5, "No observations", transform=ax.transAxes, ha="center")
        return
    lo = min(float(values.min()) for values in present)
    hi = max(float(values.max()) for values in present)
    log_x = lo > 0 and hi / lo > 30
    if lo == hi:
        bins = np.linspace(lo - 0.5, hi + 0.5, 3)
    else:
        bins = np.geomspace(lo, hi, 61) if log_x else np.linspace(lo, hi, 51)
    for name, values in clean.items():
        if len(values):
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.6,
                    color=colors[name], label=f"{name} (n={len(values):,})")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Density (log scale)")
    ax.legend(fontsize=8)


def _survival(ax, series: dict[str, np.ndarray], colors: dict[str, str]) -> None:
    for name, values in series.items():
        values = np.sort(np.asarray(values))
        if len(values):
            unique, first = np.unique(values, return_index=True)
            ax.step(unique, (len(values) - first) / len(values), where="pre",
                    color=colors[name], label=name)
    ax.set(xlabel="Cell activation (mm proxy)", ylabel="Fraction >= activation",
           title="Full-range survival", yscale="log")
    if any(len(values) for values in series.values()):
        ax.legend(fontsize=8)


def plot_pileup_comparison(analyses: dict[str, ClusterAnalysis]) -> None:
    events = {value.event for value in analyses.values()}
    if len(events) != 1 or len({value.threshold for value in analyses.values()}) != 1:
        raise ValueError("Pileup plots require the same selected event and threshold")
    event = next(iter(events))
    for detector in DETECTOR_VOLUMES:
        cell_frames = {name: value.cells[value.cells["detector_type"] == detector]
                       for name, value in analyses.items()}
        cluster_frames = {name: value.clusters[value.clusters["detector_type"] == detector]
                          for name, value in analyses.items()}
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        cell_values = {name: frame["value"].to_numpy() for name, frame in cell_frames.items()}
        _comparison_hist(axes[0, 0], cell_values, PILEUP_COLORS, "All cell activations", "Activation (mm proxy)")
        _survival(axes[0, 1], cell_values, PILEUP_COLORS)
        for ax, field, title, xlabel in (
            (axes[0, 2], "activation_max", "Cluster maxima", "Maximum activation (mm proxy)"),
            (axes[1, 0], "n_cells", "Cluster sizes", "Cells per cluster"),
            (axes[1, 1], "activation_sum", "Cluster activation sums", "Sum (mm proxy)")):
            _comparison_hist(ax, {name: frame[field].to_numpy() for name, frame in cluster_frames.items()},
                             PILEUP_COLORS, title, xlabel)
        ax = axes[1, 2]
        for name, analysis in analyses.items():
            cells = cell_frames[name]
            cells = cells[cells["truth_complete"]]
            clusters = cluster_frames[name]
            clusters = clusters[clusters["truth_complete"]]
            cell_rate = cells.groupby("known_particles")["above_tail"].mean()
            cluster_rate = clusters.assign(above=clusters["activation_max"] > analysis.threshold).groupby(
                "known_particles")["above"].mean()
            ax.plot(cell_rate.index, cell_rate.values, "o-", color=PILEUP_COLORS[name], label=f"{name}: cells")
            ax.plot(cluster_rate.index, cluster_rate.values, "s--", color=PILEUP_COLORS[name], label=f"{name}: clusters")
        threshold = next(iter(analyses.values())).threshold
        ax.set(title=f"Fraction above {threshold:g}: complete truth only",
               xlabel="Parent-cluster particle count", ylabel="Fraction", ylim=(-0.02, 1.02))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        fig.suptitle(f"{detector}: pu0 vs pu200, local event {event}")
        plt.show()

        groups = ("1 particle", "2 particles", "3+ particles", "Unknown/incomplete")
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
        for ax, group in zip(axes, groups):
            values = {name: frame.loc[frame["contributor_group"] == group, "value"].to_numpy()
                      for name, frame in cell_frames.items()}
            _comparison_hist(ax, values, PILEUP_COLORS, group, "Activation (mm proxy)")
        fig.suptitle(f"{detector}: cell activation by parent-cluster truth, local event {event}")
        plt.show()


def summarize_atlas_pixel_images(images, regions, pixel_regions, event_id="") -> dict[str, np.ndarray]:
    """Describe only nonzero pixels in the supplied ATLAS windows, without calibration."""
    images = np.asarray(images)
    regions = np.asarray(regions)
    if images.ndim != 3 or images.shape[0] != len(regions) or min(images.shape[1:]) < 1:
        raise ValueError("ATLAS images must have shape (hits, height, width), aligned with regions")
    if not np.all(np.isfinite(images)) or np.any(images < 0):
        raise ValueError("ATLAS image codes must be finite and nonnegative")
    pixel_mask = np.isin(regions, list(pixel_regions))
    if not pixel_mask.any():
        raise ValueError("No pixel-region hits found in ATLAS input")
    pixels = images[pixel_mask]
    active = pixels > 0
    count = active.sum(axis=(1, 2))
    def span(occupied):
        size = occupied.shape[1]
        first = occupied.argmax(axis=1)
        last = size - 1 - occupied[:, ::-1].argmax(axis=1)
        return np.where(count > 0, last - first + 1, 0)
    width0 = span(active.any(axis=1))
    width1 = span(active.any(axis=2))
    occupied_box = width0 * width1
    occupancy = np.divide(count, occupied_box, out=np.full(len(count), np.nan), where=occupied_box > 0)
    minor = np.minimum(width0, width1)
    aspect = np.divide(np.maximum(width0, width1), minor,
                       out=np.full(len(count), np.nan), where=minor > 0)
    edge = active[:, 0, :].any(axis=1) | active[:, -1, :].any(axis=1)
    edge |= active[:, :, 0].any(axis=1) | active[:, :, -1].any(axis=1)
    return {"event_id": np.asarray(str(event_id)), "image_shape": np.asarray(images.shape[1:]),
            "pixel_regions": np.asarray(sorted(pixel_regions)), "n_cells": count,
            "width0": width0, "width1": width1, "occupancy": occupancy, "aspect_ratio": aspect,
            "touches_edge": edge, "values": pixels[active],
            "contains_code15": (pixels == 15).any(axis=(1, 2)),
            "contains_overflow15": (pixels >= 15).any(axis=(1, 2)),
            "excluded_hits": np.asarray(int((~pixel_mask).sum())),
            "excluded_nonempty_images": np.asarray(int(np.any(images[~pixel_mask] > 0, axis=(1, 2)).sum()))}


def _read_atlas_pixel_file(path: str) -> dict[str, np.ndarray]:
    import torch
    graph = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    configs = graph.config if isinstance(graph.config, list) else [graph.config]
    pixel_regions = {int(region) for config in configs
                     for region, settings in config.get("region_labels", {}).items()
                     if settings.get("hardware") == "PIXEL"}
    if not pixel_regions:
        raise ValueError("ATLAS graph config has no PIXEL region labels")
    return summarize_atlas_pixel_images(graph.cluster_images.cpu().numpy(), graph.hit_region.cpu().numpy(),
                                         pixel_regions, graph.event_id)


def load_atlas_pixels(path: Path, python_executable: Path | str | None = None) -> dict[str, np.ndarray]:
    """Use a PyTorch-capable interpreter and return numeric arrays in memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run([str(python_executable or sys.executable), str(Path(__file__).resolve()),
                             "--atlas-pixels", str(path.resolve())], capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"ATLAS reader failed in {python_executable or sys.executable}:\n"
                           f"{result.stderr.decode(errors='replace')}")
    with np.load(io.BytesIO(result.stdout), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def atlas_pixel_report(atlas: dict[str, np.ndarray]) -> pd.DataFrame:
    values = atlas["values"]
    counts = atlas["n_cells"]
    nonempty = counts > 0
    return pd.DataFrame([{
        "event_id": str(atlas["event_id"]), "image_shape": tuple(atlas["image_shape"].tolist()),
        "pixel_regions": atlas["pixel_regions"].tolist(), "pixel_images": len(counts),
        "empty_pixel_images": int((~nonempty).sum()), "nonzero_cell_entries": len(values),
        "excluded_nonpixel_hits": int(atlas["excluded_hits"]),
        "excluded_nonempty_images": int(atlas["excluded_nonempty_images"]),
        "window_edge_fraction": atlas["touches_edge"].mean(),
        "window_edge_fraction_nonempty": atlas["touches_edge"][nonempty].mean() if nonempty.any() else np.nan,
        "minimum_nonzero_code": values.min() if len(values) else np.nan,
        "maximum_code": values.max() if len(values) else np.nan,
        "fraction_at_code_15": (values == 15).mean() if len(values) else np.nan,
    }])


def plot_atlas_pixel_comparison(analyses: dict[str, ClusterAnalysis], atlas: dict[str, np.ndarray]) -> None:
    frames = {name: value.clusters[value.clusters["detector_type"] == "Pixels"]
              for name, value in analyses.items()}
    atlas_name = "ATLAS pu200"
    colors = {**PILEUP_COLORS, atlas_name: PUBLIC_COLOR}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fields = (("n_cells", "Observed nonzero cells", "Cells per cluster"),
              ("width0", "Native channel0 span", "Channels (ATLAS image columns = eta)"),
              ("width1", "Native channel1 span", "Channels (ATLAS image rows = phi)"),
              ("occupancy", "Bounding-box occupancy", "Active cells / box area"),
              ("aspect_ratio", "Major/minor span", "Aspect ratio"))
    for ax, (field, title, xlabel) in zip(axes.flat, fields):
        series = {name: frame[field].to_numpy() for name, frame in frames.items()}
        series[atlas_name] = atlas[field]
        _comparison_hist(ax, series, colors, title, xlabel)
    ax = axes.flat[-1]
    fraction = atlas["touches_edge"].mean()
    ax.bar([0, 1], [1 - fraction, fraction], color=PUBLIC_COLOR)
    ax.set_xticks([0, 1], ["No edge contact", "Touches window edge"])
    ax.set(ylabel="Fraction of pixel images", ylim=(0, 1), title="ATLAS window-boundary diagnostic")
    selected_event = next(iter(analyses.values())).event
    fig.suptitle(f"Pixel shape: ColliderML local event {selected_event}, ATLAS event {str(atlas['event_id'])}\n"
                 "Native channel axes; ATLAS windowed images vs full ColliderML clusters")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    values = {name: analysis.cells.loc[analysis.cells["detector_type"] == "Pixels", "value"].to_numpy()
              for name, analysis in analyses.items()}
    _comparison_hist(axes[0], values, PILEUP_COLORS, "ColliderML native pixel activation", "Path-length proxy (mm)")
    codes, counts = np.unique(atlas["values"], return_counts=True)
    if len(codes):
        axes[1].bar(codes, counts / counts.sum(), color=PUBLIC_COLOR)
    axes[1].set(xlabel="Native ToT code", ylabel="Fraction of nonzero entries (log scale)", yscale="log",
                title="ATLAS native pixel codes (different units)")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Native amplitudes: no cross-detector calibration or rescaling")
    plt.show()


ATHENA_QUANTIZATION_REVISION = "46f1900866d629c4241b75f4f178d3038abd5b1c"
CODE_PERCENTILES = (50, 90, 99, 99.5, 99.9, 99.99)


def select_atlas_reference_files(folder: Path, count: int = 20) -> list[Path]:
    """Choose evenly spaced files; alternating positions define disjoint cohorts."""
    files = sorted(Path(folder).glob("*.pyg"))
    if count < 4 or count > len(files):
        raise ValueError(f"Choose between 4 and {len(files)} ATLAS reference files")
    indices = np.rint(np.linspace(0, len(files) - 1, count)).astype(int)
    return [files[index] for index in indices]


def _histogram_percentile(counts, percentile):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        return np.nan
    rank = (counts.sum() - 1) * percentile / 100
    cumulative = counts.cumsum()
    lower = np.searchsorted(cumulative, int(np.floor(rank)), side="right")
    upper = np.searchsorted(cumulative, int(np.ceil(rank)), side="right")
    return lower + (upper - lower) * (rank - np.floor(rank))


def collect_atlas_code_reference(files, python_executable=None, resamples=1000,
                                 seed=7, reader=None) -> dict:
    """Read one graph at a time; retain only per-event counts, never image arrays."""
    files = [Path(path) for path in files]
    if len(files) < 4 or len(set(path.resolve() for path in files)) != len(files):
        raise ValueError("ATLAS reference needs at least four distinct files")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    reader = reader or (lambda path: load_atlas_pixels(path, python_executable))
    records, event_ids = [], set()
    native_histograms = {cohort: np.zeros(256, dtype=np.int64) for cohort in ("calibration", "held_out")}
    for index, path in enumerate(files):
        atlas = reader(path)
        values = np.asarray(atlas["values"])
        if not len(values) or not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{path}: expected nonempty integer pixel codes")
        if np.any((values < 1) | (values > 255)):
            raise ValueError(f"{path}: ATLAS code outside positive uint8 range")
        event_id = str(atlas["event_id"])
        if event_id in event_ids:
            raise ValueError(f"Repeated ATLAS event {event_id}")
        event_ids.add(event_id)
        native_hist = np.bincount(values.astype(int), minlength=256)
        hist = np.r_[native_hist[:15], native_hist[15:].sum()]
        cohort = "calibration" if index % 2 == 0 else "held_out"
        native_histograms[cohort] += native_hist
        cluster_count = len(atlas["n_cells"])
        code15_clusters = np.asarray(atlas["contains_code15"], dtype=bool)
        overflow_clusters = np.asarray(atlas["contains_overflow15"], dtype=bool)
        if (len(code15_clusters) != cluster_count or len(overflow_clusters) != cluster_count
                or cluster_count == 0 or np.asarray(atlas["n_cells"]).sum() != len(values)):
            raise ValueError(f"{path}: invalid pixel-cluster summary")
        row = dict(path=str(path), event_id=event_id,
                   cohort=cohort,
                   clusters=cluster_count, nonzero_cells=int(hist.sum()),
                   native_max_code=int(values.max()), cells_above15=int(native_hist[16:].sum()),
                   native_code15_cells=int(native_hist[15]),
                   code15_clusters=int(code15_clusters.sum()),
                   overflow15_clusters=int(overflow_clusters.sum()),
                   edge_clusters=int(np.asarray(atlas["touches_edge"]).sum()))
        row.update({f"code_{code}": int(hist[code]) for code in range(1, 16)})
        row.update({f"q{p:g}": _histogram_percentile(native_hist, p) for p in CODE_PERCENTILES})
        row["code15_cell_fraction"] = native_hist[15] / hist.sum()
        row["overflow15_cell_fraction"] = hist[15] / hist.sum()
        row["code15_cluster_fraction"] = code15_clusters.mean()
        row["overflow15_cluster_fraction"] = overflow_clusters.mean()
        records.append(row)
        del atlas, values, code15_clusters, overflow_clusters
    events = pd.DataFrame(records)
    histograms, pooled, intervals = {}, [], []
    rng = np.random.default_rng(seed)
    for cohort, group in events.groupby("cohort", sort=False):
        matrix = group[[f"code_{code}" for code in range(1, 16)]].to_numpy()
        hist = np.r_[0, matrix.sum(axis=0)]
        histograms[cohort] = hist
        row = dict(cohort=cohort, events=len(group), nonzero_cells=int(hist.sum()),
                   clusters=int(group.clusters.sum()),
                   native_max_code=int(group.native_max_code.max()),
                   cells_above15=int(group.cells_above15.sum()),
                   code15_cell_fraction=group.native_code15_cells.sum() / hist.sum(),
                   overflow15_cell_fraction=hist[15] / hist.sum(),
                   code15_cluster_fraction=group.code15_clusters.sum() / group.clusters.sum(),
                   overflow15_cluster_fraction=group.overflow15_clusters.sum() / group.clusters.sum(),
                   window_edge_fraction=group.edge_clusters.sum() / group.clusters.sum())
        row.update({f"q{p:g}": _histogram_percentile(native_histograms[cohort], p) for p in CODE_PERCENTILES})
        pooled.append(row)
        # Resample whole events so correlated cells do not masquerade as independent trials.
        draws = rng.integers(0, len(group), size=(resamples, len(group)))
        for metric, numerator, denominator in (
                ("code15_cell_fraction", group.native_code15_cells.to_numpy(), group.nonzero_cells.to_numpy()),
                ("overflow15_cell_fraction", group.code_15.to_numpy(), group.nonzero_cells.to_numpy()),
                ("code15_cluster_fraction", group.code15_clusters.to_numpy(), group.clusters.to_numpy()),
                ("overflow15_cluster_fraction", group.overflow15_clusters.to_numpy(), group.clusters.to_numpy())):
            fractions = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
            low, high = np.quantile(fractions, [0.025, 0.975])
            intervals.append(dict(cohort=cohort, metric=metric, lower95=low, upper95=high,
                                  resamples=resamples, seed=seed))
    return dict(events=events, pooled=pd.DataFrame(pooled),
                intervals=pd.DataFrame(intervals), histograms=histograms,
                native_histograms=native_histograms)


def _positive_activations(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Expected a nonempty vector of finite positive activations")
    return values


def athena_charge_for_tot(tot, threshold=600., c=4500.):
    if not np.isfinite([tot, threshold, c]).all() or not 0 < tot < 14 or threshold <= 0 or c <= 0:
        raise ValueError("Require 0 < target ToT < 14 and positive calibration parameters")
    return (c * tot + 14 * threshold) / (14 - tot)


def athena_response(values, gain, threshold=600., c=4500., activation_min=0.01):
    values = _positive_activations(values)
    minimum_charge = athena_charge_for_tot(1., threshold, c)
    if not np.isfinite([gain, activation_min]).all() or gain <= 0 or activation_min <= 0:
        raise ValueError("Gain and activation minimum must be finite and positive")
    if gain * activation_min < minimum_charge * (1 - 1e-12):
        raise ValueError("Gain maps the retained-cell threshold below ToT 1")
    if np.any(values < activation_min):
        raise ValueError("Activation below the declared retained-cell threshold")
    # Algebraically equivalent to 14 * (Q - threshold) / (Q + C), avoiding Q overflow.
    response = 14 * (1 - ((threshold + c) / gain) / (values + c / gain))
    return np.maximum(response, 1.)  # Protect roundoff at the exact minimum-code boundary.


def fit_athena_gains(values, targets=(12., 13., 13.5), percentile=99.5,
                     threshold=600., c=4500., activation_min=0.01):
    values = _positive_activations(values)
    if not 0 < percentile < 100 or not len(targets) or len(set(targets)) != len(targets):
        raise ValueError("Choose a percentile inside (0,100) and distinct targets")
    reference = np.percentile(values, percentile)
    rows = []
    for target in targets:
        gain = athena_charge_for_tot(target, threshold, c) / reference
        minimum_tot = athena_response(np.array([activation_min]), gain, threshold, c, activation_min)[0]
        rows.append(dict(target_tot=target, gain=gain, calibration_percentile=percentile,
                         reference_activation=reference, threshold=threshold, c=c,
                         activation_min=activation_min, minimum_retained_tot=minimum_tot))
    return pd.DataFrame(rows)


def athena_integer_codes(response, rng=None):
    response = np.asarray(response, dtype=float)
    if not np.isfinite(response).all() or np.any(response < 1) or np.any(response > 14):
        raise ValueError("Continuous occupancy-preserving ToT must be inside 1..14")
    signal = response if rng is None else rng.normal(response, 0.4 + 0.02 * response)
    truncated = np.trunc(signal)
    return np.clip(truncated, 1, 14).astype(np.uint8), {
        "lower_floor_cells": int((truncated < 1).sum()),
        "upper_clamp_cells": int((truncated > 14).sum())}


def fit_empirical_boundaries(values, histogram):
    values = _positive_activations(values)
    histogram = np.asarray(histogram, dtype=float)
    if (histogram.shape != (16,) or not np.isfinite(histogram).all()
            or np.any(histogram < 0) or histogram[0] != 0 or histogram.sum() <= 0):
        raise ValueError("Expected positive-code counts in a 16-bin histogram, with bin zero empty")
    probabilities = histogram[1:].cumsum()[:-1] / histogram.sum()
    return np.quantile(values, probabilities, method="linear")


def empirical_codes(values, boundaries):
    values = _positive_activations(values)
    boundaries = np.asarray(boundaries, dtype=float)
    if (boundaries.shape != (14,) or not np.isfinite(boundaries).all()
            or np.any(np.diff(boundaries) < 0)):
        raise ValueError("Expected fourteen finite, nondecreasing boundaries")
    return (1 + np.searchsorted(boundaries, values, side="right")).astype(np.uint8)


def weighted_channel_centroids(cells, weights):
    """Return centroids in first-seen full-key order, without mutating cells."""
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(cells),) or not np.isfinite(weights).all()
            or np.any(weights <= 0)):
        raise ValueError("Every retained cell needs a finite positive weight")
    keys = pd.MultiIndex.from_frame(cells[MEASUREMENT_KEY])
    inverse, unique = pd.factorize(keys, sort=False)
    sums = np.bincount(inverse, weights=weights, minlength=len(unique))
    centroids = np.column_stack([
        np.bincount(inverse, weights=weights * cells[channel].to_numpy()) / sums
        for channel in ("channel0", "channel1")])
    return centroids


@dataclass
class PixelQuantizationStudy:
    gains: pd.DataFrame
    boundaries: np.ndarray
    cells: dict
    clusters: dict
    weights: dict
    shifts: dict
    summary: pd.DataFrame
    code_comparison: pd.DataFrame


def run_pixel_quantization(analyses, calibration_values, atlas_reference,
                           targets=(12., 13., 13.5), percentile=99.5,
                           threshold=600., c=4500., activation_min=0.01, seed=7,
                           include_smearing=False, empirical_boundaries=None):
    """Calibrate once externally, then apply identical maps to selected events."""
    gains = fit_athena_gains(calibration_values, targets, percentile, threshold, c, activation_min)
    bounds = (fit_empirical_boundaries(calibration_values, atlas_reference["histograms"]["calibration"])
              if empirical_boundaries is None else np.asarray(empirical_boundaries, dtype=float).copy())
    empirical_codes([activation_min], bounds)
    cells_by_sample, clusters_by_sample, weights, shifts, rows, comparisons = {}, {}, {}, {}, [], []
    for sample, analysis in analyses.items():
        cells = analysis.cells.loc[analysis.cells.detector_type == "Pixels"].reset_index(drop=True)
        if cells.empty:
            raise ValueError(f"{sample}: selected event has no pixel cells")
        keys = cells[MEASUREMENT_KEY].drop_duplicates()
        clusters = keys.merge(analysis.clusters, on=MEASUREMENT_KEY, validate="one_to_one", sort=False)
        native = weighted_channel_centroids(cells, cells.value)
        cells_by_sample[sample], clusters_by_sample[sample] = cells, clusters
        variants = {"empirical": empirical_codes(cells.value, bounds)}
        clamp_counts = {"empirical": dict(lower_floor_cells=0, upper_clamp_cells=0)}
        for model_index, gain_row in gains.iterrows():
            prefix = f"athena_T{gain_row.target_tot:g}"
            continuous = athena_response(cells.value, gain_row.gain, threshold, c, activation_min)
            variants[prefix + "_continuous"] = continuous
            variants[prefix + "_integer"], clamp_counts[prefix + "_integer"] = athena_integer_codes(continuous)
            if include_smearing:
                # Stable streams per sample, event, and configured gain index.
                sample_index = list(PILEUP_COLORS).index(sample)
                rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index, analysis.event, model_index]))
                variants[prefix + "_smeared"], clamp_counts[prefix + "_smeared"] = athena_integer_codes(continuous, rng)
        for variant, value in variants.items():
            key = (sample, variant)
            weights[key] = value
            centers = weighted_channel_centroids(cells, value)
            shift = {"native_to_transformed": np.linalg.norm(centers - native, axis=1)}
            if not variant.endswith("_continuous") and variant != "empirical":
                prefix = variant.rsplit("_", 1)[0]
                continuous_centers = weighted_channel_centroids(cells, variants[prefix + "_continuous"])
                shift["continuous_to_transformed"] = np.linalg.norm(centers - continuous_centers, axis=1)
            shifts[key] = shift
            is_code = not variant.endswith("_continuous")
            hist = np.bincount(value.astype(int), minlength=16) if is_code else None
            saturated = (value == (15 if variant == "empirical" else 14)) if is_code else np.zeros(len(value), bool)
            parent_ids = pd.factorize(pd.MultiIndex.from_frame(cells[MEASUREMENT_KEY]), sort=False)[0]
            saturated_clusters = np.bincount(parent_ids, weights=saturated, minlength=len(clusters)) > 0
            fractions = dict(code14_cell_fraction=float(np.mean(value == 14)) if is_code else np.nan,
                             code15_cell_fraction=float(np.mean(value == 15)) if is_code else np.nan,
                             upper_code_cluster_fraction=float(saturated_clusters.mean()) if is_code else np.nan)
            for transition, distances in shift.items():
                for grouping in (None, "n_cells", "contributor_group"):
                    groups = [("all", np.arange(len(clusters)))] if grouping is None else clusters.groupby(grouping, sort=True).indices.items()
                    for group, indices in groups:
                        indices = np.asarray(indices)
                        d = distances[indices]
                        rows.append(dict(sample=sample, event=analysis.event, variant=variant,
                                         transition=transition, grouping=grouping or "all", group=group,
                                         clusters=len(indices), cells=int(clusters.iloc[indices].n_cells.sum()),
                                         median=np.median(d), q95=np.quantile(d, .95),
                                         q99=np.quantile(d, .99), maximum=d.max(),
                                         **(fractions if grouping is None else {}),
                                         **(clamp_counts.get(variant, {}) if grouping is None else {})))
            if is_code:
                for cohort, atlas_hist in atlas_reference["histograms"].items():
                    for code in range(1, 16):
                        observed = hist[code] / hist.sum()
                        reference = atlas_hist[code] / atlas_hist.sum()
                        comparisons.append(dict(sample=sample, event=analysis.event, variant=variant,
                                                atlas_cohort=cohort, code=code, fraction=observed,
                                                atlas_fraction=reference, difference=observed - reference))
    return PixelQuantizationStudy(gains, bounds, cells_by_sample, clusters_by_sample,
                                  weights, shifts, pd.DataFrame(rows), pd.DataFrame(comparisons))


def pixel_calibration_values(root: Path, event=0):
    table = _pixel_event_table(root, "pu200", "cells", event, columns=["value"])
    return _positive_activations(table["value"].to_numpy())


def _pixel_event_table(root, sample, kind, event, columns=None):
    path = Path(root) / sample / f"{kind}.parquet"
    selection = ((ds.field("local_event") == event)
                 & (ds.field("geometry_id") >= pa.scalar(16 << 56, type=pa.uint64()))
                 & (ds.field("geometry_id") < pa.scalar(19 << 56, type=pa.uint64())))
    table = ds.dataset(path, format="parquet").to_table(columns=columns, filter=selection)
    if not len(table):
        raise ValueError(f"{sample}: event {event} has no pixel {kind}")
    return table


def load_pixel_analysis(root, sample, event):
    """Load one event's pixels only, preserving the full-key integrity checks."""
    hits = _pixel_event_table(root, sample, "hits", event)
    analysis = build_cluster_analysis(hits, _pixel_event_table(root, sample, "cells", event), sample, event)
    if {"local0", "local1"} <= set(hits.column_names):
        analysis.clusters = analysis.clusters.merge(
            hits.select(MEASUREMENT_KEY + ["local0", "local1"]).to_pandas(),
            on=MEASUREMENT_KEY, validate="one_to_one", sort=False)
    return analysis


def _study_events(events, name):
    events = tuple(events)
    if (not events or any(not isinstance(e, (int, np.integer)) or e < 0 for e in events)
            or len(set(events)) != len(events)):
        raise ValueError(f"{name} must contain distinct nonnegative integer event IDs")
    return tuple(int(e) for e in events)


def _counted_quantiles(support, counts, probabilities):
    """Exact NumPy linear quantiles of a sorted, count-compressed population."""
    ranks = np.asarray(probabilities) * (counts.sum() - 1)
    cumulative = counts.cumsum()
    low = support[np.searchsorted(cumulative, np.floor(ranks), side="right")]
    high = support[np.searchsorted(cumulative, np.ceil(ranks), side="right")]
    return low + (high - low) * (ranks - np.floor(ranks))


def _merge_activation_counts(support, counts, values):
    event_support, event_counts = np.unique(values, return_counts=True)
    combined, inverse = np.unique(np.concatenate([support, event_support]), return_inverse=True)
    combined_counts = np.zeros(len(combined), dtype=np.int64)
    np.add.at(combined_counts, inverse, np.concatenate([counts, event_counts]))
    return combined, combined_counts


@dataclass
class EmpiricalCalibration:
    events: tuple
    boundaries: np.ndarray
    stability: pd.DataFrame
    atlas_histogram: np.ndarray


@dataclass
class LinearCalibration:
    activation_min: float
    overflow_edge: float
    target_overflow_fraction: float
    boundaries: np.ndarray


def fit_linear_calibration(empirical, activation_min=0.01):
    """Fourteen equal-width bins below the empirical calibration overflow edge."""
    edge = float(empirical.boundaries[-1])
    if not np.isfinite([activation_min, edge]).all() or not 0 < activation_min < edge:
        raise ValueError("Linear range needs 0 < activation minimum < overflow edge")
    histogram = np.asarray(empirical.atlas_histogram)
    fit_empirical_boundaries([1.], histogram)
    return LinearCalibration(activation_min, edge, histogram[15] / histogram.sum(),
                             np.linspace(activation_min, edge, 15)[1:])


def linear_codes(values, calibration):
    values = _positive_activations(values)
    if np.any(values < calibration.activation_min):
        raise ValueError("Activation below the declared linear range minimum")
    return empirical_codes(values, calibration.boundaries)


def fit_multievent_empirical(root, atlas_reference, events=range(25),
                            checkpoints=(1, 2, 5, 10, 25), progress=print):
    """Pool exact activation counts sequentially, never retaining event cell tables."""
    events = _study_events(events, "Calibration events")
    histogram = np.asarray(atlas_reference["histograms"]["calibration"], dtype=float).copy()
    fit_empirical_boundaries([1.], histogram)  # Validate reference counts before reading events.
    probabilities = histogram[1:].cumsum()[:-1] / histogram.sum()
    checkpoints = set(checkpoints) | {len(events)}
    if any(not isinstance(n, (int, np.integer)) or n < 1 or n > len(events) for n in checkpoints):
        raise ValueError("Checkpoints must be positive counts within the calibration cohort")
    support, counts, rows = np.array([], dtype=float), np.array([], dtype=np.int64), []
    for index, event in enumerate(events, 1):
        values = pixel_calibration_values(root, event)
        support, counts = _merge_activation_counts(support, counts, values)
        if index in checkpoints:
            boundaries = _counted_quantiles(support, counts, probabilities)
            rows.extend(dict(calibration_events=index, last_event=event, cells=int(counts.sum()),
                             boundary=code, activation=value)
                        for code, value in enumerate(boundaries, 1))
        if progress is not None:
            progress(f"Empirical calibration {index}/{len(events)}: pu200 event {event}, {len(values):,} pixel cells")
        del values
    stability = pd.DataFrame(rows)
    final = stability.query("calibration_events == @index").activation.to_numpy(copy=True)
    stability["final_activation"] = stability.boundary.map(dict(enumerate(final, 1)))
    stability["relative_change_to_final"] = stability.activation / stability.final_activation - 1
    return EmpiricalCalibration(events, final, stability, histogram)


@dataclass
class EmpiricalValidation:
    events: pd.DataFrame
    pooled: pd.DataFrame
    shifts: pd.DataFrame
    survival: pd.DataFrame
    convergence: pd.DataFrame
    code_comparison: pd.DataFrame
    largest_shifts: pd.DataFrame


def _shift_statistics(values):
    return dict(median=float(np.median(values)), q95=float(np.quantile(values, .95)),
                q99=float(np.quantile(values, .99)), maximum=float(np.max(values)))


def validate_multievent_empirical(root, calibration, atlas_reference,
                                 held_out_events=range(25, 50), samples=("pu0", "pu200"),
                                 progress=print, loader=None):
    """Apply frozen maps sequentially; retain only cluster shifts and compact counts."""
    held_out = _study_events(held_out_events, "Held-out events")
    if set(calibration.events) & set(held_out):
        raise ValueError("Calibration and held-out ColliderML events must be disjoint")
    if not samples or len(set(samples)) != len(samples) or not set(samples) <= set(PILEUP_COLORS):
        raise ValueError("Choose distinct pu0/pu200 samples")
    if not np.array_equal(calibration.atlas_histogram,
                          atlas_reference["histograms"]["calibration"]):
        raise ValueError("ATLAS calibration reference changed after fitting")
    loader = loader or load_pixel_analysis
    records, shift_records, extremes, convergence = [], [], [], []
    distance_parts, histograms, cohort_totals = {}, {}, {}
    checkpoints = {int(n): frame.sort_values("boundary").activation.to_numpy()
                   for n, frame in calibration.stability.groupby("calibration_events")}
    # Full-range survival curves use exact shift support, not clipped display bins.
    for sample in samples:
        for event in (*calibration.events, *held_out):
            cohort = "calibration" if event in calibration.events else "held_out"
            analysis = loader(root, sample, event)
            cells = analysis.cells
            if not (cells.detector_type == "Pixels").all():
                raise ValueError("Campaign validation expects pixel-only event analysis")
            clusters = cells[MEASUREMENT_KEY].drop_duplicates().merge(
                analysis.clusters, on=MEASUREMENT_KEY, validate="one_to_one", sort=False)
            native = weighted_channel_centroids(cells, cells.value)
            codes = empirical_codes(cells.value, calibration.boundaries)
            shift = np.linalg.norm(weighted_channel_centroids(cells, codes) - native, axis=1)
            parent = pd.factorize(pd.MultiIndex.from_frame(cells[MEASUREMENT_KEY]), sort=False)[0]
            overflow_clusters = np.bincount(parent, weights=codes == 15, minlength=len(clusters)) > 0
            hist = np.bincount(codes, minlength=16)
            key = (sample, cohort)
            histograms[key] = histograms.get(key, np.zeros(16, dtype=np.int64)) + hist
            totals = cohort_totals.setdefault(key, dict(events=0, cells=0, clusters=0, overflow_clusters=0))
            for name, value in dict(events=1, cells=len(cells), clusters=len(clusters),
                                    overflow_clusters=int(overflow_clusters.sum())).items():
                totals[name] += value
            record = dict(sample=sample, cohort=cohort, event=event, cells=len(cells),
                          clusters=len(clusters), code15_cell_fraction=float(np.mean(codes == 15)),
                          code15_cluster_fraction=float(overflow_clusters.mean()), **_shift_statistics(shift))
            record.update({f"code_{code}": int(hist[code]) for code in range(1, 16)})
            records.append(record)
            for grouping in ("all", "n_cells", "contributor_group"):
                groups = [("all", np.arange(len(clusters)))] if grouping == "all" else clusters.groupby(grouping).indices.items()
                for group, indices in groups:
                    d = shift[indices]
                    distance_parts.setdefault((*key, grouping, group), []).append(d)
                    shift_records.append(dict(sample=sample, cohort=cohort, event=event,
                                              grouping=grouping, group=group, clusters=len(d),
                                              **_shift_statistics(d)))
            for index in np.argsort(-shift, kind="stable")[:3]:
                row = clusters.iloc[index]
                extremes.append(dict(sample=sample, cohort=cohort,
                                     **{name: row[name] for name in MEASUREMENT_KEY},
                                     n_cells=int(row.n_cells), contributor_group=row.contributor_group,
                                     displacement=float(shift[index])))
            if cohort == "held_out":
                for count, bounds in checkpoints.items():
                    h = np.bincount(empirical_codes(cells.value, bounds), minlength=16)
                    convergence.append(dict(sample=sample, event=event, calibration_events=count,
                                            **{f"code_{code}": int(h[code]) for code in range(1, 16)}))
            if progress is not None:
                progress(f"Empirical validation: {sample} event {event} ({cohort}), {len(clusters):,} clusters")
            del analysis, cells, clusters, native, codes, shift, parent
    pooled_shifts, survival = [], []
    for (sample, cohort, grouping, group), parts in distance_parts.items():
        values = np.concatenate(parts)
        pooled_shifts.append(dict(sample=sample, cohort=cohort, event="pooled", grouping=grouping,
                                  group=group, clusters=len(values), **_shift_statistics(values)))
        if grouping in ("all", "contributor_group"):
            support, counts = np.unique(values, return_counts=True)
            # Keep exact values for statistics; display an evenly spaced rank subset of the ECDF.
            indices = np.unique(np.linspace(0, len(support) - 1, min(2000, len(support)), dtype=int))
            fractions = counts[::-1].cumsum()[::-1] / counts.sum()
            survival.extend(dict(sample=sample, cohort=cohort, grouping=grouping, group=group,
                                 displacement=support[i], fraction=fractions[i]) for i in indices)
    comparisons, pooled = [], []
    for (sample, cohort), histogram in histograms.items():
        totals = cohort_totals[(sample, cohort)]
        pooled.append(dict(sample=sample, cohort=cohort, **totals,
                           code15_cell_fraction=histogram[15] / histogram.sum(),
                           code15_cluster_fraction=totals["overflow_clusters"] / totals["clusters"]))
        for atlas_cohort, atlas_hist in atlas_reference["histograms"].items():
            for code in range(1, 16):
                fraction = histogram[code] / histogram.sum()
                reference = atlas_hist[code] / atlas_hist.sum()
                comparisons.append(dict(sample=sample, cohort=cohort, atlas_cohort=atlas_cohort,
                                        code=code, fraction=fraction, atlas_fraction=reference,
                                        difference=fraction - reference))
    convergence_rows = []
    for (sample, count), frame in pd.DataFrame(convergence).groupby(["sample", "calibration_events"]):
        histogram = frame[[f"code_{code}" for code in range(1, 16)]].sum().to_numpy()
        target = np.asarray(atlas_reference["histograms"]["held_out"])[1:]
        difference = histogram / histogram.sum() - target / target.sum()
        convergence_rows.append(dict(sample=sample, calibration_events=count,
                                     total_variation=0.5 * np.abs(difference).sum(),
                                     maximum_code_deviation=np.abs(difference).max(),
                                     code15_fraction=histogram[-1] / histogram.sum()))
    return EmpiricalValidation(pd.DataFrame(records), pd.DataFrame(pooled),
                               pd.concat([pd.DataFrame(shift_records), pd.DataFrame(pooled_shifts)], ignore_index=True),
                               pd.DataFrame(survival), pd.DataFrame(convergence_rows), pd.DataFrame(comparisons),
                               pd.DataFrame(extremes).sort_values("displacement", ascending=False).reset_index(drop=True))


def plot_multievent_empirical(calibration, validation, atlas_reference):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for boundary, frame in calibration.stability.groupby("boundary"):
        axes[0].plot(frame.calibration_events, frame.relative_change_to_final * 100,
                     ".-", label=f"{boundary}/{boundary + 1}")
    axes[0].set(xlabel="Calibration event count", ylabel="Boundary change from final (%)",
                title="Exact pooled activation boundaries")
    axes[0].legend(ncol=3, fontsize=7)
    for sample, frame in validation.convergence.groupby("sample"):
        axes[1].plot(frame.calibration_events, frame.total_variation, "o-",
                     color=PILEUP_COLORS[sample], label=sample)
    axes[1].set(xlabel="Calibration event count", ylabel="Total variation distance",
                title="Held-out ColliderML vs held-out ATLAS")
    axes[1].legend()
    plt.show()
    for cohort in ("calibration", "held_out"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        comparison = validation.code_comparison.query("cohort == @cohort and atlas_cohort == 'held_out'")
        for sample, frame in comparison.groupby("sample"):
            color = PILEUP_COLORS[sample]
            axes[0].plot(frame.code, frame.fraction, "o-", color=color, label=sample)
            frame = validation.survival.query("sample == @sample and cohort == @cohort and grouping == 'all'")
            axes[1].step(frame.displacement, frame.fraction, where="post", color=color, label=sample)
            frame = validation.events.query("sample == @sample and cohort == @cohort")
            axes[2].plot(frame.event, frame.code15_cell_fraction, "o-", color=color, label=f"{sample} cells")
            axes[2].plot(frame.event, frame.code15_cluster_fraction, ":", color=color, label=f"{sample} clusters")
        for atlas_cohort, style in (("calibration", ":"), ("held_out", "--")):
            hist = atlas_reference["histograms"][atlas_cohort]
            axes[0].plot(np.arange(1, 16), hist[1:] / hist.sum(), style,
                         color=PUBLIC_COLOR, label=f"ATLAS {atlas_cohort}")
        axes[0].set(yscale="log", xlabel="Positive pixel code", ylabel="Cell fraction", title="Frozen empirical mapping")
        axes[1].set(yscale="log", xlabel="Centroid displacement (channel units)",
                    ylabel="Fraction >= displacement", title="Full-range pooled survival")
        axes[2].set(xlabel="ColliderML event", ylabel="Code-15 fraction", title="Separate cell/cluster denominators")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.suptitle(f"ColliderML {cohort} events")
        plt.show()
    groups = ("1 particle", "2 particles", "3+ particles", "Unknown/incomplete")
    fig, axes = plt.subplots(1, 4, figsize=(19, 4), constrained_layout=True)
    for ax, group in zip(axes, groups):
        for sample, color in PILEUP_COLORS.items():
            frame = validation.survival.query("sample == @sample and cohort == 'held_out' and grouping == 'contributor_group' and group == @group")
            if len(frame):
                ax.step(frame.displacement, frame.fraction, where="post", color=color, label=sample)
        ax.set(yscale="log", xlabel="Displacement (channel units)", ylabel="Fraction >= displacement", title=group)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    plt.show()


def quantization_source_checksums(root: Path):
    """Hash the six source Parquets before/after this in-memory analysis."""
    result = {}
    for sample in PILEUP_COLORS:
        for name in ("hits", "cells", "particles"):
            path = Path(root) / sample / f"{name}.parquet"
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(path)] = digest.hexdigest()
    return result


def plot_atlas_code_reference(reference):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for cohort, style in (("calibration", "-"), ("held_out", "--")):
        hist = reference["histograms"][cohort]
        axes[0].plot(np.arange(1, 16), hist[1:] / hist.sum(), style, color=PUBLIC_COLOR, label=cohort)
        events = reference["events"].query("cohort == @cohort")
        for ax, metric in zip(axes[1:], ("code15_cell_fraction", "code15_cluster_fraction")):
            ax.plot(np.arange(len(events)), events[metric], style + "o", color=PUBLIC_COLOR, label=cohort)
            ax.set(xlabel="Event index within cohort", ylabel="Fraction", title=metric.replace("_", " "))
    axes[0].set(yscale="log", xlabel="ATLAS pixel code (>=15 folded to 15)",
                ylabel="Nonzero-cell fraction", title="ATLAS reference cohorts")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_pixel_quantization(study, atlas_reference, show_continuous=False):
    values = np.concatenate([frame.value.to_numpy() for frame in study.cells.values()])
    x = np.geomspace(study.gains.activation_min.iloc[0], values.max(), 500)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for row in study.gains.itertuples():
        response = athena_response(x, row.gain, row.threshold, row.c, row.activation_min)
        if show_continuous:
            ax.plot(x, response, label=f"Athena-inspired T{row.target_tot:g} continuous diagnostic")
        else:
            ax.step(x, athena_integer_codes(response)[0], where="post",
                    label=f"Athena-inspired T{row.target_tot:g} integer")
    ax.step(x, empirical_codes(x, study.boundaries), where="post", color="black", label="Empirical ATLAS match")
    ax.set(xscale="log", xlabel="Native pixel activation (mm proxy)", ylabel="Response / code" if show_continuous else "Integer code",
           ylim=(.5, 15.5), title="Frozen mappings: no per-event normalization")
    ax.legend()
    plt.show()
    for variant in sorted({variant for _, variant in study.weights}):
        continuous = variant.endswith("_continuous")
        if continuous and not show_continuous:
            continue
        transitions = ["native_to_transformed"]
        if not continuous and variant != "empirical":
            transitions.append("continuous_to_transformed")
        columns = 1 + len(transitions)
        fig, axes = plt.subplots(1, columns, figsize=(5.3 * columns, 4), constrained_layout=True)
        for sample, color in PILEUP_COLORS.items():
            weights = study.weights[(sample, variant)]
            if continuous:
                axes[0].hist(weights, bins=np.linspace(1, 14, 53), density=True, histtype="step", color=color, label=sample)
            else:
                hist = np.bincount(weights.astype(int), minlength=16)
                axes[0].plot(np.arange(1, 16), hist[1:] / hist.sum(), "o-", color=color, label=sample)
            for index, transition in enumerate(transitions, start=1):
                distances = study.shifts[(sample, variant)].get(transition)
                if distances is None:
                    continue
                support, counts = np.unique(distances, return_counts=True)
                axes[index].step(support, np.cumsum(counts[::-1])[::-1] / counts.sum(),
                                 where="post", color=color, label=f"{sample}, event {study.clusters[sample].local_event.iloc[0]}")
        if not continuous:
            for cohort, style in (("calibration", ":"), ("held_out", "--")):
                hist = atlas_reference["histograms"][cohort]
                axes[0].plot(np.arange(1, 16), hist[1:] / hist.sum(), style, color=PUBLIC_COLOR, label=f"ATLAS {cohort}")
        axes[0].set(yscale="log", xlabel="Continuous ToT" if continuous else "Positive code (ATLAS >=15 folded to 15)",
                    ylabel="Density" if continuous else "Cell fraction", title="Pixel amplitude")
        for index, transition in enumerate(transitions, start=1):
            title = "Native to transformed" if transition == "native_to_transformed" else "Continuous to integer/smeared"
            axes[index].set(yscale="log", xlabel="Centroid displacement (channel units)",
                            ylabel="Fraction >= displacement", title=title)
        for ax in axes:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.suptitle(variant)
        plt.show()


def plot_quantization_examples(study, variant="empirical", per_sample=3):
    """Show raw and transformed values on identical actual channel addresses."""
    records = []
    for sample in study.clusters:
        distances = study.shifts[(sample, variant)]["native_to_transformed"]
        indices = np.argsort(-distances, kind="stable")[:per_sample]
        clusters, cells = study.clusters[sample], study.cells[sample]
        weights = study.weights[(sample, variant)]
        fig, axes = plt.subplots(len(indices), 2, figsize=(11, 3.2 * len(indices)), squeeze=False,
                                 constrained_layout=True)
        for row, cluster_index in enumerate(indices):
            cluster = clusters.iloc[cluster_index]
            mask = np.ones(len(cells), dtype=bool)
            for key in MEASUREMENT_KEY:
                mask &= cells[key].to_numpy() == cluster[key]
            selected = cells.loc[mask]
            for ax, amplitudes, title in ((axes[row, 0], selected.value.to_numpy(), "Native activation (mm proxy)"),
                                          (axes[row, 1], weights[mask], variant)):
                image_values, extent = _cluster_cell_image(selected.channel0, selected.channel1, amplitudes)
                shown = ax.imshow(image_values, extent=extent, origin="lower", interpolation="nearest",
                                  cmap="viridis", aspect="equal")
                center = np.average(selected[["channel0", "channel1"]], axis=0, weights=amplitudes)
                ax.plot(*center, "+", color="red", markersize=12)
                ax.set(aspect="equal", xlabel="channel0", ylabel="channel1", title=title,
                       xlim=(selected.channel0.min() - 1, selected.channel0.max() + 1),
                       ylim=(selected.channel1.min() - 1, selected.channel1.max() + 1))
                fig.colorbar(shown, ax=ax)
            record = {key: cluster[key] for key in MEASUREMENT_KEY}
            record.update(sample=sample, cells=len(selected), displacement=distances[cluster_index],
                          contributor_group=cluster.contributor_group)
            records.append(record)
            axes[row, 0].set_title(f"{sample} event {cluster.local_event}, measurement {cluster.measurement_id}\nNative activation")
            axes[row, 1].set_title(f"{variant}; shift {distances[cluster_index]:.4g} channels")
        plt.show()
    return pd.DataFrame(records)


def plot_campaign_examples(root, calibration, validation, per_sample=2):
    """Reload only deterministic largest held-out examples, one event at a time."""
    records = []
    for sample in validation.events["sample"].unique():
        examples = validation.largest_shifts.query("sample == @sample and cohort == 'held_out'").head(per_sample)
        for example in examples.to_dict("records"):
            analysis = load_pixel_analysis(root, sample, int(example["local_event"]))
            mask = np.ones(len(analysis.cells), dtype=bool)
            cluster_mask = np.ones(len(analysis.clusters), dtype=bool)
            for key in MEASUREMENT_KEY:
                mask &= analysis.cells[key].to_numpy() == example[key]
                cluster_mask &= analysis.clusters[key].to_numpy() == example[key]
            cells = analysis.cells.loc[mask].reset_index(drop=True)
            clusters = analysis.clusters.loc[cluster_mask].reset_index(drop=True)
            if len(clusters) != 1 or len(cells) != example["n_cells"]:
                raise ValueError("Largest-shift example membership changed")
            codes = empirical_codes(cells.value, calibration.boundaries)
            distance = np.linalg.norm(weighted_channel_centroids(cells, codes)
                                      - weighted_channel_centroids(cells, cells.value), axis=1)
            if not np.isclose(distance[0], example["displacement"], rtol=1e-12, atol=1e-12):
                raise ValueError("Largest-shift example displacement changed")
            study = PixelQuantizationStudy(pd.DataFrame(), calibration.boundaries,
                                            {sample: cells}, {sample: clusters},
                                            {(sample, "empirical"): codes},
                                            {(sample, "empirical"): {"native_to_transformed": distance}},
                                            pd.DataFrame(), pd.DataFrame())
            records.append(plot_quantization_examples(study, per_sample=1))
            del analysis, study
    return pd.concat(records, ignore_index=True)


class RootMeasurementTruth:
    """Index event spans once; measurement IDs are the original per-event row positions."""

    fields = ("event_nr", "volume_id", "layer_id", "surface_id", "rec_loc0", "rec_loc1",
              "true_loc0", "true_loc1", "residual_loc0", "residual_loc1")

    def __init__(self, path):
        import uproot
        self.path = Path(path)
        # Synchronous local I/O avoids stalled async basket notifications on this runtime.
        self.root = uproot.open(self.path, handler=uproot.source.file.MemmapSource,
                                decompression_executor=uproot.TrivialExecutor(),
                                interpretation_executor=uproot.TrivialExecutor())
        try:
            self.tree = self.root["measurements"]
            missing = set(self.fields) - set(self.tree.keys())
            if missing:
                raise ValueError(f"{path}: missing measurement truth branches {sorted(missing)}")
            events = self.tree["event_nr"].array(library="np")
            changes = np.r_[0, np.flatnonzero(events[1:] != events[:-1]) + 1, len(events)]
            self.spans = {}
            for start, stop in zip(changes[:-1], changes[1:]):
                if start < stop:
                    self.spans.setdefault(int(events[start]), []).append((int(start), int(stop)))
        except Exception:
            self.root.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.root.close()

    def read(self, event):
        if event not in self.spans:
            raise ValueError(f"{self.path}: missing ROOT event {event}")
        spans = self.spans[event]
        arrays = self.tree.arrays(list(self.fields), entry_start=spans[0][0],
                                 entry_stop=spans[-1][1], library="np")
        selected = arrays["event_nr"] == event
        frame = pd.DataFrame({name: arrays[name][selected] for name in self.fields})
        frame["measurement_id"] = np.arange(len(frame), dtype=np.uint64)
        frame["local_event"] = event
        return frame.drop(columns="event_nr")


def pixel_segmentation(path):
    with Path(path).open() as stream:
        config = json.load(stream)
    result = {}
    for entry in config["entries"]:
        volume = int(entry["volume"])
        if volume not in DETECTOR_VOLUMES["Pixels"]:
            continue
        geometric = entry["value"]["geometric"]
        axes = geometric["segmentation"]["binningdata"]
        if (geometric["digital"] or len(axes) != 2 or geometric["indices"] != [0, 1]
                or [axis["value"] for axis in axes] != ["binX", "binY"]
                or any(axis["type"] != "equidistant" for axis in axes)):
            raise ValueError("Residual study requires analog, equidistant X/Y pixel segmentation")
        result[volume] = tuple(dict(minimum=float(axis["min"]), bins=int(axis["bins"]),
                                   pitch=(float(axis["max"]) - float(axis["min"])) / int(axis["bins"]))
                               for axis in axes)
        if any(axis["pitch"] <= 0 or axis["bins"] < 1 for axis in result[volume]):
            raise ValueError("Invalid pixel segmentation range")
    if set(result) != set(DETECTOR_VOLUMES["Pixels"]):
        raise ValueError("Digitization config must define all three pixel volumes")
    return result


def physical_pixel_centroids(cells, clusters, weights, segmentation):
    centers = weighted_channel_centroids(cells, weights)
    result = np.empty_like(centers)
    for volume in clusters.volume_id.unique():
        if volume not in segmentation:
            raise ValueError(f"Missing pixel segmentation for volume {volume}")
        rows = clusters.volume_id.to_numpy() == volume
        selected = cells.loc[cells.volume_id == volume]
        for axis, spec in enumerate(segmentation[volume]):
            addresses = selected[f"channel{axis}"].to_numpy()
            if np.any((addresses < 0) | (addresses >= spec["bins"])):
                raise ValueError("Channel outside configured pixel segmentation")
            result[rows, axis] = spec["minimum"] + (centers[rows, axis] + .5) * spec["pitch"]
    return result


def _residual_statistics(values):
    values = np.asarray(values)
    return dict(bias=float(values.mean()), median=float(np.median(values)),
                rms=float(np.sqrt(np.mean(values ** 2))), std=float(values.std()),
                central68_half_width=float(np.diff(np.quantile(values, [.16, .84]))[0] / 2),
                abs_q95=float(np.quantile(np.abs(values), .95)),
                abs_q99=float(np.quantile(np.abs(values), .99)), maximum_abs=float(np.abs(values).max()))


@dataclass
class PixelResidualStudy:
    coverage: pd.DataFrame
    summary: pd.DataFrame
    histograms: pd.DataFrame
    survival: pd.DataFrame
    codes: pd.DataFrame
    largest_residuals: pd.DataFrame
    fit_statistics: pd.DataFrame | None = None


def run_pixel_residual_study(root, empirical, linear, atlas_reference, digi_config,
                              acts_roots, events=range(25, 50), histogram_edges=None,
                              alignment_tolerance_mm=1e-4, progress=print,
                              analysis_loader=None, truth_loader=None,
                              linear_variants=None, fit_window_um=30.):
    """Signed local residuals for complete, one-SimHit pixel clusters; no decoding."""
    events = _study_events(events, "Residual events")
    if set(events) & set(empirical.events):
        raise ValueError("Residual validation events must be held out from calibration")
    if not np.array_equal(empirical.atlas_histogram, atlas_reference["histograms"]["calibration"]):
        raise ValueError("ATLAS calibration reference changed after fitting")
    expected_linear = fit_linear_calibration(empirical, linear.activation_min)
    if not np.array_equal(linear.boundaries, expected_linear.boundaries):
        raise ValueError("Linear boundaries must use the frozen calibration overflow edge")
    edges = np.asarray(np.linspace(-100, 100, 201) if histogram_edges is None else histogram_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Residual histogram edges must be finite and strictly increasing")
    if not np.isfinite(alignment_tolerance_mm) or alignment_tolerance_mm <= 0:
        raise ValueError("Alignment tolerance must be finite and positive")
    if not np.isfinite(fit_window_um) or fit_window_um <= 0:
        raise ValueError("Fit window must be finite and positive")
    models = {"linear": linear} if linear_variants is None else dict(linear_variants)
    if not models or set(models) & {"native", "empirical"}:
        raise ValueError("Choose nonempty, distinctly named linear variants")
    segmentation = pixel_segmentation(digi_config)
    analysis_loader = analysis_loader or load_pixel_analysis
    coverage, event_stats, parts, histograms, code_counts, examples = [], [], {}, {}, {}, []
    fit_rows = []
    with ExitStack() as stack:
        readers = ({sample: stack.enter_context(RootMeasurementTruth(path))
                    for sample, path in acts_roots.items()} if truth_loader is None else {})
        for sample in PILEUP_COLORS:
            if truth_loader is None and sample not in readers:
                raise ValueError(f"Missing ACTS measurement ROOT path for {sample}")
            for event in events:
                analysis = analysis_loader(root, sample, event)
                cells = analysis.cells
                if not (cells.detector_type == "Pixels").all():
                    raise ValueError("Residual study expects pixel-only analysis")
                clusters = cells[MEASUREMENT_KEY].drop_duplicates().merge(
                    analysis.clusters, on=MEASUREMENT_KEY, validate="one_to_one", sort=False)
                if clusters.run.nunique() != 1 or clusters.measurement_id.duplicated().any():
                    raise ValueError("A single ROOT truth path supports only one run per sample")
                truth = truth_loader(sample, event) if truth_loader else readers[sample].read(event)
                truth = truth.rename(columns={name: f"root_{name}" for name in truth.columns
                                               if name not in ("local_event", "measurement_id")})
                aligned = clusters.merge(truth, on=["local_event", "measurement_id"], how="left",
                                          validate="one_to_one", indicator=True, sort=False)
                if not (aligned._merge == "both").all():
                    raise ValueError(f"{sample} event {event}: missing ROOT measurement alignment")
                geometry = aligned.hit_geometry_id.to_numpy(dtype=np.uint64)
                for name, shift, mask in (("volume_id", 56, 0xff), ("layer_id", 36, 0xfff),
                                           ("surface_id", 8, 0xfffff)):
                    if not np.array_equal((geometry >> shift) & mask, aligned[f"root_{name}"]):
                        raise ValueError("ROOT/Parquet measurement geometry differs")
                native = physical_pixel_centroids(cells, clusters, cells.value, segmentation)
                root_native = aligned[["root_rec_loc0", "root_rec_loc1"]].to_numpy()
                stored_native = aligned[["local0", "local1"]].to_numpy()
                if (not np.isfinite(root_native).all() or not np.isfinite(stored_native).all()
                        or not np.allclose(native, root_native, rtol=0, atol=alignment_tolerance_mm)
                        or not np.allclose(stored_native, root_native, rtol=0, atol=alignment_tolerance_mm)):
                    raise ValueError("Native centroid differs from ROOT/Parquet local position; check segmentation/alignment")
                true_position = aligned[["root_true_loc0", "root_true_loc1"]].to_numpy()
                root_residual = aligned[["root_residual_loc0", "root_residual_loc1"]].to_numpy()
                eligible = (clusters.truth_complete & (clusters.simhit_count == 1)).to_numpy()
                if np.any(eligible & (clusters.known_particles.to_numpy() != 1)):
                    raise ValueError("Complete one-SimHit clusters must have exactly one particle")
                finite = np.isfinite(true_position).all(axis=1) & np.isfinite(root_residual).all(axis=1)
                use = eligible & finite
                if not np.allclose((root_native - true_position)[use], root_residual[use],
                                   rtol=0, atol=alignment_tolerance_mm):
                    raise ValueError("Stored ROOT residual disagrees with reconstructed-minus-truth convention")
                coverage.append(dict(sample=sample, event=event, pixel_clusters=len(clusters),
                                     incomplete_truth=int((~clusters.truth_complete).sum()),
                                     complete_multiple_simhits=int((clusters.truth_complete & (clusters.simhit_count > 1)).sum()),
                                     complete_zero_simhits=int((clusters.truth_complete & (clusters.simhit_count == 0)).sum()),
                                     eligible_single_simhit=int(eligible.sum()), nonfinite_truth=int((eligible & ~finite).sum()),
                                     residual_clusters=int(use.sum()),
                                     maximum_native_alignment_um=float(np.abs(native - root_native).max() * 1000)))
                weights = {"empirical": empirical_codes(cells.value, empirical.boundaries),
                           **{name: linear_codes(cells.value, model) for name, model in models.items()}}
                if linear_variants is None and not np.array_equal(weights["empirical"] == 15, weights["linear"] == 15):
                    raise ValueError("Linear and empirical maps disagree on overflow membership")
                parent = pd.factorize(pd.MultiIndex.from_frame(cells[MEASUREMENT_KEY]), sort=False)[0]
                for variant, codes in weights.items():
                    key = (sample, variant)
                    totals = code_counts.setdefault(key, dict(histogram=np.zeros(16, dtype=np.int64),
                                                               clusters=0, overflow_clusters=0))
                    totals["histogram"] += np.bincount(codes, minlength=16)
                    totals["clusters"] += len(clusters)
                    totals["overflow_clusters"] += int((np.bincount(parent, weights=codes == 15,
                                                                    minlength=len(clusters)) > 0).sum())
                if use.any():
                    used_clusters = clusters.loc[use].reset_index(drop=True)
                    positions = {"native": native, **{variant: physical_pixel_centroids(cells, clusters, codes, segmentation)
                                                        for variant, codes in weights.items()}}
                    for variant, position in positions.items():
                        residuals = (position[use] - true_position[use]) * 1000  # mm -> micrometres
                        for axis in range(2):
                            values = residuals[:, axis]
                            for grouping in ("all", "volume_id"):
                                groups = [("all", np.arange(len(values)))] if grouping == "all" else used_clusters.groupby(grouping).indices.items()
                                for group, indices in groups:
                                    selected = values[indices]
                                    core = selected[(selected >= -fit_window_um) & (selected < fit_window_um)]
                                    fit_rows.append(dict(sample=sample, variant=variant, axis=f"local{axis}",
                                                         event=event, grouping=grouping, group=group,
                                                         window_um=fit_window_um, n_full=len(selected), n_fit=len(core),
                                                         sum_x=float(core.sum()), sum_x2=float(np.square(core).sum())))
                            key = (sample, variant, f"local{axis}")
                            state = histograms.setdefault(key, dict(counts=np.zeros(len(edges) - 1, dtype=np.int64),
                                                                    clusters=0, below=0, above=0))
                            state["counts"] += np.histogram(values, bins=edges)[0]
                            state["clusters"] += len(values)
                            state["below"] += int((values < edges[0]).sum())
                            state["above"] += int((values > edges[-1]).sum())
                            for grouping in ("all", "volume_id", "n_cells"):
                                groups = [("all", np.arange(len(values)))] if grouping == "all" else used_clusters.groupby(grouping).indices.items()
                                for group, indices in groups:
                                    parts.setdefault((*key, grouping, group), []).append(values[indices])
                            event_stats.append(dict(sample=sample, variant=variant, axis=f"local{axis}", event=event,
                                                    grouping="all", group="all", clusters=len(values), **_residual_statistics(values)))
                            for index in np.argsort(-np.abs(values), kind="stable")[:2]:
                                row = used_clusters.iloc[index]
                                examples.append(dict(sample=sample, variant=variant, axis=f"local{axis}",
                                                     **{name: row[name] for name in MEASUREMENT_KEY}, n_cells=int(row.n_cells),
                                                     residual_um=float(values[index]), absolute_um=float(abs(values[index]))))
                if progress is not None:
                    progress(f"Truth residuals: {sample} event {event}, {use.sum():,}/{len(clusters):,} single-SimHit clusters")
                del analysis, cells, clusters, truth, aligned
    if not parts:
        raise ValueError("No finite complete single-SimHit clusters available for residuals")
    pooled, survival, histogram_rows, code_rows = [], [], [], []
    for (sample, variant, axis, grouping, group), arrays in parts.items():
        values = np.concatenate(arrays)
        pooled.append(dict(sample=sample, variant=variant, axis=axis, event="pooled", grouping=grouping,
                           group=group, clusters=len(values), **_residual_statistics(values)))
        if grouping == "all":
            support, counts = np.unique(np.abs(values), return_counts=True)
            indices = np.unique(np.linspace(0, len(support) - 1, min(len(support), 2000), dtype=int))
            fractions = counts[::-1].cumsum()[::-1] / len(values)
            survival.extend(dict(sample=sample, variant=variant, axis=axis, absolute_um=support[i],
                                 fraction=fractions[i]) for i in indices)
    for (sample, variant, axis), state in histograms.items():
        histogram_rows.extend(dict(sample=sample, variant=variant, axis=axis, left=edges[i], right=edges[i + 1],
                                   count=int(count), density=count / (state["clusters"] * (edges[i + 1] - edges[i])),
                                   total_clusters=state["clusters"], below_range=state["below"], above_range=state["above"])
                              for i, count in enumerate(state["counts"]))
    for (sample, variant), state in code_counts.items():
        histogram = state["histogram"]
        for cohort, atlas_hist in atlas_reference["histograms"].items():
            for code in range(1, 16):
                fraction = histogram[code] / histogram.sum()
                target = atlas_hist[code] / atlas_hist.sum()
                code_rows.append(dict(sample=sample, variant=variant, atlas_cohort=cohort, code=code,
                                      cells=int(histogram.sum()), clusters=state["clusters"], fraction=fraction,
                                      atlas_fraction=target, difference=fraction - target,
                                      code15_cluster_fraction=state["overflow_clusters"] / state["clusters"]))
    return PixelResidualStudy(pd.DataFrame(coverage),
                               pd.concat([pd.DataFrame(event_stats), pd.DataFrame(pooled)], ignore_index=True),
                               pd.DataFrame(histogram_rows), pd.DataFrame(survival), pd.DataFrame(code_rows),
                               pd.DataFrame(examples).sort_values("absolute_um", ascending=False).reset_index(drop=True),
                               pd.DataFrame(fit_rows))


def plot_linear_comparison(empirical, linear, analyses, atlas_reference):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    x = np.linspace(linear.activation_min, linear.overflow_edge * 1.1, 1000)
    axes[0].step(x, empirical_codes(x, empirical.boundaries), where="post", color="black", label="Empirical")
    axes[0].step(x, linear_codes(x, linear), where="post", color="gray", label="Linear bins")
    axes[0].axvline(linear.overflow_edge, color=PUBLIC_COLOR, linestyle=":", label="Shared overflow edge")
    axes[0].set(xlabel="Activation (mm proxy; linear axis)", ylabel="Integer code", title="Frozen 1-14 + overflow 15")
    for ax, variant in zip(axes[1:], ("empirical", "linear")):
        for sample, analysis in analyses.items():
            values = analysis.cells.loc[analysis.cells.detector_type == "Pixels", "value"]
            codes = empirical_codes(values, empirical.boundaries) if variant == "empirical" else linear_codes(values, linear)
            histogram = np.bincount(codes, minlength=16)
            ax.plot(np.arange(1, 16), histogram[1:] / histogram.sum(), "o-", color=PILEUP_COLORS[sample],
                    label=f"{sample} event {analysis.event}")
        for cohort, style in (("calibration", ":"), ("held_out", "--")):
            histogram = atlas_reference["histograms"][cohort]
            ax.plot(np.arange(1, 16), histogram[1:] / histogram.sum(), style, color=PUBLIC_COLOR, label=f"ATLAS {cohort}")
        ax.set(yscale="log", xlabel="Positive pixel code", ylabel="Cell fraction", title=variant.capitalize())
    for ax in axes:
        ax.legend(fontsize=8)
    plt.show()


def plot_pixel_residuals(study):
    variants = list(study.histograms.variant.unique())
    for axis in ("local0", "local1"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for column, sample in enumerate(PILEUP_COLORS):
            for variant in variants:
                name = "linear_ATLAS" if variant == "linear" else variant
                color = METHOD_COLORS.get(name, "#555555")
                label = METHOD_LABELS.get(name, name)
                hist = study.histograms.query("sample == @sample and variant == @variant and axis == @axis")
                survival = study.survival.query("sample == @sample and variant == @variant and axis == @axis")
                if hist.empty:
                    continue
                axes[0, column].stairs(hist.density.to_numpy(), np.r_[hist.left.to_numpy(), hist.right.iloc[-1]],
                                       color=color, linewidth=2, label=label)
                axes[1, column].step(survival.absolute_um, survival.fraction, where="post", color=color,
                                     linewidth=1.8, marker=METHOD_MARKERS.get(name, "o"), markersize=4,
                                     markevery=max(1, len(survival) // 12), label=label)
            axes[0, column].set(xlabel=f"{axis}: reconstructed - ROOT truth (micrometres)", ylabel="Density",
                                title=f"{sample}: signed residuals")
            axes[1, column].set(yscale="log", xlabel=f"Absolute {axis} residual (micrometres)",
                                ylabel="Fraction >= residual", title=f"{sample}: full-range tails")
        for ax in axes.flat:
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        fig.suptitle("Complete single-SimHit pixel clusters; ROOT-reference residuals")
        plt.show()
    coded = list(study.codes.variant.unique())
    fig, axes = plt.subplots(1, len(coded), figsize=(5 * len(coded), 4), squeeze=False, constrained_layout=True)
    for ax, variant in zip(axes.flat, coded):
        for sample, color in PILEUP_COLORS.items():
            frame = study.codes.query("sample == @sample and variant == @variant and atlas_cohort == 'held_out'")
            ax.plot(frame.code, frame.fraction, "o-", color=color, label=sample)
        ax.plot(frame.code, frame.atlas_fraction, "--", color=PUBLIC_COLOR, label="ATLAS held out")
        ax.set(yscale="log", xlabel="Integer code", ylabel="Fraction of all pixel cells",
               title=METHOD_LABELS.get(variant, variant))
        ax.legend()
    plt.show()


@dataclass
class LinearThresholdScan:
    events: tuple
    variants: dict
    calibration: pd.DataFrame


def fit_linear_threshold_scan(root, empirical, percentiles=(99.9, 99.5, 99.),
                               activation_min=0.01, progress=print):
    """Fit lower overflow edges on the same pu200 calibration population only."""
    percentiles = tuple(percentiles)
    if (not percentiles or len(set(percentiles)) != len(percentiles)
            or not np.isfinite(percentiles).all() or any(not 0 < p < 100 for p in percentiles)):
        raise ValueError("Choose distinct finite percentiles inside (0,100)")
    events = _study_events(empirical.events, "Calibration events")
    variants = {"linear_ATLAS": fit_linear_calibration(empirical, activation_min)}
    support, counts = np.array([], dtype=float), np.array([], dtype=np.int64)
    for index, event in enumerate(events, 1):
        values = pixel_calibration_values(root, event)
        support, counts = _merge_activation_counts(support, counts, values)
        if progress is not None:
            progress(f"Linear scan calibration {index}/{len(events)}: pu200 event {event}")
        del values
    if support[0] < activation_min:
        raise ValueError("Calibration activation below the declared linear range minimum")
    edges = _counted_quantiles(support, counts, np.asarray(percentiles) / 100)
    for percentile, edge in zip(percentiles, edges):
        if not activation_min < edge <= empirical.boundaries[-1]:
            raise ValueError("Scan overflow edges must exceed the minimum and not exceed the ATLAS baseline edge")
        variants[f"linear_P{percentile:g}"] = LinearCalibration(
            activation_min, float(edge), 1 - percentile / 100,
            np.linspace(activation_min, edge, 15)[1:])
    rows = []
    for name, model in variants.items():
        histogram = np.bincount(linear_codes(support, model), weights=counts, minlength=16)
        rows.append(dict(variant=name, calibration_events=len(events), cells=int(counts.sum()),
                         target_percentile=100 * (1 - model.target_overflow_fraction),
                         overflow_edge=model.overflow_edge, bin_width=(model.overflow_edge - activation_min) / 14,
                         target_overflow_fraction=model.target_overflow_fraction,
                         actual_calibration_overflow_fraction=histogram[15] / histogram.sum(),
                         calibration_code1_or2_fraction=histogram[1:3].sum() / histogram.sum()))
    return LinearThresholdScan(events, variants, pd.DataFrame(rows))


@dataclass
class LinearScanStudy:
    events: pd.DataFrame
    summary: pd.DataFrame
    overflow: pd.DataFrame
    codes: pd.DataFrame
    survival: pd.DataFrame
    signed_histograms: pd.DataFrame
    largest_shifts: pd.DataFrame


def _scan_shift_statistics(distances):
    return {**_shift_statistics(distances),
            **{f"fraction_gt_{str(t).replace('.', '_')}": float(np.mean(distances > t))
               for t in (0.1, 0.5, 1.0)}}


def run_linear_threshold_scan(root, empirical, scan, events=range(25, 50),
                               progress=print, loader=None, signed_edges=None):
    """Frozen quantizers; all pixel clusters, including merged and incomplete truth."""
    events = _study_events(events, "Scan held-out events")
    if tuple(empirical.events) != scan.events or set(events) & set(scan.events):
        raise ValueError("Scan must use the same calibration cohort and disjoint held-out events")
    baseline = scan.variants.get("linear_ATLAS")
    if baseline is None or not np.array_equal(baseline.boundaries,
                                             fit_linear_calibration(empirical, baseline.activation_min).boundaries):
        raise ValueError("Scan baseline differs from the frozen empirical calibration")
    edges = np.asarray(np.linspace(-1, 1, 201) if signed_edges is None else signed_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Signed shift histogram edges must be finite and strictly increasing")
    loader = loader or load_pixel_analysis
    rows, summaries, parts, code_counts, signed, largest = [], [], {}, {}, {}, []
    signed_parts = {}
    for sample in PILEUP_COLORS:
        for event in events:
            analysis = loader(root, sample, event)
            cells = analysis.cells
            if not (cells.detector_type == "Pixels").all():
                raise ValueError("Linear scan expects pixel-only event analysis")
            clusters = cells[MEASUREMENT_KEY].drop_duplicates().merge(
                analysis.clusters, on=MEASUREMENT_KEY, validate="one_to_one", sort=False)
            native = weighted_channel_centroids(cells, cells.value)
            parent = pd.factorize(pd.MultiIndex.from_frame(cells[MEASUREMENT_KEY]), sort=False)[0]
            for name in ("empirical", *scan.variants):
                codes = (empirical_codes(cells.value, empirical.boundaries) if name == "empirical"
                         else linear_codes(cells.value, scan.variants[name]))
                delta = weighted_channel_centroids(cells, codes) - native
                distance = np.linalg.norm(delta, axis=1)
                contains_overflow = np.bincount(parent, weights=codes == 15, minlength=len(clusters)) > 0
                key = (sample, name)
                totals = code_counts.setdefault(key, dict(histogram=np.zeros(16, dtype=np.int64),
                                                           clusters=0, overflow_clusters=0, events=0))
                totals["histogram"] += np.bincount(codes, minlength=16)
                totals["clusters"] += len(clusters)
                totals["overflow_clusters"] += int(contains_overflow.sum())
                totals["events"] += 1
                rows.append(dict(sample=sample, variant=name, event=event, clusters=len(clusters), cells=len(cells),
                                 code15_cell_fraction=float(np.mean(codes == 15)),
                                 code15_cluster_fraction=float(contains_overflow.mean()),
                                 code1_or2_fraction=float(np.mean(codes <= 2)), **_scan_shift_statistics(distance)))
                for grouping in ("all", "n_cells", "contributor_group", "contains_overflow"):
                    if grouping == "all":
                        groups = [("all", np.arange(len(clusters)))]
                    elif grouping == "contains_overflow":
                        groups = [(flag, np.flatnonzero(contains_overflow == flag)) for flag in (False, True)]
                    else:
                        groups = clusters.groupby(grouping).indices.items()
                    for group, indices in groups:
                        if len(indices):
                            parts.setdefault((*key, grouping, group), []).append(distance[indices])
                signed_parts.setdefault(key, []).append(delta)
                for axis in range(2):
                    values = delta[:, axis]
                    state = signed.setdefault((*key, axis), dict(counts=np.zeros(len(edges) - 1, dtype=np.int64),
                                                                 total=0, below=0, above=0))
                    state["counts"] += np.histogram(values, bins=edges)[0]
                    state["total"] += len(values)
                    state["below"] += int((values < edges[0]).sum())
                    state["above"] += int((values > edges[-1]).sum())
                for index in np.argsort(-distance, kind="stable")[:3]:
                    row = clusters.iloc[index]
                    largest.append(dict(sample=sample, variant=name,
                                        **{column: row[column] for column in MEASUREMENT_KEY},
                                        n_cells=int(row.n_cells), contributor_group=row.contributor_group,
                                        contains_overflow=bool(contains_overflow[index]),
                                        delta0=float(delta[index, 0]), delta1=float(delta[index, 1]),
                                        displacement=float(distance[index])))
            if progress is not None:
                progress(f"Linear threshold scan: {sample} event {event}, {len(clusters):,} pixel clusters")
            del analysis, cells, clusters
    survival, overflow, frequencies, histogram_rows = [], [], [], []
    for (sample, variant, grouping, group), arrays in parts.items():
        values = np.concatenate(arrays)
        summaries.append(dict(sample=sample, variant=variant, grouping=grouping, group=group,
                              clusters=len(values), **_scan_shift_statistics(values)))
        if grouping == "all":
            support, counts = np.unique(values, return_counts=True)
            indices = np.unique(np.linspace(0, len(support) - 1, min(2000, len(support)), dtype=int))
            fractions = counts[::-1].cumsum()[::-1] / len(values)
            survival.extend(dict(sample=sample, variant=variant, displacement=support[i], fraction=fractions[i])
                            for i in indices)
    summary = pd.DataFrame(summaries)
    for (sample, variant), arrays in signed_parts.items():
        values = np.concatenate(arrays)
        mask = (summary["sample"] == sample) & (summary.variant == variant) & (summary.grouping == "all")
        for axis in range(2):
            summary.loc[mask, f"mean_delta{axis}"] = values[:, axis].mean()
            summary.loc[mask, f"signed_q01_delta{axis}"] = np.quantile(values[:, axis], .01)
            summary.loc[mask, f"signed_q99_delta{axis}"] = np.quantile(values[:, axis], .99)
    for (sample, variant), totals in code_counts.items():
        hist = totals["histogram"]
        overflow.append(dict(sample=sample, variant=variant, events=totals["events"], cells=int(hist.sum()),
                             clusters=totals["clusters"], code1_or2_fraction=hist[1:3].sum() / hist.sum(),
                             code15_cell_fraction=hist[15] / hist.sum(),
                             code15_cluster_fraction=totals["overflow_clusters"] / totals["clusters"]))
        frequencies.extend(dict(sample=sample, variant=variant, code=code, fraction=hist[code] / hist.sum())
                           for code in range(1, 16))
    for (sample, variant, axis), state in signed.items():
        histogram_rows.extend(dict(sample=sample, variant=variant, axis=f"local{axis}",
                                   left=edges[i], right=edges[i + 1], count=int(count),
                                   density=count / (state["total"] * (edges[i + 1] - edges[i])),
                                   total_clusters=state["total"], below_range=state["below"], above_range=state["above"])
                              for i, count in enumerate(state["counts"]))
    return LinearScanStudy(pd.DataFrame(rows), summary, pd.DataFrame(overflow), pd.DataFrame(frequencies),
                            pd.DataFrame(survival), pd.DataFrame(histogram_rows),
                            pd.DataFrame(largest).sort_values("displacement", ascending=False).reset_index(drop=True))


def plot_linear_threshold_scan(empirical, scan, study, atlas_reference):
    names = ["empirical", *scan.variants]
    styles = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.linspace(scan.variants["linear_ATLAS"].activation_min, empirical.boundaries[-1] * 1.1, 2000)
    for index, name in enumerate(names):
        codes = empirical_codes(x, empirical.boundaries) if name == "empirical" else linear_codes(x, scan.variants[name])
        ax.step(x, codes, where="post", color=METHOD_COLORS.get(name), linewidth=2,
                linestyle=styles[index % len(styles)], label=METHOD_LABELS.get(name, name))
    ax.set(xlabel="Activation (mm proxy)", ylabel="Integer code", title="Frozen mappings; code 15 is overflow")
    ax.legend()
    plt.show()
    for sample in PILEUP_COLORS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for index, name in enumerate(names):
            style = styles[index % len(styles)]
            color = METHOD_COLORS.get(name, "#555555")
            label = METHOD_LABELS.get(name, name)
            marker = METHOD_MARKERS.get(name, "o")
            frame = study.codes.query("sample == @sample and variant == @name")
            axes[0, 0].plot(frame.code, frame.fraction, linestyle=style, color=color,
                             marker=marker, markersize=4, linewidth=1.8, label=label)
            frame = study.survival.query("sample == @sample and variant == @name")
            axes[0, 1].step(frame.displacement, frame.fraction, where="post", linestyle=style, color=color,
                             linewidth=1.8, marker=marker, markevery=max(1, len(frame) // 12), markersize=4, label=label)
            for axis, ax in enumerate(axes[1]):
                axis_name = f"local{axis}"
                frame = study.signed_histograms.query("sample == @sample and variant == @name and axis == @axis_name")
                ax.stairs(frame.density.to_numpy(), np.r_[frame.left.to_numpy(), frame.right.iloc[-1]],
                          linestyle=style, color=color, linewidth=2, label=label)
        for cohort, style in (("calibration", ":"), ("held_out", "--")):
            hist = atlas_reference["histograms"][cohort]
            axes[0, 0].plot(np.arange(1, 16), hist[1:] / hist.sum(), style, color="#333333",
                             linewidth=2, label=f"ATLAS {cohort}")
        axes[0, 0].set(yscale="log", xlabel="Code", ylabel="Cell fraction", title="Interior-code occupancy vs ATLAS")
        axes[0, 1].set(yscale="log", xlabel="Distance from native (channel units)",
                       ylabel="Fraction >= displacement", title="Full-range displacement survival")
        for axis, ax in enumerate(axes[1]):
            ax.set(yscale="log", xlabel=f"Signed local{axis} change from native (channels)",
                   ylabel="Density per channel", title="Central display; clipped counts reported separately")
        for ax in axes.flat:
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        fig.suptitle(f"{sample}: all held-out pixel clusters; NOT truth residuals")
        plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4), constrained_layout=True)
    x = np.arange(len(names))
    for sample, color in PILEUP_COLORS.items():
        frame = study.summary.query("sample == @sample and grouping == 'all'").set_index("variant").loc[names]
        axes[0].plot(x, frame.q99, "o-", color=color, label=sample)
        axes[1].plot(x, frame.fraction_gt_0_5, "o-", color=color, label=sample)
        frame = study.overflow.query("sample == @sample").set_index("variant").loc[names]
        axes[2].plot(x, frame.code15_cell_fraction, "o-", color=color, label=f"{sample} cells")
        axes[2].plot(x, frame.code15_cluster_fraction, ":", color=color, label=f"{sample} clusters")
    axes[0].set(ylabel="99th percentile displacement (channels)", title="Position-change tail")
    axes[1].set(ylabel="Fraction moving > 0.5 channel", title="Same clusters, frozen mappings")
    axes[2].set(yscale="log", ylabel="Code-15 fraction", title="Different cell and cluster denominators")
    for ax in axes:
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.legend(fontsize=8)
    plt.show()


def plot_linear_scan_examples(root, empirical, scan, study, variants=None, per_sample=1):
    variants = list(scan.variants)[1:] if variants is None else list(variants)
    records = []
    for variant in variants:
        if variant != "empirical" and variant not in scan.variants:
            raise ValueError(f"Unknown scan variant {variant}")
        for sample in PILEUP_COLORS:
            examples = study.largest_shifts.query("sample == @sample and variant == @variant").head(per_sample)
            for example in examples.to_dict("records"):
                analysis = load_pixel_analysis(root, sample, int(example["local_event"]))
                mask = np.ones(len(analysis.cells), dtype=bool)
                cluster_mask = np.ones(len(analysis.clusters), dtype=bool)
                for key in MEASUREMENT_KEY:
                    mask &= analysis.cells[key].to_numpy() == example[key]
                    cluster_mask &= analysis.clusters[key].to_numpy() == example[key]
                cells = analysis.cells.loc[mask].reset_index(drop=True)
                clusters = analysis.clusters.loc[cluster_mask].reset_index(drop=True)
                if len(clusters) != 1 or len(cells) != example["n_cells"]:
                    raise ValueError("Scan example membership changed")
                codes = (empirical_codes(cells.value, empirical.boundaries) if variant == "empirical"
                         else linear_codes(cells.value, scan.variants[variant]))
                distance = np.linalg.norm(weighted_channel_centroids(cells, codes)
                                          - weighted_channel_centroids(cells, cells.value), axis=1)
                if not np.isclose(distance[0], example["displacement"], atol=1e-12, rtol=1e-12):
                    raise ValueError("Scan example displacement changed")
                selected = PixelQuantizationStudy(pd.DataFrame(), empirical.boundaries, {sample: cells},
                    {sample: clusters}, {(sample, variant): codes},
                    {(sample, variant): {"native_to_transformed": distance}}, pd.DataFrame(), pd.DataFrame())
                frame = plot_quantization_examples(selected, variant=variant, per_sample=1)
                frame["variant"] = variant
                records.append(frame)
                del analysis, selected
    return pd.concat(records, ignore_index=True)


def gaussian_core_fit(statistics, window_um, minimum_entries=100):
    """Unbinned Gaussian MLE conditioned on the common [-window, window) core."""
    from scipy.optimize import minimize
    from scipy.special import ndtr
    n, total, squared = np.asarray(statistics, dtype=float)
    if not np.isfinite(window_um) or window_um <= 0:
        raise ValueError("Gaussian fit window must be finite and positive")
    invalid = dict(fit_status="insufficient_entries", mean_um=np.nan, sigma_um=np.nan)
    if n < minimum_entries or not np.isfinite([n, total, squared]).all():
        return invalid
    mean, second = total / n, squared / n
    variance = second - mean * mean
    if variance <= 1e-12:
        return {**invalid, "fit_status": "degenerate_distribution"}
    def objective(parameters):
        mu, log_sigma = parameters
        sigma = np.exp(log_sigma)
        probability = ndtr((window_um - mu) / sigma) - ndtr((-window_um - mu) / sigma)
        if probability <= 0 or not np.isfinite(probability):
            return 1e100
        return log_sigma + .5 * (second - 2 * mu * mean + mu * mu) / sigma ** 2 + np.log(probability)
    bounds = [(-window_um, window_um), (np.log(window_um * 1e-4), np.log(window_um * 10))]
    result = minimize(objective, [mean, np.log(np.sqrt(variance))], method="L-BFGS-B", bounds=bounds)
    mu, log_sigma = result.x
    boundary = abs(mu) >= window_um * .999 or log_sigma <= bounds[1][0] + 1e-5 or log_sigma >= bounds[1][1] - 1e-5
    status = "ok" if result.success and not boundary else "fit_boundary" if boundary else "optimizer_failed"
    return dict(fit_status=status, mean_um=float(mu), sigma_um=float(np.exp(log_sigma)))


def fit_residual_resolutions(study, bootstrap_resamples=200, seed=7, minimum_entries=100,
                              progress=print):
    """Fit residual cores; whole-event paired bootstrap compares sigma to native."""
    from scipy.special import ndtr
    stats = study.fit_statistics
    if stats is None or stats.empty:
        raise ValueError("Rerun residual collection to populate Gaussian fit statistics")
    if bootstrap_resamples < 0:
        raise ValueError("Bootstrap resamples cannot be negative")
    keys = ["sample", "axis", "grouping", "group"]
    rows = []
    rng = np.random.default_rng(seed)
    for group_key, group_frame in stats.groupby(keys, sort=False):
        sample, axis, grouping, group = group_key
        if group_frame.window_um.nunique() != 1:
            raise ValueError("All methods must use the same fit window")
        window = float(group_frame.window_um.iloc[0])
        native_events = sorted(group_frame.query("variant == 'native'").event.unique())
        draws = (rng.integers(0, len(native_events), size=(bootstrap_resamples, len(native_events)))
                 if grouping == "all" and bootstrap_resamples and len(native_events) >= 2 else None)
        fitted, boot = {}, {}
        for variant, frame in group_frame.groupby("variant", sort=False):
            if frame.event.duplicated().any() or sorted(frame.event.unique()) != native_events:
                raise ValueError("Residual fit methods must share the same events")
            matrix = frame.set_index("event").loc[native_events, ["n_fit", "sum_x", "sum_x2"]].to_numpy()
            result = gaussian_core_fit(matrix.sum(axis=0), window, minimum_entries)
            full = int(frame.n_full.sum())
            fitted[variant] = dict(sample=sample, axis=axis, grouping=grouping, group=group,
                                    variant=variant, events=len(frame), clusters=full,
                                    core_clusters=int(matrix[:, 0].sum()), core_fraction=matrix[:, 0].sum() / full,
                                    fit_window_um=window, reference="ROOT SimHit local (provisional)", **result)
            if draws is not None and result["fit_status"] == "ok":
                results = [gaussian_core_fit(matrix[draw].sum(axis=0), window, minimum_entries) for draw in draws]
                boot[variant] = np.array([r["sigma_um"] if r["fit_status"] == "ok" else np.nan for r in results])
        native = fitted["native"]
        for variant, row in fitted.items():
            if row["fit_status"] == native["fit_status"] == "ok":
                row["sigma_ratio_to_native"] = row["sigma_um"] / native["sigma_um"]
                row["sigma_change_percent"] = 100 * (row["sigma_ratio_to_native"] - 1)
            if variant in boot and "native" in boot:
                valid = np.isfinite(boot[variant]) & np.isfinite(boot["native"])
                row["bootstrap_valid"] = int(valid.sum())
                if valid.sum() >= max(2, .8 * bootstrap_resamples):
                    row["sigma_lower95"], row["sigma_upper95"] = np.quantile(boot[variant][valid], [.025, .975])
                    ratio = boot[variant][valid] / boot["native"][valid]
                    row["ratio_lower95"], row["ratio_upper95"] = np.quantile(ratio, [.025, .975])
            if row["fit_status"] == "ok" and grouping == "all":
                hist = study.histograms.query("sample == @sample and variant == @variant and axis == @axis")
                hist = hist.loc[(hist.left >= -window) & (hist.right <= window)]
                sigma, mu = row["sigma_um"], row["mean_um"]
                norm = ndtr((window - mu) / sigma) - ndtr((-window - mu) / sigma)
                expected = row["core_clusters"] * (ndtr((hist.right.to_numpy() - mu) / sigma)
                                                    - ndtr((hist.left.to_numpy() - mu) / sigma)) / norm
                use = expected >= 5
                ndof = int(use.sum()) - 3
                row["pearson_chi2_ndof"] = (float(np.sum((hist["count"].to_numpy()[use] - expected[use]) ** 2 / expected[use]) / ndof)
                                              if ndof > 0 else np.nan)
            rows.append(row)
        if progress is not None and grouping == "all":
            progress(f"Gaussian residual-core fits: {sample} {axis}; {bootstrap_resamples} paired whole-event resamples")
    return pd.DataFrame(rows)


def plot_residual_resolution_fits(study, fits, samples=("pu200",)):
    """Small multiples separate residual distributions from their Gaussian fits."""
    from scipy.special import ndtr
    from scipy.stats import norm
    methods = list(study.histograms.variant.unique())
    for sample in samples:
        for axis in ("local0", "local1"):
            fig, axes = plt.subplots(int(np.ceil(len(methods) / 3)), 3, figsize=(15, 7),
                                      squeeze=False, sharex=True, sharey=True, constrained_layout=True)
            for ax, variant in zip(axes.flat, methods):
                method = "linear_ATLAS" if variant == "linear" else variant
                color = METHOD_COLORS.get(method, "#555555")
                hist = study.histograms.query("sample == @sample and variant == @variant and axis == @axis")
                row = fits.query("sample == @sample and variant == @variant and axis == @axis and grouping == 'all'").iloc[0]
                window = row.fit_window_um
                ax.stairs(hist.density.to_numpy(), np.r_[hist.left.to_numpy(), hist.right.iloc[-1]],
                          color=color, linewidth=2, label="Residual data")
                ax.axvspan(-window, window, color="gray", alpha=.08, label="Common fit window")
                if row.fit_status == "ok":
                    x = np.linspace(-window, window, 400)
                    norm_window = ndtr((window - row.mean_um) / row.sigma_um) - ndtr((-window - row.mean_um) / row.sigma_um)
                    density = row.core_fraction * norm.pdf(x, row.mean_um, row.sigma_um) / norm_window
                    ax.plot(x, density, color="black", linestyle="--", linewidth=2,
                            label=f"Gaussian: sigma={row.sigma_um:.2f} um")
                    ax.set_title(f"{METHOD_LABELS.get(method, method)}\nmean={row.mean_um:.2f} um; core={100 * row.core_fraction:.1f}%")
                else:
                    ax.set_title(f"{METHOD_LABELS.get(method, method)}: {row.fit_status}")
                ax.set(xlim=(-2 * window, 2 * window), xlabel="Reconstructed - ROOT truth (micrometres)", ylabel="Density")
                ax.legend(fontsize=8)
                ax.grid(alpha=.15)
            for ax in list(axes.flat)[len(methods):]:
                ax.set_visible(False)
            fig.suptitle(f"{sample} {axis}: Gaussian core widths; provisional ROOT truth reference")
            plt.show()
    pooled = fits.query("grouping == 'all'")
    names = list(pooled.variant.unique())
    for axis in ("local0", "local1"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        for sample, marker, offset in (("pu0", "o", -.12), ("pu200", "s", .12)):
            for index, variant in enumerate(names):
                frame = pooled.query("sample == @sample and variant == @variant and axis == @axis")
                if frame.empty or frame.iloc[0].fit_status != "ok":
                    continue
                row = frame.iloc[0]
                method = "linear_ATLAS" if variant == "linear" else variant
                color = METHOD_COLORS.get(method, "#555555")
                for ax, value, low, high in ((axes[0], "sigma_um", "sigma_lower95", "sigma_upper95"),
                                              (axes[1], "sigma_ratio_to_native", "ratio_lower95", "ratio_upper95")):
                    y = row.get(value, np.nan)
                    if not np.isfinite(y):
                        continue
                    lo, hi = row.get(low, np.nan), row.get(high, np.nan)
                    ax.plot(index + offset, y, marker=marker, color=color, markersize=7,
                            label=sample if index == 0 else None)
                    if np.isfinite([lo, hi]).all():
                        ax.vlines(index + offset, lo, hi, color=color, linewidth=2)
        axes[0].set(ylabel="Fitted Gaussian core sigma (micrometres)", title=f"{axis}: residual-distribution fits")
        axes[1].axhline(1, color="black", linestyle=":")
        axes[1].set(ylabel="Fitted sigma / native sigma", title="Paired event-bootstrap 95% intervals")
        for ax in axes:
            ax.set_xticks(range(len(names)), [METHOD_LABELS.get(n, n) for n in names], rotation=25, ha="right")
            ax.legend()
            ax.grid(alpha=.15)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--atlas-pixels":
        raise SystemExit("Usage: paired_vs_published.py --atlas-pixels graph.pyg")
    payload = _read_atlas_pixel_file(sys.argv[2])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    sys.stdout.buffer.write(buffer.getvalue())
