"""Versioned paired Parquet interface; no simulation or truth matching lives here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCHEMA_VERSION = "paired-cern-compatible-v1"
SCHEMA_KEY = b"colliderml.paired_schema"
CONTEXT_KEY = b"colliderml.paired_context"
CONTEXT = ("campaign", "dataset", "version", "run")
KEY = [*CONTEXT, "event_id", "measurement_id"]
LEGACY_FIELDS = {
    "x": pa.float32(), "y": pa.float32(), "z": pa.float32(),
    "volume_id": pa.uint8(), "layer_id": pa.uint16(), "surface_id": pa.uint32(),
    "true_x": pa.float32(), "true_y": pa.float32(), "true_z": pa.float32(),
    "time": pa.float32(), "detector": pa.uint8(),
}
EXTRA_FIELDS = {
    "measurement_id": pa.uint64(), "geometry_id": pa.uint64(),
    "local0": pa.float64(), "local1": pa.float64(),
    "particle_id": pa.uint64(), "converted_particle_id": pa.uint64(),
    "particle_ids": pa.list_(pa.uint64()), "particle_ids_complete": pa.bool_(),
    "unresolved_simhit_ids": pa.list_(pa.uint64()), "simhit_ids": pa.list_(pa.uint64()),
}
HIT_SCHEMA = pa.schema([pa.field("event_id", pa.uint32())] +
                       [pa.field(k, pa.list_(v)) for k, v in (LEGACY_FIELDS | EXTRA_FIELDS).items()])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def event_offset(value, run, events):
    if value is None:
        if run != 0:
            raise ValueError("Nonzero runs require an explicit --event-id-offset (never run × events)")
        value = 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or events < 1 or value + events > 2**32:
        raise ValueError("Event range must fit uint32 and contain at least one event")
    return value


def validate_ranges(shards):
    ranges = sorted((s["event_id_start"], s["event_id_stop_exclusive"]) for s in shards)
    for lo, hi in ranges:
        event_offset(lo, 0, hi - lo)
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping event ID ranges")


def metadata(context, offset, events, old=None):
    event_offset(offset, 0, events)
    result = dict(old or {})
    result[SCHEMA_KEY] = SCHEMA_VERSION.encode()
    result[CONTEXT_KEY] = json.dumps({**context, "event_id_start": offset,
        "event_id_stop_exclusive": offset + events,
        "source_event_mapping": "source_local_event = event_id - event_id_start"}, sort_keys=True).encode()
    return result


def context(path_or_schema):
    schema = path_or_schema if isinstance(path_or_schema, pa.Schema) else pq.read_schema(path_or_schema)
    return json.loads((schema.metadata or {}).get(CONTEXT_KEY, b"{}"))


def source_event(path, event):
    c = context(path)
    local = event - c.get("event_id_start", 0)
    if c and not c["event_id_start"] <= event < c["event_id_stop_exclusive"]:
        raise ValueError(f"Event {event} is outside this shard")
    return local


def canonical_flat(table):
    """Normalize an old flat table in memory. Never add old aliases on disk."""
    names = {"local_event": "event_id", "global_x": "x", "global_y": "y", "global_z": "z"}
    return table.rename_columns([names.get(n, n) for n in table.column_names])


def unnest(table):
    if "measurement_id" not in table.column_names or not pa.types.is_list(table.schema.field("measurement_id").type):
        return canonical_flat(table)
    parts = []
    c = context(table.schema)
    for batch in table.to_batches(max_chunksize=1):
        n = len(batch.column("measurement_id")[0].values)
        columns = {f.name: batch.column(i)[0].values for i, f in enumerate(batch.schema) if f.name != "event_id"}
        if any(len(a) != n for a in columns.values()):
            raise ValueError("Nested hit lists are not aligned")
        columns["event_id"] = pa.array([batch.column("event_id")[0].as_py()] * n, pa.uint32())
        for k in CONTEXT:
            if k in c:
                columns[k] = pa.array([c[k]] * n, pa.uint32() if k == "run" else pa.string())
        parts.append(pa.table(columns))
    if parts:
        return pa.concat_tables(parts)
    return pa.table({f.name: pa.array([], f.type.value_type) for f in table.schema if f.name != "event_id"})


def read_event(path, event):
    schema = pq.read_schema(path)
    name = "event_id" if "event_id" in schema.names else "local_event"
    table = ds.dataset(path, format="parquet").to_table(filter=ds.field(name) == event)
    return unnest(table)


def iter_events(path):
    """Yield one old flat event or one new nested event; retain Arrow buffers."""
    f = pq.ParquetFile(path)
    if "measurement_id" in f.schema_arrow.names and pa.types.is_list(f.schema_arrow.field("measurement_id").type):
        for b in f.iter_batches(batch_size=1):
            yield int(b.column("event_id")[0].as_py()), unnest(pa.Table.from_batches([b], schema=f.schema_arrow))
        return
    name = "local_event" if "local_event" in f.schema_arrow.names else "event_id"
    parts, current = [], None
    for b in f.iter_batches(batch_size=65536):
        t = pa.Table.from_batches([b]); ids = t[name].to_numpy()
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for a, z in zip(starts[:-1], starts[1:]):
            ev = int(ids[a])
            if current is not None and ev != current:
                if ev < current:
                    raise ValueError("Flat input must be ordered by event")
                yield current, canonical_flat(pa.concat_tables(parts)); parts = []
            current = ev; parts.append(t.slice(int(a), int(z-a)))
    if current is not None:
        yield current, canonical_flat(pa.concat_tables(parts))


def array_equal(a, b):
    """Exact primitive/nested values and null masks, with NaN equal to NaN."""
    if a.type != b.type or len(a) != len(b) or a.null_count != b.null_count:
        return False
    if pa.types.is_list(a.type):
        if not pc.equal(pc.list_value_length(a), pc.list_value_length(b)).fill_null(True).to_numpy(zero_copy_only=False).all():
            return False
        return array_equal(pc.list_flatten(a), pc.list_flatten(b))
    if pa.types.is_floating(a.type):
        return (np.array_equal(pc.is_null(a).to_numpy(zero_copy_only=False), pc.is_null(b).to_numpy(zero_copy_only=False)) and
                np.array_equal(a.to_numpy(zero_copy_only=False), b.to_numpy(zero_copy_only=False), equal_nan=True))
    return a.equals(b)


def hit_event(hits, converted, public_event, md):
    """Attach existing converter values by verified row position; never match truth."""
    hits = canonical_flat(hits)
    n = len(hits)
    missing = set(LEGACY_FIELDS) - set(converted.column_names)
    if missing:
        raise ValueError(f"Converted tracker hits lack required fields: {sorted(missing)}")
    fields = {name: converted[name][0].values for name in LEGACY_FIELDS}
    if any(len(a) != n for a in fields.values()):
        raise ValueError("Converted tracker-hit counts differ")
    ids = hits['measurement_id'].to_numpy()
    if not np.array_equal(ids, np.arange(n)):
        raise ValueError("Missing, duplicate or shuffled measurement IDs")
    geom = hits['geometry_id'].to_numpy()
    for name, shift, mask in (("volume_id",56,255),("layer_id",36,4095),("surface_id",8,1048575)):
        if not np.array_equal((geom >> shift) & mask, fields[name].to_numpy()):
            raise ValueError("Converted geometry differs")
    maximum = 0.
    for name in ('x','y','z'):
        a, b = hits[name].to_numpy(), fields[name].to_numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,rtol=0,atol=2e-4):
            raise ValueError("Converted position differs")
        maximum = max(maximum, float(np.max(np.abs(a-b), initial=0.)))
    for name, typ in EXTRA_FIELDS.items():
        fields[name] = hits[name].combine_chunks() if name in hits.column_names else pa.nulls(n,typ)
    columns = [pa.array([public_event], pa.uint32())]
    for name, typ in (LEGACY_FIELDS | EXTRA_FIELDS).items():
        a = pc.cast(fields[name], typ, safe=True)
        columns.append(pa.ListArray.from_arrays(pa.array([0,n],pa.int32()), a))
    return pa.Table.from_arrays(columns, schema=HIT_SCHEMA.with_metadata(md)), maximum


def rewrite_cells(table, offset, version, md):
    table = canonical_flat(table)
    ids = pc.add(pc.cast(table['event_id'],pa.uint64()), offset)
    table = table.set_column(table.schema.get_field_index('event_id'),'event_id',pc.cast(ids,pa.uint32(),safe=True))
    if 'version' in table.column_names:
        table = table.set_column(table.schema.get_field_index('version'),'version',pa.array([version]*len(table)))
    return table.replace_schema_metadata(md)


def particle_schema(source, md):
    return pa.schema([pa.field(f.name, pa.uint32() if f.name == 'event_id' else f.type) for f in source], metadata=md)


def rewrite_particles(table, offset, md):
    values=pc.cast(pc.add(pc.cast(table['event_id'],pa.uint64()),offset),pa.uint32(),safe=True)
    return table.set_column(table.schema.get_field_index('event_id'),'event_id',values).replace_schema_metadata(md)


def resolve_sample(root, sample, event=None, run=None):
    root=Path(root)
    if (root/sample/'report.json').is_file():
        return {k:root/sample/(k+'.parquet') for k in ('hits','cells','particles')},root/sample/'report.json'
    manifest_path=root/'manifest.json'
    if manifest_path.is_file():
        m=json.loads(manifest_path.read_text()); candidates=[]
        for s in m['shards']:
            if run is not None and s['run']!=run: continue
            if event is not None and 'event_id_start' in s and not s['event_id_start']<=event<s['event_id_stop_exclusive']: continue
            candidates.append(s)
        if len(candidates)!=1: raise ValueError('Select one shard by global event_id or explicit run')
        s=candidates[0]
        return {k:root/v['path'] for k,v in s['files'][sample].items()},root/s['provenance']/(sample+'_report.json')
    paths={}
    for k in ('hits','cells','particles'):
        files=sorted(root.glob(f'*_{sample}_{k}/data/*/*.parquet'))
        if len(files)!=1: raise ValueError(f'Expected one {sample} {k} shard, found {len(files)}')
        paths[k]=files[0]
    reports=list(root.glob(f'provenance/run-*/{sample}_report.json'))
    if len(reports)!=1: raise ValueError('Select a unique report')
    return paths,reports[0]
