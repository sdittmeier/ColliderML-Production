"""Verified PODIO entry selection and strict DDG4 duplicate-event recovery."""

from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def leaf_names(tree):
    return [name for name in tree.keys() if not tree[name].branches]


def equal_entries(left, right, left_entry, right_entry, ignore_timestamp=False):
    names = leaf_names(left)
    if set(names) != set(leaf_names(right)):
        raise ValueError("Recovery changed the branch schema")
    checked = 0
    for name in names:
        if ignore_timestamp and name == "EventHeader/EventHeader.timeStamp":
            continue
        a = left[name].array(entry_start=left_entry, entry_stop=left_entry + 1)
        b = right[name].array(entry_start=right_entry, entry_stop=right_entry + 1)
        if not ak.almost_equal(a, b, rtol=0, atol=0):
            raise ValueError(f"Different event content in branch {name}; refusing duplicate recovery")
        checked += 1
    if not checked:
        raise ValueError("No event branches were compared")
    return checked


def duplicate_entries(tree, events):
    if tree.num_entries != 2 * events:
        raise ValueError("Recovery requires exactly two copies of every requested event")
    numbers = ak.to_list(tree["EventHeader/EventHeader.eventNumber"].array())
    groups = {}
    for entry, number in enumerate(numbers):
        if len(number) != 1:
            raise ValueError("Expected exactly one EventHeader per entry")
        groups.setdefault(int(number[0]), []).append(entry)
    if set(groups) != set(range(events)) or any(len(group) != 2 for group in groups.values()):
        raise ValueError("Recovery requires two copies of each logical event ID")
    checked = 0
    for group in groups.values():
        checked += equal_entries(tree, tree, *group, ignore_timestamp=True)
    return sorted(group[0] for group in groups.values()), checked


def recover(source, destination, events):
    """Write a separate, fully verified podio file; never alter the source."""
    import pyedm4hep  # noqa: F401 - load EDM4hep dictionaries before reading frames
    from podio.root_io import Reader
    from ROOT import podio

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError(f"Recovery destination exists: {destination}")
    with uproot.open(source) as original:
        keep, checked = duplicate_entries(original["events"], events)
    reader = Reader(str(source))
    categories = tuple(reader.categories)
    writer = podio.ROOTWriter(str(destination))
    for category in categories:
        for entry, frame in enumerate(reader.get(category)):
            if category != "events" or entry in keep:
                writer.writeFrame(frame._frame, category)
    writer.finish()
    with uproot.open(source) as original, uproot.open(destination) as result:
        for category in categories:
            left, right = original[category], result[category]
            selected = keep if category == "events" else list(range(left.num_entries))
            if right.num_entries != len(selected):
                raise ValueError(f"Recovery changed category count: {category}")
            for new_entry, old_entry in enumerate(selected):
                equal_entries(left, right, old_entry, new_entry)
    return dict(original_entries=2 * events, recovered_events=events, kept_entries=keep,
                duplicate_branch_comparisons=checked, copied_categories=list(categories),
                ignored_duplicate_field="EventHeader.timeStamp",
                policy="Exact duplicate contents required except timestamp; all categories preserved")


def selection_fields(left, right, category):
    """Permit only the documented old-to-current EDM4hep colorFlow removal."""
    names, written = leaf_names(left), set(leaf_names(right))
    omitted = set(names) - written
    allowed = {"MCParticles/MCParticles.colorFlow.a", "MCParticles/MCParticles.colorFlow.b"} if category == "events" else set()
    if written - set(names) or omitted - allowed:
        raise ValueError(f"Selection changed category schema: {category}; missing={sorted(omitted)}, added={sorted(written - set(names))}")
    return [name for name in names if name in written], sorted(omitted)


def verify_selection_content(original, selected, names):
    """Verify values exactly, allowing only known EventHeader storage casts."""
    casts = {}
    header_types = {"EventHeader/EventHeader.eventNumber": np.dtype("uint64"),
                    "EventHeader/EventHeader.runNumber": np.dtype("uint32")}
    for name in names:
        left, right = original[name], selected[name]
        if name in header_types:
            source_dtype = ak.to_numpy(ak.flatten(left)).dtype
            selected_dtype = ak.to_numpy(ak.flatten(right)).dtype
            if source_dtype == np.dtype("int32") and selected_dtype == header_types[name]:
                left = ak.values_astype(left, selected_dtype)
                casts[name] = dict(source_dtype=str(source_dtype), selected_dtype=str(selected_dtype))
        if not ak.almost_equal(left, right, rtol=0, atol=0):
            raise ValueError(f"Selection changed event content: {name}")
    return casts


def select_event_range(source, destination, start, events):
    """Copy a contiguous ROOT-entry range, preserving PODIO relations/metadata."""
    source, destination = Path(source), Path(destination)
    if start < 0 or events < 1:
        raise ValueError("Require a nonnegative start and positive event count")
    if destination.exists():
        raise ValueError(f"Selection destination exists: {destination}")
    with uproot.open(source) as original:
        if start + events > original["events"].num_entries:
            raise ValueError("Selected ROOT entries exceed the input")
    import pyedm4hep  # noqa: F401
    from podio.root_io import Reader
    from ROOT import podio

    destination.parent.mkdir(parents=True, exist_ok=True)
    reader = Reader(str(source))
    categories = tuple(reader.categories)
    writer = podio.ROOTWriter(str(destination))
    for category in categories:
        for entry, frame in enumerate(reader.get(category)):
            if category == "events":
                if entry >= start + events:
                    break
                if entry < start:
                    continue
            writer.writeFrame(frame._frame, category)
    writer.finish()
    checked, omitted_fields, header_casts = 0, {}, {}
    with uproot.open(source) as original, uproot.open(destination) as result:
        for category in categories:
            left, right = original[category], result[category]
            names, omitted = selection_fields(left, right, category)
            if omitted:
                omitted_fields[category] = omitted
            count = events if category == "events" else left.num_entries
            offset = start if category == "events" else 0
            if right.num_entries != count:
                raise ValueError(f"Selection changed category count: {category}")
            # Read all branches of one event together, retaining bounded memory.
            for entry in range(count):
                a = left.arrays(names, entry_start=offset + entry, entry_stop=offset + entry + 1)
                b = right.arrays(names, entry_start=entry, entry_stop=entry + 1)
                header_casts.update(verify_selection_content(a, b, names))
                checked += len(names)
    return dict(source_entry_start=start, source_entry_stop_exclusive=start + events,
                events=events, copied_categories=list(categories), compared_branch_entries=checked,
                omitted_legacy_fields=omitted_fields, header_storage_casts=header_casts,
                policy="Physical fields verified exactly; explicitly recorded EDM4hep header storage casts and obsolete colorFlow removal only. Original input retained; event headers not renumbered")
