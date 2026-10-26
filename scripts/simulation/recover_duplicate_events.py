"""Strict recovery of sequential DDG4 shared-writer duplicate events."""

from pathlib import Path

import awkward as ak
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
