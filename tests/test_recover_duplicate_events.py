"""Strict selection of duplicate writer entries without the podio runtime."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import awkward as ak
import numpy as np
import uproot

from scripts.simulation.recover_duplicate_events import duplicate_entries, selection_fields, select_event_range, verify_selection_content


class DuplicateWriterRecoveryTest(unittest.TestCase):
    def fixture(self, numbers, values, timestamps=None):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "events.root"
        data = {"EventHeader/EventHeader.eventNumber": ak.Array([[n] for n in numbers]),
                "PixelBarrelReadout/PixelBarrelReadout.cellID": ak.Array(values),
                "EventHeader/EventHeader.timeStamp": ak.Array([[n] for n in (timestamps or numbers)])}
        with uproot.recreate(path) as root:
            root.mktree("events", data)
        root = uproot.open(path)
        self.addCleanup(root.close)
        return root["events"]

    def test_only_obsolete_colorflow_omission_is_permitted(self):
        def tree(names):
            return {name: SimpleNamespace(branches=[]) for name in names}
        fields = ["MCParticles/MCParticles.PDG", "MCParticles/MCParticles.colorFlow.a", "MCParticles/MCParticles.colorFlow.b"]
        names, missing = selection_fields(tree(fields), tree(fields[:1]), "events")
        self.assertEqual(names, fields[:1])
        self.assertEqual(missing, fields[1:])
        with self.assertRaisesRegex(ValueError, "schema"):
            selection_fields(tree(fields), tree([]), "events")
        with self.assertRaisesRegex(ValueError, "schema"):
            selection_fields(tree(fields), tree(fields + ["unexpected"]), "events")
        with self.assertRaisesRegex(ValueError, "schema"):
            selection_fields(tree(fields), tree(fields[:1]), "metadata")

    def test_header_casts_preserve_ids_and_only_known_unsigned_sentinel_conversion(self):
        event, run = "EventHeader/EventHeader.eventNumber", "EventHeader/EventHeader.runNumber"
        original = {event: ak.values_astype(ak.Array([[99]]), np.int32),
                    run: ak.values_astype(ak.Array([[-1]]), np.int32)}
        selected = {event: ak.values_astype(ak.Array([[99]]), np.uint64),
                    run: ak.values_astype(ak.Array([[4294967295]]), np.uint32)}
        casts = verify_selection_content(original, selected, [event, run])
        self.assertEqual(set(casts), {event, run})
        selected[event] = ak.values_astype(ak.Array([[100]]), np.uint64)
        with self.assertRaisesRegex(ValueError, "eventNumber"):
            verify_selection_content(original, selected, [event, run])
        selected[event] = ak.values_astype(ak.Array([[99]]), np.uint64)
        selected[run] = ak.values_astype(ak.Array([[4294967294]]), np.uint32)
        with self.assertRaisesRegex(ValueError, "runNumber"):
            verify_selection_content(original, selected, [event, run])

    def test_selection_rejects_invalid_range_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source.root", Path(temporary) / "target.root"
            for start, events in [(-1, 1), (0, 0)]:
                with self.assertRaisesRegex(ValueError, "nonnegative"):
                    select_event_range(source, target, start, events)
            target.touch()
            with self.assertRaisesRegex(ValueError, "exists"):
                select_event_range(source, target, 0, 1)
            target.unlink()
            with uproot.recreate(source) as handle:
                handle["events"] = {"value": [1, 2]}
            with self.assertRaisesRegex(ValueError, "exceed"):
                select_event_range(source, target, 1, 2)

    def test_selects_first_identical_copy_in_original_order(self):
        tree = self.fixture([1, 0, 0, 1], [[7], [8, 9], [8, 9], [7]], [1, 2, 3, 4])
        keep, comparisons = duplicate_entries(tree, 2)
        self.assertEqual(keep, [0, 1])
        self.assertGreater(comparisons, 0)

    def test_different_contents_fail(self):
        with self.assertRaisesRegex(ValueError, "Different event content"):
            duplicate_entries(self.fixture([0, 0], [[7], [8]]), 1)

    def test_unexpected_event_set_fails(self):
        for numbers in ([0], [1, 1], [0, 1], [0, 0, 0, 1]):
            with self.subTest(numbers=numbers), self.assertRaises(ValueError):
                duplicate_entries(self.fixture(numbers, [[7] for _ in numbers]), 1)


if __name__ == "__main__":
    unittest.main()
