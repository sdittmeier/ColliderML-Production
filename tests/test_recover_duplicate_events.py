"""Strict selection of duplicate writer entries without the podio runtime."""

from pathlib import Path
import tempfile
import unittest

import awkward as ak
import uproot

from scripts.simulation.recover_duplicate_events import duplicate_entries


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
