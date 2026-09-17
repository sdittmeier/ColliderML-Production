"""Missing matched particles must not silently become particle zero."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"),
                     "requires pandas and pyarrow")
class NullableParticleIdTest(unittest.TestCase):
    def test_opt_in_preserves_null_and_default_keeps_legacy_zero(self) -> None:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        from scripts.postprocessing.utils.parquet_utils import build_parquet_from_flat_df

        with tempfile.TemporaryDirectory() as directory:
            frame = pd.DataFrame({"event_id": [0, 0], "particle_id": [7, float("nan")]})
            schema = {"particle_id": pa.list_(pa.uint64())}
            legacy = Path(directory) / "legacy.parquet"
            paired = Path(directory) / "paired.parquet"
            build_parquet_from_flat_df(frame, str(legacy), schema_overrides=schema)
            build_parquet_from_flat_df(frame, str(paired), schema_overrides=schema,
                                       nullable_integer_columns={"particle_id"})
            self.assertEqual(pq.read_table(legacy)["particle_id"][0].as_py(), [7, 0])
            self.assertEqual(pq.read_table(paired)["particle_id"][0].as_py(), [7, None])


if __name__ == "__main__":
    unittest.main()
