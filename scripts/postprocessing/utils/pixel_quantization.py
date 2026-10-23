"""Frozen pixel-only encodings, independent of the notebook and ACTS runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PRESETS_PATH = (Path(__file__).resolve().parents[3] /
                "configs_development/paired_clusters/pixel_quantization_v1.json")
SCHEMES = ("p99.9", "p99.5", "p99", "none")
DEFAULT_SCHEME = "p99.9"
METADATA_KEY = b"colliderml.cell_quantization"


def add_quantization_argument(parser) -> None:
    parser.add_argument("--pixel-quantization", choices=SCHEMES, default=DEFAULT_SCHEME,
                        help="Frozen extra pixel_code feature (default: p99.9); "
                             "none disables codes. Native value is always preserved.")


class PixelQuantizer:
    """Apply fixed upper-bin ties and collect cell-level encoding diagnostics."""

    def __init__(self, scheme: str = DEFAULT_SCHEME):
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown pixel quantization scheme {scheme!r}; choose {SCHEMES}")
        self.scheme = scheme
        preset_bytes = PRESETS_PATH.read_bytes()
        self.preset_sha256 = hashlib.sha256(preset_bytes).hexdigest()
        self.config = json.loads(preset_bytes)
        self.boundaries = (np.asarray(self.config["schemes"][scheme]["boundaries"], dtype=np.float64)
                           if scheme != "none" else None)
        if self.boundaries is not None:
            if (self.boundaries.shape != (14,) or not np.isfinite(self.boundaries).all()
                    or self.boundaries[0] <= self.config["activation_min"]
                    or not np.all(np.diff(self.boundaries) > 0)):
                raise ValueError("Pixel quantization preset has invalid boundaries")
            steps = np.diff(np.r_[self.config["activation_min"], self.boundaries])
            if not np.allclose(steps, steps[0], rtol=1e-14, atol=0):
                raise ValueError("Pixel quantization preset must have equal-width bins")
        self.pixel_cells = 0
        self.nonpixel_cells = 0
        self.code_counts = np.zeros(16, dtype=np.int64)
        self.genuinely_clipped_cells = 0

    def validate_digi_config(self, path: Path) -> None:
        if self.boundaries is not None:
            expected = self.config["calibration"]["digi_config_sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Pixel quantization requires the calibrated geometric digitization "
                                 "config; use --pixel-quantization none for a different config")

    def codes(self, values) -> np.ndarray:
        if self.boundaries is None:
            raise ValueError("Native mode does not define quantized codes")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Pixel activations must be a finite one-dimensional array")
        if np.any(values < self.config["activation_min"]):
            raise ValueError("Pixel activation below the calibrated minimum 0.01")
        return (1 + np.searchsorted(self.boundaries, values, side="right")).astype(np.uint8)

    def encode(self, geometry_ids, values) -> np.ndarray:
        """Return uint8 codes; internal zero sentinels are written as Parquet nulls."""
        values = np.asarray(values, dtype=np.float64)
        geometry_ids = np.asarray(geometry_ids, dtype=np.uint64)
        if values.ndim != 1 or geometry_ids.shape != values.shape:
            raise ValueError("Cell geometry and activation arrays must be aligned and one-dimensional")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Cell activations must be positive and finite")
        pixels = np.isin((geometry_ids >> 56) & 0xff, self.config["pixel_volumes"])
        result = np.zeros(len(values), dtype=np.uint8)
        if self.boundaries is not None:
            codes = self.codes(values[pixels])
            result[pixels] = codes
            self.code_counts += np.bincount(codes, minlength=16)
            self.genuinely_clipped_cells += int(np.count_nonzero(values[pixels] >= self.code15_bin_end))
        self.pixel_cells += int(np.count_nonzero(pixels))
        self.nonpixel_cells += int(np.count_nonzero(~pixels))
        return result

    @property
    def code15_bin_end(self) -> float:
        return float(self.boundaries[-1] +
                     (self.boundaries[-1] - self.config["activation_min"]) / 14)

    def report(self) -> dict:
        result = {
            "scheme": self.scheme, "enabled": self.boundaries is not None,
            "pixel_volumes": self.config["pixel_volumes"],
            "pixel_cells": self.pixel_cells, "unchanged_nonpixel_cells": self.nonpixel_cells,
            "native_value_column": "value", "native_value_storage": "float64",
            "native_value_semantics": self.config["input_units"],
            "native_values_preserved": True,
            "code_column": "pixel_code", "code_storage": "nullable uint8",
            "pixel_code_semantics": ("positive integer codes 1-15" if self.boundaries is not None
                                     else "disabled (all codes null)"),
            "nonpixel_code_semantics": "null (not calibrated)",
            "hit_positions": "unchanged native ACTS measurements; no centroid recomputation",
        }
        if self.boundaries is not None:
            result.update({
                "preset_version": self.config["version"],
                "preset_sha256": self.preset_sha256,
                "calibration": self.config["calibration"],
                "calibration_percentile": self.config["schemes"][self.scheme]["calibration_percentile"],
                "activation_min": self.config["activation_min"],
                "boundaries": self.boundaries.tolist(), "boundary_ties": "upper_bin",
                "code_min": 1, "code_max": 15,
                "code15_edge": float(self.boundaries[-1]), "code15_bin_end": self.code15_bin_end,
                "pixel_code_counts": {str(code): int(self.code_counts[code]) for code in range(1, 16)},
                "code15_cells": int(self.code_counts[15]),
                "genuinely_clipped_cells": self.genuinely_clipped_cells,
            })
        return result
