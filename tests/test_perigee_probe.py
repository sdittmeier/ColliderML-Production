"""The diagnostic must not confuse a successful status with finite parameters."""

import unittest

from notebooks.validation.probe_perigee import propagation_status


class PerigeeProbeTest(unittest.TestCase):
    def test_finite_parameters_and_path_are_required(self):
        result = dict(ok=True, has_end_parameters=True, finite_parameters=True,
                      d0=1., z0=2., path_length_mm=-10.)
        self.assertEqual(propagation_status(result), "finite_result")
        for field in ("d0", "z0", "path_length_mm"):
            with self.subTest(field=field):
                broken = dict(result, **{field: float("nan")})
                self.assertEqual(propagation_status(broken), "nonfinite_end_parameters")
        self.assertEqual(propagation_status(dict(result, finite_parameters=False)),
                         "nonfinite_end_parameters")

    def test_failure_and_missing_endpoint_are_not_success(self):
        self.assertEqual(propagation_status(dict(ok=False, error="SurfaceError")), "SurfaceError")
        self.assertEqual(propagation_status(dict(ok=True)), "no_end_parameters")


if __name__ == "__main__":
    unittest.main()
