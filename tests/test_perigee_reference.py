"""Analytic geometry and independent Lorentz integration conventions."""

import unittest

import numpy as np

from notebooks.validation.perigee_reference import analytic_perigee, reference_perigee, uniform_state, slab_solenoid_perigee


class PerigeeReferenceTest(unittest.TestCase):
    def test_straight_line_and_signed_d0(self):
        result = analytic_perigee([3, 4, 7], [1, 0, 2], 0, 2)
        self.assertAlmostEqual(result["d0"], 4.)
        self.assertAlmostEqual(result["z0"], 1.)
        self.assertAlmostEqual(result["path_length_mm"], -3 * np.sqrt(5))

    def test_exact_helix_has_transverse_minimum(self):
        for charge in (-1, 1):
            result = analytic_perigee([100, -40, 200], [.3, .4, 2], charge, 2.)
            distance = np.hypot(*result["position_mm"][:2])
            self.assertLessEqual(distance, np.hypot(100, 40))
            self.assertAlmostEqual(distance, abs(result["d0"]))
            for offset in (-.1, .1):
                r, _ = uniform_state([100, -40, 200], [.3, .4, 2], charge, 2.,
                                     result["path_length_mm"] + offset)
                self.assertGreater(np.hypot(*r[:2]), distance)

    def test_integrator_matches_analytic_both_charges_and_forward_tracks(self):
        cases = [([.01, -.02, 10], [1., .4, 2.]),
                 ([100., -40., 200.], [.3, .4, 2.]),
                 ([-58., -26., -1454.], [-.02, .1, -12.]),
                 ([40., 20., 100.], [.005, .003, .2])]
        for position, momentum in cases:
            for charge in (-1, 1):
                with self.subTest(position=position, charge=charge):
                    expected = analytic_perigee(position, momentum, charge, 2.)
                    actual = reference_perigee(position, momentum, charge, lambda _: np.array([0, 0, 2.]))
                    self.assertEqual(actual["status"], "ok")
                    self.assertGreater(actual["curvature"], 0)
                    np.testing.assert_allclose([actual[key] for key in ("d0", "z0", "path_length_mm")],
                                               [expected[key] for key in ("d0", "z0", "path_length_mm")],
                                               atol=2e-4, rtol=1e-7)

    def test_degenerate_transverse_momentum_rejected(self):
        with self.assertRaises(ValueError):
            analytic_perigee([1, 2, 3], [0, 0, 1], 1, 2)

    def test_solenoid_boundary_has_independent_piecewise_solution(self):
        position, momentum = [-58., -26., -1454.], [-.02, .1, -12.]
        exact = slab_solenoid_perigee(position, momentum, 1.)
        self.assertEqual(exact["region"], "outside_z")
        reference = reference_perigee(position, momentum, 1.,
                                      lambda r: np.array([0, 0, 2. if abs(r[2]) < 3000 else 0.]),
                                      rtol=2e-13, atol=1e-12, step_scale=.25)
        self.assertEqual(reference["status"], "ok")
        np.testing.assert_allclose([reference["d0"], reference["z0"]],
                                   [exact["d0"], exact["z0"]], atol=2e-3, rtol=0)

    def test_initial_stationary_maximum_is_not_accepted_as_a_minimum(self):
        position, momentum = [200., 0., 0.], [0., -.01, .01]
        expected = analytic_perigee(position, momentum, 1., 2.)
        actual = reference_perigee(position, momentum, 1., lambda _: np.array([0., 0., 2.]))
        self.assertEqual(actual["status"], "ok")
        self.assertLess(actual["path_length_mm"], 0)
        self.assertLess(abs(actual["d0"]), 200.)
        self.assertGreater(actual["curvature"], 0)
        np.testing.assert_allclose([actual[key] for key in ("d0", "z0", "path_length_mm")],
                                   [expected[key] for key in ("d0", "z0", "path_length_mm")], atol=2e-4)


if __name__ == "__main__":
    unittest.main()
