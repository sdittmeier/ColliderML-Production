"""Gaussian residual-distribution fits, conditional core windows and paired errors."""

import importlib.util
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(n) for n in ("numpy", "pandas", "pyarrow", "matplotlib", "scipy"))


@unittest.skipUnless(AVAILABLE, "requires notebook dependencies and scipy")
class ResolutionFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import pandas as pd
        from notebooks.validation import paired_vs_published as p
        cls.np, cls.pd, cls.p = np, pd, p

    def tearDown(self):
        self.p.plt.close("all")

    def test_fits_gaussian_not_merely_trimmed_standard_deviation(self):
        np, p = self.np, self.p
        values = np.random.default_rng(7).normal(2., 9., 250000)
        core = values[(values >= -15) & (values < 15)]
        fit = p.gaussian_core_fit([len(core), core.sum(), core @ core], 15)
        self.assertEqual(fit["fit_status"], "ok")
        self.assertAlmostEqual(fit["mean_um"], 2., delta=.1)
        self.assertAlmostEqual(fit["sigma_um"], 9., delta=.1)
        self.assertLess(core.std(), fit["sigma_um"] - 1.)
        self.assertEqual(p.gaussian_core_fit([1, 1, 1], 15)["fit_status"], "insufficient_entries")
        self.assertEqual(p.gaussian_core_fit([200, 400, 800], 15)["fit_status"], "degenerate_distribution")
        with self.assertRaises(ValueError):
            p.gaussian_core_fit([100, 0, 100], 0)

    def study(self):
        np, pd, p = self.np, self.pd, self.p
        rows, histograms = [], []
        rng = np.random.default_rng(19)
        edges = np.linspace(-60, 60, 121)
        accumulated = {}
        for sample in ("pu0", "pu200"):
            for event in range(25, 33):
                values = rng.normal(.1 * (event - 29), 6., (2000, 2))
                for variant, scale in (("native", 1.), ("linear_P99.9", 1.1)):
                    for axis in range(2):
                        residuals = values[:, axis] * scale
                        core = residuals[(residuals >= -30) & (residuals < 30)]
                        rows.append(dict(sample=sample, event=event, variant=variant, axis=f"local{axis}",
                                         grouping="all", group="all", window_um=30.,
                                         n_full=len(residuals), n_fit=len(core), sum_x=core.sum(), sum_x2=core @ core))
                        key = (sample, variant, f"local{axis}")
                        accumulated.setdefault(key, []).append(residuals)
        for (sample, variant, axis), parts in accumulated.items():
            values = np.concatenate(parts)
            counts = np.histogram(values, bins=edges)[0]
            histograms.extend(dict(sample=sample, variant=variant, axis=axis, left=edges[i], right=edges[i + 1],
                                   count=count, density=count / len(values)) for i, count in enumerate(counts))
        return p.PixelResidualStudy(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(histograms),
                                     pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(rows))

    def test_paired_event_bootstrap_ratios_reproducibility_and_fit_plots(self):
        p, np = self.p, self.np
        study = self.study()
        fits = p.fit_residual_resolutions(study, bootstrap_resamples=20, progress=None)
        native = fits.query("variant == 'native'")
        np.testing.assert_allclose(native.sigma_ratio_to_native, 1.)
        np.testing.assert_allclose(native.ratio_lower95, 1.)
        np.testing.assert_allclose(native.ratio_upper95, 1.)
        quantized = fits.query("variant == 'linear_P99.9'")
        np.testing.assert_allclose(quantized.sigma_ratio_to_native, 1.1, atol=.005)
        self.assertTrue((quantized.bootstrap_valid >= 16).all())
        self.assertTrue(np.isfinite(fits.pearson_chi2_ndof).all())
        again = p.fit_residual_resolutions(study, bootstrap_resamples=20, progress=None)
        self.pd.testing.assert_frame_equal(fits, again)
        with patch.object(p.plt, "show"):
            p.plot_residual_resolution_fits(study, fits, samples=("pu200",))
        bad = self.study()
        bad.fit_statistics = bad.fit_statistics.iloc[1:]
        with self.assertRaisesRegex(ValueError, "same events"):
            p.fit_residual_resolutions(bad, bootstrap_resamples=0, progress=None)


if __name__ == "__main__":
    unittest.main()
