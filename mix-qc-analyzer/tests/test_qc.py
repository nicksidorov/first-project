from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from pipp import SpectrumAnalysisThread
from qc_smoke import write_synthetic_phd


def worker(folder, irls=False):
    return SpectrumAnalysisThread('', str(folder), '', False, irls)


class ParserTests(unittest.TestCase):
    def test_matrix_and_last_gamma_channel(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'QC.phd'
            matrix, gamma = write_synthetic_phd(path)
            date, actual_matrix, actual_gamma = worker(folder).parse_spectrum(path)
            self.assertEqual(date, '2026-01-01')
            np.testing.assert_allclose(actual_matrix, matrix, rtol=1e-7)
            np.testing.assert_allclose(actual_gamma, gamma, rtol=1e-7)
            self.assertGreater(actual_gamma[-1], 0)

    def test_truncated_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.PHD'
            path.write_text('#Histogram\n256 256\n1 2\n')
            self.assertIsNone(worker(folder).parse_spectrum(path)[1])

    def test_missing_gamma_keeps_valid_matrix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'beta.PHD'
            write_synthetic_phd(path, include_gamma=False)
            _, matrix, gamma = worker(folder).parse_spectrum(path)
            self.assertEqual(matrix.shape, (256, 256))
            self.assertIsNone(gamma)

    def test_truncated_gamma_not_silently_zero_padded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad_gamma.PHD'
            write_synthetic_phd(path, include_gamma=False)
            with path.open('a') as file:
                file.write('\n#g_Spectrum\n256\n0 1 2 3\n#End\n')
            _, matrix, gamma = worker(folder).parse_spectrum(path)
            self.assertEqual(matrix.shape, (256, 256))
            self.assertIsNone(gamma)


class FitTests(unittest.TestCase):
    def test_filter_boundaries_and_source_preserved(self):
        matrix = np.ones((256, 256))
        actual = worker('.').apply_filters(matrix)
        self.assertEqual(actual[39, 200], 0)
        self.assertEqual(actual[40, 149], 0)
        self.assertEqual(actual[40, 150], 1)
        np.testing.assert_array_equal(matrix, np.ones((256, 256)))

    def test_irls_recomputes_weights_from_current_prediction(self):
        instance = worker('.', True)
        x = np.arange(1, 4, dtype=float)
        y = np.array([2, 5, 8], dtype=float)
        calls = []
        def fake_curve_fit(model, x_data, y_data, **kwargs):
            calls.append(kwargs['sigma'].copy())
            return np.array([4.0]), np.eye(1)
        with patch('pipp.curve_fit', side_effect=fake_curve_fit):
            parameters, _, iterations = instance.fit_poisson_mle(lambda x, a: a * x, x, y, [1.0])
        np.testing.assert_allclose(calls[0], np.sqrt(x))
        np.testing.assert_allclose(calls[1], np.sqrt(4 * x))
        self.assertEqual(iterations, 2)
        self.assertTrue(instance.mle_converged)
        self.assertEqual(parameters[0], 4)

    def test_irls_failed_iteration_is_not_reported_as_success(self):
        instance = worker('.', True)
        with patch('pipp.curve_fit', side_effect=RuntimeError('not converged')):
            with self.assertRaisesRegex(RuntimeError, 'IRLS'):
                instance.fit_poisson_mle(lambda x, a: a * x, np.array([1., 2.]), np.array([1., 2.]), [1.])
        self.assertFalse(instance.mle_converged)

    def test_iteration_limit_does_not_claim_convergence(self):
        instance = worker('.', True)
        with patch('pipp.curve_fit', return_value=(np.array([2.0]), np.eye(1))):
            instance.fit_poisson_mle(lambda x, a: a * x, np.array([1., 2.]), np.array([1., 2.]), [1.], max_iter=1)
        self.assertFalse(instance.mle_converged)

    def test_gamma_peak(self):
        x = np.arange(256, dtype=float)
        spectrum = 5000 * np.exp(-0.5 * ((x - 233.4) / 3.8) ** 2) + 20 + 0.05 * x
        fit = worker('.').fit_peak_in_region(spectrum)
        self.assertTrue(fit['success'])
        self.assertAlmostEqual(fit['peak_center'], 233.4, places=5)
        self.assertAlmostEqual(abs(fit['peak_sigma']), 3.8, places=5)
        self.assertGreater(fit['peak_area'], 0)


if __name__ == '__main__':
    unittest.main()
