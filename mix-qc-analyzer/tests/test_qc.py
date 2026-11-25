from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import openpyxl

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


class GammaExtensionTests(unittest.TestCase):
    def gamma(self):
        x = np.arange(256, dtype=float)
        return (5000*np.exp(-0.5*((x-233.4)/3.8)**2)
                + 2400*np.exp(-0.5*((x-32.1)/1.4)**2) + 20 + 0.05*x)

    def test_ba_width_and_covariance_uncertainty(self):
        result = worker('.').analyze_gamma(self.gamma())
        ba = result['ba_fit']
        self.assertTrue(ba['success'], ba)
        self.assertAlmostEqual(ba['peak_center'], 32.1, places=5)
        self.assertAlmostEqual(ba['fwhm'], 3.296748063, places=6)
        self.assertAlmostEqual(ba['fwhm_err'], 2.354820045 * ba['peak_sigma_err'], places=9)
        self.assertGreater(ba['fwhm_err'], 0)

    def test_custom_range_tracks_shifted_barium(self):
        x = np.arange(256, dtype=float)
        gamma = 3000*np.exp(-0.5*((x-43.2)/1.5)**2) + 25 + 0.15*x
        instance = SpectrumAnalysisThread('', '.', '', False, False, ba_fit_start=39, ba_fit_end=48)
        fit = instance.analyze_gamma(gamma)['ba_fit']
        self.assertTrue(fit['success'], fit)
        self.assertAlmostEqual(fit['peak_center'], 43.2, places=5)
        self.assertAlmostEqual(fit['fwhm'], 3.532230068, places=6)
        self.assertEqual((fit['fit_start'], fit['fit_end']), (39, 48))

    def test_invalid_ranges_are_rejected(self):
        for start, end in [(32, 32), (36, 28), (28, 32), (-1, 36), (250, 256)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                SpectrumAnalysisThread('', '.', '', False, False, ba_fit_start=start, ba_fit_end=end)

    def test_failed_ba_does_not_hide_cs_or_spectrum(self):
        instance = worker('.')
        real_fit = instance.fit_peak_in_region
        def fitter(spectrum, start, end):
            if start == 28:
                raise RuntimeError('test peak failure')
            return real_fit(spectrum, start, end)
        with patch.object(instance, 'fit_peak_in_region', side_effect=fitter):
            result = instance.analyze_gamma(self.gamma())
        self.assertTrue(result['cs_fit']['success'])
        self.assertFalse(result['ba_fit']['success'])
        fig = instance.make_gamma_figure('QC.phd', result)
        self.assertEqual(len(fig.data[0].x), 256)
        self.assertIn('test peak failure', str(fig.layout.annotations))

    def test_missing_gamma_exports_blank_width_with_status(self):
        instance = worker('.')
        result = instance.analyze_gamma(None)
        result.update(file='empty.phd', date='2026-01-01', success=False, error='No matrix')
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'report.xlsx')
            instance.save_to_excel([result], '', folder, path)
            wb = openpyxl.load_workbook(path, data_only=True)
            row = list(wb['Гамма-пики'].values)[1]
            self.assertIsNone(row[2])
            self.assertIsNone(row[7])
            self.assertIsNone(row[8])
            self.assertIn('Не удалось', row[11])
            wb.close()

    def test_plot_uses_stored_fits_and_local_half_height(self):
        instance = worker('.')
        result = instance.analyze_gamma(self.gamma())
        with patch.object(instance, 'fit_peak_in_region', side_effect=AssertionError('Refitting')):
            fig = instance.make_gamma_figure('QC.phd', result)
        for trace in fig.data:
            if 'гаусс + фон' in trace.name or 'линейный фон' in trace.name:
                start, end = (209, 251) if trace.name.startswith('Cs') else (28, 36)
                self.assertGreaterEqual(min(trace.x), start)
                self.assertLessEqual(max(trace.x), end)
        bracket = next(trace for trace in fig.data if trace.name.startswith('Ba FWHM'))
        A, mu, sigma, a, b = result['ba_fit']['parameters']
        ends = np.asarray(bracket.x)
        self.assertAlmostEqual(ends[1]-ends[0], result['ba_fit']['fwhm'])
        np.testing.assert_allclose(np.asarray(bracket.y)-(a*ends+b), A/2)
        np.testing.assert_allclose(instance.gauss_with_linear(ends, A, mu, sigma, a, b), bracket.y)

    def test_gamma_survives_damaged_matrix_and_beta_fit_error(self):
        instance = worker('.')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'QC.phd'
            write_synthetic_phd(path)
            text = path.read_text()
            path.write_text('#Histogram\n256 256\n1 2\n#g_Spectrum' + text.split('#g_Spectrum', 1)[1])
            result = instance.fit_spectrum_to_template(path, None, {})
            self.assertFalse(result['success'])
            self.assertTrue(result['ba_fit']['success'])
            write_synthetic_phd(path)
            def broken_template(x):
                raise RuntimeError('beta failure')
            result = instance.fit_spectrum_to_template(path, broken_template, {})
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'beta failure')
            self.assertTrue(result['ba_fit']['success'])


if __name__ == '__main__':
    unittest.main()
