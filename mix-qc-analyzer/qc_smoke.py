"""Synthetic portability checks; contains no real QC measurements."""
from pathlib import Path
import json
import math
import sys
import tempfile
import time
import traceback

import numpy as np
import openpyxl
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from pipp import APP_VERSION


def write_synthetic_phd(path, amplitude=1.0, stretch=1.0, include_gamma=True):
    x = np.arange(256, dtype=float)
    u = x / stretch
    beta = amplitude / stretch * (1800 * np.exp(-0.5 * ((u - 85) / 20) ** 2)
                                  + 700 * np.exp(-0.5 * ((u - 155) / 24) ** 2))
    matrix = np.zeros((256, 256))
    matrix[200] = beta
    gamma = 5000 * np.exp(-0.5 * ((x - 233.4) / 3.8) ** 2) + 20 + 0.05 * x
    gamma += 2400 * np.exp(-0.5 * ((x - 32.1) / 1.4) ** 2)
    lines = ['#Acquisition', '2026-01-01 12:00:00', '#Histogram', '256 256']
    lines.extend(' '.join(f'{value:.8g}' for value in row) for row in matrix)
    if include_gamma:
        lines.extend(['#g_Spectrum', '256'])
        for start in range(0, 256, 5):
            lines.append(str(start) + ' ' + ' '.join(f'{v:.8g}' for v in gamma[start:start + 5]))
    lines.extend(['#End', 'Synthetic test data only'])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return matrix, gamma


def validate_workbook(path, use_irls, ba_start, ba_end):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        assert set(workbook.sheetnames) == {'Успешные подгонки', 'Неудачные подгонки', 'Гамма-пики', 'Информация'}
        rows = list(workbook['Успешные подгонки'].values)
        assert len(rows) == 4, rows
        for row in rows[1:]:
            assert math.isfinite(row[2]) and row[2] > 0
            assert 0.85 < row[4] < 1.15
            assert row[9] in ('Да', 'Лимит итераций') if use_irls else row[9] == '—'
            if row[0] != 'no_gamma.PHD':
                assert abs(row[6] - 233.4) < 0.1
                assert row[8] > 0
                assert abs(row[12] - 3.2967) < 0.001  # Known sigma=1.4; FWHM=2.35482*sigma.
                assert math.isfinite(row[13]) and row[13] > 0
            else:
                assert row[6] == 'Не удалось'
                assert row[12] is None and row[13] is None
                assert row[16].startswith('Не удалось:')
            assert row[14:16] == (ba_start, ba_end)
        failed = list(workbook['Неудачные подгонки'].values)
        assert len(failed) == 3
        assert {row[0] for row in failed[1:]} == {'broken.PHD', 'gamma_only.PHD'}
        gamma_rows = list(workbook['Гамма-пики'].values)
        assert len(gamma_rows) == 6
        for row in gamma_rows[1:]:
            assert row[9:11] == (ba_start, ba_end)
            if row[0] in ('broken.PHD', 'no_gamma.PHD'):
                assert row[2] is None and row[7] is None
            else:
                assert abs(row[2] - 233.4) < 0.001
                assert abs(row[7] - 3.2967) < 0.001
        info = dict((row[0], row[1]) for row in workbook['Информация'].values if row[0])
        assert info['Всего обработано файлов:'] == 5
        assert info['Версия:'] == APP_VERSION
        assert ('Пуассоновский' in info['Метод фитинга:']) == use_irls
    finally:
        workbook.close()


def run_smoke(app, window, report_path):
    original_save = QFileDialog.getSaveFileName
    original_information = QMessageBox.information
    original_critical = QMessageBox.critical
    messages, errors = [], []
    QMessageBox.information = staticmethod(lambda *a, **k: messages.append(str(a[2])))
    QMessageBox.critical = staticmethod(lambda *a, **k: errors.append(str(a[2])))
    report = {'ok': False, 'version': APP_VERSION, 'frozen': bool(getattr(sys, 'frozen', False)), 'modes': []}
    try:
        with tempfile.TemporaryDirectory(prefix='QC MIX тест ') as temporary:
            root = Path(temporary)
            spectra = root / 'измерения'
            spectra.mkdir()
            template = root / 'эталон.PHD'
            write_synthetic_phd(template)
            write_synthetic_phd(spectra / 'QC_001.PHD', 1.15, 1.02)
            write_synthetic_phd(spectra / 'QC_002.phd', 0.9, 0.98)
            write_synthetic_phd(spectra / 'no_gamma.PHD', include_gamma=False)
            (spectra / 'broken.PHD').write_text('#Histogram\n256 256\n1 2 3\n', encoding='ascii')
            gamma_only = spectra / 'gamma_only.PHD'
            write_synthetic_phd(gamma_only)
            gamma_block = gamma_only.read_text(encoding='utf-8').split('#g_Spectrum', 1)[1]
            gamma_only.write_text('#Histogram\n256 256\n1 2 3\n#g_Spectrum' + gamma_block, encoding='utf-8')
            window.template_file = str(template)
            window.spectra_folder = str(spectra)
            window.check_ready()
            assert window.run_btn.isEnabled()
            assert '1.1' in window.windowTitle()
            cases = [(False, True, True, 28, 36), (True, False, True, 27, 37),
                     (True, True, False, 28, 36), (False, False, False, 28, 36)]
            for index, (use_irls, save_html, save_gamma, ba_start, ba_end) in enumerate(cases):
                output = root / f'case_{index}.xlsx'
                QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(output), 'Excel (*.xlsx)'))
                window.save_checkbox.setChecked(save_html)
                window.gamma_checkbox.setChecked(save_gamma)
                window.ba_start_spin.setValue(ba_start)
                window.ba_end_spin.setValue(ba_end)
                window.mle_checkbox.setChecked(use_irls)
                app.processEvents()
                window.run_analysis()
                deadline = time.monotonic() + 120
                thread = window.analysis_thread
                assert thread.use_poisson_mle == use_irls
                while thread.isRunning() and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.02)
                assert not thread.isRunning(), 'Analysis timed out'
                thread.wait(5000)
                app.processEvents()
                assert not errors, errors
                assert window.run_btn.isEnabled() and window.files_group.isEnabled()
                validate_workbook(output, use_irls, ba_start, ba_end)
                if save_html or save_gamma:
                    plots = Path(thread.plots_folder)
                    html_files = list(plots.glob('*.html'))
                    assert len(list(plots.glob('*_fitting.html'))) == 3 * save_html
                    assert len(list(plots.glob('*_gamma.html'))) == 3 * save_gamma
                    for path in html_files:
                        text = path.read_text(encoding='utf-8')
                        assert 'Plotly.newPlot' in text and path.stat().st_size > 1_000_000
                    assert plots.parent == output.parent
                else:
                    assert thread.plots_folder == ''
                report['modes'].append({'method': 'IRLS' if use_irls else 'fixed weights',
                                        'files': len(thread.results), 'xlsx': True,
                                        'beta_html': save_html, 'gamma_html': save_gamma,
                                        'ba_range': [ba_start, ba_end], 'ba_fwhm': True,
                                        'converged': thread.mle_converged if use_irls else None})
            assert len(messages) == 4
        report['ok'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        QFileDialog.getSaveFileName = original_save
        QMessageBox.information = original_information
        QMessageBox.critical = original_critical
        if not hasattr(window, 'analysis_thread') or not window.analysis_thread.isRunning():
            window.close()
        Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0 if report['ok'] else 1
