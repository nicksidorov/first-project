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


def write_synthetic_phd(path, amplitude=1.0, stretch=1.0, include_gamma=True):
    x = np.arange(256, dtype=float)
    u = x / stretch
    beta = amplitude / stretch * (1800 * np.exp(-0.5 * ((u - 85) / 20) ** 2)
                                  + 700 * np.exp(-0.5 * ((u - 155) / 24) ** 2))
    matrix = np.zeros((256, 256))
    matrix[200] = beta
    gamma = 5000 * np.exp(-0.5 * ((x - 233.4) / 3.8) ** 2) + 20 + 0.05 * x
    lines = ['#Acquisition', '2026-01-01 12:00:00', '#Histogram', '256 256']
    lines.extend(' '.join(f'{value:.8g}' for value in row) for row in matrix)
    if include_gamma:
        lines.extend(['#g_Spectrum', '256'])
        for start in range(0, 256, 5):
            lines.append(str(start) + ' ' + ' '.join(f'{v:.8g}' for v in gamma[start:start + 5]))
    lines.extend(['#End', 'Synthetic test data only'])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    return matrix, gamma


def validate_workbook(path, use_irls):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        assert set(workbook.sheetnames) == {'Успешные подгонки', 'Неудачные подгонки', 'Информация'}
        rows = list(workbook['Успешные подгонки'].values)
        assert len(rows) == 4, rows
        for row in rows[1:]:
            assert math.isfinite(row[2]) and row[2] > 0
            assert 0.85 < row[4] < 1.15
            assert row[9] in ('Да', 'Лимит итераций') if use_irls else row[9] == '—'
            if row[0] != 'no_gamma.PHD':
                assert abs(row[6] - 233.4) < 0.1
                assert row[8] > 0
            else:
                assert row[6] == 'Не удалось'
        failed = list(workbook['Неудачные подгонки'].values)
        assert len(failed) == 2 and failed[1][0] == 'broken.PHD'
        info = dict((row[0], row[1]) for row in workbook['Информация'].values if row[0])
        assert info['Всего обработано файлов:'] == 4
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
    report = {'ok': False, 'frozen': bool(getattr(sys, 'frozen', False)), 'modes': []}
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
            window.template_file = str(template)
            window.spectra_folder = str(spectra)
            window.check_ready()
            assert window.run_btn.isEnabled()
            for use_irls in (False, True):
                save_html = not use_irls
                output = root / ('IRLS.xlsx' if use_irls else 'fixed.xlsx')
                QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(output), 'Excel (*.xlsx)'))
                window.save_checkbox.setChecked(save_html)
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
                validate_workbook(output, use_irls)
                if save_html:
                    plots = Path(thread.plots_folder)
                    html_files = list(plots.glob('*.html'))
                    assert len(html_files) == 3
                    for path in html_files:
                        text = path.read_text(encoding='utf-8')
                        assert 'Plotly.newPlot' in text and path.stat().st_size > 1_000_000
                    assert plots.parent == output.parent
                else:
                    assert thread.plots_folder == ''
                report['modes'].append({'method': 'IRLS' if use_irls else 'fixed weights',
                                        'files': len(thread.results), 'xlsx': True,
                                        'html': save_html, 'converged': thread.mle_converged if use_irls else None})
            assert len(messages) == 2
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
