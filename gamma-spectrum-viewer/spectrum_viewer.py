#!/usr/bin/env python3
"""Просмотр гамма-спектра: uint32 big-endian, без заголовка."""
import argparse
import csv
from pathlib import Path
import struct
import sys
import tempfile


def read_spectrum(filename):
    data = Path(filename).read_bytes()
    if not data:
        raise ValueError('Файл пуст.')
    if len(data) % 4:
        raise ValueError('Размер файла должен быть кратен 4 байтам (uint32).')
    return [value[0] for value in struct.iter_unpack('>I', data)]


def make_figure(counts, title, logarithmic=False):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=list(range(len(counts))), y=counts, mode='lines',
        line=dict(color='#2563eb', width=1.4, shape='hv'),
        hovertemplate='Канал: %{x}<br>Число импульсов: %{y}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=title), template='plotly_white',
        xaxis_title='Номер канала (с 0)', yaxis_title='Число импульсов',
        yaxis_type='log' if logarithmic else 'linear',
        margin=dict(l=75, r=30, t=65, b=65),
        dragmode='zoom', hovermode='closest',
    )
    if not logarithmic:
        fig.update_yaxes(rangemode='tozero')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs='?', help='Бинарный файл спектра')
    args = parser.parse_args()

    from PyQt5.QtCore import QUrl
    from PyQt5.QtWidgets import (
        QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel,
        QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
    )
    from PyQt5.QtWebEngineWidgets import QWebEngineView

    class SpectrumWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Гамма-спектр')
            self.resize(1100, 720)
            self.counts = None
            self.source = None
            # HTML вместе с Plotly хранится локально: интернет не требуется.
            self.temp = tempfile.TemporaryDirectory(prefix='gamma_spectrum_')
            self.html_path = Path(self.temp.name) / 'spectrum.html'
            self.figure = None
            central = QWidget()
            layout = QVBoxLayout(central)
            controls = QHBoxLayout()
            open_button = QPushButton('Открыть бинарный файл…')
            open_button.clicked.connect(self.choose_file)
            controls.addWidget(open_button)
            self.log_scale = QCheckBox('Логарифмическая шкала Y')
            self.log_scale.toggled.connect(self.redraw)
            controls.addWidget(self.log_scale)
            self.html_button = QPushButton('Сохранить HTML…')
            self.html_button.clicked.connect(self.save_html)
            controls.addWidget(self.html_button)
            self.csv_button = QPushButton('Сохранить CSV…')
            self.csv_button.clicked.connect(self.save_csv)
            controls.addWidget(self.csv_button)
            controls.addStretch()
            layout.addLayout(controls)
            self.info = QLabel('Выберите файл: uint32, big-endian, без заголовка.')
            self.info.setWordWrap(True)
            layout.addWidget(self.info)
            self.web = QWebEngineView()
            layout.addWidget(self.web, 1)
            self.setCentralWidget(central)
            self.html_button.setEnabled(False)
            self.csv_button.setEnabled(False)
            self.statusBar().showMessage(
                'Выделение мышью — увеличение; двойной щелчок — сброс масштаба. '
                'На логарифмической шкале нулевые значения не видны.')

        def choose_file(self):
            filename, _ = QFileDialog.getOpenFileName(
                self, 'Открыть спектр', '', 'Все файлы (*);;Бинарные файлы (*.bin *.dat *.txt)')
            if filename:
                self.load_file(filename)

        def load_file(self, filename):
            try:
                counts = read_spectrum(filename)
                # Сначала готовим график, чтобы при ошибке оставить прежние данные.
                figure = make_figure(counts, 'Гамма-спектр', self.log_scale.isChecked())
                figure.write_html(str(self.html_path), include_plotlyjs=True,
                                  config={'responsive': True, 'displaylogo': False})
            except Exception as error:
                QMessageBox.warning(self, 'Не удалось открыть спектр', str(error))
                return
            self.counts, self.source, self.figure = counts, Path(filename), figure
            self.setWindowTitle(f'Гамма-спектр — {self.source.name}')
            self.info.setText(
                f'{self.source.name} | Каналов: {len(counts)} | '
                f'Всего импульсов: {sum(counts):,} | '
                f'Максимум: {max(counts)} (канал {counts.index(max(counts))})')
            self.web.load(QUrl.fromLocalFile(str(self.html_path)))
            self.html_button.setEnabled(True)
            self.csv_button.setEnabled(True)

        def redraw(self, _checked=False):
            if self.counts is None:
                return
            try:
                self.figure = make_figure(self.counts, 'Гамма-спектр', self.log_scale.isChecked())
                self.figure.write_html(str(self.html_path), include_plotlyjs=True,
                                       config={'responsive': True, 'displaylogo': False})
                self.web.load(QUrl.fromLocalFile(str(self.html_path)))
            except Exception as error:
                QMessageBox.warning(self, 'Ошибка построения', str(error))

        def save_html(self):
            filename, _ = QFileDialog.getSaveFileName(
                self, 'Сохранить интерактивный график', 'spectrum.html', 'HTML (*.html)')
            if filename:
                try:
                    self.figure.write_html(filename, include_plotlyjs=True,
                                           config={'responsive': True, 'displaylogo': False})
                except Exception as error:
                    QMessageBox.warning(self, 'Ошибка сохранения', str(error))

        def save_csv(self):
            filename, _ = QFileDialog.getSaveFileName(
                self, 'Сохранить значения', 'spectrum.csv', 'CSV (*.csv)')
            if filename:
                try:
                    with open(filename, 'w', newline='', encoding='utf-8-sig') as output:
                        writer = csv.writer(output, delimiter=';')
                        writer.writerow(['channel', 'counts'])
                        writer.writerows(enumerate(self.counts))
                except Exception as error:
                    QMessageBox.warning(self, 'Ошибка сохранения', str(error))

    app = QApplication(sys.argv[:1])
    window = SpectrumWindow()
    window.show()
    if args.file:
        window.load_file(args.file)
    result = app.exec_()
    # Закрываем Chromium до удаления временного HTML.
    window.web.close()
    window.temp.cleanup()
    return result


if __name__ == '__main__':
    sys.exit(main())
