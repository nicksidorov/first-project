"""XeMat: сохранённая компоновка PyQt5, новое математическое ядро.

Запуск: python project.py. В первой версии обязательна подготовленная NPZ
модель. Три спектра можно подключить для исходных ROI-отсчётов и DETBK.
"""

from html import escape
from pathlib import Path
import sys

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QTextBrowser,
    QTextEdit, QVBoxLayout, QWidget,
)

from process_measurement import METHOD_NAMES, process_prepared_measurement
from results_saver import format_results, names_for_results, save_results_to_file


REGRESSION_METHODS = METHOD_NAMES


class ProcessingThread(QThread):
    """Расчёт в рабочем потоке; ошибок сохранения до выбора файла нет."""

    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_file, **options):
        super().__init__()
        self.model_file = model_file
        self.options = options

    def run(self):
        try:
            self.message.emit("Чтение данных и совместный расчёт SAMPLE/GASBK...")
            self.progress.emit(10)
            results = process_prepared_measurement(self.model_file, **self.options)
            self.progress.emit(100)
            self.result_ready.emit(results)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class FileSelector:
    """Исходная строка выбора файла: название, путь и кнопка."""

    def __init__(self, label_text, path_label, callback):
        self.layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(205)
        path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        path_label.setMinimumHeight(30)
        self.button = QPushButton("Выбрать...")
        self.button.setMinimumWidth(110)
        self.button.clicked.connect(callback)
        self.layout.addWidget(label)
        self.layout.addWidget(path_label, 1)
        self.layout.addWidget(self.button)


class MainWindow(QMainWindow):
    """Окно прежнего проекта с четырьмя новыми методами и моделью NPZ."""

    def __init__(self):
        super().__init__()
        self.model_file = None
        self.sample_file = None
        self.background_file = None
        self.detector_file = None
        self.results = None
        self.thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("XeMat — совместная обработка спектров")
        self.setGeometry(100, 100, 980, 880)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.file_group = QGroupBox("Выбор файлов")
        file_layout = QVBoxLayout()
        self.model_path_label = QLabel("Не выбрана")
        self.sample_path_label = QLabel("Не выбран — отсчёты из NPZ")
        self.background_path_label = QLabel("Не выбран — отсчёты из NPZ")
        self.detector_path_label = QLabel("Не выбран — фон из NPZ")
        for label, path_label, callback in (
            ("Готовая модель NPZ:", self.model_path_label, self.select_model_file),
            ("Файл пробы SAMPLE:", self.sample_path_label, self.select_sample_file),
            ("Газовый фон GASBK:", self.background_path_label, self.select_background_file),
            ("Фон детектора DETBK:", self.detector_path_label, self.select_detector_file),
        ):
            file_layout.addLayout(FileSelector(label, path_label, callback).layout)
        hint = QLabel(
            "Выберите NPZ с матрицей M и координатами параметров. Для расчёта по спектрам "
            "добавьте все три файла; порядок ROI задаётся в NPZ. Можно считать только "
            "по готовым векторам x,b из NPZ."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #667788; padding: 4px;")
        file_layout.addWidget(hint)
        clear_spectra = QPushButton("Использовать только готовые векторы из NPZ")
        clear_spectra.clicked.connect(self.clear_spectra)
        file_layout.addWidget(clear_spectra)
        self.file_group.setLayout(file_layout)
        layout.addWidget(self.file_group)

        self.method_group = QGroupBox("Метод расчёта")
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Выберите метод:"))
        self.method_combo = QComboBox()
        for method in ("nnls", "wls", "poisson", "poisson_nonnegative"):
            self.method_combo.addItem(METHOD_NAMES[method], method)
        self.method_combo.currentIndexChanged.connect(self.invalidate_results)
        method_layout.addWidget(self.method_combo, 1)
        self.method_group.setLayout(method_layout)
        layout.addWidget(self.method_group)

        self.checkbox_fixed_roi = QCheckBox("Использовать фиксированные ROI")
        self.checkbox_fixed_roi.setToolTip("Границы ROI из сохранённого constants.py; матрицы всё равно берутся из NPZ.")
        self.checkbox_fixed_roi.toggled.connect(self.invalidate_results)
        layout.addWidget(self.checkbox_fixed_roi)
        self.calculate_button = self.create_button("РАССЧИТАТЬ", "#3498db", self.calculate)
        self.calculate_button.setEnabled(False)
        layout.addWidget(self.calculate_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.message_label = QLabel("")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(self.message_label)

        results_group = QGroupBox("Результаты расчёта активностей")
        results_layout = QVBoxLayout()
        self.results_output = QTextBrowser()
        self.results_output.setMinimumHeight(270)
        self.results_output.setStyleSheet(
            "QTextBrowser { background-color: #f9f9f9; border: 1px solid #ddd; "
            "font-family: 'Courier New', monospace; font-size: 10pt; }"
        )
        self.results_output.setVisible(False)
        results_layout.addWidget(self.results_output)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, 1)
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Подробный протокол появится после расчёта...")
        self.text_output.setMaximumHeight(160)
        layout.addWidget(self.text_output)
        self.save_button = self.create_button("Сохранить результаты в файл", "#27ae60", self.save_results)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        self.statusBar().showMessage("Выберите готовую модель NPZ")

    def create_button(self, text, color, callback):
        button = QPushButton(text)
        button.setFixedHeight(40 if text == "РАССЧИТАТЬ" else 35)
        button.setStyleSheet(f"""
            QPushButton {{ background-color: {color}; color: white;
                border-radius: 5px; font-weight: {'bold' if text == 'РАССЧИТАТЬ' else 'normal'}; }}
            QPushButton:hover {{ background-color: {'#2980b9' if color == '#3498db' else '#219653'}; }}
            QPushButton:disabled {{ background-color: #bdc3c7; }}
        """)
        button.clicked.connect(callback)
        return button

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Готовая модель XeMat", "", "NumPy архив (*.npz)")
        if path:
            self.model_file = path
            self.model_path_label.setText(path)
            self.invalidate_results()

    def _select_spectrum(self, title, attribute, path_label):
        path, _ = QFileDialog.getOpenFileName(self, title, "", "Спектры IMS (*.ims *.PHD *.phd);;Все файлы (*)")
        if path:
            setattr(self, attribute, path)
            path_label.setText(path)
            self.invalidate_results()

    def select_sample_file(self):
        self._select_spectrum("Выберите SAMPLE", "sample_file", self.sample_path_label)

    def select_background_file(self):
        self._select_spectrum("Выберите предшествующий GASBK", "background_file", self.background_path_label)

    def select_detector_file(self):
        self._select_spectrum("Выберите DETBK", "detector_file", self.detector_path_label)

    def clear_spectra(self):
        self.sample_file = self.background_file = self.detector_file = None
        self.sample_path_label.setText("Не выбран — отсчёты из NPZ")
        self.background_path_label.setText("Не выбран — отсчёты из NPZ")
        self.detector_path_label.setText("Не выбран — фон из NPZ")
        self.invalidate_results()

    def invalidate_results(self, *_):
        self.results = None
        self.save_button.setEnabled(False)
        self.results_output.setVisible(False)
        self.text_output.clear()
        self.check_files_selected()

    def check_files_selected(self):
        spectra = (self.sample_file, self.background_file, self.detector_file)
        complete = all(spectra) or not any(spectra)
        paths = [path for path in (self.model_file, *spectra) if path]
        ready = bool(self.model_file) and complete and all(Path(path).is_file() for path in paths)
        self.calculate_button.setEnabled(ready)
        if not self.model_file:
            self.statusBar().showMessage("Выберите готовую модель NPZ")
        elif not complete:
            self.statusBar().showMessage("Выберите все три спектра или используйте только векторы NPZ")
        elif not ready:
            self.statusBar().showMessage("Один из выбранных файлов не найден")
        else:
            self.statusBar().showMessage("Данные выбраны. Можно начинать расчёт.")

    def _set_busy(self, busy):
        self.file_group.setEnabled(not busy)
        self.method_group.setEnabled(not busy)
        self.checkbox_fixed_roi.setEnabled(not busy)
        self.calculate_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy and self.results is not None)
        self.progress_bar.setVisible(busy)
        if not busy:
            self.check_files_selected()

    def calculate(self):
        self.check_files_selected()
        if not self.calculate_button.isEnabled():
            QMessageBox.warning(self, "Неполные данные", "Выберите NPZ и, при необходимости, все три спектра.")
            return
        self.results = None
        self.results_output.setVisible(False)
        self.text_output.clear()
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self.thread = ProcessingThread(
            self.model_file, sample_file=self.sample_file,
            background_file=self.background_file, detector_file=self.detector_file,
            use_fixed_roi=self.checkbox_fixed_roi.isChecked(), method=self.method_combo.currentData(),
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.message.connect(self.update_message)
        self.thread.result_ready.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()

    def update_message(self, message):
        self.message_label.setText(message)
        self.statusBar().showMessage(message)

    def on_processing_finished(self, results):
        self.results = results
        self._set_busy(False)
        self.results_output.setHtml(self.generate_activities_html(results))
        self.results_output.setVisible(True)
        self.text_output.setPlainText(format_results(results))
        self.update_message("Расчёт завершён. Результаты можно сохранить в файл.")

    def generate_activities_html(self, results):
        """Прежние цвета таблиц; единицы и ограничения определяет новая модель."""
        model, solution = results["model"], results["fit"]
        names = names_for_results(results)
        activity = results.get("activities")
        physical = solution.physical_parameters
        physical_std = (
            None if solution.physical_covariance is None else
            np.sqrt(np.maximum(np.diag(solution.physical_covariance), 0))
        )
        parts = [
            "<h3 style='text-align:center'>РЕЗУЛЬТАТЫ СОВМЕСТНОГО РАСЧЁТА</h3>",
            "<p style='background:#ecf0f1;padding:5px'>",
            f"<b>Метод:</b> {escape(METHOD_NAMES[solution.method])}<br>",
            f"<b>Координаты theta:</b> {escape(model.coordinates)}; ",
            "<b>физические координаты:</b> [a_c; gamma]</p>",
            "<table width='100%' style='border-collapse:collapse' border='1' cellpadding='7'>",
            "<tr style='background:#e0e0e0'><th>Компонент</th><th>a, мБк</th><th>σ(a), мБк</th>",
            "<th>a_c, мБк/м³</th><th>σ(a_c), мБк/м³</th></tr>",
        ]

        def number(value):
            return "—" if value is None else f"{1000 * value:.6g}"

        for index, name in enumerate(names):
            a = None if activity is None or activity.a is None else activity.a[index]
            sigma_a = None if activity is None or activity.std_A is None else activity.std_A[index]
            a_c = None if physical is None else physical[index]
            sigma_c = None if physical_std is None else physical_std[index]
            if activity is not None and activity.a_c is not None:
                a_c = activity.a_c[index]
                sigma_c = None if activity.std_A_c is None else activity.std_A_c[index]
            color = "#f0f8ff" if index % 2 == 0 else "#ffffff"
            display_name = name + " (вспом.)" if name.lower().replace("-", "").startswith("rn") else name
            parts.append(f"<tr style='background:{color}'><td>{escape(display_name)}</td>")
            parts.extend(f"<td align='right'>{number(value)}</td>" for value in (a, sigma_a, a_c, sigma_c))
            parts.append("</tr>")
        parts.append("</table>")
        if any(name.lower().replace("-", "").startswith("rn") for name in names):
            parts.append("<p>Радоновая компонента a_c — вспомогательный параметр модели; она не является объёмной активностью радона в атмосфере.</p>")
        if solution.covariance is None:
            parts.append("<p>Ковариация этой оценки недоступна; ±σ не выдаётся. Прочерк означает, что значение не рассчитано.</p>")
        detection = results.get("detection")
        if detection is not None:
            parts.extend([
                "<h4>ПОРОГ L_C И MDC, мБк/м³</h4>",
                "<table width='100%' style='border-collapse:collapse' border='1' cellpadding='7'>",
                "<tr style='background:#e0e0e0'><th>Компонент</th><th>Свободная a_c</th><th>L_C</th><th>MDC</th><th>Обнаружен</th></tr>",
            ])
            for index, component in enumerate(detection.component_indices):
                parts.append(f"<tr><td>{escape(names[component])}</td>")
                parts.append(f"<td align='right'>{number(detection.free_activity[index])}</td>")
                if detection.converged[index]:
                    parts.append(f"<td align='right'>{number(detection.L_C[index])}</td><td align='right'>{number(detection.MDC[index])}</td>")
                    parts.append(f"<td>{'Да' if detection.detected[index] else 'Нет'}</td></tr>")
                else:
                    parts.append("<td colspan='3'>Расчёт MDC не сошёлся</td></tr>")
            parts.append("</table>")
            parts.append("<p>Обнаружение проверяется по свободной оценке a_c &gt; L_C.</p>")
        if results.get("warnings"):
            parts.append("<p><b>Замечания:</b><br>" + "<br>".join(escape(str(value)) for value in results["warnings"]) + "</p>")
        return "".join(parts)

    def generate_output_text(self, results):
        return format_results(results)

    def on_processing_error(self, message):
        self._set_busy(False)
        self.update_message("Ошибка обработки")
        QMessageBox.critical(self, "Ошибка", message)

    def save_results(self):
        if self.results is None:
            return
        target, _ = QFileDialog.getSaveFileName(self, "Сохранить результаты", "analysis_results.txt", "Текстовые файлы (*.txt)")
        if target:
            try:
                save_results_to_file(self.results, target)
                self.update_message(f"Результаты сохранены: {target}")
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.update_message("Дождитесь завершения текущего расчёта перед закрытием окна.")
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
