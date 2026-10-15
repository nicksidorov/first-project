"""XeMat: сохранённая компоновка PyQt5, новое математическое ядро.

Запуск: python project.py. Для трёх спектров доступны матрицы из constants.py;
подготовленная NPZ-модель остаётся отдельным вариантом входных данных.
"""

from html import escape
from pathlib import Path
import sys

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QTextBrowser,
    QTextEdit, QVBoxLayout, QWidget,
)

from process_measurement import METHOD_NAMES, process_prepared_measurement
from xemat.detectors import DetectorSelectionError
from xemat.monte_carlo_dialog import MonteCarloDialog, eligible_components
from results_saver import format_results, names_for_results, quality_for_results, save_results_to_file


REGRESSION_METHODS = METHOD_NAMES


class ProcessingThread(QThread):
    """Расчёт в рабочем потоке; ошибок сохранения до выбора файла нет."""

    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)
    warning = pyqtSignal(str)

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
        except DetectorSelectionError as exc:
            self.warning.emit(str(exc))
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


class RadonCalibrationDialog(QDialog):
    """Явный выбор прежнего приближения или трёх калиброванных H_Rn."""

    def __init__(self, mode, coefficients, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Радоновая компонента")
        layout = QVBoxLayout(self)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Прежняя модель: эффективная ε = 0,3", "legacy")
        self.mode_combo.addItem("Калиброванные коэффициенты H_Rn", "calibrated")
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(mode))
        layout.addWidget(self.mode_combo)
        note = QLabel(
            "Прежняя модель использует приближение из старой программы. "
            "Для калиброванной модели задайте H_Rn в Бк/отсчёт для соответствующих "
            "интервалов измерения. Радон остаётся вспомогательной компонентой."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.coefficient_group = QGroupBox("H_Rn, Бк/отсчёт")
        form = QFormLayout(self.coefficient_group)
        self.fields = {}
        for key, label in (
            ("sample", "Проба SAMPLE:"),
            ("gasbk", "Память в GASBK:"),
            ("sample_memory", "Память в SAMPLE:"),
        ):
            field = QLineEdit("" if not coefficients else str(coefficients[key]))
            field.setPlaceholderText("Например: 1.2e-4")
            self.fields[key] = field
            form.addRow(label, field)
        layout.addWidget(self.coefficient_group)
        self.mode_combo.currentIndexChanged.connect(self.update_mode)
        self.update_mode()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def update_mode(self, *_):
        self.coefficient_group.setEnabled(self.mode_combo.currentData() == "calibrated")

    def values(self):
        mode = self.mode_combo.currentData()
        if mode == "legacy":
            return mode, None
        coefficients = {}
        for key, field in self.fields.items():
            try:
                value = float(field.text().strip().replace(",", "."))
            except ValueError as exc:
                raise ValueError("Введите все три положительных коэффициента H_Rn.") from exc
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Все коэффициенты H_Rn должны быть конечными и больше нуля.")
            coefficients[key] = value
        return mode, coefficients

    def accept(self):
        try:
            self.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Калибровка радона", str(exc))
            return
        super().accept()


class MainWindow(QMainWindow):
    """Окно прежнего проекта: три спектра с константами или готовая модель NPZ."""

    def __init__(self):
        super().__init__()
        self.model_file = None
        self.sample_file = None
        self.background_file = None
        self.detector_file = None
        self.results = None
        self.thread = None
        self.monte_carlo_dialog = None
        self.radon_mode = "legacy"
        self.radon_h = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("XeMat — совместная обработка спектров")
        self.setGeometry(100, 100, 980, 880)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.file_group = QGroupBox("Выбор файлов")
        file_layout = QVBoxLayout()
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Источник матриц:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Матрицы из constants.py", "constants")
        self.source_combo.addItem("Готовая модель NPZ", "prepared")
        mode_layout.addWidget(self.source_combo, 1)
        self.radon_button = QPushButton("Калибровка радона...")
        self.radon_button.clicked.connect(self.configure_radon)
        mode_layout.addWidget(self.radon_button)
        file_layout.addLayout(mode_layout)
        self.model_path_label = QLabel("Не требуется для матриц из constants.py")
        self.sample_path_label = QLabel("Не выбран")
        self.background_path_label = QLabel("Не выбран")
        self.detector_path_label = QLabel("Не выбран")
        for label, path_label, callback in (
            ("Готовая модель NPZ:", self.model_path_label, self.select_model_file),
            ("Файл пробы SAMPLE:", self.sample_path_label, self.select_sample_file),
            ("Газовый фон GASBK:", self.background_path_label, self.select_background_file),
            ("Фон детектора DETBK:", self.detector_path_label, self.select_detector_file),
        ):
            selector = FileSelector(label, path_label, callback)
            if path_label is self.model_path_label:
                self.model_file_row = QWidget()
                selector.layout.setContentsMargins(0, 0, 0, 0)
                self.model_file_row.setLayout(selector.layout)
                file_layout.addWidget(self.model_file_row)
            else:
                file_layout.addLayout(selector.layout)
        self.source_hint = QLabel()
        self.source_hint.setWordWrap(True)
        self.source_hint.setStyleSheet("color: #667788; padding: 4px;")
        file_layout.addWidget(self.source_hint)
        self.clear_spectra_button = QPushButton("Использовать только готовые векторы из NPZ")
        self.clear_spectra_button.clicked.connect(self.clear_spectra)
        file_layout.addWidget(self.clear_spectra_button)
        self.file_group.setLayout(file_layout)
        layout.addWidget(self.file_group)

        self.method_group = QGroupBox("Метод расчёта")
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Выберите метод:"))
        self.method_combo = QComboBox()
        for method in ("nnls", "wls", "poisson", "poisson_nonnegative"):
            self.method_combo.addItem(METHOD_NAMES[method], method)
        self.method_combo.currentIndexChanged.connect(self.calculation_options_changed)
        method_layout.addWidget(self.method_combo, 1)
        self.method_group.setLayout(method_layout)
        layout.addWidget(self.method_group)

        self.checkbox_fixed_roi = QCheckBox("Использовать фиксированные ROI")
        self.checkbox_fixed_roi.setChecked(True)
        self.checkbox_fixed_roi.setToolTip("Границы ROI из FIXED_ROI_TABLE в constants.py. При отключении используются границы из спектров.")
        self.checkbox_fixed_roi.toggled.connect(self.calculation_options_changed)
        self.checkbox_fixed_efficiencies = QCheckBox("Фиксированные эффективности")
        self.checkbox_fixed_efficiencies.setToolTip(
            "Включено: ε ксенона из FIXED_EFFICIENCIES_SAMPLE/GASBK в constants.py "
            "для выбранного детектора. Выключено: ε из SAMPLE и GASBK. "
            "Для готовой модели NPZ не применяется."
        )
        self.checkbox_fixed_efficiencies.toggled.connect(self.calculation_options_changed)
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.checkbox_fixed_roi)
        options_layout.addWidget(self.checkbox_fixed_efficiencies)
        options_layout.addStretch()
        layout.addLayout(options_layout)
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
        self.text_output.setStyleSheet("font-family: 'Courier New', monospace; font-size: 10pt;")
        self.text_output.setLineWrapMode(QTextEdit.NoWrap)
        self.text_output.setPlaceholderText("Подробный протокол появится после расчёта...")
        self.text_output.setMaximumHeight(160)
        layout.addWidget(self.text_output)
        self.save_button = self.create_button("Сохранить результаты в файл", "#27ae60", self.save_results)
        self.save_button.setEnabled(False)
        self.monte_carlo_button = QPushButton("Монте-Карло: Lc / Ld…")
        self.monte_carlo_button.setMinimumHeight(35)
        self.monte_carlo_button.setEnabled(False)
        self.monte_carlo_button.setToolTip(
            "Отдельный эксперимент по последнему расчёту: ложные обнаружения и вероятность обнаружения при MDC."
        )
        self.monte_carlo_button.clicked.connect(self.open_monte_carlo)
        result_actions = QHBoxLayout()
        result_actions.addWidget(self.save_button, 1)
        result_actions.addWidget(self.monte_carlo_button)
        layout.addLayout(result_actions)
        self.source_combo.currentIndexChanged.connect(self.change_source)
        self.update_source_description()
        self.check_files_selected()

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
            self.source_combo.setCurrentIndex(self.source_combo.findData("prepared"))
            self.invalidate_results()

    def change_source(self, *_):
        self.invalidate_results()
        self.checkbox_fixed_roi.setChecked(self.source_combo.currentData() == "constants")
        self.update_source_description()

    def update_source_description(self):
        constants_mode = self.source_combo.currentData() == "constants"
        self.model_file_row.setVisible(not constants_mode)
        self.radon_button.setVisible(constants_mode)
        self.clear_spectra_button.setVisible(not constants_mode)
        self.checkbox_fixed_efficiencies.setEnabled(constants_mode)
        if constants_mode:
            radon = (
                "прежняя модель (ε = 0,3)" if self.radon_mode == "legacy" else
                "калиброванные коэффициенты H_Rn"
            )
            efficiencies = (
                "Эффективности SAMPLE/GASBK — фиксированные из constants.py для выбранного детектора. "
                if self.checkbox_fixed_efficiencies.isChecked() else
                "Эффективности — из файлов SAMPLE/GASBK. "
            )
            self.source_hint.setText(
                "R_s/R_g — из constants.py; времена и объём — из файлов. "
                + efficiencies + "Для памяти используется набор GASBK. Радон: " + radon + "."
            )
        else:
            self.source_hint.setText(
                "Выберите NPZ с матрицей M и координатами параметров. Для расчёта по спектрам "
                "добавьте все три файла; порядок ROI задаётся в NPZ. Можно считать только "
                "по готовым векторам x,b из NPZ."
            )
        if self.model_file is None:
            self.model_path_label.setText(
                "Не требуется для матриц из constants.py" if constants_mode else "Не выбрана"
            )
        for attribute, label, empty_npz in (
            ("sample_file", self.sample_path_label, "Не выбран — отсчёты из NPZ"),
            ("background_file", self.background_path_label, "Не выбран — отсчёты из NPZ"),
            ("detector_file", self.detector_path_label, "Не выбран — фон из NPZ"),
        ):
            if getattr(self, attribute) is None:
                label.setText("Не выбран" if constants_mode else empty_npz)

    def configure_radon(self):
        dialog = RadonCalibrationDialog(self.radon_mode, self.radon_h, self)
        if dialog.exec_() == QDialog.Accepted:
            mode, coefficients = dialog.values()
            if (mode, coefficients) != (self.radon_mode, self.radon_h):
                self.radon_mode, self.radon_h = mode, coefficients
                self.update_source_description()
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
        self.update_source_description()
        self.invalidate_results()

    def calculation_options_changed(self, *_):
        """Настройки относятся к следующему запуску; показан последний расчёт."""
        self.update_source_description()
        self.check_files_selected()
        if self.results is not None:
            changed = (
                self.method_combo.currentData() != self.results["fit"].method
                or self.checkbox_fixed_roi.isChecked() != self.results["use_fixed_roi"]
                or (
                    self.source_combo.currentData() == "constants"
                    and self.checkbox_fixed_efficiencies.isChecked() != self.results.get("use_fixed_efficiencies", False)
                )
            )
            self.update_message(
                "Настройки изменены. Показан предыдущий расчёт; нажмите «РАССЧИТАТЬ» для обновления."
                if changed else "Расчёт завершён. Результаты можно сохранить в файл."
            )

    def invalidate_results(self, *_):
        self.results = None
        self.save_button.setEnabled(False)
        self.monte_carlo_button.setEnabled(False)
        self.results_output.setVisible(False)
        self.text_output.clear()
        self.message_label.clear()
        self.check_files_selected()

    def check_files_selected(self):
        spectra = (self.sample_file, self.background_file, self.detector_file)
        constants_mode = self.source_combo.currentData() == "constants"
        complete = all(spectra) if constants_mode else all(spectra) or not any(spectra)
        model_ready = constants_mode or bool(self.model_file)
        input_paths = spectra if constants_mode else (self.model_file, *spectra)
        paths = [path for path in input_paths if path]
        ready = model_ready and complete and all(Path(path).is_file() for path in paths)
        self.calculate_button.setEnabled(ready)
        if constants_mode and not complete:
            self.statusBar().showMessage("Выберите три спектра: SAMPLE, предшествующий GASBK и DETBK")
        elif not model_ready:
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
        self.checkbox_fixed_efficiencies.setEnabled(not busy and self.source_combo.currentData() == "constants")
        self.calculate_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy and self.results is not None)
        self.monte_carlo_button.setEnabled(not busy and bool(eligible_components(self.results)))
        self.progress_bar.setVisible(busy)
        if not busy:
            self.check_files_selected()

    def calculate(self):
        self.check_files_selected()
        if not self.calculate_button.isEnabled():
            QMessageBox.warning(self, "Неполные данные", self.statusBar().currentMessage())
            return
        self.results = None
        self.results_output.setVisible(False)
        self.text_output.clear()
        self.progress_bar.setValue(0)
        self._set_busy(True)
        constants_mode = self.source_combo.currentData() == "constants"
        options = dict(
            sample_file=self.sample_file,
            background_file=self.background_file, detector_file=self.detector_file,
            use_fixed_roi=self.checkbox_fixed_roi.isChecked(), method=self.method_combo.currentData(),
        )
        if constants_mode:
            options.update(
                radon_mode=self.radon_mode, radon_h=self.radon_h,
                use_fixed_efficiencies=self.checkbox_fixed_efficiencies.isChecked(),
            )
        self.thread = ProcessingThread(None if constants_mode else self.model_file, **options)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.message.connect(self.update_message)
        self.thread.result_ready.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.warning.connect(self.on_processing_warning)
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
        """Две таблицы как в старом окне; ±σ берётся из нового ядра."""
        model, solution = results["model"], results["fit"]
        names = names_for_results(results)
        activity = results.get("activities")
        detection = results.get("detection")
        detected_components = set() if detection is None else {
            int(component) for index, component in enumerate(detection.component_indices)
            if detection.converged[index] and detection.detected[index]
        }
        physical = solution.physical_parameters
        physical_std = (
            None if solution.physical_covariance is None else
            np.sqrt(np.maximum(np.diag(solution.physical_covariance), 0))
        )
        quality = quality_for_results(results)
        reduced = "—" if quality.reduced_statistic is None else f"{quality.reduced_statistic:.5g}"
        quality_note = (
            "Для ограниченного метода ν номинальное; сравнение с 1 условно."
            if quality.nominal_dof else "Около 1 — ориентир при достаточных отсчётах."
        )
        parts = [
            "<h3 style='text-align:center'>РЕЗУЛЬТАТЫ СОВМЕСТНОГО РАСЧЁТА</h3>",
            "<p style='background:#ecf0f1;padding:5px'>",
            f"<b>Метод:</b> {escape(METHOD_NAMES[solution.method])}<br>",
            (
                f"<b>Детектор:</b> {results['metadata']['input_detector_id']}<br>"
                if results.get("metadata", {}).get("input_detector_id") is not None else ""
            ),
            "</p>",
        ]
        details = [
            "<p style='background:#ecf0f1;padding:5px'>",
            (
                "<b>ROI:</b> " + ("фиксированные" if results["use_fixed_roi"] else "из файлов спектров") + "<br>"
                if results.get("sample_file") else ""
            ),
            (
                "<b>Эффективности:</b> " + (
                    "фиксированные, constants.py" if results.get("use_fixed_efficiencies") else "из SAMPLE и GASBK"
                ) + "<br>"
                if results.get("calibration_mode") == "constants" else ""
            ),
            f"<b>Координаты theta:</b> {escape(model.coordinates)}; ",
            "<b>физические координаты:</b> [a_c; gamma]<br>",
            f"<b>{escape(quality.label)}: {reduced}</b>; ν = {quality.degrees_of_freedom}",
            " (номинальное)" if quality.nominal_dof else "",
            f"<br>{quality_note}</p>",
        ]
        if quality.degrees_of_freedom <= 0:
            details.append("<p>Приведённая статистика не определена: нет остаточных степеней свободы (ν≤0).</p>")
        elif quality.warnings and not quality.nominal_dof:
            details.append("<p>" + "<br>".join(escape(note) for note in quality.warnings) + "</p>")

        def number(value):
            return "—" if value is None or not np.isfinite(value) else f"{1000 * value:.6g}"

        def relative_error(value, sigma):
            # Относительная σ не зависит от перевода Бк → мБк.
            # Для отрицательной свободной оценки используем модуль;
            # при нуле или отсутствии σ относительная погрешность не определена.
            if value is None or sigma is None or value == 0:
                return "—"
            if not np.isfinite(value) or not np.isfinite(sigma):
                return "—"
            percent = 100 * (float(sigma) / abs(float(value)))
            return f"{percent:.1f}%" if np.isfinite(percent) else "—"

        detector_rows, concentration_rows = [], []
        for index, name in enumerate(names):
            a = None if activity is None or activity.a is None else activity.a[index]
            sigma_a = None if activity is None or activity.std_A is None else activity.std_A[index]
            a_c = None if physical is None else physical[index]
            sigma_c = None if physical_std is None else physical_std[index]
            if activity is not None and activity.a_c is not None:
                a_c = activity.a_c[index]
                sigma_c = None if activity.std_A_c is None else activity.std_A_c[index]
            detector_rows.append((a, sigma_a))
            concentration_rows.append((a_c, sigma_c))

        def activity_table(title, rows):
            table = [
                f"<h4>{title}</h4>",
                "<table width='100%' style='border-collapse:collapse' border='1' cellspacing='0' cellpadding='5'>",
                "<tr style='background:#e0e0e0'><th align='left'>Изотоп</th>",
                "<th>Значение</th><th>±σ</th><th>Отн.<br>погр., %</th></tr>",
            ]
            for index, (name, (value, sigma)) in enumerate(zip(names, rows)):
                color = "#fde2e2" if index in detected_components else (
                    "#f0f8ff" if index % 2 == 0 else "#ffffff"
                )
                display_name = "Rn 222" if name.lower().replace("-", "").replace(" ", "") == "rn222" else name
                table.append(f"<tr style='background:{color}'><td>{escape(display_name)}</td>")
                table.extend(f"<td align='right'>{cell}</td>" for cell in
                             (number(value), number(sigma), relative_error(value, sigma)))
                table.append("</tr>")
            table.append("</table>")
            return "".join(table)

        # QTextBrowser поддерживает вложенные таблицы; явная ячейка между
        # ними обеспечивает промежуток без браузерных flex/grid-стилей.
        parts.extend([
            "<table width='100%' border='0' cellspacing='0' cellpadding='0'><tr>",
            "<td width='49%' valign='top'>",
            activity_table("1. АКТИВНОСТЬ В ДЕТЕКТОРЕ (мБк)", detector_rows),
            "</td><td width='2%'>&nbsp;</td><td width='49%' valign='top'>",
            activity_table("2. ОБЪЁМНАЯ АКТИВНОСТЬ (мБк/м³)", concentration_rows),
            "</td></tr></table>",
            *details,
            "<p>Отн. погр. = 100 × σ / |значение|. При нулевом значении или отсутствии σ — прочерк.</p>",
        ])
        if detection is not None:
            parts.append("<p>Красноватой заливкой отмечено обнаружение: свободная оценка a_c &gt; L_C.</p>")
        if any(name.lower().replace("-", "").startswith("rn") for name in names):
            parts.append("<p>Радоновая компонента a_c — вспомогательный параметр модели; она не является объёмной активностью радона в атмосфере.</p>")
        if solution.covariance is None:
            parts.append("<p>Ковариация этой оценки недоступна; ±σ не выдаётся. Прочерк означает, что значение не рассчитано.</p>")
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

    def on_processing_warning(self, message):
        self.results = None
        self.results_output.clear()
        self.results_output.setVisible(False)
        self.text_output.clear()
        self._set_busy(False)
        self.update_message("Проверьте номера детекторов в выбранных файлах")
        QMessageBox.warning(self, "Проверьте детекторы", message)

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

    def open_monte_carlo(self):
        if not eligible_components(self.results) or (self.thread is not None and self.thread.isRunning()):
            return
        if self.monte_carlo_dialog is None or self.monte_carlo_dialog.source_results is not self.results:
            if self.monte_carlo_dialog is not None:
                self.monte_carlo_dialog.deleteLater()
            self.monte_carlo_dialog = MonteCarloDialog(self.results, self)
        self.monte_carlo_dialog.exec_()

    def closeEvent(self, event):
        if self.monte_carlo_dialog is not None and self.monte_carlo_dialog.is_running:
            self.monte_carlo_dialog.request_close()
            self.update_message("Эксперимент Монте-Карло останавливается. Закройте окно после завершения текущего повтора.")
            event.ignore()
        elif self.thread is not None and self.thread.isRunning():
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
