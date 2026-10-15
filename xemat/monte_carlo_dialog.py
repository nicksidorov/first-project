"""Отдельное окно эксперимента; основной расчёт спектров не изменяется."""

from html import escape
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTextBrowser,
    QVBoxLayout,
)

from results_saver import names_for_results
from .io import save_json
from .monte_carlo import run_from_results, save_monte_carlo_report


def eligible_components(results):
    """Предлагаем только компоненты с конечными рассчитанными LC и MDC."""
    if results is None or results.get("detection") is None:
        return []
    if results.get("fit") is None or not results["fit"].success:
        return []
    detection = results["detection"]
    names = names_for_results(results)
    return [
        int(component)
        for index, component in enumerate(detection.component_indices)
        if detection.converged[index]
        and not names[int(component)].lower().replace("-", "").replace(" ", "").startswith("rn")
        and np.isfinite(detection.L_C[index]) and detection.L_C[index] >= 0
        and np.isfinite(detection.MDC[index]) and detection.MDC[index] >= 0
    ]


class MonteCarloThread(QThread):
    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, results, *, component_index, n_trials, seed, method, parent=None):
        super().__init__(parent)
        self.results = results
        self.options = dict(component_index=component_index, n_trials=n_trials,
                            seed=seed, method=method)

    def run(self):
        try:
            result = run_from_results(
                self.results, **self.options, progress=self.progress.emit,
                should_cancel=self.isInterruptionRequested,
            )
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class MonteCarloDialog(QDialog):
    """Пуассоновские реализации при A=0 и A=MDC исходного расчёта."""

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.source_results = results
        self.experiment_result = None
        self.worker = None
        self._running = False
        self._close_when_finished = False
        self._worker_error = None
        self.setWindowTitle("XeMat — Монте-Карло: Lc / Ld")
        self.setModal(True)
        self.resize(940, 760)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Эксперимент по последнему расчёту спектров: пуассоновские отсчёты "
            "при A = 0 и A = MDC (Ld) исходного расчёта. Проверяем частоту ложных обнаружений "
            "и вероятность обнаружения. Активности здесь — объёмные."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.settings_group = QGroupBox("Параметры эксперимента")
        form = QFormLayout(self.settings_group)
        self.isotope_combo = QComboBox()
        names = names_for_results(results)
        for index in eligible_components(results):
            self.isotope_combo.addItem(names[index], index)
        form.addRow("Изотоп:", self.isotope_combo)
        self.method_combo = QComboBox()
        self.method_combo.addItem("МНК / NNLS", "wls")
        self.method_combo.addItem("Пуассон / неотрицательный Пуассон", "poisson")
        method = "poisson" if results["fit"].method.startswith("poisson") else "wls"
        self.method_combo.setCurrentIndex(self.method_combo.findData(method))
        form.addRow("Методы оценивания:", self.method_combo)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 1_000_000)
        self.trials_spin.setValue(1000)
        self.trials_spin.setGroupSeparatorShown(True)
        form.addRow("Повторов для каждого из двух сценариев:", self.trials_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(12345)
        self.seed_spin.setToolTip("Начальное число генератора для воспроизводимого эксперимента.")
        form.addRow("Seed генератора:", self.seed_spin)
        layout.addWidget(self.settings_group)

        assumption = QLabel(
            "В каждом повторе заново оцениваются все параметры и Lc; обнаружение: "
            "свободная оценка a_c > Lc. Для генерации фиксируются матрица M, фон b "
            "и опорные сопутствующие активности. Их истинная неопределённость и "
            "систематические погрешности этим экспериментом не моделируются. "
            "MDC для второго сценария фиксируется по исходному расчёту. При смене пары методов "
            "и том же seed моделируются те же спектры."
        )
        assumption.setWordWrap(True)
        assumption.setStyleSheet("color: #566573;")
        layout.addWidget(assumption)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Запустить эксперимент")
        self.start_button.clicked.connect(self.start_experiment)
        self.cancel_button = QPushButton("Остановить")
        self.cancel_button.clicked.connect(self.request_cancel)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 2000)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Выберите изотоп и запустите эксперимент.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.results_output = QTextBrowser()
        self.results_output.setOpenExternalLinks(False)
        layout.addWidget(self.results_output, 1)
        footer = QHBoxLayout()
        self.save_button = QPushButton("Сохранить эксперимент…")
        self.save_button.clicked.connect(self.save_results)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.reject)
        footer.addWidget(self.save_button)
        footer.addStretch()
        footer.addWidget(self.close_button)
        layout.addLayout(footer)
        self._set_running(False)
        if self.isotope_combo.count() == 0:
            self.status_label.setText("Нет изотопов с успешно рассчитанными Lc и MDC.")

    @property
    def is_running(self):
        return self._running

    def _set_running(self, running):
        self._running = running
        self.settings_group.setEnabled(not running)
        self.start_button.setEnabled(not running and self.isotope_combo.count() > 0)
        self.cancel_button.setEnabled(running)
        self.save_button.setEnabled(not running and self.experiment_result is not None)

    def start_experiment(self):
        if self._running or self.isotope_combo.currentData() is None:
            return
        self.experiment_result = None
        self._worker_error = None
        self._close_when_finished = False
        self.results_output.clear()
        self.progress_bar.setRange(0, 2 * self.trials_spin.value())
        self.progress_bar.setValue(0)
        self.status_label.setText("Подготовка опорной модели и расчёт MDC…")
        self._set_running(True)
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = MonteCarloThread(
            self.source_results, component_index=int(self.isotope_combo.currentData()),
            n_trials=self.trials_spin.value(), seed=self.seed_spin.value(),
            method=self.method_combo.currentData(), parent=self,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        # Освобождаем управление только после finished: result_ready ещё не
        # означает, что QThread завершён и его безопасно заменить или закрыть.
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, done, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        if not self.worker.isInterruptionRequested():
            self.status_label.setText(f"Выполнено повторов: {done} из {total}.")

    def _on_result(self, result):
        self.experiment_result = result
        self.results_output.setHtml(self._result_html(result))

    def _on_error(self, message):
        self._worker_error = message
        self.status_label.setText("Ошибка эксперимента: " + message)

    def _on_finished(self):
        self._set_running(False)
        if self.experiment_result is not None:
            self.status_label.setText(
                "Эксперимент остановлен. Частичные результаты можно сохранить."
                if self.experiment_result.get("cancelled") else
                "Эксперимент завершён. Результаты можно сохранить."
            )
        elif not self._worker_error:
            self.status_label.setText("Эксперимент завершён без результата.")
        if self._close_when_finished:
            self._close_when_finished = False
            super().reject()

    def request_cancel(self):
        if self._running:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Остановка после текущего повтора; результаты будут сохранены в окне…")

    def request_close(self):
        if self._running:
            self._close_when_finished = True
            self.request_cancel()
        else:
            super().reject()

    def reject(self):
        self.request_close()

    def closeEvent(self, event):
        if self._running:
            event.ignore()
            self.request_close()
        else:
            event.accept()

    @staticmethod
    def _result_html(result):
        def number(value):
            return "—" if value is None or not np.isfinite(value) else f"{value:.6g}"

        def percent(value):
            return "—" if value is None else f"{100 * value:.2f}%"

        def interval(value):
            return "—" if value is None else f"{percent(value[0])} … {percent(value[1])}"

        method = "МНК / NNLS" if result["method"] == "wls" else "Пуассон / неотрицательный Пуассон"
        parts = [
            f"<h3>{escape(str(result['component_name']))}: {method}</h3>",
            f"<p>Опорные Lc = {number(1000 * result['reference_L_C'])} мБк/м³; "
            f"MDC (Ld) = {number(1000 * result['reference_MDC'])} мБк/м³.<br>",
            f"Повторов на сценарий: {result['requested_trials_per_scenario']}; seed: {result['seed']}.</p>",
            "<table width='100%' border='1' cellspacing='0' cellpadding='6'>",
            "<tr style='background:#e0e0e0'><th>Сценарий</th><th>Обнаружения / успешные</th>"
            "<th>Частота</th><th>95% интервал</th><th>Целевой уровень</th></tr>",
        ]
        for scenario in result["scenarios"]:
            is_null = scenario["label"] == "A=0"
            expected = result["alpha"] if is_null else 1 - result["beta_ii"]
            target_label = "α" if is_null else "1 − β"
            parts.extend([
                "<tr><td>" + ("A = 0: ложные обнаружения" if is_null else "A = MDC: обнаружения") + "</td>",
                f"<td align='right'>{scenario['n_detected']} / {scenario['n_valid']}</td>",
                f"<td align='right'>{percent(scenario['detection_rate'])}</td>",
                f"<td>{interval(scenario['confidence_interval'])}</td>",
                f"<td>{target_label} = {percent(expected)}</td></tr>",
            ])
        parts.append("</table><p>95% интервал — точный биномиальный интервал по успешным повторам. "
                     "При неуспешных повторах частота условна; их влияние показано ниже. "
                     "Один эксперимент не доказывает соблюдение уровня для всех спектров.</p>")
        for scenario in result["scenarios"]:
            parts.extend([
                f"<h4>{escape(scenario['label'])}; A = {number(1000 * scenario['activity'])} мБк/м³</h4>",
                f"<p>Попыток: {scenario['n_attempted']} / {scenario['n_requested']}; "
                f"успешно: {scenario['n_valid']}; <b>неуспешно: {scenario['n_failed']}</b>.<br>",
                "Границы частоты, если все неуспешные повторы считать сначала необнаружениями, "
                f"затем обнаружениями: {interval(scenario['failure_probability_bounds'])}.<br>",
                f"Среднее Lc: {number(None if scenario['mean_L_C'] is None else 1000 * scenario['mean_L_C'])} мБк/м³; ",
                f"средняя свободная оценка: {number(None if scenario['mean_free_activity'] is None else 1000 * scenario['mean_free_activity'])} мБк/м³; ",
                f"её стандартное отклонение: {number(None if scenario['std_free_activity'] is None else 1000 * scenario['std_free_activity'])} мБк/м³.</p>",
            ])
            for reason, count in scenario.get("failure_reasons", {}).items():
                parts.append(f"<p>Неуспешные повторы ({count}): {escape(str(reason))}</p>")
        if result.get("cancelled"):
            parts.append("<p><b>Эксперимент остановлен: результаты частичные.</b></p>")
        for warning in result.get("warnings", []):
            parts.append(f"<p>{escape(str(warning))}</p>")
        return "".join(parts)

    def save_results(self):
        if self._running or self.experiment_result is None:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Сохранить эксперимент Монте-Карло", "monte_carlo_results.txt",
            "Текстовый отчёт (*.txt);;Данные эксперимента JSON (*.json)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".json" if "*.json" in selected_filter else ".txt")
        try:
            if target.suffix.lower() == ".json":
                save_json(self.experiment_result, target)
            else:
                save_monte_carlo_report(self.experiment_result, target)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return
        self.status_label.setText(f"Эксперимент сохранён: {target}")
