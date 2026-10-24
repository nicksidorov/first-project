"""Окно распределений свободных оценок; исходный расчёт остаётся неизменным."""

from html import escape
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QTabWidget, QTextBrowser,
    QVBoxLayout,
)

from .distribution_mc import eligible_components, run_from_results, save_distribution_report
from .distribution_plot import render_distribution_figure
from .distribution_pairs import render_activity_pair_figure
from .io import save_json


class _ScrollableFigureCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event):
        # Здесь колесо прокручивает большую матрицу графиков в QScrollArea.
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.wheelEvent(event)
                return
            parent = parent.parentWidget()
        event.ignore()


class DistributionThread(QThread):
    progress = pyqtSignal(int, int)
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, results, *, generation, method, n_trials, seed, parent=None):
        super().__init__(parent)
        self.results = results
        self.options = dict(generation=generation, method=method,
                            n_trials=n_trials, seed=seed)

    def run(self):
        try:
            result = run_from_results(
                self.results, **self.options, progress=self.progress.emit,
                should_cancel=self.isInterruptionRequested,
            )
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class DistributionDialog(QDialog):
    """Гистограммы объёмных активностей после повторной обработки спектров."""

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.source_results = results
        self.experiment_result = None
        self.worker = None
        self._running = False
        self._worker_error = None
        self._close_when_finished = False
        self._pairs_dirty = True
        self._pairs_plot_error = None
        self.setWindowTitle("XeMat — Монте-Карло: гистограммы активностей")
        self.setModal(True)
        self.resize(1160, 900)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Распределения свободных оценок объёмной активности. Отрицательные значения сохраняются."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.settings_group = QGroupBox("Параметры эксперимента")
        grid = QGridLayout(self.settings_group)
        self.generation_combo = QComboBox()
        self.generation_combo.addItem("По ожидаемому спектру: Poisson(μ)", "fitted")
        self.generation_combo.addItem("По наблюдаемым отсчётам: Poisson(x)", "observed")
        grid.addWidget(QLabel("Генерация:"), 0, 0)
        grid.addWidget(self.generation_combo, 0, 1, 1, 3)
        self.method_combo = QComboBox()
        self.method_combo.addItem("Свободный МНК", "wls")
        self.method_combo.addItem("Свободный Пуассон (ММП)", "poisson")
        method = "poisson" if results["fit"].method.startswith("poisson") else "wls"
        self.method_combo.setCurrentIndex(self.method_combo.findData(method))
        grid.addWidget(QLabel("Оценивание:"), 0, 4)
        grid.addWidget(self.method_combo, 0, 5)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(1, 1_000_000)
        self.trials_spin.setValue(1000)
        self.trials_spin.setGroupSeparatorShown(True)
        grid.addWidget(QLabel("Повторов:"), 1, 0)
        grid.addWidget(self.trials_spin, 1, 1)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(12345)
        self.seed_spin.setToolTip(
            "При одном источнике средних и одинаковом seed методы получают одинаковые спектры."
        )
        grid.addWidget(QLabel("Seed:"), 1, 2)
        grid.addWidget(self.seed_spin, 1, 3)
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 200)
        self.bins_spin.setValue(30)
        self.bins_spin.setToolTip("Меняет только вид гистограмм, без повторного моделирования.")
        grid.addWidget(QLabel("Интервалов гистограммы:"), 1, 4)
        grid.addWidget(self.bins_spin, 1, 5)
        layout.addWidget(self.settings_group)
        self.assumption_label = QLabel()
        self.assumption_label.setWordWrap(True)
        self.assumption_label.setStyleSheet("color: #566573;")
        layout.addWidget(self.assumption_label)
        self.generation_combo.currentIndexChanged.connect(self._update_assumption)
        self.bins_spin.valueChanged.connect(self.redraw_plot)
        self._update_assumption()

        controls = QHBoxLayout()
        self.start_button = QPushButton("Построить гистограммы")
        self.start_button.clicked.connect(self.start_experiment)
        self.cancel_button = QPushButton("Остановить")
        self.cancel_button.clicked.connect(self.request_cancel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        controls.addWidget(self.start_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.progress_bar, 1)
        layout.addLayout(controls)
        self.status_label = QLabel("Выберите параметры и запустите эксперимент.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        for control in (self.generation_combo, self.method_combo):
            control.currentIndexChanged.connect(self._mark_settings_changed)
        for control in (self.trials_spin, self.seed_spin):
            control.valueChanged.connect(self._mark_settings_changed)

        self.figure = Figure(figsize=(10, 5.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(300)
        # Графикам нужна высота для двух рядов осей. Подробная таблица
        # получает отдельную вкладку, а не отнимает место у гистограмм.
        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self.canvas, "Гистограммы")
        self.pairs_figure = Figure(figsize=(9, 9))
        self.pairs_canvas = _ScrollableFigureCanvas(self.pairs_figure)
        self.pairs_canvas.setMinimumSize(800, 800)
        self.pairs_scroll = QScrollArea()
        self.pairs_scroll.setWidgetResizable(True)
        self.pairs_scroll.setWidget(self.pairs_canvas)
        self.result_tabs.addTab(self.pairs_scroll, "Корреляции активностей")
        self.results_output = QTextBrowser()
        self.results_output.setOpenExternalLinks(False)
        self.result_tabs.addTab(self.results_output, "Статистика и пояснения")
        layout.addWidget(self.result_tabs, 1)
        footer = QHBoxLayout()
        self.save_button = QPushButton("Сохранить результаты…")
        self.save_button.clicked.connect(self.save_results)
        self.save_plot_button = QPushButton("Сохранить графики PNG…")
        self.save_plot_button.clicked.connect(self.save_plot)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.reject)
        footer.addWidget(self.save_button)
        footer.addWidget(self.save_plot_button)
        footer.addStretch()
        footer.addWidget(self.close_button)
        layout.addLayout(footer)
        self.result_tabs.currentChanged.connect(self._on_tab_changed)
        self._set_running(False)
        if not eligible_components(results):
            self.status_label.setText("Нет доступных компонентов ксенона для эксперимента.")

    @property
    def is_running(self):
        return self._running

    def _update_assumption(self):
        generation = self.generation_combo.currentData()
        text = (
            "Средние μ получаем один раз неотрицательным ММП, независимо от метода повторов. "
            if generation == "fitted" else
            "Средние равны x: если в ROI наблюдался ноль, во всех повторах там будет ноль. "
            "Единой заданной активности нет. "
        )
        if self.source_results["model"].has_detector_background:
            text += "Разыгрываются SAMPLE, GASBK и DETBK. "
        else:
            text += "Заданный фон b фиксирован. "
        text += "Калибровка фиксирована; систематика не моделируется."
        self.assumption_label.setText(text)
        self.assumption_label.setToolTip(
            "Эксперимент использует последний расчёт спектров. При генерации по модели μ "
            "заданы одной совместной неотрицательной ММП-подгонкой. В каждом повторе "
            "заново оцениваются все параметры свободным методом; для V86 — также фон DETBK. "
            "Матрицы, времена, объём и калибровка фиксированы. Сравниваются только статистические σ. "
            "Изменение параметров относится к следующему запуску."
        )

    def _mark_settings_changed(self, *_):
        if self.experiment_result is not None and not self._running:
            self.status_label.setText("Настройки изменены. Показан предыдущий результат; запустите эксперимент для обновления.")

    def _set_running(self, running):
        self._running = running
        self.settings_group.setEnabled(not running)
        self.start_button.setEnabled(not running and bool(eligible_components(self.source_results)))
        self.cancel_button.setEnabled(running)
        ready = not running and self.experiment_result is not None
        self.save_button.setEnabled(ready)
        self._update_plot_export()

    def _update_plot_export(self):
        active = self.result_tabs.currentWidget()
        ready = not self._running and self.experiment_result is not None
        if active is self.canvas:
            self.save_plot_button.setText("Сохранить гистограммы PNG…")
        elif active is self.pairs_scroll:
            self.save_plot_button.setText("Сохранить корреляции PNG…")
            ready = ready and self._pairs_plot_error is None
        else:
            self.save_plot_button.setText("Сохранить график PNG…")
            ready = False
        self.save_plot_button.setEnabled(ready)

    def _on_tab_changed(self, *_):
        if self.result_tabs.currentWidget() is self.pairs_scroll:
            try:
                self._ensure_pair_plot()
            except Exception as exc:
                self._pairs_plot_error = f"Ошибка графика корреляций: {type(exc).__name__}: {exc}"
                self.status_label.setText(self._pairs_plot_error)
        self._update_plot_export()

    def _ensure_pair_plot(self):
        if self.experiment_result is None or not self._pairs_dirty:
            return
        # Все пары берутся из тех же строк сохранённой выборки; переключение
        # вкладки не запускает новых генераций или подгонок.
        try:
            render_activity_pair_figure(self.experiment_result, self.pairs_figure, bins=self.bins_spin.value())
        except Exception as exc:
            self._pairs_plot_error = f"Ошибка графика корреляций: {type(exc).__name__}: {exc}"
            raise
        self.pairs_canvas.draw_idle()
        self._pairs_dirty = False
        self._pairs_plot_error = None

    def start_experiment(self):
        if self._running or not eligible_components(self.source_results):
            return
        self.experiment_result = None
        self._worker_error = None
        self._close_when_finished = False
        self.results_output.clear()
        self.figure.clear()
        self.canvas.draw_idle()
        self.pairs_figure.clear()
        self.pairs_canvas.draw_idle()
        self._pairs_dirty = True
        self._pairs_plot_error = None
        self._update_assumption()
        self.progress_bar.setRange(0, self.trials_spin.value())
        self.progress_bar.setValue(0)
        self.status_label.setText(
            "Подготовка средних совместным неотрицательным ММП…"
            if self.generation_combo.currentData() == "fitted" else
            "Подготовка исходного вектора наблюдений…"
        )
        self._set_running(True)
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = DistributionThread(
            self.source_results, generation=self.generation_combo.currentData(),
            method=self.method_combo.currentData(), n_trials=self.trials_spin.value(),
            seed=self.seed_spin.value(), parent=self,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        # Сигнал результата приходит до завершения QThread; закрываем окно
        # и разрешаем новый запуск только после finished.
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
        try:
            self.redraw_plot()
        except Exception as exc:
            # Численные результаты остаются доступны для сохранения даже
            # при ошибке отрисовки или отсутствии данных для гистограммы.
            self._worker_error = f"Ошибка графика: {type(exc).__name__}: {exc}"
        self._update_assumption()

    def _on_error(self, message):
        self._worker_error = message
        self.status_label.setText("Ошибка эксперимента: " + message)

    def _on_finished(self):
        self._set_running(False)
        if self._worker_error:
            self.status_label.setText(self._worker_error)
        elif self.experiment_result is not None:
            self.status_label.setText(
                "Эксперимент остановлен. Частичные результаты можно сохранить."
                if self.experiment_result.get("cancelled") else
                "Эксперимент завершён. Можно менять число интервалов и сохранять результаты."
            )
        else:
            self.status_label.setText("Эксперимент завершён без результата.")
        if self._close_when_finished:
            self._close_when_finished = False
            super().reject()

    def redraw_plot(self, *_):
        self._pairs_dirty = True
        self._pairs_plot_error = None
        if self.experiment_result is not None:
            render_distribution_figure(self.experiment_result, self.figure, bins=self.bins_spin.value())
            self.canvas.draw_idle()
            if self.result_tabs.currentWidget() is self.pairs_scroll:
                self._ensure_pair_plot()
        self._update_plot_export()

    def request_cancel(self):
        if self._running:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Остановка после текущего расчёта; частичные результаты останутся в окне…")

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
        def activity(value):
            return "—" if value is None or not np.isfinite(value) else f"{1000 * value:.5g}"

        method = "свободный МНК" if result["method"] == "wls" else "свободный Пуассон"
        source = "Poisson(μ), μ из неотрицательного ММП" if result["generation"] == "fitted" else "Poisson(x), исходные наблюдения"
        parts = [
            f"<p><b>{method}; {source}.</b> Все активности и σ — в мБк/м³.<br>",
            f"Попыток: {result['n_attempted']} / {result['n_requested']}; "
            f"успешно: {result['n_valid']}; <b>неуспешно: {result['n_failed']}</b>; seed: {result['seed']}.</p>",
            "<table width='100%' border='1' cellspacing='0' cellpadding='4'>",
            "<tr style='background:#e0e0e0'><th>Изотоп</th><th>Среднее</th><th>СКО Монте-Карло</th>"
            "<th>σ исходная</th><th>Средняя σ повторов</th><th>Медиана</th><th>Центральные 95%</th></tr>",
        ]
        original_sigma = result.get("original_stat_sigma")
        for index, summary in enumerate(result["summary"]):
            sigma = None if original_sigma is None else original_sigma[index]
            parts.append(
                f"<tr><td>{escape(str(summary['name']))}</td>"
                f"<td>{activity(summary['mean'])}</td><td>{activity(summary['std'])}</td>"
                f"<td>{activity(sigma)}</td><td>{activity(summary['mean_stat_sigma'])}</td>"
                f"<td>{activity(summary['median'])}</td>"
                f"<td>{activity(summary['q025'])} … {activity(summary['q975'])}</td></tr>"
            )
        parts.append(
            "</table><p>σ — только статистическая погрешность свободной оценки. "
            "Центральные 95% — диапазон полученных оценок, не доверительный интервал истинной активности.</p>"
        )
        if result.get("cancelled"):
            parts.append("<p><b>Эксперимент остановлен: результаты частичные.</b></p>")
        for reason, count in result.get("failure_reasons", {}).items():
            parts.append(f"<p>Неуспешные повторы ({count}): {escape(str(reason))}</p>")
        for warning in result.get("warnings", []):
            parts.append(f"<p>{escape(str(warning))}</p>")
        return "".join(parts)

    def save_results(self):
        if self._running or self.experiment_result is None:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Сохранить распределения активностей", "activity_distributions.txt",
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
                save_distribution_report(self.experiment_result, target)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return
        self.status_label.setText(f"Результаты сохранены: {target}")

    def save_plot(self):
        if self._running or self.experiment_result is None:
            return
        active = self.result_tabs.currentWidget()
        if active is self.canvas:
            figure, label, filename = self.figure, "Гистограммы", "activity_distributions.png"
        elif active is self.pairs_scroll:
            try:
                self._ensure_pair_plot()
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка графика корреляций", str(exc))
                return
            figure, label, filename = self.pairs_figure, "Корреляции активностей", "activity_correlations.png"
        else:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Сохранить: {label}", filename, "Изображение PNG (*.png)",
        )
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        try:
            figure.savefig(target, format="png", dpi=160)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return
        self.status_label.setText(f"{label} сохранены: {target}")
