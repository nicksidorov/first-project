"""
main.py - Графический интерфейс XeMAT 2.0
Только GUI, логика обработки вынесена в отдельные модули
"""

import sys
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
    QMessageBox, QGroupBox, QTextBrowser, QComboBox, QCheckBox
)
from PyQt5.QtGui import QColor

# Импорты ваших модулей (минимальные)
import process_measurement
from results_saver import save_results_to_file
from process_measurement import parse_sample_and_background_and_detector
from calculation import create_combined_vector

# =====================================
# КОНСТАНТЫ
# =====================================
# Путь к файлу детектора по умолчанию - ЗАМЕНИТЕ НА СВОЙ ПУТЬ
DEFAULT_DETECTOR_PATH = r"D:\WORK\2025\МИКС\спайк-компания\Для венцев\DETBK\DETBK20250610213044796.PHD"

# Методы регрессии
REGRESSION_METHODS = {
    'ls': 'Обычный LS (может быть отрицательным)',
    'nnls': 'Неотрицательный LS (NNLS)',
    'mle': 'Пуассоновский MLE (x ≥ 0)',
    'mle_free': 'Пуассоновский MLE (без ограничений)'
}


# =====================================
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС 
# =====================================

class ProcessingThread(QThread):
    """Поток для обработки данных в фоне"""
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, sample_file, background_file, detector_file, 
                 use_fixed_roi=False, regression_method='nnls'):
        super().__init__()
        self.sample_file = sample_file
        self.background_file = background_file
        self.detector_file = detector_file
        self.use_fixed_roi = use_fixed_roi
        self.regression_method = regression_method
    
    def run(self):
        try:
            self.message.emit("Начало обработки...")
            self.progress.emit(10)
            
            print("="*60)
            print("DEBUG: Запуск обработки")
            print(f"DEBUG: Метод регрессии: {self.regression_method}")
            print("="*60)
            
            self.message.emit("Парсинг...")
            
            try:
                results = process_measurement.parse_sample_and_background_and_detector(
                    self.sample_file, 
                    self.background_file, 
                    self.detector_file,
                    use_fixed_roi=self.use_fixed_roi,
                    regression_method=self.regression_method
                )
                print(f"DEBUG: Результаты получены, тип: {type(results)}")
                print(f"DEBUG: Ключи: {list(results.keys()) if isinstance(results, dict) else 'не словарь!'}")
            except Exception as e:
                print(f"DEBUG: Ошибка в parse_sample...: {e}")
                raise
            
            self.progress.emit(60)
            
            self.message.emit("Сохранение...")
            try:
                save_results_to_file(results, "analysis_results.txt")
                print("DEBUG: Файл успешно сохранен")
            except Exception as e:
                print(f"DEBUG: Ошибка сохранения: {e}")
            
            self.progress.emit(90)
            self.message.emit("Завершено!")
            self.progress.emit(100)
            
            print("DEBUG: Отправляю сигнал finished")
            self.finished.emit(results)
            
            print("="*60)
            print("DEBUG: Обработка завершена успешно")
            print("="*60)
            
        except Exception as e:
            print(f"DEBUG: ИСКЛЮЧЕНИЕ в run(): {type(e).__name__}: {e}")
            
            if isinstance(e, dict):
                error_msg = str(e)
            else:
                error_msg = str(e)
            
            print(f"DEBUG: Отправляю ошибку: {error_msg}")
            self.error.emit(error_msg)


class FileSelector:
    """Виджет для выбора файла"""
    def __init__(self, label_text, path_label, button_callback):
        self.layout = QHBoxLayout()
        self.label = QLabel(label_text)
        self.label.setFixedWidth(120)
        self.path_label = path_label
        self.path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        self.path_label.setMinimumHeight(30)
        self.button = QPushButton('Выбрать...')
        self.button.clicked.connect(button_callback)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.path_label, 1)
        self.layout.addWidget(self.button)


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    def __init__(self):
        super().__init__()
        self.sample_file = None
        self.background_file = None
        self.detector_file = DEFAULT_DETECTOR_PATH
        self.results = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('XeMAT 2.0')
        self.setGeometry(100, 100, 900, 850)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ===========================================
        # 1. ВЫБОР ФАЙЛОВ
        # ===========================================
        file_group = QGroupBox("Выбор файлов")
        file_layout = QVBoxLayout()
        
        self.sample_path_label = QLabel('Не выбран')
        self.background_path_label = QLabel('Не выбран')
        self.detector_path_label = QLabel(DEFAULT_DETECTOR_PATH)
        self.detector_path_label.setToolTip(DEFAULT_DETECTOR_PATH)
        
        file_selectors = [
            FileSelector('Файл пробы:', self.sample_path_label, self.select_sample_file),
            FileSelector('Файл фона:', self.background_path_label, self.select_background_file),
            FileSelector('Фон детектора:', self.detector_path_label, self.select_detector_file)
        ]
        
        for selector in file_selectors:
            file_layout.addLayout(selector.layout)
        
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # ===========================================
        # 2. ВЫБОР МЕТОДА РЕГРЕССИИ
        # ===========================================
        method_group = QGroupBox("Метод регрессии")
        method_layout = QHBoxLayout()
        
        method_label = QLabel("Выберите метод:")
        self.method_combo = QComboBox()
        self.method_combo.addItem("Неотрицательный LS (NNLS) - по умолчанию", "nnls")
        self.method_combo.addItem("Обычный LS (может быть отрицательным)", "ls")
        self.method_combo.addItem("Пуассоновский MLE (x ≥ 0)", "mle")
        self.method_combo.addItem("Пуассоновский MLE (без ограничений)", "mle_free")
        self.method_combo.setCurrentIndex(0)
        
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        
        method_group.setLayout(method_layout)
        main_layout.addWidget(method_group)
        
        # ===========================================
        # 3. ЧЕКБОКС ДЛЯ ФИКСИРОВАННЫХ ROI
        # ===========================================
        self.checkbox_fixed_roi = QCheckBox("Использовать фиксированные ROI")
        self.checkbox_fixed_roi.setToolTip("Использовать ROI из таблицы констант вместо ROI из файла")
        main_layout.addWidget(self.checkbox_fixed_roi)
        
        # ===========================================
        # 4. КНОПКА РАСЧЕТА
        # ===========================================
        self.calculate_button = self.create_button('РАССЧИТАТЬ', '#3498db', self.calculate)
        self.calculate_button.setEnabled(False)
        main_layout.addWidget(self.calculate_button)
        
        # ===========================================
        # 5. ПРОГРЕСС БАР
        # ===========================================
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # ===========================================
        # 6. СООБЩЕНИЯ
        # ===========================================
        self.message_label = QLabel('')
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        main_layout.addWidget(self.message_label)
        
        # ===========================================
        # 7. РЕЗУЛЬТАТЫ АКТИВНОСТЕЙ
        # ===========================================
        results_group = QGroupBox("Результаты расчета активностей")
        results_layout = QVBoxLayout()
        
        self.results_output = QTextBrowser()
        self.results_output.setMinimumHeight(300)
        self.results_output.setStyleSheet("""
            QTextBrowser {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        self.results_output.setVisible(False)
        
        results_layout.addWidget(self.results_output)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # ===========================================
        # 8. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ
        # ===========================================
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Дополнительная информация появится здесь после расчета...")
        self.text_output.setMaximumHeight(200)
        main_layout.addWidget(self.text_output)
        
        # ===========================================
        # 9. КНОПКА СОХРАНЕНИЯ
        # ===========================================
        self.save_button = self.create_button('Сохранить результаты в файл', '#27ae60', self.save_results)
        self.save_button.setEnabled(False)
        main_layout.addWidget(self.save_button)
        
        self.statusBar().showMessage('Готов к работе')
        
        self.check_detector_file()
    
    def check_detector_file(self):
        """Проверяет существует ли файл детектора по умолчанию"""
        import os
        if not os.path.exists(self.detector_file):
            self.detector_path_label.setText(f"⚠ {DEFAULT_DETECTOR_PATH} (ФАЙЛ НЕ НАЙДЕН)")
            self.detector_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; color: red;")
            QMessageBox.warning(self, 'Предупреждение', 
                              f'Файл детектора по умолчанию не найден:\n{DEFAULT_DETECTOR_PATH}\n\nПожалуйста, выберите файл вручную.')
        else:
            self.detector_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; color: green;")
            self.check_files_selected()
    
    def create_button(self, text, color, callback):
        """Создает стилизованную кнопку"""
        button = QPushButton(text)
        button.setFixedHeight(40 if text == 'РАССЧИТАТЬ' else 35)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 5px;
                font-weight: {'bold' if text == 'РАССЧИТАТЬ' else 'normal'};
            }}
            QPushButton:hover {{
                background-color: {'#2980b9' if color == '#3498db' else '#219653'};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
        """)
        button.clicked.connect(callback)
        return button
    
    def select_file(self, title, file_types):
        """Открывает диалог выбора файла"""
        file_name, _ = QFileDialog.getOpenFileName(self, title, '', file_types)
        return file_name
    
    def select_sample_file(self):
        """Выбор файла пробы"""
        if file_name := self.select_file('Выберите файл пробы', 'Файлы спектров (*.ims *.PHD *.phd);;Все файлы (*.*)'):
            self.sample_file = file_name
            self.sample_path_label.setText(file_name)
            self.check_files_selected()
    
    def select_background_file(self):
        """Выбор файла фона"""
        if file_name := self.select_file('Выберите файл фона', 'Файлы спектров (*.ims *.PHD *.phd);;Все файлы (*.*)'):
            self.background_file = file_name
            self.background_path_label.setText(file_name)
            self.check_files_selected()
    
    def select_detector_file(self):
        """Выбор файла фона детектора"""
        if file_name := self.select_file('Выберите файл фона детектора', 'Файлы спектров (*.ims *.PHD *.phd);;Все файлы (*.*)'):
            self.detector_file = file_name
            self.detector_path_label.setText(file_name)
            self.detector_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; color: black;")
            self.check_files_selected()
    
    def check_files_selected(self):
        """Проверяет, выбраны ли все файлы"""
        if all([self.sample_file, self.background_file, self.detector_file]):
            self.calculate_button.setEnabled(True)
            self.statusBar().showMessage('Все файлы выбраны. Можно начинать расчет.')
        else:
            self.calculate_button.setEnabled(False)
            files = []
            if self.sample_file: files.append('проба')
            if self.background_file: files.append('фон')
            if self.detector_file: 
                import os
                if self.detector_file == DEFAULT_DETECTOR_PATH:
                    if os.path.exists(self.detector_file):
                        files.append('фон детектора (по умолчанию)')
                    else:
                        files.append('фон детектора (файл не найден)')
                else:
                    files.append('фон детектора')
            if files:
                self.statusBar().showMessage(f'Выбраны: {", ".join(files)}')
    
    def calculate(self):
        """Запуск расчета"""
        if not all([self.sample_file, self.background_file, self.detector_file]):
            QMessageBox.warning(self, 'Ошибка', 'Пожалуйста, выберите все три файла!')
            return
        
        import os
        if self.detector_file == DEFAULT_DETECTOR_PATH and not os.path.exists(self.detector_file):
            QMessageBox.warning(self, 'Ошибка', 
                              f'Файл детектора по умолчанию не найден:\n{DEFAULT_DETECTOR_PATH}\n\nПожалуйста, выберите файл вручную.')
            return
        
        use_fixed_roi = self.checkbox_fixed_roi.isChecked()
        regression_method = self.method_combo.currentData()
        method_name = self.method_combo.currentText()
        
        if use_fixed_roi:
            self.statusBar().showMessage(f'Расчет с фиксированными ROI, метод: {method_name}...')
        else:
            self.statusBar().showMessage(f'Расчет с ROI из файлов, метод: {method_name}...')
        
        self.calculate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.message_label.setText('Обработка...')
        self.results_output.setVisible(False)
        self.results_output.clear()
        self.text_output.clear()
        
        self.thread = ProcessingThread(
            self.sample_file, 
            self.background_file, 
            self.detector_file,
            use_fixed_roi=use_fixed_roi,
            regression_method=regression_method
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.message.connect(self.update_message)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()
    
    def update_progress(self, value):
        """Обновление прогресс-бара"""
        self.progress_bar.setValue(value)
    
    def update_message(self, message):
        """Обновление сообщений"""
        self.message_label.setText(message)
        self.statusBar().showMessage(message)
    
    def on_processing_finished(self, results):
        """Обработка завершения расчета"""
        self.results = results
        self.progress_bar.setVisible(False)
        self.calculate_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        method_used = results.get('regression_method_used', 'nnls')
        method_names = {'ls': 'LS', 'nnls': 'NNLS', 'mle': 'MLE (x≥0)', 'mle_free': 'MLE (без огр.)'}
        self.message_label.setText(f'Обработка завершена успешно! Метод: {method_names.get(method_used, method_used)}')
        
        self.results_output.setVisible(True)
        self.results_output.setHtml(self.generate_activities_html(results))
        self.text_output.setPlainText(self.generate_output_text(results))
        
        success_msg = f'Обработка завершена успешно!\nМетод: {method_names.get(method_used, method_used)}\nРезультаты сохранены в файл analysis_results.txt'
        QMessageBox.information(self, 'Успех', success_msg)
        self.statusBar().showMessage('Готово')
    
    def generate_activities_html(self, results):
        """Генерирует HTML для отображения активности и удельной активности"""
        html = []
        
        if not results.get('all_activities'):
            return "<div style='color: red; text-align: center; padding: 20px;'>Нет данных о активностях</div>"
        
        all_act = results['all_activities']
        
        # Получаем названия изотопов
        isotopes_names = []
        if results.get('regression'):
            isotopes_names = results['regression']['isotopes']
        elif 'isotopes' in all_act:
            isotopes_names = all_act['isotopes']
        else:
            from constants import HALF_LIFE_DAYS
            isotopes_names = list(HALF_LIFE_DAYS.keys())
        
        # Добавляем информацию о методе регрессии
        method_used = results.get('regression_method_used', 'nnls')
        method_names = {'ls': 'LS (обычный)', 'nnls': 'NNLS (неотрицательный)', 'mle': 'MLE (Пуассоновский)'}
        
        html.append("<div style='text-align: center; margin-bottom: 15px;'>")
        html.append("<h3>ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ РАСЧЕТА АКТИВНОСТЕЙ</h3>")
        html.append(f"<p style='color: #2c3e50; background-color: #ecf0f1; padding: 5px; border-radius: 5px;'>")
        html.append(f"<b>Метод регрессии:</b> {method_names.get(method_used, method_used)}")
        html.append("</p>")
        html.append("</div>")
        
        # Контейнер для двух таблиц в один ряд
        html.append("<table style='width: 100%; border-collapse: separate; border-spacing: 20px 0; margin-bottom: 20px;'>")
        html.append("<tr>")
        
        # 1. Активность в детекторе
        html.append("<td style='width: 50%; vertical-align: top;'>")
        html.append("<h4>1. АКТИВНОСТЬ В ДЕТЕКТОРЕ (мБк):</h4>")
        html.append("<table style='border-collapse: collapse; width: 100%; margin-bottom: 15px; border: 1px solid #ccc;'>")
        html.append("<thead>")
        html.append("<tr style='background-color: #e0e0e0;'>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: left;'>Изотоп</th>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: right;'>Значение</th>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: right;'>±σ</th>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: right;'>Отн. погр.</th>")
        html.append("</tr>")
        html.append("</thead>")
        html.append("<tbody>")
        
        for i, iso_name in enumerate(isotopes_names):
            value = all_act['activities_in_detector'].get(iso_name) * 1000
            std = all_act['std_activities'].get(iso_name) * 1000
            rel_err = all_act['relative_errors_activities'].get(iso_name)
            
            if value is not None and std is not None:
                bg_color = "#f0f8ff" if i % 2 == 0 else "#ffffff"
                html.append(f"<tr style='background-color: {bg_color}'>")
                html.append(f"<td style='border: 1px solid #ccc; padding: 8px;'>{iso_name}</td>")
                
                if abs(value) < 1e-3:
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>{value:.3e}</td>")
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>±{std:.3e}</td>")
                else:
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>{value:.6f}</td>")
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>±{std:.6f}</td>")
                
                html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>{rel_err}</td>")
                html.append("</tr>")
        
        html.append("</tbody>")
        html.append("</table>")
        html.append("</td>")
        
        # 2. Удельная активность
        html.append("<td style='width: 50%; vertical-align: top;'>")
        html.append("<h4>2. УДЕЛЬНАЯ АКТИВНОСТЬ (мБк/м³):</h4>")
        html.append("<table style='border-collapse: collapse; width: 100%; margin-bottom: 15px; border: 1px solid #ccc;'>")
        html.append("<thead>")
        html.append("<tr style='background-color: #e0e0e0;'>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: left;'>Изотоп</th>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: right;'>Значение</th>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: right;'>±σ</th>")
        html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: right;'>Отн. погр.</th>")
        html.append("</tr>")
        html.append("</thead>")
        html.append("<tbody>")
        
        for i, iso_name in enumerate(isotopes_names):
            value = all_act['specific_activities'].get(iso_name) * 1000
            std = all_act['std_specific_activities'].get(iso_name) * 1000
            rel_err = all_act['relative_errors_specific'].get(iso_name)
            
            if value is not None and std is not None:
                bg_color = "#f0f8ff" if i % 2 == 0 else "#ffffff"
                html.append(f"<tr style='background-color: {bg_color}'>")
                html.append(f"<td style='border: 1px solid #ccc; padding: 8px;'>{iso_name}</td>")
                
                if abs(value) < 1e-3:
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>{value:.3e}</td>")
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>±{std:.3e}</td>")
                else:
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>{value:.6f}</td>")
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>±{std:.6f}</td>")
                
                html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>{rel_err}</td>")
                html.append("</tr>")
        
        html.append("</tbody>")
        html.append("</table>")
        html.append("</td>")
        
        html.append("</tr>")
        html.append("</table>")
        
        # 3. MDA
        if 'MDA' in all_act:
            html.append("<div style='margin-bottom: 20px;'>")
            html.append("<h4>3. МИНИМАЛЬНАЯ ДЕТЕКТИРУЕМАЯ АКТИВНОСТЬ (MDA), мБк/м³:</h4>")
            html.append("<table style='border-collapse: collapse; width: 100%; margin-bottom: 15px; border: 1px solid #ccc;'>")
            html.append("<thead>")
            html.append("<tr style='background-color: #e0e0e0;'>")
            html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: left;'>Изотоп</th>")
            html.append("<th style='border: 1px solid #ccc; padding: 8px; text-align: right;'>MDA</th>")
            html.append("</tr>")
            html.append("</thead>")
            html.append("<tbody>")
            
            for i, iso_name in enumerate(isotopes_names):
                mda = all_act['MDA'].get(iso_name)
                
                if mda is not None:
                    bg_color = "#f0f8ff" if i % 2 == 0 else "#ffffff"
                    html.append(f"<tr style='background-color: {bg_color}'>")
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px;'>{iso_name}</td>")
                    html.append(f"<td style='border: 1px solid #ccc; padding: 8px; text-align: right;'>{mda * 1000:.3e}</td>")
                    html.append("<tr>")
            
            html.append("</tbody>")
            html.append("</table>")
            html.append("</div>")
        
        # 4. Объемные параметры
        if 'volumes' in all_act:
            volumes = all_act['volumes']
            html.append("<div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>")
            html.append("<h4>ОБЪЕМНЫЕ ПАРАМЕТРЫ:</h4>")
            html.append(f"<p>V_xe = {volumes['V_xe']:.3f} см³</p>")
            html.append(f"<p>V_eff = V_xe / {volumes['concentration_factor']:.3f} = {volumes['V_eff']:.3f} м³</p>")
            html.append("</div>")
        
        return "".join(html)
    
    def generate_output_text(self, results):
        """Генерирует текст для дополнительной информации"""
        output = ["ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:", "="*50]
        
        method_used = results.get('regression_method_used', 'nnls')
        method_names = {'ls': 'LS (обычный)', 'nnls': 'NNLS (неотрицательный)', 'mle': 'MLE (Пуассоновский)'}
        output.append(f"\nМЕТОД РЕГРЕССИИ: {method_names.get(method_used, method_used)}")
        
        output.extend(["\nВРЕМЕННЫЕ ДАННЫЕ:", "-"*30])
        
        for title, key in [("Проба", 'sample_temporal'), ("Фон", 'background_temporal')]:
            temp = results[key]
            if temp.get('start_time'):
                output.append(f"{title} - Старт: {temp['start_time']}")
            if temp.get('real_time'):
                output.append(f"{title} - Реальное время: {temp['real_time']:.1f} сек")
            if temp.get('live_time'):
                output.append(f"{title} - Живое время: {temp['live_time']:.1f} сек")
        
        intervals = results.get('time_intervals', {})
        if 'between_start' in intervals:
            output.extend([
                f"\nВРЕМЕННОЙ ИНТЕРВАЛ τ:",
                f"  τ = {intervals['between_start']:.1f} сек",
                f"  Направление: {intervals.get('direction', 'не определено')}"
            ])
        
        if results.get('regression'):
            output.extend([
                "\n\nРЕЗУЛЬТАТЫ РЕГРЕССИИ:",
                "="*40
            ])
            
            reg = results['regression']
            output.append(f"Невязка NNLS: {reg.get('residual_nnls', 0):.6f}")
            output.append(f"Остаточная сумма квадратов (LS): {reg.get('RSS', 0):.6f}")
            
            residuals = reg.get('residuals')
            if residuals is not None and len(residuals) > 0:
                import numpy as np
                residuals_text = np.array2string(np.array(residuals), suppress_small=True)
                output.append(f"\nВектор остатков:")
                output.append(residuals_text)
        
        if results.get('all_activities'):
            all_act = results['all_activities']
            formulas = all_act.get('formulas', {})
            activity_formula = formulas.get('activity_detector', 'a = Hβ')
            specific_formula = formulas.get('specific_activity', 'a_c = (1/V_eff) × Gs × Gp × Γ × a')
            
            output.extend([
                "\nФИНАЛЬНЫЕ ФОРМУЛЫ РАСЧЕТА:",
                "-"*40,
                f"1. Активность в детекторе: {activity_formula}",
                f"2. Удельная активность: {specific_formula}",
            ])
            
            if 'volumes' in all_act:
                volumes = all_act['volumes']
                if 'V_xe' in volumes and 'V_eff' in volumes:
                    output.append(f"3. Объем: V_eff = {volumes['V_xe']:.3f} / 0.087 = {volumes['V_eff']:.3f} м³")
        
        output.extend([
            "\n" + "="*50,
            "Подробные результаты сохранены в файл: analysis_results.txt"
        ])
        
        return "\n".join(output)
    
    def on_processing_error(self, error_message):
        """Обработка ошибки расчета"""
        self.progress_bar.setVisible(False)
        self.calculate_button.setEnabled(True)
        self.message_label.setText('Ошибка обработки')
        QMessageBox.critical(self, 'Ошибка', f'Произошла ошибка при обработке:\n{error_message}')
        self.statusBar().showMessage('Ошибка')
    
    def save_results(self):
        """Сохранение результатов в файл"""
        print(f"\n🎯 GUI_SAVE: КНОПКА СОХРАНЕНИЯ НАЖАТА")
        
        if not self.results:
            QMessageBox.warning(self, 'Ошибка', 'Нет результатов для сохранения!')
            return
        
        file_name, _ = QFileDialog.getSaveFileName(
            self, 
            'Сохранить результаты', 
            'analysis_results.txt', 
            'Текстовые файлы (*.txt);;Все файлы (*.*)'
        )
        
        if file_name:
            try:
                save_results_to_file(self.results, file_name)
                QMessageBox.information(self, 'Успех', f'Результаты успешно сохранены в файл:\n{file_name}')
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Не удалось сохранить файл:\n{type(e).__name__}: {e}')


def main():
    """Основная функция запуска приложения"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()