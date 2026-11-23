import sys
import argparse
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QFileDialog, QWidget,
                             QProgressBar, QMessageBox, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import plotly.graph_objects as go
import numpy as np
from scipy import linalg
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, Alignment
from datetime import datetime

class SpectrumAnalysisThread(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, template_file, spectra_folder, plots_folder, save_spectra, use_poisson_mle, output_file=None):
        super().__init__()
        self.template_file = template_file
        self.output_file = output_file or str(Path(spectra_folder) / (
            'QC_results_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.xlsx'))
        self.results = []
        self.mle_converged = False
        self.spectra_folder = spectra_folder
        self.plots_folder = plots_folder
        self.save_spectra = save_spectra
        self.use_poisson_mle = use_poisson_mle  # Новый параметр
        self.lowess_params = {'bandwidth': 0.19, 'poly_order': 2, 'it': 3}

    def fit_poisson_mle(self, model_func, x_data, y_data, p0, max_iter=50, tol=1e-6, verbose=False):
        """
        Подгонка модели к пуассоновским данным методом максимального правдоподобия
        через итеративное перевзвешивание (IRLS).
        """
        
        self.mle_converged = False
        params = np.array(p0, dtype=float)
        y_data = np.asarray(y_data)
        x_data = np.asarray(x_data)
        
        for iteration in range(max_iter):
            params_old = params.copy()
            
            # Вычисляем ожидаемые значения при текущих параметрах
            mu = model_func(x_data, *params)
            mu = np.maximum(mu, 1e-10)  # Защита от нуля
            
            # Пуассоновские веса: sigma = sqrt(mu)
            sigmas = np.sqrt(mu)
            
            try:
                popt, pcov = curve_fit(
                    model_func, x_data, y_data, 
                    p0=params, 
                    sigma=sigmas, 
                    absolute_sigma=True,
                    maxfev=5000
                )
                params = popt
            except Exception as e:
                if verbose:
                    self.message.emit(f"  Итерация {iteration}: ошибка {e}")
                raise RuntimeError(f"Ошибка взвешенной подгонки IRLS на итерации {iteration + 1}: {e}") from e
            
            # Проверка сходимости
            rel_change = np.max(np.abs((params - params_old) / (params_old + 1e-10)))
            if verbose:
                mu_current = model_func(x_data, *params)
                mu_current = np.maximum(mu_current, 1e-10)
                log_likelihood = np.sum(y_data * np.log(mu_current) - mu_current)
                self.message.emit(f"  Итерация {iteration}: change={rel_change:.2e}, LL={log_likelihood:.1f}")
            
            if rel_change < tol:
                self.mle_converged = True
                if verbose:
                    self.message.emit(f"  Сошлось за {iteration + 1} итераций")
                break
        
        # Финальный расчёт ковариации
        mu_final = model_func(x_data, *params)
        mu_final = np.maximum(mu_final, 1e-10)
        sigmas_final = np.sqrt(mu_final)
        
        try:
            _, pcov_final = curve_fit(
                model_func, x_data, y_data,
                p0=params, sigma=sigmas_final, absolute_sigma=True
            )
            pcov = pcov_final
        except:
            pcov = np.eye(len(params)) * np.inf
        
        return params, pcov, iteration + 1

    def run(self):
        try:
            self.message.emit("=== ОБРАБОТКА ШАБЛОННОГО СПЕКТРА ===")
            
            # Создаем папку для сохранения графиков ТОЛЬКО если чекбокс отмечен
            if self.save_spectra:
                os.makedirs(self.plots_folder, exist_ok=True)
                self.message.emit(f"Создана папка для графиков: {self.plots_folder}")
            else:
                self.message.emit("Сохранение спектров отключено (папка для графиков не создается)")
            
            # 1. ОБРАБОТКА ШАБЛОННОГО СПЕКТРА
            date_template, template_matrix, template_gamma = self.parse_spectrum(self.template_file)
            
            if template_matrix is None:
                self.error_signal.emit(f"Ошибка: в шаблонном файле {self.template_file} не найдена матрица!")
                return
            
            filtered_template = self.apply_filters(template_matrix)
            x_template = np.sum(filtered_template, axis=0)
            
            # Создаем шаблонную функцию
            y_smoothed = self.create_smoothed_spectrum(x_template.copy(), **self.lowess_params)
            if not np.all(np.isfinite(y_smoothed)) or np.max(y_smoothed) <= 0:
                raise ValueError("Шаблон после фильтрации не содержит положительных конечных значений.")
            self.spectrum_func = self.create_spectrum_function(x_template.copy(), **self.lowess_params)
            
            # Сохраняем исходный шаблон (несдвинутый) для построения графиков
            self.original_template = y_smoothed.copy()
            
            self.message.emit(f"Шаблонный спектр загружен: {os.path.basename(self.template_file)}")
            self.message.emit(f"Дата шаблонного спектра: {date_template}")
            self.message.emit(f"Метод фитинга: {'Пуассоновский MLE (итеративный)' if self.use_poisson_mle else 'Обычный (фиксированные веса)'}")
            
            # 2. ОБРАБОТКА ВСЕХ СПЕКТРОВ В ПАПКЕ
            self.message.emit(f"\n=== ОБРАБОТКА СПЕКТРОВ В ПАПКЕ ===")
            
            # Находим все файлы в папке
            spectrum_files = []
            for file in os.listdir(self.spectra_folder):
                if file.lower().endswith('.phd') and os.path.isfile(os.path.join(self.spectra_folder, file)):
                    spectrum_files.append(os.path.join(self.spectra_folder, file))
            
            spectrum_files.sort(key=str.casefold)
            if not spectrum_files:
                self.error_signal.emit("В указанной папке не найдено файлов спектров!")
                return
            
            self.message.emit(f"Найдено файлов: {len(spectrum_files)}")
            
            # Обрабатываем каждый файл
            results = []
            for i, file_path in enumerate(spectrum_files, 1):
                self.message.emit(f"Обработка {i}/{len(spectrum_files)}: {os.path.basename(file_path)}")
                self.progress.emit(int((i / len(spectrum_files)) * 100))
                
                result = self.fit_spectrum_to_template(file_path, self.spectrum_func, self.lowess_params)
                if result:
                    # Сохраняем график для успешной подгонки ТОЛЬКО если чекбокс отмечен
                    if result.get('success', False) and self.save_spectra:
                        self.save_fitting_plot(file_path, result)
                    results.append(result)
            
            # 3. СОЗДАНИЕ EXCEL ФАЙЛА С РЕЗУЛЬТАТАМИ
            self.message.emit("\n=== СОЗДАНИЕ EXCEL ОТЧЕТА ===")
            
            self.results = results
            output_file = self.save_to_excel(results, self.template_file, self.spectra_folder, self.output_file)
            
            # Формируем итоговое сообщение
            successful_results = [r for r in results if r.get('success', False)]
            failed_results = [r for r in results if not r.get('success', False)]
            
            summary = f"""
=== РЕЗЮМЕ ===
Шаблон: {os.path.basename(self.template_file)}
Папка со спектрами: {self.spectra_folder}
Метод фитинга: {'Пуассоновский MLE (итеративный)' if self.use_poisson_mle else 'Обычный (фиксированные веса)'}
Сохранение спектров: {'Включено' if self.save_spectra else 'Отключено'}
Папка с графиками: {self.plots_folder if self.save_spectra else 'Не создавалась'}
Всего обработано: {len(spectrum_files)} файлов
Успешных подгонок: {len(successful_results)}
Неудачных подгонок: {len(failed_results)}
Сохранено графиков: {len(successful_results) if self.save_spectra else 0}
"""

            summary += f"\nРезультаты сохранены в файл: {output_file}"
            
            self.finished_signal.emit(summary)
            
        except Exception as e:
            self.error_signal.emit(f"Критическая ошибка: {str(e)}")

    def save_fitting_plot(self, file_path, result):
        """Сохраняет график результата фитинга в HTML формате"""
        try:
            # Получаем данные спектра
            date_str, matrix, gamma_spectrum = self.parse_spectrum(file_path)
            filtered_new = self.apply_filters(matrix)
            x_new = np.sum(filtered_new, axis=0)
            
            # Создаем график
            fig = go.Figure()
            
            # Каналы для построения
            channels = np.arange(256)
            
            # 1. Точки экспериментального спектра
            fig.add_trace(go.Scatter(
                x=channels,
                y=x_new,
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.6),
                name='Экспериментальный спектр'
            ))
            
            # 2. Подогнанный шаблон (с коэффициентом масштаба k)
            A_fit = result['A']
            k_fit = result['k']
            fitted_template = A_fit * (1/k_fit) * self.spectrum_func(channels/k_fit)
            fig.add_trace(go.Scatter(
                x=channels,
                y=fitted_template,
                mode='lines',
                line=dict(color='red', width=2),
                name=f'Подогнанный шаблон (k={k_fit:.3f})'
            ))
            
            # 3. Исходный шаблон (без сдвига, коэффициент масштаба 1)
            original_template_scaled = result['A'] * self.original_template
            fig.add_trace(go.Scatter(
                x=channels,
                y=original_template_scaled,
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name='Исходный шаблон (k=1)'
            ))
            
            # Настройка графика
            fig.update_layout(
                title=f'Результат фитинга для {os.path.basename(file_path)}<br>k = {k_fit:.4f} ± {result["k_err"]:.4f}, A = {A_fit:.2f}',
                xaxis_title='Каналы',
                yaxis_title='Интенсивность',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1
                ),
                hovermode='closest'
            )
            
            # Добавляем сетку
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Сохраняем график в HTML
            file_name = Path(file_path).stem
            output_path = os.path.join(self.plots_folder, f"{file_name}_fitting.html")
            fig.write_html(output_path)
            
            self.message.emit(f"  Сохранен график: {file_name}_fitting.html")
            
        except Exception as e:
            self.message.emit(f"  Ошибка при сохранении графика: {str(e)}")

    def parse_spectrum(self, file_path):
        date_str, matrix, gamma_spectrum = "Не найдена", None, None
        try:
            try:
                text = Path(file_path).read_text(encoding='utf-8-sig')
            except UnicodeDecodeError:
                text = Path(file_path).read_text(encoding='cp1251')
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == '#Acquisition' and i + 1 < len(lines):
                    parts = lines[i + 1].split()
                    if parts:
                        date_str = parts[0]
                    break
            for i, line in enumerate(lines):
                if line.strip() == '#Histogram':
                    rows = lines[i + 2:i + 2 + 256]
                    matrix = np.array([[float(x) for x in row.split()] for row in rows])
                    if matrix.shape != (256, 256):
                        raise ValueError('Блок #Histogram должен содержать матрицу 256 × 256.')
                    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
                        raise ValueError('Матрица содержит отрицательные или неконечные значения.')
                    break
            for i, line in enumerate(lines):
                if line.strip() != '#g_Spectrum':
                    continue
                try:
                    counts = []
                    for row in lines[i + 2:]:
                        row = row.strip()
                        if row.startswith('#'):
                            break
                        if not row:
                            continue
                        values = row.split()
                        if len(values) < 2:
                            raise ValueError('Некорректная строка гамма-спектра.')
                        counts.extend(float(value) for value in values[1:6])
                        if len(counts) >= 256:
                            break
                    if len(counts) < 256:
                        raise ValueError(f'Найдено только {len(counts)} из 256 гамма-каналов.')
                    gamma_spectrum = np.array(counts[:256])
                    if not np.all(np.isfinite(gamma_spectrum)) or np.any(gamma_spectrum < 0):
                        raise ValueError('Гамма-спектр содержит недопустимые значения.')
                except (ValueError, OverflowError) as error:
                    gamma_spectrum = None
                    self.message.emit(f"  Гамма-спектр не использован: {error}")
                break
        except (OSError, ValueError, OverflowError, UnicodeError) as error:
            self.message.emit(f"Ошибка при чтении {Path(file_path).name}: {error}")
            matrix, gamma_spectrum = None, None
        return date_str, matrix, gamma_spectrum

    def apply_filters(self, matrix):
        filtered_matrix = matrix.copy()
        y_coords, x_coords = np.ogrid[0:256, 0:256]
        mask1 = y_coords < 40
        mask2 = (x_coords + y_coords) < 190
        filtered_matrix[mask1 | mask2] = 0
        return filtered_matrix

    def tricubic(self, x):
        return np.where(np.abs(x) < 1, (1 - np.abs(x)**3)**3, 0)

    def lowess_maple(self, x, y, bandwidth=0.19, poly_order=2, it=3):
        n = len(x)
        y_smooth = y.copy()
        k = int(bandwidth * n)
        
        for iteration in range(it):
            weights = np.ones(n)
            for i in range(n):
                dist = np.abs(x - x[i])
                idx = np.argsort(dist)[:k]
                dist_sorted = dist[idx]
                d_max = dist_sorted[-1]
                w = self.tricubic(dist_sorted / d_max) * weights[idx]
                
                X = np.column_stack([np.ones_like(x[idx]), x[idx], x[idx]**2])
                W = np.diag(w)
                try:
                    beta = linalg.solve(X.T @ W @ X, X.T @ W @ y[idx])
                    y_smooth[i] = beta[0] + beta[1] * x[i] + beta[2] * x[i]**2
                except linalg.LinAlgError:
                    y_smooth[i] = y[i]
            
            if iteration < it - 1:
                residuals = y - y_smooth
                median_abs_residual = np.median(np.abs(residuals))
                if median_abs_residual > 0:
                    u = residuals / (6 * median_abs_residual)
                    weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
        
        return y_smooth

    def create_smoothed_spectrum(self, x_projection, bandwidth=0.19, poly_order=2, it=3):
        x_projection[0:5] = 0
        x_channels = np.arange(5, 256)
        y_values = x_projection[5:256]
        
        y_smoothed = self.lowess_maple(x_channels, y_values, bandwidth, poly_order, it)
        
        slope = (y_smoothed[1] - y_smoothed[0]) / (x_channels[1] - x_channels[0])
        intercept = y_smoothed[0] - slope * x_channels[0]
        x_zero = max(0, -intercept / slope) if slope != 0 else 0
        
        x_extrap = np.arange(0, 5)
        y_extrap = np.maximum(0, slope * x_extrap + intercept)
        y_extrap[x_extrap >= x_zero] = 0
        
        y_full = np.zeros(256)
        y_full[0:5] = y_extrap
        y_full[5:] = y_smoothed
        
        return y_full

    def create_spectrum_function(self, x_projection, **lowess_params):
        y_smoothed = self.create_smoothed_spectrum(x_projection, **lowess_params)
        interpolator = CubicSpline(np.arange(256), y_smoothed)
        
        def spectrum_func(x):
            x = np.asarray(x)
            result = interpolator(x)
            result[x < 0] = 0
            result[x > 255] = 0
            return result
        
        return spectrum_func

    def model_function(self, x, A, k):
        return A * (1/k) * self.spectrum_func(x/k)

    def gauss_with_linear(self, x, A, mu, sigma, a, b):
        gaussian = A * np.exp(-(x - mu)**2 / (2 * sigma**2))
        linear_bg = a * x + b
        return gaussian + linear_bg

    def fit_peak_in_region(self, spectrum, fit_start=209, fit_end=251):
        x_fit = np.arange(fit_start, fit_end + 1)
        y_fit = spectrum[fit_start:fit_end + 1]
        sigma = np.sqrt(y_fit + 1)
        
        A0 = np.max(y_fit) - np.min(y_fit)
        mu0 = x_fit[np.argmax(y_fit)]
        sigma0 = 5.0
        a0 = 0.0
        b0 = np.min(y_fit)
        
        initial_guess = [A0, mu0, sigma0, a0, b0]
        
        try:
            popt, pcov = curve_fit(self.gauss_with_linear, x_fit, y_fit, 
                                  p0=initial_guess, sigma=sigma, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            
            x_continuous = np.linspace(fit_start, fit_end, 1000)
            y_fitted = self.gauss_with_linear(x_continuous, *popt)
            y_bg = popt[3] * x_continuous + popt[4]
            peak_area = np.trapz(y_fitted - y_bg, x_continuous)
            
            results = {
                'peak_center': popt[1], 'peak_center_err': perr[1],
                'peak_area': peak_area,
                'peak_amplitude': popt[0], 'peak_amplitude_err': perr[0],
                'peak_sigma': popt[2], 'peak_sigma_err': perr[2],
                'success': True
            }
            
        except Exception as e:
            results = {
                'peak_center': 0, 'peak_center_err': 0,
                'peak_area': 0,
                'peak_amplitude': 0, 'peak_amplitude_err': 0,
                'peak_sigma': 0, 'peak_sigma_err': 0,
                'success': False,
                'error': str(e)
            }
        
        return results

    def fit_spectrum_to_template(self, new_file_path, spectrum_func, lowess_params):
        try:
            date_str, matrix, gamma_spectrum = self.parse_spectrum(new_file_path)
            
            if matrix is None:
                return {
                    'file': Path(new_file_path).name,
                    'date': date_str,
                    'success': False,
                    'error': "Не найдена матрица в файле"
                }
            
            filtered_new = self.apply_filters(matrix)
            x_new = np.sum(filtered_new, axis=0)
            
            fit_start, fit_end = 9, 220
            x_fit = np.arange(fit_start, fit_end + 1)
            y_fit = x_new[fit_start:fit_end + 1]

            A0 = np.max(y_fit) / np.max([spectrum_func(x) for x in range(256)]) * 1.0
            k0 = 1.0

            # Выбор метода фитинга
            if self.use_poisson_mle:
                # Используем Пуассоновский MLE с итеративным перевзвешиванием
                popt, pcov, n_iter = self.fit_poisson_mle(
                    self.model_function, x_fit, y_fit, 
                    p0=[A0, k0], max_iter=50, tol=1e-6, verbose=False
                )
                state = "критерий сходимости выполнен" if self.mle_converged else "лимит итераций; сходимость не подтверждена"
                self.message.emit(f"  → Пуассоновский IRLS: {n_iter} итераций, {state}")
            else:
                # Обычный метод с фиксированными весами
                sigmas = np.sqrt(np.maximum(y_fit, 1.0))
                popt, pcov = curve_fit(self.model_function, x_fit, y_fit, 
                                    p0=[A0, k0], sigma=sigmas, absolute_sigma=True)
            
            A_fit, k_fit = popt
            A_err, k_err = np.sqrt(np.diag(pcov))
            
            peak_results = None
            if gamma_spectrum is not None:
                peak_results = self.fit_peak_in_region(gamma_spectrum, 209, 251)
            else:
                peak_results = {'success': False, 'error': "Гамма-спектр не найден"}
            
            return {
                'file': Path(new_file_path).name,
                'date': date_str,
                'A': A_fit,
                'A_err': A_err,
                'k': k_fit,
                'k_err': k_err,
                'peak_center': peak_results['peak_center'] if peak_results['success'] else 0,
                'peak_center_err': peak_results['peak_center_err'] if peak_results['success'] else 0,
                'peak_area': peak_results['peak_area'] if peak_results['success'] else 0,
                'peak_success': peak_results['success'],
                'irls_converged': bool(self.mle_converged) if self.use_poisson_mle else None,
                'success': True
            }
            
        except Exception as e:
            return {
                'file': Path(new_file_path).name,
                'date': "Ошибка чтения",
                'A': 0,
                'A_err': 0,
                'k': 0,
                'k_err': 0,
                'peak_center': 0,
                'peak_center_err': 0,
                'peak_area': 0,
                'peak_success': False,
                'success': False,
                'error': str(e)
            }

    def save_to_excel(self, results, template_file, spectra_folder, output_file="fitting_results.xlsx"):
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        wb = openpyxl.Workbook()
        
        if successful_results:
            ws_success = wb.active
            ws_success.title = "Успешные подгонки"
            
            df_success = pd.DataFrame(successful_results)
            headers = ['Файл', 'Дата', 'Амплитуда (A)', 'Погрешность A', 
                      'Коэффициент масштаба (k)', 'Погрешность k',
                      'Центр пика', 'Погрешность центра', 'Площадь пика', 'Сходимость IRLS']
            ws_success.append(headers)
            
            for _, row in df_success.iterrows():
                ws_success.append([
                    row['file'],
                    row['date'],
                    round(row['A'], 6),
                    round(row['A_err'], 6),
                    round(row['k'], 6),
                    round(row['k_err'], 6),
                    round(row['peak_center'], 2) if row['peak_success'] else "Не удалось",
                    round(row['peak_center_err'], 2) if row['peak_success'] else "Не удалось",
                    round(row['peak_area'], 2) if row['peak_success'] else "Не удалось",
                    ("Да" if row['irls_converged'] else "Лимит итераций") if self.use_poisson_mle else "—"
                ])
            
            for cell in ws_success[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal='center')
            
            for column in ws_success.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                ws_success.column_dimensions[column_letter].width = adjusted_width
        else:
            ws_success = wb.active
            ws_success.title = "Успешные подгонки"
            ws_success.append(["Нет успешных результатов подгонки"])
        
        if failed_results:
            ws_failed = wb.create_sheet("Неудачные подгонки")
            df_failed = pd.DataFrame(failed_results)
            df_failed = df_failed[['file', 'date', 'error']]
            headers = ['Файл', 'Дата', 'Ошибка']
            ws_failed.append(headers)
            for _, row in df_failed.iterrows():
                ws_failed.append([row['file'], row['date'], row['error']])
            for cell in ws_failed[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal='center')
            for column in ws_failed.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                ws_failed.column_dimensions[column_letter].width = adjusted_width
        
        ws_info = wb.create_sheet("Информация")
        info_data = [
            ["ОТЧЕТ О ПОДГОНКЕ СПЕКТРОВ"],
            [""],
            ["Шаблонный спектр:", template_file],
            ["Папка со спектрами:", spectra_folder],
            ["Папка с графиками:", self.plots_folder if self.save_spectra else "Сохранение отключено"],
            ["Сохранение спектров:", "Включено" if self.save_spectra else "Отключено"],
            ["Метод фитинга:", "Пуассоновский MLE (итеративный)" if self.use_poisson_mle else "Обычный (фиксированные веса)"],
            ["Дата обработки:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
            [""],
            ["СТАТИСТИКА:"],
            ["Всего обработано файлов:", len(results)],
            ["Успешных подгонок:", len(successful_results)],
            ["Неудачных подгонок:", len(failed_results)],
            [""]
        ]
        
        if successful_results:
            df_success = pd.DataFrame(successful_results)
            stats = [
                ["Средняя амплитуда (A):", round(df_success['A'].mean(), 6)],
                ["Стандартное отклонение A:", round(df_success['A'].std(), 6)],
                ["Средний коэффициент масштаба (k):", round(df_success['k'].mean(), 6)],
                ["Стандартное отклонение k:", round(df_success['k'].std(), 6)],
                ["Минимальный k:", round(df_success['k'].min(), 6)],
                ["Максимальный k:", round(df_success['k'].max(), 6)]
            ]
            
            successful_peaks = [r for r in successful_results if r.get('peak_success', False)]
            if successful_peaks:
                df_peaks = pd.DataFrame(successful_peaks)
                peak_stats = [
                    [""],
                    ["СТАТИСТИКА ПО ПИКАМ (209-251 каналы):"],
                    ["Средний центр пика:", f"{df_peaks['peak_center'].mean():.2f} ± {df_peaks['peak_center'].std():.2f}"],
                    ["Средняя площадь пика:", f"{df_peaks['peak_area'].mean():.0f} ± {df_peaks['peak_area'].std():.0f}"],
                    ["Успешных фитирований пиков:", len(successful_peaks)]
                ]
                stats.extend(peak_stats)
            
            info_data.extend(stats)
        
        for row in info_data:
            ws_info.append(row)
        
        ws_info['A1'].font = Font(bold=True, size=14)
        
        for column in ws_info.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws_info.column_dimensions[column_letter].width = adjusted_width
        
        wb.save(output_file)
        return output_file

class SpectrumAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.template_file = ""
        self.spectra_folder = ""
        self.plots_folder = ""
        self.save_spectra = True
        self.use_poisson_mle = False  # По умолчанию используем обычный метод
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("QC analyzer")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Группа выбора файлов
        files_group = QGroupBox("QC Files")
        self.files_group = files_group
        files_layout = QVBoxLayout(files_group)
        
        # Выбор шаблонного спектра
        template_layout = QHBoxLayout()
        self.template_label = QLabel("Template spectrum: not selected")
        template_btn = QPushButton("Reference spectrum")
        template_btn.clicked.connect(self.select_template)
        template_layout.addWidget(self.template_label)
        template_layout.addWidget(template_btn)
        files_layout.addLayout(template_layout)
        
        # Выбор папки со спектрами
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Spectra folder: not selected")
        folder_btn = QPushButton("Folder")
        folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(folder_btn)
        files_layout.addLayout(folder_layout)
        
        # Чекбокс для сохранения спектров
        checkbox_layout = QHBoxLayout()
        self.save_checkbox = QCheckBox("Save spectra to folder")
        self.save_checkbox.setChecked(True)
        self.save_checkbox.stateChanged.connect(self.on_checkbox_changed)
        checkbox_layout.addWidget(self.save_checkbox)
        checkbox_layout.addStretch()
        files_layout.addLayout(checkbox_layout)
        
        # НОВЫЙ ЧЕКБОКС: Пуассоновский MLE метод
        mle_layout = QHBoxLayout()
        self.mle_checkbox = QCheckBox("Use Poisson MLE (iterative reweighting)")
        self.mle_checkbox.setChecked(False)
        self.mle_checkbox.setToolTip("Использовать итеративный метод максимального правдоподобия для Пуассоновских данных")
        mle_layout.addWidget(self.mle_checkbox)
        mle_layout.addStretch()
        files_layout.addLayout(mle_layout)
        
        layout.addWidget(files_group)
        
        # Кнопка запуска
        self.run_btn = QPushButton("Run calculation")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Текстовое поле для вывода
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
    def on_checkbox_changed(self, state):
        self.save_spectra = (state == Qt.Checked)
        
    def select_template(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a template spectrum", "", "Spectrum files (*.phd *.PHD)")
        if file_path:
            self.template_file = file_path
            self.template_label.setText(f"Шаблонный спектр: {os.path.basename(file_path)}")
            self.check_ready()
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку со спектрами")
        if folder_path:
            self.spectra_folder = folder_path
            self.folder_label.setText(f"Папка со спектрами: {folder_path}")
            self.check_ready()
    
    def check_ready(self):
        if self.template_file and self.spectra_folder:
            self.run_btn.setEnabled(True)
        else:
            self.run_btn.setEnabled(False)
    
    def run_analysis(self):
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            return
        # Choose a persistent destination, never PyInstaller's temporary __file__ folder.
        default_name = os.path.join(self.spectra_folder, 'QC_results_' +
                                    datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.xlsx')
        output_file, _ = QFileDialog.getSaveFileName(
            self, "Сохранить Excel-отчет", default_name, "Excel (*.xlsx)")
        if not output_file:
            return
        if not output_file.lower().endswith('.xlsx'):
            output_file += '.xlsx'
        output_file = os.path.abspath(output_file)
        # Получаем состояние чекбокса MLE
        self.use_poisson_mle = self.mle_checkbox.isChecked()
        
        # Создаем папку для графиков ТОЛЬКО если чекбокс отмечен
        if self.save_spectra:
            self.plots_folder = str(Path(output_file).with_suffix('')) + '_plots' 
        else:
            self.plots_folder = ""
        
        self.run_btn.setEnabled(False)
        self.files_group.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.output_text.clear()
        
        self.analysis_thread = SpectrumAnalysisThread(
            self.template_file, 
            self.spectra_folder, 
            self.plots_folder, 
            self.save_spectra,
            self.use_poisson_mle,
            output_file=output_file
        )
        self.analysis_thread.progress.connect(self.progress_bar.setValue)
        self.analysis_thread.message.connect(self.append_message)
        self.analysis_thread.finished_signal.connect(self.analysis_finished)
        self.analysis_thread.error_signal.connect(self.analysis_error)
        self.analysis_thread.finished.connect(self.analysis_thread_stopped)
        self.analysis_thread.start()
    
    def append_message(self, message):
        self.output_text.append(message)
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)
    
    def analysis_finished(self, summary):
        self.append_message(summary)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Завершено", "Анализ успешно завершен!")
    
    def analysis_error(self, error_message):
        self.append_message(f"ОШИБКА: {error_message}")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Ошибка", error_message)

    def analysis_thread_stopped(self):
        self.progress_bar.setVisible(False)
        self.files_group.setEnabled(True)
        self.check_ready()

    def closeEvent(self, event):
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            QMessageBox.information(self, "Выполняется обработка",
                                    "Дождитесь завершения обработки перед закрытием окна.")
            event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description='QC MIX analyzer')
    parser.add_argument('--self-test', metavar='REPORT', help=argparse.SUPPRESS)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = SpectrumAnalyzerApp()
    window.show()
    if args.self_test:
        from qc_smoke import run_smoke
        return run_smoke(app, window, args.self_test)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
