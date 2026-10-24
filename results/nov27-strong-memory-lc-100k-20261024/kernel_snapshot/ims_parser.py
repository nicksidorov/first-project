"""
Парсер файлов IMS 2.0 для программы анализа активности ксенона.
"""

import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any
from constants import FIXED_ROI_TABLE
def parse_ims_spectral_file(file_path: str, is_sample: bool = False) -> Dict[str, Any]:
    """Парсит файл IMS 2.0, извлекая все данные."""
    result = {
        'start_time': None,
        'collection_start': None,
        'collection_end': None,
        'real_time': None,
        'live_time': None,
        'air_volume': None,
        'xenon_volume': None,
        'histogram': None,
        'g_energy': [],
        'b_energy': [],
        'roi_limits': [],
        'g_spectrum': [],
        'b_spectrum': [],
        'roi_numbers': [],
        'efficiency': {},
        'has_histogram': False,
        'num_roi': 0,
        'num_g_energy_points': 0,
        'num_b_energy_points': 0,
        'has_efficiency': False,
        'detector_id': None,
        'detector_name': None,
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        current_section = None
        histogram_lines, spectrum_lines = [], []
        spectrum_count = 0
        
        def find_next_non_empty(start_idx):
            """Находит следующую непустую строку после start_idx."""
            for j in range(start_idx + 1, len(lines)):
                stripped = lines[j].strip()
                if stripped:
                    return stripped
            return None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            # ============================================
            # ОБРАБОТКА ВСЕХ ЗАГОЛОВКОВ С ПУСТЫМИ СТРОКАМИ
            # ============================================
            if stripped.startswith("#"):
                # Временные данные
                if stripped == "#Collection":
                    next_line = find_next_non_empty(i)
                    if next_line:
                        data_parts = next_line.split()
                        if len(data_parts) >= 5:
                            try:
                                start_str = f"{data_parts[0]} {data_parts[1]}"
                                end_str = f"{data_parts[2]} {data_parts[3]}"
                                
                                for fmt in ['%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S']:
                                    try:
                                        result['collection_start'] = datetime.strptime(start_str, fmt)
                                        result['collection_end'] = datetime.strptime(end_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                                
                                if len(data_parts) >= 5:
                                    try:
                                        result['air_volume'] = float(data_parts[4])
                                    except ValueError:
                                        pass
                            except Exception:
                                pass
                
                elif stripped == "#Acquisition":
                    next_line = find_next_non_empty(i)
                    if next_line:
                        data_parts = next_line.split()
                        if len(data_parts) >= 3:
                            try:
                                time_str = f"{data_parts[0]} {data_parts[1]}"
                                for fmt in ['%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S']:
                                    try:
                                        result['start_time'] = datetime.strptime(time_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                                
                                try:
                                    if len(data_parts) >= 4:
                                        result['real_time'] = float(data_parts[2])
                                        result['live_time'] = float(data_parts[3])
                                    elif len(data_parts) >= 3:
                                        result['real_time'] = float(data_parts[2])
                                except ValueError:
                                    pass
                            except Exception:
                                pass
                
                elif stripped == "#Header 3":
                    next_line = find_next_non_empty(i)
                    if next_line:
                        parts = next_line.split()
                        if len(parts) >= 2:
                            detector_name = parts[1]  # MIX58_221
                            result['detector_name'] = detector_name
                            
                            if '_' in detector_name:
                                last_part = detector_name.split('_')[-1]
                                if last_part.isdigit():
                                    result['detector_id'] = int(last_part)
                
                elif stripped == "#Processing" and is_sample:
                    # Для Processing может быть несколько строк
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                        if next_line.startswith('#'):
                            break
                        
                        parts = next_line.split()
                        if len(parts) >= 2:
                            try:
                                result['xenon_volume'] = float(parts[0])
                                break
                            except ValueError:
                                pass
                        j += 1
                
                # Спектральные секции
                sections = {
                    "#Histogram": "histogram",
                    "#g_Energy": "g_energy",
                    "#b_Energy": "b_energy",
                    "#ROI_Limits": "roi_limits",
                    "#g_Spectrum": "g_spectrum",
                    "#b_Spectrum": "b_spectrum",
                    "#b-gEfficiency": "efficiency"
                }
                
                if stripped in sections:
                    current_section = sections[stripped]
                    if current_section in ["histogram", "g_spectrum", "b_spectrum"]:
                        histogram_lines = [] if current_section == "histogram" else []
                        spectrum_lines = []
                        spectrum_count = 0
                    continue
            
            # ============================================
            # ОБРАБОТКА ДАННЫХ СЕКЦИЙ (без изменений)
            # ============================================
            if current_section == "g_energy":
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        result["g_energy"].append(tuple(map(float, parts[:3])))
                    except ValueError:
                        pass
            
            elif current_section == "b_energy":
                parts = stripped.split()
                if len(parts) >= 4 and parts[1] == "C":
                    try:
                        result["b_energy"].append((float(parts[0]), float(parts[2]), float(parts[3])))
                    except ValueError:
                        pass
            
            elif current_section == "roi_limits":
                parts = stripped.split()
                if len(parts) >= 5:
                    try:
                        result["roi_limits"].append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                        try:
                            result["roi_numbers"].append(int(parts[0]))
                        except ValueError:
                            result["roi_numbers"].append(len(result["roi_limits"]))
                    except ValueError:
                        pass
            
            elif current_section == "histogram":
                if stripped == "STOP":
                    current_section = None
                    continue
                
                if 'metadata_processed' not in locals():
                    metadata_processed = True
                    continue
                
                if len(histogram_lines) < 256:
                    elements = stripped.split()
                    if len(elements) != 256:
                        current_section = None
                        histogram_lines = []
                        continue
                    histogram_lines.append(elements)
                
                if len(histogram_lines) == 256:
                    try:
                        matrix = np.array(histogram_lines, dtype=float)
                        if matrix.shape == (256, 256):
                            result["histogram"] = matrix
                            result["has_histogram"] = True
                    finally:
                        current_section = None
            
            elif current_section in ["g_spectrum", "b_spectrum"]:
                if spectrum_count < 52:
                    try:
                        row = list(map(int, stripped.split()))
                        spectrum_lines.append(row)
                        spectrum_count += 1
                    except ValueError:
                        pass
                
                if spectrum_count == 52:
                    max_len = max(len(r) for r in spectrum_lines)
                    padded = [r + [0] * (max_len - len(r)) for r in spectrum_lines]
                    arr = np.array(padded)[1:, 1:].flatten()
                    arr = np.append(arr, 0)
                    result[current_section] = arr
            
            elif current_section == "efficiency":
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        roi_number = int(parts[1])
                        efficiency_value = float(parts[2])
                        isotope_name = parts[0].strip().upper()
                        
                        if isotope_name not in ['RN222', 'RN-222']:
                            if 0.0 <= efficiency_value <= 1.0:
                                result["efficiency"][roi_number] = efficiency_value
                        
                        if 'roi_to_isotope' not in result:
                            result['roi_to_isotope'] = {}
                        result['roi_to_isotope'][roi_number] = isotope_name
                        result["has_efficiency"] = True
                            
                    except (ValueError, IndexError):
                        pass
        
        # Заполняем roi_numbers если они пустые
        if not result["roi_numbers"] and result["roi_limits"]:
            result["roi_numbers"] = list(range(1, len(result["roi_limits"]) + 1))
        
        # Обновляем сводные поля
        result["num_roi"] = len(result["roi_limits"])
        result["num_g_energy_points"] = len(result["g_energy"])
        result["num_b_energy_points"] = len(result["b_energy"])
        
        # Вывод для отладки
        if result['detector_id']:
            print(f"[ПАРСЕР] {file_path}: детектор {result['detector_name']} (ID: {result['detector_id']})")
        
        return result
        
    except FileNotFoundError:
        raise ValueError(f"Файл не найден: {file_path}")
    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла {file_path}: {e}")


def fit_energy_calibration(g_energy_data: List[Tuple], b_energy_data: List[Tuple]) -> Tuple:
    """Создает калибровочные функции для перевода энергии в каналы."""
    if not g_energy_data or not b_energy_data:
        raise ValueError("Недостаточно данных для калибровки")
    
    g_energies = [row[0] for row in g_energy_data]
    g_channels = [row[1] for row in g_energy_data]
    b_energies = [row[0] for row in b_energy_data]
    b_channels = [row[1] for row in b_energy_data]

    g_coeffs = np.polyfit(g_energies, g_channels, 2)
    b_coeffs = np.polyfit(b_energies, b_channels, 2)

    return lambda e: np.polyval(g_coeffs, e), lambda e: np.polyval(b_coeffs, e)


def convert_roi_energy_to_channels(roi_energy_matrix: List[List[float]], 
                                   gamma_func, beta_func) -> np.ndarray:
    """Конвертирует границы ROI из энергии в каналы."""
    return np.array([
        [beta_func(row[0]), beta_func(row[1]), gamma_func(row[2]), gamma_func(row[3])]
        for row in roi_energy_matrix
    ])


def get_roi_counts_vector(
    spectral_data: Dict[str, Any], 
    use_fixed_roi: bool = False
) -> Dict[int, float]:
    """Возвращает словарь отсчетов для всех ROI в спектральных данных."""
    # Проверка
    if spectral_data.get("histogram") is None:
        raise ValueError("Гистограмма не загружена")
    
    # Калибровка
    gamma_to_ch, beta_to_ch = fit_energy_calibration(
        spectral_data["g_energy"], 
        spectral_data["b_energy"]
    )
    
    # Выбираем какие ROI использовать
    if use_fixed_roi:
        # Используем фиксированные ROI из таблицы
        roi_limits = list(FIXED_ROI_TABLE.values())
        roi_numbers = list(FIXED_ROI_TABLE.keys())
    else:
        # Используем ROI из файла
        if not spectral_data.get("roi_limits"):
            raise ValueError("ROI не найдены в файле")
        roi_limits = spectral_data["roi_limits"]
        roi_numbers = spectral_data.get("roi_numbers", [])
    
    # Конвертация энергий в каналы
    roi_channels = convert_roi_energy_to_channels(
        roi_limits, 
        gamma_to_ch, 
        beta_to_ch
    )
    
    # Подсчет
    histogram = spectral_data["histogram"]
    roi_counts = {}
    
    for idx, (beta_min, beta_max, gamma_min, gamma_max) in enumerate(roi_channels):
        roi_number = roi_numbers[idx] if idx < len(roi_numbers) else idx + 1
        
        # Преобразуем в целые числа с ограничением
        beta_min_int = int(np.clip(np.round(beta_min), 0, 255))
        beta_max_int = int(np.clip(np.round(beta_max), 0, 255))
        gamma_min_int = int(np.clip(np.round(gamma_min), 0, 255))
        gamma_max_int = int(np.clip(np.round(gamma_max), 0, 255))
        
        # Корректируем если min > max
        if beta_min_int > beta_max_int:
            beta_min_int, beta_max_int = beta_max_int, beta_min_int
        if gamma_min_int > gamma_max_int:
            gamma_min_int, gamma_max_int = gamma_max_int, gamma_min_int
        
        # Суммируем
        roi_region = histogram[gamma_min_int:gamma_max_int+1, beta_min_int:beta_max_int+1]
        roi_counts[roi_number] = float(np.sum(roi_region))
    
    return roi_counts