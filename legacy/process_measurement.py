import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

# Импорты из ваших модулей
from ims_parser import parse_ims_spectral_file, get_roi_counts_vector
import calculation
from constants import XENON_ISOTOPES, ROI_INFO
from results_saver import save_results_to_file
from calculation import create_combined_vector, create_combined_vector2


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ДАННЫХ
# ============================================================================

def process_measurement_data(
    sample_file: str, 
    background_file: str, 
    detector_file: str, 
    use_fixed_roi: bool = False,
    regression_method: str = 'nnls'  # 'ls', 'nnls', 'mle', 'mle_free'
) -> Dict[str, Any]:
    """
    Основная функция обработки данных.
    
    Args:
        sample_file: Путь к файлу пробы
        background_file: Путь к файлу фона
        detector_file: Путь к файлу фона детектора
        use_fixed_roi: Если True, использовать фиксированные ROI из таблицы FIXED_ROI_TABLE
        regression_method: Метод регрессии 
            - 'ls': обычный метод наименьших квадратов
            - 'nnls': неотрицательный метод наименьших квадратов
            - 'mle': Пуассоновский MLE с ограничением x ≥ 0
            - 'mle_free': Пуассоновский MLE без ограничений
    
    Returns:
        Dict[str, Any]: Словарь с результатами обработки
    """
    print("=" * 80)
    print("НАЧАЛО ОБРАБОТКИ")
    print("=" * 80)
    
    # Показываем, какие ROI используются
    if use_fixed_roi:
        print(f"РЕЖИМ: Используются ФИКСИРОВАННЫЕ ROI из таблицы")
    else:
        print(f"РЕЖИМ: Используются ROI из файлов")
    
    print(f"РЕЖИМ РЕГРЕССИИ: {regression_method.upper()}")
    
    # 1. Парсинг файлов
    print("\n[1/6] Парсинг файлов...")
    parsed_files = {
        'sample': parse_ims_spectral_file(sample_file, is_sample=True),
        'background': parse_ims_spectral_file(background_file, is_sample=False),
        'detector': parse_ims_spectral_file(detector_file, is_sample=False)
    }
    
    # ============================================
    # ТОЛЬКО ВЫВОД ДЛЯ ОТЛАДКИ
    # ============================================
    print("\n" + "="*40)
    print("ОТЛАДКА: ДЕТЕКТОРЫ")
    print("="*40)
    
    detector_info = {}
    
    for key in ['sample', 'background', 'detector']:
        data = parsed_files[key]
        detector_id = data.get('detector_id')
        detector_name = data.get('detector_name')
        
        if detector_id:
            print(f"{key:12} -> ID: {detector_id:3}  Name: {detector_name}")
            detector_info[f'{key}_id'] = detector_id
            detector_info[f'{key}_name'] = detector_name
        else:
            print(f"{key:12} -> НЕ ОПРЕДЕЛЕН")
            detector_info[f'{key}_id'] = None
            detector_info[f'{key}_name'] = None
    
    print("="*40)
    
    def create_temporal_dict(parsed_data):
        return {
            'start_time': parsed_data.get('start_time'),
            'collection_start': parsed_data.get('collection_start'),
            'collection_end': parsed_data.get('collection_end'),
            'real_time': parsed_data.get('real_time'),
            'live_time': parsed_data.get('live_time'),
            'air_volume': parsed_data.get('air_volume'),
            'xenon_volume': parsed_data.get('xenon_volume')
        }
    
    # Базовые результаты
    results = {
        'sample_file': sample_file,
        'background_file': background_file,
        'detector_file': detector_file,
        'detector_info': detector_info,
        'regression_method_used': regression_method,
        'sample_temporal': create_temporal_dict(parsed_files['sample']),
        'background_temporal': create_temporal_dict(parsed_files['background']),
        'detector_temporal': create_temporal_dict(parsed_files['detector']),
        'sample_roi_counts': None,
        'background_roi_counts': None,
        'detector_roi_counts': None,
        'sample_efficiencies': parsed_files['sample'].get("efficiency", {}),
        'background_efficiencies': parsed_files['background'].get("efficiency", {}),
        'time_intervals': {},
        'use_fixed_roi': use_fixed_roi
    }
    
    # 2. ИЗВЛЕЧЕНИЕ ROI ОТСЧЕТОВ
    print("[2/6] Извлечение ROI отсчетов...")
    for key in ['sample', 'background', 'detector']:
        data = parsed_files[key]
        if data["has_histogram"] and data["num_roi"] > 0:
            try:
                results[f'{key}_roi_counts'] = get_roi_counts_vector(data, use_fixed_roi=use_fixed_roi)
                print(f"  {key}: извлечено {len(results[f'{key}_roi_counts'])} ROI")
                if use_fixed_roi:
                    print(f"    (использованы фиксированные ROI из таблицы)")
            except Exception as e:
                print(f"  Ошибка при извлечении ROI для {key}: {e}")
    
    # 3. РАСЧЕТ ВРЕМЕННЫХ ПАРАМЕТРОВ
    print("[3/6] Расчет временных параметров...")
    sample_start = results['sample_temporal']['start_time']
    bg_start = results['background_temporal']['start_time']
    
    if sample_start and bg_start:
        time_diff = (sample_start - bg_start).total_seconds()
        direction = "ФОН раньше" if time_diff > 0 else "ПРОБА раньше" if time_diff < 0 else "Одновременно"
        results['time_intervals']['between_start'] = time_diff
        results['time_intervals']['direction'] = direction
        print(f"  τ = {time_diff:.1f} сек ({direction})")
    
    # Рассчитываем t_a и t_p
    temp = results['sample_temporal']
    t_a = t_p = None
    if temp['collection_start'] and temp['collection_end'] and temp['start_time']:
        t_a = (temp['collection_end'] - temp['collection_start']).total_seconds()
        t_p = (temp['start_time'] - temp['collection_end']).total_seconds()
        results['t_a'] = t_a
        results['t_p'] = t_p
        print(f"  t_a = {t_a/3600:.2f} ч (время отбора)")
        print(f"  t_p = {t_p/3600:.2f} ч (время ожидания)")
    
    # 4. СОЗДАНИЕ МАТРИЦ
    print("[4/6] Создание матриц...")
    try:
        t_sl = results['sample_temporal'].get('live_time')
        t_gl = results['background_temporal'].get('live_time')
        t_sr = results['sample_temporal'].get('real_time')
        t_gr = results['background_temporal'].get('real_time')
        tau = results['time_intervals'].get('between_start', 0)
        
        # R матрицы
        if t_sl and t_gl:
            R_matrices = calculation.create_R_matrices(t_sl, t_gl)
            results.update(R_matrices)
            print("  ✓ Матрицы R созданы")
        
        # F матрица
        if all(v is not None for v in [t_sl, t_gl, t_sr, t_gr]):
            F_matrix = calculation.create_F_matrix(t_sl, t_gl, t_sr, t_gr, tau)
            results['F_matrix'] = F_matrix
            print("  ✓ Матрица F создана")
        
        # Матрицы поправок на время
        if t_a is not None and t_p is not None:
            results.update(calculation.create_chain_correction_matrix(t_p, t_a))
            results.update(calculation.create_Gp_matrix(t_p))
            results.update(calculation.create_Gs_matrix(t_a))
            print("  ✓ Матрицы поправок созданы")
        
        # H матрица
        if t_sl and t_sr and results['sample_efficiencies']:
            results['H_matrix'] = calculation.create_H_matrix(t_sl, t_sr, results['sample_efficiencies'])
            print("  ✓ Матрица H создана")
        
        # Временные параметры
        results['time_parameters'] = {
            't_sl': t_sl, 't_gl': t_gl, 't_sr': t_sr, 't_gr': t_gr,
            't_a': t_a, 't_p': t_p, 'tau': tau,
            't_a_hours': t_a/3600 if t_a else None,
            't_p_hours': t_p/3600 if t_p else None
        }
        
    except Exception as e:
        print(f"  ✗ Ошибка при создании матриц: {e}")
        results.update({
            'R_c': None, 'R_g1': None, 'R_g2': None,
            'F_matrix': None, 'H_matrix': None,
            'Gamma': np.eye(5), 'Gp': np.eye(5), 'Gs': np.eye(5)
        })
    
    # 5. РЕГРЕССИЯ С ПОГРЕШНОСТЯМИ
    print("[5/6] Регрессия с погрешностями...")
    if (results['sample_roi_counts'] is not None and 
        results['background_roi_counts'] is not None and 
        results['detector_roi_counts'] is not None and
        results['sample_temporal'].get('live_time') and
        results['background_temporal'].get('live_time') and
        results['detector_temporal'].get('live_time') and
        results.get('R_c') is not None and
        results.get('R_g1') is not None and
        results.get('R_g2') is not None and
        results.get('F_matrix') is not None):
        
        try:
            # Combined vector
            combined = create_combined_vector(
                sample_counts=results['sample_roi_counts'],
                background_counts=results['background_roi_counts'],
                detector_counts=results['detector_roi_counts'],
                sample_time=results['sample_temporal']['live_time'],
                background_time=results['background_temporal']['live_time'],
                detector_time=results['detector_temporal']['live_time']
            )
            results['combined_vector'] = combined
            
            # Combined vector2 (c и b раздельно)
            combined2 = create_combined_vector2(
                sample_counts=results['sample_roi_counts'],
                background_counts=results['background_roi_counts'],
                detector_counts=results['detector_roi_counts'],
                sample_time=results['sample_temporal']['live_time'],
                background_time=results['background_temporal']['live_time'],
                detector_time=results['detector_temporal']['live_time']
            )
            results['combined_vector2'] = combined2

            # ============================================
            # MLE (с ограничениями) - всегда считаем для информации
            # ============================================
            regression_result_mle = calculation.solve_poisson_mle_for_spectrum(
                R_c=results['R_c'],
                R_g1=results['R_g1'],
                R_g2=results['R_g2'],
                F=results['F_matrix'],
                c_vector=results['combined_vector2']['c'],
                b_vector=results['combined_vector2']['b'],
                nonnegative=True
            )
            results['regression_mle_full'] = regression_result_mle
            print("  ✓ MLE (с ограничениями) выполнена")

            # ============================================
            # MLE (без ограничений) - всегда считаем для информации
            # ============================================
            regression_result_mle_free = calculation.solve_poisson_mle_for_spectrum(
                R_c=results['R_c'],
                R_g1=results['R_g1'],
                R_g2=results['R_g2'],
                F=results['F_matrix'],
                c_vector=results['combined_vector2']['c'],
                b_vector=results['combined_vector2']['b'],
                nonnegative=False,
                use_ls_initial=True  # ← используем LS как начальное приближение

            )
            results['regression_mle_free_full'] = regression_result_mle_free
            print("  ✓ MLE (без ограничений) выполнена")

            # ============================================
            # NNLS + LS (всегда считаем, для информации)
            # ============================================
            regression_result = calculation.solve_nnls_regression_with_simple_errors(
                R_c=results['R_c'],
                R_g1=results['R_g1'],
                R_g2=results['R_g2'],
                F=results['F_matrix'],
                c_vector=results['combined_vector']['combined_vector'],
                sample_counts=results['sample_roi_counts'],
                background_counts=results['background_roi_counts'],
                confidence_level=0.95
            )
            results['regression_nnls_full'] = regression_result
            
            # Сохраняем полный результат NNLS (с A, W_new и т.д.)
            results['regression'] = regression_result
            print("  ✓ NNLS регрессия выполнена")
            
            # ============================================
            # ВЫБОР МЕТОДА ДЛЯ АКТИВНОСТЕЙ
            # ============================================
            print(f"\n  ВЫБРАН МЕТОД: {regression_method.upper()}")
            
            if regression_method == 'ls':
                results['regression_selected'] = {
                    'beta': regression_result['beta_ls'],
                    'gamma': regression_result['gamma_ls'],
                    'std_beta': regression_result['std_beta'],
                    'std_gamma': regression_result['std_gamma'],
                    'residuals': regression_result['residuals_vec'],
                    'cov_beta': regression_result['cov_beta'],
                    'method': 'ls',
                    'isotopes': regression_result['isotopes']
                }
                print(f"    Использую beta_ls (обычный LS)")
                
            elif regression_method == 'nnls':
                results['regression_selected'] = {
                    'beta': regression_result['beta_nnls'],
                    'gamma': regression_result['gamma_nnls'],
                    'std_beta': regression_result['std_beta'],
                    'std_gamma': regression_result['std_gamma'],
                    'residuals': regression_result['residuals_vec'],
                    'cov_beta': regression_result['cov_beta'],
                    'method': 'nnls',
                    'isotopes': regression_result['isotopes']
                }
                print(f"    Использую beta_nnls (неотрицательный LS)")
                
            elif regression_method == 'mle':
                results['regression_selected'] = {
                    'beta': regression_result_mle['beta'],
                    'gamma': regression_result_mle['gamma'],
                    'std_beta': regression_result_mle['beta_std'],
                    'std_gamma': regression_result_mle['gamma_std'],
                    'residuals': None,
                    'cov_beta': regression_result_mle.get('cov_beta'),
                    'method': 'mle',
                    'isotopes': regression_result['isotopes']
                }
                print(f"    Использую beta_mle (Poisson MLE с ограничениями)")
                
            elif regression_method == 'mle_free':
                results['regression_selected'] = {
                    'beta': regression_result_mle_free['beta'],
                    'gamma': regression_result_mle_free['gamma'],
                    'std_beta': regression_result_mle_free['beta_std'],
                    'std_gamma': regression_result_mle_free['gamma_std'],
                    'residuals': None,
                    'cov_beta': regression_result_mle_free.get('cov_beta'),
                    'method': 'mle_free',
                    'isotopes': regression_result['isotopes']
                }
                print(f"    Использую beta_mle (Poisson MLE без ограничений)")
            
            # Добавляем выбранный метод в полный regression
            results['regression']['selected_beta'] = results['regression_selected']['beta']
            results['regression']['selected_gamma'] = results['regression_selected']['gamma']
            results['regression']['selected_std_beta'] = results['regression_selected']['std_beta']
            results['regression']['selected_std_gamma'] = results['regression_selected']['std_gamma']
            results['regression']['selected_method'] = results['regression_selected']['method']
            results['regression']['selected_residuals'] = results['regression_selected']['residuals']
            results['regression']['selected_cov_beta'] = results['regression_selected']['cov_beta']
            
            # Для батч-скрипта
            results['regression']['residuals'] = results['regression_selected']['residuals']
            
        except Exception as e:
            print(f"  ✗ Ошибка при регрессии: {e}")
            traceback.print_exc()
            results['regression'] = None
    else:
        print("  ✗ Недостаточно данных для регрессии")
        missing_data = []
        if results['sample_roi_counts'] is None: missing_data.append('sample_roi_counts')
        if results['background_roi_counts'] is None: missing_data.append('background_roi_counts')
        if results['detector_roi_counts'] is None: missing_data.append('detector_roi_counts')
        if results.get('R_c') is None: missing_data.append('R_c')
        print(f"    Отсутствуют: {', '.join(missing_data)}")
        results['regression'] = None
    
    # 6. РАСЧЕТ АКТИВНОСТЕЙ С СИСТЕМАТИЧЕСКИМИ ПОГРЕШНОСТЯМИ
    print("[6/6] Расчет активностей с систематическими погрешностями...")
    if (results.get('regression') and 
        results.get('H_matrix') is not None and
        results.get('Gamma') is not None and
        results.get('Gp') is not None and
        results.get('Gs') is not None and
        results['sample_temporal'].get('xenon_volume')):
        
        print("  ✓ Все необходимые данные для расчета активностей есть")
        
        try:
            print("  Вызываю calculate_all_activities_with_systematic_errors...")
            all_activities = calculation.calculate_all_activities_with_systematic_errors(results)
            
            if all_activities:
                print(f"  ✓ calculate_all_activities_with_systematic_errors вернул результат")
                print(f"  Ключи: {list(all_activities.keys())}")
                if 'MDA' in all_activities:
                    print(f"  MDA найден! Значения: {all_activities['MDA']}")
                results['all_activities'] = all_activities
            else:
                print("  ✗ calculate_all_activities_with_systematic_errors вернул None")
                
        except Exception as e:
            print(f"  ✗ Ошибка при расчете всех активностей: {e}")
            traceback.print_exc()
    else:
        print("  ✗ Недостаточно данных для расчета активностей")
        missing = []
        if not results.get('regression'): missing.append('regression')
        if results.get('H_matrix') is None: missing.append('H_matrix')
        if results.get('Gamma') is None: missing.append('Gamma')
        if results.get('Gp') is None: missing.append('Gp')
        if results.get('Gs') is None: missing.append('Gs')
        if not results['sample_temporal'].get('xenon_volume'): missing.append('xenon_volume')
        print(f"    Отсутствуют: {missing}")
    
    print("\n" + "=" * 80)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80)
    
    return results


# ============================================================================
# ОБЕРТКА ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ
# ============================================================================

def parse_sample_and_background_and_detector(
    sample_file: str, 
    background_file: str, 
    detector_file: str, 
    use_fixed_roi: bool = False,
    regression_method: str = 'nnls'
) -> Dict[str, Any]:
    """
    Обертка для обратной совместимости со старым кодом.
    """
    return process_measurement_data(
        sample_file, 
        background_file, 
        detector_file, 
        use_fixed_roi,
        regression_method
    )