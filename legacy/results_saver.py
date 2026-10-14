# results_saver.py
import numpy as np
from typing import Dict, Any
from constants import XENON_ISOTOPES, ROI_INFO
import os
import traceback

def save_results_to_file(results: Dict[str, Any], output_file: str = "results.txt") -> None:
    """Сохраняет все результаты в текстовый файл."""
    
    print(f"\n{'='*80}")
    print(f"🚀 DEBUG_SAVER: НАЧАЛО СОХРАНЕНИЯ")
    print(f"   Файл: {output_file}")
    print(f"   Полный путь: {os.path.abspath(output_file)}")
    print(f"   Директория: {os.path.dirname(os.path.abspath(output_file)) or '.'}")
    print(f"   Результаты keys: {list(results.keys())[:10]}...")
    
    try:
        # 1. Проверяем директорию
        full_path = os.path.abspath(output_file)
        dir_name = os.path.dirname(full_path) or '.'
        
        print(f"\n📁 Проверка директории:")
        print(f"   Директория: {dir_name}")
        print(f"   Существует: {os.path.exists(dir_name)}")
        
        if not os.path.exists(dir_name):
            print(f"   Создаем директорию...")
            os.makedirs(dir_name, exist_ok=True)
            print(f"   Директория создана: {os.path.exists(dir_name)}")
        
        # 2. Проверяем права на запись
        print(f"\n🔐 Проверка прав на запись:")
        if os.path.exists(full_path):
            print(f"   Файл уже существует: ДА")
            can_write = os.access(full_path, os.W_OK)
            print(f"   Можем писать в файл: {'ДА' if can_write else 'НЕТ'}")
            if not can_write:
                raise PermissionError(f"Нет прав на запись в файл {full_path}")
        else:
            print(f"   Файл не существует, будет создан")
            can_create = os.access(dir_name, os.W_OK)
            print(f"   Можем создать файл в директории: {'ДА' if can_create else 'НЕТ'}")
            if not can_create:
                raise PermissionError(f"Нет прав на создание файла в директории {dir_name}")
        
        # 3. Проверяем данные
        print(f"\n📊 Проверка данных:")
        if not results:
            raise ValueError("Нет данных для сохранения (пустой словарь results)")
        
        required_keys = ['sample_file', 'background_file', 'detector_file']
        missing_keys = [k for k in required_keys if k not in results]
        if missing_keys:
            print(f"   ВНИМАНИЕ: Отсутствуют ключи: {missing_keys}")
        
        # 4. ОТКРЫВАЕМ ФАЙЛ И ПИШЕМ
        print(f"\n💾 Начинаем запись в файл...")
        
        with open(full_path, 'w', encoding='utf-8') as f:
            print(f"   ✅ Файл успешно открыт для записи")
            
            # === ЗАГОЛОВОК ===
            f.write(f"{'='*80}\n")
            f.write("РЕЗУЛЬТАТЫ ПАРСИНГА ФАЙЛОВ IMS 2.0 (СИСТЕМАТИКА ВКЛЮЧЕНА)\n")
            
            # Определяем какой метод использовался
            method_used = results.get('regression_method_used', 'nnls')
            method_names = {'ls': 'LS (обычный)', 'nnls': 'NNLS (неотрицательный)', 'mle': 'MLE (Пуассоновский)'}
            f.write(f"Метод регрессии: {method_names.get(method_used, method_used)}\n")
            f.write("Используются погрешности ±σ из МНК + систематика\n")
            f.write(f"{'='*80}\n\n")
            print(f"   📝 Записали заголовок (метод: {method_used})")
            
            # === ФАЙЛЫ ===
            f.write("ФАЙЛЫ ДАННЫХ:\n")
            f.write("-"*40 + "\n")
            f.write(f"Файл пробы:           {results.get('sample_file', 'НЕИЗВЕСТНО')}\n")
            f.write(f"Файл фона:            {results.get('background_file', 'НЕИЗВЕСТНО')}\n")
            f.write(f"Файл фона детектора:  {results.get('detector_file', 'НЕИЗВЕСТНО')}\n\n")
            print(f"   📝 Записали информацию о файлах")
            
            # === ВРЕМЕННЫЕ ДАННЫЕ ===
            for title, key in [("1. ДАННЫЕ ПРОБЫ", 'sample_temporal'), 
                              ("2. ДАННЫЕ ФОНА", 'background_temporal')]:
                if key in results:
                    temp = results[key]
                    f.write(f"{title}\n")
                    f.write("-"*40 + "\n")
                    if temp.get('start_time'):
                        f.write(f"   Старт измерения:      {temp['start_time']}\n")
                    if temp.get('real_time'):
                        f.write(f"   Реальное время:       {temp['real_time']:.1f} сек\n")
                    if temp.get('live_time'):
                        f.write(f"   Живое время:          {temp['live_time']:.1f} сек\n")
                    if key == 'sample_temporal' and temp.get('air_volume'):
                        f.write(f"   Объем воздуха:        {temp['air_volume']} м³\n")
                    if key == 'sample_temporal' and temp.get('xenon_volume'):
                        f.write(f"   Объем ксенона:        {temp['xenon_volume']} см³\n")
                    f.write("\n")
            
            # === ВРЕМЕННЫЕ ИНТЕРВАЛЫ ===
            if 'time_intervals' in results:
                intervals = results['time_intervals']
                if 'between_start' in intervals:
                    f.write("3. ВРЕМЕННЫЕ ИНТЕРВАЛЫ\n")
                    f.write("-"*40 + "\n")
                    tau = intervals['between_start']
                    f.write(f"   τ (старт пробы - старт фона): {tau:.1f} сек\n")
                    f.write(f"   Направление:                {intervals.get('direction', 'не определено')}\n")
                    f.write("\n")
            
            if 'time_parameters' in results:
                tp = results['time_parameters']
                f.write(f"   t_a (время пробоотбора): {tp.get('t_a_hours', 0):.2f} ч\n")
                f.write(f"   t_p (время ожидания): {tp.get('t_p_hours', 0):.2f} ч\n")
                f.write("\n")
            
            print(f"   📝 Записали временные данные")
            
            # === ТАБЛИЦА ROI ОТСЧЕТОВ ===
            f.write("4. ТАБЛИЦА ОТСЧЕТОВ В ROI:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'ROI':>4}  {'Проба':>12}  {'Фон':>12}  {'Фон детектора':>15}\n")
            f.write("-"*70 + "\n")
            
            all_roi_numbers = set()
            for key in ['sample_roi_counts', 'background_roi_counts', 'detector_roi_counts']:
                if results.get(key):
                    all_roi_numbers.update(results[key].keys())
            
            total_counts = [0, 0, 0]
            for roi_num in sorted(all_roi_numbers):
                sample_counts = results.get('sample_roi_counts', {})
                background_counts = results.get('background_roi_counts', {})
                detector_counts = results.get('detector_roi_counts', {})
                
                counts = [
                    sample_counts.get(roi_num, 0) if sample_counts else 0,
                    background_counts.get(roi_num, 0) if background_counts else 0,
                    detector_counts.get(roi_num, 0) if detector_counts else 0
                ]
                f.write(f"{roi_num:4d}  {counts[0]:12.1f}  {counts[1]:12.1f}  {counts[2]:15.1f}\n")
                total_counts = [t + c for t, c in zip(total_counts, counts)]
            
            f.write("-"*70 + "\n")
            f.write(f"{'Σ':>4}  {total_counts[0]:12.1f}  {total_counts[1]:12.1f}  {total_counts[2]:15.1f}\n\n")
            print(f"   📝 Записали таблицу ROI")
            
            # === МАТРИЦЫ ДЛЯ ОТЛАДКИ ===
            f.write("5. МАТРИЦЫ ДЛЯ ОТЛАДКИ\n")
            f.write("="*80 + "\n")
            
            isotopes = list(XENON_ISOTOPES.keys())
            
            # Матрица F
            if results.get('F_matrix') is not None:
                f.write("Матрица F (фон → проба):\n")
                f.write("-"*40 + "\n")
                F = results['F_matrix']
                
                f.write("     " + " ".join(f"{iso:>12}" for iso in isotopes) + "\n")
                for i in range(len(isotopes)):
                    f.write(f"{isotopes[i]:<5}")
                    for j in range(len(isotopes)):
                        f.write(f"{F[i, j]:12.3e}")
                    f.write("\n")
                f.write("\n")
            else:
                f.write("Матрица F не создана\n")
            
            # Матрица H
            if results.get('H_matrix') is not None:
                f.write("\nМатрица H (нормировка):\n")
                f.write("-"*40 + "\n")
                H = results['H_matrix']
                
                f.write("     " + " ".join(f"{iso:>12}" for iso in isotopes) + "\n")
                for i in range(len(isotopes)):
                    f.write(f"{isotopes[i]:<5}")
                    for j in range(len(isotopes)):
                        if i == j:
                            f.write(f"{H[i, j]:12.3e}")
                        else:
                            f.write(f"{' ':12}")
                    f.write("\n")
                f.write("\n")
            else:
                f.write("Матрица H не создана\n")
            
            # Матрица Gamma
            if results.get('Gamma') is not None:
                f.write("\nМатрица Gamma (поправка на цепочку распада):\n")
                f.write("-"*40 + "\n")
                Gamma = results['Gamma']
                
                f.write("     " + " ".join(f"{iso:>12}" for iso in isotopes) + "\n")
                for i in range(len(isotopes)):
                    f.write(f"{isotopes[i]:<5}")
                    for j in range(len(isotopes)):
                        if abs(Gamma[i, j]) > 1e-12:
                            f.write(f"{Gamma[i, j]:12.3e}")
                        else:
                            f.write(f"{' ':12}")
                    f.write("\n")
                f.write("\n")
            else:
                f.write("Матрица Gamma не создана\n")
            
            print(f"   📝 Записали матрицы (F, H, Gamma)")
            
            # === УДЛИНЕННЫЙ ВЕКТОР (combined_vector) ===
            if results.get('combined_vector') and 'combined_vector' in results['combined_vector']:
                combined_vec = results['combined_vector']['combined_vector']
                if combined_vec is not None and len(combined_vec) > 0:
                    f.write("\n8. УДЛИНЕННЫЙ ВЕКТОР (combined_vector):\n")
                    f.write("="*80 + "\n")
                    f.write("Вектор наблюдений y (14 элементов):\n")
                    f.write("-"*40 + "\n")
                    
                    # Описание элементов
                    roi_list = [1, 2, 3, 5, 6, 7, 8]
                    descriptions = [f"Проба ROI{r}" for r in roi_list] + [f"Фон ROI{r}" for r in roi_list]
                    
                    for i, (val, desc) in enumerate(zip(combined_vec, descriptions)):
                        f.write(f"  {i+1:2d}. {desc:15} = {val:12.3e}\n")
                    f.write("\n")
                    print(f"   📝 Записали удлиненный вектор (combined_vector)")
            
            # === РЕЗУЛЬТАТЫ РЕГРЕССИИ ===
            if results.get('regression'):
                reg = results['regression']
                f.write("\n6. РЕЗУЛЬТАТЫ РЕГРЕССИИ\n")
                f.write("="*80 + "\n")
                f.write(f"Использованный метод: {reg.get('method', method_used).upper()}\n")
                f.write(f"Невязка NNLS: {reg.get('residual_nnls', 0):.6f}\n")
                f.write(f"Остаточная сумма квадратов (LS): {reg.get('RSS', 0):.6f}\n\n")
                
                # ========== ПОЛУЧАЕМ ОЦЕНКИ ИЗ ВЫБРАННОГО МЕТОДА ==========
                if 'selected_beta' in reg:
                    beta_vals = reg['selected_beta']
                    method_label = reg.get('selected_method', method_used).upper()
                    f.write(f"ОЦЕНКИ ПАРАМЕТРОВ β (пробные активности) - МЕТОД: {method_label}:\n")
                elif 'beta_nnls' in reg and 'beta_ls' in reg:
                    if method_used == 'ls':
                        beta_vals = reg['beta_ls']
                        f.write("ОЦЕНКИ ПАРАМЕТРОВ β (пробные активности) - МЕТОД: LS (обычный):\n")
                    elif method_used == 'nnls':
                        beta_vals = reg['beta_nnls']
                        f.write("ОЦЕНКИ ПАРАМЕТРОВ β (пробные активности) - МЕТОД: NNLS (неотрицательный):\n")
                    elif method_used == 'mle' and 'beta' in reg:
                        beta_vals = reg['beta']
                        f.write("ОЦЕНКИ ПАРАМЕТРОВ β (пробные активности) - МЕТОД: MLE (Пуассоновский):\n")
                    else:
                        beta_vals = reg.get('beta_nnls', [])
                        f.write("ОЦЕНКИ ПАРАМЕТРОВ β (пробные активности):\n")
                elif 'beta' in reg:
                    beta_vals = reg['beta']
                    f.write(f"ОЦЕНКИ ПАРАМЕТРОВ β (пробные активности) - МЕТОД: {reg.get('method', method_used).upper()}:\n")
                else:
                    beta_vals = []
                    f.write("ОЦЕНКИ ПАРАМЕТРОВ β (пробные активности):\n")
                
                f.write("-"*80 + "\n")
                f.write(f"{'Изотоп':<10} {'Значение β':>15} {'±σ(β)':>15} {'Отн. погр., %':>15}\n")
                f.write("-"*80 + "\n")
                
                std_beta = reg.get('selected_std_beta', reg.get('std_beta', []))
                if len(std_beta) == 0 and 'std_beta' in reg:
                    std_beta = reg['std_beta']
                
                iso_list = reg.get('isotopes', isotopes)
                
                for i, iso in enumerate(iso_list):
                    if i < len(beta_vals):
                        beta_val = beta_vals[i]
                        std_val = std_beta[i] if i < len(std_beta) else 0
                        rel_error = (std_val / abs(beta_val)) * 100 if abs(beta_val) > 1e-12 else 0.0
                        f.write(f"{iso:<10} {beta_val:15.3e} ±{std_val:14.3e} {rel_error:15.1f}\n")
                f.write("\n")
                
                # ========== gamma (фон) ==========
                f.write("ОЦЕНКИ ПАРАМЕТРОВ γ (вектор фона):\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Изотоп':<10} {'Значение γ':>15} {'±σ(γ)':>15} {'Отн. погр., %':>15}\n")
                f.write("-"*80 + "\n")
                
                if 'selected_gamma' in reg:
                    gamma_vals = reg['selected_gamma']
                elif 'gamma_nnls' in reg:
                    gamma_vals = reg['gamma_nnls']
                elif 'gamma' in reg:
                    gamma_vals = reg['gamma']
                else:
                    gamma_vals = []
                
                std_gamma = reg.get('std_gamma', [])
                
                for i, iso in enumerate(iso_list):
                    if i < len(gamma_vals):
                        gamma_val = gamma_vals[i]
                        std_val = std_gamma[i] if i < len(std_gamma) else 0
                        rel_error = (std_val / abs(gamma_val)) * 100 if abs(gamma_val) > 1e-12 else 0.0
                        f.write(f"{iso:<10} {gamma_val:15.3e} ±{std_val:14.3e} {rel_error:15.1f}\n")
                f.write("\n")
            
            print(f"   📝 Записали регрессию (с учетом выбранного метода)")
            
            # === ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ АКТИВНОСТЕЙ ===
            if results.get('all_activities'):
                all_act = results['all_activities']
                
                f.write(f"7. ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ РАСЧЕТА АКТИВНОСТЕЙ\n")
                f.write("="*80 + "\n\n")
                
                # Объемные параметры
                if 'volumes' in all_act:
                    volumes = all_act['volumes']
                    f.write("ОБЪЕМНЫЕ ПАРАМЕТРЫ:\n")
                    f.write("-"*40 + "\n")
                    f.write(f"  V_xe = {volumes.get('V_xe', 0):.3f} см³\n")
                    f.write(f"  V_eff = V_xe / {volumes.get('concentration_factor', 1):.3f} = {volumes.get('V_eff', 0):.3f} м³\n\n")
                
                # 1. Активность в детекторе
                f.write("1. АКТИВНОСТЬ В ДЕТЕКТОРЕ (Бк):\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Изотоп':<10} {'Значение':>15} {'±σ':>15} {'Отн. погр.':>15}\n")
                f.write("-"*80 + "\n")
                
                isotopes_names = results.get('regression', {}).get('isotopes', isotopes)
                
                for i, iso_name in enumerate(isotopes_names):
                    if iso_name in all_act.get('activities_in_detector', {}):
                        value = all_act['activities_in_detector'][iso_name]
                        std = all_act.get('std_activities', {}).get(iso_name, 0)
                        rel_err = all_act.get('relative_errors_activities', {}).get(iso_name, "—")
                        
                        if abs(value) < 1e-3:
                            f.write(f"{iso_name:<10} {value:15.3e} ±{std:14.3e} {rel_err:>15}\n")
                        else:
                            f.write(f"{iso_name:<10} {value:15.6f} ±{std:14.6f} {rel_err:>15}\n")
                
                f.write("\n")
                
                # 2. Удельная активность
                f.write("2. УДЕЛЬНАЯ АКТИВНОСТЬ (Бк/м³):\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Изотоп':<10} {'Значение':>15} {'±σ':>15} {'Отн. погр.':>15}\n")
                f.write("-"*80 + "\n")
                
                for i, iso_name in enumerate(isotopes_names):
                    if iso_name in all_act.get('specific_activities', {}):
                        value = all_act['specific_activities'][iso_name]
                        std = all_act.get('std_specific_activities', {}).get(iso_name, 0)
                        rel_err = all_act.get('relative_errors_specific', {}).get(iso_name, "—")
                        
                        if abs(value) < 1e-3:
                            f.write(f"{iso_name:<10} {value:15.3e} ±{std:14.3e} {rel_err:>15}\n")
                        else:
                            f.write(f"{iso_name:<10} {value:15.6f} ±{std:14.6f} {rel_err:>15}\n")
                
                f.write("\n")
                
                # 3. MDA
                if 'MDA' in all_act:
                    f.write("3. МИНИМАЛЬНАЯ ДЕТЕКТИРУЕМАЯ АКТИВНОСТЬ (MDA), мБк/м³:\n")
                    f.write("-"*80 + "\n")
                    f.write(f"{'Изотоп':<10} {'MDA, мБк/м³':>20}\n")
                    f.write("-"*80 + "\n")
                    
                    for iso_name in isotopes_names:
                        mda = all_act['MDA'].get(iso_name)
                        if mda is not None:
                            f.write(f"{iso_name:<10} {mda * 1000:20.3e}\n")
                    f.write("\n")
                
                # 4. Корреляционная матрица
                if 'correlation_matrices' in all_act:
                    corr_matrices = all_act['correlation_matrices']
                    
                    if 'activities' in corr_matrices:
                        f.write("4. КОРРЕЛЯЦИОННАЯ МАТРИЦА АКТИВНОСТЕЙ В ДЕТЕКТОРЕ:\n")
                        f.write("-"*80 + "\n")
                        corr_act = np.array(corr_matrices['activities'])
                        
                        f.write("     " + " ".join(f"{iso:>12}" for iso in isotopes_names) + "\n")
                        for i, iso in enumerate(isotopes_names):
                            f.write(f"{iso:<5}")
                            for j in range(len(isotopes_names)):
                                f.write(f"{corr_act[i, j]:12.4f}")
                            f.write("\n")
                        f.write("\n")
                    
                    if 'specific' in corr_matrices:
                        f.write("5. КОРРЕЛЯЦИОННАЯ МАТРИЦА УДЕЛЬНЫХ АКТИВНОСТЕЙ (Бк/м³):\n")
                        f.write("-"*80 + "\n")
                        corr_spec = np.array(corr_matrices['specific'])
                        
                        f.write("     " + " ".join(f"{iso:>12}" for iso in isotopes_names) + "\n")
                        for i, iso in enumerate(isotopes_names):
                            f.write(f"{iso:<5}")
                            for j in range(len(isotopes_names)):
                                f.write(f"{corr_spec[i, j]:12.4f}")
                            f.write("\n")
                        f.write("\n")
                
                print(f"   📝 Записали финальные результаты (активности, MDA, корреляции)")
            
            # === ВЕКТОР ОСТАТКОВ ===
            if results.get('regression') and results['regression'].get('residuals') is not None:
                residuals = results['regression']['residuals']
                if isinstance(residuals, np.ndarray):
                    f.write("\nВЕКТОР ОСТАТКОВ:\n")
                    f.write("="*80 + "\n")
                    f.write(np.array2string(residuals, suppress_small=True))
                    f.write("\n\n")
            
            # === ЗАВЕРШЕНИЕ ===
            f.write(f"{'='*80}\n")
            f.write("АНАЛИЗ ЗАВЕРШЕН\n")
            method_names_short = {'ls': 'LS', 'nnls': 'NNLS', 'mle': 'MLE'}
            f.write(f"Метод: {method_names_short.get(method_used, method_used)} оценки + МНК погрешности ±σ + систематика + MDA\n")
            f.write("β - проба, γ - фон\n")
            f.write(f"{'='*80}\n")
            print(f"   ✅ Запись в файл завершена")
        
        # 5. ПРОВЕРЯЕМ РЕЗУЛЬТАТ
        print(f"\n✅ СОХРАНЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
        print(f"   Файл: {full_path}")
        print(f"   Размер: {os.path.getsize(full_path)} байт")
        print(f"   Существует: {os.path.exists(full_path)}")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ СОХРАНЕНИИ:")
        print(f"   Тип ошибки: {type(e).__name__}")
        print(f"   Сообщение: {e}")
        print(f"   Traceback:")
        traceback.print_exc()
        print(f"{'='*80}\n")
        raise