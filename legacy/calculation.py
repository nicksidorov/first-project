# ============================================================================
# МАТЕМАТИЧЕСКИЕ ФУНКЦИИ
# ============================================================================
import numpy as np
from scipy.optimize import nnls, minimize, Bounds
from typing import Dict, Any, Optional, List, Tuple
from constants import XENON_ISOTOPES, ROI_INFO, Rs, Rg, C1, DELTA_V_XE, DELTA_EFF
from scipy import stats
import warnings


def create_combined_vector2(
    sample_counts: Dict[int, float], 
    background_counts: Dict[int, float],
    detector_counts: Dict[int, float], 
    sample_time: float,
    background_time: float, 
    detector_time: float
) -> Dict[str, Any]:
    """
    Создает два вектора длиной 14:
    1. Вектор наблюдений c (7 проба + 7 газовый фон)
    2. Вектор детекторного фона b (умноженный на коэффициенты времени)
    
    Параметры:
    ----------
    sample_counts : Dict[int, float]
        Отсчеты пробы по ROI
    background_counts : Dict[int, float]
        Отсчеты газового фона по ROI
    detector_counts : Dict[int, float]
        Отсчеты детекторного фона по ROI
    sample_time : float
        Время измерения пробы (сек)
    background_time : float
        Время измерения газового фона (сек)
    detector_time : float
        Время измерения детекторного фона (сек)
    
    Возвращает:
    -----------
    Dict[str, Any]:
        - 'c': вектор наблюдений (длина 14)
        - 'b': вектор детекторного фона (длина 14)
        - 'sample_factor': коэффициент времени для пробы
        - 'background_factor': коэффициент времени для фона
        - 'description': описание элементов векторов
    """
    
    if detector_time <= 0:
        raise ValueError("Время измерения детекторного фона должно быть > 0")
    
    # Коэффициенты пересчета
    sample_factor = sample_time / detector_time
    background_factor = background_time / detector_time
    
    # Список ROI
    roi_list = [1, 2, 3, 5, 6, 7, 8]
    
    # Инициализируем векторы
    c_vector = []  # Вектор наблюдений (проба + газовый фон)
    b_vector = []  # Вектор детекторного фона
    
    # Словари для исправленных значений (если нужно)
    sample_corr = {}
    background_corr = {}
    detector_sample = {}
    detector_background = {}
    
    # 1. Проба (первые 7 элементов)
    for roi in roi_list:
        # Наблюдения пробы
        sample_val = sample_counts.get(roi, 0.0)
        c_vector.append(sample_val)
        
        # Детекторный фон для пробы
        detector_val = detector_counts.get(roi, 0.0)
        b_detector_sample = detector_val * sample_factor
        b_vector.append(b_detector_sample)
        
        # Сохраняем для отладки
        detector_sample[roi] = b_detector_sample
        sample_corr[roi] = sample_val - b_detector_sample
    
    # 2. Газовый фон (следующие 7 элементов)
    for roi in roi_list:
        # Наблюдения газового фона
        bg_val = background_counts.get(roi, 0.0)
        c_vector.append(bg_val)
        
        # Детекторный фон для газового фона
        detector_val = detector_counts.get(roi, 0.0)
        b_detector_bg = detector_val * background_factor
        b_vector.append(b_detector_bg)
        
        # Сохраняем для отладки
        detector_background[roi] = b_detector_bg
        background_corr[roi] = max(0, bg_val - b_detector_bg)
    
    # Описание элементов
    description = [f"Проба ROI{r}" for r in roi_list] + [f"Газовый фон ROI{r}" for r in roi_list]
    
    return {
        'c': np.array(c_vector, dtype=float),          # Вектор наблюдений (14)
        'b': np.array(b_vector, dtype=float),          # Вектор детекторного фона (14)
        'sample_factor': sample_factor,
        'background_factor': background_factor,
        'sample_corrected': sample_corr,               # Проба - детекторный фон
        'background_corrected': background_corr,       # Газ.фон - детекторный фон
        'detector_sample': detector_sample,            # Дет.фон для пробы
        'detector_background': detector_background,    # Дет.фон для газ.фона
        'description': description,
        'roi_list': roi_list
    }




# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================
def create_combined_vector(
    sample_counts: Dict[int, float], background_counts: Dict[int, float],
    detector_counts: Dict[int, float], sample_time: float,
    background_time: float, detector_time: float
) -> Dict[str, Any]:
    """
    Создает объединенный вектор из отсчетов пробы и фона.
    
    ВАЖНО: НЕ обрезает отрицательные значения!
    Отрицательные значения - это реальная информация,
    которая должна быть передана в регрессию.
    """
    if not detector_time or detector_time <= 0:
        raise ValueError("Не указано время измерения фона детектора")
    
    sample_factor = sample_time / detector_time
    background_factor = background_time / detector_time
    roi_list = [1, 2, 3, 5, 6, 7, 8]
    
    all_roi = set(list(sample_counts.keys()) + list(background_counts.keys()) + list(detector_counts.keys()))
    sample_corr, background_corr = {}, {}
    
    for roi in all_roi:
        detector = detector_counts.get(roi, 0.0)
        # ИСПРАВЛЕНО: убираем max(0, ...), оставляем как есть (могут быть отрицательные)
        sample_corr[roi] = sample_counts.get(roi, 0.0) - detector * sample_factor
        background_corr[roi] = background_counts.get(roi, 0.0) - detector * background_factor
    
    combined_vector = []
    for roi in roi_list:
        combined_vector.append(sample_corr.get(roi, 0.0))
    for roi in roi_list:
        combined_vector.append(background_corr.get(roi, 0.0))
    
    return {
        'combined_vector': np.array(combined_vector),
        'sample_corrected': sample_corr,
        'background_corrected': background_corr,
        'description': [f"Проба ROI{r}" for r in roi_list] + [f"Фон ROI{r}" for r in roi_list]
    }


def phi_T(T: float) -> float:
    """
    Рассчитывает функцию Φ(T) для поправки на распад.
    """
    lambda_3 = XENON_ISOTOPES['Xe133']
    lambda_5 = XENON_ISOTOPES['Xe133m']
    
    exp_lambda3_T = np.exp(-lambda_3 * T)
    exp_lambda5_T = np.exp(-lambda_5 * T)
    lambda_diff = lambda_5 - lambda_3
    
    numerator = exp_lambda3_T * lambda_5 - exp_lambda5_T * lambda_3
    first_part = 1 - numerator / lambda_diff
    second_part = 1 / (1 - exp_lambda5_T)
    
    return first_part * second_part * C1


def G_lambda_t(lambda_val: float, t_sec: float) -> float:
    """Рассчитывает функцию G(λ, t) = λt / (1 - exp(-λt))."""
    if t_sec is None or t_sec <= 0:
        return 1.0
    
    exponent = lambda_val * t_sec
    
    if exponent < 1e-12:
        return 1.0 + exponent/2.0 + exponent**2/12.0
    if exponent > 100:
        return exponent
    
    return exponent / (1.0 - np.exp(-exponent))


# ============================================================================
# ФУНКЦИИ СОЗДАНИЯ МАТРИЦ
# ============================================================================

def create_matrix_with_decay_correction(base_matrix: np.ndarray, T: float) -> np.ndarray:
    """Создает матрицу с поправкой на распад для времени T."""
    matrix = base_matrix.copy()
    matrix[:, 4] += phi_T(T) * base_matrix[:, 2]
    return matrix


def create_R_matrices(t_sl: float, t_gl: float) -> Dict[str, np.ndarray]:
    """Создает матрицы R_c, R_g1, R_g2 с поправкой на распад."""
    return {
        'R_c': create_matrix_with_decay_correction(Rs, t_sl),
        'R_g1': create_matrix_with_decay_correction(Rg, t_sl),
        'R_g2': create_matrix_with_decay_correction(Rg, t_gl)
    }


def create_F_matrix(t_sl: float, t_gl: float, t_sr: float, t_gr: float, tau: float) -> np.ndarray:
    """
    Создает матрицу учета фона F (5×5).
    """
    lambda_vals = list(XENON_ISOTOPES.values())
    F = np.zeros((5, 5))
    
    time_norm = t_sl / t_gl if t_gl else 1.0
    
    # Диагональные элементы
    for i, λ in enumerate(lambda_vals):
        G_bg = G_lambda_t(λ, t_gr)
        G_sample = G_lambda_t(λ, t_sr)
        G_ratio = G_bg / G_sample if G_sample != 0 else 1.0
        
        F[i, i] = np.exp(-λ * tau) * G_ratio * time_norm
    
    # Элемент для цепочки Xe133m→Xe133
    λ3 = lambda_vals[2]
    λ5 = lambda_vals[4]
    
    G_bg_5 = G_lambda_t(λ5, t_gr)
    G_sample_3 = G_lambda_t(λ3, t_sr)
    G_ratio_35 = G_bg_5 / G_sample_3 if G_sample_3 != 0 else 1.0
    
    λ_diff = λ5 - λ3
    exp_diff = np.exp(-λ3 * tau) - np.exp(-λ5 * tau)
    λ_ratio = λ3 / λ_diff if abs(λ_diff) > 1e-15 else 0.0
    
    F[2, 4] = G_ratio_35 * time_norm * C1 * λ_ratio * exp_diff
    
    return F


def create_chain_correction_matrix(t_p: float, t_a: float) -> Dict[str, Any]:
    """
    Создает матрицу поправки на цепочку распада Γ (5×5).
    """
    lambda_3 = XENON_ISOTOPES['Xe133']
    lambda_5 = XENON_ISOTOPES['Xe133m']
    lambda_diff = lambda_5 - lambda_3
    
    # Φ(t_a)
    exp_lambda3_ta = np.exp(-lambda_3 * t_a)
    exp_lambda5_ta = np.exp(-lambda_5 * t_a)
    numerator = exp_lambda3_ta * lambda_5 - exp_lambda5_ta * lambda_3
    phi_ta = (1 - numerator / lambda_diff) / (1 - exp_lambda5_ta)
    
    # γ
    exp_lambda_diff_tp = np.exp(lambda_diff * t_p)
    gamma_value = (lambda_3 / lambda_diff) * (exp_lambda_diff_tp - 1) + phi_ta * exp_lambda_diff_tp
    
    # Матрица Γ
    Gamma = np.eye(5)
    Gamma[2, 4] = -gamma_value
    
    return {
        'Gamma': Gamma,
        'gamma': gamma_value
    }


def create_Gp_matrix(t_p: float) -> Dict[str, Any]:
    """Создает диагональную матрицу поправки на время ожидания Gp (5×5)."""
    lambda_values = list(XENON_ISOTOPES.values())
    Gp = np.eye(5)
    
    for i, lambda_i in enumerate(lambda_values):
        Gp[i, i] = np.exp(lambda_i * t_p)
    
    return {
        'Gp': Gp,
        'Gp_info': {
            't_p': t_p,
            'isotopes': list(XENON_ISOTOPES.keys()),
            'lambda_values': lambda_values
        }
    }


def create_Gs_matrix(t_a: float) -> Dict[str, Any]:
    """Создает диагональную матрицу поправки на время пробоотбора Gs (5×5)."""
    lambda_values = list(XENON_ISOTOPES.values())
    Gs = np.eye(5)
    
    for i, lambda_i in enumerate(lambda_values):
        Gs[i, i] = G_lambda_t(lambda_i, t_a)
    
    return {
        'Gs': Gs,
        'Gs_info': {
            't_a': t_a,
            'isotopes': list(XENON_ISOTOPES.keys()),
            'lambda_values': lambda_values
        }
    }


def create_H_matrix(t_sl: float, t_sr: float, efficiencies_dict: Dict[int, float]) -> np.ndarray:
    """
    Создает диагональную матрицу нормировки H (5×5).
    """
    isotopes = list(XENON_ISOTOPES.keys())
    lambda_values = list(XENON_ISOTOPES.values())
    H = np.zeros((5, 5))
    
    for i, (iso, lambda_i) in enumerate(zip(isotopes, lambda_values)):
        roi_info = ROI_INFO[iso]
        roi_num = roi_info['roi']
        br = roi_info['br']
        
        # Поправка на распад за время измерения
        if t_sr and t_sr > 0:
            G_i = G_lambda_t(lambda_i, t_sr)
        else:
            G_i = 1.0
        
        # Эффективность детектирования
        eff = 0.3 if iso == 'Rn222' else efficiencies_dict.get(roi_num, 0.0)
        
        # Диагональный элемент Hᵢᵢ
        if t_sl and t_sl > 0 and eff > 0:
            H[i, i] = G_i / (t_sl * eff * br)
    
    return H


# ============================================================================
# ФУНКЦИИ РАСЧЕТА LC И LD (MDA)
# ============================================================================
def calculate_LC_Ld_all_isotopes(
    A: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    k: float = 1.645,
    max_iter: int = 50,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Расчет LC и Ld (MDA) для всех изотопов.
    """
    
    n_isotopes = 5
    LC = np.zeros(n_isotopes)
    Ld = np.zeros(n_isotopes)
    
    isotope_names = ['Rn222', 'Xe135', 'Xe133', 'Xe131m', 'Xe133m']
    
    print("\n" + "="*80)
    print("РАСЧЕТ LC И MDA ДЛЯ ВСЕХ ИЗОТОПОВ")
    print("="*80)
    
    for j in range(n_isotopes):
        print(f"\n{'='*60}")
        print(f"ИЗОТОП: {isotope_names[j]} (индекс {j})")
        print(f"{'='*60}")
        
        r_j = A[:, j].copy()
        y_without = y - r_j * x[j]
        
        # ============================================
        # Расчет σ₀ и LC
        # ============================================
        print(f"\n  Шаг 1: Расчет σ₀")
        
        W = np.diag(1.0 / np.maximum(np.abs(y_without), 1e-12))
        cov = np.linalg.pinv(A.T @ W @ A)
        cov_beta = cov[:5, :5]
        cov_a = T @ cov_beta @ T.T
        sigma_0 = np.sqrt(cov_a[j, j])
        
        LC[j] = k * sigma_0
        print(f"    σ₀ = {sigma_0:.3e} Бк/м³")
        print(f"    LC = {LC[j]:.3e} Бк/м³")
        
        # ============================================
        # Итерационный поиск Ld
        # Условие: t = k × σ(t) + k × σ₀
        # где t — удельная активность (Бк/м³)
        # ============================================
        print(f"\n  Шаг 2: Итерационный поиск Ld")
        print(f"    Условие: t = {k} × σ(t) + {k} × σ₀")
        
        # Начальное приближение для t (удельная активность, Бк/м³)
        t = LC[j]  # начинаем с LC
        print(f"    Начальное приближение t = {t:.3e} Бк/м³")
        
        Ld_val = None
        
        for iteration in range(max_iter):
            # Переводим t (Бк/м³) в β для добавления в y
            # a_c = T @ β → β = T^{-1} @ a_c
            # try:
            #     T_inv = np.linalg.inv(T)
            #     a_c_signal = np.zeros(5)
            #     a_c_signal[j] = t
            #     beta_signal = T_inv @ a_c_signal
            #     beta_signal_j = beta_signal[j]
            # except:
            beta_signal_j = t / T[j, j] if T[j, j] > 0 else t
            
            # Добавляем сигнал в единицах β
            y_signal = y_without + r_j * beta_signal_j
            
            # Весовая матрица с сигналом
            W_signal = np.diag(1.0 / np.maximum(np.abs(y_signal), 1e-12))
            
            # Ковариация с сигналом
            cov_signal = np.linalg.pinv(A.T @ W_signal @ A)
            cov_beta_signal = cov_signal[:5, :5]
            cov_a_signal = T @ cov_beta_signal @ T.T
            sigma_t = np.sqrt(cov_a_signal[j, j])
            
            # Целевое значение
            target = k * sigma_t + k * sigma_0
            
            # Разница
            diff = abs(t - target)
            rel_diff = diff / (abs(t) + 1e-12)
            
            print(f"    Итер {iteration+1:2d}: t={t:.3e}, σ(t)={sigma_t:.3e}, target={target:.3e}, diff={rel_diff:.2e}")
            
            if rel_diff < tol:
                print(f"    ✓ Сходимость!")
                Ld_val = t
                break
            
            # Обновляем t (простая итерация)
            t = target
        
        if Ld_val is None:
            Ld_val = LC[j] * 2  # fallback: Ld = 2 × LC
            print(f"    Сходимость не достигнута, Ld = 2 × LC = {Ld_val:.3e}")
        
        Ld[j] = Ld_val
        
        print(f"\n  ИТОГО ДЛЯ {isotope_names[j]}:")
        print(f"    σ₀ = {sigma_0:.3e} Бк/м³")
        print(f"    LC = {LC[j]:.3e} Бк/м³")
        print(f"    MDA (Ld) = {Ld[j]:.3e} Бк/м³")
        print(f"    Отношение Ld/LC = {Ld[j]/LC[j]:.2f}")
    
    return LC, Ld

def calculate_LC_Ld_iterative(
    A: np.ndarray, 
    W0: np.ndarray, 
    T: np.ndarray, 
    component_index: int, 
    k: float = 1.645, 
    max_iter: int = 20,
    tolerance: float = 1e-9
) -> Tuple[float, float]:
    """
    Итерационный расчет критического уровня (LC) и предела обнаружения (Ld/MDA).
    
    Математика:
    -----------
    LC = k * σ₀          (критический уровень)
    Ld = k * σ₀ + k * σ(Ld)  (предел обнаружения, MDA)
    
    где:
      σ₀ = √[T @ Σ_β @ T.T][component_index, component_index]
      Σ_β = (A.T @ W @ A)^{-1}
      σ(Ld) - стандартное отклонение при добавлении сигнала Ld
    
    Параметры:
    ----------
    A : (n, m) матрица дизайна
    W0 : (n, n) диагональная весовая матрица (обратные отсчеты)
    T : (p, 5) матрица преобразования β -> активности
    component_index : int, индекс в векторе a = T @ β
    k : float, коэффициент доверия (1.645 для 95%)
    max_iter : int, максимальное число итераций
    tolerance : float, точность сходимости
    
    Возвращает:
    -----------
    Ld : float, предел обнаружения (MDA)
    LC : float, критический уровень
    """
    print("\n" + "="*80)
    print("РАСЧЕТ LC И Ld (MDA) - ИТЕРАЦИОННЫЙ МЕТОД")
    print("="*80)
    
    # 1. Проверка входных данных
    print(f"\n1. ПРОВЕРКА ВХОДНЫХ ДАННЫХ:")
    print(f"   A.shape = {A.shape}")
    print(f"   W0.shape = {W0.shape}")
    print(f"   T.shape = {T.shape}")
    print(f"   component_index = {component_index}")
    print(f"   k = {k} (доверительный коэффициент)")
    print(f"   max_iter = {max_iter}")
    print(f"   tolerance = {tolerance}")
    
    if W0.ndim != 2 or W0.shape[0] != W0.shape[1]:
        raise ValueError("W0 должна быть квадратной матрицей")
    
    # Извлекаем диагональ для вычислений
    W0_diag = np.diag(W0)
    print(f"   W0_diag (первые 5 элементов): {W0_diag[:5]}")
    
    # 2. Расчет LC (критического уровня) по фоновым данным
    print(f"\n2. РАСЧЕТ КРИТИЧЕСКОГО УРОВНЯ (LC):")
    
    print(f"   Вычисляем Aᵀ·W₀·A...")
    ATW0A = A.T @ W0 @ A
    print(f"   (Aᵀ·W₀·A).shape = {ATW0A.shape}")
    
    try:
        print(f"   Инвертируем матрицу (Aᵀ·W₀·A)...")
        Cov_beta = np.linalg.inv(ATW0A)
        print(f"   Инверсия успешна")
    except np.linalg.LinAlgError:
        print(f"   Внимание: использование псевдообратной матрицы")
        Cov_beta = np.linalg.pinv(ATW0A)
    
    print(f"   Cov_beta.shape = {Cov_beta.shape}")
    print(f"   Берем подматрицу Cov_beta[0:5, 0:5]...")
    Cov_beta_subset = Cov_beta[:5, :5]
    
    print(f"   Вычисляем Σ_a = T @ Cov_beta[0:5,0:5] @ Tᵀ...")
    Cov_a = T @ Cov_beta_subset @ T.T
    print(f"   Cov_a.shape = {Cov_a.shape}")
    
    print(f"   Извлекаем дисперсию для компонента {component_index}...")
    variance_0 = Cov_a[component_index, component_index]
    sigma_0 = np.sqrt(variance_0)
    LC = k * sigma_0
    
    print(f"\n   РЕЗУЛЬТАТ РАСЧЕТА LC:")
    print(f"   Дисперсия σ₀² = {variance_0:.6e}")
    print(f"   Стандартное отклонение σ₀ = {sigma_0:.6e}")
    print(f"   Критический уровень LC = k × σ₀ = {k} × {sigma_0:.6e} = {LC:.6e}")
    
    # 3. Итерационный расчет Ld
    print(f"\n3. ИТЕРАЦИОННЫЙ РАСЧЕТ ПРЕДЕЛА ОБНАРУЖЕНИЯ (Ld):")
    print(f"   Начальное приближение: Ld₀ = LC = {LC:.6e}")
    
    Ld = LC  # Начальное приближение
    converged = False
    
    print(f"\n   НАЧАЛО ИТЕРАЦИЙ:")
    print(f"   {'Итерация':<8} {'Ld':<15} {'Ld_new':<15} {'σ(Ld)':<15} {'Отн. изменение':<15} {'Сходимость'}")
    print(f"   {'-'*8} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    for iteration in range(max_iter):
        # 3.1. Добавляем сигнал Ld к отсчетам
        signal_contribution = Ld * A[:, component_index]
        print(f"\n   Итерация {iteration+1}/{max_iter}:")
        print(f"     Сигнал Ld = {Ld:.6e}")
        print(f"     A[:,{component_index}] (первые 5): {A[:5, component_index]}")
        print(f"     Вклад сигнала (первые 5): {signal_contribution[:5]}")
        
        # 3.2. Обновляем отсчеты: counts = 1/W0_diag + signal
        background_counts = 1.0 / W0_diag
        print(f"     Фоновые отсчеты (первые 5): {background_counts[:5]}")
        
        counts_new = background_counts + signal_contribution
        counts_new = np.maximum(counts_new, 1e-10)  # Защита от нуля
        print(f"     Новые отсчеты (первые 5): {counts_new[:5]}")
        
        # 3.3. Новые веса: W_new = diag(1/counts_new)
        W_new_diag = 1.0 / counts_new
        W_new = np.diag(W_new_diag)
        print(f"     Новые веса W_new (первые 5): {W_new_diag[:5]}")
        
        # 3.4. Новая ковариация с добавленным сигналом
        print(f"     Вычисляем Aᵀ·W_new·A...")
        ATW_new_A = A.T @ W_new @ A
        
        try:
            Cov_beta_new = np.linalg.inv(ATW_new_A)
            print(f"     Инверсия успешна")
        except np.linalg.LinAlgError:
            print(f"     Внимание: использование псевдообратной матрицы")
            Cov_beta_new = np.linalg.pinv(ATW_new_A)
        
        Cov_beta_new_subset = Cov_beta_new[:5, :5]
        print(f"     Вычисляем новую Σ_a...")
        Cov_a_new = T @ Cov_beta_new_subset @ T.T
        
        variance_Ld = Cov_a_new[component_index, component_index]
        sigma_Ld = np.sqrt(variance_Ld)
        print(f"     Новая дисперсия σ(Ld)² = {variance_Ld:.6e}")
        print(f"     Новое стандартное отклонение σ(Ld) = {sigma_Ld:.6e}")
        
        # 3.5. Новое значение Ld
        Ld_new = LC + k * sigma_Ld
        print(f"     Новый Ld = LC + k × σ(Ld) = {LC:.6e} + {k} × {sigma_Ld:.6e} = {Ld_new:.6e}")
        
        # 3.6. Проверка сходимости
        rel_diff = abs(Ld_new - Ld) / (0.5 * (abs(Ld_new) + abs(Ld)) + 1e-12)
        
        # Вывод в компактной таблице
        status = "✓ СХОДИМОСТЬ" if rel_diff < tolerance else "продолжаем"
        print(f"   {iteration+1:<8} {Ld:<15.6e} {Ld_new:<15.6e} {sigma_Ld:<15.6e} {rel_diff:<15.2e} {status}")
        
        if rel_diff < tolerance:
            converged = True
            Ld = Ld_new
            print(f"\n     ✓ Достигнута сходимость на итерации {iteration+1}")
            print(f"     Относительное изменение: {rel_diff:.2e} < tolerance = {tolerance}")
            break
        
        Ld = Ld_new
    
    if not converged:
        print(f"\n   ⚠ ВНИМАНИЕ: Сходимость не достигнута за {max_iter} итераций!")
        print(f"   Последнее относительное изменение: {rel_diff:.2e}")
        print(f"   Требуемая точность: {tolerance}")
    
    # 4. Итоговые результаты
    print(f"\n4. ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   {'Параметр':<25} {'Значение':<20} {'Единицы':<15}")
    print(f"   {'-'*25} {'-'*20} {'-'*15}")
    print(f"   {'Критический уровень LC':<25} {LC:<20.6e} {'Бк/м³':<15}")
    print(f"   {'Предел обнаружения Ld':<25} {Ld:<20.6e} {'Бк/м³':<15}")
    print(f"   {'Отношение Ld/LC':<25} {Ld/LC:<20.3f} {'':<15}")
    print(f"   {'σ₀ (фон)':<25} {sigma_0:<20.6e} {'Бк/м³':<15}")
    print(f"   {'σ(Ld)':<25} {sigma_Ld:<20.6e} {'Бк/м³':<15}")
    
    
    print(f"\n" + "="*80)
    print("РАСЧЕТ ЗАВЕРШЕН")
    print("="*80)
    
    return Ld, LC



# ============================================================================
# ФУНКЦИИ РЕГРЕССИИ
# ============================================================================

def solve_nnls_regression_with_simple_errors(
    R_c: np.ndarray, R_g1: np.ndarray, R_g2: np.ndarray, F: np.ndarray, 
    c_vector: np.ndarray, sample_counts: Optional[Dict[int, float]] = None,
    background_counts: Optional[Dict[int, float]] = None,
    confidence_level: Optional[float] = None
) -> Dict[str, Any]:
    """
    Решает задачу NNLS и вычисляет погрешности через обычный МНК.
    """
    # 1. Создаем матрицу регрессии
    R_g1_F = R_g1 @ F
    A_upper = np.hstack([R_c, R_g1_F])
    A_lower = np.hstack([np.zeros((7, 5)), R_g2])
    A = np.vstack([A_upper, A_lower])
    y = c_vector.flatten()
    
    n_sample, n_background = 7, 7
    n_params_beta, n_params_gamma = 5, 5
    n = len(y)
    p = n_params_beta + n_params_gamma
    
    # 2. Вычисляем веса для МНК
    weights = np.ones(n)
    if sample_counts is not None and background_counts is not None:
        roi_list = [1, 2, 3, 5, 6, 7, 8]
        for i, roi in enumerate(roi_list):
            N_sample = sample_counts.get(roi, 0.0)
            N_background = background_counts.get(roi, 0.0)
            
            weight_sample = 1.0 / (N_sample + 1.0)
            weight_background = 1.0 / (N_background + 1.0)
            
            weights[i] = weight_sample
            weights[i + len(roi_list)] = weight_background
    
    # 3. Взвешенный NNLS
    W_sqrt = np.sqrt(weights)
    A_weighted = A * W_sqrt[:, np.newaxis]
    y_weighted = y * W_sqrt
    
    x_nnls, residual_nnls = nnls(A_weighted, y_weighted)
    beta_nnls = x_nnls[:n_params_beta]
    gamma_nnls = x_nnls[n_params_beta:]
    
    # 4. Обычный МНК (без ограничений)
    x_ls, residuals, rank, s = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
    beta_ls = x_ls[:n_params_beta]
    gamma_ls = x_ls[n_params_beta:]
    
    # 5. Ковариационная матрица для пуассоновских ошибок
    W = np.diag(weights)
    A_T_W_A = A.T @ W @ A
    cov_θ = np.linalg.inv(A_T_W_A)
    
    # Стандартные ошибки
    std_θ = np.sqrt(np.diag(cov_θ))
    std_beta = std_θ[:n_params_beta]
    std_gamma = std_θ[n_params_beta:]
    
    cov_beta = cov_θ[:n_params_beta, :n_params_beta]
    cov_gamma = cov_θ[n_params_beta:, n_params_gamma:]
    
    # 6. Остатки
    y_hat = A @ x_ls
    residuals_vec = y - y_hat
    RSS = residuals_vec.T @ W @ residuals_vec
    
    residuals_vec = residuals_vec * np.sqrt(weights)
    df = n - p
    chi_sq_reduced = RSS / df if df > 0 else RSS
    
    # 7. Расчет W_new для третьего изотопа
    j = 2  # Индекс третьего изотопа Xe-133
    a_j = A[:, j]
    β_j = beta_nnls[j]
    y_contribution = β_j * a_j
    
    try:
        W_inv = np.linalg.inv(W)
        diag_y_contrib = np.diag(y_contribution)
        diff_matrix = W_inv - diag_y_contrib
        W_new = np.linalg.inv(diff_matrix)
    except np.linalg.LinAlgError:
        print("WARNING: Не удалось вычислить W_new, использую W")
        W_new = W
    
    # 8. Новая ковариационная матрица
    try:
        A_T_W_new_A = A.T @ W_new @ A
        cov_new = np.linalg.inv(A_T_W_new_A)
    except np.linalg.LinAlgError:
        print("WARNING: Не удалось инвертировать A'·W_new·A, использую псевдообратную")
        cov_new = np.linalg.pinv(A_T_W_new_A)
    
    # 9. Имена изотопов (ДОБАВЛЕНО!)
    # Получаем имена из HALF_LIFE_DAYS или используем стандартные
    try:
        from constants import HALF_LIFE_DAYS
        isotope_names = list(HALF_LIFE_DAYS.keys())
    except ImportError:
        # Стандартные имена для 5 изотопов
        isotope_names = ['Rn222', 'Xe135', 'Xe133', 'Xe131m', 'Xe133m']
    
    # 10. Базовый результат
    result = {
        # NNLS оценки
        'beta_nnls': beta_nnls,
        'gamma_nnls': gamma_nnls,
        'x_nnls': x_nnls,
        'residual_nnls': residual_nnls,
        'residuals_vec' : residuals_vec,
        # Погрешности
        'std_beta': std_beta,
        'std_gamma': std_gamma,
        'cov_beta': cov_beta,
        'cov_gamma': cov_gamma,
        'cov_θ' : cov_θ,
        # Матрицы для расчетов
        'A': A,
        'W': W,
        'W_new': W_new,
        'cov_new': cov_new,
        
        
        # Для информации
        'gamma_ls': gamma_ls,
        'beta_ls' : beta_ls,
        'RSS': chi_sq_reduced,
        'weights': weights,
        'isotopes': isotope_names,  # ИЗМЕНЕНО: теперь имена, а не индексы
        
        # Структура задачи
        'problem_structure': {
            'n_sample': n_sample,
            'n_background': n_background,
            'n_params_beta': n_params_beta,
            'n_params_gamma': n_params_gamma,
            'matrix_A_shape': A.shape,
            'vector_b_length': len(y)
        }
    }
    
    return result


# ============================================================================
# ФУНКЦИИ РАСЧЕТА ПОГРЕШНОСТЕЙ С СИСТЕМАТИКОЙ
# ============================================================================

def propagate_errors_with_systematic(
    beta: np.ndarray, cov_beta_stat: np.ndarray, H: np.ndarray, 
    Gamma: np.ndarray, Gp: np.ndarray, Gs: np.ndarray, V_eff: float,
    delta_eff: Optional[np.ndarray] = None, delta_V: Optional[float] = None
) -> Dict[str, Any]:
    """
    Распространяет погрешности с учетом статистических и систематических составляющих.
    """
    if delta_eff is None:
        delta_eff = DELTA_EFF
    if delta_V is None:
        delta_V = DELTA_V_XE
    
    # 1. Расчет активностей
    a_vector = H @ beta
    a_c_vector = (1.0 / V_eff) * Gs @ Gp @ Gamma @ a_vector


    # 2. Ковариация активности в детекторе
    cov_a_stat = H @ cov_beta_stat @ H.T          # Статистическая
    cov_a_sys_eff = np.diag((a_vector * delta_eff)**2)  # Систематика эффективности
    cov_a_total = cov_a_stat + cov_a_sys_eff

    # 3. Преобразование к удельной активности
    T = (1.0 / V_eff) * Gs @ Gp @ Gamma
    cov_c_stat_eff = T @ cov_a_total @ T.T        # Статистика + эффективность

    # 4. Систематическая погрешность от объема
    cov_c_sys_vol = (delta_V**2) * np.outer(a_c_vector, a_c_vector)

    # 5. Итоговая ковариация удельной активности
    cov_c_total = cov_c_stat_eff + cov_c_sys_vol

    # 6. Стандартные отклонения
    std_a_total = np.sqrt(np.diag(cov_a_total))
    std_c_total = np.sqrt(np.diag(cov_c_total))
    std_c_stat_eff = np.sqrt(np.diag(cov_c_stat_eff))
    std_c_sys_vol = np.sqrt(np.diag(cov_c_sys_vol))
    
    # 7. Относительные погрешности (%)
    rel_std_a_total = np.zeros_like(std_a_total)
    for i in range(len(a_vector)):
        if abs(a_vector[i]) > 1e-12:
            rel_std_a_total[i] = std_a_total[i] / abs(a_vector[i]) * 100
        else:
            rel_std_a_total[i] = np.nan
    
    rel_std_c_total = np.zeros_like(std_c_total)
    for i in range(len(a_c_vector)):
        if abs(a_c_vector[i]) > 1e-12:
            rel_std_c_total[i] = std_c_total[i] / abs(a_c_vector[i]) * 100
        else:
            rel_std_c_total[i] = np.nan
    
    # 8. Корреляционные матрицы
    def safe_correlation(cov_matrix: np.ndarray) -> np.ndarray:
        n = cov_matrix.shape[0]
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0
                elif cov_matrix[i, i] > 0 and cov_matrix[j, j] > 0:
                    corr[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
                else:
                    corr[i, j] = 0.0
        return corr
    
    corr_a_total = safe_correlation(cov_a_total)
    corr_c_total = safe_correlation(cov_c_total)
    corr_c_sys_vol = safe_correlation(cov_c_sys_vol)
    
    return {
        'a_vector': a_vector,
        'a_c_vector': a_c_vector,
        
        'cov_a_total': cov_a_total,
        'cov_c_total': cov_c_total,
        
        'corr_a_total': corr_a_total,
        'corr_c_total': corr_c_total,
        'corr_c_sys_vol': corr_c_sys_vol,
        
        'std_a_total': std_a_total,
        'std_c_total': std_c_total,
        'std_c_components': {
            'stat_eff': std_c_stat_eff,
            'sys_vol': std_c_sys_vol,
            'total': std_c_total
        },
        
        'rel_std_a_total': rel_std_a_total,
        'rel_std_c_total': rel_std_c_total,
    }



def calculate_all_activities_with_systematic_errors(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Рассчитывает все активности с учетом погрешностей и MDA.
    """
    print(f"\n{'='*60}")
    print("Расчет активностей с MDA")
    print(f"{'='*60}")
    
    required_keys = ['regression', 'H_matrix', 'Gamma', 'Gp', 'Gs', 'sample_temporal']
    missing = set(required_keys) - set(results.keys())
    
    if missing:
        print(f"Недостаточно данных: отсутствуют {missing}")
        return None
    
    try:
        regression = results['regression']
        H = results['H_matrix']
        Gamma = results['Gamma']
        Gp = results['Gp']
        Gs = results['Gs']
        
        V_xe = results['sample_temporal'].get('xenon_volume')
        if not V_xe or V_xe <= 0:
            print("Объем ксенона не указан или некорректен")
            return None
        
        V_eff = V_xe / 0.087
        isotopes = regression['isotopes']
        
        # ========== ПОЛУЧАЕМ BETA ИЗ ВЫБРАННОГО МЕТОДА ==========
        if 'regression_selected' in results:
            beta = results['regression_selected']['beta']
            print(f"Использую beta из выбранного метода: {results['regression_selected'].get('method', 'unknown')}")
            
            if 'cov_beta' in results['regression_selected']:
                cov_beta_stat = results['regression_selected']['cov_beta']
            else:
                cov_beta_stat = regression.get('cov_beta', np.eye(len(beta)) * 0.01)
        else:
            if 'beta' in regression:
                beta = regression['beta']
                print(f"Использую beta из regression")
            elif 'beta_ls' in regression:
                beta = regression['beta_ls']
                print("Использую beta_ls (обратная совместимость)")
            else:
                raise KeyError("В regression нет beta или beta_ls")
            
            if 'cov_beta' in regression:
                cov_beta_stat = regression['cov_beta']
            elif 'std_beta' in regression:
                std_beta = regression['std_beta']
                cov_beta_stat = np.diag(std_beta ** 2)
                print("  Использую диагональную ковариацию из std_beta")
            else:
                cov_beta_stat = np.eye(len(beta)) * 0.01
                print("  ВНИМАНИЕ: Использую приближенную ковариацию")
        
        print(f"Изотопы: {isotopes}")
        print(f"beta: {beta}")
        print(f"V_eff: {V_eff}")
        
        # 1. Расчет погрешностей с систематикой
        error_results = propagate_errors_with_systematic(
            beta=beta,
            cov_beta_stat=cov_beta_stat,
            H=H,
            Gamma=Gamma,
            Gp=Gp,
            Gs=Gs,
            V_eff=V_eff,
            delta_eff=DELTA_EFF,
            delta_V=DELTA_V_XE
        )
        
# ========== НОВЫЙ РАСЧЕТ MDA с использованием calculate_LC_Ld_all_isotopes ==========
        MDA_dict = {}
        LC_dict = {}
        
        # Получаем необходимые данные для расчета MDA
        A = regression.get('A')  # матрица дизайна (14×10)
        if A is None:
            print("WARNING: Невозможно рассчитать MDA - отсутствует A")
            for iso in isotopes:
                MDA_dict[iso] = None
                LC_dict[iso] = None
        else:
            # Получаем решение x (10 элементов: [β₁...β₅, γ₁...γ₅])
            if 'beta_nnls' in regression and 'gamma_nnls' in regression:
                x_nnls = np.concatenate([regression['beta_nnls'], regression['gamma_nnls']])
                print("Использую x из NNLS (beta_nnls + gamma_nnls)")
            elif 'selected_beta' in regression and 'selected_gamma' in regression:
                x_nnls = np.concatenate([regression['selected_beta'], regression['selected_gamma']])
                print(f"Использую x из выбранного метода: {regression.get('selected_method', 'unknown')}")
            else:
                x_nnls = np.concatenate([beta, np.zeros(5)])
                print("ВНИМАНИЕ: Использую приближенный x (gamma = 0)")
            
            # ========== ИСПРАВЛЕНИЕ: ВСЕГДА используем combined_vector2['c'] ==========
            # Получаем вектор наблюдений y (14 элементов) из combined_vector2['c']
            if 'combined_vector2' in results and 'c' in results['combined_vector2']:
                y = results['combined_vector2']['c']
                print("Использую y из combined_vector2['c']")
            elif 'combined_vector' in results and 'combined_vector' in results['combined_vector']:
                y = results['combined_vector']['combined_vector']
                print("Использую y из combined_vector['combined_vector']")
            else:
                y = np.zeros(14)
                print("WARNING: Нет вектора наблюдений y, использую нули")
            
            if y is None or len(y) != 14:
                print("WARNING: Нет корректного вектора наблюдений y")
                for iso in isotopes:
                    MDA_dict[iso] = None
                    LC_dict[iso] = None
            else:
                # Матрица преобразования T
                T = (1 / V_eff) * Gs @ Gp @ Gamma @ H
                
                print(f"\nРасчет MDA для изотопов (новая итерационная процедура):")
                
                # Вызываем новую функцию
                LC_vec, Ld_vec = calculate_LC_Ld_all_isotopes(
                    A=A,
                    x=x_nnls,
                    y=y,
                    T=T,
                    k=1.645,
                    max_iter=50,
                    tol=1e-6
                )
                
                # Заполняем словари
                for i, iso in enumerate(isotopes):
                    LC_dict[iso] = LC_vec[i]
                    MDA_dict[iso] = Ld_vec[i]
                    
                    # Статус обнаружения
                    activity = error_results['a_c_vector'][i]
                    status = "обнаружен" if activity > Ld_vec[i] else "не обнаружен"
                    print(f"{iso}: Активность: {activity:.3e}, LC: {LC_vec[i]:.3e}, MDA: {Ld_vec[i]:.3e}, Статус: {status}")       
        # 3. Создание словарей результатов
        activities_dict = dict(zip(isotopes, error_results['a_vector']))
        specific_activities_dict = dict(zip(isotopes, error_results['a_c_vector']))
        std_activities = dict(zip(isotopes, error_results['std_a_total']))
        std_specific_activities = dict(zip(isotopes, error_results['std_c_total']))
        
        # 4. Формирование финального результата
        final_result = {
            'isotopes': isotopes,
            'activities_in_detector': activities_dict,
            'specific_activities': specific_activities_dict,
            'std_activities': std_activities,
            'std_specific_activities': std_specific_activities,
            'MDA': MDA_dict,
            'LC': LC_dict,
            
            'relative_errors_activities': {
                iso: f"{error_results['rel_std_a_total'][i]:.1f}%" 
                if not np.isnan(error_results['rel_std_a_total'][i]) else "не определена"
                for i, iso in enumerate(isotopes)
            },
            'relative_errors_specific': {
                iso: f"{error_results['rel_std_c_total'][i]:.1f}%" 
                if not np.isnan(error_results['rel_std_c_total'][i]) else "не определена"
                for i, iso in enumerate(isotopes)
            },
            
            'correlation_matrices': {
                'activities': error_results['corr_a_total'].tolist(),
                'specific': error_results['corr_c_total'].tolist(),
                'sys_vol': error_results['corr_c_sys_vol'].tolist()
            },
            
            'volumes': {
                'V_xe': V_xe,
                'V_eff': V_eff,
                'concentration_factor': 0.087,
            },
            'systematic_params': {
                'delta_V': DELTA_V_XE,
                'delta_eff': DELTA_EFF.tolist(),
                'delta_V_percent': DELTA_V_XE * 100,
                'delta_eff_percent': (DELTA_EFF * 100).tolist()
            },
            'calculation_details': {
                'beta_vector': beta.tolist(),
                'method': 'NNLS + полные погрешности + итерационный MDA',
                'k_factor': 1.645,
                'confidence_level': 0.95
            }
        }
        
        # 5. Детали для третьего изотопа
        third_iso = isotopes[2] if len(isotopes) > 2 else None
        if third_iso and third_iso in MDA_dict:
            activity_val = specific_activities_dict.get(third_iso, 0)
            mda_val = MDA_dict.get(third_iso)
            lc_val = LC_dict.get(third_iso)
            
            if mda_val is not None:
                detection_status = "обнаружен" if activity_val > mda_val else "не обнаружен"
            else:
                detection_status = "MDA не рассчитан"
            
            final_result['third_isotope_details'] = {
                'isotope': third_iso,
                'index': 2,
                'activity': activity_val if activity_val is not None else 0,
                'MDA': mda_val if mda_val is not None else 0,
                'LC': lc_val if lc_val is not None else 0,
                'detection_status': detection_status
            }
        
        return final_result
        
    except Exception as e:
        print(f"Ошибка в calculate_all_activities_with_systematic_errors: {e}")
        import traceback
        traceback.print_exc()
        return None




def solve_poisson_mle_point_estimate(
    A: np.ndarray,
    c: np.ndarray, 
    b: np.ndarray,
    method: str = 'L-BFGS-B',
    max_iter: int = 1000,
    tol: float = 1e-8,
    nonnegative: bool = True,
    use_ls_initial: bool = True  # ← новый параметр
) -> Dict[str, Any]:
    """
    ТОЧЕЧНАЯ ОЦЕНКА для максимума правдоподобия Пуассона.
    
    Параметры:
    ----------
    nonnegative : bool, default=True
        Если True: x ≥ 0 (ограничение неотрицательности)
        Если False: без ограничений (x может быть отрицательным)
    use_ls_initial : bool, default=True
        Использовать LS решение как начальное приближение (лучше для сходимости)
    """
    
    n_samples, n_features = A.shape
    
    if len(c) != n_samples:
        raise ValueError(f"c имеет длину {len(c)}, но ожидается {n_samples}")
    if len(b) != n_samples:
        raise ValueError(f"b имеет длину {len(b)}, но ожидается {n_samples}")
    
    def negative_log_likelihood(x: np.ndarray) -> float:
        λ = b + A @ x
        λ = np.maximum(λ, 1e-12)
        return -np.sum(c * np.log(λ) - λ)
    
    def gradient(x: np.ndarray) -> np.ndarray:
        λ = b + A @ x
        λ = np.maximum(λ, 1e-12)
        return A.T @ (1 - c / λ)
    
    # ========== УЛУЧШЕННОЕ НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ ==========
    if use_ls_initial:
        # Используем LS решение как начальное приближение
        try:
            x_ls, _, _, _ = np.linalg.lstsq(A, c - b, rcond=None)
            # Обрезаем слишком большие значения
            x_ls = np.clip(x_ls, -1e6, 1e6)
            x0 = x_ls
            print("  Использую LS решение как начальное приближение")
        except:
            # Fallback
            x0 = np.ones(n_features) * 0.1 if nonnegative else np.zeros(n_features)
            print("  Использую стандартное начальное приближение")
    else:
        x0 = np.ones(n_features) * 0.1 if nonnegative else np.zeros(n_features)
    
    # ========== ГРАНИЦЫ ==========
    if nonnegative:
        bounds = Bounds(lb=0, ub=np.inf)
    else:
        # Для MLE без ограничений используем широкие границы
        bounds = Bounds(lb=-1e6, ub=1e6)  # ← широкие, но не бесконечные
    
    # ========== ОПТИМИЗАЦИЯ ==========
    result = minimize(
        fun=negative_log_likelihood,
        x0=x0,
        method=method,
        jac=gradient,
        bounds=bounds,
        options={
            'maxiter': max_iter,
            'ftol': tol,
            'gtol': 1e-8,      # ← добавили tolerance для градиента
            'disp': False
        }
    )
    
    # ========== ПРОВЕРКА СХОДИМОСТИ ==========
    if not result.success:
        print(f"  WARNING: Оптимизация не сошлась! {result.message}")
        # Пробуем другой метод
        if method == 'L-BFGS-B':
            print("  Пробую метод 'BFGS' без границ...")
            result2 = minimize(
                fun=negative_log_likelihood,
                x0=x0,
                method='BFGS',
                jac=gradient,
                options={'maxiter': max_iter, 'gtol': 1e-8}
            )
            if result2.success:
                result = result2
                print("  ✓ Метод BFGS сошелся")
    
    # Не обрезаем результат если nonnegative=False
    if nonnegative:
        x_opt = np.maximum(result.x, 0)
    else:
        x_opt = result.x
    
    λ_opt = b + A @ x_opt
    λ_opt_safe = np.maximum(λ_opt, 1e-12)
    log_likelihood = np.sum(c * np.log(λ_opt_safe) - λ_opt_safe)
    
    # Ковариационная матрица
    try:
        W = np.diag(1 / λ_opt_safe)
        fisher_info = A.T @ W @ A
        cov_matrix = np.linalg.inv(fisher_info)
        std_errors = np.sqrt(np.diag(cov_matrix))
        
        if not np.all(np.linalg.eigvalsh(cov_matrix) > 0):
            cov_matrix = np.linalg.pinv(fisher_info)
            std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0))
            
    except np.linalg.LinAlgError:
        W = np.diag(1 / λ_opt_safe)
        fisher_info = A.T @ W @ A
        cov_matrix = np.linalg.pinv(fisher_info)
        std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0))
    
    return {
        'x_opt': x_opt,
        'lambda_opt': λ_opt,
        'cov_matrix': cov_matrix,
        'std_errors': std_errors,
        'log_likelihood': log_likelihood,
        'success': result.success,
        'message': result.message,
        'n_iter': result.nit,
        'fun': result.fun,
        'nonnegative': nonnegative,
        'n_samples': n_samples,
        'n_features': n_features
    }


def solve_poisson_mle_for_spectrum(
    R_c: np.ndarray,
    R_g1: np.ndarray, 
    R_g2: np.ndarray,
    F: np.ndarray,
    c_vector: np.ndarray,
    b_vector: np.ndarray,
    method: str = 'L-BFGS-B',
    nonnegative: bool = True,
    use_ls_initial: bool = True  # ← новый параметр
) -> Dict[str, Any]:
    """
    Точечная оценка для задачи спектрального анализа.
    
    Параметры:
    ----------
    nonnegative : bool, default=True
        Если True: x ≥ 0 (ограничение неотрицательности)
        Если False: без ограничений (x может быть отрицательным)
    use_ls_initial : bool, default=True
        Использовать LS решение как начальное приближение
    """
    # 1. Строим матрицу дизайна
    R_g1_F = R_g1 @ F
    A_upper = np.hstack([R_c, R_g1_F])
    A_lower = np.hstack([np.zeros((7, 5)), R_g2])
    A = np.vstack([A_upper, A_lower])
    
    c = c_vector.flatten()
    b = b_vector.flatten()
    
    # 2. Пуассоновский MLE
    result = solve_poisson_mle_point_estimate(
        A, c, b, method, 
        nonnegative=nonnegative,
        use_ls_initial=use_ls_initial
    )
    
    # 3. Разделяем на beta и gamma
    n_beta = 5
    result['beta'] = result['x_opt'][:n_beta]
    result['gamma'] = result['x_opt'][n_beta:]
    
    if result['cov_matrix'] is not None:
        cov_matrix = result['cov_matrix']
        result['cov_beta'] = cov_matrix[:n_beta, :n_beta]
        result['cov_gamma'] = cov_matrix[n_beta:, n_beta:]
        result['cov_beta_gamma'] = cov_matrix[:n_beta, n_beta:]
    
    if result['std_errors'] is not None:
        std_errors = result['std_errors']
        result['beta_std'] = std_errors[:n_beta]
        result['gamma_std'] = std_errors[n_beta:]
    
    return result