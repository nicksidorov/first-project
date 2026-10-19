"""Критические уровни Керри и MDC в координатах V86 [a_V; gamma; zeta].

Статья: mu_i^(0) = S_V theta_(V,+) - a_(V,i,+) s_(V,i),
K_i(L) = [S_Vᵀ diag(1 / mu_i(L)) S_V]^{-1},
L_C = z_(1-alpha) sqrt(K_i(0)[i,i]),
L_D = L_C + z_(1-beta_II) sqrt(K_i(L_D)[i,i]), MDC = L_D.

Имена аргументов S_c и b сохранены для совместимости: S_c обозначает S_V,
а b=0 для V86. В прежней модели b остаётся известным аддитивным фоном.
Вспомогательная неотрицательная подгонка задаёт только нулевой спектр;
решение об обнаружении использует основную неограниченную оценку.
Здесь нет повторной подгонки и нет вычитания из наблюдаемых отсчётов.
"""

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import svd
from scipy.stats import norm


Array = NDArray[np.float64]


def _finite_vector(value: ArrayLike, name: str, n: int) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (n,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: требуется конечный вектор длины {n}")
    return result.copy()


def model_covariance(S_c: ArrayLike, mu: ArrayLike, *, rcond: float | None = None) -> Array:
    """Статистическая ковариация неограниченной полной модели при данном mu.

    Это обратная информация Фишера, не ковариация NNLS/ML+.
    Все столбцы (включая память, радон, фон zeta и нулевые оценки) сохраняются.
    Статистика DETBK входит через его строки и столбцы zeta полной S_V;
    отдельно добавлять дисперсию DETBK к полученной ковариации не нужно.
    Масштабирование столбцов и SVD избегают квадратного ухудшения
    обусловленности при формировании нормальных уравнений S_cᵀ W S_c.
    Неидентифицируемые модели отклоняются вместо конечного MDC из pinv.
    """
    S = np.asarray(S_c, dtype=float)
    if S.ndim != 2 or not S.size or not np.all(np.isfinite(S)):
        raise ValueError("S_c: требуется непустая конечная матрица")
    expected = _finite_vector(mu, "mu", S.shape[0])
    if np.any(expected <= 0):
        raise ValueError("Для ковариации MDC требуется mu > 0 во всех ROI; нули не заменяются epsilon")
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0 or rcond >= 1):
        raise ValueError("rcond должен принадлежать [0,1)")

    # A = sqrt(W) S_c, W = diag(1/mu). Не собираем огромную диагональную W.
    A = S / np.sqrt(expected)[:, None]
    if not np.all(np.isfinite(A)):
        raise ValueError("Переполнение при нормировании S_c по ожидаемым отсчётам")
    # Два шага нормирования предохраняют вычисление норм от переполнения.
    peaks = np.max(np.abs(A), axis=0)
    if np.any(peaks == 0):
        raise ValueError("S_c содержит нулевой столбец: параметр неидентифицируем")
    scaled = A / peaks
    lengths = np.linalg.norm(scaled, axis=0)
    normalized = scaled / lengths
    _, singular_values, Vh = svd(normalized, full_matrices=False, check_finite=False)
    tolerance = (
        np.finfo(float).eps * max(normalized.shape) if rcond is None else rcond
    ) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < S.shape[1]:
        raise ValueError(f"Полная S_c неидентифицируема: ранг {rank} < {S.shape[1]}; MDC не определён")

    # (Aᵀ A)^{-1} = D^{-1} V diag(1/s²) Vᵀ D^{-1}, D_jj = peaks_j lengths_j.
    covariance_factor = (Vh.T / singular_values) / lengths[:, None] / peaks[:, None]
    covariance = covariance_factor @ covariance_factor.T
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Ковариация не представима в float64 при данных масштабах")
    return (covariance + covariance.T) / 2


@dataclass(frozen=True)
class DetectionResult:
    """Пороги в Бк/м³; индексы соответствуют исходным столбцам S_c.

    При несходимости MDC=NaN; last_iterate содержит диагностическое
    приближение, не заменяющее рассчитанный предел обнаружения.
    mode='fixed' обозначает явно выбранное приближение с K_i(0),
    mode='iterative' — основной алгоритм статьи с пересчётом K_i(L).
    """

    component_indices: NDArray[np.int64]
    L_C: Array
    MDC: Array
    sigma0: Array
    sigma_at_MDC: Array
    detected: NDArray[np.bool_]
    free_activity: Array
    positive_activity: Array
    null_spectra: Array
    null_covariances: Array
    iterations: NDArray[np.int64]
    converged: NDArray[np.bool_]
    last_iterate: Array
    mode: str
    alpha: float
    beta_ii: float
    warnings: tuple[str, ...]

    @property
    def L_D(self) -> Array:
        """В данной параметризации MDC и L_D совпадают без пересчёта объёма."""
        return self.MDC


def compute_detection_limits(
    S_c: ArrayLike,
    b: ArrayLike,
    physical_free: ArrayLike,
    physical_positive: ArrayLike,
    *,
    component_indices: Iterable[int] | None = None,
    alpha: float = 0.05,
    beta_ii: float = 0.05,
    mode: str = "iterative",
    rtol: float = 1e-8,
    max_iter: int = 200,
    rcond: float | None = None,
) -> DetectionResult:
    """Вычислить LC/MDC по двум уже выполненным подгонкам полной модели.

    Для V86 physical_free и physical_positive имеют координаты
    [a_V; gamma; zeta]; для прежней модели — [a_c; gamma].
    Первый вектор сохраняет знаки; второй — NNLS либо ML+ именно в этих
    физических координатах. Удаляется только вклад проверяемого a_V;
    память gamma, фон zeta и все мешающие компоненты остаются. Строки
    GASBK и DETBK нулевого спектра совпадают с неотрицательной подгонкой.

    По умолчанию для модели V86 размера 21×17 и прежних 10 столбцов
    проверяются четыре изотопа Xe
    (Python-индексы 1,2,3,4), а радон остаётся nuisance-параметром.
    Для иных размерностей component_indices задаются явно.

    Стандартная версия учитывает статистику всех наблюдений, в том числе
    DETBK в полной модели V86, при фиксированной калибровке S_V.
    mode='fixed': MDC=(z_(1-alpha)+z_(1-beta_II))*sigma0 — приближение
    постоянной дисперсии, отдельно от основного итерационного метода.
    Ни отрицательные, ни нулевые ожидаемые отсчёты здесь не исправляются.
    """
    S = np.asarray(S_c, dtype=float)
    if S.ndim != 2 or not S.size or not np.all(np.isfinite(S)):
        raise ValueError("S_c: требуется непустая конечная матрица")
    n_rows, n_parameters = S.shape
    background = _finite_vector(b, "b", n_rows)
    if np.any(background < 0):
        raise ValueError("Аддитивный фон b не может быть отрицательным")
    free = _finite_vector(physical_free, "physical_free", n_parameters)
    positive = _finite_vector(physical_positive, "physical_positive", n_parameters)
    if np.any(positive < 0):
        raise ValueError("Вспомогательная оценка должна иметь все физические параметры >= 0")
    if component_indices is None:
        if n_parameters != 10 and S.shape != (21, 17):
            raise ValueError("Для нестандартного числа параметров задайте component_indices явно")
        component_indices = range(1, 5)
    supplied = list(component_indices)
    if not supplied or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in supplied):
        raise ValueError("component_indices должен содержать целые индексы проверяемых объёмных активностей")
    indices = np.asarray(supplied, dtype=np.int64)
    if np.unique(indices).size != indices.size or np.any(indices < 0) or np.any(indices >= n_parameters):
        raise ValueError("component_indices: недопустимые или повторяющиеся индексы")
    if mode not in ("iterative", "fixed"):
        raise ValueError("mode должен быть 'iterative' либо 'fixed'")
    if not np.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha должна принадлежать (0,0.5)")
    if not np.isfinite(beta_ii) or not 0 < beta_ii < 0.5:
        raise ValueError("beta_ii должна принадлежать (0,0.5)")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol должна быть положительной")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter должен быть положительным целым")

    k_alpha, k_beta = norm.ppf(1 - alpha), norm.ppf(1 - beta_ii)
    n_components = len(indices)
    L_C = np.empty(n_components)
    MDC = np.full(n_components, np.nan)
    sigma0 = np.empty(n_components)
    sigma_end = np.full(n_components, np.nan)
    null_spectra = np.empty((n_components, n_rows))
    null_covariances = np.empty((n_components, n_parameters, n_parameters))
    iterations = np.zeros(n_components, dtype=np.int64)
    converged = np.zeros(n_components, dtype=bool)
    last_iterate = np.empty(n_components)
    messages = []

    for output_index, i in enumerate(indices):
        # mu_i^(0) = mu_+ - z_+[i] s_i. Эта эквивалентная форма избегает
        # потери точности при вычитании большого сигнала из полного mu_+.
        nuisance = positive.copy()
        nuisance[i] = 0
        mu0 = background + S @ nuisance
        if np.any(mu0 <= 0):
            raise ValueError(
                f"Компонента {i}: нулевой модельный спектр требует mu_i^(0) > 0; "
                "нельзя подставлять abs(mu), epsilon или изменять наблюдения"
            )
        K0 = model_covariance(S, mu0, rcond=rcond)
        null_spectra[output_index] = mu0
        null_covariances[output_index] = K0
        sigma0[output_index] = np.sqrt(K0[i, i])
        L_C[output_index] = k_alpha * sigma0[output_index]

        if mode == "fixed":
            MDC[output_index] = (k_alpha + k_beta) * sigma0[output_index]
            last_iterate[output_index] = MDC[output_index]
            sigma_end[output_index] = sigma0[output_index]
            converged[output_index] = True
            continue

        L = L_C[output_index]  # Начальное L_D^(0) по статье.
        for iteration in range(1, max_iter + 1):
            # Нулевой спектр, полный S_c и мешающие оценки зафиксированы.
            mu_L = mu0 + L * S[:, i]
            K_L = model_covariance(S, mu_L, rcond=rcond)
            updated = L_C[output_index] + k_beta * np.sqrt(K_L[i, i])
            iterations[output_index] = iteration
            relative_change = abs(updated - L) / updated
            L = updated
            if relative_change < rtol:
                final_covariance = model_covariance(S, mu0 + L * S[:, i], rcond=rcond)
                sigma_end[output_index] = np.sqrt(final_covariance[i, i])
                MDC[output_index] = L
                converged[output_index] = True
                break
        last_iterate[output_index] = L
        if not converged[output_index]:
            messages.append(f"Компонента {i}: MDC не сошёлся за {max_iter} итераций; результат NaN")

    if mode == "fixed":
        messages.append("MDC рассчитан в приближении постоянной нулевой дисперсии, без её роста с сигналом")
    messages.append("Пороги используют нормальное асимптотическое приближение; малые отсчёты требуют проверки Монте-Карло")
    return DetectionResult(
        component_indices=indices,
        L_C=L_C,
        MDC=MDC,
        sigma0=sigma0,
        sigma_at_MDC=sigma_end,
        detected=free[indices] > L_C,  # Не NNLS/ML+ и не сравнение с MDC.
        free_activity=free[indices].copy(),
        positive_activity=positive[indices].copy(),
        null_spectra=null_spectra,
        null_covariances=null_covariances,
        iterations=iterations,
        converged=converged,
        last_iterate=last_iterate,
        mode=mode,
        alpha=alpha,
        beta_ii=beta_ii,
        warnings=tuple(messages),
    )
