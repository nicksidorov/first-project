"""Активности по подготовленным преобразованиям статьи XeMat.

Обозначения сохранены: a = H @ beta, a_tilde = Gamma @ a,
a_c = T_c @ beta, T_c = G_s @ G_p @ Gamma @ H / V0.
H переводит отсчёты beta в Бк на начало SAMPLE; T_c — в Бк/м³.
Этот модуль не строит физические матрицы по временам и калибровкам.
Радоновая первая компонента a_c — вспомогательный параметр; она
не является атмосферной объёмной активностью радона.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve


Array = NDArray[np.float64]


def _vector(value: ArrayLike, name: str, n: int | None = None) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: требуется конечный одномерный вектор")
    if n is not None and result.shape != (n,):
        raise ValueError(f"{name}: ожидается длина {n}")
    return result.copy()


def _matrix(value: ArrayLike, name: str, n: int) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (n, n) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: требуется конечная матрица {n} × {n}")
    return result.copy()


def _covariance(value: ArrayLike, name: str, n: int) -> Array:
    result = _matrix(value, name, n)
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name}: ковариационная матрица должна быть симметричной")
    result = (result + result.T) / 2
    if np.any(np.diag(result) < 0):
        raise ValueError(f"{name}: отрицательная дисперсия")
    # Нормирование перед проверкой защищает разные масштабы дисперсий.
    scales = np.sqrt(np.diag(result))
    zero = scales == 0
    if np.any(result[zero, :] != 0):
        raise ValueError(f"{name}: нулевая дисперсия имеет ненулевую ковариацию")
    positive = ~zero
    if np.any(positive):
        block = result[np.ix_(positive, positive)]
        normalized = block / np.outer(scales[positive], scales[positive])
        if np.min(np.linalg.eigvalsh(normalized)) < -1e-10:
            raise ValueError(f"{name}: матрица не положительно полуопределённая")
    return result


def covariance_to_correlation(covariance: ArrayLike) -> Array:
    """Корреляция Пирсона; при нулевой дисперсии значение неопределено (NaN)."""
    value = np.asarray(covariance, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("Ковариационная матрица должна быть квадратной")
    value = _covariance(value, "covariance", value.shape[0])
    denominator = np.outer(np.sqrt(np.diag(value)), np.sqrt(np.diag(value)))
    result = np.full_like(value, np.nan)
    np.divide(value, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def compose_activity_transform(
    H: ArrayLike, Gamma: ArrayLike, G_p: ArrayLike, G_s: ArrayLike, V0: float
) -> Array:
    """Собрать только алгебраическое T_c из уже подготовленных матриц статьи."""
    H_value = np.asarray(H, dtype=float)
    if H_value.ndim != 2 or H_value.shape[0] != H_value.shape[1]:
        raise ValueError("H должна быть квадратной")
    n = H_value.shape[0]
    H_value = _matrix(H_value, "H", n)
    Gamma_value = _matrix(Gamma, "Gamma", n)
    G_p_value = _matrix(G_p, "G_p", n)
    G_s_value = _matrix(G_s, "G_s", n)
    if not np.isfinite(V0) or V0 <= 0:
        raise ValueError("V0 должен быть положительным конечным объёмом воздуха")
    # Порядок множителей существенен: Gamma содержит цепочечную поправку.
    return (G_s_value @ G_p_value @ Gamma_value @ H_value) / V0


@dataclass(frozen=True)
class ActivityResult:
    """Оценки и отдельно статистическая/полная неопределённость.

    None означает отсутствие подготовленного преобразования или подходящей
    статистической ковариации. В частности, K_beta=None для граничного NNLS
    не превращается в выдуманную ковариацию ограниченной оценки.
    """

    beta: Array
    a: Array | None
    a_tilde: Array | None
    a_c: Array | None
    T_c: Array | None
    K_A_stat: Array | None
    K_A_sys: Array | None
    K_A: Array | None
    K_A_c_stat: Array | None
    K_A_c_sys: Array | None
    K_A_c: Array | None
    std_A: Array | None
    std_A_c: Array | None
    correlation_A: Array | None
    correlation_A_c: Array | None


def compute_activities(
    beta: ArrayLike,
    K_beta: ArrayLike | None = None,
    *,
    H: ArrayLike | None = None,
    T_c: ArrayLike | None = None,
    Gamma: ArrayLike | None = None,
    G_p: ArrayLike | None = None,
    G_s: ArrayLike | None = None,
    V0: float | None = None,
    delta_I: ArrayLike | None = None,
    delta_V0: float = 0.0,
) -> ActivityResult:
    """Пересчёт beta и K_beta с явным распространением неопределённостей.

    delta_I — независимые относительные систематические неопределённости
    чувствительностей I_i^(c). delta_V0 — относительная неопределённость
    общего объёма, дающая коррелированную добавку delta_V0² a_c a_cᵀ.
    При наличии только T_c вычисляются только объёмные активности.
    Для полного пересчёта можно передать H и T_c либо все H,Gamma,G_p,G_s,V0.
    Подготовленные матрицы считаются фиксированными; неопределённость всего
    спектрального отклика R в этих формулах не включена.
    """
    beta_value = _vector(beta, "beta")
    n = beta_value.size
    if n == 0:
        raise ValueError("beta не должен быть пустым")
    K_value = None if K_beta is None else _covariance(K_beta, "K_beta", n)
    H_value = None if H is None else _matrix(H, "H", n)
    T_value = None if T_c is None else _matrix(T_c, "T_c", n)
    Gamma_value = None if Gamma is None else _matrix(Gamma, "Gamma", n)

    if any(item is not None for item in (G_p, G_s, V0)):
        if any(item is None for item in (H, Gamma, G_p, G_s, V0)):
            raise ValueError("Для сборки T_c требуются H, Gamma, G_p, G_s и V0")
        composed = compose_activity_transform(H, Gamma, G_p, G_s, V0)
        if T_value is not None and not np.allclose(T_value, composed, rtol=1e-9, atol=0):
            raise ValueError("Переданное T_c не соответствует H, Gamma, G_p, G_s, V0")
        T_value = composed

    if H_value is None and T_value is None:
        raise ValueError("Требуется подготовленная матрица H или T_c")
    if Gamma_value is not None and H_value is None:
        raise ValueError("Для a_tilde = Gamma @ H @ beta требуется H")
    if not np.isfinite(delta_V0) or delta_V0 < 0:
        raise ValueError("delta_V0 должна быть конечной и неотрицательной")
    if delta_V0 > 0 and T_value is None:
        raise ValueError("Для неопределённости объёма требуется T_c")
    relative = np.zeros(n) if delta_I is None else _vector(delta_I, "delta_I", n)
    if np.any(relative < 0):
        raise ValueError("delta_I должны быть неотрицательными")
    if np.any(relative > 0) and H_value is None:
        raise ValueError("Для неопределённости чувствительностей требуется H")

    # Знаки сохраняются: неограниченные оценки могут быть отрицательными.
    a = None if H_value is None else H_value @ beta_value
    a_tilde = None if Gamma_value is None else Gamma_value @ a
    a_c = None if T_value is None else T_value @ beta_value

    # K_A^stat = H K_beta Hᵀ; K_Ac^stat = T_c K_beta T_cᵀ.
    K_A_stat = None if H_value is None or K_value is None else H_value @ K_value @ H_value.T
    K_c_stat = None if T_value is None or K_value is None else T_value @ K_value @ T_value.T
    K_A_sys = None if a is None else np.diag((a * relative) ** 2)
    K_c_sys = None if a_c is None else delta_V0**2 * np.outer(a_c, a_c)
    if T_value is not None and np.any(relative > 0):
        # P = T_c H^{-1} = G_s G_p Gamma / V0. solve избегает явного inv(H).
        P = solve(H_value.T, T_value.T).T
        K_c_sys = K_c_sys + P @ K_A_sys @ P.T

    K_A = None if K_A_stat is None else K_A_stat + K_A_sys
    K_c = None if K_c_stat is None else K_c_stat + K_c_sys
    std_A = None if K_A is None else np.sqrt(np.maximum(np.diag(K_A), 0))
    std_c = None if K_c is None else np.sqrt(np.maximum(np.diag(K_c), 0))
    return ActivityResult(
        beta=beta_value,
        a=a,
        a_tilde=a_tilde,
        a_c=a_c,
        T_c=T_value,
        K_A_stat=K_A_stat,
        K_A_sys=K_A_sys,
        K_A=K_A,
        K_A_c_stat=K_c_stat,
        K_A_c_sys=K_c_sys,
        K_A_c=K_c,
        std_A=std_A,
        std_A_c=std_c,
        correlation_A=None if K_A is None else covariance_to_correlation(K_A),
        correlation_A_c=None if K_c is None else covariance_to_correlation(K_c),
    )
