"""Качество подгонки: χ² и пуассоновский девианс на степень свободы."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .model import JointModel

if TYPE_CHECKING:
    from .solvers import FitResult


@dataclass(frozen=True)
class FitQuality:
    statistic_name: str
    statistic: float
    n_observations: int
    rank: int
    degrees_of_freedom: int
    reduced_statistic: float | None
    nominal_dof: bool
    warnings: tuple[str, ...] = ()

    @property
    def symbol(self) -> str:
        return "χ²" if self.statistic_name == "chi2" else "D"

    @property
    def label(self) -> str:
        return "Приведённый χ²/ν" if self.statistic_name == "chi2" else "Приведённый девианс D/ν"


def poisson_deviance(x, mu) -> float:
    """D=2Σ[x ln(x/mu)+mu−x], eq:poisson_deviance; x=0 даёт вклад 2mu."""
    observed, expected = np.asarray(x, dtype=float), np.asarray(mu, dtype=float)
    if observed.ndim != 1 or expected.shape != observed.shape:
        raise ValueError("x и mu должны быть векторами одинаковой длины")
    if (not np.all(np.isfinite(observed)) or not np.all(np.isfinite(expected))
            or np.any(observed < 0) or np.any(observed != np.floor(observed))):
        raise ValueError("Нужны конечные mu и неотрицательные целочисленные x")
    positive = observed > 0
    if np.any(expected < 0) or np.any(expected[positive] <= 0):
        return float("inf")

    x_positive, mu_positive = observed[positive], expected[positive]
    delta = (mu_positive - x_positive) / x_positive
    terms = np.empty_like(x_positive)
    tiny = np.abs(delta) < 1e-4
    close = (np.abs(delta) < 0.5) & ~tiny
    far = ~(tiny | close)
    # Ряд delta−log(1+delta) сохраняет малую разность при mu≈x.
    d = delta[tiny]
    terms[tiny] = x_positive[tiny] * d**2 * (0.5 + d * (-1/3 + d * (1/4 + d * (-1/5 + d/6))))
    terms[close] = x_positive[close] * (delta[close] - np.log1p(delta[close]))
    # Разность логарифмов не переполняется при большом отношении x/mu.
    terms[far] = (mu_positive[far] - x_positive[far]
                  + x_positive[far] * (np.log(x_positive[far]) - np.log(mu_positive[far])))
    return float(2 * (np.sum(expected[~positive]) + np.sum(terms)))


def compute_fit_quality(model: JointModel, solution: "FitResult") -> FitQuality:
    """ν=N−rank; для ограниченных методов это номинальная нормировка.

    МНК берёт готовый взвешенный критерий решателя, сохраняя внешнюю W/Ω.
    Девианс считается по исходным x и полным mu, включая фиксированный фон.
    Диагностика не меняет оценки, ковариации и пороги обнаружения.
    """
    quadratic = solution.method in ("wls", "nnls")
    if not quadratic and solution.method not in ("poisson", "poisson_nonnegative"):
        raise ValueError("Неизвестный метод для диагностики подгонки")
    statistic = float(solution.objective) if quadratic else poisson_deviance(model.x, solution.mu)
    n_observations, rank = model.x.size, int(solution.rank)
    dof = n_observations - rank
    constrained = solution.method in ("nnls", "poisson_nonnegative")
    warnings = []
    if constrained:
        warnings.append(
            "Для ограниченного метода ν=N−ранг — номинальное число степеней свободы; "
            "при активных ограничениях стандартное χ²-распределение неприменимо."
        )
    if not quadratic and np.any(solution.mu == 0):
        warnings.append("При граничном mu=0 приближение девианса χ²-распределением может быть неприменимо.")
    if dof <= 0:
        warnings.append("Приведённая статистика не определена: ν≤0 (нет остаточных степеней свободы).")
    if not solution.success:
        warnings.append("Приведённая статистика недоступна: сходимость расчёта не подтверждена.")
    if not np.isfinite(statistic):
        warnings.append("Приведённая статистика недоступна: исходная статистика не конечна.")
    reduced = float(statistic / dof) if dof > 0 and solution.success and np.isfinite(statistic) else None
    return FitQuality(
        statistic_name="chi2" if quadratic else "poisson_deviance",
        statistic=statistic, n_observations=int(n_observations), rank=rank,
        degrees_of_freedom=int(dof), reduced_statistic=reduced,
        nominal_dof=constrained, warnings=tuple(warnings),
    )
