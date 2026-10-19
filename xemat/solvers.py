"""Четыре оценивателя статьи XeMat для подготовленной модели ``mu=M@theta+b``.

МНК/NNLS используют одну квадратичную форму. По умолчанию
W=diag(1/where(x==0,1,x)), eq:wls_weights: заменяется дисперсия нулевого
отсчёта, а не сам отсчёт. Пуассоновские методы используют исходный x.
В V86 b=0, x=[SAMPLE;GASBK;DETBK], z=[a_V;gamma;zeta].
Неотрицательность относится ко всему физическому z, а не к beta.
Прежний формат z=[a_c;gamma] с известным b также поддерживается.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import LinearConstraint, linprog, minimize, nnls

from .model import JointModel


@dataclass
class FitResult:
    theta: np.ndarray
    physical_parameters: np.ndarray | None
    mu: np.ndarray
    covariance: np.ndarray | None
    physical_covariance: np.ndarray | None
    method: str
    objective: float
    success: bool
    message: str
    rank: int
    condition_number: float
    warnings: list[str] = field(default_factory=list)
    iterations: int = 0
    coordinates: str = "beta_gamma"


def _whiten(x, weights, covariance):
    """C такой, что ||C r||²=r.T W r; обратные SPD матрицы не строятся."""
    if weights is not None and covariance is not None:
        raise ValueError("Передайте weights или observation_covariance, а не оба")
    n = len(x)
    if weights is None and covariance is None:
        return 1 / np.sqrt(np.where(x == 0, 1.0, x))
    supplied = weights if covariance is None else covariance
    a = np.asarray(supplied, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("Веса/ковариация должны быть конечными")
    if a.ndim == 1 and covariance is None:
        if a.shape != (n,) or np.any(a <= 0):
            raise ValueError("Вектор весов должен иметь длину x и положительные элементы")
        return np.sqrt(a)
    if a.shape != (n, n) or not np.allclose(a, a.T, rtol=1e-12, atol=0):
        raise ValueError("Матрица весов/ковариации должна быть симметричной размера len(x)")
    try:
        L = cholesky(a, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Матрица весов/ковариации должна быть положительно определённой") from exc
    if covariance is None:
        return L.T  # W=L@L.T, поэтому C=L.T.
    return solve_triangular(L, np.eye(n), lower=True)  # Ω=L@L.T; W=Ω^{-1}.


def _apply_whitening(C, a):
    # Диагональные веса храним вектором: для спектра не нужна плотная n*n W.
    return C * a if C.ndim == 1 and a.ndim == 1 else (
        C[:, None] * a if C.ndim == 1 else C @ a)


def _scaled_svd(A):
    """Нормировка столбцов сохраняет координаты; SVD выявляет неоднозначность."""
    maxabs = np.max(np.abs(A), axis=0) if A.shape[0] else np.zeros(A.shape[1])
    maxabs = np.where(maxabs == 0, 1.0, maxabs)
    norms = np.linalg.norm(A / maxabs, axis=0)
    # maxabs предотвращает переполнение при возведении больших элементов в
    # квадрат. Если сама норма не представима, достаточно масштаба maxabs.
    with np.errstate(over="ignore"):
        scales = maxabs * norms
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, maxabs)
    B = A / scales
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    cutoff = np.finfo(float).eps * max(B.shape) * (s[0] if len(s) else 0)
    rank = int(np.sum(s > cutoff))
    condition = float(s[0] / s[-1]) if rank == A.shape[1] else float("inf")
    return scales, U, s, Vt, rank, condition


def _covariance(A):
    """(A.T A)^-1 через SVD, без ложной конечной ошибки при неполном ранге."""
    scales, _, s, Vt, rank, condition = _scaled_svd(A)
    if rank != A.shape[1]:
        return None, rank, condition
    # H H.T=diag(1/scales) V diag(1/s²) V.T diag(1/scales).
    H = (Vt.T / s) / scales[:, None]
    return H @ H.T, rank, condition


def _result(model, method, theta, covariance, objective, success, message,
            rank, condition, warnings, iterations=0):
    Q = model.coordinate_transform
    physical = model.to_physical(theta)
    physical_covariance = None if covariance is None or Q is None else Q @ covariance @ Q.T
    return FitResult(
        theta=theta, physical_parameters=physical, mu=model.M @ theta + model.b,
        covariance=covariance, physical_covariance=physical_covariance,
        method=method, objective=float(objective), success=bool(success),
        message=str(message), rank=rank, condition_number=condition,
        warnings=warnings, iterations=iterations, coordinates=model.coordinates,
    )


def _least_squares(model, method, weights, covariance, max_iter):
    C = _whiten(model.x, weights, covariance)
    # В V86 model.b=0: используем все исходные отсчёты, включая DETBK.
    y = _apply_whitening(C, model.x - model.b)
    A = _apply_whitening(C, model.physical_matrix if method == "nnls" else model.M)
    scales, U, s, Vt, rank, condition = _scaled_svd(A)
    warnings = []
    if rank != A.shape[1]:
        warnings.append("Неполный ранг: параметры не идентифицируются однозначно; ковариация не вычисляется")
    elif condition > 1e8:
        warnings.append("Сильное перекрытие компонент: большая обусловленность нормированных столбцов")
    if method == "nnls":
        # u=scales*z, scales>0: u>=0 эквивалентно a_V,gamma,zeta>=0.
        u, _ = nnls(A / scales, y, maxiter=max_iter)
        z = u / scales
        theta = model.from_physical(z)
        K = None
        warnings.append("Ограниченная оценка: обычная обратная информационная матрица не является её ковариацией")
    else:
        # theta=diag(1/scales) V diag(1/s) U.T x_b; это eq:wls_solution
        # без образования M.T W M, которое возводит обусловленность в квадрат.
        u = Vt[:rank].T @ ((U[:, :rank].T @ y) / s[:rank])
        theta = u / scales
        K, _, _ = _covariance(A)
    residual = _apply_whitening(C, model.x - (model.M @ theta + model.b))
    result = _result(model, method, theta, K, residual @ residual, True,
                     "Решена взвешенная линейная задача", rank, condition, warnings)
    if method == "nnls":
        result.physical_parameters = z
    return result


def _poisson_basis(A, count_scale, nonnegative):
    """Переход v -> параметры; нормированные независимые направления mu.

    Для свободной задачи A@T≈count_scale*U. Для ограниченной задачи оставляем
    диагональный переход: он сохраняет прямоугольную область z>=0, тогда как
    SVD при сильном перекрытии превратил бы её в численно узкий конус.
    Направления нулевого пространства ограниченной задачи также сохраняются.
    """
    scales, U, s, Vt, rank, condition = _scaled_svd(A)
    if nonnegative:
        T = np.diag(count_scale / scales)
        B = A @ T
    else:
        T = (Vt[:rank].T / s[:rank]) / scales[:, None] * count_scale
        # Вычисляем A@T явно, чтобы оптимизируемое mu соответствовало исходной
        # модели также с учётом округления сильно перекрывающихся компонент.
        B = A @ T
    return T, B, rank, condition


def _poisson_start(B, b, x, P):
    """Начальная точка: линейные ограничения сохраняются во всех решателях.

    LP ищет положительный запас для строк x>0. t<=1 ограничивает только
    вспомогательный запас, а не физические параметры. Для x=0 допустимо mu=0.
    """
    v = np.linalg.lstsq(B, x - b, rcond=None)[0]
    mu = B @ v + b
    if np.all(mu >= 0) and np.all(mu[x > 0] > 0) and (P is None or np.all(P @ v >= 0)):
        return v
    row_scale = np.maximum(np.maximum(np.linalg.norm(B, axis=1), np.abs(b)), 1.0)
    C = B / row_scale[:, None]
    lower = -b / row_scale
    positive = (x > 0).astype(float)
    # C v >= lower + positive*t.
    G = np.column_stack((-C, positive))
    h = -lower
    if P is not None:
        G = np.vstack((G, np.column_stack((-P, np.zeros(P.shape[0])))))
        h = np.r_[h, np.zeros(P.shape[0])]
    objective = np.r_[np.zeros(B.shape[1]), -1.0]
    lp = linprog(objective, A_ub=G, b_ub=h,
                 bounds=[(None, None)] * B.shape[1] + [(0, 1)], method="highs")
    if not lp.success or (np.any(x > 0) and lp.x[-1] <= 0):
        raise ValueError("Не существует допустимых ожидаемых отсчётов для конечного пуассоновского правдоподобия")
    v = lp.x[:-1]
    mu = B @ v + b
    if np.any(mu < 0) or np.any(mu[x > 0] <= 0) or (P is not None and np.any(P @ v < -1e-12)):
        raise ValueError("Не удалось построить численно допустимую начальную точку пуассоновского метода")
    return v


def _restore_mean_roundoff(model, theta, physical=None):
    """Устранить лишь отрицательный остаток арифметики в mu=M@theta+b.

    Отсчёты и mu не обрезаются. Используется малая выпуклая комбинация
    параметров с допустимой точкой: обычно theta=0, иначе точка линейной
    задачи. Значимые нарушения области не исправляются этим способом.
    """
    mu = model.M @ theta + model.b
    bad = mu < 0
    if not np.any(bad):
        return theta, physical, False
    roundoff = 64 * np.finfo(float).eps * (np.abs(model.M) @ np.abs(theta) + model.b + 1)
    if np.any(mu < -roundoff):
        return theta, physical, False
    target = np.zeros_like(theta)
    target_physical = np.zeros_like(physical) if physical is not None else None
    target_mu = model.b
    if np.any(target_mu[bad] <= 0):
        A = model.physical_matrix if physical is not None else model.M
        scales, *_ = _scaled_svd(A)
        B = A / scales
        rows = np.maximum(np.maximum(np.linalg.norm(B, axis=1), model.b), 1)
        required = ((model.x > 0) | bad).astype(float)
        lp = linprog(np.r_[np.zeros(B.shape[1]), -1.0],
                     A_ub=np.column_stack((-B / rows[:, None], required)),
                     b_ub=model.b / rows,
                     bounds=[(0, None) if physical is not None else (None, None)] * B.shape[1] + [(0, 1)],
                     method="highs")
        if not lp.success or lp.x[-1] <= 0:
            return theta, physical, False
        target_parameters = lp.x[:-1] / scales
        if physical is not None:
            target_physical = target_parameters
            target = model.from_physical(target_physical)
        else:
            target = target_parameters
        target_mu = model.M @ target + model.b
    if np.any(target_mu[bad] <= 0):
        return theta, physical, False
    alpha = max(16 * np.finfo(float).eps,
                float(np.max(8 * roundoff[bad] / (target_mu[bad] - mu[bad]))))
    for _ in range(8):
        if alpha > 1e-4:
            break
        if physical is not None:
            corrected_physical = (1 - alpha) * physical + alpha * target_physical
            corrected = model.from_physical(corrected_physical)
        else:
            corrected_physical = None
            corrected = (1 - alpha) * theta + alpha * target
        expected = model.M @ corrected + model.b
        if np.all(expected >= 0) and np.all(expected[model.x > 0] > 0):
            return corrected, corrected_physical, True
        alpha *= 4
    return theta, physical, False


def _poisson(model, method, max_iter, tol):
    nonnegative = method == "poisson_nonnegative"
    A = model.physical_matrix if nonnegative else model.M
    x, b = model.x, model.b
    count_scale = max(1.0, float(np.mean(x)), float(np.mean(np.abs(b))))
    T, B, rank, condition = _poisson_basis(A, count_scale, nonnegative)
    if B.shape[1] == 0:
        if np.any(b[x > 0] <= 0):
            raise ValueError("Положительные отсчёты невозможны при нулевой матрице и фоне")
        theta = np.zeros(model.M.shape[1])
        value = float(np.sum(b) - np.sum(x[x > 0] * np.log(b[x > 0])))
        return _result(model, method, theta, None, value, True,
                       "Модель не зависит от параметров", 0, float("inf"),
                       ["Нулевой ранг; ковариация не вычисляется"])
    # P v>=0 означает z=T v>=0. Нормировка каждой строки не меняет знак.
    P = T / np.linalg.norm(T, axis=1)[:, None] if nonnegative else None
    initial = _poisson_start(B, b, x, P)
    row_scale = np.maximum(np.maximum(np.linalg.norm(B, axis=1), np.abs(b)), 1.0)
    constraints = [LinearConstraint(B / row_scale[:, None], -b / row_scale, np.inf)]
    if P is not None:
        constraints.append(LinearConstraint(P, 0, np.inf))
    positive = x > 0

    def value(v):
        mu = B @ v + b
        if np.any(mu < 0) or np.any(mu[positive] <= 0):
            return np.inf  # Расширенный критерий: недопустимый mu никогда не обрезается.
        # Половина девианса отличается от -ell_P лишь константой по данным.
        observed, expected = x[positive], mu[positive]
        delta = (expected - observed) / observed
        close = np.abs(delta) < 0.5
        loss_terms = np.empty_like(observed)
        loss_terms[close] = observed[close] * (delta[close] - np.log1p(delta[close]))
        # При mu<<x delta может округлиться к -1: log(mu) остаётся конечным.
        loss_terms[~close] = (expected[~close] - observed[~close]
                              - observed[~close] * (np.log(expected[~close]) - np.log(observed[~close])))
        loss = np.sum(mu[~positive]) + np.sum(loss_terms)
        return float(loss / count_scale)

    def gradient(v):
        mu = B @ v + b
        if np.any(mu[positive] <= 0):
            # На границе log не определён; эта точка не может быть принята.
            return np.full(B.shape[1], np.nan)
        residual = np.ones_like(x)
        residual[positive] -= x[positive] / mu[positive]
        return B.T @ residual / count_scale

    def hessian(v):
        mu = B @ v + b
        weighted = B[positive] * (np.sqrt(x[positive]) / mu[positive])[:, None]
        return weighted.T @ weighted / count_scale

    def stationarity(v):
        """Невязка ККТ: grad f=C_active.T lambda, lambda>=0."""
        g = gradient(v)
        if not np.all(np.isfinite(g)):
            return np.inf
        blocks = [constraint.A for constraint in constraints]
        lower = np.concatenate([np.broadcast_to(constraint.lb, constraint.A.shape[0])
                                for constraint in constraints])
        C_all = np.vstack(blocks)
        slacks = C_all @ v - lower
        active = slacks <= 1e-8 * (1 + np.abs(lower))
        if not np.any(active):
            return float(np.linalg.norm(g, np.inf))
        try:
            multipliers, _ = nnls(C_all[active].T, g, maxiter=max_iter)
        except RuntimeError:
            return np.inf
        residual = g - C_all[active].T @ multipliers
        return float(np.linalg.norm(residual, np.inf) / max(1.0, np.linalg.norm(g, np.inf)))

    def polish(v):
        # Ньютоновское уточнение с равенствами на активных границах. Нулевые
        # отсчёты дают линейный член, поэтому обычный Hessian может быть
        # вырожденным; седловая система учитывает эти границы явно.
        for _ in range(30):
            g = gradient(v)
            mu = B @ v + b
            if not np.all(np.isfinite(g)) or np.any(mu < 0) or stationarity(v) <= tol:
                break
            H = hessian(v)
            eq_rows, eq_slack = [], []
            active_mean = (~positive) & (mu / row_scale <= 1e-7)
            if np.any(active_mean):
                eq_rows.append(B[active_mean] / row_scale[active_mean, None])
                eq_slack.append(mu[active_mean] / row_scale[active_mean])
            if P is not None:
                slack = P @ v
                active_physical = slack <= 1e-7
                if np.any(active_physical):
                    eq_rows.append(P[active_physical])
                    eq_slack.append(slack[active_physical])
            try:
                if eq_rows:
                    E = np.vstack(eq_rows)
                    saddle = np.block([[H, -E.T], [E, np.zeros((len(E), len(E)))]] )
                    rhs = np.r_[-g, -np.concatenate(eq_slack)]
                    step = np.linalg.lstsq(saddle, rhs, rcond=None)[0][:len(v)]
                else:
                    step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                break
            if g @ step >= 0 or np.linalg.norm(step, np.inf) == 0:
                break
            alpha = 1.0
            change = B @ step
            falling = change < 0
            if np.any(falling):
                alpha = min(alpha, 0.99 * float(np.min(-mu[falling] / change[falling])))
            if P is not None:
                slack, change = P @ v, P @ step
                falling = change < 0
                if np.any(falling):
                    alpha = min(alpha, 0.99 * float(np.min(-slack[falling] / change[falling])))
            if alpha <= 0:
                break
            old = value(v)
            old_stationarity = stationarity(v)
            for _ in range(50):
                candidate = v + alpha * step
                candidate_value = value(candidate)
                indistinguishable = (np.isfinite(candidate_value)
                                     and candidate_value <= old + 64 * np.finfo(float).eps * max(1, abs(old))
                                     and stationarity(candidate) < 0.9 * old_stationarity)
                if candidate_value <= old + 1e-4 * alpha * (g @ step) or indistinguishable:
                    v = candidate
                    break
                alpha *= 0.5
            else:
                break
        return v

    if nonnegative and np.all(A >= 0):
        # Для S_c>=0 и b>=0 границы z>=0 уже гарантируют mu>=0; log-критерий
        # дополнительно исключает mu=0 при x>0. Здесь прямые границы устойчивее
        # общей линейной системы и сохраняются без верхних ограничений.
        optimum = minimize(value, initial, jac=gradient, method="L-BFGS-B",
                           bounds=[(0, None)] * B.shape[1],
                           options={"maxiter": max_iter, "ftol": min(tol, 1e-14),
                                    "gtol": tol, "maxls": 100})
    else:
        optimum = minimize(value, initial, jac=gradient, method="SLSQP",
                           constraints=constraints,
                           options={"maxiter": max_iter, "ftol": min(tol, 1e-12)})
    v = polish(optimum.x)
    mu = B @ v + b
    valid = np.all(mu >= 0) and np.all(mu[positive] > 0)
    if P is not None:
        valid = valid and np.all(P @ v >= -1e-12)
    # Второй метод решает ту же задачу с теми же ограничениями, включая z>=0.
    if not optimum.success or not valid or stationarity(v) > max(1e-7, tol * 1000):
        restart = v if valid else initial
        retry = minimize(value, restart, jac=gradient, hess=hessian, method="trust-constr",
                         constraints=constraints,
                         options={"maxiter": max_iter, "gtol": tol,
                                  "xtol": tol, "barrier_tol": tol})
        retry_valid = np.all(B @ retry.x + b >= 0) and np.all((B @ retry.x + b)[positive] > 0)
        if P is not None:
            retry_valid = retry_valid and np.all(P @ retry.x >= -1e-12)
        if retry_valid and (not valid or value(retry.x) < value(v) or retry.success):
            optimum = retry
            v, valid = polish(retry.x), retry_valid
    parameters = T @ v
    if nonnegative:
        # Только остаток округления на активной границе z=0. Значимое
        # отрицательное решение не исправляется и не объявляется допустимым.
        roundoff = 64 * np.finfo(float).eps * (np.abs(T) @ np.abs(v) + 1.0)
        numerical_zero = (parameters < 0) & (parameters >= -roundoff)
        parameters[numerical_zero] = 0.0
    # Решение в физических координатах переводится во входные; это не
    # постобрезание beta, которое нарушило бы цепочечное преобразование T_c.
    theta = model.from_physical(parameters) if nonnegative else parameters
    theta, corrected_physical, corrected_mean = _restore_mean_roundoff(
        model, theta, parameters if nonnegative else None)
    if nonnegative:
        parameters = corrected_physical
    mu = model.M @ theta + b
    valid = np.all(mu >= 0) and np.all(mu[positive] > 0)
    if nonnegative:
        valid = valid and np.all(parameters >= 0)
    warnings = []
    if corrected_mean:
        warnings.append("Отрицательный остаток округления mu устранён малой допустимой поправкой параметров; mu не обрезался")
    kkt_residual = stationarity(v)
    stationary = kkt_residual <= max(1e-7, tol * 1000)
    if not stationary:
        warnings.append(f"Условия оптимальности ККТ не подтверждены: невязка {kkt_residual:.3g}")
    K = None
    if nonnegative:
        warnings.append("Ограниченная оценка: обычная обратная информационная матрица не является её ковариацией")
    elif valid:
        # Свободная задача решается в SVD-координатах: theta=T@v. Нулевая
        # компонента theta может получиться вычитанием больших слагаемых.
        # Граница mu=0 проверяется с учётом округления всего перехода,
        # иначе остаток ~1e-14 в DETBK ошибочно выглядит внутренней точкой
        # с допустимой обычной ковариацией. Сами mu и theta не изменяем.
        parameter_magnitudes = np.abs(T) @ np.abs(v)
        mean_roundoff = 32 * np.finfo(float).eps * (np.abs(model.M) @ parameter_magnitudes + b + 1.0)
        zero_rows = mu <= mean_roundoff
        if corrected_mean or np.any(zero_rows & np.any(model.M != 0, axis=1)):
            warnings.append("Граничный mu=0: обычная асимптотическая ковариация неприменима")
        else:
            keep = mu > 0
            K, rank, condition = _covariance(model.M[keep] / np.sqrt(mu[keep])[:, None])
    if rank != model.M.shape[1]:
        warnings.append("Неполный ранг: параметры не идентифицируются однозначно; ковариация не вычисляется")
    elif condition > 1e8:
        warnings.append("Сильное перекрытие компонент: большая обусловленность нормированных столбцов")
    if not valid:
        warnings.append("Найденная точка нарушает область допустимых математических ожиданий")
    # Возвращаем -ell_P без log(x!); это eq:poisson_loglikelihood.
    objective = float(np.sum(mu) - np.sum(x[positive] * np.log(mu[positive]))) if valid else np.inf
    result = _result(model, method, theta, K, objective, valid and stationary,
                     optimum.message, rank, condition, warnings, int(optimum.nit))
    if nonnegative:
        # z — решаемые координаты, сохраняем их непосредственно; Q@theta
        # может внести новый остаток округления после обратного преобразования.
        result.physical_parameters = parameters
    return result


def fit(model: JointModel, method="wls", *, weights=None,
        observation_covariance=None, max_iter=2000, tol=1e-10) -> FitResult:
    """Оценить совместную модель SAMPLE/GASBK/DETBK (или прежнюю NPZ).

    ``theta`` возвращается в model.coordinates, ``physical_parameters`` —
    в [a_V;gamma;zeta] для V86, если переход определён. ``nnls`` и
    ``poisson_nonnegative`` требуют этого перехода. ``weights``: положительный
    вектор или SPD матрица W; ``observation_covariance``: SPD матрица Ω.
    Ковариация свободной оценки условна на фиксированные M и T_V;
    в V86 она включает совместно оцениваемый фон zeta. Для прежней
    двухблочной модели координаты [a_c;gamma] и сдвиг b фиксирован.
    """
    aliases = {"ls": "wls", "mle_free": "poisson", "mle": "poisson_nonnegative"}
    method = aliases.get(method, method)
    if method not in ("wls", "nnls", "poisson", "poisson_nonnegative"):
        raise ValueError("Неизвестный метод: wls, nnls, poisson, poisson_nonnegative")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter <= 0:
        raise ValueError("max_iter должен быть положительным целым")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol должен быть конечным положительным числом")
    if method in ("wls", "nnls"):
        return _least_squares(model, method, weights, observation_covariance, int(max_iter))
    if weights is not None or observation_covariance is not None:
        raise ValueError("Пуассоновский критерий использует исходные отсчёты и не принимает веса МНК")
    return _poisson(model, method, int(max_iter), tol)
