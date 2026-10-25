"""Совместная модель V86: x=[c;d;b_DETBK], mu=M@[beta;gamma;zeta].

Для прежних подготовленных данных сохранена модель с фиксированным
аддитивным фоном. Поле ``b`` всегда означает такой аддитивный фон,
а не наблюдения DETBK: в модели V86 ``b`` равно нулю.

Все физические матрицы передаются подготовленными. Здесь нет вычисления
калибровок, поправок распада или цепочек из времён измерения.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, solve


def _array(value, ndim, name):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: требуется конечный массив размерности {ndim}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class JointModel:
    M: np.ndarray
    x: np.ndarray
    b: np.ndarray
    n_sample_parameters: int
    T_c: np.ndarray | None = None
    coordinates: str = "beta_gamma"
    n_detector_parameters: int = 0

    def __post_init__(self):
        for name, ndim in (("M", 2), ("x", 1), ("b", 1)):
            object.__setattr__(self, name, _array(getattr(self, name), ndim, name))
        m, p = self.M.shape
        if not m or not p or self.x.shape != (m,) or self.b.shape != (m,):
            raise ValueError("Размеры M, x, b не согласованы или пусты")
        n = self.n_sample_parameters
        if isinstance(n, (bool, np.bool_)) or int(n) != n or not 0 < n <= p:
            raise ValueError("n_sample_parameters должен быть целым от 1 до числа столбцов M")
        object.__setattr__(self, "n_sample_parameters", int(n))
        if np.any(self.x < 0) or np.any(self.x != np.floor(self.x)):
            raise ValueError("x содержит исходные неотрицательные целочисленные отсчёты")
        if np.any(self.b < 0):
            raise ValueError("Фиксированный детекторный фон b должен быть неотрицательным")
        detector_n = self.n_detector_parameters
        if (isinstance(detector_n, (bool, np.bool_)) or not isinstance(detector_n, (int, float, np.integer, np.floating))
                or not np.isfinite(detector_n) or int(detector_n) != detector_n
                or not 0 <= detector_n < p):
            raise ValueError("n_detector_parameters должен быть целым от 0 до числа столбцов M − 1")
        object.__setattr__(self, "n_detector_parameters", int(detector_n))
        if self.coordinates not in ("beta_gamma", "a_c_gamma", "beta_gamma_zeta", "a_V_gamma_zeta"):
            raise ValueError("coordinates: ожидается beta_gamma, a_c_gamma, beta_gamma_zeta или a_V_gamma_zeta")
        if self.has_detector_background:
            self._validate_joint_detector_model()
        elif self.coordinates in ("beta_gamma_zeta", "a_V_gamma_zeta"):
            raise ValueError("Для координат с zeta требуется n_detector_parameters > 0")
        if self.T_c is not None:
            T_c = _array(self.T_c, 2, "T_c")
            if T_c.shape != (n, n):
                raise ValueError("T_c должна быть квадратной матрицей для параметров пробы")
            # Проверяем ранг после масштабирования: единицы столбцов могут различаться.
            scales = np.linalg.norm(T_c, axis=0)
            if np.any(scales == 0) or np.linalg.matrix_rank(T_c / scales) != n:
                raise ValueError("T_c должна быть невырожденной")
            object.__setattr__(self, "T_c", T_c)

    def _validate_joint_detector_model(self):
        """Проверить, что zeta — отсчёты DETBK, а не фон иной нормировки."""
        k, n = self.n_detector_parameters, self.n_sample_parameters
        if self.coordinates not in ("beta_gamma_zeta", "a_V_gamma_zeta"):
            raise ValueError("Совместная модель DETBK требует координат beta_gamma_zeta или a_V_gamma_zeta")
        if self.M.shape != (3 * k, 2 * n + k):
            raise ValueError("Для SAMPLE/GASBK/DETBK нужна M размера 3k × (2n+k)")
        if np.any(self.b != 0):
            raise ValueError("В совместной модели DETBK аддитивный b должен быть нулевым; отсчёты DETBK входят в x")
        if not np.allclose(self.M[2 * k:, :2 * n], 0, rtol=0, atol=1e-12):
            raise ValueError("DETBK не должен содержать отклики пробы или памяти")
        if not np.allclose(self.M[2 * k:, 2 * n:], np.eye(k), rtol=1e-12, atol=1e-12):
            raise ValueError("Блок zeta в DETBK должен быть единичным: zeta задана в отсчётах DETBK")
        if not np.allclose(self.M[k:2 * k, :n], 0, rtol=0, atol=1e-12):
            raise ValueError("GASBK не должен содержать отклик свежей пробы")
        for label, block in (("SAMPLE", self.M[:k, 2 * n:]),
                             ("GASBK", self.M[k:2 * k, 2 * n:])):
            kappa = block[0, 0]
            if kappa <= 0 or not np.allclose(block, kappa * np.eye(k), rtol=1e-12, atol=0):
                raise ValueError(f"Блок zeta в {label} должен быть kappa × I с kappa > 0")

    @property
    def is_activity_coordinates(self):
        return self.coordinates in ("a_c_gamma", "a_V_gamma_zeta")

    @property
    def n_memory_parameters(self):
        return self.M.shape[1] - self.n_sample_parameters - self.n_detector_parameters

    @property
    def has_detector_background(self):
        return self.n_detector_parameters > 0

    @property
    def input_coordinate_label(self):
        return {"beta_gamma": "[beta; gamma]", "a_c_gamma": "[a_c; gamma]",
                "beta_gamma_zeta": "[beta; gamma; zeta]",
                "a_V_gamma_zeta": "[a_V; gamma; zeta]"}[self.coordinates]

    @property
    def physical_coordinate_label(self):
        return "[a_V; gamma; zeta]" if self.has_detector_background else "[a_c; gamma]"

    @property
    def T_V(self):
        """Обозначение V86 для преобразования объёмных активностей (ранее T_c)."""
        return self.T_c

    @property
    def coordinate_transform(self):
        """Q=diag(T_V,I_n,I_k) для V86; прежние координаты также поддержаны."""
        if self.is_activity_coordinates:
            return np.eye(self.M.shape[1])
        if self.T_c is None:
            return None
        return block_diag(self.T_c, np.eye(self.M.shape[1] - self.n_sample_parameters))

    @property
    def physical_matrix(self):
        """S_V=M Q^{-1} в V86 (ранее S_c); без явного обращения Q."""
        Q = self.coordinate_transform
        if Q is None:
            raise ValueError(f"Для физических координат {self.physical_coordinate_label} необходима T_V (T_c)")
        return solve(Q.T, self.M.T).T

    def to_physical(self, theta):
        Q = self.coordinate_transform
        return None if Q is None else Q @ np.asarray(theta, dtype=float)

    def from_physical(self, parameters):
        Q = self.coordinate_transform
        if Q is None:
            raise ValueError("Для обратного преобразования необходима T_c")
        return solve(Q, np.asarray(parameters, dtype=float))


def build_joint_matrix(R_c, R_g1, R_g2, F, *, kappa_s=None, kappa_g=None):
    """R_F V86, eq:joint_model; без kappa возвращается прежняя двухблочная M."""
    R_c, R_g1, R_g2, F = (
        _array(v, 2, name) for v, name in
        ((R_c, "R_c"), (R_g1, "R_g1"), (R_g2, "R_g2"), (F, "F"))
    )
    if R_c.shape[0] != R_g1.shape[0] or R_g1.shape[1] != F.shape[0]:
        raise ValueError("Размеры откликов SAMPLE и матрицы переноса F не согласованы")
    if R_g2.shape[1] != F.shape[1]:
        raise ValueError("Размеры отклика GASBK и матрицы переноса F не согласованы")
    if kappa_s is not None or kappa_g is not None:
        kappa_s, kappa_g = (
            _positive_kappa(value, name) for value, name in
            ((kappa_s, "kappa_s"), (kappa_g, "kappa_g"))
        )
        k, n = R_c.shape
        if (not k or not n or R_g1.shape != (k, n) or R_g2.shape != (k, n)
                or F.shape != (n, n)):
            raise ValueError("В совместной модели DETBK отклики должны быть k × n, а F — n × n")
        return np.block([
            [R_c, R_g1 @ F, kappa_s * np.eye(k)],
            [np.zeros((k, n)), R_g2, kappa_g * np.eye(k)],
            [np.zeros((k, n)), np.zeros((k, n)), np.eye(k)],
        ])
    return np.block([[R_c, R_g1 @ F],
                     [np.zeros((R_g2.shape[0], R_c.shape[1])), R_g2]])


def _positive_kappa(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name}: нужен положительный конечный скаляр")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: нужен положительный конечный скаляр") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name}: нужен положительный конечный скаляр")
    return result
