"""Совместная модель статьи: x=[c;d], mu=M@theta+b_det.

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
        if self.coordinates not in ("beta_gamma", "a_c_gamma"):
            raise ValueError("coordinates: ожидается beta_gamma или a_c_gamma")
        if self.T_c is not None:
            T_c = _array(self.T_c, 2, "T_c")
            if T_c.shape != (n, n):
                raise ValueError("T_c должна быть квадратной матрицей для параметров пробы")
            # Проверяем ранг после масштабирования: единицы столбцов могут различаться.
            scales = np.linalg.norm(T_c, axis=0)
            if np.any(scales == 0) or np.linalg.matrix_rank(T_c / scales) != n:
                raise ValueError("T_c должна быть невырожденной")
            object.__setattr__(self, "T_c", T_c)

    @property
    def coordinate_transform(self):
        """Q: theta во входных координатах -> z=[a_c;gamma]."""
        if self.coordinates == "a_c_gamma":
            return np.eye(self.M.shape[1])
        if self.T_c is None:
            return None
        return block_diag(self.T_c, np.eye(self.M.shape[1] - self.n_sample_parameters))

    @property
    def physical_matrix(self):
        """S_c=M Q^{-1}, eq:concentration_response_matrix; без явной inverse."""
        Q = self.coordinate_transform
        if Q is None:
            raise ValueError("Для физических координат [a_c;gamma] необходима T_c")
        return solve(Q.T, self.M.T).T

    def to_physical(self, theta):
        Q = self.coordinate_transform
        return None if Q is None else Q @ np.asarray(theta, dtype=float)

    def from_physical(self, parameters):
        Q = self.coordinate_transform
        if Q is None:
            raise ValueError("Для обратного преобразования необходима T_c")
        return solve(Q, np.asarray(parameters, dtype=float))


def build_joint_matrix(R_c, R_g1, R_g2, F):
    """R_F=[[R_c, R_g1@F],[0,R_g2]]; отклики пробы и памяти различны."""
    R_c, R_g1, R_g2, F = (
        _array(v, 2, name) for v, name in
        ((R_c, "R_c"), (R_g1, "R_g1"), (R_g2, "R_g2"), (F, "F"))
    )
    if R_c.shape[0] != R_g1.shape[0] or R_g1.shape[1] != F.shape[0]:
        raise ValueError("Размеры откликов SAMPLE и матрицы переноса F не согласованы")
    if R_g2.shape[1] != F.shape[1]:
        raise ValueError("Размеры отклика GASBK и матрицы переноса F не согласованы")
    return np.block([[R_c, R_g1 @ F],
                     [np.zeros((R_g2.shape[0], R_c.shape[1])), R_g2]])
