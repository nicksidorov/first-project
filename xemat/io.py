"""Безопасный обмен подготовленными массивами NPZ (без pickle)."""
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .model import JointModel, build_joint_matrix


def load_prepared_model(path, *, x=None, b=None):
    """Вернуть (JointModel, extras); x может включать три исходных спектра V86.

    ``b`` — только фиксированный аддитивный сдвиг прежней модели. Отсчёты
    DETBK V86 находятся в последней трети x; их нельзя передавать как b.
    В V86 отсутствие b означает нулевой сдвиг. T_V — новое имя T_c,
    G_a — новое имя поправки пробоотбора G_s; прежние имена сохранены.
    """
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = ("n_sample_parameters", "coordinates")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("В NPZ отсутствуют поля: " + ", ".join(missing))
    n_value = data.pop("n_sample_parameters")
    if n_value.size != 1 or n_value.dtype.kind not in "iu":
        raise ValueError("n_sample_parameters в NPZ: один целочисленный скаляр")
    coordinate_value = data.pop("coordinates")
    if coordinate_value.size != 1 or coordinate_value.dtype.kind != "U":
        raise ValueError("coordinates в NPZ: одна строка Unicode")
    coordinates = str(coordinate_value.item())
    joint_detector = coordinates in ("beta_gamma_zeta", "a_V_gamma_zeta")
    built_from_responses = "M" not in data
    if not built_from_responses:
        M = data.pop("M")
    elif all(name in data for name in ("R_c", "R_g1", "R_g2", "F")):
        if coordinates not in ("beta_gamma", "beta_gamma_zeta"):
            raise ValueError("Матрицы R_c,R_g1,R_g2,F строят R_F в координатах beta_gamma "
                             "или beta_gamma_zeta; для физических координат передайте готовую M")
        if joint_detector and not all(name in data for name in ("kappa_s", "kappa_g")):
            raise ValueError("Для построения совместной модели DETBK нужны kappa_s и kappa_g")
        if not joint_detector and any(name in data for name in ("kappa_s", "kappa_g")):
            raise ValueError("kappa_s/kappa_g требуют координат beta_gamma_zeta")
        M = build_joint_matrix(
            *(data.pop(name) for name in ("R_c", "R_g1", "R_g2", "F")),
            kappa_s=data.get("kappa_s"), kappa_g=data.get("kappa_g"),
        )
    else:
        raise ValueError("Нужна M или четыре матрицы R_c,R_g1,R_g2,F")
    if "n_detector_parameters" in data:
        detector_value = data.pop("n_detector_parameters")
        if detector_value.size != 1 or detector_value.dtype.kind not in "iu":
            raise ValueError("n_detector_parameters в NPZ: один целочисленный скаляр")
        detector_n = int(detector_value.item())
    else:
        # Координаты обязаны явно объявлять zeta; размер не угадывается
        # для старых моделей произвольной размерности.
        if joint_detector and (M.ndim != 2 or M.shape[0] % 3):
            raise ValueError("Совместная M должна иметь 3k строк")
        detector_n = M.shape[0] // 3 if joint_detector else 0
    T_c, T_V = data.get("T_c"), data.get("T_V")
    if T_c is not None and T_V is not None:
        if T_c.shape != T_V.shape or not np.allclose(T_c, T_V, rtol=1e-12, atol=0):
            raise ValueError("T_c и T_V — имена одной матрицы; значения в NPZ должны совпадать")
    if T_V is not None:
        T_c = T_V
        data.setdefault("T_c", T_V.copy())
    G_s, G_a = data.get("G_s"), data.get("G_a")
    if G_s is not None and G_a is not None:
        if G_s.shape != G_a.shape or not np.allclose(G_s, G_a, rtol=1e-12, atol=0):
            raise ValueError("G_s и G_a — имена одной матрицы; значения в NPZ должны совпадать")
    if G_a is not None:
        data.setdefault("G_s", G_a.copy())
    stored_x, stored_b = data.pop("x", None), data.pop("b", None)
    if x is None:
        x = stored_x
    if b is None:
        b = stored_b
    if b is None and joint_detector and M.ndim == 2:
        b = np.zeros(M.shape[0])
    if x is None or b is None:
        raise ValueError("Нужны x,b в NPZ или три файла спектров")
    model = JointModel(M, x, b, int(n_value.item()), T_c, coordinates, detector_n)
    if "weights" in data and "observation_covariance" in data:
        raise ValueError("Укажите weights или observation_covariance, не оба поля")
    return model, data


def jsonable(value):
    """Строгий JSON: отсутствующие ковариации -> null, никаких NaN/Infinity."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def save_json(result, path):
    Path(path).write_text(json.dumps(jsonable(result), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
