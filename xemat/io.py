"""Безопасный обмен подготовленными массивами NPZ (без pickle)."""
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .model import JointModel, build_joint_matrix


def load_prepared_model(path, *, x=None, b=None):
    """Вернуть (JointModel, extras). Векторы спектров могут заменить x,b из NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = ("n_sample_parameters", "coordinates")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("В NPZ отсутствуют поля: " + ", ".join(missing))
    built_from_responses = "M" not in data
    if "M" in data:
        M = data.pop("M")
    elif all(name in data for name in ("R_c", "R_g1", "R_g2", "F")):
        M = build_joint_matrix(*(data.pop(name) for name in ("R_c", "R_g1", "R_g2", "F")))
    else:
        raise ValueError("Нужна M или четыре матрицы R_c,R_g1,R_g2,F")
    n_value = data.pop("n_sample_parameters")
    if n_value.size != 1 or n_value.dtype.kind not in "iu":
        raise ValueError("n_sample_parameters в NPZ: один целочисленный скаляр")
    coordinate_value = data.pop("coordinates")
    if coordinate_value.size != 1 or coordinate_value.dtype.kind != "U":
        raise ValueError("coordinates в NPZ: одна строка Unicode")
    if built_from_responses and str(coordinate_value.item()) != "beta_gamma":
        raise ValueError("Матрицы R_c,R_g1,R_g2,F строят R_F в координатах beta_gamma; "
                         "для a_c_gamma передайте готовую M=S_c")
    stored_x, stored_b = data.pop("x", None), data.pop("b", None)
    if x is None:
        x = stored_x
    if b is None:
        b = stored_b
    if x is None or b is None:
        raise ValueError("Нужны x,b в NPZ или три файла спектров")
    model = JointModel(M, x, b, int(n_value.item()), data.get("T_c"),
                       str(coordinate_value.item()))
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
