"""Согласованность детекторов в #Header 3 файлов SAMPLE/GASBK/DETBK."""

from collections.abc import Mapping
from numbers import Integral

from constants import DETECTOR_RESPONSE_MATRICES


class DetectorSelectionError(ValueError):
    """Набор спектров требует исправления; GUI показывает предупреждение."""


def validate_detector_ids(records: Mapping[str, Mapping]) -> int:
    """Вернуть общий номер известного детектора без неявной подстановки.

    Калибровки относятся к конкретному детектору. Поэтому отсутствие
    номера или смесь 221/222 не позволяют выбирать матрицу по одному SAMPLE.
    В сообщении сохраняются роли всех проверенных файлов и их номера.
    """
    ids = {}
    invalid = []
    descriptions = []
    for role, data in records.items():
        raw = data.get("detector_id")
        descriptions.append(f"{role}: {raw if raw is not None else 'номер не найден'}")
        if isinstance(raw, Integral) and not isinstance(raw, bool):
            ids[role] = int(raw)
        elif isinstance(raw, str) and raw.strip().isascii() and raw.strip().isdigit():
            ids[role] = int(raw.strip())
        else:
            invalid.append(role)
    details = "; ".join(descriptions)
    if not records or invalid:
        raise DetectorSelectionError(
            "Не удалось определить номер детектора в #Header 3. "
            f"{details}. Выберите файлы с указанным номером детектора."
        )
    if len(set(ids.values())) != 1:
        raise DetectorSelectionError(
            "Номера детекторов не совпадают. "
            f"{details}. SAMPLE, GASBK и DETBK должны относиться к одному детектору. "
            "Расчёт остановлен; выберите согласованный комплект файлов."
        )
    detector_id = next(iter(ids.values()))
    if detector_id not in DETECTOR_RESPONSE_MATRICES:
        available = ", ".join(str(value) for value in sorted(DETECTOR_RESPONSE_MATRICES))
        raise DetectorSelectionError(
            f"Нет калибровки для детектора {detector_id}. {details}. "
            f"Поддерживаются детекторы: {available}."
        )
    return detector_id
