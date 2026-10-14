"""Расчёт без окна: тот же адаптер и те же методы, что в GUI."""
import argparse

from .io import save_json


def main(argv=None):
    from process_measurement import METHOD_NAMES, process_prepared_measurement
    parser = argparse.ArgumentParser(description="XeMat: готовая совместная модель NPZ")
    parser.add_argument('model', help='Подготовленные M,x,b и метаданные NPZ')
    parser.add_argument('--method', choices=list(METHOD_NAMES), default='wls')
    parser.add_argument('--sample')
    parser.add_argument('--gasbk')
    parser.add_argument('--detbk')
    parser.add_argument('--fixed-roi', action='store_true')
    parser.add_argument('--output', default='results.json')
    args = parser.parse_args(argv)
    try:
        result = process_prepared_measurement(
            args.model, method=args.method, sample_file=args.sample,
            background_file=args.gasbk, detector_file=args.detbk,
            use_fixed_roi=args.fixed_roi)
        save_json(result, args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Ошибка расчёта: {exc}\n")
    solution = result['fit']
    print(f"{METHOD_NAMES[args.method]}; ранг={solution.rank}; критерий={solution.objective:.8g}")
    print(f"Координаты входа/выхода theta: {result['model'].coordinates}")
    print(f"Результаты: {args.output}")
    for message in result['warnings']:
        print(f"Примечание: {message}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
