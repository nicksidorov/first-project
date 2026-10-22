"""Render distributions of per-observation iterative MDC estimates.

This module only reads the audited histogram_summary.json and
histogram_values.npz. It does not generate counts, fit spectra, or change Ld95.
All stored histogram values and displayed activities are in mBq/m3.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'xemat-hist-matplotlib'))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
DATES = {'may22': '22 мая', 'may26': '26 мая'}
FAMILIES = {'wls': 'МНК', 'poisson': 'ММП'}
ISOTOPES = {1: 'Xe135', 2: 'Xe133', 3: 'Xe131m', 4: 'Xe133m'}
SCENARIOS = {'null': 'A = 0', 'signal': 'A = Ld95'}
COLORS = {'null': '#7954a1', 'signal': '#008c88'}


def number(value, digits=4):
    return f'{float(value):.{digits}f}'.replace('.', ',')


def integer(value):
    return f'{int(value):,}'.replace(',', '\u202f')


def array_key(curve, scenario):
    return f"{curve['dataset']}_{curve['family']}_i{curve['target']}_{scenario}"


def ordered_curves(summary):
    return sorted(summary['curves'], key=lambda c: (
        list(DATES).index(c['dataset']), list(FAMILIES).index(c['family']), c['target']))


def load_checked(root):
    summary = json.loads((root/'histogram_summary.json').read_text(encoding='utf-8'))
    expected = {(d, f, i) for d in DATES for f in FAMILIES for i in ISOTOPES}
    curves = summary['curves']
    found = {(c['dataset'], c['family'], c['target']) for c in curves}
    if len(curves) != len(expected) or found != expected:
        raise ValueError('Expected sixteen date/method/isotope combinations')
    arrays = {}
    with np.load(root/'histogram_values.npz', allow_pickle=False) as archive:
        for curve in curves:
            if curve['isotope'] != ISOTOPES[curve['target']]:
                raise ValueError('Target and isotope name disagree')
            for key in ('nominal_MDC_mBq_m3', 'signal_A_mBq_m3', 'cutoff_c'):
                if not np.isfinite(curve[key]) or curve[key] <= 0:
                    raise ValueError(f'Invalid positive reference {key}')
            for scenario in SCENARIOS:
                key = array_key(curve, scenario)
                a = np.array(archive[key], dtype=float, copy=True)
                if a.ndim != 1 or not len(a):
                    raise ValueError(f'{key}: expected a nonempty one-dimensional array')
                if np.any(np.isinf(a)):
                    raise ValueError(f'{key}: represent unavailable MDC by NaN, not infinity')
                valid = a[np.isfinite(a)]
                if not len(valid) or np.any(valid <= 0):
                    raise ValueError(f'{key}: no valid positive MDC values')
                record = curve[scenario]
                if (record['n_attempted'], record['n_valid'], record['n_missing']) != (
                        len(a), len(valid), len(a)-len(valid)):
                    raise ValueError(f'{key}: array lengths and reported counts disagree')
                quantiles = np.quantile(valid, [.025, .5, .975])
                actual = dict(mean=float(np.mean(valid)), median=float(quantiles[1]),
                    sd=float(np.std(valid, ddof=1)), q025=float(quantiles[0]), q975=float(quantiles[2]))
                for field, value in actual.items():
                    if not np.isclose(record[field], value, rtol=1e-10, atol=1e-12):
                        raise ValueError(f'{key}: statistic {field} disagrees with raw MDC values')
                arrays[key] = a
    return summary, arrays


def shared_bins(null, signal):
    """Use all successful values, with one shared set of edges within a pair."""
    both = np.concatenate([null, signal])
    low, high = float(both.min()), float(both.max())
    if low == high:
        margin = max(abs(low)*.02, 1e-12)
        return np.linspace(low-margin, high+margin, 13)
    preliminary = np.histogram_bin_edges(both, bins='fd')
    count = min(60, max(16, len(preliminary)-1))
    return np.linspace(low, high, count+1)


def render_figure(root, curves, arrays, dataset, family):
    stem = f'mdc_hist_{dataset}_{family}'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'}):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))
        for ax, curve in zip(axes.flat, curves):
            values = {scenario: arrays[array_key(curve, scenario)] for scenario in SCENARIOS}
            valid = {scenario: a[np.isfinite(a)] for scenario, a in values.items()}
            edges = shared_bins(valid['null'], valid['signal'])
            for scenario in SCENARIOS:
                heights = np.histogram(valid[scenario], bins=edges)[0] * 100 / len(valid[scenario])
                if not np.isclose(heights.sum(), 100):
                    raise ValueError('Histogram bins lost successful observations')
                ax.stairs(heights, edges, fill=True, color=COLORS[scenario], alpha=.19, linewidth=0)
                ax.stairs(heights, edges, color=COLORS[scenario], linewidth=1.5)
            ax.axvline(curve['nominal_MDC_mBq_m3'], color='#888888', ls=':', lw=1.7)
            ax.axvline(curve['signal_A_mBq_m3'], color='#222222', ls='--', lw=1.5)
            low = min(edges[0], curve['nominal_MDC_mBq_m3'], curve['signal_A_mBq_m3'])
            high = max(edges[-1], curve['nominal_MDC_mBq_m3'], curve['signal_A_mBq_m3'])
            margin = max((high-low)*.04, 1e-10)
            ax.set_xlim(low-margin, high+margin)
            ax.set_ylim(bottom=0)
            ax.margins(y=.09)
            ax.set_title(curve['isotope'], fontweight='bold', fontsize=13, pad=31)
            counts = (f"A = Ld95 = {number(curve['signal_A_mBq_m3'], 5)} мБк/м³\n"
                      f"N₀ = {integer(curve['null']['n_valid'])}; N_Ld = {integer(curve['signal']['n_valid'])}")
            ax.text(.5, 1.015, counts, transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
            ax.set_xlabel('Оценка MDC, мБк/м³', fontsize=11)
            ax.set_ylabel('Доля проб в интервале, %', fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: number(value, 3)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='y', alpha=.17)
        handles = [
            Patch(facecolor=COLORS['null'], edgecolor=COLORS['null'], alpha=.45,
                  label='A = 0: пустая проба'),
            Patch(facecolor=COLORS['signal'], edgecolor=COLORS['signal'], alpha=.45,
                  label='A = Ld95: один введённый изотоп'),
            Line2D([], [], color='#888888', ls=':', lw=1.7,
                   label='Формульный MDC при заданном чистом фоне'),
            Line2D([], [], color='#222222', ls='--', lw=1.5,
                   label='Ld95 после подстройки Lc'),
        ]
        fig.suptitle(f"Гистограммы оценки MDC · {DATES[dataset]} · {FAMILIES[family]}",
                     fontsize=17, fontweight='bold', y=.99)
        fig.text(.5, .94, 'Стандартная итерационная формула по каждой реализации; множитель c в MDC не введён',
                 ha='center', fontsize=10)
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .026),
                   ncol=2, frameon=False, fontsize=9.6, columnspacing=1.8)
        fig.text(.5, .006, 'Каждая гистограмма нормирована на 100% доступных MDC; границы столбцов общие внутри пары.',
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.20, hspace=.62, wspace=.28)
        fig.savefig(root/(stem+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(root/(stem+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    return stem


def write_csv(root, summary):
    fields = ['Комплект', 'Метод', 'Изотоп', 'Условие', 'Истинная A, мБк/м3',
              'Попыток', 'Доступных MDC', 'Пропусков MDC', 'Среднее MDC, мБк/м3',
              'Медиана MDC, мБк/м3', 'SD MDC, мБк/м3', 'Квантиль 2,5%, мБк/м3',
              'Квантиль 97,5%, мБк/м3', 'Формульный MDC заданного чистого фона, мБк/м3',
              'Ld95 после подстройки, мБк/м3', 'c']
    with (root/'histogram_statistics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(fields)
        for curve in ordered_curves(summary):
            for scenario, label in SCENARIOS.items():
                record = curve[scenario]
                writer.writerow([DATES[curve['dataset']], FAMILIES[curve['family']], curve['isotope'],
                    label, number(0 if scenario=='null' else curve['signal_A_mBq_m3'], 8),
                    record['n_attempted'], record['n_valid'], record['n_missing'],
                    *[number(record[field], 8) for field in ('mean', 'median', 'sd', 'q025', 'q975')],
                    number(curve['nominal_MDC_mBq_m3'], 8), number(curve['signal_A_mBq_m3'], 8),
                    number(curve['cutoff_c'], 8)])


def median_range(record):
    return (f"{number(record['median'])} [{number(record['q025'])}–{number(record['q975'])}]")


def content_blocks(summary):
    blocks = [{'type': 'page_break'},
        {'type': 'heading', 'text': '11. Гистограммы MDC, оценённого по отдельным наблюдениям'},
        {'type': 'paragraph', 'text':
         'В этом разделе каждая проба даёт собственную оценку MDC: программа подгоняет наблюдаемые '
         'отсчёты и применяет стандартную итерационную формулу с α = β = 0,05. Неотрицательная '
         'вспомогательная подгонка задаёт нулевой спектр после удаления целевого свежего вклада; '
         'оценки остальных компонентов и детекторного фона сохраняются. Множитель c в эту формулу MDC '
         'не вводится. Это распределение расчётных MDC отдельных проб, а не повторных оценок Ld95 '
         'по кривым мощности.'},
        {'type': 'paragraph', 'text':
         'Рассмотрены два фиксированных истинных уровня: A = 0 и A = Ld95. Для второго уровня '
         'использована ранее найденная активность Ld95 после подстройки Lc, отдельно для изотопа, '
         'метода и майского комплекта; сигнальные A для МНК и ММП поэтому немного различаются. '
         'В сигнальной серии ненулевой только выбранный свежий изотоп: истинная газовая память '
         'и остальные изотопы равны нулю. '
         'На рисунках указаны введённые значения; в генерации использованы полные числа, без округления. '
         'Их погрешность Монте-Карло в новом эксперименте не разыгрывалась.'},
        {'type': 'paragraph', 'text':
         'При A = 0 повторно использованы 10 000 проверочных наблюдений каждого комплекта и уже '
         'полученные подгонки: итерационный MDC рассчитан заново. При A = Ld95 сгенерированы новые '
         '3000 наблюдений на каждое из 16 сочетаний, всего 48 000 новых векторов. Разыгрываются '
         'все 21 отсчёт SAMPLE/GASBK/DETBK; при обработке оцениваются все 17 параметров. '
         'Систематические ошибки не включены. В гистограммы входят все доступные MDC независимо '
         'от того, обнаружен изотоп или нет; пропуски учитываются отдельно.'},
        {'type': 'paragraph', 'text':
         'По горизонтали — оценка MDC, по вертикали — доля проб, попавших в соответствующий '
         'интервал столбца. Фиолетовый цвет обозначает A = 0, бирюзовый — A = Ld95. '
         'Каждая гистограмма нормирована на 100% доступных MDC, поэтому сравниваются формы '
         'распределений при разных числах повторов. Границы столбцов общие внутри каждой пары; '
         'хвосты распределения не обрезаны.'},
        {'type': 'paragraph', 'text':
         'Серая пунктирная линия — формульный MDC при заданном истинном чистом фоне. Для него '
         'использована полная информация Фишера, включая статистическую неопределённость оценки '
         'DETBK. Чёрная штриховая линия — '
         'численный Ld95 откалиброванного правила, который использован как A для сигнальной серии. '
         'Это разные характеристики: отличие гистограммы стандартного MDC от чёрной линии само '
         'по себе не является оценкой смещения относительно того же самого параметра. В таблицах '
         'приведена медиана и центральные 95% значений MDC (квантили 2,5% и 97,5%). '
         'Это разброс отдельных оценок, а не 95% доверительный интервал истинного Ld95.'},
    ]
    blocks.append({'type': 'paragraph', 'text':
        'Средние, стандартные отклонения, медианы, квантили и числа попыток сохранены в '
        'histogram_statistics.csv. Эти гистограммы характеризуют случайность оценки MDC '
        'при фиксированных калибровках и выбранном истинном фоне. Ранее найденные коэффициенты c '
        'и итоговые Ld95 по новым сериям не перенастраивались.'})
    blocks.append({'type': 'paragraph', 'text':
        'MDC зависит от оценённых по наблюдениям фона, памяти и остальных компонентов. '
        'Даже после удаления целевого вклада эти оценки остаются случайными; их совместное '
        'распределение может меняться при наличии сигнала. Поэтому распределение MDC может '
        'зависеть от истинной активности.'})
    for group_index, (dataset, family) in enumerate(
            (pair for dataset in DATES for pair in [(dataset, f) for f in FAMILIES]), start=1):
        selected = [c for c in ordered_curves(summary) if c['dataset']==dataset and c['family']==family]
        blocks.extend([
            {'type': 'page_break'},
            {'type': 'heading', 'text': f"11.{group_index}. {DATES[dataset]} · {FAMILIES[family]}"},
            {'type': 'image', 'file': f'mdc_hist_{dataset}_{family}.png',
             'caption': f"Рисунок {4+group_index}. Распределения стандартного MDC по наблюдениям: "
                        f"{DATES[dataset]}, {FAMILIES[family]}. N на панели — число доступных оценок."},
            {'type': 'paragraph', 'text': 'Все значения в таблице — мБк/м³. В квадратных скобках — '
                                         'центральные 95% значений, а не доверительный интервал Ld95.'},
            {'type': 'table', 'headers': ['Изотоп', 'A = 0: медиана [2,5–97,5%]',
                                         'A = Ld95: медиана [2,5–97,5%]', 'Введённая A = Ld95'],
             'rows': [[c['isotope'], median_range(c['null']), median_range(c['signal']),
                       number(c['signal_A_mBq_m3'], 5)] for c in selected],
             'widths': [.13, .33, .33, .21]},
        ])
        missing = sum(c[s]['n_missing'] for c in selected for s in SCENARIOS)
        if missing:
            details = '; '.join(
                f"{c['isotope']}, {SCENARIOS[s]}: {c[s]['n_valid']}/{c[s]['n_attempted']} доступных"
                for c in selected for s in SCENARIOS if c[s]['n_missing'])
            blocks.append({'type': 'paragraph', 'text':
                'Есть недоступные MDC: '+details+'. Они не заменены новыми реализациями. '
                'Гистограммы и квантили условны относительно успешного расчёта MDC.'})
        else:
            blocks.append({'type': 'paragraph', 'text':
                'Для каждого изотопа доступны все '+integer(selected[0]['null']['n_attempted'])+
                ' MDC при A = 0 и все '+integer(selected[0]['signal']['n_attempted'])+
                ' MDC при A = Ld95; пропусков в этих гистограммах нет.'})
    increases = [100*(c['signal']['median']/c['null']['median']-1)
                 for c in summary['curves']]
    example = next(c for c in summary['curves']
                   if (c['dataset'], c['family'], c['target']) == ('may22', 'poisson', 4))
    blocks.append({'type': 'paragraph', 'text':
        'В этих сериях при A = Ld95 медианы стандартного MDC выше, чем при A = 0, на '+
        number(min(increases), 2)+'–'+number(max(increases), 2)+'%. Стандартные отклонения и '
        'ширина центральных 95% диапазонов также увеличились, особенно для Xe131m и Xe133m, '
        'где заметен правый хвост. Для примера, 22 мая, '
        'ММП, Xe133m: '+median_range(example['null'])+' при A = 0 и '+
        median_range(example['signal'])+' при A = Ld95, мБк/м³. '
        'МНК и ММП дают одну и ту же качественную картину.'})
    return blocks


def render_addendum(root: Path | None = None):
    root = Path(root or ROOT).resolve()
    summary, arrays = load_checked(root)
    for dataset in DATES:
        for family in FAMILIES:
            selected = [c for c in ordered_curves(summary)
                        if c['dataset']==dataset and c['family']==family]
            render_figure(root, selected, arrays, dataset, family)
    write_csv(root, summary)
    blocks = content_blocks(summary)
    (root/'hist_blocks.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
    return blocks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    blocks = render_addendum(args.root)
    print(f'Rendered four histogram figures, CSV and {len(blocks)} report blocks.')
