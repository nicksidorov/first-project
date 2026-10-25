"""Build the numerical report from saved April results. No fitting or plots."""
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parent
VARIANTS=('original_034','test_035','test_036')
VALUES=(.34,.35,.36)
METHODS=('wls','nnls','poisson','poisson_nonnegative')
ISOTOPES=('Xe135','Xe133','Xe131m','Xe133m')


def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def number(value):return '—' if value in ('',None) else f'{float(value):.7g}'


def main():
    source=json.loads((ROOT/'baseline/scenario.json').read_text());meta=source['metadata']
    summaries=rows('summary.csv');activities=rows('concentrations.csv');rois=rows('roi_diagnostics.csv')
    summary={(r['variant'],r['method_key']):r for r in summaries}
    activity={(r['variant'],r['method_key'],r['isotope']):r for r in activities}
    roi={(r['variant'],r['method_key'],r['block'],int(r['roi'])):r for r in rois}
    lines=['# XeMat: апрельская пара, три коэффициента Xe133 → SAMPLE ROI 7','',
        'Обработаны **SPIKE20260425010733595.PHD** (роль SAMPLE), **GASBK20260424152856639.PHD** '
        'и прежний **DETBK20250610213044796.PHD**. Детектор 221, фиксированные ROI и эффективности. '
        'Сравниваются Rs[ROI7, Xe133] = **0.34, 0.35 и 0.36**. Все 12 расчётов четырьмя методами сошлись. '
        'Эллипсы и другие графики не строились.','',
        '**Основной результат:** в этой паре увеличение коэффициента ухудшает согласие. '
        'Для свободного ММП D/ν растёт **5.97364 → 7.46624 → 12.16389**. '
        'При исходном 0.34 ROI 7 согласуется с наблюдениями; основное несогласие находится в SAMPLE ROI 8.','',
        '**Матрицы памяти Rg сохранены.** В пределах этого сравнения не меняются R_g0, R_g1, R_g2, F, '
        'эффективности, времена, объём, наблюдения и преобразование к активностям. '
        'Оценки параметров памяти и DETBK каждый раз получаются из полной совместной подгонки. '
        'Рабочий constants.py остаётся исходным; замена сделана только в копии реестра внутри отдельного процесса.','',
        '## Общий девианс и ROI 7: свободный пуассоновский ММП','',
        'Во всех вариантах в SAMPLE ROI 7 наблюдалось **10449 отсчётов**. '
        'Число степеней свободы для каждого фиксированного варианта ν=21−17=4.','',
        '| Rs | μ ROI 7 | x − μ | Вклад D ROI 7 | Общий D | D/ν |',
        '|---:|---:|---:|---:|---:|---:|']
    for v,rvalue in zip(VARIANTS,VALUES):
        r=summary[v,'poisson']
        lines.append(f"| {rvalue:.2f} | {float(r['sample_ROI7_mu']):.6f} | {float(r['sample_ROI7_residual']):+.6f} | {float(r['sample_ROI7_D']):.6f} | {float(r['statistic']):.6f} | {float(r['reduced_statistic']):.6f} |")
    lines += ['', 'При 0.34 модель недооценивает ROI 7 лишь на 12.91 отсчёта. '
        'При 0.35 и 0.36 появляется завышение на 220.13 и 449.79 отсчёта. '
        'Сравниваются результаты полной повторной подгонки всех параметров.','',
        '## Концентрации: свободный пуассоновский ММП','',
        'Единицы — **mBq/m³**. Отрицательные свободные оценки сохранены.','',
        '| Изотоп | Rs=0.34 | Rs=0.35 | Rs=0.36 |', '|---|---:|---:|---:|']
    for isotope in ISOTOPES:
        lines.append('| '+isotope+' | '+' | '.join(f"{float(activity[v,'poisson',isotope]['concentration_mBq_m3']):.6f}" for v in VARIANTS)+' |')
    lines += ['', '## Сопоставление критериев всех методов','',
        '| Rs | LS χ²/ν | NNLS χ²/ν | Poisson ML D/ν | Nonnegative Poisson ML D/ν |',
        '|---:|---:|---:|---:|---:|']
    for v,rvalue in zip(VARIANTS,VALUES):
        lines.append(f'| {rvalue:.2f} | '+' | '.join(f"{float(summary[v,m]['reduced_statistic']):.6f}" for m in METHODS)+' |')
    lines += ['', 'Для NNLS и неотрицательного ММП ν=4 — номинальное число степеней свободы. '
        'Для МНК основной критерий — взвешенный χ², для ММП — пуассоновский девианс. '
        'В CSV дополнительно сохранён девианс при оценках МНК, отдельно от его собственного критерия.','',
        '## Локализация несогласия по ROI','',
        'Вклады свободного ММП вычислены как Dᵢ=2[xᵢ ln(xᵢ/μᵢ)+μᵢ−xᵢ]; при xᵢ=0 вклад равен 2μᵢ. '
        'Их сумма совпадает с полным девиансом.','',
        '| Область | D при 0.34 | D при 0.35 | D при 0.36 |','|---|---:|---:|---:|']
    for block,number_ in [('SAMPLE',3),('SAMPLE',7),('SAMPLE',8),('GASBK',7)]:
        lines.append(f'| {block} ROI {number_} | '+' | '.join(f"{float(roi[v,'poisson',block,number_]['deviance_contribution']):.6f}" for v in VARIANTS)+' |')
    r=roi['original_034','poisson','SAMPLE',8]
    share=100*float(r['deviance_contribution'])/float(summary['original_034','poisson']['statistic'])
    lines += ['',f"При 0.34 в SAMPLE ROI 8 наблюдалось {float(r['x']):.0f}, модель предсказывает "
        f"{float(r['mu']):.3f}; остаток {float(r['residual']):+.3f}. Эта область даёт "
        f"**{share:.2f}%** всего девианса. Изменение ROI 7 немного уменьшает невязку ROI 8, "
        'но увеличивает несогласие в ROI 7 и ROI 3; общий критерий становится хуже.','',
        '| Блок | D при 0.34 | D при 0.35 | D при 0.36 |','|---|---:|---:|---:|']
    for block in ('SAMPLE','GASBK','DETBK'):
        lines.append('| '+block+' | '+' | '.join(f"{float(summary[v,'poisson']['D_'+block]):.6f}" for v in VARIANTS)+' |')
    lines += ['', 'В августовском комплекте увеличение той же ячейки улучшало согласие; '
        'для текущей пары направление эффекта противоположно. Это показывает, что улучшение '
        'на одном спектре не устанавливает универсальное новое значение калибровки. '
        'Причина различия между измерениями в этой проверке не исследовалась.','',
        '## Все оценки и обычные LC/MDC','',
        'Все числа — mBq/m³. σ обозначает одно стандартное отклонение; '
        'для ограниченных оценок обычная ковариация не выдаётся, поэтому σ оставлена пустой. '
        'Обнаружение определяется свободной оценкой > LC в соответствующей семье методов. '
        'LC/MDC получены обычным workflow заново для каждого варианта; дополнительных исследований порогов не было.','',
        '| Rs | Метод | Изотоп | Концентрация | σ stat | σ total | LC | MDC | Обнаружение |',
        '|---:|---|---|---:|---:|---:|---:|---:|---|']
    for r in activities:
        lines.append('| '+' | '.join([r['Rs_ROI7_Xe133'],r['method'],r['isotope'],
            number(r['concentration_mBq_m3']),number(r['sigma_stat_mBq_m3']),number(r['sigma_total_mBq_m3']),
            number(r['LC_mBq_m3']),number(r['MDC_mBq_m3']),'да' if r['detected']=='True' else 'нет'])+' |')
    lines += ['', '## Эффективности и branching','',
        'Эффективности фиксированы из constants.py для 221, одинаковые для SAMPLE/GASBK. '
        'Они не содержат branching; чувствительность I=ε×BR.','',
        '| Изотоп | ε SAMPLE | ε GASBK | Branching | ε×BR |', '|---|---:|---:|---:|---:|']
    for isotope,es,eg in zip(meta['efficiency_isotope_names'],meta['sample_efficiencies'],meta['gasbk_efficiencies']):
        br=source['branching'][isotope]
        lines.append(f'| {isotope} | {es:.6f} | {eg:.6f} | {br:.4f} | {es*br:.9f} |')
    lines += ['| Rn222 | 0.300000 | 0.300000 | 0.3700 | 0.111000000 |','',
        'Для радона сохранено приближение legacy. В обычных полных σ использованы текущие '
        'δI=0.001 и δV₀=0.001 (по 0.1%); в критерий девианса эти систематические вклады не добавляются.','',
        '## Времена и исходные файлы','',
        '| Роль | Файл | Начало измерения | Реальное / живое время, s |','|---|---|---|---:|']
    for role,key,m in [('SAMPLE','sample_file','sample_temporal'),('GASBK','background_file','background_temporal'),('DETBK','detector_file','detector_temporal')]:
        t=meta[m]
        lines.append(f"| {role} | {source['input_files'][key]['file']} | {t['start_time']} | {t['real_time']} / {t['live_time']} |")
    t=meta['time_parameters']
    lines += ['', 'SPIKE имеет статус **PREL** в заголовке. Его фактическое живое время '
        '**27844.14 s** использовано без подмены стандартной длительностью. Начало измерения '
        'из #Acquisition — 24 апреля 2026, хотя имя файла содержит 25 апреля.','',
        f"Объём Xe={meta['xenon_volume_cm3']} cm³; V₀=V_Xe/0.087=**{meta['effective_air_volume_m3']:.12g} m³**. "
        f"Пробоотбор={t['t_a']} s; ожидание={t['t_p']} s; интервал начал GASBK→SAMPLE={t['tau']} s. "
        'Поправки на распад и врастание пересобраны из этой пары, а не взяты из августовской модели.','',
        'Базовая ячейка имеет индекс Python Rs_221[5,2]. После R_c=R_c0@C_s_c '
        'меняются M[5,2] и M[5,4] через дочерний вклад Xe133 в столбце Xe133m. '
        'Это необходимое следствие одной базовой замены. Все другие элементы M неизменны между тремя апрельскими вариантами.','',
        '## Сохранённые файлы','',
        '- `summary.csv` — критерии, сходимость и ROI 7 для всех 12 расчётов.',
        '- `concentrations.csv` — 48 строк концентраций, погрешностей, LC/MDC и обнаружений.',
        '- `roi_diagnostics.csv` — 252 строки: исходные отсчёты, ожидания, остатки и вклады девианса во всех ROI.',
        '- `original_034/`, `test_035/`, `test_036/` — обычные оформленные TXT и полные JSON каждого метода.',
        '- `inputs/`, `kernel_snapshot/`, `baseline/`, `scenario.json` — новые исходные измерения, ядро и полная модель.',
        '- `prepare_baseline.py` — четыре исходных расчёта для этой пары; `run_sensitivity.py` — восемь расчётов двух замен.',
        '- `build_report.py` — оформление сохранённых чисел без повторной подгонки; `verify_sensitivity.py` — независимая проверка.', '',
        f"Версия ядра: `{source['production_commit']}`. Новая калибровка в приложение не записывалась. "
        'Монте-Карло и поиск оптимального коэффициента не запускались. Контрольные суммы входят в архив.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
