"""Build the report from saved tables and metadata, without MCMC or fits."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_csv(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def jread(name):
    return json.loads((ROOT/name).read_text())


def main():
    s, d, source = jread('settings.json'), jread('diagnostics.json'), jread('source/scenario.json')
    conc, ratios = read_csv('concentrations.csv'), read_csv('ratios.csv')
    chain = read_csv('chain_diagnostics.csv')
    plot = jread('plot_metadata.json')
    manifest = jread('source_manifest.json')
    a = np.load(ROOT/'source/measurement_arrays.npz')
    metadata = source['metadata']
    scenario = dict(analysis_date='2026-10-25', analysis='Bayesian posterior of original August real sample',
        original_archive=manifest['archive'], original_archive_sha256=manifest['archive_sha256'],
        original_production_commit=source['production_commit'], input_files=source['input_files'],
        original_scenario=source, original_counts=a['x'].astype(int).tolist(),
        physical_parameter_names=s['parameter_names'], physical_parameter_units=s['parameter_units'],
        numerical_scaling=s['scale'], calibration_uncertainties_included=False,
        model_equation='mu = S z = M Q^{-1} z, Q=diag(T_V,I_12), b=0',
        priors='Constant on the nonnegative physical orthant; independent improper HalfFlat factors',
        random_observation_generation=False, LC_LD_MDC_recomputed=False,
        new_MAP_optimization=False, response_matrices_modified=False)
    (ROOT/'scenario.json').write_text(json.dumps(scenario,indent=2,ensure_ascii=False)+'\n')
    params = [dict(index=i,name=n,native_units=u,summary_units=('mBq/m3 (legacy auxiliary)' if i==0 else 'mBq/m3') if i<5 else u)
              for i,(n,u) in enumerate(zip(s['parameter_names'],s['parameter_units']))]
    (ROOT/'parameters.json').write_text(json.dumps(dict(axes=['chain','draw','parameter'],shape=s['posterior_shape'],
        parameters=params,concentration_order=s['concentration_order'],concentration_units='mBq/m3',
        ratio_order=s['ratio_order'],ratio_units='dimensionless',
        note='First radon coordinate is the original legacy auxiliary parameter, not an atmospheric radon concentration'),indent=2)+'\n')
    versions = dict(python=s['python'],packages=s['software'],platform=s['platform'])
    (ROOT/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in s['software'].items())+'\n')
    lines = [
        '# XeMat: байесовские концентрации и отношения — исходная августовская проба',
        '',
        'Расчёт от 25 октября 2026 года. Исходный архив: `XeMat_Aug17_Xe133_Xe131m_ellipse.zip`. '
        'Отклики, кинетика и калибровки сохранены. Сравнение относится к одной исходной модели с фиксированными калибровками.',
        '',
        '## Данные и модель',
        '',
        '| Блок | Исходный файл | Начало измерения из PHD | Реальное время, с | Живое время, с |',
        '|---|---|---|---:|---:|',
    ]
    for role, key, temporal in [('SAMPLE','sample_file','sample_temporal'),('GASBK','background_file','background_temporal'),('DETBK','detector_file','detector_temporal')]:
        t=metadata[temporal]
        lines.append(f"| {role} | {source['input_files'][key]['file']} | {t['start_time']} | {t['real_time']} | {t['live_time']} |")
    lines += ['', 'Все файлы относятся к детектору **221**. Фиксированные ROI в порядке **1, 2, 3, 5, 6, 7, 8**; '
        'их границы и все матрицы сохранены в `source/scenario.json` и `source/measurement_arrays.npz`. '
        'Исходные отсчёты не вычитались и не заменялись модельными средними:', '',
        '| Блок | ROI 1 | ROI 2 | ROI 3 | ROI 5 | ROI 6 | ROI 7 | ROI 8 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for block, counts in zip(['SAMPLE','GASBK','DETBK'],a['x'].reshape(3,7)):
        lines.append('| '+block+' | '+' | '.join(str(int(v)) for v in counts)+' |')
    t=metadata['time_parameters']
    lines += ['', f"Время сбора {t['t_a']} с, ожидания {t['t_p']} с, интервал начала GASBK → SAMPLE {t['tau']} с. "
        f"Объём Xe {metadata['xenon_volume_cm3']} см³; использованный эквивалентный объём воздуха V₀ = {metadata['effective_air_volume_m3']:.12g} м³. "
        'Заголовочный объём воздуха сохранён как метаданные; применяется исходное преобразование XeMat.', '',
        '| Изотоп | ε SAMPLE | ε GASBK | Branching |', '|---|---:|---:|---:|']
    for i,name in enumerate(['Xe135','Xe133','Xe131m','Xe133m']):
        lines.append(f"| {name} | {metadata['sample_efficiencies'][i]:.6f} | {metadata['gasbk_efficiencies'][i]:.6f} | {source['branching'][name]:.4f} |")
    lines += [
        '', 'Чувствительность использует ε × branching. Радон: исходный режим `legacy`, ε = 0.3, branching = 0.37. '
        'Его первая координата после исходного пересчёта — вспомогательный параметр, не атмосферная концентрация радона.',
        '', r'$$x_j\mid z\sim\operatorname{Poisson}(\mu_j),\qquad \mu=S z=M Q^{-1}z,\quad Q=\operatorname{diag}(T_V,I_{12}).$$',
        '', r'Здесь $z=[a_V;\gamma;\zeta]$: пять компонент свежей пробы, пять компонент памяти и семь средних DETBK. '
        'Совместно варьируются все 17 параметров при всех 21 исходных наблюдениях. '
        'Свежий ксенон ограничен именно в атмосферных координатах после кинетического пересчёта. '
        'Входной аддитивный `b=0`: детекторный фон находится внутри `S @ z`, а не вычитается.',
        '',
        '## Prior и алгоритм',
        '', r'$$p(z)\propto\mathbf{1}_{\{z\geq0\}},\qquad p(z\mid x)\propto\prod_{j=1}^{21}\frac{\mu_j^{x_j}e^{-\mu_j}}{x_j!}\,\mathbf{1}_{\{z\geq0\}}.$$',
        '', 'Prior постоянный без верхней границы; он несобственный, но posterior здесь нормируем. '
        'Матрица S неотрицательна, сумма каждого её столбца положительна, поэтому правдоподобие имеет экспоненциально убывающие хвосты, умноженные на полином.',
        '', 'Использован PyMC NUTS: `u ~ HalfFlat`, `z = scale * u`. Постоянные масштабы — статистические σ исходного ММП, '
        'они служат только для численной параметризации и не задают ширину prior. '
        'PyMC автоматически вводит `q=log(u)` и якобиан `+sum(q)`; вручную второй якобиан не добавлялся. '
        'Проверено совпадение программной лог-плотности с пуассоновским правдоподобием плюс один якобиан.',
        '', 'Опорная MAP — сохранённое исходное неотрицательное пуассоновское решение в физических координатах. '
        'Новая оптимизация не выполнялась. Начала четырёх цепей различны и строго положительны, сохранены в `initial_states.npz`.',
        '', f"Четыре цепи: по {s['tune_per_chain']} warmup и {s['draws_per_chain']} сохранённых итераций, всего {d['posterior_draws']} совместных выборок. "
        f"`target_accept={s['target_accept']}`, `max_treedepth={s['max_treedepth']}`, `init=adapt_diag`. "
        f"Seed sampler = {s['sampler_seed']}, seed начальных состояний = {s['initial_seed']}. "
        'Данные не моделировались заново; Lc, Ld, MDC не пересчитывались.',
        '', '## Сходимость и точность Монте-Карло', '',
        f"Проверены все 17 физических параметров и оба отношения: max R̂ = **{d['max_rhat']:.6f}**; "
        f"минимальные bulk-ESS / tail-ESS среди них — **{d['min_bulk_ess_all']:.0f} / {d['min_tail_ess_all']:.0f}**. "
        f"Для четырёх концентраций и двух отношений: **{d['min_bulk_ess_published']:.0f} / {d['min_tail_ess_published']:.0f}**. "
        'Увеличение длительности по заданным критериям не потребовалось.', '',
        '| Цепь | Divergences после warmup | Divergences в warmup | E-BFMI | Средняя вероятность принятия | Макс. глубина дерева |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in chain:
        lines.append(f"| {r['chain']} | {r['divergences']} | {r['warmup_divergences']} | {float(r['bfmi']):.3f} | {float(r['mean_acceptance_rate']):.3f} | {r['max_tree_depth']} |")
    lines += ['', 'Во время настройки было 12 divergences; после настройки — 0, достижений предельной глубины — 0. '
        'Warmup сохранён отдельно в NetCDF и исключён из сводных характеристик. Все 20 000 последующих итераций сохранены и использованы; отбраковки нет. '
        'MCSE средних, SD, медиан и обеих границ интервала приведены в CSV. MCSE описывает точность MCMC, а posterior SD — неопределённость параметра.',
        '', '## Концентрации', '', 'Все значения в **mBq m⁻³**. Интервалы posterior — равновероятностные по хвостам 95% **credible intervals**. '
        'σ ММП — исключительно статистическая; ошибки эффективностей/объёма не добавлялись и ковариация на приведённый девианс не умножалась.', '',
        '| Изотоп | Свободный ММП | σ ММП | MAP | Posterior mean | Median | SD | 95% credible interval | MCSE mean |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in conc:
        f=lambda k: float(r[k])
        lines.append(f"| {r['quantity']} | {f('free_ml'):.6g} | {f('free_ml_statistical_sd'):.6g} | {f('saved_map'):.6g} | {f('mean'):.6g} | {f('median'):.6g} | {f('sd'):.6g} | [{f('q025'):.6g}, {f('q975'):.6g}] | {f('mcse_mean'):.3g} |")
    lines += ['', '## Отношения из совместных выборок', '',
        r'Каждая реализация даёт $r_1=a_V(^{131m}\mathrm{Xe})/a_V(^{133}\mathrm{Xe})$ и $r_2=a_V(^{133m}\mathrm{Xe})/a_V(^{133}\mathrm{Xe})$. '
        'Числитель и знаменатель взяты из одной итерации одной цепи; корреляции сохранены.', '',
        r'Для ММП использован дельта-метод: $K_r=G K_z^{\mathrm{stat}}G^T$, где $K_z$ — полная ковариация 17 совместных параметров. '
        'Для одного отношения градиент имеет компоненты `1/a_den` по числителю и `−a_num/a_den²` по знаменателю; перекрёстный ковариационный член сохранён.', '',
        '| Отношение | ММП | σ delta | Posterior mean | Median | SD | 95% credible interval | MCSE mean |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in ratios:
        f=lambda k: float(r[k])
        lines.append(f"| {r['quantity']} | {f('free_ml'):.7g} | {f('free_ml_statistical_sd'):.7g} | {f('mean'):.7g} | {f('median'):.7g} | {f('sd'):.7g} | [{f('q025'):.7g}, {f('q975'):.7g}] | {f('mcse_mean'):.3g} |")
    lines += ['', '## Что видно при сравнении', '',
        '- Для Xe133 и Xe131m центры и ширины близки к исходному ММП. Отношение Xe131m/Xe133 также практически совпадает с нормальным приближением: SD различаются примерно на 0.1%, что меньше точности оценки этой SD по MCMC.',
        '- Для Xe133m и Xe133m/Xe133 posterior имеет влияние границы нуля: SD примерно на 6.1% меньше, нижние границы credible intervals выше нормальных границ ММП. '
        'Это различие распределений при выбранном prior, а не доказательство большей точности или лучшего покрытия.',
        '- Для Xe135 свободный ММП отрицательный, а MAP практически нулевой. Posterior положителен и асимметричен: среднее около 0.208 mBq m⁻³, медиана около 0.155. '
        'Это ожидаемое следствие ограничения и интегрирования по совместным неизвестным; сравниваются разные статистические конструкции.',
        '- Неотрицательность задана для всех 17 координат, в том числе памяти и фона; поэтому Bayesian результат не следует понимать как простое обрезание отдельных одномерных нормальных кривых ММП.',
        '', 'Корреляции (posterior / исходный статистический ММП): ' + '; '.join(
            f"{p['isotopes'][0]}–{p['isotopes'][1]}: {p['posterior_correlation']:.4f} / {p['ml_stat_correlation']:.4f}" for p in plot['joint']['pairs']) + '.',
        '', 'Положительная нижняя граница равновероятностного credible interval **не используется как обнаружение**: '
        'при непрерывном prior на положительной полуоси вероятность строго положительного параметра равна единице по построению. '
        'Ни тест обнаружения, ни частотное покрытие этих интервалов в данной работе не оценивались. '
        'Выводы условны относительно исходной модели и фиксированных калибровок.',
        '', '## Рисунки и файлы', '',
        '![Joint posterior concentrations](bayesian_joint_concentrations.png)', '',
        'Облака — все 20 000 posterior draws. Эллипсы — исходное **нормальное приближение ММП**, с χ²₂,₀․₉₅ = 5.991464547; '
        'они не являются апостериорными областями. Отмечены свободный ММП и сохранённая MAP.', '',
        '![Posterior ratios](bayesian_ratios.png)', '',
        'Гистограммы нормированы по всем выборкам. Линиями показаны медианы и равновероятностные credible intervals. '
        'Нормальное приближение дельта-метода не обрезано при нуле и не перенормировано; '
        f"для r₂ его вероятность отрицательного хвоста {100*plot['ratios']['ratios'][1]['normal_probability_below_zero']:.3f}%.", '',
        '- `posterior.npz`: `z[chain,draw,parameter]` = 4×5000×17, четыре концентрации в mBq/m³, два отношения, названия, единицы и sampler statistics. '
        'Нативные первые пять координат `z` — Bq/m³; `gamma` — опорные отсчёты GASBK; `zeta` — средние DETBK.',
        '- `posterior.nc`: те же posterior draws плюс warmup отдельно и полная диагностика NUTS. '
        '`parameters.json` описывает порядок и единицы.',
        '- `concentrations.csv`, `ratios.csv`, `posterior_summary_all.csv`: среднее, медиана, SD, квантили и сравнение с исходным ММП.',
        '- `diagnostics.csv`, `chain_diagnostics.csv`, `diagnostics.json`: R̂, ESS, MCSE и диагностика NUTS.',
        '- `original_mle.npz`, `original_mle_parameters.csv`, `original_mle_covariance_*.csv`: исходные оценки, MAP и статистические ковариации; '
        '`posterior_covariances.npz` и соответствующие CSV — эмпирические posterior ковариации (делитель N−1) и корреляции.',
        '- `source/`: точные файлы из исходного архива — PHD, подготовленные матрицы, два сохранённых ММП-результата и снимок ядра. '
        'В исходных JSON присутствуют прежние результаты Lc/MDC, но в этом исследовании они не пересчитывались и не использовались.',
        '- `scenario.json`, `settings.json`, `initial_states.npz`, `versions.json`, `source_manifest.json`: параметры данных, prior, seeds, версии и контрольные суммы.',
        '- `bayesian_joint_concentrations.{pdf,png}`, `bayesian_ratios.{pdf,png}`: два векторных PDF и PNG 300 dpi. '
        '`plot_bayesian.py` меняет оформление без MCMC; `figures.tex` содержит английские подписи и вставку PDF.',
        '- `sample_posterior.py`, `summarize_posterior.py`, `build_report.py`, `verify_bayesian.py`, `package.py`: воспроизведение и проверка. '
        '`verification.json`, `checksums.json` — результаты проверки и SHA-256.',
        '', '## Воспроизведение', '',
        f"Исходное ядро: `{source['production_commit']}`. Python {s['python']}; PyMC {s['software']['pymc']}; "
        f"ArviZ {s['software']['arviz']}; NumPy {s['software']['numpy']}; SciPy {s['software']['scipy']}. Полные основные версии закреплены в `requirements.txt`.",
        '', '```bash', 'python -m pip install -r requirements.txt', '# Only change figures; no sampling:',
        'python plot_bayesian.py', '# Recompute summaries / checks from archived draws:',
        'python summarize_posterior.py', 'python build_report.py', 'python verify_bayesian.py',
        '# Re-run MCMC only if needed (overwrites posterior files):', 'python sample_posterior.py',
        'python summarize_posterior.py', 'python plot_bayesian.py', 'python build_report.py', 'python verify_bayesian.py', '```',
        '', 'Seed и начальные состояния воспроизводимы; между платформами и версиями компилятора возможны численные различия траекторий MCMC. '
        'Сохранённые выборки позволяют воспроизводить все приведённые таблицы и рисунки без повторного расчёта цепей.',
        '', '## English summary for the manuscript', '',
        'The original 21-count joint Poisson model was analysed with a constant prior on the nonnegative orthant of all 17 physical parameters. '
        'Four NUTS chains retained 20,000 joint draws after adaptation, with no post-warmup divergences and satisfactory rank-normalized R-hat and effective sample sizes. '
        'Posterior uncertainty for 133Xe, 131mXe and the 131mXe/133Xe ratio was close to the original statistical ML normal approximation. '
        'For 133mXe and its ratio to 133Xe, the posterior standard deviations were approximately 6% smaller and the lower equal-tailed credible limits differed more visibly. '
        'The weak 135Xe component showed a marked boundary effect. These are model-conditional differences, not evidence of superior coverage or detection performance. '
        'All calibration quantities were fixed, and the ML comparison used statistical covariance only.', '']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report, scenario, units and version metadata written.')


if __name__ == '__main__':
    main()
