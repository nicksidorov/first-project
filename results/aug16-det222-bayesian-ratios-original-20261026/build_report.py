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
    assert source['detector'] == 222 and d['status'] == 'passed'
    conc, ratios = read_csv('concentrations.csv'), read_csv('ratios.csv')
    chain = read_csv('chain_diagnostics.csv')
    plot = jread('plot_metadata.json')
    manifest = jread('source_manifest.json')
    a = np.load(ROOT/'source/measurement_arrays.npz')
    metadata = source['metadata']
    scenario = dict(analysis_date='2026-10-26', analysis='Bayesian posterior of original detector-222 August real sample',
        source_kind=manifest['source_kind'], calibration_config=jread('calibration_config.json'),
        calibration_config_sha256=manifest['calibration_config_sha256'],
        original_production_commit=source['production_commit'], input_files=source['input_files'],
        original_scenario=source, original_counts=a['x'].astype(int).tolist(),
        physical_parameter_names=s['parameter_names'], physical_parameter_units=s['parameter_units'],
        numerical_scaling=s['scale'], calibration_uncertainties_included=False,
        model_equation='mu = S z = M Q^{-1} z, Q=diag(T_V,I_12), b=0',
        priors='Constant on the nonnegative physical orthant; independent improper HalfFlat factors',
        random_observation_generation=False, LC_LD_MDC_recomputed=False,
        baseline_fits_new_for_this_measurement=True, additional_MAP_optimization_for_MCMC=False, response_matrices_modified=False)
    (ROOT/'scenario.json').write_text(json.dumps(scenario,indent=2,ensure_ascii=False)+'\n')
    params = [dict(index=i,name=n,native_units=u,summary_units=('mBq/m3 (legacy auxiliary)' if i==0 else 'mBq/m3') if i<5 else u)
              for i,(n,u) in enumerate(zip(s['parameter_names'],s['parameter_units']))]
    (ROOT/'parameters.json').write_text(json.dumps(dict(axes=['chain','draw','parameter'],shape=s['posterior_shape'],
        parameters=params,concentration_order=s['concentration_order'],concentration_units='mBq/m3',
        ratio_order=s['ratio_order'],ratio_units='dimensionless',
        note='First radon coordinate is the original legacy auxiliary parameter, not an atmospheric radon concentration'),indent=2)+'\n')
    versions = dict(python=s['python'],packages=s['software'],platform=s['platform'])
    (ROOT/'versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    lines = [
        '# XeMat: байесовские концентрации и отношения — детектор 222, проба 16 августа 2025 года',
        '',
        'Расчёт от 26 октября 2026 года по трём PHD-файлам, предоставленным пользователем. '
        'Использованы исходные `Rs_222` и `Rg_222` из `constants.py`, фиксированные ROI и новые фиксированные эффективности пользователя. '
        'Кинетика пересчитана по временам этого комплекта. Сравнение относится к одной модели с фиксированными калибровками.',
        '',
        '## Данные и модель',
        '',
        '| Блок | Исходный файл | Начало измерения из PHD | Реальное время, с | Живое время, с |',
        '|---|---|---|---:|---:|',
    ]
    for role, key, temporal in [('SAMPLE','sample_file','sample_temporal'),('GASBK','background_file','background_temporal'),('DETBK','detector_file','detector_temporal')]:
        t=metadata[temporal]
        lines.append(f"| {role} | {source['input_files'][key]['file']} | {t['start_time']} | {t['real_time']} | {t['live_time']} |")
    lines += ['', 'Все файлы относятся к детектору **222**. Фиксированные ROI в порядке **1, 2, 3, 5, 6, 7, 8**; '
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
        '', 'Для SAMPLE и GASBK применён один и тот же пользовательский массив ε = [0.251173, 0.3850, 0.244096, 0.244096], '
        'в порядке Xe135, Xe133, Xe131m, Xe133m. Он сохранён в `calibration_config.json` и передан в локальную копию реестра эффективностей '
        'при сборке матриц. Файлы ядра и матрицы из `constants.py` не изменялись. Для ROI 7 ← Xe133: Rs = 0.347, Rg = 0.18.',
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
        '', 'Для этого комплекта один раз рассчитаны свободный и неотрицательный пуассоновский ММП штатными решателями XeMat. '
        'Опорная MAP — это сохранённое неотрицательное решение в физических координатах; отдельная дополнительная оптимизация для MCMC не выполнялась. Начала четырёх цепей различны и строго положительны, сохранены в `initial_states.npz`.',
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
    lines += ['', f"Во время настройки было {sum(int(r['warmup_divergences']) for r in chain)} divergences; после настройки — {d['divergences']}, достижений предельной глубины — {d['max_treedepth_hits']}. "
        f"Warmup сохранён отдельно в NetCDF и исключён из сводных характеристик. Все {d['posterior_draws']} последующих итераций сохранены и использованы; отбраковки нет. "
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
    # These comparisons are calculated from THIS detector-222 posterior.
    by_conc = {row['quantity']: row for row in conc}
    sd_changes = {row['quantity']: 100*(float(row['sd_ratio_posterior_to_ml'])-1) for row in conc+ratios}
    ratio_sd_mcse_percent = [100*float(row['mcse_sd'])/float(row['sd']) for row in ratios]
    lines += ['', '## Что видно при сравнении', '',
        '- Для Xe133 posterior близок к нормальному приближению ММП по центру и ширине. '
        f"Среднее {float(by_conc['Xe133']['mean']):.6g} против ММП {float(by_conc['Xe133']['free_ml']):.6g} mBq m⁻³; SD отличается на {sd_changes['Xe133']:+.2f}%.",
        f"- Для Xe131m и Xe133m posterior SD отличаются от статистических σ ММП на {sd_changes['Xe131m']:+.2f}% и {sd_changes['Xe133m']:+.2f}% соответственно. "
        'Различия ширины небольшие; сдвиги средних и медиан приведены в таблицах вместе с MCSE. '
        'Это свойства распределений в данной модели, а не доказательство большей точности или лучшего покрытия.',
        f"- Для отношений Xe131m/Xe133 и Xe133m/Xe133 SD отличаются от дельта-метода на {sd_changes['Xe131m/Xe133']:+.2f}% и {sd_changes['Xe133m/Xe133']:+.2f}%. "
        f"Относительная MCSE этих posterior SD составляет {ratio_sd_mcse_percent[0]:.2f}% и {ratio_sd_mcse_percent[1]:.2f}%; небольшие расхождения ширины следует оценивать с учётом этой точности. "
        'Обе формы распределений в основном близки к нормальным приближениям, при сохранённой неотрицательной области posterior.',
        f"- Для Xe135 свободный ММП положительный ({float(by_conc['Xe135']['free_ml']):.6g} mBq m⁻³), а совместная MAP практически нулевая. "
        f"Posterior положителен и асимметричен: среднее {float(by_conc['Xe135']['mean']):.6g}, медиана {float(by_conc['Xe135']['median']):.6g}, SD {float(by_conc['Xe135']['sd']):.6g} mBq m⁻³. "
        'Неотрицательность накладывается одновременно на свежую пробу, память и фон; поэтому MAP не получается простым занулением отрицательных координат свободной оценки.',
        '- Все 17 координат интегрируются совместно. Posterior нельзя понимать как независимое обрезание одномерных нормальных кривых ММП.',
        '', 'Корреляции (posterior / исходный статистический ММП): ' + '; '.join(
            f"{p['isotopes'][0]}–{p['isotopes'][1]}: {p['posterior_correlation']:.4f} / {p['ml_stat_correlation']:.4f}" for p in plot['joint']['pairs']) + '.',
        '', 'Положительная нижняя граница равновероятностного credible interval **не используется как обнаружение**: '
        'при непрерывном prior на положительной полуоси вероятность строго положительного параметра равна единице по построению. '
        'Ни тест обнаружения, ни частотное покрытие этих интервалов в данной работе не оценивались. '
        'Выводы условны относительно исходной модели и фиксированных калибровок.',
        '', '## Рисунки и файлы', '',
        '![Joint posterior concentrations](bayesian_joint_concentrations.png)', '',
        f"Облака — все {d['posterior_draws']} posterior draws. "
        'Эллипсы — исходное **нормальное приближение ММП**, с χ²₂,₀․₉₅ = 5.991464547; '
        'они не являются апостериорными областями. Отмечены свободный ММП и сохранённая MAP.', '',
        '![Posterior ratios](bayesian_ratios.png)', '',
        'Гистограммы нормированы по всем выборкам. Линиями показаны медианы и равновероятностные credible intervals. '
        'Нормальное приближение дельта-метода не обрезано при нуле и не перенормировано; '
        f"для r₂ его вероятность отрицательного хвоста {100*plot['ratios']['ratios'][1]['normal_probability_below_zero']:.3f}%.", '',
        f"- `posterior.npz`: `z[chain,draw,parameter]` = {s['posterior_shape'][0]}×{s['posterior_shape'][1]}×{s['posterior_shape'][2]}, "
        'четыре концентрации в mBq/m³, два отношения, названия, единицы и sampler statistics. '
        'Нативные первые пять координат `z` — Bq/m³; `gamma` — опорные отсчёты GASBK; `zeta` — средние DETBK.',
        '- `posterior.nc`: те же posterior draws плюс warmup отдельно и полная диагностика NUTS. '
        '`parameters.json` описывает порядок и единицы.',
        '- `concentrations.csv`, `ratios.csv`, `posterior_summary_all.csv`: среднее, медиана, SD, квантили и сравнение с исходным ММП.',
        '- `diagnostics.csv`, `chain_diagnostics.csv`, `diagnostics.json`: R̂, ESS, MCSE и диагностика NUTS.',
        '- `original_mle.npz`, `original_mle_parameters.csv`, `original_mle_covariance_*.csv`: исходные оценки, MAP и статистические ковариации; '
        '`posterior_covariances.npz` и соответствующие CSV — эмпирические posterior ковариации (делитель N−1) и корреляции.',
        '- `source/`: точные пользовательские PHD, подготовленные матрицы, два сохранённых ММП-результата для этого комплекта и неизменённый снимок ядра. '
        'Lc, Ld и MDC отсутствуют в этом расчёте; поле `detection` в результатах равно `null`.',
        '- `scenario.json`, `settings.json`, `initial_states.npz`, `versions.json`, `source_manifest.json`: параметры данных, prior, seeds, версии и контрольные суммы.',
        '- `bayesian_joint_concentrations.{pdf,png}`, `bayesian_ratios.{pdf,png}`: два векторных PDF и PNG 300 dpi. '
        '`plot_bayesian.py` меняет оформление без MCMC; `figures.tex` содержит английские подписи и вставку PDF.',
        '- `calibration_config.json` фиксирует пользовательские ε и branching; `prepare_model.py` воспроизводит матрицы и две исходные ММП-оценки без расчёта пределов обнаружения.',
        '- `sample_posterior.py`, `summarize_posterior.py`, `build_report.py`, `verify_bayesian.py`, `package.py`: воспроизведение и проверка. '
        '`verification.json`, `checksums.json` — результаты проверки и SHA-256.',
        '', '## Воспроизведение', '',
        f"Исходное ядро: `{source['production_commit']}`. Python {s['python']}; PyMC {s['software']['pymc']}; "
        f"ArviZ {s['software']['arviz']}; NumPy {s['software']['numpy']}; SciPy {s['software']['scipy']}. Полные основные версии закреплены в `requirements.txt`.",
        '', '```bash', 'python -m pip install -r requirements.txt', '# Only change figures; no sampling:',
        'python plot_bayesian.py', '# Recompute summaries / checks from archived draws:',
        'python summarize_posterior.py', 'python build_report.py', 'python verify_bayesian.py',
        '# Optional: rebuild the original model and two ML fits:', 'python prepare_model.py',
        '# Re-run MCMC only if needed (overwrites posterior files):', 'python sample_posterior.py',
        'python summarize_posterior.py', 'python plot_bayesian.py', 'python build_report.py', 'python verify_bayesian.py', '```',
        '', 'Seed и начальные состояния воспроизводимы; между платформами и версиями компилятора возможны численные различия траекторий MCMC. '
        'Сохранённые выборки позволяют воспроизводить все приведённые таблицы и рисунки без повторного расчёта цепей.',
        '', '## English summary for the manuscript', '',
        'The original detector-222 response matrices and the user-supplied fixed efficiencies were used for a joint Poisson analysis of all 21 SAMPLE, GASBK and DETBK counts. '
        'A constant prior was assigned on the nonnegative orthant of all 17 physical parameters. '
        f"Four NUTS chains retained {d['posterior_draws']:,} joint draws after adaptation, with {d['divergences']} post-warmup divergences, maximum rank-normalized R-hat {d['max_rhat']:.6f}, and satisfactory effective sample sizes. "
        'Posterior uncertainty for 133Xe, 131mXe, 133mXe and both ratios was close to the original statistical ML normal approximation. '
        f"For the 131mXe/133Xe and 133mXe/133Xe ratios, the posterior standard deviations differed from the delta-method values by {sd_changes['Xe131m/Xe133']:+.2f}% and {sd_changes['Xe133m/Xe133']:+.2f}%, respectively. "
        f"The corresponding relative Monte Carlo standard errors of posterior SD were {ratio_sd_mcse_percent[0]:.2f}% and {ratio_sd_mcse_percent[1]:.2f}%, so these small width differences should not be overinterpreted. "
        'The weak 135Xe component showed a marked boundary effect: the joint nonnegative ML/MAP estimate was essentially zero despite a positive free ML estimate. '
        'The posterior was obtained by joint inference over the signal, memory and detector-background parameters, rather than by truncating separate Gaussian marginals. '
        'These model-conditional results do not establish superior frequentist coverage or detection performance. '
        'All calibration quantities were fixed, and the ML comparison used statistical covariance only.', '']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report, scenario, units and version metadata written.')


if __name__ == '__main__':
    main()
