"""Tables and plot from the saved sensitivity results; no fits."""
from pathlib import Path
import csv
import json
import os
import tempfile

ROOT=Path(__file__).resolve().parent
CACHE=Path(tempfile.gettempdir())/'xemat-roi7-plot-cache'
CACHE.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(CACHE/'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(CACHE))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    summary=rows('summary.csv');activities=rows('concentrations.csv');roi=rows('roi_diagnostics.csv')
    by_key={(r['variant'],r['method_key']):r for r in summary}
    variants=['original_034','test_035','test_036']
    values=[.34,.35,.36]
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':11,
                         'pdf.fonttype':42,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8))
    fig.subplots_adjust(left=.075,right=.98,bottom=.24,top=.78,wspace=.30)
    for key,label,color,marker in [('wls','LS','#1f77b4','o'),('poisson','Poisson ML','#ff7f0e','s')]:
        selected=[by_key[v,key] for v in variants]
        axes[0].plot(values,[float(r['reduced_statistic']) for r in selected],color=color,marker=marker,label=label,lw=1.4)
        axes[1].plot(values,[float(r['sample_ROI7_residual']) for r in selected],color=color,marker=marker,label=label,lw=1.4)
    axes[0].axhline(1,color='.5',lw=1,ls='--');axes[1].axhline(0,color='.5',lw=1,ls='--')
    axes[0].set_ylabel(r'Reduced criterion $\chi^2/\nu$ or $D/\nu$')
    axes[1].set_ylabel(r'SAMPLE ROI 7 residual $x-\mu$ [counts]')
    axes[0].set_title('(a) Full joint fit');axes[1].set_title('(b) ROI 7 after refitting')
    for ax in axes:
        ax.set_xticks(values);ax.set_xticklabels(['0.34','0.35','0.36'])
        ax.set_xlabel(r'Fresh-sample response $R_s[\mathrm{ROI}\ 7,\,^{133}\mathrm{Xe}]$')
        ax.grid(alpha=.2)
    fig.suptitle('Sensitivity to one fresh-sample response coefficient',y=.97,fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    fig.text(.5,.055,'Same August observations; all parameters refitted; memory-response matrices unchanged.',ha='center',fontsize=10)
    fig.savefig(ROOT/'roi7_sensitivity.pdf');fig.savefig(ROOT/'roi7_sensitivity.png',dpi=300);plt.close(fig)
    lines=['# Чувствительность августовского расчёта к Rs[ROI 7, Xe133]', '',
        'Проверены ровно два изменения исходного коэффициента **0.34 → 0.35** и **0.34 → 0.36** '
        '(+0.01 и +0.02, то есть +2.94% и +5.88%).', '',
        '**Матрица памяти Rg не менялась.** Неизменны также обе её временные формы R_g1/R_g2, '
        'перенос памяти F, времена, объём, эффективности, branching и исходные наблюдения SAMPLE/GASBK/DETBK. '
        'Параметры памяти и DETBK переоцениваются совместно с пробой обычным способом.', '',
        'В каждом новом варианте выполнен обычный расчёт всеми четырьмя методами. Все восемь новых подгонок сошлись; '
        'исходные четыре результата при 0.34 взяты из сохранённого расчёта. '
        'Рабочий constants.py не редактировался: для сравнения используется временная копия реестра матриц в отдельном процессе.', '',
        '![Sensitivity comparison](roi7_sensitivity.png)', '',
        '## Девианс свободного пуассоновского ММП', '',
        'В SAMPLE ROI 7 наблюдалось **28048 отсчётов** во всех вариантах. '
        'Ниже указана модель после полной повторной подгонки, а не результат подстановки старых активностей в новую матрицу.', '',
        '| Коэффициент Rs | μ ROI 7 | x − μ | D ROI 7 | Общий D | D/ν, ν=4 |',
        '|---:|---:|---:|---:|---:|---:|']
    for v,value in zip(variants,values):
        r=by_key[v,'poisson']
        lines.append(f"| {value:.2f} | {float(r['sample_ROI7_mu']):.3f} | {float(r['sample_ROI7_residual']):+.3f} | {float(r['sample_ROI7_D']):.6f} | {float(r['statistic']):.6f} | {float(r['reduced_statistic']):.6f} |")
    lines += ['', 'Оба изменения резко уменьшают невязку. При 0.36 знак остатка ROI 7 уже меняется: '
        'модель предсказывает больше отсчётов, чем наблюдалось. Общий девианс у 0.36 немного ниже, '
        'чем у 0.35, но это сравнение трёх фиксированных вариантов, а не поиск оптимального коэффициента.', '',
        '## Оценки четырёх концентраций', '',
        'Свободный пуассоновский ММП, **mBq/m³**. Отрицательные оценки сохранены.', '',
        '| Изотоп | Rs=0.34 | Rs=0.35 | Rs=0.36 |', '|---|---:|---:|---:|']
    for isotope in ['Xe135','Xe133','Xe131m','Xe133m']:
        vals=[next(float(r['concentration_mBq_m3']) for r in activities if r['variant']==v and r['method_key']=='poisson' and r['isotope']==isotope) for v in variants]
        lines.append('| '+isotope+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines += ['', 'При увеличении этой ячейки модель объясняет ROI 7 меньшим общим количеством свежего Xe133. '
        'Его вклад в других ROI также уменьшается, и совместная подгонка перераспределяет отсчёты между примесями. '
        'Поэтому меняется не только остаток ROI 7: оценки Xe131m и Xe133m заметно растут.', '',
        '## Сопоставление всех методов', '',
        '| Rs | LS χ²/ν | NNLS χ²/ν | Poisson ML D/ν | Nonnegative Poisson ML D/ν |',
        '|---:|---:|---:|---:|---:|']
    for v,value in zip(variants,values):
        lines.append(f'| {value:.2f} | '+' | '.join(f"{float(by_key[v,m]['reduced_statistic']):.6f}" for m in ['wls','nnls','poisson','poisson_nonnegative'])+' |')
    lines += ['', 'Для ограниченных методов ν=4 — номинальное число степеней свободы. '
        'В каждом фиксированном варианте коэффициент Rs не включён в вектор подгоняемых параметров. '
        'Выбор калибровки по минимуму критерия на этой же пробе потребовал бы отдельного учёта такого подбора.', '',
        '## Где остаются невязки', '',
        'Вклады в девианс свободного ММП после подгонки:', '',
        '| Область | Rs=0.34 | Rs=0.35 | Rs=0.36 |', '|---|---:|---:|---:|']
    for number in [3,7,8]:
        vals=[next(float(r['deviance_contribution']) for r in roi if r['variant']==v and r['method_key']=='poisson' and r['block']=='SAMPLE' and int(r['roi'])==number) for v in variants]
        lines.append(f'| SAMPLE ROI {number} | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines += ['', 'После изменения основной оставшийся вклад даёт SAMPLE ROI 8. '
        'Улучшение по одному комплекту показывает чувствительность результата к Rs, '
        'но не устанавливает новое калибровочное значение: его следует проверять на независимых измерениях.', '',
        '## Условия и точное изменение модели', '',
        'Детектор 221; SAMPLE20250817072414190.PHD, GASBK20250816152938777.PHD и прежний '
        'DETBK20250610213044796.PHD. Фиксированные ROI [1,2,3,5,6,7,8]. '
        'Эффективности SAMPLE/GASBK одинаковы: Xe135 0.241169, Xe133 0.367784, '
        'Xe131m 0.233937, Xe133m 0.233937; branching 0.9020, 0.3700, 0.5400, 0.5577 соответственно. '
        'Эффективности не содержат branching. Радон — прежняя модель ε=0.3, BR=0.37.', '',
        'Заменена только базовая ячейка **Rs_221[5,2]** (индексы Python начинаются с нуля). '
        'Пересборка R_c=R_c0@C_s_c сохраняет врастание дочернего Xe133: поэтому в полной M '
        'меняются M[5,2] на δ и M[5,4] на δ×0.04051830888018503. '
        'Изменение второго элемента необходимо для согласованного столбца Xe133m. '
        'Все остальные элементы M и все матрицы памяти проверены на совпадение.', '',
        '## Сохранённые материалы', '',
        '- `summary.csv` — критерии и остаток ROI 7 для всех вариантов и методов.',
        '- `roi_diagnostics.csv` — все 21 наблюдение, μ, остатки и вклады девианса для каждого расчёта.',
        '- `concentrations.csv` — концентрации, σ, собственные LC/MDC и обнаружения каждого варианта.',
        '- `original_034/`, `test_035/`, `test_036/` — полные сохранённые результаты; для новых вариантов также обычный TXT-экспорт.',
        '- `baseline/`, `inputs/`, `kernel_snapshot/`, `scenario.json` — исходные данные и неизменённое ядро с происхождением.',
        '- `run_sensitivity.py` — повторение ровно двух новых вариантов; `build_report.py` — оформление сохранённых таблиц без подгонки.',
        '- `construction_checks.json`, `verification.json` — проверки неизменности памяти и согласованности результатов.', '',
        'Версия ядра: `69ff5a62ccaaddeb6372f7f003109ce5ff6cf333`. '
        'Монте-Карло, автоматический поиск коэффициента и постоянная замена калибровки не выполнялись.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
