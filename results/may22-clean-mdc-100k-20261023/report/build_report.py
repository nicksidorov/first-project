"""Render the audited, single-May-configuration 100k XeMat study.

This presentation layer reads completed results only. It never generates counts,
fits spectra or changes a threshold. See REPORT_SCHEMA.md for the input contract.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'xemat-100k-figures'))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = {'wls': 'МНК', 'poisson': 'ММП'}
METHOD_COLORS = {'wls': '#147b9d', 'poisson': '#c86117'}
HIST_COLORS = {'null': '#7954a1', 'signal': '#008c88'}
ISOTOPES = {1: 'Xe135', 2: 'Xe133', 3: 'Xe131m', 4: 'Xe133m'}
TITLE = 'XeMat: чистая проба и проверка Lc на 100 000 реализаций'


def num(value, digits=4):
    if value is None:
        return 'не определено'
    if not np.isfinite(value):
        return '∞' if value > 0 else '−∞'
    return f'{value:.{digits}f}'.replace('.', ',')


def integer(value):
    return f'{int(value):,}'.replace(',', '\u202f')


def ci_text(interval, scale=1, digits=4):
    if interval is None:
        return 'не определён'
    return '–'.join(num(None if v is None else v*scale, digits) for v in interval)


def rate_text(record):
    return f"{num(100*record['rate'], 3)}% [{ci_text(record['ci95'], 100, 3)}%]"


def hist_key(curve, scenario):
    return f"may22_{curve['family']}_i{curve['target']}_{scenario}"


def read_checked(root):
    data = json.loads((root/'report_data.json').read_text(encoding='utf-8'))
    if data.get('verification', {}).get('status') != 'passed':
        raise ValueError('The report requires a completed independent audit')
    if data.get('histogram_verification', {}).get('status') != 'passed':
        raise ValueError('The report requires a completed independent histogram audit')
    if data.get('dataset') != 'may22':
        raise ValueError('This report covers the single may22 configuration')
    if any(data['n_by_stage'][key] != 100000 for key in ('calibration', 'validation')):
        raise ValueError('Expected 100000 calibration and 100000 independent validation observations')
    if data['n_signal_per_point'] != 20000:
        raise ValueError('Expected 20000 observations per signal activity')
    curves = sorted(data['curves'], key=lambda c: c['target'])
    if [(c['target'], c['isotope']) for c in curves] != list(ISOTOPES.items()):
        raise ValueError('Expected the four Xe isotope curves')
    data['curves'] = curves
    for c in curves:
        if c['dataset'] != 'may22':
            raise ValueError('Mixed datasets in the one-configuration report')
        for method in METHODS:
            m = c['methods'][method]
            if m['calibration']['n'] != 100000:
                raise ValueError('Calibration sample count mismatch')
            for rule in ('standard', 'calibrated'):
                a = m['validation'][rule]
                if a['n'] != 100000 or a['n_valid'] + a['missing'] != a['n']:
                    raise ValueError('Validation accounting mismatch')
                if not np.isclose(a['rate'], a['k']/a['n_valid']):
                    raise ValueError('Probability and counts disagree')
            if not m['formal_points']:
                raise ValueError('Power points are not yet available')
    h = data['histograms']
    digest = hashlib.sha256((root/'histogram_values.npz').read_bytes()).hexdigest()
    if digest != h['values_sha256']:
        raise ValueError('Histogram values checksum mismatch')
    if digest != data['histogram_verification']['histogram_values_sha256']:
        raise ValueError('Histogram values differ from the independently audited archive')
    expected = {(f, i) for f in METHODS for i in ISOTOPES}
    if {(c['family'], c['target']) for c in h['curves']} != expected or len(h['curves']) != 8:
        raise ValueError('Expected eight method/isotope histogram pairs')
    arrays = {}
    with np.load(root/'histogram_values.npz', allow_pickle=False) as archive:
        expected_keys = {hist_key(c, s) for c in h['curves'] for s in ('null', 'signal')}
        if set(archive.files) != expected_keys:
            raise ValueError('Unexpected histogram array set')
        for c in h['curves']:
            if c['dataset'] != 'may22' or c['isotope'] != ISOTOPES[c['target']]:
                raise ValueError('Histogram isotope mapping mismatch')
            reference = curves[c['target']-1]
            m = reference['methods'][c['family']]
            for actual, expected_value in [
                (c['nominal_MDC_mBq_m3'], reference['nominal_MDC_mBq_m3']),
                (c['signal_A_mBq_m3'], m['calibrated_ld_mBq_m3']),
                (c['cutoff_c'], m['calibration']['c']),
            ]:
                if not np.isclose(actual, expected_value, rtol=1e-12, atol=0):
                    raise ValueError('Histogram reference differs from the main study')
            for scenario, count in [('null', 100000), ('signal', 20000)]:
                key = hist_key(c, scenario)
                a = np.array(archive[key], dtype=float, copy=True)
                valid = a[np.isfinite(a)]
                if a.shape != (count,) or np.any(np.isinf(a)) or np.any(valid <= 0):
                    raise ValueError('Invalid histogram array: '+key)
                r = c[scenario]
                if (r['n_attempted'], r['n_valid'], r['n_missing']) != (count, len(valid), count-len(valid)):
                    raise ValueError('Histogram count accounting mismatch: '+key)
                q = np.quantile(valid, [.025, .5, .975])
                actual = dict(mean=float(valid.mean()), median=float(q[1]), sd=float(valid.std(ddof=1)),
                              q025=float(q[0]), q975=float(q[2]))
                for name, value in actual.items():
                    if not np.isclose(r[name], value, rtol=1e-10, atol=1e-12):
                        raise ValueError('Histogram statistic mismatch: '+key+'/'+name)
                arrays[key] = a
    return data, arrays


def save(root, fig, name):
    fig.savefig(root/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(root/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def alpha_limits(data):
    endpoints = [5.]
    for c in data['curves']:
        for f in METHODS:
            for rule in ('standard', 'calibrated'):
                endpoints.extend(100*np.asarray(c['methods'][f]['validation'][rule]['ci95']))
    low, high = min(endpoints), max(endpoints)
    margin = max(.15, (high-low)*.15)
    return np.floor((low-margin)*4)/4, np.ceil((high+margin)*4)/4


def alpha_figure(root, data, rule):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True, sharey=True)
    for ax, (f, label) in zip(axes, METHODS.items()):
        records = [c['methods'][f]['validation'][rule] for c in data['curves']]
        rates = np.array([r['rate']*100 for r in records])
        intervals = np.array([r['ci95'] for r in records])*100
        y = np.arange(4)
        ax.errorbar(rates, y, xerr=[rates-intervals[:,0], intervals[:,1]-rates],
                    fmt='o', color=METHOD_COLORS[f], ms=6, capsize=4, lw=1.8)
        for pos, rate in zip(y, rates):
            ax.annotate(num(rate,3)+'%', (rate,pos), xytext=(0,10),
                        textcoords='offset points', ha='center', fontsize=10)
        ax.axvline(5, color='#555555', ls='--', lw=1.3)
        ax.set_yticks(y, list(ISOTOPES.values()))
        ax.tick_params(labelleft=True)
        ax.set_ylim(3.45, -.65)
        ax.set_xlim(*alpha_limits(data))
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: num(value,2)))
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Ложные обнаружения, %')
        ax.grid(axis='x', alpha=.17)
    title = 'Исходное правило Â > Lc' if rule == 'standard' else 'После подстройки: Â > c · Lc'
    fig.suptitle(title+': проверка уровня 5%', fontsize=16, fontweight='bold', y=.99)
    fig.text(.5,.91,'Пустая проба · условия 22 мая · одна точка на изотоп',ha='center',fontsize=11)
    fig.text(.5,.012,'Точка: доля обнаружений из 100 000 проб. Полоса: 95% ДИ частоты. Пунктир: 5%.',
             ha='center',fontsize=10)
    fig.subplots_adjust(top=.79,bottom=.17,wspace=.25)
    save(root,fig,'alpha_'+rule)


def power_figure(root, data):
    fig, axes = plt.subplots(2,2,figsize=(10.5,8.0),sharex=True,sharey=True)
    all_x=[];all_y=[]
    for ax,c in zip(axes.flat,data['curves']):
        for f,label in METHODS.items():
            m=c['methods'][f];points=sorted(m['formal_points'],key=lambda p:p['factor'])
            x=np.array([p['factor'] for p in points]);all_x.extend(x)
            y=np.array([p['calibrated']['rate']*100 for p in points])
            ci=np.array([p['calibrated']['ci95'] for p in points])*100;all_y.extend(ci[:,0])
            ax.errorbar(x,y,yerr=[y-ci[:,0],ci[:,1]-y],color=METHOD_COLORS[f],
                        fmt='o-',lw=1.5,capsize=3,ms=4,label=label)
            if m['calibrated_ld_mBq_m3'] is not None:
                ax.plot(m['calibrated_ld_mBq_m3']/c['nominal_MDC_mBq_m3'],95,
                        marker='|',color=METHOD_COLORS[f],ms=15,markeredgewidth=2)
            check=next(h for h in data['histograms']['curves']
                       if h['target']==c['target'] and h['family']==f)
            rate=check['signal']['detection']['calibrated']
            hx=check['signal_A_mBq_m3']/c['nominal_MDC_mBq_m3'];hy=100*rate['rate']
            low,high=np.array(rate['ci95'])*100;all_x.append(hx);all_y.append(low)
            ax.errorbar([hx],[hy],yerr=[[hy-low],[high-hy]],color=METHOD_COLORS[f],
                        fmt='D',mfc='white',lw=1.2,capsize=3,ms=5,zorder=6)
        ax.axhline(95,color='#555555',ls=':',lw=1.3)
        ax.axvline(1,color='#999999',ls='--',lw=1)
        ax.set_title(c['isotope'],fontweight='bold');ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:num(value,2)))
    span=max(all_x)-min(all_x)
    for ax in axes.flat:
        ax.set_xlim(min(all_x)-span*.035,max(all_x)+span*.035)
        ax.set_ylim(max(0,np.floor(min(all_y)-1)),100)
    for ax in axes[1]:ax.set_xlabel('Введённая активность / формульный MDC')
    for ax in axes[:,0]:ax.set_ylabel('Обнаружение, %')
    fig.suptitle('Мощность после подстройки Lc · условия 22 мая',fontsize=16,fontweight='bold',y=.99)
    fig.text(.5,.94,'Для каждой точки — '+integer(data['n_signal_per_point'])+' проб с одним введённым изотопом',
             ha='center',fontsize=11)
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([],[],marker='D',mfc='white',color='#444444',ls='None',ms=5))
    labels.append('Независимая проверка')
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.048),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.006,'Кружки: сетка мощности; ромбы: независимая проверка при оценке Ld95.\n'
             'Полосы: 95% ДИ. Отметки на уровне 95%: интерполированные оценки Ld95.',
             ha='center',va='bottom',fontsize=9.2)
    fig.subplots_adjust(top=.85,bottom=.18,hspace=.34,wspace=.16)
    save(root,fig,'power_calibrated')


def histogram_figure(root, data, arrays, method):
    curves=sorted([c for c in data['histograms']['curves'] if c['family']==method],key=lambda c:c['target'])
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.7))
    for ax,c in zip(axes.flat,curves):
        vals={s:arrays[hist_key(c,s)][np.isfinite(arrays[hist_key(c,s)])] for s in ('null','signal')}
        both=np.concatenate(list(vals.values()));lo,hi=both.min(),both.max()
        count=min(65,max(20,len(np.histogram_bin_edges(both,bins='fd'))-1))
        if lo==hi:lo,hi=lo*.98,hi*1.02
        edges=np.linspace(lo,hi,count+1)
        for s,a in vals.items():
            counts=np.histogram(a,bins=edges)[0]
            if counts.sum()!=len(a):raise ValueError('Histogram bins exclude successful observations')
            heights=counts*100/len(a)
            ax.stairs(heights,edges,fill=True,color=HIST_COLORS[s],alpha=.19,linewidth=0)
            ax.stairs(heights,edges,color=HIST_COLORS[s],linewidth=1.5)
        ax.axvline(c['nominal_MDC_mBq_m3'],color='#888888',ls=':',lw=1.7)
        ax.axvline(c['signal_A_mBq_m3'],color='#222222',ls='--',lw=1.5)
        lo=min(lo,c['nominal_MDC_mBq_m3'],c['signal_A_mBq_m3'])
        hi=max(hi,c['nominal_MDC_mBq_m3'],c['signal_A_mBq_m3'])
        margin=max((hi-lo)*.04,1e-10);ax.set_xlim(lo-margin,hi+margin)
        ax.set_ylim(bottom=0);ax.margins(y=.09)
        ax.set_title(c['isotope'],fontsize=13,fontweight='bold',pad=31)
        subtitle=f"A = оценка Ld95 = {num(c['signal_A_mBq_m3'],5)} мБк/м³\n"+\
                 f"N₀ = {integer(c['null']['n_valid'])}; N_Ld = {integer(c['signal']['n_valid'])}"
        ax.text(.5,1.015,subtitle,transform=ax.transAxes,ha='center',va='bottom',fontsize=9)
        ax.set_xlabel('Оценка MDC, мБк/м³');ax.set_ylabel('Доля проб в интервале, %',fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:num(value,3)))
        ax.grid(axis='y',alpha=.17)
    handles=[Patch(facecolor=HIST_COLORS['null'],edgecolor=HIST_COLORS['null'],alpha=.45,label='A = 0: пустая проба'),
             Patch(facecolor=HIST_COLORS['signal'],edgecolor=HIST_COLORS['signal'],alpha=.45,label='A = оценка Ld95: один введённый изотоп'),
             Line2D([],[],color='#888888',ls=':',lw=1.7,label='Формульный MDC при заданном чистом фоне'),
             Line2D([],[],color='#222222',ls='--',lw=1.5,label='Оценка Ld95 после подстройки Lc')]
    fig.suptitle('Гистограммы оценки MDC · 22 мая · '+METHODS[method],fontsize=17,fontweight='bold',y=.99)
    fig.text(.5,.94,'Стандартная итерационная формула по каждой реализации; множитель c в MDC не введён',ha='center',fontsize=10)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.026),ncol=2,frameon=False,fontsize=9.6,columnspacing=1.8)
    fig.text(.5,.006,'Гистограммы нормированы на 100% доступных MDC; интервалы столбцов общие внутри пары.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.09,right=.98,top=.80,bottom=.20,hspace=.62,wspace=.28)
    save(root,fig,'mdc_hist_'+method)


def content(data):
    blocks=[];curves=data['curves'];conditions=data['conditions'];times=conditions['times'];stage=data['n_by_stage']
    def p(text):blocks.append(dict(type='paragraph',text=text))
    def h(text):blocks.append(dict(type='heading',text=text))
    def page():blocks.append(dict(type='page_break'))
    def table(headers,rows,widths=None):
        b=dict(type='table',headers=headers,rows=rows)
        if widths:b['widths']=widths
        blocks.append(b)
    def image(file,caption):blocks.append(dict(type='image',file=file,caption=caption))
    h('1. Условия и объём исследования')
    p('Рассмотрен один набор условий — комплект 22 мая, детектор 221, фиксированные ROI и матрицы R. '
      'В чистой пробе истинные активности всех компонентов пробы и газовой памяти, включая радон, равны нулю. '
      'Остаётся детекторный фон. Для исследования обнаружения вводим по одному свежему изотопу Xe.')
    e=data['efficiencies']
    table(['Изотоп','ε SAMPLE','ε GASBK'],[[iso,num(a,6),num(g,6)] for iso,a,g in
          zip(e['isotope_order'],e['sample'],e['gasbk'])])
    p('Эффективности ε указаны без branching; коэффициенты ветвления учитываются отдельно. '
      'Живое время SAMPLE / GASBK / DETBK: '+ ' / '.join(num(times[k]/3600,2) for k in ('t_sl','t_gl','t_bl'))+
      ' часа. Эффективный объём воздуха: '+num(conditions['effective_air_volume_m3'])+' м³.')
    p('В каждом повторе генерируем все 21 пуассоновский отсчёт SAMPLE/GASBK/DETBK и заново оцениваем '
      'все 17 параметров. Истинное среднее детекторного фона фиксировано для данного сценария; '
      'новые отсчёты DETBK случайны, фон оценивается совместно с пробой и памятью. Систематика не разыгрывается.')
    table(['Этап','Число векторов наблюдений'],[
        ['Подстройка порога',integer(stage['calibration'])],
        ['Независимая проверка пустой пробы',integer(stage['validation'])],
        ['Кривые мощности',integer(stage['signal']+stage.get('signal_extension',0))],
        ['Гистограммы при A = Ld95',integer(stage['hist_signal'])]])
    p('Гистограммы при A = 0 используют проверочные пустые пробы повторно; они не добавляют новых наблюдений. '
      'МНК и ММП в основной серии применяются к одним и тем же отсчётам. Вальд не используется.')
    h('2. Исходное правило обнаружения')
    p('Обнаружение: Â > Lc. Â — свободная оценка активности МНК или ММП. В каждой реализации '
      'выполняется соответствующая неотрицательная вспомогательная подгонка, из её ожидаемого спектра '
      'удаляется оценённый вклад целевого свежего изотопа и рассчитывается Lc. Оценки памяти, '
      'других компонентов и детекторного фона сохраняются. Исходные наблюдаемые отсчёты не вычитаются.')
    p('При A = 0 каждое обнаружение ложное. Цель 5% соответствует примерно 5000 срабатываниям '
      'на 100 000 проверочных проб для одного изотопа. Порог Lc пересчитывается в каждом повторе.')
    page();h('3. Проверка ложноположительных по исходному Lc')
    image('alpha_standard.png','Рисунок 1. Исходное правило Â > Lc на 100 000 независимых проверочных пробах.')
    p('Слева МНК, справа ММП. В каждой панели на каждый изотоп приходится одна точка: процент '
      'ложных обнаружений. Полоса — индивидуальный 95% биномиальный доверительный интервал этой частоты, '
      'то есть погрешность самого Монте-Карло. Пунктир показывает целевые 5%.')
    p('Если весь интервал справа от 5%, наблюдается превышение уровня; если слева — более консервативное '
      'правило. Пересечение с 5% означает совместимость с уровнем с учётом погрешности, но не доказывает '
      'точного равенства. Интервалы не скорректированы на одновременную проверку всех изотопов и методов.')
    table(['Изотоп','МНК: частота [95% ДИ]','ММП: частота [95% ДИ]'],
          [[c['isotope'],*[rate_text(c['methods'][f]['validation']['standard']) for f in METHODS]] for c in curves],
          [.16,.42,.42])
    page();h('4. Подстройка Lc к уровню 5%')
    order=curves[0]['methods']['wls']['calibration']['k_order']
    p('На отдельной серии из 100 000 пустых проб построена статистика R = Â/Lc. Для каждого '
      'метода и изотопа множитель c выбран как '+integer(order)+'-е значение R по возрастанию. '
      'После этого c фиксируется; проверочная серия для выбора c не используется.')
    p('Откалиброванное правило: Â > c · Lc. При c > 1 порог повышается, при c < 1 понижается. '
      'Сам Lc по-прежнему рассчитывается по каждой пробе. Множитель c относится к порогу решения, '
      'а не к готовому MDC; итоговый Ld95 определяется заново по мощности.')
    table(['Изотоп','c для МНК [95% ДИ]','c для ММП [95% ДИ]'],
          [[c['isotope'],*[num(c['methods'][f]['calibration']['c'])+' ['+
             ci_text(c['methods'][f]['calibration']['ci95_c'])+']' for f in METHODS]] for c in curves],[.16,.42,.42])
    h('5. Независимая проверка после подстройки')
    image('alpha_calibrated.png','Рисунок 2. Правило Â > c · Lc. Проверочные отсчёты те же, что на рисунке 1; изменилось правило решения.')
    table(['Изотоп','МНК: частота [95% ДИ]','ММП: частота [95% ДИ]'],
          [[c['isotope'],*[rate_text(c['methods'][f]['validation']['calibrated']) for f in METHODS]] for c in curves],
          [.16,.42,.42])
    records=[c['methods'][f]['validation']['calibrated'] for c in curves for f in METHODS]
    compatible=sum(r['ci95'][0]<=.05<=r['ci95'][1] for r in records)
    p('Проверочные частоты после подстройки лежат в диапазоне '+num(100*min(r['rate'] for r in records),3)+
      '–'+num(100*max(r['rate'] for r in records),3)+'%. Целевые 5% входят в '+str(compatible)+
      ' из 8 индивидуальных 95% интервалов. Конечная калибровочная выборка также вносит '
      'неопределённость; после проверки c не подстраивался повторно.')
    page();h('6. Формульные пределы и Ld95 исходного правила')
    p('Далее все активности указаны в мБк/м³. Формульные Lc и MDC вычислены по полной информации '
      'Фишера для заданного истинного чистого фона, включая статистическую неопределённость оценки DETBK. '
      'Формульный Lc — ориентир; в моделируемых пробах он не используется как постоянный порог.')
    p('Численный предел Ld95 получен по кривой мощности: истинная активность фиксируется, '
      'наблюдения многократно разыгрываются, Lc пересчитывается и измеряется доля обнаружений. '
      'Ld95 соответствует мощности 95%; это оценка с погрешностью Монте-Карло.')
    table(['Изотоп','Формула Lc','Формула MDC','МНК: Ld95\nисходный Lc','ММП: Ld95\nисходный Lc'],
          [[c['isotope'],num(c['nominal_LC_mBq_m3']),num(c['nominal_MDC_mBq_m3']),
            *[num(c['methods'][f]['standard_ld_mBq_m3']) for f in METHODS]] for c in curves],
          [.15,.16,.17,.26,.26])
    table(['Изотоп','МНК: 95% ДИ исходного Ld95','ММП: 95% ДИ исходного Ld95'],
          [[c['isotope'],*[ci_text(c['methods'][f]['standard_ci95_mBq_m3']) for f in METHODS]] for c in curves],
          [.16,.42,.42])
    p('Исходные Ld95 соответствуют ложноположительным частотам рисунка 1. Сравнивать их без '
      'учёта фактического уровня ошибок нельзя: более низкий предел может сопровождаться большим числом ложных обнаружений.')
    h('7. Оценки Ld95 после подстройки Lc')
    table(['Изотоп','Формульный MDC','МНК: Ld95 [95% ДИ]','ММП: Ld95 [95% ДИ]'],
          [[c['isotope'],num(c['nominal_MDC_mBq_m3']),*[num(c['methods'][f]['calibrated_ld_mBq_m3'])+
            ' ['+ci_text(c['methods'][f]['conditional_ci95_mBq_m3'])+']' for f in METHODS]] for c in curves],
          [.14,.18,.34,.34])
    p('Интервалы этой таблицы условны на фактически найденных c. При сигнальном моделировании '
      'c не меняется; Lc пересчитывается по каждому наблюдению. Ld95 не получен умножением формульного MDC на c. '
      'Независимая проверка мощности при этих активностях приведена в разделе 10.1.')
    page();h('8. Кривые мощности после подстройки')
    image('power_calibrated.png','Рисунок 3. Мощность откалиброванного правила; каждый узел рассчитан на '+
          integer(data['n_signal_per_point'])+' наблюдениях. Синий — МНК, оранжевый — ММП.')
    p('По горизонтали — введённая активность, делённая на формульный MDC. Значение 1 соответствует '
      'формуле. По вертикали — доля обнаружений. Полосы отражают 95% интервалы мощности при '
      'фиксированных c. Кружки — исходная сетка мощности; цветные отметки на горизонтали 95% '
      'показывают интерполированные оценки Ld95. Ромбы — независимая проверка мощности при этих '
      'активностях, подробно разобранная в разделе 10.1.')
    h('9. Неопределённость и область применимости')
    p('Использовано '+integer(data['bootstrap_repetitions'])+' bootstrap-повторов. Помимо условных '
      'интервалов рассчитаны интервалы с учётом случайности калибровки c. Они характеризуют оценку '
      'Ld95 для идеального порога c₀ — истинного 95-го квантиля нулевой статистики. '
      'Ошибка интерполяции между узлами мощности и систематика в эти интервалы не включены.')
    table(['Изотоп','МНК: 95% ДИ с учётом c','ММП: 95% ДИ с учётом c'],
          [[c['isotope'],*[ci_text(c['methods'][f]['total_ci95_mBq_m3']) for f in METHODS]] for c in curves],
          [.16,.42,.42])
    censor=[]
    for c in curves:
        for f in METHODS:
            for name,b in c['methods'][f].get('bootstrap',{}).items():
                missing=sum(b.get(k,0) for k in ('n_below_grid','n_above_grid','n_ambiguous'))
                if missing:censor.append(f"{c['isotope']} / {METHODS[f]} / {name}: {missing}")
    if censor:
        p('Bootstrap-пересечения за пределами сетки или неоднозначные пересечения не удалялись: '+
          '; '.join(censor)+'. Неопределённые границы отмечены в таблицах.')
    p('Результаты относятся к чистому фону и указанной временной конфигурации. Перенос подстройки '
      'на смеси с памятью или другой фон отдельно не проверялся. Уровень 5% относится к одному '
      'изотопу. Неопределённость выбранного истинного среднего фона и калибровок не разыгрывалась.')
    failures=len(data['errors']);hist_failures=len(data.get('histogram_errors',[]))
    p('Записей ошибок основной серии: '+integer(failures)+', анализа гистограмм: '+integer(hist_failures)+
      '. Это отдельные журналы, и ошибки переиспользованной проверочной серии могут входить в оба. '
      'Неуспешные реализации не заменялись новыми. '
      'Недоступность ковариации свободного ММП учитывается отдельно и сама по себе не означает '
      'неуспех отдельного расчёта Lc/MDC по неотрицательному пилоту. Тест Вальда не применяется.')
    page();h('10. Гистограммы MDC по отдельным наблюдениям')
    p('Каждая проба даёт собственную оценку MDC стандартной итерационной формулой с α = β = 0,05. '
      'Нулевая модель строится той же процедурой удаления целевого свежего вклада из '
      'неотрицательной подгонки. Множитель c в эту формулу MDC не вводится.')
    p('Рассмотрены A = 0 и A = Ld95. Во втором случае вводится один свежий изотоп, а истинная '
      'память и остальные компоненты равны нулю. Активность Ld95 берётся из новой итоговой таблицы '
      'после подстройки, отдельно для метода и изотопа, и остаётся фиксированной. '
      'Погрешность этого значения при генерации не разыгрывается.')
    p('Для A = 0 использованы все 100 000 независимых проверочных проб. Для A = Ld95 создано '
      'по 20 000 новых наблюдений на каждую пару метод/изотоп. Включены все доступные MDC, '
      'независимо от того, произошло ли обнаружение. Никакого отбора по результату теста нет.')
    p('Фиолетовый цвет — A = 0, бирюзовый — A = Ld95. По горизонтали показан MDC, по вертикали — '
      'доля проб в интервале столбца. Каждая гистограмма нормирована на 100%; интервалы столбцов '
      'общие внутри пары, хвосты не обрезаны. Серая линия — формульный MDC заданного чистого фона, '
      'чёрная — численный Ld95 после подстройки.')
    p('Это распределения случайной оценки MDC по одной пробе, а не повторных оценок Ld95 '
      'по кривой мощности. Центральные 95% значений в таблицах — квантили 2,5% и 97,5%; '
      'они не являются доверительным интервалом истинного Ld95. Отличие от чёрной линии '
      'само по себе не означает смещения оценки того же самого параметра.')
    p('Увеличение числа реализаций Монте-Карло уменьшает погрешность оценённых частот и Ld95. '
      'При прежних временах измерения и остальных условиях внутренний разброс MDC одной пробы '
      'сохраняется; большая выборка позволяет точнее оценить форму этого распределения и его квантили.')
    h('10.1. Независимый контроль мощности при A = Ld95')
    p('На новых наблюдениях, созданных для сигнальных гистограмм, дополнительно проверено '
      'правило Â > c · Lc. Эти пробы не участвовали в определении Ld95. В таблице — '
      'фактическая мощность при фиксированной ранее оценённой активности Ld95 и её '
      '95% биномиальный интервал Клоппера–Пирсона. Ни c, ни Ld95 по этой проверке не менялись.')
    control=[];below_control=[]
    for f in METHODS:
        for c in sorted([h for h in data['histograms']['curves'] if h['family']==f],key=lambda h:h['target']):
            r=c['signal']['detection']['calibrated']
            attempted=r.get('n_attempted',r.get('n'))
            control.append([c['isotope'],METHODS[f],num(c['signal_A_mBq_m3'],5),
                            str(r['k'])+' / '+integer(attempted),rate_text(r)])
            if r['ci95'][1] < .95:
                below_control.append(c['isotope']+' ('+METHODS[f]+'): '+num(100*r['rate'],3)+'%')
    table(['Изотоп','Метод','A = Ld95,\nмБк/м³','Обнаружено / N','Мощность [95% ДИ]'],control,[.13,.12,.17,.19,.39])
    p('Проверочная мощность может отличаться от 95% из-за случайности новой серии, '
      'погрешности ранее оценённого Ld95 и интерполяции кривой мощности. Интервал в таблице '
      'отражает только биномиальную погрешность новой проверки при фиксированных A и c.')
    if below_control:
        p('В независимой проверке получено '+ '; '.join(below_control)+'. В этих случаях даже '
          'верхняя граница 95% интервала мощности ниже 95%. Таким образом, выбранные оценки '
          'Ld95 для этих сочетаний изотопа и метода не подтвердили целевую мощность; их точность '
          'требует уточнения. Коэффициенты c и активности по результатам этой проверки не менялись.')
        p('Аудит подтвердил правильность заданной активности, ожидаемого спектра, единиц и решений '
          'об обнаружении. Причина отклонения не установлена: результаты не позволяют отдельно '
          'оценить вклад погрешности найденного Ld95 и интерполяции кривой мощности.')
    for i,f in enumerate(METHODS,start=2):
        page();h('10.'+str(i)+'. Гистограммы MDC · '+METHODS[f])
        image('mdc_hist_'+f+'.png','Рисунок '+str(2+i)+'. Стандартный итерационный MDC по наблюдениям · '+METHODS[f]+'.')
        selected=sorted([c for c in data['histograms']['curves'] if c['family']==f],key=lambda c:c['target'])
        table(['Изотоп','A = 0: медиана [2,5–97,5%]','A = Ld95: медиана [2,5–97,5%]','Введённая A'],
              [[c['isotope'],*[num(c[s]['median'])+' ['+num(c[s]['q025'])+'–'+num(c[s]['q975'])+']'
                for s in ('null','signal')],num(c['signal_A_mBq_m3'],5)] for c in selected],[.13,.33,.33,.21])
        p('Все значения таблицы — мБк/м³. Скобки показывают центральные 95% отдельных оценок MDC. '
          'Недоступных MDC в этих распределениях: '+integer(sum(c[s]['n_missing'] for c in selected for s in ('null','signal')))+'.')
    h('11. Что показывают результаты')
    ordinary=[c['methods'][f]['validation']['standard'] for c in curves for f in METHODS]
    p('На независимой проверке исходное правило дало частоты от '+num(100*min(r['rate'] for r in ordinary),3)+
      '% до '+num(100*max(r['rate'] for r in ordinary),3)+'%; после подстройки — от '+
      num(100*min(r['rate'] for r in records),3)+'% до '+num(100*max(r['rate'] for r in records),3)+
      '%. Эти частоты и интервалы нужно рассматривать вместе с Ld95, чтобы сравнивать чувствительность правил при сопоставимом уровне ошибок.')
    if below_control:
        p('Отдельный контроль при выбранных Ld95 выявил '+ '; '.join(below_control)+
          '. Их 95% интервалы целиком ниже 95%; поэтому достижение целевой мощности '
          'для всех изотопов этим исследованием не подтверждено. Это ограничение численных '
          'оценок Ld95 сохранено в результатах без повторной подстройки.')
    ratios=[100*(c['signal']['median']/c['null']['median']-1) for c in data['histograms']['curves']]
    p('При переходе от A = 0 к A = Ld95 изменение медианы обычной оценки MDC составляет от '+
      num(min(ratios),2)+'% до '+num(max(ratios),2)+'%. MDC остаётся случайной характеристикой '
      'обработанной пробы: он зависит от оценённых фона, памяти и других компонентов. '
      'Полные средние, стандартные отклонения и квантили доступны в таблице histogram_statistics.csv.')
    p('Коэффициенты c и численные Ld95 относятся к этой чистой конфигурации с заданным истинным '
      'средним DETBK. Их применимость к произвольной радиоактивной смеси или другому фону '
      'требует отдельной проверки.')
    return dict(title=TITLE,blocks=blocks)


def write_documents(root, document):
    md=['# '+document['title'],''];plain=[document['title'],''];body=['<h1>'+html.escape(document['title'])+'</h1>']
    for b in document['blocks']:
        kind=b['type']
        if kind=='page_break':body.append('<div class="page-break"></div>');continue
        if kind in ('heading','paragraph'):
            text=b['text'];md += [('## ' if kind=='heading' else '')+text,''];plain += [text,'']
            tag='h2' if kind=='heading' else 'p';body.append(f'<{tag}>'+html.escape(text)+f'</{tag}>')
        elif kind=='table':
            md += ['| '+' | '.join(v.replace('\n','<br>') for v in b['headers'])+' |','|'+'|'.join(['---']*len(b['headers']))+'|']
            md += ['| '+' | '.join(str(v).replace('\n','<br>') for v in row)+' |' for row in b['rows']];md.append('')
            plain.append(' | '.join(b['headers']).replace('\n',' '));plain += [' | '.join(str(v).replace('\n',' ') for v in row) for row in b['rows']];plain.append('')
            rows=['<tr>'+''.join('<th>'+html.escape(v).replace('\n','<br>')+'</th>' for v in b['headers'])+'</tr>']
            rows += ['<tr>'+''.join('<td>'+html.escape(str(v)).replace('\n','<br>')+'</td>' for v in row)+'</tr>' for row in b['rows']]
            body.append('<div class="table-wrap"><table>'+''.join(rows)+'</table></div>')
        elif kind=='image':
            md += ['!['+b['caption']+']('+b['file']+')','',b['caption'],''];plain += [b['caption']+' Файл: '+b['file'],'']
            encoded=base64.b64encode((root/b['file']).read_bytes()).decode('ascii')
            body.append('<figure><img src="data:image/png;base64,'+encoded+'" alt="'+html.escape(b['caption'],quote=True)+'"><figcaption>'+html.escape(b['caption'])+'</figcaption></figure>')
    (root/'report_ru.md').write_text('\n'.join(md),encoding='utf-8')
    (root/'report_ru.txt').write_text('\n'.join(plain),encoding='utf-8')
    css='body{max-width:1050px;margin:35px auto;padding:0 22px;color:#172b3a;font:17px/1.55 system-ui,sans-serif}h1{font-size:29px;line-height:1.25}h2{font-size:23px;margin-top:38px;color:#125b75}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #d5e0e6;text-align:left}th{background:#eaf3f7}tr:nth-child(even){background:#f7fafb}.table-wrap{overflow-x:auto}figure{margin:20px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#465966}@media print{body{font-size:11pt;margin:0}h2{break-after:avoid}table{font-size:8pt}.page-break{break-before:page}figure{break-inside:avoid}}'
    (root/'report_ru.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(TITLE)+'</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>',encoding='utf-8')


def write_csv(root,data):
    with (root/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream,delimiter=';')
        writer.writerow(['Изотоп','Метод','Проверочных проб','Доступных решений','Ложных обнаружений Lc','Частота Lc, %','95% ДИ Lc, %','c','95% ДИ c','Ложных обнаружений cLc','Частота cLc, %','95% ДИ cLc, %','Формульный Lc, мБк/м3','Формульный MDC, мБк/м3','Ld95 исходный, мБк/м3','95% ДИ исходного Ld95','Ld95 после подстройки, мБк/м3','95% ДИ при фиксированном c','95% ДИ с учётом c'])
        for c in data['curves']:
            for f,label in METHODS.items():
                m=c['methods'][f];a=m['validation']['standard'];v=m['validation']['calibrated']
                writer.writerow([c['isotope'],label,a['n'],a['n_valid'],a['k'],num(100*a['rate'],4),ci_text(a['ci95'],100,4),num(m['calibration']['c'],8),ci_text(m['calibration']['ci95_c'],digits=8),v['k'],num(100*v['rate'],4),ci_text(v['ci95'],100,4),num(c['nominal_LC_mBq_m3'],8),num(c['nominal_MDC_mBq_m3'],8),num(m['standard_ld_mBq_m3'],8),ci_text(m['standard_ci95_mBq_m3'],digits=8),num(m['calibrated_ld_mBq_m3'],8),ci_text(m['conditional_ci95_mBq_m3'],digits=8),ci_text(m['total_ci95_mBq_m3'],digits=8)])
    with (root/'histogram_statistics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream,delimiter=';')
        writer.writerow(['Метод','Изотоп','Условие','Истинная A, мБк/м3','Попыток','Доступных MDC','Пропусков','Среднее MDC','Медиана MDC','SD MDC','Квантиль 2,5%','Квантиль 97,5%','Формульный MDC','Ld95 после подстройки','c'])
        for c in sorted(data['histograms']['curves'],key=lambda c:(list(METHODS).index(c['family']),c['target'])):
            for scenario,label in [('null','A = 0'),('signal','A = Ld95')]:
                r=c[scenario]
                writer.writerow([METHODS[c['family']],c['isotope'],label,num(0 if scenario=='null' else c['signal_A_mBq_m3'],8),r['n_attempted'],r['n_valid'],r['n_missing'],*[num(r[k],8) for k in ('mean','median','sd','q025','q975')],num(c['nominal_MDC_mBq_m3'],8),num(c['signal_A_mBq_m3'],8),num(c['cutoff_c'],8)])
    with (root/'power_validation.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream,delimiter=';')
        writer.writerow(['Метод','Изотоп','A = Ld95, мБк/м3','Проб','Доступных решений','Обнаружений','Мощность, %','95% ДИ мощности, %','c'])
        for c in sorted(data['histograms']['curves'],key=lambda c:(list(METHODS).index(c['family']),c['target'])):
            r=c['signal']['detection']['calibrated']
            writer.writerow([METHODS[c['family']],c['isotope'],num(c['signal_A_mBq_m3'],8),r.get('n_attempted',r.get('n')),r['n_valid'],r['k'],num(100*r['rate'],4),ci_text(r['ci95'],100,4),num(c['cutoff_c'],8)])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root.resolve();data,arrays=read_checked(root)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
    for rule in ('standard','calibrated'):alpha_figure(root,data,rule)
    power_figure(root,data)
    for f in METHODS:histogram_figure(root,data,arrays,f)
    document=content(data)
    (root/'content.json').write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
    write_documents(root,document);write_csv(root,data)
    print('Rendered the completed, independently audited single-configuration study. No new Monte Carlo.')


if __name__=='__main__':main()
