"""Report nominal formula and independent Monte Carlo power checks."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import numpy as np
from analysis_tools import load_groups,rate,crossing,FAMILIES

ROOT=Path(__file__).resolve().parent
LABELS={'wls':'МНК','poisson':'ММП'}
DATES={'may22':'22 мая','may26':'26 мая'}
ISOTOPES={1:'Xe135',2:'Xe133',3:'Xe131m',4:'Xe133m'}


def curve_data(groups,dataset,target):
    null=next(g for g in groups if g['dataset']==dataset and g['stage']=='null')
    screen=sorted([g for g in groups if g['dataset']==dataset and g['target']==target and g['stage']=='screen'],key=lambda g:g['factor'])
    # Pool independent confirmation/extension batches only when activities coincide.
    merged={}
    for g in groups:
        if g['dataset']!=dataset or g['target']!=target or g['stage'] not in ('confirm','extend'):continue
        entry=merged.setdefault(g['factor'],{f:[] for f in FAMILIES})
        for f in FAMILIES:entry[f].append(g['decisions'][f][:,0])
    formal=[dict(factor=x,decisions={f:np.concatenate(v[f]) for f in FAMILIES}) for x,v in sorted(merged.items())]
    return null,screen,formal


def bootstrap(formal,seed):
    rng=np.random.default_rng(seed);B=2000;x=np.array([p['factor'] for p in formal])
    probabilities={f:[] for f in FAMILIES}
    states=np.array([(i,j) for i in (-1,0,1) for j in (-1,0,1)])
    for p in formal:
        joint=np.stack([p['decisions'][f] for f in FAMILIES],axis=1)
        counts=np.array([np.sum(np.all(joint==state,axis=1)) for state in states])
        draws=rng.multinomial(len(joint),counts/len(joint),size=B)
        for col,f in enumerate(FAMILIES):
            good=states[:,col]>=0;yes=states[:,col]==1
            denominator=draws[:,good].sum(axis=1)
            probabilities[f].append(np.divide(draws[:,yes].sum(axis=1),denominator,
                out=np.full(B,np.nan),where=denominator>0))
    result={}
    for f,values in probabilities.items():
        y=np.stack(values,axis=1)
        roots=[];below=above=0
        for p in y:
            value=crossing(x,p)
            if value is None:
                # Keep out-of-grid draws in the percentile distribution.
                # Discarding them would condition the interval on crossing.
                if np.all(p>.95):value=-np.inf;below+=1
                elif np.all(p<.95):value=np.inf;above+=1
                else:raise RuntimeError('Ambiguous bootstrap crossing; extend the grid')
            roots.append(value)
        roots=np.array(roots);good=roots[np.isfinite(roots)]
        ci=np.quantile(roots,[.025,.975],method='inverted_cdf')
        if not np.all(np.isfinite(ci)):
            raise RuntimeError('Bootstrap interval extends outside the grid; add an independent point')
        result[f]=dict(n_repetitions=B,n_crossings=len(good),
            n_below_grid=below,n_above_grid=above,approximate_ci95_factor=ci.tolist(),
            bootstrap_sd_conditional_on_grid_factor=float(np.std(good,ddof=1)) if len(good)>1 else None)
    return result


def summarize(manifest,groups,errors):
    curves=[]
    for dataset,data in manifest['datasets'].items():
        for target,isotope in ISOTOPES.items():
            null,screen,formal=curve_data(groups,dataset,target)
            assert null['n']==5000 and len(screen)==4 and len(formal)>=3
            nominal=data['nominal_MDC'][target-1]*1000
            x=np.array([p['factor'] for p in formal]);boot=bootstrap(formal,[2026102209,list(DATES).index(dataset),target])
            methods={}
            for family in FAMILIES:
                alpha=rate(null['decisions'][family][:,target-1])
                points=[dict(factor=p['factor'],activity_mBq_m3=p['factor']*nominal,**rate(p['decisions'][family])) for p in formal]
                y=[p['rate'] for p in points];root=crossing(x,y)
                if root is None:raise RuntimeError(f'Need independent extension: {dataset} {target} {family}: {list(zip(x,y))}')
                policy=json.loads((ROOT/'interval_policy.json').read_text())
                assert len(formal)+1<=policy['max_grid_points_including_null']
                confidence=policy['confidence_per_grid_point']
                bounds=[(0.,rate(null['decisions'][family][:,target-1],confidence))]
                bounds.extend((p['factor'],rate(p['decisions'][family],confidence)) for p in formal)
                lower=max((a for a,r in bounds if r['conservative_ci'][1]<.95),default=0.)
                upper=min((a for a,r in bounds if r['conservative_ci'][0]>.95),default=None)
                bootstrap_result=boot[family]
                methods[family]=dict(ld_mBq_m3=root*nominal,ld_factor=root,formal_points=points,
                    relative_difference_from_formula_percent=(root-1)*100,
                    nominal_formula_power=rate(next(g for g in screen if g['factor']==1.)['decisions'][family][:,0]),
                    null=alpha,approximate_bootstrap_ci95_mBq_m3=[v*nominal if v is not None else None for v in bootstrap_result['approximate_ci95_factor']],
                    grid_bound95_mBq_m3=[lower*nominal,upper*nominal if upper is not None else None],
                    simultaneous_point_confidence=confidence,bootstrap=bootstrap_result,
                    empirical_reversals=int(np.sum(np.diff(y)<0)),
                    missing_completion_ld_range_mBq_m3=[
                        z*nominal if z is not None else None for z in [crossing(x,[p['bounds'][1] for p in points]),crossing(x,[p['bounds'][0] for p in points])]])
            curves.append(dict(dataset=dataset,target=target,isotope=isotope,nominal_LC_mBq_m3=data['nominal_LC'][target-1]*1000,
                nominal_MDC_mBq_m3=nominal,methods=methods,
                screening=[dict(factor=g['factor'],methods={f:rate(g['decisions'][f][:,0]) for f in FAMILIES}) for g in screen]))
    return dict(scope=manifest['scope'],unit='mBq/m3',n_vectors=sum(g['n'] for g in groups),
        n_by_stage={stage:sum(g['n'] for g in groups if g['stage']==stage) for stage in ('null','screen','confirm','extend')},
        validation_vectors_excluded=400,curves=curves,errors=errors,
        interval_note='Local linear interpolation of independent confirmation points; paired bootstrap is approximate and omits interpolation-shape error; grid bounds invert simultaneous binomial intervals under monotone true power',
        scope_note='Background rates fixed to original DETBK ROI counts/live time; all 17 parameters remain fitted; no calibration of nominal alpha')


def plots(summary):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    colors={'wls':'#087eaa','poisson':'#d9650b'}
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    for c in summary['curves']:
        ax=axes[list(DATES).index(c['dataset']),c['target']-1]
        for f in FAMILIES:
            m=c['methods'][f];p=m['formal_points'];x=np.array([v['factor'] for v in p]);y=np.array([v['rate'] for v in p])*100
            error=np.array([v['ci95'] for v in p]).T*100
            ax.errorbar(x,y,yerr=np.stack([y-error[0],error[1]-y]),color=colors[f],marker='o' if f=='wls' else 's',
                        markersize=4,lw=1.6,capsize=3,alpha=.95)
            sx=np.array([v['factor'] for v in c['screening']]);sy=[v['methods'][f]['rate']*100 for v in c['screening']]
            ax.plot(sx,sy,linestyle='none',marker='o' if f=='wls' else 's',fillstyle='none',color=colors[f],alpha=.6,markersize=5)
            ax.plot(m['ld_factor'],95,marker='|',ms=16,mew=2,color=colors[f])
        ax.axhline(95,color='.3',ls=':',lw=1.3)
        ax.axvline(1,color='.4',ls='--',lw=1)
        ax.set_title(f"{DATES[c['dataset']]} — {c['isotope']}",fontsize=12)
        ax.text(.98,.05,f"Формула: {c['nominal_MDC_mBq_m3']:.3f} мБк/м³",transform=ax.transAxes,ha='right',fontsize=9)
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(78,100.5)
        if c['target']==1:ax.set_ylabel('Вероятность обнаружения, %')
        if c['dataset']=='may26':ax.set_xlabel('Истинная A / формульный MDC')
    fig.suptitle('Чистый детекторный фон: проверка номинального MDC полной моделью',fontsize=17,fontweight='bold',y=.985)
    fig.text(.5,.945,'Память и прочие изотопы нулевые при генерации; SAMPLE, GASBK и DETBK оцениваются совместно',ha='center',fontsize=11)
    handles=[Line2D([],[],color=colors[f],marker='o' if f=='wls' else 's',label=LABELS[f]) for f in FAMILIES]
    handles.extend([Line2D([],[],color='.4',marker='o',ls='none',fillstyle='none',label='Поиск: 500 реализаций'),
                    Line2D([],[],color='.4',marker='o',ls='-',label='Подтверждение: 3000 реализаций'),
                    Line2D([],[],color='.4',ls='--',label='Формульный MDC')])
    fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=10,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=[0,.06,1,.92],h_pad=2)
    fig.savefig(ROOT/'nominal_power.pdf');fig.savefig(ROOT/'nominal_power.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5.7),sharey=True,sharex=True)
    for dataset,ax in zip(DATES,axes):
        for c in (v for v in summary['curves'] if v['dataset']==dataset):
            for f,offset in [('wls',-.10),('poisson',.10)]:
                m=c['methods'][f];ci=np.array(m['approximate_bootstrap_ci95_mBq_m3'])/c['nominal_MDC_mBq_m3'];y=c['target']-1+offset
                ax.hlines(y,ci[0],ci[1],color=colors[f],lw=2)
                ax.plot(m['ld_factor'],y,'o' if f=='wls' else 's',color=colors[f],ms=6)
        ax.axvline(1,color='.3',ls='--',lw=1)
        ax.set_title(DATES[dataset],fontsize=14);ax.set_yticks(range(4),list(ISOTOPES.values()))
        ax.set_xlabel('Ld95 по мощности / формульный MDC')
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Номинальный предел: формула и фактическая мощность',fontsize=16,fontweight='bold')
    fig.legend(handles=handles[:2],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.04))
    fig.text(.5,.015,'Полосы — приближённые 95% интервалы интерполяции по парному bootstrap; неопределённость модели не включена.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.92])
    fig.savefig(ROOT/'nominal_comparison.pdf');fig.savefig(ROOT/'nominal_comparison.png',dpi=170);plt.close(fig)


def text_report(summary,manifest):
    lines=['# Номинальный MDC: чистый детекторный фон и полная модель XeMat','',
        'Два майских комплекта задают времена, сбор, задержки, объёмы и калибровки. Проба, газовая память и радон в генерирующей смеси равны нулю; затем вводится один изотоп ксенона. Все 17 параметров продолжают оцениваться при обработке.','',
        'Средний фон задан по исходному DETBK детектора 221, без использования смеси из совместной подгонки реальных SAMPLE/GASBK. По ROI [1,2,3,5,6,7,8] zeta=[522,854,494,66,64,89,97] отсчётов за 259200 с. SAMPLE, GASBK и DETBK разыгрываются заново по Пуассону.','',
        f"Всего **{summary['n_vectors']:,} новых реализаций**: {summary['n_by_stage']['null']:,} на нуле, {summary['n_by_stage']['screen']:,} для поиска, {summary['n_by_stage']['confirm']:,} для подтверждения и {summary['n_by_stage']['extend']:,} для уточнения сетки. Технические 400 реализаций исключены. МНК и ММП обрабатывают одни и те же векторы. Пороги не калибровались к фактическим 5%; использована действующая процедура Lc с вычетом целевого изотопа из неотрицательной подгонки.",'',
        '## Формульные номинальные значения','',
        'Единицы — мБк/м³. Эти значения вычислены итерационно при точно заданной фоновой смеси, с полной информационной матрицей всех 17 параметров. При таком задании они общие для МНК и ММП; различия способов оценивания проявляются в Монте-Карло.','',
        r'В координатах $\eta=[a_V;\gamma;\zeta]$ положено $\eta_0=[0;0;\zeta_D]$, $\mu_i(A)=S_V(\eta_0+A e_i)+b$ и $K_i(A)=(S_V^T\operatorname{diag}(\mu_i(A))^{-1}S_V)^{-1}$. Использованы $L_C=z_{0.95}\sqrt{K_i(0)_{ii}}$ и положительный корень $MDC=L_C+z_{0.95}\sqrt{K_i(MDC)_{ii}}$. Обращение записано здесь как математическая формула; вычисления выполняются устойчивыми разложениями.',
        '', 'Формула использует нормальное приближение. Точное численное решение этого уравнения само по себе не гарантирует ровно 95% мощности при пуассоновских наблюдениях и случайном оценённом Lc. Для этого и выполнена отдельная проверка Монте-Карло.','',
        '| Изотоп | Lc, 22 мая | MDC, 22 мая | Lc, 26 мая | MDC, 26 мая |',
        '|---|---:|---:|---:|---:|']
    for i,iso in ISOTOPES.items():
        a=manifest['datasets']['may22'];b=manifest['datasets']['may26']
        lines.append(f"| {iso} | {a['nominal_LC'][i-1]*1000:.5f} | {a['nominal_MDC'][i-1]*1000:.5f} | {b['nominal_LC'][i-1]*1000:.5f} | {b['nominal_MDC'][i-1]*1000:.5f} |")
    lines.extend(['','## Активность для фактических 95% обнаружения','',
        'Ниже — приближённые Ld95, найденные по независимым подтверждающим точкам мощности. Это характеристика действующей процедуры, включая случайный Lc и повторное оценивание фона.','',
        '| Комплект | Изотоп | Формульный MDC | Ld95 МНК | Ld95 ММП | Отклонение ММП от формулы |',
        '|---|---|---:|---:|---:|---:|'])
    for c in summary['curves']:
        w=c['methods']['wls'];p=c['methods']['poisson']
        lines.append(f"| {DATES[c['dataset']]} | {c['isotope']} | {c['nominal_MDC_mBq_m3']:.4f} | {w['ld_mBq_m3']:.4f} | {p['ld_mBq_m3']:.4f} | {p['relative_difference_from_formula_percent']:+.1f}% |")
    lines.extend(['',
        'На этих двух комплектах формульный MDC и оценка Ld95 близки для Xe135/Xe133. Для Xe131m оценка Ld95 по ММП выше формулы примерно на 7%, для Xe133m — на 9–10%. МНК даёт близкие пределы, но у него выше частота ложных обнаружений Xe133; поэтому меньшее Ld само по себе не означает преимущество метода. Эти отклонения относятся к данному чистому фону и не являются универсальной поправкой к MDC.',
        '', '![Сопоставление](nominal_comparison.png)','','![Мощность около номинального MDC](nominal_power.png)','',
        '## Ложные обнаружения на чистом фоне','',
        'В нулевой серии нет ни одного изотопа и нет памяти. Одна серия из 5000 векторов на комплект используется для всех четырёх проверок. Интервалы ниже — индивидуальные точные биномиальные 95% интервалы.','',
        '| Комплект | Изотоп | МНК: частота и 95% ДИ | ММП: частота и 95% ДИ |',
        '|---|---|---|---|'])
    for c in summary['curves']:
        values=[]
        for f in FAMILIES:
            r=c['methods'][f]['null'];ci=r['ci95'];values.append(f"{r['rate']*100:.2f}% ({ci[0]*100:.2f}–{ci[1]*100:.2f}%)")
        lines.append(f"| {DATES[c['dataset']]} | {c['isotope']} | {' | '.join(values)} |")
    lines.extend(['','## Неопределённость Ld95','',
        'Точечный Ld95 — локальная линейная интерполяция подтверждающих точек. Приближённый парный bootstrap (2000 повторов) учитывает случайность Монте-Карло, но не ошибку формы кривой между узлами. Пересечения, уходящие за сетку, учитываются в порядке квантилей как цензурированные слева/справа, а не исключаются. Если граница 95% интервала оказывается вне сетки, требуется новая независимая точка. Сеточные границы консервативнее: они получены инверсией одновременных биномиальных интервалов при предположении монотонной истинной мощности.','',
        '| Комплект | Изотоп | Метод | Ld95 | Приближённый 95% ДИ | Сеточные границы 95% | Bootstrap с пересечением |',
        '|---|---|---|---:|---|---|---|'])
    for c in summary['curves']:
        for f in FAMILIES:
            m=c['methods'][f];ci=m['approximate_bootstrap_ci95_mBq_m3'];grid=m['grid_bound95_mBq_m3'];boot=m['bootstrap']
            gridtext=f"{grid[0]:.4f}–{grid[1]:.4f}" if grid[1] is not None else f"от {grid[0]:.4f}, верхняя граница не достигнута"
            lines.append(f"| {DATES[c['dataset']]} | {c['isotope']} | {LABELS[f]} | {m['ld_mBq_m3']:.4f} | {ci[0]:.4f}–{ci[1]:.4f} | {gridtext} | {boot['n_crossings']}/{boot['n_repetitions']} |")
    lines.extend(['','## Сохранённые параметры измерений','',
        '| Параметр | 22 мая | 26 мая |','|---|---:|---:|'])
    for label,key in [('SAMPLE: живое время, с','t_sl'),('SAMPLE: реальное время, с','t_sr'),('GASBK: живое время, с','t_gl'),('GASBK: реальное время, с','t_gr'),('DETBK: живое время, с','t_bl'),('Сбор, с','t_a'),('Задержка после сбора, с','t_p'),('Разность начал SAMPLE/GASBK, с','tau')]:
        vals=[manifest['datasets'][d]['metadata']['time_parameters'][key] for d in DATES]
        lines.append(f'| {label} | {vals[0]} | {vals[1]} |')
    for label,func in [('Объём Xe, см³',lambda m:m['xenon_volume_cm3']),('Расчётный эффективный объём воздуха, м³',lambda m:m['effective_air_volume_m3']),('Поле air_volume исходного SAMPLE, м³',lambda m:m['sample_temporal']['air_volume'])]:
        vals=[func(manifest['datasets'][d]['metadata']) for d in DATES]
        lines.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} |')
    lines.extend(['','Сохранена конвенция объёма существующего ядра: объёмные активности рассчитываются через effective_air_volume_m3, полученный из объёма Xe. Поле air_volume показано отдельно для прозрачности. Эффективности SAMPLE/GASBK [0.2057,0.314,0.258,0.258] и branching из constants.py сохранены. Матрицы R — для детектора 221. Все матрицы и метаданные находятся в manifest.json.','',
        '## Проверки и ограничения','',
        f"Ошибок подгонки/расчёта Lc: {len(summary['errors'])}. Недоступные решения, если возникают, сохраняются и учитываются отдельно; новые реализации взамен не генерируются.",'',
        'Формульные MDC независимо проверены через проекцию мешающих столбцов и решение скалярного уравнения методом Брента. Проверено совпадение Lc в fixed/iterative, воспроизведение отсчётов по seed, сохранение полной модели, нулевая память генератора и отсутствие изменений математического ядра.','',
        'Чистый фон не означает известный без ошибки фон при анализе: zeta оценивается заново по каждому совместному комплекту. Фоновые средние для генерации фиксированы по одному исходному DETBK; неопределённость самих этих принятых средних и калибровок не включена. Систематические ошибки не моделировались. Следовательно, результаты условны относительно данного детектора, фонового уровня и майских параметров.','',
        '## Файлы','',
        '- [График сопоставления PDF](nominal_comparison.pdf)\n- [Проверка мощности PDF](nominal_power.pdf)\n- [Таблица CSV](results.csv)\n- [Полная численная сводка](summary.json)\n- [Протокол](protocol_ru.md)\n- [Архив с данными](XeMat_nominal_MDC_May.zip)',''])
    text='\n'.join(lines)
    (ROOT/'report_ru.md').write_text(text,encoding='utf-8');(ROOT/'report_ru.txt').write_text(text,encoding='utf-8')


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text());groups,errors=load_groups()
    planned_n=sum(job['n'] for name in ('initial_plan.json','confirmation_plan.json','extension_plan.json')
                  if (ROOT/name).exists() for job in json.loads((ROOT/name).read_text()))
    assert sum(g['n'] for g in groups)==planned_n,'Some planned Monte Carlo batches are incomplete'
    summary=summarize(manifest,groups,errors)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (ROOT/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['dataset','isotope','nominal_LC_mBq_m3','nominal_MDC_mBq_m3','wls_Ld95_mBq_m3','poisson_Ld95_mBq_m3','wls_false_positive_rate','poisson_false_positive_rate'])
        for c in summary['curves']:
            w.writerow([c['dataset'],c['isotope'],c['nominal_LC_mBq_m3'],c['nominal_MDC_mBq_m3'],c['methods']['wls']['ld_mBq_m3'],c['methods']['poisson']['ld_mBq_m3'],c['methods']['wls']['null']['rate'],c['methods']['poisson']['null']['rate']])
    plots(summary);text_report(summary,manifest)
    print('Completed summary:',summary['n_vectors'],'vectors;',len(errors),'errors')
    for c in summary['curves']:print(c['dataset'],c['isotope'],c['nominal_MDC_mBq_m3'],{f:c['methods'][f]['ld_mBq_m3'] for f in FAMILIES})


if __name__=='__main__':main()
