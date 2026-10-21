"""Summarize completed MDC sampling series; no fitting or data selection."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import os
import numpy as np
from scipy.stats import binomtest, t

ROOT=Path(__file__).resolve().parent
FAMILIES=('wls','poisson')
LABELS={'wls':'МНК','poisson':'ММП'}
DATES={'may22':'22 мая','may26':'26 мая'}


def proportions(k,n):
    ci=binomtest(int(k),int(n)).proportion_ci(.95,method='exact')
    return dict(k=int(k),n=int(n),rate=float(k/n),ci95=[float(ci.low),float(ci.high)])


def mean_ci(values):
    mean=float(np.mean(values));se=float(np.std(values,ddof=1)/np.sqrt(len(values)))
    width=float(t.ppf(.975,len(values)-1)*se)
    return dict(mean=mean,mc_standard_error=se,mean_ci95=[mean-width,mean+width])


def load():
    grouped={}
    errors=[]
    for dataset in DATES:
        for scenario in ('null','signal'):
            parts={}
            paths=sorted((ROOT/'chunks').glob(f'{dataset}_{scenario}_*.npz'))
            assert len(paths)==10,(dataset,scenario,len(paths))
            for path in paths:
                a=np.load(path)
                meta=json.loads(path.with_suffix('.json').read_text())
                assert not meta['validation']
                for k in a.files:parts.setdefault(k,[]).append(a[k])
                errors.extend(dict(e,dataset=dataset,scenario=scenario) for e in meta['errors'])
            merged={k:np.concatenate(v,axis=0) for k,v in parts.items()}
            assert np.array_equal(merged['indices'],np.arange(10000))
            grouped[dataset,scenario]=merged
    return grouped,errors


def summarize(manifest,grouped,errors):
    cells=[]
    method_comparisons=[]
    signal_comparisons=[]
    for (dataset,scenario),a in grouped.items():
        data=manifest['datasets'][dataset]
        for family in FAMILIES:
            mdc=a[family+'_MDC']*1000
            valid=np.isfinite(mdc);v=mdc[valid]
            ref=data['reference_ld'][family]['estimate']*1000
            approx_ci=data['reference_ld'][family]['approximate_bootstrap_ci95']
            grid_ci=data['reference_ld'][family]['grid_bound95_mBq_m3']
            decisions=a[family+'_detected'];known=decisions>=0
            activity=a[family+'_free'][valid,4]*1000
            lc=a[family+'_LC'][valid]*1000
            stats=mean_ci(v)
            errors_vs_ref=v-ref
            below=proportions(np.sum(v<ref),len(v))
            det=proportions(np.sum(decisions[known]),np.sum(known))
            missing=int(np.sum(~known));det['n_missing']=missing
            det['missing_completion_rate_bounds']=[det['k']/len(decisions),(det['k']+missing)/len(decisions)]
            cells.append(dict(dataset=dataset,scenario=scenario,family=family,
                activity_mBq_m3=0 if scenario=='null' else data['signal_activity']*1000,
                n_total=len(mdc),n_valid=len(v),n_missing=int(np.sum(~valid)),
                **stats,median=float(np.median(v)),sd=float(np.std(v,ddof=1)),
                central95=[float(x) for x in np.quantile(v,[.025,.975])],
                q05_q95=[float(x) for x in np.quantile(v,[.05,.95])],
                min=float(np.min(v)),max=float(np.max(v)),
                cv_percent=float(np.std(v,ddof=1)/np.mean(v)*100),
                reference_ld=ref,reference_approximate_ci95=approx_ci,reference_grid_bound95=grid_ci,
                bias=float(np.mean(errors_vs_ref)),relative_bias_percent=float(np.mean(errors_vs_ref)/ref*100),
                rmse=float(np.sqrt(np.mean(errors_vs_ref**2))),
                below_reference=below,
                bias_sensitivity_to_reference_approximate_ci=[stats['mean']-approx_ci[1],stats['mean']-approx_ci[0]],
                bias_sensitivity_to_reference_grid_ci=[stats['mean']-grid_ci[1],stats['mean']-grid_ci[0]],
                below_reference_sensitivity_approximate_ci=[float(np.mean(v<lim)) for lim in approx_ci],
                original_reported_mdc=data['original_reported'][family]['MDC']*1000,
                oracle_formula_mdc=data['oracle_formula_MDC']*1000,
                mean_minus_oracle=stats['mean']-data['oracle_formula_MDC']*1000,
                oracle_minus_reference=data['oracle_formula_MDC']*1000-ref,
                detection=det,
                activity_MDC_correlation=float(np.corrcoef(activity,v)[0,1]),
                activity_LC_correlation=float(np.corrcoef(activity,lc)[0,1]),
                iterations_max=int(np.max(a[family+'_iterations'])),
                equation_residual_max=float(np.nanmax(a[family+'_equation_relative_residual'])),
                free_covariance_unavailable=int(np.sum(~a[family+'_free_covariance_available']))))
        paired=np.isfinite(a['wls_MDC'])&np.isfinite(a['poisson_MDC'])
        delta=(a['wls_MDC'][paired]-a['poisson_MDC'][paired])*1000
        method_comparisons.append(dict(dataset=dataset,scenario=scenario,n_paired=int(paired.sum()),
            definition='WLS MDC minus Poisson MDC on identical observations',**mean_ci(delta),
            sd=float(np.std(delta,ddof=1)),central95=np.quantile(delta,[.025,.975]).tolist(),
            correlation=float(np.corrcoef(a['wls_MDC'][paired],a['poisson_MDC'][paired])[0,1])))
    for dataset in DATES:
        for family in FAMILIES:
            null=next(c for c in cells if c['dataset']==dataset and c['scenario']=='null' and c['family']==family)
            signal=next(c for c in cells if c['dataset']==dataset and c['scenario']=='signal' and c['family']==family)
            delta=signal['mean']-null['mean']
            se=float(np.hypot(null['mc_standard_error'],signal['mc_standard_error']))
            signal_comparisons.append(dict(dataset=dataset,family=family,
                definition='Independent signal scenario mean minus null scenario mean',
                difference=delta,mc_standard_error=se,approximate_ci95=[delta-1.96*se,delta+1.96*se],
                relative_to_null_percent=delta/null['mean']*100))
    return dict(scope=manifest['scope'],unit='mBq/m3',n_new_vectors=40000,
        n_application_MDC_calculations=80000,validation_vectors_excluded=400,
        cells=cells,paired_method_comparisons=method_comparisons,
        signal_vs_null_mean_comparisons=signal_comparisons,errors=errors,
        reference_note='Prior independent MC Ld95 estimates are kept fixed, not refitted. Mean CIs condition on those references.',
        central95_note='Central range of reported MDC across repeat measurements, not a CI for mean or true Ld.',
        distribution_note='All trials retained; summaries of MDC conditional on available MDC if any failures; no replacements.')


def plots(summary,manifest,grouped):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    colors={'wls':'#147eab','poisson':'#df6c13'}
    fig,axes=plt.subplots(2,2,figsize=(13.6,9.7),sharex='col',sharey=True)
    for col,dataset in enumerate(DATES):
        d=manifest['datasets'][dataset]
        values=np.concatenate([grouped[dataset,s][f+'_MDC']*1000 for s in ('null','signal') for f in FAMILIES])
        values=values[np.isfinite(values)]
        bins=np.linspace(values.min()-.001,values.max()+.001,45)
        ref=d['reference_ld']['poisson']['estimate']*1000
        refci=d['reference_ld']['poisson']['approximate_bootstrap_ci95']
        oracle=d['oracle_formula_MDC']*1000
        for row,scenario in enumerate(('null','signal')):
            ax=axes[row,col]
            for family in FAMILIES:
                v=grouped[dataset,scenario][family+'_MDC']*1000;v=v[np.isfinite(v)]
                ax.hist(v,bins=bins,weights=np.full(len(v),100/len(v)),histtype='step',
                        color=colors[family],linewidth=1.8,label=LABELS[family])
            ax.axvspan(*refci,color='gray',alpha=.17,zorder=0)
            ax.axvline(ref,color='#242424',ls='--',lw=1.6)
            ax.axvline(oracle,color='#555555',ls=':',lw=1.5)
            a=0 if scenario=='null' else d['signal_activity']*1000
            ax.set_title(f'{DATES[dataset]}: истинная A = {a:.3f} мБк/м³',fontsize=13,pad=10)
            cell=next(c for c in summary['cells'] if c['dataset']==dataset and c['scenario']==scenario and c['family']=='poisson')
            text=f"ММП: среднее {cell['mean']:.3f}\n95% значений: {cell['central95'][0]:.3f}–{cell['central95'][1]:.3f}"
            ax.text(.98,.96,text,transform=ax.transAxes,ha='right',va='top',fontsize=10,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.88))
            ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel('Доля реализаций в интервале, %')
            if row==1:ax.set_xlabel('Выдаваемый MDC, мБк/м³')
    handles=[Line2D([],[],color=colors[f],lw=2,label=LABELS[f]) for f in FAMILIES]
    handles.extend([Line2D([],[],color='#242424',ls='--',label='Ld95 по мощности ММП–Lc'),
                    Line2D([],[],color='#555555',ls=':',label='Формула MDC при истинной смеси'),
                    Patch(facecolor='gray',alpha=.17,label='Приближённый 95% ДИ Ld95')])
    fig.suptitle('Xe133m: распределение MDC при повторных измерениях',fontsize=17,fontweight='bold',y=.985)
    fig.text(.5,.945,'По 10 000 новых реализаций на панель; МНК и ММП обрабатывают одинаковые отсчёты',ha='center',fontsize=11)
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005),fontsize=10)
    fig.tight_layout(rect=[0,.085,1,.92],h_pad=2)
    fig.savefig(ROOT/'mdc_distributions.png',dpi=170)
    fig.savefig(ROOT/'mdc_distributions.pdf')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True,sharex=True)
    labels=['МНК, A=0','ММП, A=0','МНК, A≈Ld','ММП, A≈Ld']
    for col,dataset in enumerate(DATES):
        ax=axes[col]
        cells=[c for c in summary['cells'] if c['dataset']==dataset]
        for i,c in enumerate(cells):
            color=colors[c['family']]; ref=c['reference_ld']
            ax.hlines(i,c['central95'][0]/ref,c['central95'][1]/ref,color=color,lw=5,alpha=.5)
            ax.plot(c['median']/ref,i,'o',color=color,ms=7)
            ax.plot(c['mean']/ref,i,'D',color=color,ms=5)
        ax.axvline(1,color='black',ls='--',lw=1.2)
        ax.set_title(DATES[dataset],fontsize=14)
        ax.set_yticks(range(4),labels)
        ax.set_xlabel('Выдаваемый MDC / собственный Ld95 метода')
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Разброс MDC относительно эталона по мощности',fontsize=16,fontweight='bold')
    fig.text(.5,.025,'Полоса — центральные 95% значений MDC; круг — медиана; ромб — среднее. Эталон здесь зафиксирован.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.065,1,.92])
    fig.savefig(ROOT/'mdc_relative_ranges.png',dpi=170)
    fig.savefig(ROOT/'mdc_relative_ranges.pdf')
    plt.close(fig)


def report(summary,manifest):
    lines=['# Xe133m: насколько случаен выдаваемый программой MDC','',
        'Два майских комплекта, по 10 000 независимых новых реализаций при A=0 и при прежнем Ld95 ММП–Lc. Всего **40 000 векторов SAMPLE/GASBK/DETBK и 80 000 расчётов MDC** (МНК и ММП на одинаковых данных). Код приложения не менялся.','',
        'Рассчитывался именно **итерационный MDC действующей процедуры Lc с удалением вклада свежего Xe133m из неотрицательной оценки**. Вальд в этом исследовании не используется. Все объёмные активности ниже — мБк/м³.','',
        '## Что получилось','',
        'Средние выдаваемые MDC оказались ниже прежних точечных эталонов Ld95 во всех восьми сочетаниях. Для ММП разность составляет около −8,2% при A=0 и −5,6% при A≈Ld на 22 мая; около −5,1% и −2,3% соответственно на 26 мая. Это оценки смещения относительно приближённого эталона, а не универсальные поправочные коэффициенты.','',
        'При переходе от A=0 к A≈Ld среднее MDC ММП увеличивается примерно на 2,8–2,9%, и распределение становится шире. Значит, удаление целевого вклада из подгонки не обеспечивает инвариантности распределения MDC к истинной активности: сохраняемые мешающие параметры тоже оцениваются по случайным данным, свойства которых зависят от сигнала.','',
        'МНК и ММП дают близкие распределения, но MDC МНК в среднем немного ниже: на 0,0011–0,0036 мБк/м³ в этих сериях. Разница намного меньше разброса MDC от измерения к измерению.','',
        'Практический вывод: выдаваемый MDC следует рассматривать как случайную оценку чувствительности. Для статьи полезно отдельно приводить её смещение и разброс, фактическую мощность правила и его ложноположительную частоту. Эти результаты относятся к двум зафиксированным генерирующим смесям.','',
        '## Распределение выдаваемого MDC','',
        'Диапазон 2,5–97,5% описывает разброс результатов по повторным измерениям. Это не доверительный интервал среднего и не доверительный интервал истинного Ld.','',
        '| SAMPLE | Истинная A | Оценивание | Среднее MDC | Медиана | SD | 2,5–97,5% значений | Ld95 по мощности | Смещение среднего |',
        '|---|---:|---|---:|---:|---:|---|---:|---:|']
    for c in summary['cells']:
        lines.append(f"| {DATES[c['dataset']]} | {c['activity_mBq_m3']:.3f} | {LABELS[c['family']]} | {c['mean']:.4f} | {c['median']:.4f} | {c['sd']:.4f} | {c['central95'][0]:.4f}–{c['central95'][1]:.4f} | {c['reference_ld']:.4f} | {c['relative_bias_percent']:+.2f}% |")
    lines.extend(['','![Распределения MDC](mdc_distributions.png)','',
        '![Относительный разброс](mdc_relative_ranges.png)','',
        'Ненулевой уровень генерации одинаков для МНК и ММП: 0,383547 для 22 мая и 0,325126 для 26 мая. Для МНК собственные эталоны немного отличаются (0,383810 и 0,325843); в смещениях и относительных графиках использован собственный эталон каждого метода.','',
        '## Насколько точно определено среднее','',
        'Следующие интервалы отражают только ошибку Монте-Карло среднего MDC. Неопределённость эталона Ld95 рассматривается отдельно.','',
        '| SAMPLE | Сценарий | Метод | 95% ДИ среднего MDC | MDC ниже точечного эталона | Корреляция свободной активности и MDC |',
        '|---|---|---|---|---:|---:|'])
    for c in summary['cells']:
        ci=c['mean_ci95']; rate=c['below_reference']['rate']*100
        lines.append(f"| {DATES[c['dataset']]} | {'A=0' if c['scenario']=='null' else 'A≈Ld95'} | {LABELS[c['family']]} | {ci[0]:.5f}–{ci[1]:.5f} | {rate:.2f}% | {c['activity_MDC_correlation']:.3f} |")
    lines.extend(['','Доля MDC ниже эталона — описательная характеристика точечной оценки. От неё не требуется быть равной 5%: такое требование относилось бы к отдельно построенной верхней доверительной границе Ld.','',
        '## Меняется ли распределение при наличии сигнала','',
        '| SAMPLE | Метод | Разность средних MDC: A≈Ld минус A=0 | 95% ДИ разности | Изменение относительно среднего при A=0 |',
        '|---|---|---:|---|---:|'])
    for c in summary['signal_vs_null_mean_comparisons']:
        ci=c['approximate_ci95']
        lines.append(f"| {DATES[c['dataset']]} | {LABELS[c['family']]} | {c['difference']:+.5f} | {ci[0]:+.5f}…{ci[1]:+.5f} | {c['relative_to_null_percent']:+.2f}% |")
    lines.extend(['','## Парное сравнение способов оценивания','',
        '| SAMPLE | Сценарий | Среднее MDC(МНК) − MDC(ММП) | 95% ДИ разности | Корреляция MDC двух методов |',
        '|---|---|---:|---|---:|'])
    for c in summary['paired_method_comparisons']:
        ci=c['mean_ci95']
        lines.append(f"| {DATES[c['dataset']]} | {'A=0' if c['scenario']=='null' else 'A≈Ld95'} | {c['mean']:+.5f} | {ci[0]:+.5f}…{ci[1]:+.5f} | {c['correlation']:.4f} |")
    lines.extend(['','## Проверка обнаружения на новых реализациях','',
        '| SAMPLE | Истинная A | Метод | Обнаружений | Частота | 95% точный биномиальный ДИ |',
        '|---|---:|---|---|---:|---|'])
    for c in summary['cells']:
        p=c['detection'];ci=p['ci95']
        lines.append(f"| {DATES[c['dataset']]} | {c['activity_mBq_m3']:.4f} | {LABELS[c['family']]} | {p['k']}/{p['n']} | {p['rate']*100:.2f}% | {ci[0]*100:.2f}–{ci[1]*100:.2f}% |")
    lines.extend(['','В этих строках обнаружение проверяется при фиксированной истинной A, а не при случайном MDC данной реализации. Прежняя кривая мощности не пересчитывалась и с новыми данными не объединялась.','',
        '## Формула при известных мешающих параметрах','',
        'Если подставить в процедуру истинную генерирующую смесь, то после удаления свежего Xe133m формульный MDC одинаков при A=0 и при A≈Ld. Он равен:',''])
    for dataset,d in manifest['datasets'].items():
        lines.append(f"- {DATES[dataset]}: **{d['oracle_formula_MDC']*1000:.6f}**; прежний Ld95 ММП–Lc по мощности — **{d['reference_ld']['poisson']['estimate']*1000:.6f}**.")
    lines.extend(['','Совпадение этого формульного значения с исходным MDC ММП обусловлено построением генерирующей смеси из первоначальной ML+ подгонки. Это не независимое подтверждение точности формулы. Разность со средним случайным MDC показывает эффект оценивания мешающих параметров; разность с Ld по мощности также включает отличие формульного приближения от реально работающей процедуры со случайным порогом.','',
        '## Неопределённость эталона и область выводов',''])
    for dataset,d in manifest['datasets'].items():
        for family,ref in d['reference_ld'].items():
            ci=ref['approximate_bootstrap_ci95'];grid=ref['grid_bound95_mBq_m3']
            lines.append(f"- {DATES[dataset]}, {LABELS[family]}: Ld95={ref['estimate']*1000:.6f}; приближённый bootstrap 95% ДИ {ci[0]:.6f}–{ci[1]:.6f}; консервативные сеточные границы {grid[0]:.6f}–{grid[1]:.6f}.")
    lines.extend(['','Эталон получен интерполяцией прежней кривой мощности. Bootstrap отражает численную неопределённость этого приближения и не включает ошибку формы кривой между узлами. В JSON отдельно сохранена чувствительность смещения среднего к обоим видам границ эталона. Из небольших различий с точечным Ld нельзя делать вывод без этой оговорки.','',
        'Это условное параметрическое исследование двух фиксированных смесей, заданных подгонкой реальных спектров. Оно не проверяет правильность R и не включает неопределённость калибровок или генерирующей смеси. Исходные MDC отдельных реальных измерений не позволяют сами по себе судить о систематическом смещении во всей базе.','',
        '## Численная проверка','',
        f"Журнал содержит {len(summary['errors'])} ошибок подгонки/расчёта MDC. Пропуски не заменялись новыми реализациями. Технические 400 реализаций исключены из таблиц.",'',
        'Ковариация для MDC проверяется при построенном нулевом спектре. Отсутствие обычной свободной ковариации ММП на границе не означает отсутствия MDC: это разные расчёты. Проверки воспроизведения seed, непересечения блоков, решения уравнения MDC независимым методом и совпадения с полным вызовом приложения сохранены в verification.json.','',
        '## Файлы','',
        '- [Графики распределений, PDF](mdc_distributions.pdf)\n- [Относительный разброс, PDF](mdc_relative_ranges.pdf)\n- [Таблица результатов, CSV](summary.csv)\n- [Полные численные сводки, JSON](summary.json)\n- [Протокол](protocol_ru.md)\n- [Компактные данные всех реализаций](trials_compact.npz)',''])
    content='\n'.join(lines)
    (ROOT/'report_ru.md').write_text(content,encoding='utf-8')
    (ROOT/'report_ru.txt').write_text(content,encoding='utf-8')


def compact(grouped):
    out={}
    for i,((dataset,scenario),a) in enumerate(grouped.items()):
        n=len(a['indices'])
        part=dict(dataset_index=np.full(n,list(DATES).index(dataset),dtype=np.int8),
                  scenario_index=np.full(n,['null','signal'].index(scenario),dtype=np.int8),
                  index=a['indices'],x=a['x'])
        for f in FAMILIES:
            for name in ('MDC','LC','sigma0','sigma_end','detected','iterations','equation_relative_residual','free_covariance_available'):
                part[f+'_'+name]=a[f+'_'+name]
            for name in ('free','positive'):
                part[f+'_'+name+'_activity']=a[f+'_'+name][:,4]
        for k,v in part.items():out.setdefault(k,[]).append(v)
    np.savez_compressed(ROOT/'trials_compact.npz',**{k:np.concatenate(v) for k,v in out.items()})
    (ROOT/'trials_compact_schema.json').write_text(json.dumps(dict(
        datasets=list(DATES),scenarios=['null','signal'],unit='Bq/m3',
        rows=40000,observations='21 raw integer SAMPLE/GASBK/DETBK counts in model order',
        missing='NaN for unavailable floating results, -1 for unavailable decision',
        note='Physical target index4; full model, truth and seeds in manifest.json. Full fitted vectors remain in local chunks.'),indent=2))


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    grouped,errors=load()
    summary=summarize(manifest,grouped,errors)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    columns=['dataset','scenario','family','n_valid','n_missing','activity_mBq_m3','mean','median','sd','reference_ld','bias','relative_bias_percent','rmse','mc_standard_error','cv_percent']
    with (ROOT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns+['quantile_025','quantile_975','fraction_below_reference','detection_rate'])
        writer.writeheader()
        for c in summary['cells']:
            row={k:c[k] for k in columns}
            row.update(quantile_025=c['central95'][0],quantile_975=c['central95'][1],
                       fraction_below_reference=c['below_reference']['rate'],detection_rate=c['detection']['rate'])
            writer.writerow(row)
    plots(summary,manifest,grouped)
    report(summary,manifest)
    compact(grouped)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
