"""Readable new-efficiency-only report; renders audited results without new fits."""
from pathlib import Path
import base64
import csv
import html
import json
import os
import tempfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'xemat-brief-matplotlib'))
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'brief_data.json').read_text(encoding='utf-8'))
DATES = {'may22':'22 мая', 'may26':'26 мая'}
METHODS = {'wls':'МНК', 'poisson':'ММП'}
COLORS = {'wls':'#147b9d', 'poisson':'#c86117'}
ISOTOPES = ['Xe135','Xe133','Xe131m','Xe133m']
CURVES = DATA['curves']
TITLE = 'XeMat: Lc и номинальный MDC при новых эффективностях 221'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.facecolor':'white'})


def num(value, digits=4):
    return f'{value:.{digits}f}'.replace('.', ',')


def ci_text(interval, scale=1, digits=4):
    return '–'.join(num(v*scale,digits) for v in interval)


def rate_text(record):
    return f"{num(100*record['rate'],2)}% [{ci_text(record['ci95'],100,2)}%]"


def ordered(dataset):
    return sorted((c for c in CURVES if c['dataset']==dataset),key=lambda c:c['target'])


def save(fig, name):
    fig.savefig(ROOT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def alpha_figure(rule, name, title):
    fig, axes = plt.subplots(2,2,figsize=(11,7.4),sharex=True)
    for row,(method,label) in enumerate(METHODS.items()):
        for col,(dataset,date) in enumerate(DATES.items()):
            ax=axes[row,col]
            records=[c['methods'][method]['validation'][rule] for c in ordered(dataset)]
            rates=np.array([v['rate']*100 for v in records])
            intervals=np.array([v['ci95'] for v in records])*100
            y=np.arange(4)
            ax.errorbar(rates,y,xerr=[rates-intervals[:,0],intervals[:,1]-rates],
                        fmt='o',color=COLORS[method],markersize=6,capsize=4,lw=1.8)
            for yy,rate in zip(y,rates):
                ax.annotate(num(rate,2)+'%',(rate,yy),xytext=(0,9),
                            textcoords='offset points',ha='center',fontsize=10)
            ax.axvline(5,color='#555555',ls='--',lw=1.3)
            ax.set_yticks(y,ISOTOPES)
            ax.set_ylim(3.5,-.65)
            ax.set_xlim(3.7,7.4)
            ax.set_xticks([4,4.5,5,5.5,6,6.5,7])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:num(v,1)))
            ax.set_title(date+' · '+label,fontweight='bold')
            ax.grid(axis='x',alpha=.17)
            if row==1:ax.set_xlabel('Ложные обнаружения, %')
    fig.suptitle(title,fontsize=17,fontweight='bold',y=.99)
    fig.text(.5,.93,'Истинная активность = 0; в каждом графике одна точка на изотоп',ha='center',fontsize=11)
    fig.text(.5,.018,'Точка: доля обнаружений из 10 000 проб. Полоса: её 95% ДИ. Пунктир: целевые 5%.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.84,bottom=.12,hspace=.35,wspace=.25)
    save(fig,name)


def power_figure(dataset):
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.4),sharex=True,sharey=True)
    for ax,c in zip(axes.flat,ordered(dataset)):
        for f,label in METHODS.items():
            m=c['methods'][f]
            x=np.array([p['factor'] for p in m['formal_points']])
            p=np.array([p['calibrated']['rate']*100 for p in m['formal_points']])
            ci=np.array([p['calibrated']['ci95'] for p in m['formal_points']])*100
            ax.errorbar(x,p,yerr=[p-ci[:,0],ci[:,1]-p],color=COLORS[f],fmt='o-',
                        lw=1.6,capsize=3,markersize=4,label=label)
            ax.plot(m['calibrated_ld_mBq_m3']/c['nominal_MDC_mBq_m3'],95,
                    marker='|',color=COLORS[f],markersize=15,markeredgewidth=2)
        ax.axhline(95,color='#555555',ls=':',lw=1.3)
        ax.axvline(1,color='#999999',ls='--',lw=1)
        ax.set_title(c['isotope'],fontweight='bold')
        ax.set_xlim(.925,1.195); ax.set_ylim(87.5,100)
        ax.set_xticks([.94,1.02,1.10,1.18])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:num(v,2)))
        ax.grid(alpha=.15)
    for ax in axes[1]:ax.set_xlabel('Введённая активность / формульный MDC')
    for ax in axes[:,0]:ax.set_ylabel('Обнаружение, %')
    fig.suptitle(DATES[dataset]+': мощность после подстройки Lc',fontsize=16,fontweight='bold',y=.99)
    fig.text(.5,.93,'Для каждой точки — 3000 проб с одним введённым изотопом',ha='center',fontsize=11)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False)
    fig.text(.5,.005,'Полосы: 95% ДИ мощности. Цветные отметки на уровне 95%: найденный Ld95.',ha='center',fontsize=9.5)
    fig.subplots_adjust(top=.84,bottom=.16,hspace=.32,wspace=.16)
    save(fig,'power_'+dataset)


def content():
    b=[]
    def heading(text):b.append({'type':'heading','text':text})
    def p(text):b.append({'type':'paragraph','text':text})
    def table(headers,rows,widths=None):
        block={'type':'table','headers':headers,'rows':rows}
        if widths:block['widths']=widths
        b.append(block)
    def image(file,caption):b.append({'type':'image','file':file,'caption':caption})
    def page():b.append({'type':'page_break'})

    heading('1. Что именно исследовали')
    p('Чистая проба: истинные активности всех компонентов пробы и газовой памяти равны нулю. '
      'Остаётся детекторный фон. Для исследования обнаружения вводим по одному свежему изотопу Xe. '
      'Рассмотрены условия комплектов «22 мая» и «26 мая»: времена, объёмы, фиксированные ROI и матрицы R детектора 221.')
    table(['Изотоп','ε SAMPLE','ε GASBK'],[[iso,num(a,6),num(g,6)] for iso,a,g in
          zip(DATA['efficiencies']['isotope_order'],DATA['efficiencies']['sample'],DATA['efficiencies']['gasbk'])])
    p('Эффективности ε — доли без branching; коэффициенты ветвления учитываются отдельно. '
      'Живое время SAMPLE / GASBK / DETBK: 14 / 7 / 72 часа. Эффективный объём воздуха: '
      '14,3826 м³ для 22 мая и 13,5674 м³ для 26 мая.')
    p('В каждом повторе разыгрываем все 21 пуассоновский отсчёт SAMPLE/GASBK/DETBK и заново оцениваем '
      'все 17 параметров. Истинное среднее детекторного фона фиксировано по принятому DETBK; '
      'отсчёты нового измерения DETBK случайны, и фон заново подгоняется. Систематические ошибки не включены.')
    p('Основные результаты Lc и Ld95 получены на 156 000 реализаций: 40 000 для настройки порогов, '
      '20 000 для независимой проверки пустой пробы и 96 000 для определения мощности. '
      'Раздел 11 дополнен гистограммами оценок MDC: для них повторно использованы эти 20 000 пустых проб '
      'и рассчитаны 48 000 новых проб при фиксированных A = Ld95. Пороги c и ранее найденные Ld95 не перенастраивались.')
    heading('2. Наше исходное правило обнаружения')
    p('Обнаружение: Â > Lc. Â — свободная оценка активности по МНК или ММП. '
      'Lc в каждой реализации пересчитывается нашей процедурой: неотрицательная вспомогательная подгонка, '
      'удаление целевого свежего вклада из ожидаемого спектра и расчёт порога. '
      'Удаляется вклад из модели ожиданий; наблюдаемые отсчёты при этом не вычитаются.')
    p('При истинной активности A = 0 каждое срабатывание — ложное обнаружение. '
      'Цель 5% означает около 500 ложных обнаружений на 10 000 повторов для одного проверяемого изотопа.')
    page()
    heading('3. Ложные обнаружения по исходному Lc')
    image('alpha_standard.png','Рисунок 1. Только исходное правило Â > Lc; подстройки здесь нет.')
    p('Как читать рисунок: сверху показан МНК, снизу — ММП; слева условия 22 мая, справа — 26 мая. '
      'В каждом небольшом графике на каждый изотоп приходится ровно одна точка.')
    p('Точка — процент ложных обнаружений среди 10 000 пустых проб. Например, Xe133 / МНК / 22 мая: '
      '642 срабатывания из 10 000, то есть 6,42%. Это частота решений, а не активность и не отдельный спектр.')
    p('Горизонтальная полоса — 95% доверительный интервал этой частоты, то есть погрешность самого Монте-Карло. '
      'Для примера 6,42% интервал равен 5,95–6,92%. Пунктир — целевые 5%. '
      'Весь интервал справа от пунктира указывает на превышение уровня; слева — на более консервативное правило. '
      'Пересечение с 5% означает совместимость с этим уровнем с учётом погрешности, но не доказывает точное равенство.')
    p('Методы применялись к одним и тем же наблюдениям. Точки разных методов и изотопов поэтому связаны; '
      'это не 16 независимых экспериментов. Интервалы индивидуальные, без поправки на одновременное рассмотрение всех точек.')
    page()
    heading('4. Числа для исходного Lc')
    table(['Комплект','Изотоп','МНК: частота [95% ДИ]','ММП: частота [95% ДИ]'],
          [[DATES[c['dataset']],c['isotope'],*[rate_text(c['methods'][f]['validation']['standard']) for f in METHODS]] for c in CURVES],
          [.14,.14,.36,.36])
    p('Наиболее заметное превышение — Xe133 при МНК: 6,42% и 6,62%. '
      'При ММП для Xe133 26 мая получено 5,58%; при МНК для Xe133m 22 мая — 5,47%. '
      'Есть и значения достоверно ниже 5% по отдельным интервалам: Xe131m обоими методами 26 мая и Xe133m при ММП 26 мая.')
    heading('5. Как выполняли подстройку к уровню 5%')
    p('На отдельной серии из 20 000 пустых проб для каждого комплекта строили R = Â / Lc. '
      'Для каждого изотопа и метода выбирали множитель c: 19 001-е значение R по возрастанию, '
      'то есть оценку верхнего 5%-го порога. После этого c фиксировали.')
    p('Новое правило: Â > c · Lc. При c > 1 порог повышается, при c < 1 — понижается. '
      'Сам Lc по-прежнему вычисляется заново для каждой реализации. '
      'Проверочные 10 000 пустых проб на комплект не использовались для выбора или повторной настройки c.')
    table(['Комплект','Изотоп','c для МНК','c для ММП'],
          [[DATES[c['dataset']],c['isotope'],*[num(c['methods'][f]['calibration']['c']) for f in METHODS]] for c in CURVES])
    p('Пример Xe133 / МНК / 22 мая: c = 1,0941, то есть каждый рассчитанный Lc повышаем примерно на 9,4%. '
      'Это множитель порога Lc, а не готового MDC: итоговый предел обнаружения нужно заново определить по мощности.')
    page()
    heading('6. Независимая проверка после подстройки')
    image('alpha_calibrated.png','Рисунок 2. Только откалиброванное правило Â > c · Lc. Оси и расположение панелей совпадают с рисунком 1.')
    p('Эти точки рассчитаны на тех же проверочных пробах, что и рисунок 1. '
      'Изменилось правило решения, а не наблюдения. Все 16 частот лежат в диапазоне 4,67–5,18%; '
      'все индивидуальные 95% интервалы включают 5%. Это подтверждает согласие с целевым уровнем '
      'в исследованных условиях с точностью данного Монте-Карло.')
    table(['Комплект','Изотоп','МНК: частота [95% ДИ]','ММП: частота [95% ДИ]'],
          [[DATES[c['dataset']],c['isotope'],*[rate_text(c['methods'][f]['validation']['calibrated']) for f in METHODS]] for c in CURVES],
          [.14,.14,.36,.36])
    page()
    heading('7. Формульные Lc/MDC и MDC исходного правила')
    p('Все дальнейшие активности — мБк/м³. Формульные Lc и MDC рассчитываются по полной информации '
      'Фишера для заданного чистого фона. Формульный Lc ниже — ориентир для этих условий; '
      'в моделируемых пробах он не используется как неизменный порог.')
    p('Численный MDC обозначаем Ld95: вводим фиксированную истинную активность одного изотопа, '
      'много раз разыгрываем отсчёты и применяем правило обнаружения. По кривой мощности находим '
      'активность, при которой срабатывает около 95% проб. Это численная оценка с погрешностью, а не математически точная константа.')
    table(['Комплект','Изотоп','Формула Lc','Формула MDC','Ld95 МНК\nисходный Lc','Ld95 ММП\nисходный Lc'],
          [[DATES[c['dataset']],c['isotope'],num(c['nominal_LC_mBq_m3']),num(c['nominal_MDC_mBq_m3']),
            *[num(c['methods'][f]['standard_ld_mBq_m3']) for f in METHODS]] for c in CURVES],
          [.13,.13,.15,.17,.21,.21])
    p('Численные MDC исходного правила относятся к ложноположительным частотам рисунка 1, '
      'которые не всегда равны 5%. Поэтому для сравнения при согласованном уровне приводим ниже результаты после подстройки.')
    heading('8. Итоговые MDC после подстройки Lc')
    p('В таблице — Ld95 для правила Â > c · Lc. В квадратных скобках — 95% интервалы Монте-Карло '
      'при фиксированном найденном c. У каждого метода собственная подстройка к 5%.')
    table(['Комплект','Изотоп','Формульный MDC','МНК: Ld95 [95% ДИ]','ММП: Ld95 [95% ДИ]'],
          [[DATES[c['dataset']],c['isotope'],num(c['nominal_MDC_mBq_m3']),
            *[num(c['methods'][f]['calibrated_ld_mBq_m3'])+' ['+ci_text(c['methods'][f]['conditional_ci95_mBq_m3'])+']' for f in METHODS]] for c in CURVES],
          [.12,.13,.18,.285,.285])
    p('МНК и ММП дают близкие пределы. Для Xe131m и Xe133m численные MDC после подстройки '
      'примерно на 5–12% выше формульных; для Xe135 и Xe133 разница составляет около 0,4–3,6%. '
      'Сравнение методов следует делать с учётом интервалов, а не только последних цифр.')
    page()
    heading('9. Как получены окончательные MDC: кривые мощности')
    p('Здесь показано только откалиброванное правило. Одна точка соответствует 3000 пробам '
      'при одной заданной истинной активности. По вертикали — процент обнаружений, по горизонтали — '
      'введённая активность, делённая на формульный MDC. Значение 1 соответствует формуле. '
      'Цветные отметки на горизонтали 95% показывают найденный Ld95.')
    image('power_may22.png','Рисунок 3. Условия 22 мая; синий — МНК, оранжевый — ММП. Полосы отражают погрешность оценки мощности при фиксированных c.')
    page()
    image('power_may26.png','Рисунок 4. Условия 26 мая. Множитель c не меняется при увеличении введённой активности; сам Lc вычисляется в каждой реализации заново.')
    p('Например, Xe133 / МНК / 22 мая: формула даёт 0,6507 мБк/м³, исходное правило — '
      '0,6458 мБк/м³ при 6,42% ложных обнаружений, а после подстройки — 0,6688 мБк/м³ '
      'при проверочной частоте 4,85%. Последнее число найдено по мощности, а не умножением формульного MDC на c.')
    heading('10. Неопределённость и границы применимости')
    p('Интервалы итоговой таблицы относятся к фактически выбранным c. Если учитывать также '
      'случайность обучения порога по 20 000 пробам, получаются отдельные интервалы, приведённые ниже. '
      'Они характеризуют оценку Ld95 для идеального порога c₀ — истинного 95-го квантиля нулевой статистики. '
      'В обоих случаях использовано 2000 bootstrap-повторов; ошибка интерполяции между узлами и систематика не включены.')
    table(['Комплект','Изотоп','МНК: 95% ДИ с учётом c','ММП: 95% ДИ с учётом c'],
          [[DATES[c['dataset']],c['isotope'],*[ci_text(c['methods'][f]['total_ci95_mBq_m3']) for f in METHODS]] for c in CURVES],
          [.14,.14,.36,.36])
    p('Для Xe133m 22 мая у МНК 2 из 2000 bootstrap-пересечений оказались выше сетки. '
      'Они сохранены в анализе; 95% интервал остался конечным. Более консервативная верхняя сеточная '
      'граница с учётом неопределённости c для Xe133m 22 мая не определена у обоих методов. '
      'Дополнительные сигнальные точки для её замыкания не рассчитывались.')
    p('Подстройка относится к чистому фону и указанным условиям. Её перенос на произвольные смеси '
      'с памятью или другой фон отдельно не проверялся. Уровень 5% относится к одному изотопу, '
      'а не к вероятности хотя бы одного ложного обнаружения среди четырёх. '
      'Неопределённость выбранного истинного фонового среднего и калибровок не моделировалась.')
    p('В исходном исследовании ошибок подгонки и расчёта Lc нет. В 577 из 156 000 реализаций обычная ковариация свободного '
      'ММП была недоступна на границе ожидаемых отсчётов, однако отдельный расчёт Lc был успешен; '
      'эти реализации не исключались. Вальд в данном исследовании не используется.')
    if (ROOT/'histogram_summary.json').exists():
        from hist_report import render_addendum
        b.extend(render_addendum())
    return {'title':TITLE,'blocks':b}


def write_documents(doc):
    md=['# '+doc['title'],''];plain=[doc['title'],''];body=['<h1>'+html.escape(doc['title'])+'</h1>']
    for b in doc['blocks']:
        t=b['type']
        if t=='page_break':body.append('<div class="page-break"></div>');continue
        if t in ('heading','paragraph'):
            text=b['text'];md += [('## ' if t=='heading' else '')+text,''];plain += [text,'']
            tag='h2' if t=='heading' else 'p';body.append(f'<{tag}>'+html.escape(text)+f'</{tag}>')
        elif t=='table':
            md += ['| '+' | '.join(v.replace('\n','<br>') for v in b['headers'])+' |','|'+'|'.join(['---']*len(b['headers']))+'|']
            md += ['| '+' | '.join(str(v).replace('\n','<br>') for v in r)+' |' for r in b['rows']];md.append('')
            plain.append(' | '.join(b['headers']).replace('\n',' '))
            plain += [' | '.join(str(v).replace('\n',' ') for v in r) for r in b['rows']];plain.append('')
            rows=['<tr>'+''.join('<th>'+html.escape(v).replace('\n','<br>')+'</th>' for v in b['headers'])+'</tr>']
            rows += ['<tr>'+''.join('<td>'+html.escape(str(v)).replace('\n','<br>')+'</td>' for v in r)+'</tr>' for r in b['rows']]
            body.append('<div class="table-wrap"><table>'+''.join(rows)+'</table></div>')
        elif t=='image':
            md += ['!['+b['caption']+']('+b['file']+')','',b['caption'],''];plain += [b['caption']+' Файл: '+b['file'],'']
            data=base64.b64encode((ROOT/b['file']).read_bytes()).decode('ascii')
            body.append('<figure><img src="data:image/png;base64,'+data+'" alt="'+html.escape(b['caption'],quote=True)+'"><figcaption>'+html.escape(b['caption'])+'</figcaption></figure>')
    (ROOT/'report_ru.md').write_text('\n'.join(md),encoding='utf-8')
    (ROOT/'report_ru.txt').write_text('\n'.join(plain),encoding='utf-8')
    css='body{max-width:1050px;margin:35px auto;padding:0 22px;color:#172b3a;font:17px/1.55 system-ui,sans-serif}h1{font-size:29px;line-height:1.25}h2{font-size:23px;margin-top:38px;color:#125b75}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #d5e0e6;text-align:left}th{background:#eaf3f7}tr:nth-child(even){background:#f7fafb}.table-wrap{overflow-x:auto}figure{margin:20px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#465966}@media print{body{font-size:11pt;margin:0}h2{break-after:avoid}table{font-size:8pt}.page-break{break-before:page}figure{break-inside:avoid}}'
    (ROOT/'report_ru.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(TITLE)+'</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>',encoding='utf-8')


def write_csv():
    with (ROOT/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream,delimiter=';')
        w.writerow(['Комплект','Изотоп','Метод','Число проверочных проб','Ложных обнаружений Lc','Частота Lc, %','95% ДИ Lc, %','c','Ложных обнаружений cLc','Частота cLc, %','95% ДИ cLc, %','Формульный Lc, мБк/м3','Формульный MDC, мБк/м3','Ld95 исходный, мБк/м3','95% ДИ исходного Ld95','Ld95 после подстройки, мБк/м3','95% ДИ при фиксированном c','95% ДИ с учётом c'])
        for c in CURVES:
            for f,label in METHODS.items():
                m=c['methods'][f];a=m['validation']['standard'];v=m['validation']['calibrated']
                w.writerow([DATES[c['dataset']],c['isotope'],label,a['n'],a['k'],num(100*a['rate'],2),ci_text(a['ci95'],100,2),num(m['calibration']['c']),v['k'],num(100*v['rate'],2),ci_text(v['ci95'],100,2),num(c['nominal_LC_mBq_m3']),num(c['nominal_MDC_mBq_m3']),num(m['standard_ld_mBq_m3']),ci_text(m['standard_ci95_mBq_m3']),num(m['calibrated_ld_mBq_m3']),ci_text(m['conditional_ci95_mBq_m3']),ci_text(m['total_ci95_mBq_m3'])])


if __name__=='__main__':
    assert len(CURVES)==8 and not DATA['errors']
    assert sum(DATA['n_by_stage'].values())==156000
    alpha_figure('standard','alpha_standard','Наше правило Lc: проверка уровня 5%')
    alpha_figure('calibrated','alpha_calibrated','После подстройки c · Lc: проверка уровня 5%')
    for dataset in DATES:power_figure(dataset)
    doc=content();(ROOT/'content.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    write_documents(doc);write_csv()
    print('Rendered new-efficiency-only figures, report and CSV; no new Monte Carlo.')
