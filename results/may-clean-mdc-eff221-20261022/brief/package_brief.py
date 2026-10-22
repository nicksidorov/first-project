"""Package only the new-efficiency brief, checking every archived byte."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
NAMES=['README.md','requirements.txt','brief_data.json','content.json',
       'build_brief.py','render_pdf.py','package_brief.py',
       'report_ru.md','report_ru.txt','report_ru.html','report_ru.pdf','results.csv',
       'hist_report.py','histogram_summary.json','histogram_values.npz',
       'histogram_statistics.csv','hist_blocks.json','histogram_verification.json']
NAMES += [name+'.'+ext for name in ('alpha_standard','alpha_calibrated','power_may22','power_may26') for ext in ('png','pdf')]
NAMES += [f'mdc_hist_{date}_{family}.{ext}' for date in ('may22','may26')
          for family in ('wls','poisson') for ext in ('png','pdf')]


def main():
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES}
    (ROOT/'sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    destination=ROOT/'XeMat_new_efficiencies_brief.zip'
    with ZipFile(destination,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(NAMES+['sha256.json']):z.write(ROOT/name,name)
    with ZipFile(destination) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(NAMES+['sha256.json'])
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    print(json.dumps({'archive':str(destination),'bytes':destination.stat().st_size,
        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'files':len(NAMES)+1}))


if __name__=='__main__':main()
