"""Build the readable XeMat report ZIP and verify every stored file."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
ARCHIVE='XeMat_May22_clean_MDC_100k_report.zip'
NAMES=['README.md','requirements.txt','REPORT_SCHEMA.md','prepare_report_data.py',
       'build_report.py','render_pdf.py','package_report.py','report_data.json',
       'histogram_values.npz','content.json','report_ru.md','report_ru.txt',
       'report_ru.html','report_ru.pdf','results.csv','histogram_statistics.csv',
       'power_validation.csv']
NAMES += [stem+'.'+ext for stem in ('alpha_standard','alpha_calibrated',
    'power_calibrated','mdc_hist_wls','mdc_hist_poisson') for ext in ('png','pdf')]


def main():
    data=json.loads((ROOT/'report_data.json').read_text(encoding='utf-8'))
    if any(data.get(name,{}).get('status')!='passed' for name in ('verification','histogram_verification')):
        raise ValueError('Cannot package an unaudited report')
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES}
    (ROOT/'sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    destination=ROOT/ARCHIVE
    with ZipFile(destination,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(NAMES+['sha256.json']):archive.write(ROOT/name,name)
    if destination.stat().st_size >= 90 * 1024**2:
        raise ValueError('Report ZIP must be smaller than 90 MiB')
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC verification failed')
        if set(archive.namelist())!=set(NAMES+['sha256.json']):raise ValueError('Unexpected ZIP contents')
        for name,digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('ZIP content mismatch: '+name)
    print(json.dumps(dict(archive=ARCHIVE,bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),files=len(NAMES)+1)))


if __name__=='__main__':main()
