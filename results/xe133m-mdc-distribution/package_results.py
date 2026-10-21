"""Create a portable research bundle, excluding large redundant fit chunks."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
NAMES=['README.md','report_ru.md','report_ru.txt','protocol_ru.md','manifest.json',
       'summary.json','summary.csv','trials_compact.npz','trials_compact_schema.json',
       'mdc_distributions.pdf','mdc_distributions.png','mdc_relative_ranges.pdf','mdc_relative_ranges.png',
       'previous_power_summary.json','verification.json','validation_review.json',
       'run_study.py','summarize.py','verify_results.py','package_results.py',
       'review_summary.py','summary_review.json']
for name in NAMES:
    assert (ROOT/name).is_file(),name
checksums={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NAMES}
(ROOT/'sha256.json').write_text(json.dumps(checksums,indent=2))
destination=ROOT/'XeMat_Xe133m_MDC_distribution_May.zip'
with ZipFile(destination,'w',ZIP_DEFLATED,compresslevel=6) as archive:
    for name in NAMES+['sha256.json']:
        archive.write(ROOT/name,name)
with ZipFile(destination) as archive:
    assert archive.testzip() is None
    for name,digest in checksums.items():
        assert hashlib.sha256(archive.read(name)).hexdigest()==digest
print(f'Archive verified: {destination.name}, {destination.stat().st_size} bytes, {len(NAMES)+1} files')
