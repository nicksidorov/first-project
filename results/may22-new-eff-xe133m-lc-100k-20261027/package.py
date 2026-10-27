"""Archive saved Monte Carlo results and code; no fits or simulations."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
NAME='XeMat_May22_Xe133m_LC_new_eff_100k'
EXCLUDE={'checksums.json','bundle_verification.json','portable_verification.json',
         'package.log','publication.json','audit_partial.json'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    verification=json.loads((ROOT/'audit_result.json').read_text())
    assert verification['status'].lower() in ('pass','passed')
    assert verification['mode']=='complete' and verification['n_rows']==100000
    assert verification['replicates_sha256']==digest((ROOT/'replicates.npz').read_bytes())
    assert verification['summary_sha256']==digest((ROOT/'summary.csv').read_bytes())
    rng_verification=json.loads((ROOT/'rng_verification.json').read_text())
    assert rng_verification['status']=='PASS' and rng_verification['n_rows_verified']==100000
    assert rng_verification['replicates_sha256']==verification['replicates_sha256']
    scenario=json.loads((ROOT/'scenario.json').read_text())
    assert scenario['n_generated']==100000 and scenario['target_index']==4
    assert scenario['no_mdc_or_ld_calculation']
    assert (ROOT/'replicates.npz').is_file() and (ROOT/'summary.csv').is_file()
    assert len(list((ROOT/'batch_metadata').glob('batch_*.json')))==100
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.zip')
        and p.name not in EXCLUDE and 'points' not in p.relative_to(ROOT).parts
        and not any(part.startswith('.') or part=='__pycache__' for part in p.relative_to(ROOT).parts))
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in files}
    (ROOT/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(ROOT/'checksums.json')
    target=ROOT/(NAME+'.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:z.write(p,NAME+'/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for rel,info in manifest.items():
            data=z.read(NAME+'/'+rel)
            assert len(data)==info['bytes'] and digest(data)==info['sha256']
        assert len(z.namelist())==len(manifest)+1
    record=dict(status='passed',archive=target.name,bytes=target.stat().st_size,
        sha256=digest(target.read_bytes()),member_count=len(files),all_member_checksums_verified=True,
        raw_batch_npz_omitted='All rows and fields are retained in merged replicates.npz; metadata retained for each batch',
        no_MDC_or_LD_calculated=True,production_code_unchanged=True,article_unchanged=True)
    (ROOT/'bundle_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
