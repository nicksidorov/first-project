"""Compact all raw trials and package reviewable research results."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np

ROOT=Path(__file__).resolve().parent


def compact():
    blocks=[];arrays={};offset=0
    for path in sorted((ROOT/'points').glob('*.json')):
        meta=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));n=meta['n']
        values=dict(x=a['x'],row_in_block=a['indices'],
            dataset=np.full(n,['may22','may26'].index(meta['dataset']),dtype=np.int8),
            target=np.full(n,meta['target'],dtype=np.int8))
        cols=np.array(meta['targets'])-1
        for f in ('wls','poisson'):
            values[f+'_free_activities']=a[f+'_free'][:,1:5]
            values[f+'_positive_activities']=a[f+'_positive'][:,1:5]
            values[f+'_LC']=np.full((n,4),np.nan)
            values[f+'_LC'][:,cols]=a[f+'_LC']
            values[f+'_decision']=np.full((n,4),-1,dtype=np.int8)
            values[f+'_decision'][:,cols]=a[f+'_decision']
        for key,v in values.items():arrays.setdefault(key,[]).append(v)
        blocks.append(dict(file=path.name,start=offset,stop=offset+n,**meta));offset+=n
    merged={k:np.concatenate(v) for k,v in arrays.items()}
    for f in ('wls','poisson'):
        good=merged[f+'_decision']>=0
        assert np.array_equal((merged[f+'_free_activities']>merged[f+'_LC'])[good],merged[f+'_decision'][good])
    np.savez_compressed(ROOT/'trials_compact.npz',**merged)
    (ROOT/'trials_compact_index.json').write_text(json.dumps(dict(rows=offset,unit='Bq/m3',
        isotope_columns=['Xe135','Xe133','Xe131m','Xe133m'],dataset_ids=['may22','may26'],
        note='LC/decision columns not evaluated in a signal trial are NaN/-1; use block targets to distinguish from actual missing decisions. Full original fit vectors remain in local point files.',
        blocks=blocks),ensure_ascii=False,indent=2))
    return offset


def main():
    rows=compact();summary=json.loads((ROOT/'summary.json').read_text());assert rows==summary['n_vectors']
    names=['README.md','report_ru.md','report_ru.txt','protocol_ru.md','manifest.json','interval_policy.json',
        'nominal_formula.csv','results.csv','summary.json','nominal_comparison.pdf','nominal_comparison.png',
        'nominal_power.pdf','nominal_power.png','trials_compact.npz','trials_compact_index.json',
        'initial_plan.json','confirmation_plan.json','confirmation_selection.json','validation_plan.json',
        'validation_review.json','verification.json','study.py','analysis_tools.py','report.py','verify.py','package.py']
    for name in ('extension_plan.json','extension_selection.json'):
        if (ROOT/name).exists():names.append(name)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    (ROOT/'sha256.json').write_text(json.dumps(hashes,indent=2));names.append('sha256.json')
    destination=ROOT/'XeMat_nominal_MDC_May.zip'
    with ZipFile(destination,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name in names:archive.write(ROOT/name,name)
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    print('Verified archive',destination.name,'bytes',destination.stat().st_size,'files',len(names),'raw vectors',rows)


if __name__=='__main__':main()
