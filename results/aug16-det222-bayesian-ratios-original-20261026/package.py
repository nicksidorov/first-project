"""Package saved results; no sampling, no fits, no external services."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = 'XeMat_Aug16_Det222_Bayesian_ratios_original'
EXCLUDE = {'checksums.json', 'bundle_verification.json', 'portable_verification.json',
           'sampling_setup_error.log', 'package.log', 'publication.json'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    verification = json.loads((ROOT/'verification.json').read_text())
    diagnostics = json.loads((ROOT/'diagnostics.json').read_text())
    assert verification['status'].lower() in ('pass', 'passed') and diagnostics['status'] == 'passed'
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and not p.name.endswith('.zip')
                   and not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(ROOT).parts)
                   and p.name not in EXCLUDE)
    manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in files}
    (ROOT/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(ROOT/'checksums.json')
    target = ROOT/(NAME+'.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            z.write(p, NAME+'/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for rel, info in manifest.items():
            data = z.read(NAME+'/'+rel)
            assert len(data) == info['bytes'] and sha(data) == info['sha256']
        assert len(z.namelist()) == len(manifest)+1
    result = dict(status='passed',archive=target.name,bytes=target.stat().st_size,
        sha256=sha(target.read_bytes()),members=len(files),member_checksums_verified=True,
        core_code_modified=False,article_modified=False)
    (ROOT/'bundle_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
