"""Package the checked measurement and figures; no fitting or simulation."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = 'XeMat_Aug17_Xe133_Xe131m_ellipse'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for name in ('verification.json', 'figure_verification.json'):
        audit = json.loads((ROOT/name).read_text())
        assert audit['status']=='passed', name
        for path,digest in audit['files_sha256'].items():
            assert sha(ROOT/path)==digest, path
    excluded={'checksums.json','bundle_verification.json','portable_verification.json','package.log'}
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.suffix not in ('.zip','.pyc') and p.name not in excluded)
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    (ROOT/'checksums.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    files.append(ROOT/'checksums.json')
    destination=ROOT/(NAME+'.zip')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files:archive.write(path,NAME+'/'+str(path.relative_to(ROOT)))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for path in files:
            assert hashlib.sha256(archive.read(NAME+'/'+str(path.relative_to(ROOT)))).hexdigest()==sha(path)
    verification=dict(status='passed',archive=destination.name,root_folder=NAME,
        bytes=destination.stat().st_size,sha256=sha(destination),
        files=[str(p.relative_to(ROOT)) for p in files],all_member_hashes_checked=True)
    (ROOT/'bundle_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print({k:v for k,v in verification.items() if k!='files'})


if __name__ == '__main__':
    main()
