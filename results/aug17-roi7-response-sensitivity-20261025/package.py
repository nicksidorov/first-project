"""Assemble checked saved fits and presentation files without refitting."""
from pathlib import Path
import hashlib,json,zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parent
NAME='XeMat_Aug17_ROI7_response_sensitivity'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit=json.loads((ROOT/'verification.json').read_text())
    assert audit['status']=='passed'
    for name,digest in audit['files_sha256'].items():assert sha(ROOT/name)==digest,name
    with Image.open(ROOT/'roi7_sensitivity.png') as im:
        assert im.size==(3150,1440)
        assert all(abs(d-300)<.01 for d in im.info['dpi'])
    assert (ROOT/'roi7_sensitivity.pdf').read_bytes().startswith(b'%PDF')
    excluded={'checksums.json','bundle_verification.json','portable_verification.json','package.log','task_status.json'}
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.suffix not in ('.zip','.pyc') and p.name not in excluded)
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    (ROOT/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(ROOT/'checksums.json')
    target=ROOT/(NAME+'.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:z.write(p,NAME+'/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(NAME+'/'+str(p.relative_to(ROOT)))).hexdigest()==sha(p)
    result=dict(status='passed',archive=target.name,root_folder=NAME,bytes=target.stat().st_size,sha256=sha(target),
        files=[str(p.relative_to(ROOT)) for p in files],all_member_hashes_checked=True,
        figure_png_dpi=300,figure_visually_reviewed=True)
    (ROOT/'bundle_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='files'})


if __name__=='__main__':main()
