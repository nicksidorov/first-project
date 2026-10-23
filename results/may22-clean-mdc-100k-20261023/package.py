"""Package the audited study as ordinary ZIP files, each smaller than 90 MiB.

The metadata ZIP and all raw parts extract into one common directory. This
script uses only the Python standard library; it does not simulate or fit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
PREFIX = 'XeMat_May22_clean_MDC_100k'
LIMIT = 90 * 1024**2
TARGET_RAW_BYTES = 80 * 1024**2  # Space for ZIP overhead and incompressible data.
CORE = (
    'README.md', 'requirements.txt', 'package.py',
    'manifest.json', 'run_lock.json', 'statistical_design.md',
    'statistical_design_review.md', 'prepare.py', 'study.py',
    'calibration.py', 'analyze.py', 'histogram_analysis.py', 'plan_followups.py',
    'audit_design.py', 'design_review.json', 'verify_main.py', 'verify_histograms.py',
    'technical_plan.json', 'technical_review.json',
    'calibration_plan.json', 'validation_plan.json', 'signal_plan.json',
    'histogram_plan.json', 'histogram_plan.lock.json',
    'calibration.json', 'calibration_bootstrap.npz', 'cutoff.json',
    'summary.json', 'analysis_bootstrap.npz',
    'histogram_summary.json', 'histogram_values.npz',
    'verification.json', 'histogram_verification.json',
)
REPORT = (
    'README.md', 'requirements.txt', 'REPORT_SCHEMA.md', 'prepare_report_data.py',
    'build_report.py', 'render_pdf.py', 'package_report.py',
    'report_data.json', 'histogram_values.npz', 'content.json',
    'report_ru.md', 'report_ru.txt', 'report_ru.html', 'report_ru.pdf',
    'results.csv', 'histogram_statistics.csv', 'power_validation.csv', 'sha256.json',
) + tuple(f'{figure}.{extension}' for figure in (
    'alpha_standard', 'alpha_calibrated', 'power_calibrated',
    'mdc_hist_wls', 'mdc_hist_poisson') for extension in ('png', 'pdf'))
AUDITS = {
    'verification.json': {
        'manifest_sha256': 'manifest.json', 'run_lock_sha256': 'run_lock.json',
        'summary_sha256': 'summary.json', 'calibration_sha256': 'calibration.json',
        'cutoff_sha256': 'cutoff.json', 'analysis_bootstrap_sha256': 'analysis_bootstrap.npz',
        'audit_file_sha256': 'verify_main.py',
    },
    'histogram_verification.json': {
        'manifest_sha256': 'manifest.json', 'run_lock_sha256': 'run_lock.json',
        'summary_sha256': 'summary.json', 'histogram_summary_sha256': 'histogram_summary.json',
        'histogram_values_sha256': 'histogram_values.npz',
        'histogram_plan_sha256': 'histogram_plan.json', 'audit_file_sha256': 'verify_histograms.py',
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def safe_file(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe path: ' + name)
    path = root.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink(), 'Missing/nonregular file: ' + name)
    path.resolve().relative_to(root.resolve())
    return path


def write_json(path, value):
    Path(path).write_bytes(encoded(value))


def stem(job):
    result = f"{job['dataset']}_{job['stage']}_i{job['target']}_f{round(job['factor']*10000):05d}_c{job['chunk']}"
    return result + '_' + job['family'] if job['stage'] == 'hist_signal' else result


def validate_payload(root):
    """Check successful audits, frozen inputs, complete plans and raw file hashes."""
    metadata = list(CORE) + ['report/' + name for name in REPORT]
    metadata += [p.name for p in sorted(root.glob('extension_*.json'))]
    metadata += [p.name for p in sorted(root.glob('*.log'))]
    for optional in ('completion.json', 'analysis_review.json',
                     'verification_inputs.json', 'verification_calibration.json'):
        if (root/optional).exists():
            metadata.append(optional)
    require(len(metadata) == len(set(metadata)), 'Duplicate metadata selection')
    for name in metadata:
        safe_file(root, name)
    locked = read(root/'run_lock.json')
    for name, digest in locked.items():
        require(sha(safe_file(root, name)) == digest, 'Locked file changed: ' + name)
    manifest = read(root/'manifest.json')
    require(list(manifest['datasets']) == ['may22'], 'Expected one May 22 configuration')
    require(manifest['calibration_n_per_dataset'] == manifest['validation_n_per_dataset'] == 100000,
            'Wrong planned null sample size')
    require(manifest['signal_n_per_point'] == 20000, 'Wrong power sample size')
    for name, fields in AUDITS.items():
        audit = read(root/name)
        require(audit.get('status') == 'passed', name + ': audit has not passed')
        for field, source in fields.items():
            require(audit.get(field) == sha(root/source), name + ': changed ' + source)
    # Tie each packaged formal batch to the bytes inspected by the audits.
    # A JSON/NPZ pair that agrees internally may still have changed after audit.
    audited_batches = dict(read(root/'verification.json')['source_batches'])
    for name, digests in read(root/'histogram_summary.json')['source_batches'].items():
        if name in audited_batches:
            require(audited_batches[name] == digests, 'Audits disagree on batch: ' + name)
        audited_batches[name] = digests
    for name in ('design_review.json', 'technical_review.json'):
        require(read(root/name).get('status') == 'passed', name + ': review did not pass')
    cutoff = read(root/'cutoff.json')
    require(cutoff['frozen'] and cutoff['validation_not_used'], 'Thresholds not independently frozen')
    require(cutoff['calibration_sha256'] == sha(root/'calibration.json'), 'Calibration changed')
    require(cutoff['bootstrap_sha256'] == sha(root/'calibration_bootstrap.npz'), 'Calibration bootstrap changed')
    hist_lock = read(root/'histogram_plan.lock.json')
    for key, source in [('plan_sha256', 'histogram_plan.json'), ('manifest_sha256', 'manifest.json'),
                        ('summary_sha256', 'summary.json'), ('cutoff_sha256', 'cutoff.json')]:
        require(hist_lock[key] == sha(root/source), 'Histogram lock: ' + source)
    report_data = read(root/'report/report_data.json')
    for name in ('manifest.json', 'summary.json', 'histogram_summary.json',
                 'histogram_values.npz', 'verification.json', 'histogram_verification.json'):
        require(report_data['source_hashes'].get(name) == sha(root/name), 'Report source: ' + name)
    require(sha(root/'report/histogram_values.npz') == sha(root/'histogram_values.npz'), 'Report values differ')
    require((root/'report/report_ru.pdf').read_bytes().startswith(b'%PDF-'), 'Report PDF missing/invalid')
    for name, digest in read(root/'report/sha256.json').items():
        require(sha(safe_file(root/'report', name)) == digest, 'Report checksum: ' + name)

    plans = [('calibration_plan.json', 'points'), ('validation_plan.json', 'points'),
             ('signal_plan.json', 'points'), ('histogram_plan.json', 'points'),
             ('technical_plan.json', 'technical')]
    if (root/'extension_plan.json').exists():
        plans.append(('extension_plan.json', 'points'))
    groups, seen, counts, formal_batches = [], set(), {}, set()
    expected_by_folder = {'points': set(), 'technical': set()}
    for plan_name, folder in plans:
        for job in read(root/plan_name):
            name = stem(job)
            require(name not in seen, 'Repeated batch: ' + name)
            seen.add(name)
            paths = [f'{folder}/{name}.json', f'{folder}/{name}.npz']
            for path in paths:
                safe_file(root, path)
                expected_by_folder[folder].add(Path(path).name)
            meta = read(root/paths[0])
            require(all(meta[k] == value for k, value in job.items()), 'Batch differs from plan: ' + name)
            require(meta['manifest_sha256'] == sha(root/'manifest.json'), 'Batch manifest: ' + name)
            require(meta['npz_sha256'] == sha(root/paths[1]), 'Raw NPZ changed: ' + name)
            if folder == 'points':
                formal_batches.add(name)
                current = {'json': sha(root/paths[0]), 'npz': meta['npz_sha256']}
                require(audited_batches.get(name) == current, 'Batch changed after audit: ' + name)
            counts[job['stage']] = counts.get(job['stage'], 0) + job['n']
            groups.append(dict(files=paths, stage=job['stage'], n=job['n'], batch=name))
    require(formal_batches == set(audited_batches), 'Formal plans and audited batches differ')
    for folder, expected in expected_by_folder.items():
        actual = {p.name for p in (root/folder).iterdir() if p.is_file()}
        require(actual == expected, 'Unplanned/missing files in ' + folder)
    require(counts.get('calibration') == counts.get('validation') == 100000, 'Incomplete null plans')
    require(counts.get('signal') == 320000 and counts.get('hist_signal') == 160000, 'Incomplete signal plans')
    require(counts.get('technical') == manifest['technical_vectors_excluded'], 'Technical count')
    summary = read(root/'summary.json')
    for key, stage in [('calibration', 'calibration'), ('validation', 'validation'),
                       ('signal', 'signal'), ('signal_extension', 'extend')]:
        require(summary['n_new_by_stage'].get(key, 0) == counts.get(stage, 0), 'Summary count: ' + key)
    hist = read(root/'histogram_summary.json')
    require(hist['n_reused_null_vectors'] == 100000 and hist['n_new_signal_vectors'] == 160000, 'Histogram vectors')
    require(hist['n_mdc_attempts'] == 960000, 'Histogram attempt count')
    require(hist['n_mdc_valid'] + hist['n_mdc_missing'] == hist['n_mdc_attempts'], 'Histogram missing accounting')
    require(hist['values_sha256'] == sha(root/'histogram_values.npz'), 'Histogram values hash')
    return sorted(metadata), groups, counts


def split_groups(root, groups, target=TARGET_RAW_BYTES):
    """Keep each metadata/NPZ pair together; estimate conservatively before compression."""
    parts, current, size = [], [], 0
    for group in groups:
        length = sum((root/name).stat().st_size for name in group['files'])
        require(length < target, 'A single batch exceeds the part budget: ' + group['batch'])
        if current and size + length > target:
            parts.append(current); current, size = [], 0
        current.append(group); size += length
    if current:
        parts.append(current)
    return parts


def file_records(root, names):
    return {name: dict(bytes=(root/name).stat().st_size, sha256=sha(root/name)) for name in names}


def zip_write(path, root, names, part_name, records):
    part_record = encoded(dict(part=part_name, prefix=PREFIX, algorithm='SHA-256',
                               files={name: records[name] for name in names}))
    internal = f'package_parts/{part_name}.json'
    with ZipFile(path, 'w', ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zipped:
        for name in sorted(names) + [internal]:
            info = ZipInfo(f'{PREFIX}/{name}', date_time=(2026, 10, 23, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            payload = part_record if name == internal else (root/name).read_bytes()
            zipped.writestr(info, payload, compresslevel=6)
    require(path.stat().st_size < LIMIT, 'ZIP is not smaller than 90 MiB: ' + path.name)
    with ZipFile(path) as zipped:
        expected = {f'{PREFIX}/{name}' for name in names + [internal]}
        require(len(zipped.namelist()) == len(expected) and set(zipped.namelist()) == expected, 'ZIP members')
        require(zipped.testzip() is None, 'ZIP CRC failure: ' + path.name)
        for name in names:
            require(hashlib.sha256(zipped.read(f'{PREFIX}/{name}')).hexdigest() == records[name]['sha256'],
                    'ZIP member hash: ' + name)
        require(zipped.read(f'{PREFIX}/{internal}') == part_record, 'Part manifest differs')
    return dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path),
                payload_files=len(names), part_manifest=internal, kind=part_name)


def create_package(root):
    metadata, groups, counts = validate_payload(root)
    raw_names = [name for group in groups for name in group['files']]
    require(len(set(metadata+raw_names)) == len(metadata+raw_names), 'Duplicate payload name')
    records = file_records(root, metadata+raw_names)
    parts = split_groups(root, groups)
    archives = []
    # Use a temporary sibling directory: completed archives replace their names only after verification.
    with tempfile.TemporaryDirectory(prefix='.package-', dir=root) as temporary:
        temporary = Path(temporary)
        for number, part in enumerate(parts, start=1):
            label = f'raw_{number:02d}'
            names = [name for group in part for name in group['files']]
            result = zip_write(temporary/f'{PREFIX}_{label}.zip', root, names, label, records)
            result['vectors_by_stage'] = {stage: sum(g['n'] for g in part if g['stage']==stage)
                                          for stage in sorted({g['stage'] for g in part})}
            archives.append(result)
        lines = ['# Части полных данных XeMat', '',
                 'Для чтения отчёта эти архивы не нужны. Для воспроизведения распакуйте metadata и **все** части данных в одну папку.', '',
                 '| Архив | Размер, MiB | SHA-256 |', '|---|---:|---|']
        for archive in archives:
            lines.append(f"| [{archive['name']}]({archive['name']}) | {archive['bytes']/1024**2:.2f} | `{archive['sha256']}` |")
        lines += ['', 'Каждая часть — самостоятельный ZIP, а не фрагмент двоичного файла. '
                  'Их общая корневая папка: `'+PREFIX+'`.', '']
        (root/'DATA_ARCHIVES.md').write_text('\n'.join(lines), encoding='utf-8')
        write_json(root/'data_archives.json', dict(prefix=PREFIX, archives=archives, counts=counts))
        metadata += ['DATA_ARCHIVES.md', 'data_archives.json']
        records.update(file_records(root, metadata[-2:]))
        write_json(root/'checksums.json', dict(algorithm='SHA-256', files=records,
            note='Payload files; this checksum index and internal part manifests are verified separately.'))
        metadata.append('checksums.json')
        all_records = {**records, **file_records(root, ['checksums.json'])}
        archives.insert(0, zip_write(temporary/f'{PREFIX}_metadata.zip', root, metadata, 'metadata', all_records))
        require(file_records(root, list(records)) == records, 'Source changed during packaging')
        require(read(root/'checksums.json')['files'] == records, 'Checksum index changed')
        for archive in archives:
            os.replace(temporary/archive['name'], root/archive['name'])
    index = dict(schema_version=1, prefix=PREFIX, archive_limit_bytes=LIMIT,
                 counts=counts, checksum_index_sha256=sha(root/'checksums.json'), archives=archives)
    write_json(root/'package_index.json', index)
    for archive in archives:
        (root/(archive['name']+'.sha256')).write_text(f"{archive['sha256']}  {archive['name']}\n", encoding='utf-8')
    result = dict(status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  zip_count=len(archives), payload_files=len(records), all_archives_below_limit=True,
                  crc='passed', member_sha256='passed', sources_unchanged=True,
                  package_index_sha256=sha(root/'package_index.json'),
                  verification_sha256=sha(root/'verification.json'),
                  histogram_verification_sha256=sha(root/'histogram_verification.json'))
    write_json(root/'package_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def verify_extracted(root):
    metadata, groups, counts = validate_payload(root)
    record = read(root/'checksums.json')
    require(record['algorithm'] == 'SHA-256', 'Wrong checksum algorithm')
    expected = set(metadata + [name for group in groups for name in group['files']] +
                   ['DATA_ARCHIVES.md', 'data_archives.json'])
    require(set(record['files']) == expected, 'Checksum file membership differs')
    require(file_records(root, sorted(expected)) == record['files'], 'Extracted payload checksums differ')
    require(read(root/'data_archives.json')['counts'] == counts, 'Data archive count index differs')
    print(f'PASS: {len(expected)} extracted payload files, both scientific audits, all planned batches.')


def verify_archives(directory):
    index = read(directory/'package_index.json')
    require(index['prefix'] == PREFIX and index['archive_limit_bytes'] == LIMIT, 'Archive index specification')
    for archive in index['archives']:
        path = safe_file(directory, archive['name'])
        require(path.stat().st_size == archive['bytes'] < LIMIT, 'Archive size: ' + path.name)
        require(sha(path) == archive['sha256'], 'Archive SHA-256: ' + path.name)
        with ZipFile(path) as zipped:
            require(zipped.testzip() is None, 'Archive CRC: ' + path.name)
            part_name = f"{PREFIX}/{archive['part_manifest']}"
            part = json.loads(zipped.read(part_name))
            expected = {f'{PREFIX}/{name}' for name in part['files']} | {part_name}
            require(set(zipped.namelist()) == expected and len(zipped.namelist()) == len(expected), 'Archive members')
            for name, record in part['files'].items():
                payload = zipped.read(f'{PREFIX}/{name}')
                require(len(payload) == record['bytes'] and hashlib.sha256(payload).hexdigest() == record['sha256'],
                        'Archive member checksum: ' + name)
        print('PASS:', archive['name'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true', help='Verify all extracted files without changing them')
    mode.add_argument('--verify-archives', action='store_true', help='Verify downloaded ZIPs using package_index.json')
    parser.add_argument('--archives-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.verify_archives:
        verify_archives(args.archives_dir.resolve())
    elif args.verify_only:
        verify_extracted(ROOT)
    else:
        create_package(ROOT)


if __name__ == '__main__':
    main()
