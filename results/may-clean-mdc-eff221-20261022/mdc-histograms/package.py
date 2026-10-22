"""Package only the explicitly listed, audited MDC histogram experiment.

Uses Python's standard library. Run --verify-only immediately after extraction
to validate the archived payload against checksums and the successful audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE_STEM = 'XeMat_MDC_histograms_eff221_20261022'
CORE_FILES = (
    'README.md', 'requirements.txt', 'package.py',
    'manifest.json', 'plan.json', 'run_lock.json', 'technical_plan.json',
    'technical_review.json', 'completion.json',
    'hist_compute.py', 'analyze_histograms.py', 'verify_histograms.py',
    'summary.json', 'values.npz',
    'compute.log', 'technical.log', 'verification.json', 'verification.log',
    'verification_portable.json', 'verification_portable.log',
)
EXPECTED_AUDIT_COUNTS = {
    'formal_batches': 68, 'poisson_vectors_checked': 68000,
    'reused_null_vectors_checked': 20000, 'new_signal_vectors_checked': 48000,
    'reused_fit_vectors_checked': 80000, 'refit_attempts': 352,
    'refits_confirmed': 352, 'independent_MDC_roots_checked': 416,
    'independent_roots_unavailable': 0, 'compact_arrays_checked': 32,
    'mdc_attempts_checked': 208000, 'valid_MDC': 208000,
    'unavailable_MDC': 0, 'error_records': 0,
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')


def stem(job):
    return '{dataset}_{stage}_i{target}_{family}_c{chunk}'.format(**job)


def validate_audit(name, manifest, full):
    audit = read(name)
    require(audit['status'] == 'passed', name + ': failed audit')
    expected_mode = 'full_source_audit' if full else 'portable_without_old_sources'
    require(audit['source_validation_mode'] == expected_mode, name + ': mode')
    if full:
        require(audit['unavailable_source_checks'] == [], name + ': incomplete source audit')
    else:
        require(len(audit['unavailable_source_checks']) == 3, name + ': source limitations')
    for key, expected in EXPECTED_AUDIT_COUNTS.items():
        if not full and key == 'reused_fit_vectors_checked':
            expected = 0  # Original-source fits are deliberately unavailable here.
        require(audit[key] == expected, f'{name}: {key}')
    for key, filename in (
        ('audit_file_sha256', 'verify_histograms.py'),
        ('manifest_sha256', 'manifest.json'),
        ('run_lock_sha256', 'run_lock.json'),
        ('summary_sha256', 'summary.json'),
        ('values_sha256', 'values.npz'),
    ):
        require(audit[key] == sha(ROOT / filename), f'{name}: changed {filename}')
    require(audit['production_sha256'] == manifest['production_sha256'],
            name + ': production hashes differ')


def validated_payload():
    """Require successful scientific checks and exact planned file membership."""
    files = list(CORE_FILES)
    for name in files:
        require((ROOT / name).is_file(), 'Missing required file: ' + name)
    lock = read('run_lock.json')
    require(set(lock) == {'manifest.json', 'plan.json', 'technical_plan.json', 'hist_compute.py'},
            'Unexpected lock contents')
    for name, expected in lock.items():
        require(sha(ROOT / name) == expected, 'Locked file changed: ' + name)
    manifest = read('manifest.json')
    summary = read('summary.json')
    require(summary['manifest_sha256'] == sha(ROOT / 'manifest.json'), 'summary manifest')
    require(summary['plan_sha256'] == sha(ROOT / 'plan.json'), 'summary plan')
    require(summary['values_sha256'] == sha(ROOT / 'values.npz'), 'summary values')
    require(summary['n_mdc_attempts'] == summary['n_mdc_valid'] == 208000, 'MDC count')
    require(summary['n_mdc_missing'] == 0 and summary['errors'] == [], 'MDC errors')
    require(summary['n_reused_null_vectors'] == 20000, 'Null count')
    require(summary['n_new_signal_vectors'] == 48000, 'Signal count')
    require(summary['technical_vectors_excluded'] == 12, 'Technical count')
    require(summary['mdc_mode'] == 'standard_iterative_alpha_beta_0.05_without_c', 'MDC mode')
    complete = read('completion.json')
    require(complete == {
        'status': 'completed', 'counts': {'null': 20000, 'signal_ld': 48000},
        'batches': 68, 'MDC_values': 208000, 'invalid_MDC': 0,
        'errors': 0, 'locked_hashes_unchanged': True,
    }, 'Completion record differs')
    technical = read('technical_review.json')
    require(technical['status'] == 'passed' and technical['excluded_technical_vectors'] == 12,
            'Technical review did not pass')
    require(technical['fit_errors'] == 0 and technical['Lc_matches_fixed_exactly'],
            'Technical review findings')
    validate_audit('verification.json', manifest, full=True)
    validate_audit('verification_portable.json', manifest, full=False)
    counts = {'null': 0, 'signal_ld': 0, 'technical': 0}
    for plan_name, folder, expected_batches in (
        ('plan.json', 'points', 68), ('technical_plan.json', 'technical', 4),
    ):
        plan = read(plan_name)
        require(len(plan) == expected_batches, 'Plan batch count: ' + plan_name)
        expected_names = set()
        for job in plan:
            name = stem(job)
            meta_name = f'{folder}/{name}.json'
            raw_name = f'{folder}/{name}.npz'
            expected_names.update((name + '.json', name + '.npz'))
            require((ROOT / raw_name).is_file(), 'Missing array: ' + raw_name)
            meta = read(meta_name)
            require(all(meta[key] == value for key, value in job.items()), meta_name + ': job')
            require(meta['manifest_sha256'] == sha(ROOT / 'manifest.json'), meta_name + ': manifest')
            require(meta['npz_sha256'] == sha(ROOT / raw_name), meta_name + ': NPZ changed')
            require(meta['errors'] == [], meta_name + ': errors')
            counts[job['stage']] += job['n']
            if folder == 'points':
                require(summary['source_batches'][name] == {
                    'json': sha(ROOT / meta_name), 'npz': sha(ROOT / raw_name),
                }, meta_name + ': summary source hashes')
            files.extend((meta_name, raw_name))
        actual_names = {path.name for path in (ROOT / folder).iterdir() if path.is_file()}
        require(actual_names == expected_names, 'Unexpected/missing files: ' + folder)
    require(counts == {'null': 20000, 'signal_ld': 48000, 'technical': 12}, 'Vector counts')
    require(len(set(files)) == len(files), 'Duplicate payload names')
    return sorted(files)


def validate_checksums(files):
    recorded = read('checksums.json')
    require(recorded['algorithm'] == 'SHA-256', 'Checksum algorithm')
    actual = {name: sha(ROOT / name) for name in files}
    require(recorded['files'] == actual, 'Payload checksums differ')
    return actual


def create_archive(files):
    payload_hashes = {name: sha(ROOT / name) for name in files}
    write('checksums.json', {
        'algorithm': 'SHA-256',
        'scope': 'Every archive payload file except this checksum index; ZIP checked externally',
        'files': payload_hashes,
    })
    payload_hashes['checksums.json'] = sha(ROOT / 'checksums.json')
    archive = ROOT / (ARCHIVE_STEM + '.zip')
    temporary = ROOT / (ARCHIVE_STEM + '.tmp.zip')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as zipped:
        for name in sorted(payload_hashes):
            info = zipfile.ZipInfo(f'{ARCHIVE_STEM}/{name}', date_time=(2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, (ROOT / name).read_bytes(), compresslevel=6)
    with zipfile.ZipFile(temporary) as zipped:
        expected_members = {f'{ARCHIVE_STEM}/{name}' for name in payload_hashes}
        require(set(zipped.namelist()) == expected_members, 'ZIP member list')
        require(len(zipped.namelist()) == len(expected_members), 'Duplicate ZIP members')
        require(zipped.testzip() is None, 'ZIP CRC failure')
        for name, digest in payload_hashes.items():
            actual = hashlib.sha256(zipped.read(f'{ARCHIVE_STEM}/{name}')).hexdigest()
            require(actual == digest, 'ZIP payload hash: ' + name)
    # Recheck the live payload to detect a concurrent edit during packaging.
    require({name: sha(ROOT / name) for name in payload_hashes} == payload_hashes,
            'Payload changed while packaging')
    os.replace(temporary, archive)
    digest = sha(archive)
    (ROOT / (archive.name + '.sha256')).write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    result = {
        'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'archive': archive.name, 'archive_bytes': archive.stat().st_size,
        'archive_sha256': digest, 'members_checked': len(payload_hashes),
        'payload_files_in_checksum_index': len(files),
        'zip_crc': 'passed', 'all_member_sha256': 'passed',
        'locked_hashes_unchanged': True, 'full_scientific_audit': 'verification.json',
        'scientific_audit_sha256': sha(ROOT / 'verification.json'),
        'checksum_index_sha256': payload_hashes['checksums.json'],
    }
    write('package_verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true',
                        help='Check the extracted payload; do not write anything')
    args = parser.parse_args()
    files = validated_payload()
    if args.verify_only:
        validate_checksums(files)
        print(f'PASS: {len(files)} payload files, full audit, locked inputs and all batch hashes')
    else:
        create_archive(files)


if __name__ == '__main__':
    main()
