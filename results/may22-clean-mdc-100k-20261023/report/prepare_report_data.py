"""Assemble presentation inputs from the completed new study, without refitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT.parent)
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();source=args.study.resolve();output=args.output.resolve()
    manifest=read(source/'manifest.json');summary=read(source/'summary.json')
    hist=read(source/'histogram_summary.json');verification=read(source/'verification.json')
    hist_verification=read(source/'histogram_verification.json')
    if verification.get('status')!='passed' or hist_verification.get('status')!='passed':
        raise ValueError('Independent audit has not passed; no final report data will be created')
    for audit,checks in [
        (verification,{'manifest_sha256':'manifest.json','run_lock_sha256':'run_lock.json',
                       'summary_sha256':'summary.json','calibration_sha256':'calibration.json',
                       'cutoff_sha256':'cutoff.json','analysis_bootstrap_sha256':'analysis_bootstrap.npz'}),
        (hist_verification,{'manifest_sha256':'manifest.json','run_lock_sha256':'run_lock.json',
                            'summary_sha256':'summary.json','histogram_summary_sha256':'histogram_summary.json',
                            'histogram_values_sha256':'histogram_values.npz','histogram_plan_sha256':'histogram_plan.json'}),
    ]:
        for field,name in checks.items():
            if audit[field]!=sha(source/name):
                raise ValueError('Audit hash does not match '+name)
    if set(manifest['datasets'])!={'may22'} or len(summary['curves'])!=4:
        raise ValueError('Expected only the new may22 study')
    if manifest['calibration_n_per_dataset']!=100000 or manifest['validation_n_per_dataset']!=100000:
        raise ValueError('Wrong study sample size')
    if hist['values_sha256']!=sha(source/'histogram_values.npz'):
        raise ValueError('Histogram values do not match the audited summary')
    # Keep the power summary immutable: its digest locks the later A=Ld95 series.
    counts=dict(summary['n_new_by_stage'])
    counts.setdefault('signal_extension',0)
    counts['hist_signal']=hist['n_new_signal_vectors']
    if counts['calibration']!=100000 or counts['validation']!=100000 or counts['hist_signal']!=160000:
        raise ValueError('Completed observation counts differ from the report plan')
    meta=manifest['datasets']['may22']['metadata']
    inputs=('manifest.json','summary.json','histogram_summary.json','histogram_values.npz',
            'verification.json','histogram_verification.json')
    data=dict(dataset='may22',verification=verification,histogram_verification=hist_verification,
        efficiencies=manifest['efficiency_update'],
        conditions=dict(times=meta['time_parameters'],effective_air_volume_m3=meta['effective_air_volume_m3']),
        n_by_stage=counts,n_signal_per_point=manifest['signal_n_per_point'],
        bootstrap_repetitions=manifest['bootstrap']['repetitions'],
        technical_vectors_excluded=manifest['technical_vectors_excluded'],
        curves=summary['curves'],histograms=hist,
        errors=summary['errors'],histogram_errors=hist.get('errors',[]),
        covariance_diagnostics=summary.get('covariance_diagnostics',[]),
        source_hashes={name:sha(source/name) for name in inputs},
        production_commit=manifest['production_commit'],systematics=False,wald_used=False)
    output.mkdir(exist_ok=True,parents=True)
    shutil.copy2(source/'histogram_values.npz',output/'histogram_values.npz')
    (output/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Prepared report_data.json and histogram_values.npz from the new audited study only.')


if __name__=='__main__':main()
