"""Compute the four ordinary fits for the new April pair at original Rs=0.34.

No August results or times are reused. This is the same unmodified workflow
as the GUI, including its standard activity, covariance and LC/MDC outputs.
No Monte Carlo and no ellipses are computed.
"""
from pathlib import Path
import hashlib
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'kernel_snapshot'))
import numpy as np
import scipy
from constants import FIXED_ROI_TABLE, ROI_INFO
from process_measurement import process_prepared_measurement, METHOD_NAMES
from results_saver import format_results
from xemat.io import jsonable, save_json

INPUTS=dict(sample_file='SPIKE20260425010733595.PHD',background_file='GASBK20260424152856639.PHD',
            detector_file='DETBK20250610213044796.PHD')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    output=ROOT/'baseline';output.mkdir(exist_ok=True)
    previous=None
    for method in METHOD_NAMES:
        result=process_prepared_measurement(
            **{key:str(ROOT/'inputs'/name) for key,name in INPUTS.items()},
            method=method,use_fixed_roi=True,use_fixed_efficiencies=True,radon_mode='legacy')
        assert result['fit'].success and result['metadata']['input_detector_id']==221
        assert result['calibration_matrices']['R_c0'][5,2]==.34
        if previous is not None:
            for key in ('M','x','b','T_c'):
                np.testing.assert_array_equal(getattr(result['model'],key),getattr(previous,key))
        previous=result['model']
        save_json(result,output/f'result_{method}.json')
        (output/f'result_{method}.txt').write_text(format_results(result),encoding='utf-8')
        print(method,'success',result['fit'].success,'reduced criterion',result['fit_quality'].reduced_statistic,
              'concentrations mBq/m3',1000*result['activities'].a_c[1:],flush=True)
    model=result['model']
    np.savez_compressed(output/'model_arrays.npz',M=model.M,x=model.x,b=model.b,T_V=model.T_c,
        physical_matrix=model.physical_matrix,**{key:np.asarray(v) for key,v in result['calibration_matrices'].items()})
    source=dict(production_commit='69ff5a62ccaaddeb6372f7f003109ce5ff6cf333',
        input_files={key:dict(file=name,sha256=sha(ROOT/'inputs'/name)) for key,name in INPUTS.items()},
        kernel_sha256={str(p.relative_to(ROOT/'kernel_snapshot')):sha(p) for p in sorted((ROOT/'kernel_snapshot').rglob('*.py'))},
        fixed_roi=True,fixed_efficiencies=True,radon_mode='legacy',detector=221,
        response_model='Original detector 221 Rs/Rg; Rs[5,2]=0.34',
        metadata=jsonable(result['metadata']),fixed_roi_bounds=FIXED_ROI_TABLE,
        branching={key:v['br'] for key,v in ROI_INFO.items()},concentration_units='mBq/m3',
        isotope_order=['Rn222','Xe135','Xe133','Xe131m','Xe133m'],
        sample_header_status='PREL',sample_data_type='SPIKEPHD',
        baseline_origin='Computed for the current April pair by prepare_baseline.py',
        n_current_baseline_fits=4,no_monte_carlo=True,no_ellipses=True,
        software=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__),
        preparation_script_sha256=sha(__file__))
    save_json(source,output/'scenario.json')


if __name__=='__main__':main()
