"""Evaluate paired temporal CatBoost feature changes at a fixed blend weight."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from model import combine
from constraints import project_lobt
from validate import compare, verdict
sys.path.insert(0, 'output/experiments/neighbor_proxy')
from normal_correction import apply_normal_correction

parser = argparse.ArgumentParser()
parser.add_argument('splits', nargs='+')
args = parser.parse_args()
out = Path('output/experiments/cat_temporal')
records = []
results = {}
for split in args.splits:
    old = pd.read_parquet(out / ('baseline_' + split + '.parquet'))
    new = pd.read_parquet(out / ('neighbor_' + split + '.parquet'))
    base = pd.read_parquet('output/experiments/neighbor_proxy/residual_member_' + split + '.parquet')
    control = pd.read_parquet('output/experiments/tail_aux_labels/control_' + split + '.parquet')
    context = pd.read_parquet('output/experiments/tail_neighbor/' + split + '.parquet')
    assert all(base.id.equals(f.id) for f in [old, new, control, context])
    assert np.array_equal(base.y, old.y) and np.array_equal(old.y, new.y)
    rome = base.ADEP_mvt.eq('LIRF')
    missing = ~base.has_lobt
    p15 = control.p.copy()
    p15.loc[rome & missing] = .5 * (control.loc[rome & missing, 'p'] + context.loc[rome & missing, 'p'])
    scale = np.where(~missing & base.gap_LOBT_flt.le(6000), .8, 1.)
    result = base.copy()
    result['pred'] = apply_normal_correction(base.pred, base.gap_LOBT_flt,
        base.gap_SCHED_TIME_UTC_mvt, p15, old.mu, old.mu + (new.mu-old.mu)*scale)
    result.to_parquet(out / ('v18_transfer_' + split + '.parquet'), index=False)
    results[split] = compare(split, result, base, 'Cat-neighbor-quarter-transfer-v18-proxy')
    normal = base.has_lobt & base.gap_LOBT_flt.le(6000) & base.y.le(6000)
    records.append({'split': split, 'rows': len(base),
        'base_rmse': float(np.sqrt(np.mean((base.y-base.pred)**2))),
        'candidate_rmse': float(np.sqrt(np.mean((base.y-result.pred)**2))),
        'normal_raw_cat_baseline_rmse': float(np.sqrt(np.mean((base.loc[normal,'y']-old.loc[normal,'mu'])**2))),
        'normal_raw_cat_neighbor_rmse': float(np.sqrt(np.mean((base.loc[normal,'y']-new.loc[normal,'mu'])**2)))})
record = {'comparison': 'Fixed .25 paired weather CatBoost change transferred to corrected no-weather v18 temporal proxy; not an exact production ensemble.',
    'splits': records, 'verdict': verdict(results), 'submitted': False}
(out/'v18_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
