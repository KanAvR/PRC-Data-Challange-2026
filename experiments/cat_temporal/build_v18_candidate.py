"""Build and independently audit the fixed CatBoost correction to immutable v18."""
import sys, json, hashlib, subprocess
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from featcache import _cache_path
sys.path.insert(0, 'output/experiments/neighbor_proxy')
from normal_correction import apply_normal_correction

source = Path('output/submitting_v18.parquet')
expected = 'dc02ea9dad7dd15ce703f4ef192b89d1ad36ac1dc39fd0e0a05532f364db6b58'
out = Path('output/experiments/cat_temporal')
dest = Path('output/submitting_v18_cat_neighbor_candidate.parquet')
assert subprocess.check_output(['git','branch','--show-current'],text=True).strip() == 'lobt-refinement'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
if dest.exists():
    raise SystemExit('Candidate already exists; preserved without overwriting.')
base = pd.read_parquet(source)
raw = pd.read_parquet('dataset/ranking.parquet', columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt'])
gaps = pd.read_parquet(_cache_path('dataset/ranking.parquet'), columns=['gap_LOBT_flt','gap_SCHED_TIME_UTC_mvt'])
assert len(raw) == len(gaps) and raw.MVT_ID_mvt.is_unique
features = pd.concat([raw,gaps],axis=1).set_index('MVT_ID_mvt').loc[base.MVT_ID_mvt]
assert features.PHASE_mvt.eq('DEP').all()
ids = features.index
pred = base.TAXITIME_SEC_mvt.to_numpy()
gap = features.gap_LOBT_flt.to_numpy()
sched = features.gap_SCHED_TIME_UTC_mvt.to_numpy()
missing = ~np.isfinite(gap)
rome = features.ADEP_mvt.eq('LIRF').to_numpy()
oldp = pd.read_parquet('output/experiments/baseline_identity/tail_probabilities.parquet').set_index('MVT_ID_mvt')
context = pd.read_parquet('output/experiments/tail_neighbor/ranking_probabilities.parquet').set_index('MVT_ID_mvt')
assert oldp.index.is_unique and context.index.is_unique
p = np.zeros(len(ids))
p[missing] = oldp.loc[ids[missing], 'p5'].to_numpy()
sel = missing & rome
p[sel] = .5*p[sel] + .5*context.loc[ids[sel], 'p'].to_numpy()
inputs = {}
def mu(mode):
    path = out / ('ranking_' + mode + '.parquet')
    manifest = json.loads((out / ('ranking_' + mode + '_manifest.json')).read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['sha256'] and manifest['mode'] == mode
    f = pd.read_parquet(path).set_index('MVT_ID_mvt')
    assert f.index.is_unique and len(f) == len(ids) and ids.isin(f.index).all()
    inputs[mode] = digest
    values = f.loc[ids,'mu'].to_numpy()
    assert np.isfinite(values).all()
    return values
old, new = mu('baseline'), mu('neighbor')
scale = np.where(~missing & (gap<=6000), .8, 1.)
values = apply_normal_correction(pred,gap,sched,p,old,old+(new-old)*scale)
# Independent reconstruction, including the exact existing boundary rule.
lo, hi = np.maximum(0,gap-3660), gap+3660
inside = ~missing & (gap<=6000) & (pred>lo+1e-6) & (pred<hi-1e-6)
delta = .25*(new-old)*scale
audit = pred.copy()
audit[inside] = np.clip(pred[inside]+delta[inside],lo[inside],hi[inside])
share = np.where(np.nan_to_num(sched[missing],nan=0)>6000,1-p[missing],1)
audit[missing] = np.maximum(0,pred[missing]+share*delta[missing])
assert np.allclose(values,audit,rtol=0,atol=1e-10)
protected = ~missing & ~inside
assert np.array_equal(values[protected],pred[protected])
assert np.isfinite(values).all() and (values>=0).all()
candidate = base.copy()
candidate['TAXITIME_SEC_mvt'] = values.astype('float64')
assert len(candidate)==344841 and candidate.MVT_ID_mvt.is_unique
assert candidate.MVT_ID_mvt.equals(base.MVT_ID_mvt) and candidate.dtypes.equals(base.dtypes)
candidate.to_parquet(dest,index=False)
record = {'source_version':18,'source_sha256':expected,'source_live_rmse':265.5251,
    'candidate':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'rows':len(base),
    'weight':.25,'matched_normal_delta_scale':.8,'input_sha256':inputs,
    'changed_rows':int(np.count_nonzero(values-pred)),'protected_rows':int(protected.sum()),
    'delta_quantiles_sec':np.quantile(values-pred,[0,.01,.5,.99,1]).tolist(),
    'arithmetic_audit_passed':True,'uploaded':False,
    'limitations':'Original v14 ensemble is unknown; normal-share probabilities approximate its preserved mixture. Temporal validation and fresh remote quota/version checks remain separate gates.'}
(out/'v18_candidate_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
