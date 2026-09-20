"""Build a separate neighborhood candidate on v14; never upload or overwrite v14."""
import sys, time, json, hashlib, subprocess
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
import lightgbm as lgb
from data import load_training, align_categoricals, flight_ids
from featcache import build_features_cached
from encode import add_target_encodings
from model import MU_PARAMS
from normal_correction import apply_normal_correction

out = Path('output/experiments/neighbor_proxy')
source = Path('output/experiments/tail_rmse/submitting_v14.parquet')
expected = '35b0d58c08481ba083673810eedf942c599ab905280f1b760ec14681e853c57d'
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'lobt-refinement'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
print('loading full training and ranking features', flush=True)
x, y, meta = load_training()
raw = pd.read_parquet('dataset/ranking.parquet')
dep = raw.PHASE_mvt.eq('DEP')
z = build_features_cached('dataset/ranking.parquet', raw).loc[dep].copy().reset_index(drop=True)
ids = raw.loc[dep, 'MVT_ID_mvt'].to_numpy()
def extras(file, ordered_ids):
    e = pd.read_parquet(out / file).set_index('MVT_ID_mvt')
    assert e.index.is_unique and pd.Index(ordered_ids).isin(e.index).all()
    return e.loc[ordered_ids].reset_index(drop=True)
x = pd.concat([x, extras('training.parquet', meta.MVT_ID_mvt.to_numpy())], axis=1)
z = pd.concat([z, extras('ranking.parquet', ids)], axis=1)
align_categoricals(x, z)
add_target_encodings(x, y, meta.month.to_numpy(), meta.flight.to_numpy(), z, flight_ids(raw.loc[dep]))
path = out / 'ranking_weather_neighbor.parquet'
if not path.exists():
    normal = y <= 6000
    print('training full neighborhood model', flush=True)
    t = time.time()
    def progress(env):
        if (env.iteration + 1) % 300 == 0:
            print(env.iteration + 1, 'rounds', round(time.time() - t), 'seconds', flush=True)
    model = lgb.train(dict(MU_PARAMS, num_threads=4),
        lgb.Dataset(x.loc[normal], label=y[normal]), num_boost_round=900, callbacks=[progress])
    model.save_model(str(out / 'full_weather_neighbor.txt'))
    mu = np.maximum(0, model.predict(z, num_threads=4))
    pd.DataFrame({'MVT_ID_mvt': ids, 'mu': mu}).to_parquet(path, index=False)
def ordered_mu(file):
    frame = pd.read_parquet(file).set_index('MVT_ID_mvt')
    assert frame.index.is_unique and pd.Index(ids).isin(frame.index).all()
    return frame.loc[ids, 'mu'].to_numpy()
baseline_mu = ordered_mu('output/experiments/route_delay/ranking_weather_base.parquet')
new_mu = ordered_mu(path)
v14 = pd.read_parquet(source)
baseline = v14.set_index('MVT_ID_mvt').loc[ids, 'TAXITIME_SEC_mvt'].to_numpy()
gap = z.gap_LOBT_flt.to_numpy()
missing = ~np.isfinite(gap)
prob = pd.read_parquet('output/experiments/baseline_identity/tail_probabilities.parquet').set_index('MVT_ID_mvt')
assert prob.index.is_unique and pd.Index(ids[missing]).isin(prob.index).all()
p = np.zeros(len(ids))
p[missing] = prob.loc[ids[missing], 'p5'].to_numpy()
prediction = apply_normal_correction(baseline, gap, z.gap_SCHED_TIME_UTC_mvt, p, baseline_mu, new_mu)
assert np.array_equal(prediction[gap > 6000], baseline[gap > 6000])
candidate = v14.copy()
candidate['TAXITIME_SEC_mvt'] = candidate.MVT_ID_mvt.map(pd.Series(prediction, index=ids)).astype('float64')
assert len(candidate) == 344841 and candidate.MVT_ID_mvt.is_unique
assert candidate.MVT_ID_mvt.equals(v14.MVT_ID_mvt) and candidate.dtypes.equals(v14.dtypes)
assert np.isfinite(candidate.TAXITIME_SEC_mvt).all() and candidate.TAXITIME_SEC_mvt.ge(0).all()
destination = Path('output/submitting_neighbor_candidate.parquet')
candidate.to_parquet(destination, index=False)
delta = prediction - baseline
manifest = {'source_version': 14, 'source_score': 268.4067, 'source_sha256': expected,
    'candidate_file': str(destination), 'candidate_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    'rows': len(candidate), 'changed': int(np.count_nonzero(delta)), 'uploaded': False,
    'weight': .25, 'normal_model': '900-round weather LGB plus neighbor flight-plan proxies',
    'rule': 'Add a quarter normal-model difference to v14. Matched long-gap and boundary predictions are preserved. Missing-LOBT changes are weighted by the normal contribution of the frozen five-seed classifier; existing v14 predictions are retained as the starting point.',
    'limitation': 'v14 exact training recipe is unconfirmed; the normal-contribution probability estimate is not claimed to reproduce its exact classifier.',
    'delta_quantiles_sec': dict(zip(['min', 'p01', 'median', 'p99', 'max'], np.quantile(delta, [0, .01, .5, .99, 1]).tolist()))}
(out / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2), flush=True)
