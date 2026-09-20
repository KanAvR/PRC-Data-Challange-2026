"""Build separate residual and residual-plus-Rome candidates from immutable v15."""
import sys, json, hashlib, time, subprocess
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
import lightgbm as lgb
from data import load_training, align_categoricals, flight_ids
from featcache import build_features_cached
from encode import add_target_encodings
from model import MU_PARAMS
from anchor import normal_anchor
from normal_correction import apply_normal_correction

out = Path('output/experiments/neighbor_proxy')
source = Path('output/submitting_v15.parquet')
expected = '551d4f8e5af1bc94331e3c5ace81ba322eeaad27e38bad68794a6b0df971068a'
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'lobt-refinement'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
print('loading full weather and neighborhood features', flush=True)
x, y, meta = load_training()
raw = pd.read_parquet('dataset/ranking.parquet')
dep = raw.PHASE_mvt.eq('DEP')
z = build_features_cached('dataset/ranking.parquet', raw).loc[dep].reset_index(drop=True)
ids = raw.loc[dep, 'MVT_ID_mvt'].to_numpy()
def extra(file, ordered_ids):
    e = pd.read_parquet(out / file).set_index('MVT_ID_mvt')
    assert e.index.is_unique
    return e.loc[ordered_ids].reset_index(drop=True)
x = pd.concat([x, extra('training.parquet', meta.MVT_ID_mvt)], axis=1)
z = pd.concat([z, extra('ranking.parquet', ids)], axis=1)
x['normal_anchor'], z['normal_anchor'] = normal_anchor(x), normal_anchor(z)
align_categoricals(x, z)
add_target_encodings(x, y, meta.month.to_numpy(), meta.flight.to_numpy(), z, flight_ids(raw.loc[dep]))
path = out / 'ranking_weather_anchor.parquet'
if not path.exists():
    modelpath = out / 'full_weather_anchor.txt'
    if modelpath.exists():
        model = lgb.Booster(model_file=str(modelpath))
    else:
        use = y <= 6000
        print('training full residual model', flush=True)
        start = time.time()
        def progress(env):
            if (env.iteration + 1) % 300 == 0:
                print(env.iteration + 1, 'rounds', round(time.time()-start), 'seconds', flush=True)
        model = lgb.train(dict(MU_PARAMS, num_threads=4), lgb.Dataset(x.loc[use], label=y[use]-x.loc[use, 'normal_anchor'].to_numpy()), num_boost_round=900, callbacks=[progress])
        model.save_model(str(modelpath))
    mu = np.maximum(0, model.predict(z, num_threads=4) + z.normal_anchor.to_numpy())
    pd.DataFrame({'MVT_ID_mvt':ids, 'mu':mu}).to_parquet(path, index=False)
def ordered_mu(file):
    f = pd.read_parquet(file).set_index('MVT_ID_mvt')
    assert f.index.is_unique
    return f.loc[ids, 'mu'].to_numpy()
plain_mu = ordered_mu(out / 'ranking_weather_neighbor.parquet')
anchor_mu = ordered_mu(path)
v15 = pd.read_parquet(source)
prediction = v15.set_index('MVT_ID_mvt').loc[ids, 'TAXITIME_SEC_mvt'].to_numpy()
gap = z.gap_LOBT_flt.to_numpy()
missing = ~np.isfinite(gap)
rome = z.ADEP_mvt.eq('LIRF').to_numpy()
oldp = pd.read_parquet('output/experiments/baseline_identity/tail_probabilities.parquet').set_index('MVT_ID_mvt')
context = pd.read_parquet('output/experiments/tail_neighbor/ranking_probabilities.parquet').set_index('MVT_ID_mvt')
p15 = np.zeros(len(ids))
p15[missing] = oldp.loc[ids[missing], 'p5'].to_numpy()
sel = rome & missing
p15[sel] = .5 * p15[sel] + .5 * context.loc[ids[sel], 'p'].to_numpy()
anchored = apply_normal_correction(prediction, gap, z.gap_SCHED_TIME_UTC_mvt, p15, plain_mu, anchor_mu)
expert = pd.read_parquet('output/experiments/airport_experts/ranking_LIRF_weather_neighbor.parquet').set_index('MVT_ID_mvt')
member_mu = anchored.copy()
sel = rome & ~missing & (gap <= 6000)
member_mu[sel] = expert.loc[ids[sel], 'mu'].to_numpy()
combined = apply_normal_correction(anchored, gap, z.gap_SCHED_TIME_UTC_mvt, p15, anchored, member_mu, weight=.2)
assert np.array_equal(combined[missing | ~rome], anchored[missing | ~rome])
manifests = {}
for name, values in [('anchor', anchored), ('anchor_rome', combined)]:
    assert np.array_equal(values[gap > 6000], prediction[gap > 6000])
    candidate = v15.copy()
    candidate['TAXITIME_SEC_mvt'] = candidate.MVT_ID_mvt.map(pd.Series(values, index=ids)).astype('float64')
    assert len(candidate) == 344841 and candidate.MVT_ID_mvt.is_unique
    assert candidate.MVT_ID_mvt.equals(v15.MVT_ID_mvt) and candidate.dtypes.equals(v15.dtypes)
    assert np.isfinite(candidate.TAXITIME_SEC_mvt).all() and candidate.TAXITIME_SEC_mvt.ge(0).all()
    dest = Path('output/submitting_' + name + '_candidate.parquet')
    candidate.to_parquet(dest, index=False)
    delta = values-prediction
    manifests[name] = {'source_version':15, 'source_score':267.0668, 'source_sha256':expected,
        'file':str(dest), 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(), 'rows':len(candidate),
        'changed_rows':int(np.count_nonzero(delta)), 'uploaded':False, 'normal_weight':.25,
        'Rome_member_weight':.2 if name=='anchor_rome' else 0,
        'delta_quantiles_sec':np.quantile(delta,[0,.01,.5,.99,1]).tolist(),
        'limitation':'Normal-share probabilities estimate the preserved v15 mixture; original v14 recipe is unconfirmed. Require completed temporal validation and fresh user approval before upload.'}
(out / 'anchor_candidate_manifests.json').write_text(json.dumps(manifests,indent=2)+'\n')
print(json.dumps(manifests,indent=2),flush=True)
