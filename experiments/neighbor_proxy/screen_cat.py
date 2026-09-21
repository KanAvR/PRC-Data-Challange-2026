"""Check the same neighborhood signals in the existing CatBoost family."""
import sys
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from data import load_training, align_categoricals
from encode import add_target_encodings
from model import train_mu_cat, CAT_CTR1_PARAMS, _CatMu, combine
from constraints import project_lobt
from validate import SPLITS, compare

split = sys.argv[1] if len(sys.argv) > 1 else 'inner'
out = Path('output/experiments/neighbor_proxy')
path = out / ('cat_weather_' + split + '.parquet')
if not path.exists():
    x, y, meta = load_training()
    extra = pd.read_parquet(out / 'training.parquet').set_index('MVT_ID_mvt')
    assert extra.index.is_unique
    x = pd.concat([x, extra.loc[meta.MVT_ID_mvt].reset_index(drop=True)], axis=1)
    del extra
    sp = SPLITS[split]
    va = meta.month.isin(sp['valid']).to_numpy()
    tr = ~meta.month.isin(sp['valid'] + sp['drop']).to_numpy()
    a, b = x.loc[tr].copy(), x.loc[va].copy()
    align_categoricals(a, b)
    add_target_encodings(a, y[tr], meta.loc[tr, 'month'].to_numpy(),
        meta.loc[tr, 'flight'].to_numpy(), b, meta.loc[va, 'flight'].to_numpy())
    modelpath = out / ('cat_weather_' + split + '.cbm')
    if modelpath.exists():
        model = _CatMu(CatBoostRegressor().load_model(str(modelpath)))
    else:
        print('training neighborhood CatBoost', split, flush=True)
        params = dict(CAT_CTR1_PARAMS, thread_count=4, verbose=250,
            allow_writing_files=True, save_snapshot=True, snapshot_interval=60,
            train_dir=str(out / ('cat_training_' + split)), snapshot_file='snapshot.cbsnapshot')
        model = train_mu_cat(a, y[tr], params=params)
        model.model.save_model(str(modelpath))
    pd.DataFrame({'id': meta.loc[va, 'MVT_ID_mvt'].to_numpy(), 'y': y[va],
        'mu': np.maximum(0, model.predict(b))}).to_parquet(path, index=False)

new = pd.read_parquet(path)
old = pd.read_parquet('output/experiments/aobt_residual/cat_ctr1_weather_' + split + '.parquet')
base = pd.read_parquet('output/experiments/route_delay/weather_base_' + split + '.parquet')
neighbor = pd.read_parquet(out / ('weather_' + split + '.parquet'))
assert base.id.equals(new.id) and base.id.equals(old.id) and base.id.equals(neighbor.id)
assert np.array_equal(base.y, new.y) and np.array_equal(base.y, old.y)
def prediction(mu):
    r = base.copy()
    r['mu'] = np.asarray(mu)
    r['pred'] = project_lobt(combine(r.mu.to_numpy(), r.p.to_numpy(), r.v.to_numpy()), r.gap_LOBT_flt.to_numpy())
    high = r.gap_LOBT_flt > 6000
    r.loc[high, 'pred'] = base.loc[high, 'pred']
    return r
cat_control, cat_new = prediction(old.mu), prediction(new.mu)
compare(split, cat_new, cat_control, 'Cat-neighborhood-only')
control = prediction(.5 * old.mu + .5 * neighbor.mu)
candidate = prediction(.5 * new.mu + .5 * neighbor.mu)
candidate.to_parquet(out / ('cat_ensemble_' + split + '.parquet'), index=False)
compare(split, candidate, control, 'Cat-increment-with-neighbor-LGB')
