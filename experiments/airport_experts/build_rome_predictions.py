"""Cache the validated Rome expert for later local candidate assembly; no upload."""
import sys
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
import lightgbm as lgb
from data import align_categoricals, flight_ids
from featcache import _cache_path
from encode import add_target_encodings
from model import MU_PARAMS
from weather_subset import load_weather_airport

out = Path('output/experiments/airport_experts')
meta = pd.read_parquet('output/experiments/meta.parquet')
use = meta.ADEP_mvt.eq('LIRF').to_numpy()
y = np.load('output/experiments/y.npy')[use]
meta = meta.loc[use].reset_index(drop=True)
x = load_weather_airport('LIRF', y)
raw = pd.read_parquet('dataset/ranking.parquet', filters=[('ADEP_mvt', '=', 'LIRF')]).reset_index(drop=True)
z = pd.read_parquet(_cache_path('dataset/ranking.parquet'), filters=[('ADEP_mvt', '=', 'LIRF')]).reset_index(drop=True)
assert len(raw) == len(z)
gap = (raw.MVT_TIME_UTC_mvt - raw.AOBT_3_flt).dt.total_seconds().to_numpy()
assert np.allclose(gap, z.gap_AOBT_3_flt.to_numpy(), equal_nan=True)
dep = raw.PHASE_mvt.eq('DEP')
raw, z = raw.loc[dep].reset_index(drop=True), z.loc[dep].reset_index(drop=True)
for frame, file, ids in [(x, 'training.parquet', meta.id), (z, 'ranking.parquet', raw.MVT_ID_mvt)]:
    e = pd.read_parquet('output/experiments/neighbor_proxy/' + file).set_index('MVT_ID_mvt')
    assert e.index.is_unique
    e = e.loc[ids].reset_index(drop=True)
    for col in e:
        frame[col] = e[col].to_numpy()
align_categoricals(x, z)
add_target_encodings(x, y, meta.month.to_numpy(), meta.flight.to_numpy(), z, flight_ids(raw))
modelpath = out / 'full_LIRF_weather_neighbor.txt'
if modelpath.exists():
    model = lgb.Booster(model_file=str(modelpath))
else:
    normal = y <= 6000
    print('training full Rome expert', int(normal.sum()), flush=True)
    model = lgb.train(dict(MU_PARAMS, num_threads=2, num_leaves=63, min_data_in_leaf=50), lgb.Dataset(x.loc[normal], label=y[normal]), num_boost_round=900)
    model.save_model(str(modelpath))
mu = np.maximum(0, model.predict(z, num_threads=2))
assert np.isfinite(mu).all()
pd.DataFrame({'MVT_ID_mvt': raw.MVT_ID_mvt, 'mu': mu}).to_parquet(out / 'ranking_LIRF_weather_neighbor.parquet', index=False)
print('saved Rome predictions', len(mu), '; no submission or upload', flush=True)
