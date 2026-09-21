"""Cache a full-data context classifier; no submission or upload."""
import sys
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from model import NO_LOBT_FEATURES, CAT_CLF_PARAMS, _fit_seeds_cat
from featcache import _cache_path
from context_features import add_context

out = Path('output/experiments/tail_neighbor')
CAT_CLF_PARAMS['thread_count'] = 2
cols = NO_LOBT_FEATURES + ['gap_LOBT_flt', 'dep_off_apt_30m', 'dep_off_apt_next_30m']
x = pd.read_parquet('output/experiments/X.parquet', columns=cols)
no = x.gap_LOBT_flt.isna().to_numpy()
y = np.load('output/experiments/y.npy')[no]
meta = pd.read_parquet('output/experiments/meta.parquet', columns=['id']).loc[no]
x = x.loc[no].reset_index(drop=True)
extra = pd.read_parquet('output/experiments/neighbor_proxy/training.parquet').set_index('MVT_ID_mvt').loc[meta.id].reset_index(drop=True)
added = add_context(x, extra)
del extra
raw = pd.read_parquet('dataset/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt'])
z = pd.read_parquet(_cache_path('dataset/ranking.parquet'), columns=cols)
use = raw.PHASE_mvt.eq('DEP') & z.gap_LOBT_flt.isna()
ids = raw.loc[use, 'MVT_ID_mvt'].to_numpy()
z = z.loc[use].reset_index(drop=True)
extra = pd.read_parquet('output/experiments/neighbor_proxy/ranking.parquet').set_index('MVT_ID_mvt').loc[ids].reset_index(drop=True)
assert added == add_context(z, extra)
print('training five full-data classifiers', len(x), len(z), flush=True)
models = _fit_seeds_cat(x, (y > 6000).astype(int), NO_LOBT_FEATURES + added)
for i, m in enumerate(models):
    m.model.save_model(str(out / ('full_context_' + str(i) + '.cbm')))
p = np.mean([m.predict(z[NO_LOBT_FEATURES + added]) for m in models], axis=0)
assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
pd.DataFrame({'MVT_ID_mvt': ids, 'p': p}).to_parquet(out / 'ranking_probabilities.parquet', index=False)
print('saved ranking probabilities; no submission created', flush=True)
