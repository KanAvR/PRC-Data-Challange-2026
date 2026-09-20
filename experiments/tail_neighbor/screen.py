"""Test whether nearby flight-plan reporting explains missing-LOBT tails."""
import sys
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from model import NO_LOBT_FEATURES, CAT_CLF_PARAMS, _fit_seeds_cat, analytic_tail_value, combine
from validate import SPLITS, compare, verdict
from context_features import add_context

CAT_CLF_PARAMS['thread_count'] = 2
out = Path('output/experiments/tail_neighbor')
meta = pd.read_parquet('output/experiments/meta.parquet')
y = np.load('output/experiments/y.npy')
cols = NO_LOBT_FEATURES + ['gap_LOBT_flt', 'dep_off_apt_30m', 'dep_off_apt_next_30m']
x = pd.read_parquet('output/experiments/X.parquet', columns=cols)
no = x.gap_LOBT_flt.isna().to_numpy()
x, y, meta = x.loc[no].reset_index(drop=True), y[no], meta.loc[no].reset_index(drop=True)
extra = pd.read_parquet('output/experiments/neighbor_proxy/training.parquet').set_index('MVT_ID_mvt').loc[meta.id].reset_index(drop=True)
added = add_context(x, extra)
del extra
checks = {}
for split in sys.argv[1:] or ['inner', 'dec_jun', 'mar_sep', 'holdout']:
    sp = SPLITS[split]
    va = meta.month.isin(sp['valid']).to_numpy()
    tr = ~meta.month.isin(sp['valid'] + sp['drop']).to_numpy()
    base = pd.read_parquet('output/experiments/tail_aux_labels/control_' + split + '.parquet')
    use = ~base.has_lobt.to_numpy()
    assert np.array_equal(base.loc[use, 'id'], meta.loc[va, 'id'])
    print('training', split, flush=True)
    selected = NO_LOBT_FEATURES + added
    models = _fit_seeds_cat(x.loc[tr], (y[tr] > 6000).astype(int), selected)
    p = np.mean([m.predict(x.loc[va, selected]) for m in models], axis=0)
    r = base.copy()
    r.loc[use, 'p'] = p
    r.loc[use, 'pred'] = combine(base.loc[use, 'mu'].to_numpy(), p, analytic_tail_value(x.loc[va]))
    r.to_parquet(out / (split + '.parquet'), index=False)
    checks[split] = compare(split, r, base, 'neighbor-tail-vs-paired-control')
    print(verdict(checks), flush=True)
