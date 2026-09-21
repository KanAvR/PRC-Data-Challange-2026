"""Fixed-capacity airport experts blended with the new neighborhood control."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
import lightgbm as lgb
from data import align_categoricals
from encode import add_target_encodings
from model import MU_PARAMS, combine
from constraints import project_lobt
from validate import SPLITS, compare, verdict

parser = argparse.ArgumentParser()
parser.add_argument('airport')
parser.add_argument('--weather', action='store_true')
parser.add_argument('--bounded-tail', action='store_true', help='fit all labels only when training labels stay below 12000 seconds')
parser.add_argument('splits', nargs='*', default=['inner'])
args = parser.parse_args()
out = Path('output/experiments/airport_experts')
meta = pd.read_parquet('output/experiments/meta.parquet')
use = meta.ADEP_mvt.eq(args.airport).to_numpy()
if not use.any():
    raise ValueError('unknown airport')
y = np.load('output/experiments/y.npy')[use]
meta = meta.loc[use].reset_index(drop=True)
if args.weather:
    from weather_subset import load_weather_airport
    x = load_weather_airport(args.airport, y)
else:
    x = pd.read_parquet('output/experiments/X.parquet', filters=[('ADEP_mvt', '=', args.airport)]).reset_index(drop=True)
x = x.drop(columns=[c for c in x if c.startswith('extra_')])
assert len(x) == len(meta)
extra = pd.read_parquet('output/experiments/neighbor_proxy/training.parquet').set_index('MVT_ID_mvt')
x = pd.concat([x, extra.loc[meta.id].reset_index(drop=True)], axis=1)
del extra
checks = {}
for split in args.splits or ['inner']:
    sp = SPLITS[split]
    va = meta.month.isin(sp['valid']).to_numpy()
    tr = ~meta.month.isin(sp['valid'] + sp['drop']).to_numpy()
    base = pd.read_parquet('output/experiments/neighbor_proxy/' + ('weather_' if args.weather else '') + split + '.parquet')
    target = base.ADEP_mvt.eq(args.airport).to_numpy()
    assert np.array_equal(base.loc[target, 'id'], meta.loc[va, 'id'])
    a, b = x.loc[tr].copy(), x.loc[va].copy()
    align_categoricals(a, b)
    add_target_encodings(a, y[tr], meta.loc[tr, 'month'].to_numpy(),
        meta.loc[tr, 'flight'].to_numpy(), b, meta.loc[va, 'flight'].to_numpy())
    if args.bounded_tail and y[tr].max() > 12000:
        raise ValueError('training airport has unbounded reporting tails')
    normal = np.ones(tr.sum(), dtype=bool) if args.bounded_tail else y[tr] <= 6000
    print('training', args.airport, split, normal.sum(), 'rows', flush=True)
    m = lgb.train(dict(MU_PARAMS, num_threads=2, num_leaves=63, min_data_in_leaf=50),
        lgb.Dataset(a.loc[normal], label=y[tr][normal]), num_boost_round=900)
    local = np.maximum(0, m.predict(b, num_threads=2))
    mu = .5 * base.loc[target, 'mu'].to_numpy() + .5 * local
    pred = project_lobt(combine(mu, base.loc[target, 'p'].to_numpy(), base.loc[target, 'v'].to_numpy()),
        b.gap_LOBT_flt.to_numpy())
    if args.bounded_tail:
        # This expert estimates the unconditional mean, including long taxis.
        # Blend predictions directly rather than adding a second tail mixture.
        local_pred = project_lobt(local, b.gap_LOBT_flt.to_numpy())
        pred = .5 * base.loc[target, 'pred'].to_numpy() + .5 * local_pred
    high = b.gap_LOBT_flt.to_numpy() > 6000
    pred[high] = base.loc[target, 'pred'].to_numpy()[high]
    r = base.copy()
    r.loc[target, 'mu'], r.loc[target, 'pred'] = mu, pred
    label = args.airport + ('_weather' if args.weather else '') + ('_bounded' if args.bounded_tail else '')
    r.to_parquet(out / (label + '_' + split + '.parquet'), index=False)
    checks[split] = compare(split, r, base, label + '-expert')
    print(verdict(checks), flush=True)
