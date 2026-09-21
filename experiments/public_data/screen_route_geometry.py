import sys, time, argparse
from pathlib import Path
sys.path.insert(0, 'output/experiments/neighbor_proxy')
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
import lightgbm as lgb
from data import align_categoricals, load_training
from encode import add_target_encodings
from model import MU_PARAMS, combine
from constraints import project_lobt
from validate import SPLITS, compare, verdict

out = Path('output/experiments/neighbor_proxy')
parser = argparse.ArgumentParser()
parser.add_argument('--features', choices=['neighbor', 'ground', 'both'], default='neighbor')
parser.add_argument('--weather', action='store_true')
parser.add_argument('--anchor', action='store_true')
parser.add_argument('--wide', action='store_true', help='retain proxy gaps through 12000 seconds')
parser.add_argument('splits', nargs='*', default=['inner'])
args = parser.parse_args()
if args.weather:
    x, y, meta = load_training()
else:
    x = pd.read_parquet('output/experiments/X.parquet')
    x = x.drop(columns=[c for c in x if c.startswith('extra_')])
    y = np.load('output/experiments/y.npy')
    meta = pd.read_parquet('output/experiments/meta.parquet')
ids = pd.read_parquet('output/experiments/raw_dep.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt
neighbor_file = 'wide_training.parquet' if args.wide else 'training.parquet'
files = {'neighbor': [neighbor_file], 'ground': ['ground_training.parquet'],
         'both': [neighbor_file, 'ground_training.parquet']}[args.features]
extra = pd.concat([pd.read_parquet(out / f).set_index('MVT_ID_mvt') for f in files], axis=1)
assert extra.index.is_unique and ids.isin(extra.index).all()
extra = extra.reindex(ids).reset_index(drop=True)
assert len(extra) == len(x)
x = pd.concat([x, extra], axis=1)
from route_geometry import build
geo = build(x)
print('route feature coverage', geo.notna().mean().to_dict(), flush=True)
x = pd.concat([x, geo], axis=1)
if args.anchor:
    if args.features == 'ground':
        raise ValueError('the anchor requires neighbor proxy features')
    from anchor import normal_anchor
    x['normal_anchor'] = normal_anchor(x, limit=12000 if args.wide else 6000)
results = {}
prefix = ('weather_' if args.weather else '') + ('wide_' if args.wide else '') + ('anchor_' if args.anchor else '') + (args.features + '_' if args.features != 'neighbor' else '')
prefix += 'route_'
for split in args.splits or ['inner']:
    sp = SPLITS[split]
    va = meta.month.isin(sp['valid']).to_numpy()
    tr = ~meta.month.isin(sp['valid'] + sp['drop']).to_numpy()
    a, b = x.loc[tr].copy(), x.loc[va].copy()
    align_categoricals(a, b)
    add_target_encodings(a, y[tr], meta.loc[tr, 'month'].values,
        meta.loc[tr, 'flight'].values, b, meta.loc[va, 'flight'].values)
    normal = y[tr] <= 6000
    at = a.normal_anchor.to_numpy() if args.anchor else np.zeros(len(a))
    bt = b.normal_anchor.to_numpy() if args.anchor else np.zeros(len(b))
    print('training', split, flush=True)
    t = time.time()
    def progress(env):
        if (env.iteration + 1) % 300 == 0:
            print(split, env.iteration + 1, 'rounds', round(time.time() - t), 'seconds', flush=True)
    m = lgb.train(dict(MU_PARAMS, num_threads=4),
        lgb.Dataset(a.loc[normal], label=y[tr][normal] - at[normal]), num_boost_round=900, callbacks=[progress])
    mu = np.maximum(0, m.predict(b, num_threads=4) + bt)
    base = pd.read_parquet('output/experiments/route_delay/weather_base_' + split + '.parquet'
        if args.weather else 'output/validate/study_specialist/' + split + '.parquet')
    assert np.array_equal(base.y, y[va])
    r = base.copy()
    r['mu'] = mu
    r['pred'] = project_lobt(combine(mu, base.p.to_numpy(), base.v.to_numpy()), b.gap_LOBT_flt.to_numpy())
    high = b.gap_LOBT_flt.to_numpy() > 6000
    r.loc[high, 'pred'] = base.loc[high, 'pred']
    r.to_parquet(out / (prefix + split + '.parquet'), index=False)
    results[split] = compare(split, r, base, 'neighbor-proxy')
    pd.Series(m.feature_importance('gain'), index=a.columns).sort_values(ascending=False).to_csv(out / (prefix + split + '_importance.csv'))
    print('done', split, round(time.time() - t), verdict(results), flush=True)
