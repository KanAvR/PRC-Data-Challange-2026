"""Paired temporal checks of weather CatBoost with and without neighbours."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from data import load_training, align_categoricals
from encode import add_target_encodings
from model import train_mu_cat, CAT_CTR1_PARAMS, _CatMu
from validate import SPLITS

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['baseline', 'neighbor'])
parser.add_argument('splits', nargs='+', choices=list(SPLITS))
args = parser.parse_args()
out = Path('output/experiments/cat_temporal')
out.mkdir(exist_ok=True)
x, y, meta = load_training()
if args.mode == 'neighbor':
    extra = pd.read_parquet('output/experiments/neighbor_proxy/training.parquet').set_index('MVT_ID_mvt')
    assert extra.index.is_unique and meta.MVT_ID_mvt.isin(extra.index).all()
    x = pd.concat([x, extra.loc[meta.MVT_ID_mvt].reset_index(drop=True)], axis=1)
    del extra
for split in args.splits:
    stem = args.mode + '_' + split
    path = out / (stem + '.parquet')
    if path.exists():
        print('preserved existing predictions', stem, flush=True)
        continue
    sp = SPLITS[split]
    va = meta.month.isin(sp['valid']).to_numpy()
    tr = ~meta.month.isin(sp['valid'] + sp['drop']).to_numpy()
    a, b = x.loc[tr].copy(), x.loc[va].copy()
    align_categoricals(a, b)
    add_target_encodings(a, y[tr], meta.loc[tr, 'month'].to_numpy(),
        meta.loc[tr, 'flight'].to_numpy(), b, meta.loc[va, 'flight'].to_numpy())
    modelpath = out / (stem + '.cbm')
    if modelpath.exists():
        model = _CatMu(CatBoostRegressor().load_model(str(modelpath)))
    else:
        print('training', stem, flush=True)
        params = dict(CAT_CTR1_PARAMS, thread_count=4, verbose=250,
            allow_writing_files=True, save_snapshot=True, snapshot_interval=60,
            train_dir=str(out / (stem + '_training')), snapshot_file='snapshot.cbsnapshot')
        model = train_mu_cat(a, y[tr], params=params)
        model.model.save_model(str(modelpath))
    mu = np.maximum(0, model.predict(b))
    assert np.isfinite(mu).all()
    pd.DataFrame({'id': meta.loc[va, 'MVT_ID_mvt'].to_numpy(), 'y': y[va],
        'mu': mu}).to_parquet(path, index=False)
    print('completed', stem, len(mu), flush=True)
    del a, b, model, mu
