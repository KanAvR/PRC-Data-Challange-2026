"""Fit one paired weather CatBoost member for ranking after temporal validation."""
import argparse, sys, json, hashlib
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from data import load_training, align_categoricals, flight_ids
from featcache import build_features_cached
from encode import add_target_encodings
from model import train_mu_cat, CAT_CTR1_PARAMS, _CatMu

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['baseline', 'neighbor'])
args = parser.parse_args()
out = Path('output/experiments/cat_temporal')
path = out / ('ranking_' + args.mode + '.parquet')
if path.exists():
    raise SystemExit('Ranking predictions already exist; preserved without overwriting.')
x, y, meta = load_training()
raw = pd.read_parquet('dataset/ranking.parquet')
dep = raw.PHASE_mvt.eq('DEP')
z = build_features_cached('dataset/ranking.parquet', raw).loc[dep].reset_index(drop=True)
ids = raw.loc[dep, 'MVT_ID_mvt'].to_numpy()
if args.mode == 'neighbor':
    def extra(file, ordered):
        e = pd.read_parquet('output/experiments/neighbor_proxy/' + file).set_index('MVT_ID_mvt')
        assert e.index.is_unique and pd.Index(ordered).isin(e.index).all()
        return e.loc[ordered].reset_index(drop=True)
    x = pd.concat([x, extra('training.parquet', meta.MVT_ID_mvt)], axis=1)
    z = pd.concat([z, extra('ranking.parquet', ids)], axis=1)
align_categoricals(x, z)
add_target_encodings(x, y, meta.month.to_numpy(), meta.flight.to_numpy(), z, flight_ids(raw.loc[dep]))
modelpath = out / ('full_' + args.mode + '.cbm')
if modelpath.exists():
    model = _CatMu(CatBoostRegressor().load_model(str(modelpath)))
else:
    params = dict(CAT_CTR1_PARAMS, thread_count=4, verbose=250,
        allow_writing_files=True, save_snapshot=True, snapshot_interval=60,
        train_dir=str(out / ('full_' + args.mode + '_training')), snapshot_file='snapshot.cbsnapshot')
    print('training full', args.mode, flush=True)
    model = train_mu_cat(x, y, params=params)
    model.model.save_model(str(modelpath))
assert list(model.model.feature_names_) == list(z.columns)
mu = np.maximum(0, model.predict(z))
assert len(mu) == 344841 and np.isfinite(mu).all() and pd.Index(ids).is_unique
pd.DataFrame({'MVT_ID_mvt': ids, 'mu': mu}).to_parquet(path, index=False)
record = {'mode': args.mode, 'rows': len(mu), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'model_sha256': hashlib.sha256(modelpath.read_bytes()).hexdigest(), 'features': list(z.columns),
    'training_rows': len(y), 'normal_training_rows': int((y <= 6000).sum()), 'uploaded': False}
(out / ('ranking_' + args.mode + '_manifest.json')).write_text(json.dumps(record, indent=2)+'\n')
print('completed', args.mode, record['sha256'], flush=True)
