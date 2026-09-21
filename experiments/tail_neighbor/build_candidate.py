"""Add the validated half-tail correction to the separate normal candidate."""
import sys, json, hashlib, subprocess
from pathlib import Path
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd

assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'lobt-refinement'
original = Path('output/experiments/tail_rmse/submitting_v14.parquet')
source_hash = '35b0d58c08481ba083673810eedf942c599ab905280f1b760ec14681e853c57d'
assert hashlib.sha256(original.read_bytes()).hexdigest() == source_hash
normal_file = Path('output/submitting_neighbor_candidate.parquet')
candidate = pd.read_parquet(normal_file)
v14 = pd.read_parquet(original)
assert candidate.MVT_ID_mvt.equals(v14.MVT_ID_mvt)
old = pd.read_parquet('output/experiments/baseline_identity/tail_probabilities.parquet').set_index('MVT_ID_mvt')
new = pd.read_parquet('output/experiments/tail_neighbor/ranking_probabilities.parquet').set_index('MVT_ID_mvt')
assert old.index.is_unique and new.index.is_unique and old.index.difference(new.index).empty
new = new.loc[old.index]
airport = pd.read_parquet('dataset/ranking.parquet', columns=['MVT_ID_mvt', 'ADEP_mvt']).set_index('MVT_ID_mvt').loc[old.index, 'ADEP_mvt']
base_mu = pd.read_parquet('output/experiments/route_delay/ranking_weather_base.parquet').set_index('MVT_ID_mvt').loc[old.index, 'mu'].to_numpy()
new_mu = pd.read_parquet('output/experiments/neighbor_proxy/ranking_weather_neighbor.parquet').set_index('MVT_ID_mvt').loc[old.index, 'mu'].to_numpy()
mu = base_mu + .25 * (new_mu - base_mu)
v = np.maximum(0, np.nan_to_num(old.v.to_numpy(), nan=0))
v = np.where(v >= 40000, v * 1.08, v)
eligible = (v > 6000) & airport.eq('LIRF').to_numpy()
delta = np.where(eligible, .5 * (new.p.to_numpy() - old.p5.to_numpy()) * (v - mu), 0.)
assert np.isfinite(delta).all()
mapped = candidate.MVT_ID_mvt.map(pd.Series(delta, index=old.index)).fillna(0).to_numpy()
candidate['TAXITIME_SEC_mvt'] = np.maximum(0, candidate.TAXITIME_SEC_mvt.to_numpy() + mapped)
assert len(candidate) == 344841 and candidate.MVT_ID_mvt.is_unique
assert candidate.dtypes.equals(v14.dtypes) and np.isfinite(candidate.TAXITIME_SEC_mvt).all()
output = Path('output/submitting_neighbor_rome_tail_candidate.parquet')
candidate.to_parquet(output, index=False)
manifest = {
    'source_version': 14, 'source_score': 268.4067, 'source_sha256': source_hash,
    'normal_candidate_sha256': hashlib.sha256(normal_file.read_bytes()).hexdigest(),
    'candidate_file': str(output), 'candidate_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'rows': len(candidate), 'uploaded': False,
    'normal_weight': .25, 'tail_weight': .5,
    'tail_airport': 'LIRF',
    'validation_log': 'output/experiments/tail_neighbor/combined_rome.log',
    'tail_changed_rows': int(np.count_nonzero(mapped)),
    'tail_delta_quantiles_sec': np.quantile(delta, [0, .01, .5, .99, 1]).tolist(),
    'limitation': 'Local controls use frozen v9 components. The v14 recipe is unconfirmed; paired-model differences are applied to its preserved predictions. Local improvement is not a verified leaderboard improvement.'}
Path('output/experiments/tail_neighbor/candidate_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2), flush=True)
