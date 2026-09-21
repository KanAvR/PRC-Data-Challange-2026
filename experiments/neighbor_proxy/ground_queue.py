"""Active ground-traffic counts from provided flight-plan timestamps only."""
from pathlib import Path
import numpy as np
import pandas as pd

INPUTS = ['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt', 'ADEP_mvt', 'RUNWAY_mvt', 'AOBT_3_flt']

def build(raw):
    d = raw.loc[raw.PHASE_mvt == 'DEP', INPUTS].copy().reset_index(drop=True)
    out = pd.DataFrame({'MVT_ID_mvt': d.MVT_ID_mvt})
    for scope, keys in [('apt', ['ADEP_mvt']), ('rwy', ['ADEP_mvt', 'RUNWAY_mvt'])]:
        for _, g in d.groupby(keys, dropna=False, observed=True):
            stop = pd.to_datetime(g.MVT_TIME_UTC_mvt, utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[s]').astype(np.int64)
            start = pd.to_datetime(g.AOBT_3_flt, utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[s]').astype(np.int64)
            gap = (g.MVT_TIME_UTC_mvt - g.AOBT_3_flt).dt.total_seconds().to_numpy()
            valid = np.isfinite(gap) & (gap > 0) & (gap <= 7200)
            starts, stops = np.sort(start[valid]), np.sort(stop[valid])
            for label, query in [('start', np.where(valid, start, stop - 900)),
                                 ('10m_before', stop - 600), ('takeoff', stop)]:
                count = np.searchsorted(starts, query, side='right') - np.searchsorted(stops, query, side='right')
                count -= (valid & (start <= query) & (query < stop)).astype(int)
                assert (count >= 0).all()
                out.loc[g.index, 'ground_' + scope + '_' + label] = count
    return out

if __name__ == '__main__':
    out = Path('output/experiments/neighbor_proxy')
    parts = []
    for path in sorted(Path('dataset').glob('training_*.parquet')):
        print(path, flush=True)
        parts.append(build(pd.read_parquet(path, columns=INPUTS)))
    pd.concat(parts, ignore_index=True).to_parquet(out / 'ground_training.parquet', index=False)
    build(pd.read_parquet('dataset/ranking.parquet', columns=INPUTS)).to_parquet(out / 'ground_ranking.parquet', index=False)
