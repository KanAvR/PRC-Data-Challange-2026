"""Nearby departures' flight-plan taxi proxies; no airport target timestamps."""
import sys
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
import pandas as pd
from features import _window_mean, _window_counts

INPUTS = ['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt', 'ADEP_mvt',
          'RUNWAY_mvt', 'AOBT_3_flt', 'LOBT_flt']

def build(raw, max_proxy_gap=6000):
    d = raw.loc[raw.PHASE_mvt == 'DEP', INPUTS].copy().reset_index(drop=True)
    out = pd.DataFrame({'MVT_ID_mvt': d.MVT_ID_mvt})
    for scope, keys in [('apt', ['ADEP_mvt']), ('rwy', ['ADEP_mvt', 'RUNWAY_mvt'])]:
        for _, g in d.groupby(keys, dropna=False, observed=True):
            ts = pd.to_datetime(g.MVT_TIME_UTC_mvt, utc=True).dt.tz_localize(None).to_numpy(dtype='datetime64[s]')
            for source in ['AOBT_3_flt', 'LOBT_flt']:
                gap = (g.MVT_TIME_UTC_mvt - g[source]).dt.total_seconds().to_numpy()
                # A feature-only normal-proxy filter, identically applied to ranking.
                gap = np.where((gap >= 0) & (gap <= max_proxy_gap), gap, np.nan)
                for after in [False, True]:
                    direction = 'after' if after else 'before'
                    means = _window_mean(ts, gap, ts, [15, 30, 60], after=after)
                    counts = _window_counts(ts[np.isfinite(gap)], ts, [15, 30, 60], after=after)
                    for w in means:
                        key = 'proxy_' + scope + '_' + source + '_' + direction + '_' + str(w)
                        out.loc[g.index, key] = means[w]
                        if w == 30:
                            out.loc[g.index, key + '_count'] = counts[w]
    return out

if __name__ == '__main__':
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-gap', type=float, default=6000)
    parser.add_argument('--output-prefix', default='')
    args = parser.parse_args()
    parts = []
    for path in sorted(Path('dataset').glob('training_*.parquet')):
        print(path, flush=True)
        parts.append(build(pd.read_parquet(path, columns=INPUTS), args.max_gap))
    out = Path('output/experiments/neighbor_proxy')
    pd.concat(parts, ignore_index=True).to_parquet(out / (args.output_prefix + 'training.parquet'), index=False)
    build(pd.read_parquet('dataset/ranking.parquet', columns=INPUTS), args.max_gap).to_parquet(
        out / (args.output_prefix + 'ranking.parquet'), index=False)
