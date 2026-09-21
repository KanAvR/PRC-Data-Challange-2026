import numpy as np

def normal_anchor(frame, limit=6000):
    """Own AOBT proxy, with a nearby-flight fallback when the report is absent."""
    own = frame.gap_AOBT_3_flt.to_numpy(dtype=float)
    total = np.zeros(len(frame))
    count = np.zeros(len(frame))
    for direction in ['before', 'after']:
        key = 'proxy_apt_AOBT_3_flt_' + direction + '_30'
        mean = frame[key].to_numpy(dtype=float)
        n = frame[key + '_count'].to_numpy(dtype=float)
        valid = np.isfinite(mean) & np.isfinite(n) & (n > 0)
        total[valid] += mean[valid] * n[valid]
        count[valid] += n[valid]
    peer = np.divide(total, count, out=np.zeros(len(frame)), where=count > 0)
    return np.clip(np.where(np.isfinite(own), own, peer), 0, limit)
