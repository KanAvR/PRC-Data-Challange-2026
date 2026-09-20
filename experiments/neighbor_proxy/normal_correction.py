"""Transfer a normal-model improvement without replacing tail predictions."""
import numpy as np
from constraints import LOBT_MARGIN_SEC, project_lobt

def apply_normal_correction(pred, gap_lobt, gap_sched, p_missing, baseline_mu, new_mu, weight=.25):
    pred, gap, sched, p, base, new = [np.asarray(v, dtype=float) for v in
        (pred, gap_lobt, gap_sched, p_missing, baseline_mu, new_mu)]
    if pred.ndim != 1 or any(v.shape != pred.shape for v in (gap, sched, p, base, new)):
        raise ValueError('all inputs must be equal-length vectors')
    if not 0 <= weight <= 1 or not all(np.isfinite(v).all() for v in (pred, base, new)):
        raise ValueError('invalid correction weight or nonfinite prediction')
    missing = ~np.isfinite(gap)
    if not np.isfinite(p[missing]).all() or ((p[missing] < 0) | (p[missing] > 1)).any():
        raise ValueError('missing-LOBT probabilities must lie in [0,1]')
    lo, hi = np.maximum(0, gap - LOBT_MARGIN_SEC), gap + LOBT_MARGIN_SEC
    matched = (~missing) & (gap <= 6000) & (pred > lo + 1e-6) & (pred < hi - 1e-6)
    out = pred.copy()
    out[matched] = project_lobt(pred[matched] + weight * (new[matched] - base[matched]), gap[matched])
    # Below the schedule gate the mixture's prediction is entirely mu. Above
    # it only (1-p) of a normal-model change enters the final prediction.
    normal_share = np.where(np.nan_to_num(sched[missing], nan=0) > 6000, 1 - p[missing], 1)
    out[missing] = np.maximum(0, pred[missing] + weight * normal_share * (new[missing] - base[missing]))
    return out
