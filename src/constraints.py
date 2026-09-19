"""Timing support observed in matched airport/NM departure records.

Across 2,062,448 nonnegative 2025 departure labels with LOBT, the difference
between airport block time and LOBT is in [-3606, 3606] seconds. This looks
like a one-hour matching window with timestamp jitter; it is an empirical
property, not a documented guarantee for new data. Use a full minute of slack.
No constraint is inferred from scheduled time when the flight-plan join failed.
"""
import numpy as np

LOBT_MARGIN_SEC = 3660.0


def check_lobt_support(y, gap_lobt, margin=LOBT_MARGIN_SEC):
    """Fail visibly if new labelled data invalidates the empirical assumption."""
    y, gap = np.asarray(y, dtype=float), np.asarray(gap_lobt, dtype=float)
    if y.ndim != 1 or y.shape != gap.shape:
        raise ValueError("y and gap_lobt must be equally sized one-dimensional arrays")
    matched = np.isfinite(y) & (y >= 0) & np.isfinite(gap)
    violations = matched & (np.abs(y - gap) > margin)
    if violations.any():
        raise ValueError(f"{int(violations.sum())} training labels violate the LOBT interval; "
                         "review the data or disable --lobt-bounds")


def project_lobt(pred, gap_lobt, margin=LOBT_MARGIN_SEC):
    """Project matched rows onto their feasible nonnegative taxi-time interval.

    If a target lies in this interval, projection cannot increase its squared
    error. Missing/nonfinite LOBT and impossible intervals leave the prediction
    alone. Inputs are positional, never aligned by pandas index.
    """
    pred = np.asarray(pred, dtype=float)
    gap = np.asarray(gap_lobt, dtype=float)
    if pred.ndim != 1 or pred.shape != gap.shape:
        raise ValueError("pred and gap_lobt must be equally sized one-dimensional arrays")
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    valid = np.isfinite(gap) & (gap + margin >= 0)
    out = pred.copy()
    out[valid] = np.clip(pred[valid], np.maximum(0.0, gap[valid] - margin),
                         gap[valid] + margin)
    return out
