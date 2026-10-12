"""--trace-spec: in the long-taxi specialist's domain, trust the own trace's stand exit.

The specialist (high_lobt.py) fits y - gap_LOBT on ~360 matched long-taxi rows a
year, ~40 of them with a trace stand exit, so it cannot learn that the exit is
the answer. Where it was seen, it is: on the four 2025 splits the exit plus the
airport's median offset was within 300s of the label on every EHAM, EDDM and
EDDF specialist row (RMSE 58 / 50 / 30 against the specialist's 799 / 1074 /
1017; NEXT_STEPS.md, B47). Those rows are de-icing days (EHAM 2025-01-05 and
01-18), which recur in 2026 (EHAM 2026-01-02..07).

Which airports to trust is fitted from the training labels: an airport's long
exits (LONG_EXIT_SEC..NORMAL_MAX_SEC) must land within TOL_SEC of the label on
at least MIN_SHARE of at least MIN_ROWS rows. On 2025 that picks EHAM, EDDF and
EDDM (EGLL 0.29-0.40, LSZH 0.17-0.33, LIRF's labels are not the physical taxi).
The estimate is projected into the LOBT window like every matched prediction.
"""
import numpy as np
import pandas as pd

from constraints import project_lobt

EXIT_COL = "adsb_leave_stand"
LONG_EXIT_SEC = 3600.0
NORMAL_MAX_SEC = 6000.0
TOL_SEC = 300.0
MIN_ROWS = 5
MIN_SHARE = 0.8


class TraceTrust:
    def __init__(self, offset, reliable, table):
        self.offset, self.reliable, self.table = offset, reliable, table

    def estimate(self, X):
        """Exit plus the airport's offset, NaN where no exit or no trusted airport."""
        if EXIT_COL not in X:
            return np.full(len(X), np.nan)
        apt = X["ADEP_mvt"].astype(str)
        off = apt.map(self.offset).where(apt.isin(self.reliable))
        return (X[EXIT_COL].astype(float) + off.astype(float)).to_numpy()


def fit_trace_trust(X, y):
    """Per-airport offset (median label - exit on normal rows) and the trusted airports."""
    y = np.asarray(y, dtype=float)
    if EXIT_COL not in X:
        return TraceTrust({}, [], pd.DataFrame())
    ex = X[EXIT_COL].to_numpy(dtype=float)
    apt = X["ADEP_mvt"].astype(str).to_numpy()
    seen = np.isfinite(ex) & np.isfinite(y)
    nrm = seen & (y <= NORMAL_MAX_SEC)
    offset = pd.Series(y[nrm] - ex[nrm]).groupby(apt[nrm]).median()
    long_ = seen & (ex > LONG_EXIT_SEC) & (ex <= NORMAL_MAX_SEC)
    off_long = pd.Series(apt[long_]).map(offset).to_numpy(dtype=float)
    hit = np.abs(y[long_] - ex[long_] - off_long) <= TOL_SEC
    table = pd.Series(hit).groupby(apt[long_]).agg(["size", "mean"])
    reliable = table[(table["size"] >= MIN_ROWS) & (table["mean"] >= MIN_SHARE)].index.tolist()
    return TraceTrust(offset.to_dict(), reliable, table)


def apply_trace_spec(pred, X, trust, rows):
    """pred with the specialist's rows (mask `rows`) replaced by the projected trace
    estimate where one exists at a trusted airport. Returns (pred, changed mask)."""
    out = np.asarray(pred, dtype=float).copy()
    est = trust.estimate(X)
    use = np.asarray(rows, bool) & np.isfinite(est)
    if use.any():
        gap = X["gap_LOBT_flt"].to_numpy(dtype=float)
        out[use] = project_lobt(est[use], gap[use])
    return out, use
