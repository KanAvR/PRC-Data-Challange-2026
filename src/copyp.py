"""--clf-copy-p: the schedule-copy probability, learned on normal rows, for the tail.

LIRF records the scheduled time as block time on some delayed departures
(NEXT_STEPS.md section 5, B17, B21). On the no-flight-plan tail candidates that
copy IS the tail label, but the tail classifier sees only ~900 LIRF candidates a
year. The same recording habit shows on ~20k delayed *normal* rows, where the
label sits within SCHED_COPY_S of gap_SCHED. A classifier fitted there learns who
gets copied (operator, stand, hour, how late the inbound aircraft was) from far
more rows; its probability is handed to the no-LOBT tail classifier as one
column.

Inputs are limited to columns unmatched rows also have, and the raw delay
(gap_SCHED) is left out: normal rows stop at 6000s while tail candidates start
there, so the classifier must learn the habit, not the delay. Fit rows get the
probability out of month (a model that never saw their month), apply rows the
full fit, like the target encodings.
"""
import numpy as np

import lightgbm as lgb

from model import (CAT_COLS, NO_LOBT_FEATURES, SCHED_CLF_PARAMS, SCHED_COPY_S, TAIL_THR,
                   with_inbound)

COL = "p_sched_copy"
DELAYED_S = 1800   # copies on on-time flights are not the habit, just punctuality
ROUNDS = 300
FEATURES = [c for c in NO_LOBT_FEATURES if c != "gap_SCHED_TIME_UTC_mvt"] + [
    "secs_since_prev_arr_block", "inbound_after_sched"]


def _cols(X):
    return [c for c in FEATURES if c in X.columns]


def _fit(X, label):
    cols = _cols(X)
    cats = [c for c in CAT_COLS if c in cols]
    return lgb.train(SCHED_CLF_PARAMS, lgb.Dataset(X[cols], label=label, categorical_feature=cats),
                     num_boost_round=ROUNDS), cols


def add_copy_p(X_fit, y_fit, months, *X_apply):
    """Add COL to X_fit (out of month) and to each frame in X_apply (full fit)."""
    Xf = with_inbound(X_fit)
    g = Xf["gap_SCHED_TIME_UTC_mvt"].to_numpy(dtype=float)
    train = (y_fit <= TAIL_THR) & np.isfinite(g) & (g > DELAYED_S)
    label = (np.abs(y_fit - np.nan_to_num(g)) <= SCHED_COPY_S).astype(int)
    months = np.asarray(months)
    p = np.full(len(Xf), np.nan)
    for m in np.unique(months):
        sel = months == m
        booster, cols = _fit(Xf[train & ~sel], label[train & ~sel])
        p[sel] = booster.predict(Xf.loc[sel, cols])
    X_fit[COL] = p
    booster, cols = _fit(Xf[train], label[train])
    for Xa in X_apply:
        if Xa is not None:
            Xa[COL] = booster.predict(with_inbound(Xa)[cols])
