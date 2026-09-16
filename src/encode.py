"""Out-of-fold target encodings: historical mean taxi time per key.

LightGBM splits a 5000-level categorical such as ADEP_STAND_RUNWAY by sorting
levels on gradient statistics of the current node, which is noisy for rare
levels. A smoothed per-level mean computed out of fold gives it the same
information as a single ordered numeric column. The flight-number key is not a
model input at all otherwise.

Encodings are fitted on normal rows only (y <= TAIL_THR): a single 80000s tail
row would otherwise define its level's mean. Training rows are encoded
out-of-fold by month, so a row never sees its own month's labels; rows being
predicted are encoded from every training row.
"""
import numpy as np
import pandas as pd

from model import TAIL_THR

PRIOR_WEIGHT = 30

# name -> columns of X (or "flight", which comes from the raw data)
TE_KEYS = {
    "te_stand_rwy": ["ADEP_STAND_RUNWAY"],
    "te_stand": ["ADEP_STAND"],
    "te_airline": ["ADEP_AIRLINE"],
    "te_actype": ["ADEP_AIRCRAFT_TYPE"],
    "te_flight": ["flight"],
    "te_rwy_hour": ["ADEP_RUNWAY", "hour_int"],
}


def _key(X, flight, cols):
    parts = []
    for c in cols:
        if c == "flight":
            parts.append(pd.Series(flight, index=X.index).astype(str))
        elif c == "hour_int":
            parts.append(X["hour"].astype(int).astype(str))
        else:
            parts.append(X[c].astype(str))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key.values


def _encode(k_fit, y, folds, k_apply):
    normal = y <= TAIL_THR
    prior = y[normal].mean()
    codes, uniq = pd.factorize(np.concatenate([k_fit, k_apply]))
    c_fit, c_apply = codes[:len(k_fit)], codes[len(k_fit):]
    n_lvl = len(uniq)

    fold_ids, fold_of = np.unique(folds, return_inverse=True)
    s = np.zeros((len(fold_ids), n_lvl))
    n = np.zeros((len(fold_ids), n_lvl))
    np.add.at(s, (fold_of[normal], c_fit[normal]), y[normal])
    np.add.at(n, (fold_of[normal], c_fit[normal]), 1)
    s_all, n_all = s.sum(axis=0), n.sum(axis=0)

    def smooth(ss, nn):
        return (ss + PRIOR_WEIGHT * prior) / (nn + PRIOR_WEIGHT)

    s_oof = s_all[c_fit] - s[fold_of, c_fit]
    n_oof = n_all[c_fit] - n[fold_of, c_fit]
    fit_mean, fit_n = smooth(s_oof, n_oof), n_oof
    app_mean, app_n = smooth(s_all[c_apply], n_all[c_apply]), n_all[c_apply]
    return (fit_mean, np.log1p(fit_n)), (app_mean, np.log1p(app_n))


def add_target_encodings(X_fit, y_fit, folds, flight_fit, X_apply, flight_apply):
    """Add te_* columns to both frames in place."""
    for name, cols in TE_KEYS.items():
        (fm, fn), (am, an) = _encode(_key(X_fit, flight_fit, cols), y_fit, np.asarray(folds),
                                     _key(X_apply, flight_apply, cols))
        X_fit[name], X_fit[name + "_n"] = fm, fn
        X_apply[name], X_apply[name + "_n"] = am, an
