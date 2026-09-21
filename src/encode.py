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

# Shrinkage toward the global prior. 30 was never tested and is too strong:
# sweeping the axis on the four-split harness gives a broad plateau over 3-10
# (mean -0.345 to -0.215 on normal rows) decaying to nothing by 30, while 200
# costs +0.24 and dropping the encodings entirely (--no-te) costs +0.96 -- so
# the encodings are worth ~1.1 RMSE and want MORE say, not less. 10 is chosen
# over the marginally better 3 as the conservative end of the plateau: more
# shrinkage means less risk of overfitting rare levels on an unseen year.
# Unlike a new ensemble member this survives averaging (two-member harness
# -0.20 pooled robust against -0.23 for LightGBM alone), because it changes the
# te_* columns that every mu family consumes. Live: v14 268.4067 -> v16 267.777
# on an otherwise identical pipeline, i.e. -0.63 RMSE.
PRIOR_WEIGHT = 10

# name -> columns of X (or "flight", which comes from the raw data)
TE_KEYS = {
    "te_stand_rwy": ["ADEP_STAND_RUNWAY"],
    "te_stand": ["ADEP_STAND"],
    "te_airline": ["ADEP_AIRLINE"],
    "te_actype": ["ADEP_AIRCRAFT_TYPE"],
    "te_flight": ["flight"],
    "te_rwy_hour": ["ADEP_RUNWAY", "hour_int"],
}

# Candidate additions, opt-in via --te-keys. Each crosses two things the shipped
# set only holds separately: the stand-to-runway route by time of day, the
# aircraft type by the runway it is sent to, and the operator by runway (an
# airline's stands cluster, so its taxi differs by departure direction).
EXTRA_TE_KEYS = {
    "te_stand_rwy_hour": ["ADEP_STAND_RUNWAY", "hour_int"],
    "te_actype_rwy": ["ADEP_AIRCRAFT_TYPE", "RUNWAY_mvt"],
    "te_airline_rwy": ["ADEP_AIRLINE", "RUNWAY_mvt"],
}

KEY_SETS = {
    "base": list(TE_KEYS),
    "plus_stand_rwy_hour": list(TE_KEYS) + ["te_stand_rwy_hour"],
    "plus_actype_rwy": list(TE_KEYS) + ["te_actype_rwy"],
    "plus_airline_rwy": list(TE_KEYS) + ["te_airline_rwy"],
    "plus_all": list(TE_KEYS) + list(EXTRA_TE_KEYS),
    # ablations: does each shipped key earn its place?
    "drop_flight": [k for k in TE_KEYS if k != "te_flight"],
    "drop_rwy_hour": [k for k in TE_KEYS if k != "te_rwy_hour"],
    "drop_actype": [k for k in TE_KEYS if k != "te_actype"],
}

ALL_KEYS = {**TE_KEYS, **EXTRA_TE_KEYS}


def _fold_ids(folds, size):
    """Coarsen the out-of-fold grouping. size=1 is the shipped month-level
    scheme; larger blocks make a row's encoding exclude more neighbouring
    months, which is stricter but leaves less data behind each level."""
    folds = np.asarray(folds)
    if size <= 1:
        return folds
    uniq = np.unique(folds)  # month strings sort chronologically
    block = {m: i // size for i, m in enumerate(uniq)}
    return np.array([block[f] for f in folds])


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


def add_target_encodings(X_fit, y_fit, folds, flight_fit, X_apply, flight_apply,
                         keys=None, counts=True, fold_size=1):
    """Add te_* columns to both frames in place.

    keys: names from KEY_SETS' vocabulary (default: the shipped TE_KEYS).
    counts: also add the _n log-count column beside each encoding.
    fold_size: months per out-of-fold block (1 = the shipped scheme).
    """
    names = list(TE_KEYS) if keys is None else list(keys)
    ids = _fold_ids(folds, fold_size)
    for name in names:
        cols = ALL_KEYS[name]
        (fm, fn), (am, an) = _encode(_key(X_fit, flight_fit, cols), y_fit, ids,
                                     _key(X_apply, flight_apply, cols))
        X_fit[name], X_apply[name] = fm, am
        if counts:
            X_fit[name + "_n"], X_apply[name + "_n"] = fn, an
