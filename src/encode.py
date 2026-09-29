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
import zlib

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
        elif c == "slot":
            # scheduled local time of day, in 30-minute slots: a renamed callsign
            # keeps its route and its slot in the timetable
            sched_min = (X["hour_local"] * 60 - X["gap_SCHED_TIME_UTC_mvt"] / 60) % 1440
            parts.append((sched_min // 30).fillna(-1).astype(int).astype(str))
        else:
            parts.append(X[c].astype(str))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key.values


def _encode(k_fit, y, folds, k_apply, unseen_apply=None, prior_fit=None, prior_apply=None,
            drop_fit=None, count_oof_basis=False):
    """unseen_apply: apply rows to encode as if their level were never seen,
    the way a renamed callsign is at serve time (see add_target_encodings).
    prior_fit / prior_apply: per-row prior to shrink toward (default: the global
    mean), for a hierarchical backoff. drop_fit: fit rows to encode as unseen,
    so the model trains on the regime serving will show it."""
    normal = y <= TAIL_THR
    glob = y[normal].mean()
    prior_fit = np.full(len(k_fit), glob) if prior_fit is None else prior_fit
    prior_apply = np.full(len(k_apply), glob) if prior_apply is None else prior_apply
    codes, uniq = pd.factorize(np.concatenate([k_fit, k_apply]))
    c_fit, c_apply = codes[:len(k_fit)], codes[len(k_fit):]
    n_lvl = len(uniq)

    fold_ids, fold_of = np.unique(folds, return_inverse=True)
    s = np.zeros((len(fold_ids), n_lvl))
    n = np.zeros((len(fold_ids), n_lvl))
    np.add.at(s, (fold_of[normal], c_fit[normal]), y[normal])
    np.add.at(n, (fold_of[normal], c_fit[normal]), 1)
    s_all, n_all = s.sum(axis=0), n.sum(axis=0)

    def smooth(ss, nn, prior):
        return (ss + PRIOR_WEIGHT * prior) / (nn + PRIOR_WEIGHT)

    s_oof = s_all[c_fit] - s[fold_of, c_fit]
    n_oof = n_all[c_fit] - n[fold_of, c_fit]
    fit_mean, fit_n = smooth(s_oof, n_oof, prior_fit), n_oof
    if drop_fit is not None:
        fit_mean, fit_n = np.where(drop_fit, prior_fit, fit_mean), np.where(drop_fit, 0.0, fit_n)
    app_mean = smooth(s_all[c_apply], n_all[c_apply], prior_apply)
    app_n = n_all[c_apply]
    if count_oof_basis:
        # Fit rows never see their own fold's labels. Put apply-side counts on
        # the exact average OOF basis for each level, including levels whose
        # observations are concentrated in a small number of months.
        sq = np.square(n).sum(axis=0)
        effective = n_all - np.divide(sq, n_all, out=np.zeros_like(n_all), where=n_all > 0)
        app_n = effective[c_apply]
    masked = None
    if unseen_apply is not None:
        masked = (np.where(unseen_apply, prior_apply, app_mean),
                  np.log1p(np.where(unseen_apply, 0.0, app_n)))
    return (fit_mean, np.log1p(fit_n)), (app_mean, np.log1p(app_n)), masked


# --adsb-offset-te: the label minus the departure's own trace stand exit is a
# steady offset per airport (LEMD median -124s, EHAM -8s) that plausibly also
# varies by stand (contact gate vs remote) and operator (who records the block
# time, and when). Encode it out of fold, the stand and operator levels shrunk
# toward the airport's, and hand the model the corrected estimate directly.
OFFSET_ANCHOR = "adsb_leave_stand"
OFFSET_KEYS = {"te_off_stand": ["ADEP_STAND"], "te_off_airline": ["ADEP_AIRLINE"]}
OFFSET_COLS = ["te_off_apt", *OFFSET_KEYS, "adsb_exit_off_stand"]
OFFSET_ANCHOR_CLIP = 3600.0
OFFSET_CLIP = 900.0  # a tow or a wrong match must not drag a stand's mean


def add_offset_encodings(X_fit, y_fit, folds, X_apply, X_apply_masked=None):
    """Add OFFSET_COLS to both frames in place (needs --adsb columns)."""
    def anchor(frame):
        return np.clip(frame[OFFSET_ANCHOR].to_numpy(dtype=float), 0, OFFSET_ANCHOR_CLIP)
    a_fit = anchor(X_fit)
    ok = np.isfinite(a_fit) & (y_fit <= TAIL_THR)
    # rows without a stand exit leave the fit by carrying a "tail" target
    o = np.where(ok, np.clip(y_fit - np.nan_to_num(a_fit), -OFFSET_CLIP, OFFSET_CLIP), TAIL_THR + 1.0)
    ids = np.asarray(folds)
    (apt_f, _), (apt_a, _), _ = _encode(_key(X_fit, None, ["ADEP_mvt"]), o, ids,
                                        _key(X_apply, None, ["ADEP_mvt"]))
    cols = {"te_off_apt": (apt_f, apt_a)}
    for name, keycols in OFFSET_KEYS.items():
        (fm, _), (am, _), _ = _encode(_key(X_fit, None, keycols), o, ids, _key(X_apply, None, keycols),
                                      prior_fit=apt_f, prior_apply=apt_a)
        cols[name] = (fm, am)
    for frame, i in [(X_fit, 0)] + [(f, 1) for f in (X_apply, X_apply_masked) if f is not None]:
        for name, pair in cols.items():
            frame[name] = pair[i]
        frame["adsb_exit_off_stand"] = anchor(frame) + frame["te_off_stand"]


# Backoff for te_flight when --te-flight-backoff is on: slot, then route, then
# the global mean. Each level is shrunk toward the next.
FLIGHT_BACKOFF = [["ADEP_AIRLINE", "ADES_mvt"], ["ADEP_AIRLINE", "ADES_mvt", "slot"]]


def _hashed(keys, frac, salt):
    """Whole keys selected by a stable hash, so every row of a level goes together."""
    h = np.array([zlib.crc32((salt + str(k)).encode()) % 10000 for k in keys])
    return h < frac * 10000


def add_target_encodings(X_fit, y_fit, folds, flight_fit, X_apply, flight_apply,
                         keys=None, counts=True, fold_size=1,
                         X_apply_masked=None, flight_unseen=None,
                         flight_backoff=False, flight_dropout=0.0, airport_prior=False,
                         count_oof_basis=False):
    """Add te_* columns to both frames in place.

    keys: names from KEY_SETS' vocabulary (default: the shipped TE_KEYS).
    counts: also add the _n log-count column beside each encoding.
    fold_size: months per out-of-fold block (1 = the shipped scheme).
    X_apply_masked, flight_unseen: evaluation only. X_apply_masked (a copy of
    X_apply) gets the same columns, except that on rows where flight_unseen is
    True every flight-keyed encoding is the unseen fallback. Airlines reissue
    alphanumeric callsigns each IATA season, so 37% of July 2026 flights are
    absent from 2025 while only ~2% of a 2025 validation month is; this is how
    validation sees that.
    flight_backoff: an unseen flight falls back to its slot (airport, airline,
    destination, scheduled 30-min slot), then its route, not the global mean.
    flight_dropout: that fraction of flight keys, by stable hash, is encoded as
    unseen on the fit rows too, so the model learns the fallback regime.
    airport_prior: every key shrinks toward its airport's mean instead of the
    global one (EDDM has 4.6% of ranking rows on stands 2025 never saw).
    """
    names = list(TE_KEYS) if keys is None else list(keys)
    ids = _fold_ids(folds, fold_size)
    apt_fit = apt_app = None
    if airport_prior:
        (apt_fit, _), (apt_app, _), _ = _encode(_key(X_fit, flight_fit, ["ADEP_mvt"]), y_fit, ids,
                                                _key(X_apply, flight_apply, ["ADEP_mvt"]))
    back_fit, back_app = apt_fit, apt_app
    if flight_backoff:
        for cols in FLIGHT_BACKOFF:
            (back_fit, _), (back_app, _), _ = _encode(
                _key(X_fit, flight_fit, cols), y_fit, ids, _key(X_apply, flight_apply, cols),
                prior_fit=back_fit, prior_apply=back_app)
    for name in names:
        cols = ALL_KEYS[name]
        is_flight = "flight" in cols
        mask = flight_unseen if (X_apply_masked is not None and is_flight) else None
        drop = _hashed(flight_fit, flight_dropout, "dropout|") if (is_flight and flight_dropout > 0) else None
        (fm, fn), (am, an), masked = _encode(_key(X_fit, flight_fit, cols), y_fit, ids,
                                             _key(X_apply, flight_apply, cols), mask,
                                             prior_fit=back_fit if is_flight else apt_fit,
                                             prior_apply=back_app if is_flight else apt_app,
                                             drop_fit=drop,
                                             count_oof_basis=count_oof_basis)
        X_fit[name], X_apply[name] = fm, am
        if counts:
            X_fit[name + "_n"], X_apply[name + "_n"] = fn, an
        if X_apply_masked is not None:
            mm, mn = masked if masked is not None else (am, an)
            X_apply_masked[name] = mm
            if counts:
                X_apply_masked[name + "_n"] = mn
