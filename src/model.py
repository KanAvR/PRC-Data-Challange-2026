"""Two-component model for taxi-out time.

The normal regressor is trained without labels above 6000 seconds and shrinks
extreme taxi times. The mixture below supplies a separate analytic tail value.
The default pipeline then uses high_lobt.py to refine matched long departures
with a bounded residual model; unmatched departures retain this mixture.

    E[y|x] = (1 - p(x)) * mu(x) + p(x) * v(x)

mu  regressor on normal rows only, where a GBM is well behaved
p   P(target > TAIL_THR | x); needs a SMALL feature set -- with all 45 features
    it memorises the ~600 training positives and collapses to AUC 0.52
v   analytic tail value: gap_LOBT where LOBT exists (RMSE 1719 on those rows),
    else gap_SCHED (within 60s on 79.5% of the rows where LOBT is absent)
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import CAT_COLS, PROXY_MAX_SEC

TAIL_THR = 6000
N_SEEDS = 5  # classifier AUC swings ~0.015 between seeds, worth ~7 RMSE points

# deliberately small: more features destroy this classifier. Adding the runway
# queue / neighbour-delay features costs ~10 RMSE -- they encode which specific
# disruption episode a row belongs to, and those episodes do not recur.
TAIL_FEATURES = [
    "gap_LOBT_flt", "gap_SCHED_TIME_UTC_mvt", "gap_IOBT_flt", "gap_EOBT_1_flt",
    "gap_AOBT_3_flt", "missing_LOBT_flt", "missing_AOBT_3_flt", "n_missing_times",
    "hour", "dow", "month", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "AIRLINE_mvt",
]

# Rows without LOBT are ~65% of the tail and are the hard ones. Pooled, they get
# swamped by the LOBT-present rows where gap_LOBT nearly gives the answer away.
# A dedicated model for them wins on all four temporal splits.
NO_LOBT_FEATURES = [
    "gap_SCHED_TIME_UTC_mvt", "hour", "dow", "month", "ADEP_mvt",
    "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "n_missing_times",
    "AIRLINE_mvt",
]

MU_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.02, num_leaves=511,
    min_data_in_leaf=30, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1,
    force_col_wise=True,  # 63 features x 1.5M rows: skip the auto row/col probe
)

CLF_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
    min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.8,
    bagging_freq=1, verbose=-1,
)


def analytic_tail_value(X):
    """MVT_TIME minus the block time that was actually recorded on these rows."""
    sched = np.clip(np.nan_to_num(X["gap_SCHED_TIME_UTC_mvt"].values.astype(float), nan=0.0), 0, None)
    lobt = X["gap_LOBT_flt"].values.astype(float)
    v = np.where(~np.isnan(lobt), np.nan_to_num(lobt, nan=0.0), sched)
    return np.clip(v, 0, None)


# A few takeoff times are stamped a day late, so the label is a normal taxi plus
# 86400s (block 15:29, takeoff 16:03 the next day). 14 such rows in 2025 carry 56%
# of the tail's squared error. Most have no LOBT and a gap_SCHED of 14-30h. Below
# 14h the no-LOBT tails are almost all v-explained (block = schedule), which a
# 12h bound wrongly pulled towards DAY; the bound was set on months other than
# Jan/Jul so the holdout stays a clean check.
DAY = 86400.0
DAY_WINDOW = (14 * 3600.0, 30 * 3600.0)


def day_late_candidates(X):
    gs = X["gap_SCHED_TIME_UTC_mvt"].values.astype(float)
    return (~_has_lobt(X)) & (gs > DAY_WINDOW[0]) & (gs < DAY_WINDOW[1])


def fit_day_late_rate(X, y):
    """P(day-late | tail, candidate), Laplace-smoothed: only ~30 rows inform it."""
    cand_tail = day_late_candidates(X) & (y > TAIL_THR)
    n_day = int(((y >= DAY) & (y < DAY + TAIL_THR) & cand_tail).sum())
    return (n_day + 1) / (int(cand_tail.sum()) + 2)


def day_late_weight(models, X):
    return np.where(day_late_candidates(X), models["q_day"], 0.0)


# Day-late labels sit at DAY + 770..1,990s in 2025 (median ~960) whatever mu
# says for the row, so a fixed base beats DAY + mu (mu is 1.4-3k on these rows).
DAY_LATE_BASE = 960.0
# P(day-late | tail candidate) rises steeply with gap_SCHED. Unmatched 2025 rows
# with a corrupted callsign (ITYTY680 for ITY680): 0 of 35 below 50k, 6 of 10 at
# 50-66k, 3 of 3 at 66-100k; the rest have label = gap_SCHED. One constant rate
# over the whole window is too high low down and too low high up.
DAY_STEP_EDGES = (DAY_WINDOW[0], 66000.0, DAY_WINDOW[1])


def fit_day_late_steps(X, y):
    """P(day-late | tail, candidate) per gap_SCHED step, Laplace-smoothed."""
    gs = X["gap_SCHED_TIME_UTC_mvt"].values.astype(float)
    cand_tail = day_late_candidates(X) & (y > TAIL_THR)
    is_day = (y >= DAY) & (y < DAY + TAIL_THR)
    rates = []
    for lo, hi in zip(DAY_STEP_EDGES[:-1], DAY_STEP_EDGES[1:]):
        sel = cand_tail & (gs >= lo) & (gs < hi)
        rates.append((int((sel & is_day).sum()) + 1) / (int(sel.sum()) + 2))
    return tuple(rates)


def day_late_step_weight(models, X):
    gs = X["gap_SCHED_TIME_UTC_mvt"].values.astype(float)
    cand = day_late_candidates(X)
    q = np.zeros(len(X))
    for (lo, hi), rate in zip(zip(DAY_STEP_EDGES[:-1], DAY_STEP_EDGES[1:]), models["q_day_steps"]):
        q[cand & (gs >= lo) & (gs < hi)] = rate
    return q


def tail_options(args, models, X):
    """combine() keyword arguments for the day-late and cap options.

    Shared by validate.py and predict.py, so a validated configuration is the one
    that ships."""
    cap = bool(getattr(args, "tail_cap", False))
    if getattr(args, "day_late_step", False):
        return {"q_day": day_late_step_weight(models, X), "day_value": DAY + DAY_LATE_BASE, "cap": cap}
    if getattr(args, "day_late", False):
        return {"q_day": day_late_weight(models, X), "cap": cap}
    return {"cap": cap}


# More departures ahead of you cannot make your taxi shorter. Constraining the
# congestion and queue counts to be monotone increasing is free regularisation:
# it removes a family of fits the physics forbids, which may generalise better to
# an unseen year. Deliberately limited to the counts where "more implies slower"
# is near-tautological -- NOT the gap_SCHED neighbour means or arrival taxi
# means, whose direction is arguable.
MONOTONE_UP = (
    ["queue_ahead_apt", "queue_ahead_rwy"]
    + [f"dep_off_apt_{w}m" for w in (5, 15, 30, 60, 120)]
    + [f"dep_off_rwy_{w}m" for w in (5, 15, 30, 60)]
)


def monotone_vector(columns):
    """LightGBM wants one entry per feature, in column order."""
    return [1 if c in MONOTONE_UP else 0 for c in columns]


def seasonal_weight(train_months, valid_months, tau):
    """Up-weight training months seasonally close to the months being scored.

    Every month is currently weighted equally and the model is handed `month` to
    sort it out. The ranking set is Jan+Jul, so months adjacent to those may
    deserve more say. Distance is circular (December is one month from January)
    and the weight decays as exp(-d / tau), normalised to mean 1 so the effective
    sample size does not change.
    """
    tm = np.asarray([int(m.split("-")[1]) for m in train_months])
    vm = [int(m.split("-")[1]) for m in valid_months]
    d = np.min(np.stack([np.minimum(np.abs(tm - v), 12 - np.abs(tm - v)) for v in vm]), axis=0)
    w = np.exp(-d / float(tau))
    return w / w.mean()


def train_mu(X, y, num_boost_round=900, monotone=False, weight=None):
    normal = y <= TAIL_THR
    ds = lgb.Dataset(X[normal], label=y[normal], categorical_feature=CAT_COLS,
                     weight=None if weight is None else np.asarray(weight)[normal])
    params = dict(MU_PARAMS)
    if monotone:
        params["monotone_constraints"] = monotone_vector(list(X.columns))
    return lgb.train(params, ds, num_boost_round=num_boost_round)


# LightGBM splits a categorical by sorting its levels on the node's gradient
# statistics, which with thousands of levels (ADEP_STAND_RUNWAY 12038,
# ADEP_AIRLINE 5508) memorises rare levels. Withholding those keys -- the te_*
# encodings still carry them -- takes the inner normal-row RMSE from 228.7 to
# 223.3 at 1500 rounds and halves the fit time (bench_mu.py; stronger
# cat_smooth/cat_l2/min_data_per_group only reached 226.4). But the native
# splits are also what made LightGBM differ from CatBoost: its residual
# correlation with cat rises from 0.944 to 0.966, and in the four-member
# FINAL_MU average the swap is a wash on every split (pooled robust +0.01),
# as is adding it as a fifth member (-0.01). Kept as an opt-in family only.
LGB_DROP = ["ADEP_STAND_RUNWAY", "ADEP_AIRLINE", "ADEP_STAND", "STAND_mvt", "AIRLINE_mvt",
            "ADES_mvt", "ADEP_AIRCRAFT_TYPE", "AIRCRAFT_OPERATOR_flt"]


def train_mu_nocat(X, y, num_boost_round=1500):
    drop = [c for c in LGB_DROP if c in X.columns]
    Xl = X.drop(columns=drop)
    cats = [c for c in CAT_COLS if c in Xl.columns]
    normal = y <= TAIL_THR
    ds = lgb.Dataset(Xl[normal], label=y[normal], categorical_feature=cats)
    return _DropCols(lgb.train(MU_PARAMS, ds, num_boost_round=num_boost_round), drop)


# A GBM is piecewise constant, so "the label equals this timestamp gap" costs it
# many splits to approximate. Re-targeting on y - gap makes that exactly
# representable, which is the mechanism behind high_lobt.py. Measured on the
# inner split at 900 rounds, normal rows: direct 228.83, residual on
# gap_AOBT_3_flt 227.28, on gap_LOBT_flt 228.35, on gap_EOBT_1_flt 228.42;
# averaging the direct model with the AOBT residual gives 226.69 and all four
# 226.20. Most of that is ensemble diversity, not the anchoring itself: on
# AOBT-matched rows the anchored model reaches 130.7 against the direct 133.0.
# INSIDE THE FOUR-MEMBER AVERAGE IT IS A WASH, like lgb_nocat: swapping lgb for
# the anchored member moves the mean normal-row RMSE over four splits from
# 211.67 to 211.41, and adding it as a fifth member to 211.65. blend.py calls
# the swap a win (pooled robust -1.34) but that is three extreme tail rows, not
# prediction quality -- see the mu-change caveat in NEXT_STEPS.md. Opt-in only.
ANCHOR_DEFAULT = "gap_AOBT_3_flt"

# The teammate's v17/v18 chain anchors on a different quantity, and the
# difference was not buildable here until the proxy features landed: where
# AOBT_3 is absent it falls back to the count-weighted mean of nearby
# departures' AOBT_3 gaps, then clips to [0, PROXY_MAX_SEC]. ANCHOR_DEFAULT is
# NaN on those ~1.5% of rows, so train_mu_anchor drops them to the plain target
# -- exactly the unmatched rows where an anchor would help most. The clip also
# stops a single long gap from defining a residual target.
# Ported from experiments/neighbor_proxy/anchor.py::normal_anchor.
def neighbour_anchor(X):
    own = X["gap_AOBT_3_flt"].to_numpy(dtype=float)
    total = np.zeros(len(X))
    count = np.zeros(len(X))
    for direction in ("before", "after"):
        key = f"proxy_apt_AOBT_3_flt_{direction}_30"
        if key not in X.columns:  # --no-proxy: no peers to fall back on
            return np.clip(own, 0, PROXY_MAX_SEC)
        mean = X[key].to_numpy(dtype=float)
        n = X[key + "_count"].to_numpy(dtype=float)
        ok = np.isfinite(mean) & np.isfinite(n) & (n > 0)
        total[ok] += mean[ok] * n[ok]
        count[ok] += n[ok]
    peer = np.divide(total, count, out=np.zeros(len(X)), where=count > 0)
    return np.clip(np.where(np.isfinite(own), own, peer), 0, PROXY_MAX_SEC)


class _AnchorMu:
    """Adds the anchor back, so the booster only has to learn the residual."""

    def __init__(self, booster, anchor):
        self.booster, self.anchor = booster, anchor

    def predict(self, X):
        a = X[self.anchor].to_numpy(dtype=float)
        raw = self.booster.predict(X)
        return np.clip(np.where(np.isfinite(a), raw + np.nan_to_num(a), raw), 0, None)


def train_mu_anchor(X, y, anchor=ANCHOR_DEFAULT, num_boost_round=900):
    """LightGBM on y - anchor. Rows with a missing anchor keep the plain target,
    and predict() mirrors that, so the two halves stay consistent."""
    normal = y <= TAIL_THR
    a = X[anchor].to_numpy(dtype=float)
    target = y[normal] - np.nan_to_num(a[normal])
    ds = lgb.Dataset(X[normal], label=target, categorical_feature=CAT_COLS)
    return _AnchorMu(lgb.train(MU_PARAMS, ds, num_boost_round=num_boost_round), anchor)


def train_mu_anchor_lobt(X, y):
    return train_mu_anchor(X, y, anchor="gap_LOBT_flt")


class _NbrAnchorMu(_AnchorMu):
    """Anchor computed from the frame rather than read from a column."""

    def __init__(self, booster):
        self.booster, self.anchor = booster, None

    def predict(self, X):
        a = neighbour_anchor(X)
        return np.clip(self.booster.predict(X) + a, 0, None)


def train_mu_anchor_nbr(X, y, num_boost_round=900):
    """LightGBM on y - neighbour_anchor(X). The anchor is always finite, so
    unlike train_mu_anchor there is no plain-target half to keep consistent."""
    normal = y <= TAIL_THR
    a = neighbour_anchor(X)
    ds = lgb.Dataset(X[normal], label=y[normal] - a[normal], categorical_feature=CAT_COLS)
    return _NbrAnchorMu(lgb.train(MU_PARAMS, ds, num_boost_round=num_boost_round))


# Other GBM families for mu. Each handles the high-cardinality categoricals
# (STAND, ADEP_STAND_RUNWAY) differently -- CatBoost with ordered target
# statistics, XGBoost with partition splits -- which is the diversity a blend
# needs. Round counts come from early stopping on the inner split (bench_mu.py).
CAT_MU_PARAMS = dict(
    loss_function="RMSE", iterations=3000, learning_rate=0.08, depth=8,
    l2_leaf_reg=3.0, border_count=254, one_hot_max_size=4,
    thread_count=-1, verbose=0, allow_writing_files=False,
)

XGB_MU_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
    learning_rate=0.03, max_depth=0, grow_policy="lossguide", max_leaves=511,
    min_child_weight=30, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
    max_cat_to_onehot=1, max_bin=255, nthread=-1,
)
# Measured and not used: native categorical splits overfit the thousands-level
# keys. Inner normal-row RMSE bottoms near 258 by round 200 at lr 0.1 (LightGBM
# 232.8, CatBoost 223.8), and every blend including it is worse than without.
XGB_ROUNDS = 1500


def _as_catboost(X):
    cats = [c for c in CAT_COLS if c in X.columns]
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(str)
    return X, cats


class _CatMu:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(_as_catboost(X)[0])


class _XgbMu:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X, enable_categorical=True))


def train_mu_cat(X, y, iterations=None, eval_set=None, params=None):
    """params overrides CAT_MU_PARAMS (bench_mu.py tuning). An eval_set only
    records the learning curve; the model is never cut back to its best
    iteration on it."""
    from catboost import CatBoostRegressor, Pool
    normal = y <= TAIL_THR
    Xc, cats = _as_catboost(X[normal])
    params = dict(CAT_MU_PARAMS, **(params or {}))
    if iterations:
        params["iterations"] = iterations
    model = CatBoostRegressor(**params)
    evals = None
    if eval_set is not None:
        Xe, ye = eval_set
        evals = Pool(_as_catboost(Xe)[0], ye, cat_features=cats)
    model.fit(Pool(Xc, y[normal], cat_features=cats), eval_set=evals,
              use_best_model=False, metric_period=50)
    return _CatMu(model)


def train_mu_xgb(X, y, num_boost_round=XGB_ROUNDS, eval_set=None):
    import xgboost as xgb
    normal = y <= TAIL_THR
    dtrain = xgb.QuantileDMatrix(X[normal], y[normal], enable_categorical=True,
                                 max_bin=XGB_MU_PARAMS["max_bin"])
    evals = []
    if eval_set is not None:
        evals = [(xgb.DMatrix(eval_set[0], eval_set[1], enable_categorical=True), "valid")]
    booster = xgb.train(XGB_MU_PARAMS, dtrain, num_boost_round=num_boost_round,
                        evals=evals, verbose_eval=100 if evals else False)
    return _XgbMu(booster)


class _DropCols:
    def __init__(self, model, drop):
        self.model, self.drop = model, drop

    def predict(self, X):
        return self.model.predict(X.drop(columns=self.drop))


def train_mu_cat_note(X, y):
    """CatBoost without the te_* columns. It scores the same as with them, but its
    errors differ enough (r=0.97) that the two average to ~1.8 RMSE better."""
    drop = [c for c in X.columns if c.startswith("te_")]
    return _DropCols(train_mu_cat(X.drop(columns=drop), y), drop)


# Single-feature CTRs at depth 10: 3.6x faster per iteration than the default and
# the most different of the tuned CatBoosts. On the inner split it bottoms out at
# 3000-4500 iterations and overfits after.
CAT_CTR1_PARAMS = dict(max_ctr_complexity=1, depth=10, iterations=3500)


def train_mu_cat_ctr1(X, y):
    return train_mu_cat(X, y, params=CAT_CTR1_PARAMS)


# Tuning on the inner split (bench_mu.py) found no single CatBoost better than the
# default: depth 10 with pairwise CTRs, Depthwise/Lossguide growth, fewer borders,
# subsampling and stronger L2 all converge to 223.7-227. Averaging the three
# CatBoost variants reaches 221.5 on inner normal rows, against 224.0 for the
# LightGBM + CatBoost average. Inner said LightGBM adds nothing to that, but over
# the four validate.py splits keeping it as a fourth, equal member is better on
# every split except inner (pooled -0.29 robust), so it stays. The four-member
# average is ~1.3 RMSE better than lgb + cat over the four splits but doubles the
# run time, so it is the final-submission recipe rather than the default:
FINAL_MU = ["lgb", "cat", "cat_note", "cat_ctr1"]
def train_mu_per_airport(X, y):
    """Per-airport pairs, routed by ADEP_mvt. Never use alone: swapping the global
    average for it is mixed across airports (LSZH +1.20). Blend it, weight ~0.4.
    See src/per_airport.py for the measurements."""
    from per_airport import train_mu_per_airport as _t
    return _t(X, y)


def train_mu_neural(X, y):
    """Fifth member, and the only one that is not a tree. Imported lazily so the
    pipeline still runs without torch installed.

    Measured on the inner split, normal rows: solo 232.70, worse than every tree
    member, but its residuals correlate 0.885-0.899 with them against 0.94-0.97
    among the trees themselves, and that diversity is what the average wants.
    Equal five-way weighting takes inner normal rows from 218.63 to 217.97.
    Regularisation is not the binding constraint: 12 vs 20 epochs, dropout 0.1
    vs 0.25 and a 2x wider net all land within 0.3 of each other."""
    from neural import train_mu_neural as _t
    return _t(X, y)


MU_FAMILIES = {"lgb": train_mu, "cat": train_mu_cat, "xgb": train_mu_xgb,
               "cat_note": train_mu_cat_note, "cat_ctr1": train_mu_cat_ctr1,
               "lgb_nocat": train_mu_nocat, "lgb_anchor": train_mu_anchor,
               "lgb_anchor_lobt": train_mu_anchor_lobt, "lgb_anchor_nbr": train_mu_anchor_nbr, "neural": train_mu_neural,
               "pa": train_mu_per_airport}


def _has_lobt(X):
    return ~np.isnan(X["gap_LOBT_flt"].values.astype(float))


def _fit_seeds(X, label, cols, num_boost_round, weight=None):
    cats = [c for c in CAT_COLS if c in cols]
    # Bin once, reuse for every seed. The seeds only move bagging and feature
    # sampling, never the binning, so constructing a Dataset per seed re-binned
    # the same 1.5M rows N_SEEDS times for nothing.
    ds = lgb.Dataset(X[cols], label=label, categorical_feature=cats, weight=weight,
                     free_raw_data=False)
    ds.construct()
    return [lgb.train(dict(CLF_PARAMS, seed=s, bagging_seed=s, feature_fraction_seed=s),
                      ds, num_boost_round=num_boost_round)
            for s in range(N_SEEDS)]


def _usable(names, X):
    """Keep the requested features, in order, de-duplicated.

    A repeated name makes X[cols] hand LightGBM two identical columns, which
    silently skews feature_fraction sampling and makes pandas return a DataFrame
    where LightGBM expects a Series.
    """
    seen, out = set(), []
    for c in names:
        if c in X.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return out


CAT_CLF_PARAMS = dict(
    loss_function="Logloss", iterations=600, learning_rate=0.05, depth=6,
    l2_leaf_reg=5.0, thread_count=-1, verbose=0, allow_writing_files=False,
)


class _CatClf:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(_as_catboost(X)[0])[:, 1]


def _fit_seeds_cat(X, label, cols, iterations=None, weight=None):
    """CatBoost's ordered boosting is built to resist the target leakage that makes
    the LightGBM classifier memorise its few hundred positives."""
    from catboost import CatBoostClassifier, Pool
    Xc, cats = _as_catboost(X[cols])
    pool = Pool(Xc, label, cat_features=cats, weight=weight)
    params = dict(CAT_CLF_PARAMS, **({"iterations": iterations} if iterations else {}))
    return [_CatClf(CatBoostClassifier(**params, random_seed=s).fit(pool))
            for s in range(N_SEEDS)]


CLF_FAMILIES = {"lgb": lambda X, label, cols, weight=None: _fit_seeds(X, label, cols, 300, weight),
                "cat": lambda X, label, cols, weight=None: _fit_seeds_cat(X, label, cols, weight=weight)}


# p only matters where v exceeds TAIL_THR: combine() gates v back to mu below
# it, so a row's p is inert unless it is a tail *candidate*. Log-loss over all
# 1.5M rows spends the classifier on rows it cannot affect and treats a 7000s
# candidate like an 80000s one, although their RMSE stakes differ 100-fold.
# --stakes trains on candidates only, each weighted by its stakes
# ((v - typical taxi) / TAIL_THR)^2, clipped to [1, CLIP]. With CLIP=25 this
# wins on all four splits (pooled robust -2.67, CI -4.34..-1.24 against the
# four-member weather run); candidates alone without weights is -0.92 with the
# interval spanning zero, CLIP=100 is -2.85. Submitted on its own it took the
# live score from 275.97 to 270.21. Off by default so the older recipes stay
# reproducible. With --high-lobt the pooled classifier is not fitted at all
# (fit_pooled=False): matched rows above APPLY_GAP_SEC go to the specialist and
# smaller matched gaps are gated back to mu, so their p cannot matter.
def _stakes_weights(X, y, v, clip):
    apt = X["ADEP_mvt"].astype(str).values
    normal = y <= TAIL_THR
    typical = pd.Series(apt).map(pd.Series(y[normal]).groupby(apt[normal]).median())
    typical = typical.fillna(np.median(y[normal])).values
    return np.clip(((v - typical) / TAIL_THR) ** 2, 1.0, clip)


def train_tail_clf(X, y, pooled="lgb", no_lobt="lgb", stakes=0.0, fit_pooled=True):
    label = (y > TAIL_THR).astype(int)
    pooled_cols = _usable(TAIL_FEATURES, X)
    absent_cols = _usable(NO_LOBT_FEATURES, X)
    absent = ~_has_lobt(X)
    if stakes:
        v = analytic_tail_value(X)
        cand = v > TAIL_THR
        w = _stakes_weights(X, y, v, stakes)
        sel_pooled, sel_absent = cand, cand & absent
    else:
        w = np.ones(len(X))
        sel_pooled, sel_absent = np.ones(len(X), bool), absent
    weight = (lambda sel: w[sel]) if stakes else (lambda sel: None)
    return {
        "pooled": (CLF_FAMILIES[pooled](X[sel_pooled], label[sel_pooled], pooled_cols,
                                        weight(sel_pooled)) if fit_pooled else [],
                   pooled_cols, pooled),
        "no_lobt": (CLF_FAMILIES[no_lobt](X[sel_absent], label[sel_absent], absent_cols,
                                          weight(sel_absent)), absent_cols, no_lobt),
        "q_day": fit_day_late_rate(X, y),
        "q_day_steps": fit_day_late_steps(X, y),
    }


def add_model_args(ap):
    """Model choices shared by validate.py and predict.py, so their defaults agree."""
    # defaults are the fast development configuration; see FINAL_MU for submissions
    ap.add_argument("--mu", nargs="+", default=["lgb", "cat"], choices=list(MU_FAMILIES),
                    help="mu model families, averaged with equal weight "
                         f"(final submission: {' '.join(FINAL_MU)})")
    ap.add_argument("--mu-weights", nargs="+", type=float, metavar="W",
                    help="weights for --mu, in the same order (default: equal). "
                         "Normalised automatically. The per-airport family wants "
                         "~0.4 against 0.6 shared by the four global members, i.e. "
                         "--mu lgb cat cat_note cat_ctr1 pa --mu-weights 1 1 1 1 " 
                         "2.667")
    ap.add_argument("--pa-weight", type=float, default=None,
                    help="weight of the per-airport family when 'pa' is in --mu "
                         "(default: per_airport.PA_WEIGHT). The other families "
                         "share the remainder equally.")
    ap.add_argument("--pa-clip", type=float, default=None,
                    help="cap, in seconds, on how far the per-airport prediction "
                         "may sit from the global average before blending "
                         "(default: per_airport.PA_CLIP; 0 disables)")
    ap.add_argument("--season-tau", type=float, default=0.0,
                    help="up-weight training months near the scored ones; "
                         "weight = exp(-circular month distance / tau), 0 = off")
    ap.add_argument("--monotone", action="store_true",
                    help="constrain congestion and queue counts to increase taxi time")
    ap.add_argument("--proxy", action=argparse.BooleanOptionalAction, default=False,
                    help="use the neighbour taxi-proxy features (features.PROXY_COLS): "
                         "nearby departures' MVT_TIME-to-AOBT_3/LOBT gaps, per airport "
                         "and per runway. Built unconditionally, so both arms of a "
                         "comparison share one feature cache; this flag only chooses "
                         "whether the models see them")
    ap.add_argument("--arvt", action=argparse.BooleanOptionalAction, default=False,
                    help="use the ARVT_1/ARVT_3 trajectory features (features.ARVT_COLS). "
                         "Built unconditionally like --proxy; this flag only chooses "
                         "whether the models see them")
    ap.add_argument("--te", action=argparse.BooleanOptionalAction, default=True,
                    help="add out-of-fold target encodings")
    ap.add_argument("--te-keys", default="base",
                    help="named set of target-encoding keys (encode.KEY_SETS)")
    ap.add_argument("--te-counts", action=argparse.BooleanOptionalAction, default=True,
                    help="add the _n log-count column beside each encoding")
    ap.add_argument("--te-fold-size", type=int, default=1, metavar="MONTHS",
                    help="months per out-of-fold encoding block (1 = shipped)")
    ap.add_argument("--te-flight-backoff", action=argparse.BooleanOptionalAction, default=True,
                    help="te_flight shrinks toward its slot, then route, instead of the global "
                         "mean, so a renamed callsign gets a real fallback (encode.FLIGHT_BACKOFF). "
                         "With the dropout, v24: -0.506 live; 37%% of July 2026 callsigns are "
                         "absent from 2025")
    ap.add_argument("--te-airport-prior", action=argparse.BooleanOptionalAction, default=False,
                    help="shrink every target encoding toward its airport's mean, not the "
                         "global one, so an unseen stand or key gets a local fallback")
    ap.add_argument("--te-flight-dropout", type=float, default=0.2, metavar="FRAC",
                    help="encode this fraction of flight keys as unseen on the training rows "
                         "too, so the models learn the fallback regime (0 with "
                         "--no-te-flight-backoff reproduces v22)")
    ap.add_argument("--lobt-bounds", action=argparse.BooleanOptionalAction, default=True,
                    help="project predictions onto the observed LOBT matching interval")
    ap.add_argument("--high-lobt", action=argparse.BooleanOptionalAction, default=True,
                    help="use a residual specialist when the matched LOBT gap exceeds 6000s")
    ap.add_argument("--clf-pooled", default="cat", choices=list(CLF_FAMILIES))
    ap.add_argument("--clf-nolobt", default="cat", choices=list(CLF_FAMILIES))
    ap.add_argument("--day-late", action=argparse.BooleanOptionalAction, default=False,
                    help="mix a normal-taxi-plus-one-day value into the tail of candidate rows")
    ap.add_argument("--day-late-step", action=argparse.BooleanOptionalAction, default=False,
                    help="day-late component with its rate stepped by gap_SCHED "
                         "(DAY_STEP_EDGES) and value DAY + DAY_LATE_BASE; overrides --day-late")
    ap.add_argument("--tail-cap", action=argparse.BooleanOptionalAction, default=True,
                    help="cap the V_HI_RATIO hedge at max(v, DAY + DAY_LATE_BASE), so it never "
                         "lifts a prediction above both candidate labels (v22: -0.47 live; "
                         "--no-tail-cap reproduces v20)")
    ap.add_argument("--stakes", type=float, default=0.0, metavar="CLIP",
                    help="train the tail classifiers on candidate rows (v > TAIL_THR) only, "
                         "weighted by RMSE stakes clipped to [1, CLIP]; 1 = candidates "
                         "unweighted, 0 = every row unweighted (default)")


def mu_weights(args, families):
    """Normalised weights for the chosen mu families, in order."""
    w = getattr(args, "mu_weights", None)
    if not w:
        return np.full(len(families), 1.0 / len(families))
    if len(w) != len(families):
        raise ValueError(f"--mu-weights needs {len(families)} values, got {len(w)}")
    w = np.asarray(w, dtype=float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("--mu-weights must be non-negative and not all zero")
    return w / w.sum()


def combine_mu(mus, families, args):
    """Average the mu families, handling 'pa' as a clipped offset on the rest.

    The per-airport family is not a peer of the global members: on its own it is
    worse than the average almost everywhere, and it only pays as a bounded
    nudge. So when 'pa' is present the other families are averaged first and the
    per-airport prediction is clipped around that average before blending. See
    src/per_airport.py for the measurements.
    """
    stack = [np.asarray(mus[f"mu_{f}"], dtype=float) for f in families]
    if "pa" not in families:
        return np.average(np.stack(stack), axis=0, weights=mu_weights(args, families))
    from per_airport import blend
    others = [m for f, m in zip(families, stack) if f != "pa"]
    pa = stack[families.index("pa")]
    if not others:
        return np.clip(pa, 0, None)
    glob = np.mean(np.stack(others), axis=0)
    clip = getattr(args, "pa_clip", None)
    return blend(glob, pa, weight=getattr(args, "pa_weight", None),
                 clip=None if clip is None else clip)


def predict_tail_proba(models, X):
    """Pooled model where LOBT exists, dedicated model where it does not.

    Returns calibrated p: each family's CLF_BOOST is already applied."""
    out = np.empty(len(X))
    mask = _has_lobt(X)
    for key, sel in (("pooled", mask), ("no_lobt", ~mask)):
        if not sel.any():
            continue
        boosters, cols, family = models[key]
        # With the high-LOBT specialist, matched gaps >6000 are replaced and
        # smaller gaps are gated back to mu. Their p cannot affect predictions.
        if not boosters:
            out[sel] = 0.0
            continue
        p = np.mean([b.predict(X.loc[sel, cols]) for b in boosters], axis=0)
        out[sel] = np.clip(p * CLF_BOOST[family], 0.0, 1.0)
    return out


# The LightGBM classifier is trained by unweighted log-loss on a 0.04% positive
# rate, which leaves it under-confident: the worst tail errors were rows where v is
# exact but p sits at 0.001-0.15, and scaling p by 1.35 fixed that (tuned on the
# old Jan+Jun+Jul holdout). CatBoost's probabilities need no such correction: its
# sweep bottoms at 1.15 on holdout and 0.9 on inner, so no boost sits between the
# two instead of fitting either.
CLF_BOOST = {"lgb": 1.35, "cat": 1.0}
# v underestimates on the most extreme rows (v>=40000: mean v 59865 vs mean y 65367).
# Much of that gap is the day-late rows; with q_day the correction is not applied
# to day-late candidates, which get the explicit component instead.
V_HI_THR = 40000.0
V_HI_RATIO = 1.08


def combine(mu_pred, p, v, p_boost=1.0, q_day=None, day_value=None, cap=False):
    """Gate v so a confident row never gets a tail value smaller than mu itself.

    p comes from predict_tail_proba and is already calibrated; p_boost is an extra
    multiplier for sweeps. q_day (from day_late_weight or day_late_step_weight)
    splits the tail of candidate rows between v and a day-late value, day_value
    (default DAY + mu).

    On unmatched rows the label is v itself (block recorded as the schedule) or
    a day-late DAY + ~960, so above ~81k the 1.08 hedge lifts the tail above
    both. cap=True limits it to max(v, DAY + DAY_LATE_BASE)."""
    p = np.clip(p * p_boost, 0.0, 1.0)
    q = np.zeros(len(mu_pred)) if q_day is None else np.asarray(q_day, float)
    v_hedged = np.where((v >= V_HI_THR) & (q == 0), v * V_HI_RATIO, v)
    if cap:
        v_hedged = np.where(q == 0, np.minimum(v_hedged, np.maximum(v, DAY + DAY_LATE_BASE)), v_hedged)
    v_gated = np.where(v_hedged > TAIL_THR, v_hedged, mu_pred)
    day = DAY + mu_pred if day_value is None else day_value
    tail = (1.0 - q) * v_gated + q * day
    return np.clip((1.0 - p) * mu_pred + p * tail, 0, None)
