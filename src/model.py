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

from features import CAT_COLS

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


def train_mu(X, y, num_boost_round=900):
    normal = y <= TAIL_THR
    ds = lgb.Dataset(X[normal], label=y[normal], categorical_feature=CAT_COLS)
    return lgb.train(MU_PARAMS, ds, num_boost_round=num_boost_round)


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
MU_FAMILIES = {"lgb": train_mu, "cat": train_mu_cat, "xgb": train_mu_xgb,
               "cat_note": train_mu_cat_note, "cat_ctr1": train_mu_cat_ctr1}


def _has_lobt(X):
    return ~np.isnan(X["gap_LOBT_flt"].values.astype(float))


def _fit_seeds(X, label, cols, num_boost_round):
    cats = [c for c in CAT_COLS if c in cols]
    # Bin once, reuse for every seed. The seeds only move bagging and feature
    # sampling, never the binning, so constructing a Dataset per seed re-binned
    # the same 1.5M rows N_SEEDS times for nothing.
    ds = lgb.Dataset(X[cols], label=label, categorical_feature=cats,
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


def _fit_seeds_cat(X, label, cols, iterations=None):
    """CatBoost's ordered boosting is built to resist the target leakage that makes
    the LightGBM classifier memorise its few hundred positives."""
    from catboost import CatBoostClassifier, Pool
    Xc, cats = _as_catboost(X[cols])
    pool = Pool(Xc, label, cat_features=cats)
    params = dict(CAT_CLF_PARAMS, **({"iterations": iterations} if iterations else {}))
    return [_CatClf(CatBoostClassifier(**params, random_seed=s).fit(pool))
            for s in range(N_SEEDS)]


CLF_FAMILIES = {"lgb": lambda X, label, cols: _fit_seeds(X, label, cols, 300),
                "cat": _fit_seeds_cat}


def train_tail_clf(X, y, pooled="lgb", no_lobt="lgb", fit_pooled=True):
    label = (y > TAIL_THR).astype(int)
    pooled_cols = _usable(TAIL_FEATURES, X)
    absent_cols = _usable(NO_LOBT_FEATURES, X)
    absent = ~_has_lobt(X)
    return {
        "pooled": (CLF_FAMILIES[pooled](X, label, pooled_cols) if fit_pooled else [],
                   pooled_cols, pooled),
        "no_lobt": (CLF_FAMILIES[no_lobt](X[absent], label[absent], absent_cols), absent_cols,
                    no_lobt),
        "q_day": fit_day_late_rate(X, y),
    }


def add_model_args(ap):
    """Model choices shared by validate.py and predict.py, so their defaults agree."""
    # defaults are the fast development configuration; see FINAL_MU for submissions
    ap.add_argument("--mu", nargs="+", default=["lgb", "cat"], choices=list(MU_FAMILIES),
                    help="mu model families, averaged with equal weight "
                         f"(final submission: {' '.join(FINAL_MU)})")
    ap.add_argument("--te", action=argparse.BooleanOptionalAction, default=True,
                    help="add out-of-fold target encodings")
    ap.add_argument("--lobt-bounds", action=argparse.BooleanOptionalAction, default=True,
                    help="project predictions onto the observed LOBT matching interval")
    ap.add_argument("--high-lobt", action=argparse.BooleanOptionalAction, default=True,
                    help="use a residual specialist when the matched LOBT gap exceeds 6000s")
    ap.add_argument("--clf-pooled", default="cat", choices=list(CLF_FAMILIES))
    ap.add_argument("--clf-nolobt", default="cat", choices=list(CLF_FAMILIES))
    ap.add_argument("--day-late", action=argparse.BooleanOptionalAction, default=False,
                    help="mix a normal-taxi-plus-one-day value into the tail of candidate rows")


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


def combine(mu_pred, p, v, p_boost=1.0, q_day=None):
    """Gate v so a confident row never gets a tail value smaller than mu itself.

    p comes from predict_tail_proba and is already calibrated; p_boost is an extra
    multiplier for sweeps. q_day (from day_late_weight) splits the tail of
    candidate rows between v and a day-late value, DAY + mu."""
    p = np.clip(p * p_boost, 0.0, 1.0)
    q = np.zeros(len(mu_pred)) if q_day is None else np.asarray(q_day, float)
    v = np.where((v >= V_HI_THR) & (q == 0), v * V_HI_RATIO, v)
    v_gated = np.where(v > TAIL_THR, v, mu_pred)
    tail = (1.0 - q) * v_gated + q * (DAY + mu_pred)
    return np.clip((1.0 - p) * mu_pred + p * tail, 0, None)
