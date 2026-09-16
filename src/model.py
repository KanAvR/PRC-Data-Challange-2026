"""Two-component model for taxi-out time.

A GBM predicts an average of the training labels in each leaf, so it can never
output the 6000-131167s values that dominate RMSE. Those rows are supplied
analytically instead: on them the recorded block time is the scheduled (or
latest-off-block) time, so the target equals MVT_TIME minus that timestamp.

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


def train_mu_cat(X, y, iterations=None, eval_set=None):
    from catboost import CatBoostRegressor, Pool
    normal = y <= TAIL_THR
    Xc, cats = _as_catboost(X[normal])
    params = dict(CAT_MU_PARAMS)
    if iterations:
        params["iterations"] = iterations
    model = CatBoostRegressor(**params)
    evals = None
    if eval_set is not None:
        Xe, ye = eval_set
        evals = Pool(_as_catboost(Xe)[0], ye, cat_features=cats)
    model.fit(Pool(Xc, y[normal], cat_features=cats), eval_set=evals,
              metric_period=100, verbose=100 if evals is not None else 0)
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


MU_FAMILIES = {"lgb": train_mu, "cat": train_mu_cat, "xgb": train_mu_xgb}


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


def train_tail_clf(X, y, pooled="lgb", no_lobt="lgb"):
    label = (y > TAIL_THR).astype(int)
    pooled_cols = _usable(TAIL_FEATURES, X)
    absent_cols = _usable(NO_LOBT_FEATURES, X)
    absent = ~_has_lobt(X)
    return {
        "pooled": (CLF_FAMILIES[pooled](X, label, pooled_cols), pooled_cols, pooled),
        "no_lobt": (CLF_FAMILIES[no_lobt](X[absent], label[absent], absent_cols), absent_cols,
                    no_lobt),
    }


def add_model_args(ap):
    """Model choices shared by validate.py and predict.py, so their defaults agree."""
    # defaults are the configuration that last passed validate.py on both splits
    ap.add_argument("--mu", nargs="+", default=["lgb", "cat"], choices=list(MU_FAMILIES),
                    help="mu model families, averaged with equal weight")
    ap.add_argument("--te", action=argparse.BooleanOptionalAction, default=True,
                    help="add out-of-fold target encodings")
    ap.add_argument("--clf-pooled", default="cat", choices=list(CLF_FAMILIES))
    ap.add_argument("--clf-nolobt", default="cat", choices=list(CLF_FAMILIES))


def predict_tail_proba(models, X):
    """Pooled model where LOBT exists, dedicated model where it does not.

    Returns calibrated p: each family's CLF_BOOST is already applied."""
    out = np.empty(len(X))
    mask = _has_lobt(X)
    for key, sel in (("pooled", mask), ("no_lobt", ~mask)):
        if not sel.any():
            continue
        boosters, cols, family = models[key]
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
# v underestimates on the most extreme rows (v>=40000: mean v 59865 vs mean y 65367)
V_HI_THR = 40000.0
V_HI_RATIO = 1.08


def combine(mu_pred, p, v, p_boost=1.0):
    """Gate v so a confident row never gets a tail value smaller than mu itself.

    p comes from predict_tail_proba and is already calibrated; p_boost is an extra
    multiplier for sweeps."""
    p = np.clip(p * p_boost, 0.0, 1.0)
    v = np.where(v >= V_HI_THR, v * V_HI_RATIO, v)
    v_gated = np.where(v > TAIL_THR, v, mu_pred)
    return np.clip((1.0 - p) * mu_pred + p * v_gated, 0, None)
