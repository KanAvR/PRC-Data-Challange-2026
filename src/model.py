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


def train_tail_clf(X, y, num_boost_round=300):
    label = (y > TAIL_THR).astype(int)
    pooled_cols = _usable(TAIL_FEATURES, X)
    absent_cols = _usable(NO_LOBT_FEATURES, X)
    no_lobt = ~_has_lobt(X)
    return {
        "pooled": (_fit_seeds(X, label, pooled_cols, num_boost_round), pooled_cols),
        "no_lobt": (_fit_seeds(X[no_lobt], label[no_lobt], absent_cols, num_boost_round),
                    absent_cols),
    }


def predict_tail_proba(models, X):
    """Pooled model where LOBT exists, dedicated model where it does not."""
    out = np.empty(len(X))
    mask = _has_lobt(X)
    for key, sel in (("pooled", mask), ("no_lobt", ~mask)):
        if not sel.any():
            continue
        boosters, cols = models[key]
        out[sel] = np.mean([b.predict(X.loc[sel, cols]) for b in boosters], axis=0)
    return out


# p is trained by unweighted log-loss on a 0.04% positive rate, which leaves it
# systematically under-confident: the worst remaining tail errors are rows where v
# is exact but p sits at 0.001-0.15. Boosting p is worth ~7 RMSE. Both eval splits
# pick 1.35 independently and the optimum is a flat plateau over [1.30, 1.50], so
# this is a calibration fix rather than a fitted constant.
P_BOOST = 1.35
# v underestimates on the most extreme rows (v>=40000: mean v 59865 vs mean y 65367)
V_HI_THR = 40000.0
V_HI_RATIO = 1.08


def combine(mu_pred, p, v):
    """Gate v so a confident row never gets a tail value smaller than mu itself."""
    p = np.clip(p * P_BOOST, 0.0, 1.0)
    v = np.where(v >= V_HI_THR, v * V_HI_RATIO, v)
    v_gated = np.where(v > TAIL_THR, v, mu_pred)
    return np.clip((1.0 - p) * mu_pred + p * v_gated, 0, None)
