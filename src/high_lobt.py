"""Residual specialist for departures with a large, matched LOBT gap.

The ordinary regressor excludes labels above 6000 seconds. For matched rows,
however, y - gap_LOBT stays bounded even when y is 24 hours. Learn that residual
on gaps above 3000 seconds, including long taxi labels, and apply only above
6000 seconds. No calendar-episode identifiers or leaderboard labels are used.
"""
import lightgbm as lgb
import numpy as np

from constraints import check_lobt_support, project_lobt
from features import CAT_COLS

TRAIN_GAP_SEC = 3000.0
APPLY_GAP_SEC = 6000.0


class HighLobtModel:
    def __init__(self, booster, columns):
        self.booster, self.columns = booster, columns

    def predict(self, X):
        gap = X["gap_LOBT_flt"].to_numpy(dtype=float)
        return project_lobt(gap + self.booster.predict(X[self.columns]), gap)

    def refine(self, pred, X):
        out = np.asarray(pred, dtype=float).copy()
        use = X["gap_LOBT_flt"].to_numpy(dtype=float) > APPLY_GAP_SEC
        if use.any():
            out[use] = self.predict(X.loc[use])
        return out


def train_high_lobt(X, y):
    from model import MU_PARAMS

    y = np.asarray(y, dtype=float)
    gap = X["gap_LOBT_flt"].to_numpy(dtype=float)
    check_lobt_support(y, gap)
    use = np.isfinite(y) & (y >= 0) & (gap > TRAIN_GAP_SEC)
    if not use.any():
        raise ValueError("no matched departures above the specialist's training gap")
    # The four-split study used the existing operational features. Keep weather
    # and normal-regime target encodings out of this small extrapolation model.
    columns = [c for c in X if not c.startswith(("met_", "te_"))]
    ds = lgb.Dataset(X.loc[use, columns], label=y[use] - gap[use],
                     categorical_feature=[c for c in CAT_COLS if c in columns])
    params = dict(MU_PARAMS, num_threads=4, num_leaves=31, min_data_in_leaf=50,
                  learning_rate=0.03, lambda_l2=10.0)
    return HighLobtModel(lgb.train(params, ds, num_boost_round=500), columns)
