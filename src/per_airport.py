"""Per-airport regressors for the normal regime.

One LightGBM and one CatBoost fitted on each airport's rows alone, averaged, and
applied by routing each row to its own airport's pair. Ten airports with
135k-273k training rows each, so every model still has plenty of data, and none
of them has to share splits or leaf values with nine other layouts.

This is NOT a drop-in replacement for the global average. Measured on three
airports over four splits (`output/perapt_phase1.csv`):

  * swapping the global average for the per-airport one is mixed: EGLL -0.73,
    LIRF -1.10, but LSZH **+1.20**;
  * blending the two helps in all 12 airport-split cells, and at a fixed weight
    of 0.3 every cell is between -0.95 and -2.65 on that airport's normal rows.

So use it as a blend partner, never alone. `blend_weight()` documents the weight
measured over LIRF, EGLL and LSZH; the remaining seven airports were out of
sample for that choice.

Routing uses `ADEP_mvt`, which is a model feature, so this needs nothing from the
metadata frame and works as an ordinary entry in MU_FAMILIES.
"""
import numpy as np

import lightgbm as lgb

from features import CAT_COLS
from model import CAT_MU_PARAMS, MU_PARAMS, TAIL_THR

# Mid-plateau values, not grid edges: clip 100-200s with weight 0.4-0.6 all land
# within 0.06 of each other, so neither constant is delicate. Live on v39's families
# (a blend-only change of at most weight x clip per row): w 0.4 232.8913, 0.5
# 232.9112, 0.6 233.0082; a quadratic on MSE puts the optimum near 0.42, so 0.4
# since v41. --pa-weight 0.5 reproduces v39 and earlier.
PA_WEIGHT = 0.4
PA_CLIP = 150.0  # seconds; the per-airport model may only nudge the global one
# 2000 until v35; 5000 (B22) took v38 live, -0.08. --pa-cat-iterations 2000 reproduces v35
CAT_ITERATIONS = 5000
LGB_ROUNDS = 900
MIN_ROWS = 20000  # below this, fall back to the global model for that airport


def blend_weight():
    return PA_WEIGHT


def blend(global_mu, pa_mu, weight=None, clip=None):
    """Blend the per-airport prediction into the global one as a BOUNDED OFFSET.

    Clipping matters more than the weight. Unclipped, one bad airport-split pair
    (LTFM on inner, where the per-airport model is 347.3 against the global's
    311.9) drags the whole split +0.51 the wrong way. Clipped to +/-150s the
    per-airport model can move a prediction by at most weight*clip, and every
    airport-split cell measured then improves.

    Measured deltas on normal rows against the four-member global average,
    mean over four splits: unclipped w=0.4 -0.68; clip 300s -0.99; clip 150s
    w=0.5 -1.12; clip 100s w=0.6 -1.13. A broad plateau over clip 100-200 and
    weight 0.4-0.6, so the exact values are not delicate.
    """
    w = PA_WEIGHT if weight is None else weight
    c = PA_CLIP if clip is None else clip
    g = np.asarray(global_mu, dtype=float)
    p = np.asarray(pa_mu, dtype=float)
    p = np.where(np.isfinite(p), p, g)          # uncovered airports keep the global value
    if c and c > 0:
        p = np.clip(p, g - c, g + c)
    return np.clip((1.0 - w) * g + w * p, 0, None)


def _cat_cols(X):
    return [c for c in CAT_COLS if c in X.columns]


class _PerAirportMu:
    """Holds one (lgb, cat) pair per airport and routes rows by ADEP_mvt."""

    def __init__(self, models, fallback_airports):
        self.models = models
        self.fallback_airports = fallback_airports

    def predict(self, X):
        apt = X["ADEP_mvt"].astype(str).to_numpy()
        out = np.full(len(X), np.nan)
        for a, (booster, cat_model, cols) in self.models.items():
            sel = apt == a
            if not sel.any():
                continue
            Xa = X.loc[sel, cols]
            p_lgb = booster.predict(Xa)
            Xc = Xa.copy()
            for c in _cat_cols(Xc):
                Xc[c] = Xc[c].astype(str)
            p_cat = cat_model.predict(Xc)
            out[sel] = (p_lgb + p_cat) / 2.0
        return np.clip(out, 0, None)

    def covered(self, X):
        """Rows this model can predict; the rest come back as NaN."""
        return ~np.isnan(self.predict(X)) if len(X) else np.zeros(0, bool)


def train_mu_per_airport(X, y, iterations=None, rounds=LGB_ROUNDS,
                         verbose=True):
    from catboost import CatBoostRegressor, Pool

    iterations = iterations or CAT_ITERATIONS
    normal = y <= TAIL_THR
    apt = X["ADEP_mvt"].astype(str).to_numpy()
    cols = list(X.columns)
    cats = _cat_cols(X)
    models, skipped = {}, []
    for a in sorted(set(apt)):
        sel = (apt == a) & normal
        if sel.sum() < MIN_ROWS:
            skipped.append(a)
            continue
        Xa, ya = X.loc[sel, cols], y[sel]
        ds = lgb.Dataset(Xa, label=ya, categorical_feature=cats)
        booster = lgb.train(MU_PARAMS, ds, num_boost_round=rounds)
        Xc = Xa.copy()
        for c in cats:
            Xc[c] = Xc[c].astype(str)
        cat_model = CatBoostRegressor(**dict(CAT_MU_PARAMS, iterations=iterations))
        cat_model.fit(Pool(Xc, ya, cat_features=cats))
        models[a] = (booster, cat_model, cols)
        if verbose:
            print(f"    per-airport {a}: {int(sel.sum()):,} rows", flush=True)
    if skipped and verbose:
        print(f"    skipped (under {MIN_ROWS} rows): {skipped}", flush=True)
    return _PerAirportMu(models, skipped)
