"""Rebuild the per-airport blend from saved runs and sweep weight/clip, no training.

No training: theirs_only.mu is the 4-member global average, mu_pa.mu_pa the pure
per-airport one, and pa_v9 is the w=0.5 clip=150 blend we must reproduce exactly.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from model import combine
from constraints import project_lobt
from validate import OUT_DIR, SPLITS, rmse, EXPLAIN_TOL, GAP_COLS

def load(split):
    g = pd.read_parquet(OUT_DIR / "theirs_only" / f"{split}.parquet")
    p = pd.read_parquet(OUT_DIR / "mu_pa" / f"{split}.parquet")
    assert (g.MVT_ID_mvt.values == p.MVT_ID_mvt.values).all(), "row order differs"
    g = g.copy(); g["pa"] = p["mu_pa"].values
    return g

def build(r, w, clip):
    """w, clip may be scalars or per-row arrays. Returns pred."""
    glob = r["mu"].values.astype(float)
    pa = np.where(np.isfinite(r["pa"].values), r["pa"].values, glob)
    if clip is not None:
        pa = np.clip(pa, glob - clip, glob + clip)
    mu = np.clip((1.0 - w) * glob + w * pa, 0, None)
    pred = combine(mu, r["p"].values, r["v"].values, q_day=None)
    pred = project_lobt(pred, r["gap_LOBT_flt"].values)
    high = r["high_lobt_pred"].notna().values
    pred = np.where(high, r["high_lobt_pred"].values, pred)
    return pred

def robust_mask(r):
    """validate.py's robust mask: drop tail rows matching no timestamp."""
    y = r["y"].values
    tail = y > 6000
    match = np.zeros(len(r), bool)
    for c in GAP_COLS:
        if c in r:
            match |= np.abs(r[c].values - y) <= EXPLAIN_TOL
    return ~(tail & ~match)

if __name__ == "__main__":
    print("=== exactness check: rebuild pa_v9 at w=0.5 clip=150 ===")
    for s in SPLITS:
        r = load(s)
        ref = pd.read_parquet(OUT_DIR / "pa_v9" / f"{s}.parquet")
        got = build(r, 0.5, 150.0)
        dmu = np.abs(np.clip(0.5*r["mu"].values + 0.5*np.clip(r["pa"].values, r["mu"].values-150, r["mu"].values+150), 0, None) - ref["mu"].values).max()
        d = np.abs(got - ref["pred"].values)
        print(f"  {s:8s} max |mu diff| {dmu:.3e}   max |pred diff| {d.max():.3e}   rows differing >1e-6: {(d>1e-6).sum()}")
