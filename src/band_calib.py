"""Out-of-fold calibration of mu by predicted band, applied post-hoc.

    python src/band_calib.py --baseline pa_v9 --tag pa_v9_cal

mu is well calibrated below ~2000s and under-predicts above it. This fits that
bias per band on the OTHER splits and applies it to the held-out one, so no
split is ever calibrated on itself. The four splits score disjoint month pairs,
which is what makes them usable as each other's calibration data.

Nothing is retrained. The mixture is rebuilt from the saved p, v and q_day and
then re-bounded, which reproduces a from-scratch run bit-for-bit -- verified by
--check, which rebuilds with a zero correction and asserts the predictions are
identical to the baseline's.
"""
import argparse

import numpy as np
import pandas as pd

from constraints import project_lobt
from high_lobt import APPLY_GAP_SEC
from model import TAIL_THR, combine
from validate import OUT_DIR, SPLITS, compare, report, verdict

# Wide at the bottom where mu is already calibrated and rows are dense, finer
# where the bias grows. The top edge is TAIL_THR: above it the mixture, not mu,
# is deciding the row.
BAND_EDGES = [0, 1000, 1500, 2000, 2500, 3000, 4000, TAIL_THR, np.inf]


def fit_bias(runs, shrink, per_airport=False, apt_shrink=400.0):
    """Mean (y - mu) per band over the given runs' normal rows.

    Shrunk toward zero by n/(n+shrink): the top bands hold few rows and their
    per-split bias has a standard deviation comparable to its own size.

    With per_airport, a second level is fitted per (airport, band) and shrunk
    toward that band's global value rather than toward zero -- the upper-band
    bias is strongly airport-specific (LTFM +195/+459/+599 against a global
    +60/+155/+273) but the cells are thin, so the global value is the right
    thing to fall back to.
    """
    d = pd.concat(runs, ignore_index=True)
    d = d[d["y"].values <= TAIL_THR]
    d = d.assign(_b=pd.cut(d["mu"].values, BAND_EDGES, right=False),
                 _r=d["y"].values - d["mu"].values)
    g = d.groupby("_b", observed=False)["_r"]
    n, mean = g.size(), g.mean().fillna(0.0)
    glob = (mean * n / (n + shrink)).to_dict()
    if not per_airport:
        return glob, {}
    ga = d.groupby(["ADEP_mvt", "_b"], observed=True)["_r"]
    na, ma = ga.size(), ga.mean()
    apt = {}
    for k in na.index:
        base = glob.get(k[1], 0.0)
        apt[k] = (ma[k] * na[k] + apt_shrink * base) / (na[k] + apt_shrink)
    return glob, apt


def apply_bias(r, bias, apt_bias=None):
    band = pd.cut(r["mu"].values, BAND_EDGES, right=False)
    delta = np.array([bias.get(b, 0.0) for b in band], dtype=float)
    if apt_bias:
        keys = list(zip(r["ADEP_mvt"].values, band))
        delta = np.array([apt_bias.get(k, bias.get(k[1], 0.0)) for k in keys], dtype=float)
    mu = np.clip(r["mu"].values + delta, 0, None)
    q = r["q_day"].values if bool(r.get("day_late", pd.Series([False])).iloc[0]) else None
    pred = combine(mu, r["p"].values, r["v"].values, q_day=q)
    if bool(r["lobt_bounds"].iloc[0]):
        pred = project_lobt(pred, r["gap_LOBT_flt"].values)
    if bool(r["high_lobt"].iloc[0]):
        hi = r["gap_LOBT_flt"].values > APPLY_GAP_SEC
        pred[hi] = r["high_lobt_pred"].values[hi]
    out = r.copy()
    out["mu"], out["pred"] = mu, pred
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--shrink", type=float, default=2000.0,
                    help="rows of prior weight pulling each band's bias to zero")
    ap.add_argument("--per-airport", action="store_true",
                    help="fit a second level per (airport, band), shrunk toward the global band")
    ap.add_argument("--apt-shrink", type=float, default=400.0,
                    help="rows of prior weight pulling each airport-band toward the global band")
    ap.add_argument("--check", action="store_true",
                    help="rebuild with a zero correction and assert it is bit-for-bit")
    args = ap.parse_args()

    runs = {s: pd.read_parquet(OUT_DIR / args.baseline / f"{s}.parquet") for s in SPLITS}

    if args.check:
        for s, r in runs.items():
            got = apply_bias(r, {}, {})["pred"].values
            assert np.array_equal(got, r["pred"].values), f"{s}: rebuild is not exact"
            print(f"{s:8s} rebuild exact over {len(r):,} rows")
        return

    print(f"bands {BAND_EDGES}, shrink {args.shrink:g}\n")
    results, out = {}, OUT_DIR / (args.tag or f"{args.baseline}_cal")
    out.mkdir(parents=True, exist_ok=True)
    for s, r in runs.items():
        bias, apt = fit_bias([runs[o] for o in SPLITS if o != s], args.shrink,
                             args.per_airport, args.apt_shrink)
        print(f"[{s}] bias fitted on {', '.join(o for o in SPLITS if o != s)}: "
              + "  ".join(f"{b.left:.0f}-{b.right:.0f}:{v:+.0f}" for b, v in bias.items()))
        cur = apply_bias(r, bias, apt)
        cur.to_parquet(out / f"{s}.parquet", index=False)
        report(s, cur)
        results[s] = compare(s, cur, r, args.baseline)
    print(f"\nverdict: {verdict(results)}", flush=True)


if __name__ == "__main__":
    main()
