"""Blend mu predictions from saved validate.py runs, without retraining.

Each --mu entry is TAG or TAG:COLUMN (default column "mu"). The tail parts (p, v)
come from --clf. Weights are equal unless --fit-weights is given, in which case
they are fitted on the inner split's normal rows and applied unchanged to the
holdout, so the holdout never sees the fit.

    python src/blend.py --clf airline --mu airline cat:mu_cat --compare airline
    python src/blend.py --clf airline --mu airline cat:mu_cat --fit-weights --tag blend1
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from model import CLF_BOOST, TAIL_THR, combine
from constraints import project_lobt
from validate import OUT_DIR, SPLITS, compare, report, rmse, verdict


def load_mu(spec, split):
    tag, _, col = spec.partition(":")
    return pd.read_parquet(OUT_DIR / tag / f"{split}.parquet")[col or "mu"].values


def fit_weights(mus, y, step=0.05):
    """Simplex grid search for the weights minimising RMSE."""
    k = len(mus)
    grid = np.arange(0, 1 + 1e-9, step)
    best, best_w = np.inf, None
    for w in itertools.product(grid, repeat=k - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = np.array([*w, 1 - sum(w)])
        e = rmse(y, w @ mus)
        if e < best:
            best, best_w = e, w
    return best_w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clf", required=True, help="tag whose p and v are used")
    ap.add_argument("--mu", nargs="+", required=True, help="TAG or TAG:COLUMN")
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--fit-weights", action="store_true")
    ap.add_argument("--p-boost", type=float,
                    help="extra multiplier on p (default: none, or 1.35 for old raw-p runs)")
    ap.add_argument("--day-late", action="store_true", help="use the saved q_day component")
    ap.add_argument("--lobt-bounds", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--high-lobt", action="store_true",
                    help="apply specialist predictions saved in the --clf run")
    ap.add_argument("--compare", metavar="TAG")
    ap.add_argument("--tag", help="save the blended run as a new tag")
    args = ap.parse_args()

    weights = np.full(len(args.mu), 1 / len(args.mu))
    if args.weights:
        weights = np.array(args.weights) / sum(args.weights)
    if args.fit_weights:
        base = pd.read_parquet(OUT_DIR / args.clf / "inner.parquet")
        normal = base["y"].values <= TAIL_THR
        mus = np.stack([load_mu(s, "inner")[normal] for s in args.mu])
        weights = fit_weights(mus, base["y"].values[normal])
    print("weights: " + "  ".join(f"{s}={w:.2f}" for s, w in zip(args.mu, weights)))

    tags = {args.clf, *(s.partition(":")[0] for s in args.mu)}
    splits = [s for s in SPLITS if all((OUT_DIR / t / f"{s}.parquet").exists() for t in tags)]
    deltas = {}
    for split in splits:
        r = pd.read_parquet(OUT_DIR / args.clf / f"{split}.parquet")
        mus = np.stack([load_mu(s, split) for s in args.mu])
        r["mu"] = weights @ mus
        # runs saved before CLF_BOOST existed hold raw LightGBM p, boosted by 1.35 then
        boost = args.p_boost or (1.0 if "p_calibrated" in r.columns else CLF_BOOST["lgb"])
        r["p"] = np.clip(r["p"].values * boost, 0.0, 1.0)
        r["p_calibrated"] = True
        q = r["q_day"].values if args.day_late else None
        r["pred"] = combine(r["mu"].values, r["p"].values, r["v"].values, q_day=q)
        r["pred_unconstrained"] = r["pred"]
        r["lobt_bounds"] = args.lobt_bounds
        if args.lobt_bounds:
            r["pred"] = project_lobt(r["pred"].values, r["gap_LOBT_flt"].values)
        r["high_lobt"] = args.high_lobt
        if args.high_lobt:
            if "high_lobt_pred" not in r or r["high_lobt_pred"].notna().sum() == 0:
                raise ValueError("--clf run has no saved specialist predictions")
            high = r["high_lobt_pred"].notna()
            r.loc[high, "pred"] = r.loc[high, "high_lobt_pred"]
        report(split, r)
        if args.tag:
            out = OUT_DIR / args.tag
            out.mkdir(parents=True, exist_ok=True)
            r.to_parquet(out / f"{split}.parquet", index=False)
        ref = OUT_DIR / str(args.compare) / f"{split}.parquet"
        if args.compare and ref.exists():
            deltas[split] = compare(split, r, pd.read_parquet(ref), args.compare)
    v = verdict(deltas) if args.compare else None
    if v:
        print(f"\nverdict: {v}")


if __name__ == "__main__":
    main()
