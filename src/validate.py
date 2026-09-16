"""Two-split validation: the rule every model change has to pass.

    holdout  train on the other 10 months, score Jan+Jul 2025 (mirrors ranking)
    inner    train on the 8 months left after also dropping Feb+Aug, score Feb+Aug

The inner split never sees the holdout months, so a constant tuned on one split
can be checked out of sample on the other. A change is kept only if it wins on
both.

RMSE here is dominated by a handful of tail rows (10 of 344k carry 37.5% of the
squared error), so a single number is not evidence. Every run saves its
predictions under output/validate/<tag>/, and --compare scores the current run
against a saved one with a bootstrap over days (paired, same rows).

    python src/validate.py --tag baseline
    python src/validate.py --tag airline --compare baseline
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from data import align_categoricals, load_training
from encode import add_target_encodings
from model import (MU_FAMILIES, TAIL_THR, add_model_args,
                   analytic_tail_value, combine, predict_tail_proba,
                   train_tail_clf)

SPLITS = {
    "holdout": dict(valid=["2025-01", "2025-07"], drop=[]),
    "inner": dict(valid=["2025-02", "2025-08"], drop=["2025-01", "2025-07"]),
}
OUT_DIR = pathlib.Path("output/validate")
N_BOOT = 2000
P_SWEEP = [0.8, 0.9, 1.0, 1.15, 1.35, 1.7]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def run_split(name, X, y, meta, args):
    spec = SPLITS[name]
    va = meta["month"].isin(spec["valid"]).values
    tr = ~va & ~meta["month"].isin(spec["drop"]).values
    X_tr, X_va = X.loc[tr].copy(), X.loc[va].copy()
    y_tr, y_va = y[tr], y[va]
    align_categoricals(X_tr, X_va)
    if args.te:
        add_target_encodings(X_tr, y_tr, meta.loc[tr, "month"].values, meta.loc[tr, "flight"].values,
                             X_va, meta.loc[va, "flight"].values)
    print(f"\n[{name}] train={len(X_tr)} valid={len(X_va)} "
          f"tail={int((y_tr > TAIL_THR).sum())}/{int((y_va > TAIL_THR).sum())}", flush=True)

    # several families are averaged with equal weight; each is saved on its own
    # so blend.py can re-weight them without retraining
    mus = {}
    if args.reuse_mu:
        saved = pd.read_parquet(OUT_DIR / args.reuse_mu / f"{name}.parquet")
        assert np.array_equal(saved["y"].values, y_va), "reuse-mu run has different rows"
        mus = {c: saved[c].values for c in saved.columns if c.startswith("mu_")}
        mu = saved["mu"].values
    else:
        for fam in args.mu:
            mus[f"mu_{fam}"] = np.clip(MU_FAMILIES[fam](X_tr, y_tr).predict(X_va), 0, None)
            print(f"  {fam}: done", flush=True)
        mu = np.mean(list(mus.values()), axis=0)
    clf = train_tail_clf(X_tr, y_tr, pooled=args.clf_pooled, no_lobt=args.clf_nolobt)
    p = predict_tail_proba(clf, X_va)
    v = analytic_tail_value(X_va)
    pred = combine(mu, p, v)

    return pd.DataFrame({
        "date": meta.loc[va, "date"].values, "ADEP_mvt": meta.loc[va, "ADEP_mvt"].values,
        "has_lobt": ~np.isnan(X_va["gap_LOBT_flt"].values.astype(float)),
        "y": y_va, "mu": mu, "p": p, "v": v, "pred": pred, "p_calibrated": True, **mus,
    })


def report(name, r):
    y, tail = r["y"].values, r["y"].values > TAIL_THR
    print(f"[{name}] scored over ALL rows, as the leaderboard does")
    print(f"  mixture          {rmse(y, r['pred']):8.2f}")
    print(f"  mu alone         {rmse(y, r['mu']):8.2f}")
    print(f"  normal rows      {rmse(y[~tail], r['pred'][~tail]):8.2f}   (n={int((~tail).sum())})")
    print(f"  tail rows        {rmse(y[tail], r['pred'][tail]):8.2f}   (n={int(tail.sum())})")
    for label, sel in (("all", np.ones(len(r), bool)), ("LOBT present", r["has_lobt"].values),
                       ("LOBT absent", ~r["has_lobt"].values)):
        if 0 < tail[sel].sum() < sel.sum():
            print(f"  tail AUC {label:13s} {roc_auc_score(tail[sel], r['p'][sel]):.4f}")
    # scaling the saved p needs no retraining
    sweep = "  ".join(
        f"{b:.2f}:{rmse(y, combine(r['mu'].values, r['p'].values, r['v'].values, b)):.2f}"
        for b in P_SWEEP)
    print(f"  extra p boost    {sweep}   (p already carries CLF_BOOST)")
    per_apt = r.assign(se=(r["y"] - r["pred"]) ** 2).groupby("ADEP_mvt")["se"].mean() ** 0.5
    print("  per airport      " + "  ".join(f"{a}:{v:.0f}" for a, v in per_apt.items()))


def compare(name, cur, base, base_tag):
    if len(cur) != len(base) or not np.array_equal(cur["y"].values, base["y"].values):
        print(f"[{name}] cannot compare with '{base_tag}': different validation rows")
        return
    if np.array_equal(cur["pred"].values, base["pred"].values):
        print(f"[{name}] predictions are identical to '{base_tag}': the change had no effect")
        return 0.0, 0.0, 0.0
    d = cur.assign(se_cur=(cur["y"] - cur["pred"]) ** 2,
                   se_base=(base["y"].values - base["pred"].values) ** 2)
    days = d.groupby("date")[["se_cur", "se_base"]].sum()
    n_day = d.groupby("date").size().values
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    n = n_day[idx].sum(axis=1)
    delta = (np.sqrt(days["se_cur"].values[idx].sum(axis=1) / n)
             - np.sqrt(days["se_base"].values[idx].sum(axis=1) / n))
    point = rmse(d["y"], d["pred"]) - rmse(base["y"], base["pred"])
    lo, hi = np.percentile(delta, [2.5, 97.5])
    tail = d["y"].values > TAIL_THR
    normal_delta = (rmse(d["y"][~tail], d["pred"][~tail])
                    - rmse(base["y"][~tail], base["pred"][~tail]))
    print(f"[{name}] vs '{base_tag}': {point:+.2f} RMSE  (95% day-bootstrap CI "
          f"{lo:+.2f} .. {hi:+.2f}, better in {100 * (delta < 0).mean():.0f}% of resamples)"
          f"   normal rows {normal_delta:+.2f}")
    return point, lo, hi


def verdict(results):
    """results: split -> (point, lo, hi). A point estimate below zero on ~220 tail
    rows is not a win by itself; the day-bootstrap interval has to clear zero too."""
    if len(results) < len(SPLITS) or any(r is None for r in results.values()):
        return None
    if all(hi < 0 for _, _, hi in results.values()):
        return "wins on both splits (both CIs below zero)"
    if all(pt < 0 for pt, _, _ in results.values()):
        loose = [s for s, (_, _, hi) in results.items() if hi >= 0]
        return f"better on both splits, but the CI crosses zero on {', '.join(loose)}: not established"
    return "does NOT win on both splits"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="latest", help="save predictions under output/validate/<tag>/")
    ap.add_argument("--compare", metavar="TAG", help="paired comparison against a saved run")
    add_model_args(ap)
    ap.add_argument("--reuse-mu", metavar="TAG", help="take mu from a saved run instead of training it")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = ap.parse_args()

    print("Loading training data and building features...", flush=True)
    X, y, meta = load_training()

    out = OUT_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.splits:
        r = run_split(name, X, y, meta, args)
        r.to_parquet(out / f"{name}.parquet", index=False)
        report(name, r)
        results[name] = r

    if args.compare:
        print()
        deltas = {}
        for name, r in results.items():
            path = OUT_DIR / args.compare / f"{name}.parquet"
            if not path.exists():
                print(f"[{name}] no saved run at {path}")
                continue
            deltas[name] = compare(name, r, pd.read_parquet(path), args.compare)
        v = verdict(deltas)
        if v:
            print(f"\nverdict: {v}")


if __name__ == "__main__":
    main()
