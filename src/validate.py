"""Multi-split validation: the rule every model change has to pass.

    holdout  train on the other 10 months, score Jan+Jul 2025 (mirrors ranking)
    inner    score Feb+Aug; train on the 8 months left after also dropping Jan+Jul
    dec_jun  score Dec+Jun; train without Jan+Jul either
    mar_sep  score Mar+Sep; train without Jan+Jul either

Only the holdout trains on Jan+Jul, so a constant tuned on the other splits can
be checked out of sample on it. Each scored pair is a winter and a summer month,
like the ranking set.

RMSE is dominated by a handful of tail rows, some of which no model can predict
(a block time from the previous day that matches no timestamp), so a single
number is not evidence. Every run saves its predictions under
output/validate/<tag>/, and --compare scores the current run against a saved
one with a paired bootstrap over days, per split and pooled over all splits,
both on all rows and on "robust" rows (unexplainable tail rows removed). The
verdict is a win only when the pooled robust interval is below zero and no
split is significantly worse.

    python src/validate.py --tag baseline --splits holdout inner
    python src/validate.py --tag mychange --compare baseline
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from data import align_categoricals, load_training
from constraints import check_lobt_support, project_lobt
from high_lobt import APPLY_GAP_SEC, train_high_lobt
from encode import add_target_encodings
from model import (MU_FAMILIES, TAIL_THR, add_model_args,
                   analytic_tail_value, combine, day_late_weight,
                   predict_tail_proba, train_tail_clf)

NO_JAN_JUL = ["2025-01", "2025-07"]
SPLITS = {
    "holdout": dict(valid=["2025-01", "2025-07"], drop=[]),
    "inner": dict(valid=["2025-02", "2025-08"], drop=NO_JAN_JUL),
    "dec_jun": dict(valid=["2025-12", "2025-06"], drop=NO_JAN_JUL),
    "mar_sep": dict(valid=["2025-03", "2025-09"], drop=NO_JAN_JUL),
}
OUT_DIR = pathlib.Path("output/validate")
N_BOOT = 2000
P_SWEEP = [0.8, 0.9, 1.0, 1.15, 1.35, 1.7]

# timestamp gaps saved with every run, for the robust mask and offline experiments
GAP_COLS = ["gap_SCHED_TIME_UTC_mvt", "gap_LOBT_flt", "gap_IOBT_flt",
            "gap_EOBT_1_flt", "gap_AOBT_3_flt"]
DAY = 86400.0
EXPLAIN_TOL = 600


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def unexplainable(r):
    """Tail rows whose label matches no timestamp gap, even shifted by a day.

    None if the run was saved before gap columns existed."""
    if not all(c in r.columns for c in GAP_COLS):
        return None
    y = r["y"].values
    hit = np.zeros(len(r), bool)
    for c in GAP_COLS:
        g = r[c].values.astype(float)
        hit |= np.abs(y - g) <= EXPLAIN_TOL
    hit |= (y >= DAY) & (y < DAY + TAIL_THR)
    return (y > TAIL_THR) & ~hit


def run_split(name, X, y, meta, args):
    spec = SPLITS[name]
    va = meta["month"].isin(spec["valid"]).values
    tr = ~va & ~meta["month"].isin(spec["drop"]).values
    X_tr, X_va = X.loc[tr].copy(), X.loc[va].copy()
    y_tr, y_va = y[tr], y[va]
    if args.lobt_bounds:
        check_lobt_support(y_tr, X_tr["gap_LOBT_flt"].values)
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
    clf = train_tail_clf(X_tr, y_tr, pooled=args.clf_pooled, no_lobt=args.clf_nolobt,
                         stakes=args.stakes, fit_pooled=not args.high_lobt)
    p = predict_tail_proba(clf, X_va)
    v = analytic_tail_value(X_va)
    # always saved, so blend.py can switch the day-late component on or off
    q_day = day_late_weight(clf, X_va)
    pred = combine(mu, p, v, q_day=q_day if args.day_late else None)
    pred_unconstrained = pred.copy()
    if args.lobt_bounds:
        pred = project_lobt(pred, X_va["gap_LOBT_flt"].values)
    high_pred = np.full(len(pred), np.nan)
    if args.high_lobt:
        specialist = train_high_lobt(X_tr, y_tr)
        high = X_va["gap_LOBT_flt"].values > APPLY_GAP_SEC
        if high.any():
            high_pred[high] = specialist.predict(X_va.loc[high])
            pred[high] = high_pred[high]

    return pd.DataFrame({
        "MVT_ID_mvt": meta.loc[va, "MVT_ID_mvt"].values,
        "date": meta.loc[va, "date"].values, "ADEP_mvt": meta.loc[va, "ADEP_mvt"].values,
        "has_lobt": ~np.isnan(X_va["gap_LOBT_flt"].values.astype(float)),
        "y": y_va, "mu": mu, "p": p, "v": v, "pred": pred, "q_day": q_day, "p_calibrated": True, **mus,
        "pred_unconstrained": pred_unconstrained,
        "lobt_bounds": args.lobt_bounds,
        "high_lobt": args.high_lobt, "high_lobt_pred": high_pred,
        **{c: X_va[c].values.astype(float) for c in GAP_COLS},
    })


def report(name, r):
    y, tail = r["y"].values, r["y"].values > TAIL_THR
    print(f"[{name}] scored over all retained departures (normal + tail)")
    print(f"  mixture          {rmse(y, r['pred']):8.2f}")
    if "pred_unconstrained" in r:
        print(f"  before correction {rmse(y, r['pred_unconstrained']):7.2f}   "
              f"({int((r['pred'] != r['pred_unconstrained']).sum())} rows corrected)")
    bad = unexplainable(r)
    if bad is not None:
        print(f"  robust           {rmse(y[~bad], r['pred'][~bad]):8.2f}   "
              f"({int(bad.sum())} unexplainable tail rows removed)")
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
    print(f"  extra p boost    {sweep}   (before corrections; p already carries CLF_BOOST)")
    per_apt = r.assign(se=(r["y"] - r["pred"]) ** 2).groupby("ADEP_mvt")["se"].mean() ** 0.5
    print("  per airport      " + "  ".join(f"{a}:{v:.0f}" for a, v in per_apt.items()))


def _day_sums(cur, base, keep):
    d = pd.DataFrame({"date": cur["date"].values[keep],
                      "cur": ((cur["y"] - cur["pred"]) ** 2).values[keep],
                      "base": ((base["y"] - base["pred"]) ** 2).values[keep]})
    g = d.groupby("date")
    s = g[["cur", "base"]].sum()
    return s["cur"].values, s["base"].values, g.size().values


def _boot(days, rng):
    """Paired day-level bootstrap of the RMSE difference; days = (cur, base, n)."""
    cur, base, n = days
    idx = rng.integers(0, len(n), size=(N_BOOT, len(n)))
    nn = n[idx].sum(axis=1)
    delta = np.sqrt(cur[idx].sum(axis=1) / nn) - np.sqrt(base[idx].sum(axis=1) / nn)
    point = np.sqrt(cur.sum() / n.sum()) - np.sqrt(base.sum() / n.sum())
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return point, lo, hi, (delta < 0).mean()


def _fmt(label, s):
    point, lo, hi, frac = s
    return f"{label} {point:+7.2f}  CI {lo:+7.2f} .. {hi:+7.2f}  better in {100 * frac:3.0f}%"


def compare(name, cur, base, base_tag):
    """Per-split comparison. Returns day sums for pooling, or None."""
    if len(cur) != len(base) or not np.array_equal(cur["y"].values, base["y"].values):
        print(f"[{name}] cannot compare with '{base_tag}': different validation rows")
        return None
    if np.array_equal(cur["pred"].values, base["pred"].values):
        print(f"[{name}] predictions are identical to '{base_tag}': the change had no effect")
    bad = unexplainable(cur)
    if bad is None:
        bad = unexplainable(base)
    if bad is None:
        bad = np.zeros(len(cur), bool)
    rng = np.random.default_rng(0)
    full = _day_sums(cur, base, np.ones(len(cur), bool))
    robust = _day_sums(cur, base, ~bad)
    tail = cur["y"].values > TAIL_THR
    normal_delta = (rmse(cur["y"][~tail], cur["pred"][~tail])
                    - rmse(base["y"][~tail], base["pred"][~tail]))
    print(f"[{name:7s}] vs '{base_tag}':  {_fmt('all', _boot(full, rng))}   |  "
          f"{_fmt('robust', _boot(robust, rng))}   |  normal rows {normal_delta:+.2f}")
    return {"full": full, "robust": robust, "robust_stats": _boot(robust, rng)}


def verdict(results):
    """Pool the per-split day sums and decide.

    A win needs the pooled robust interval below zero and no split whose robust
    interval is entirely above zero."""
    results = {k: v for k, v in results.items() if v is not None}
    if not results:
        return None
    rng = np.random.default_rng(1)
    pooled = {}
    for kind in ("full", "robust"):
        parts = [r[kind] for r in results.values()]
        pooled[kind] = _boot(tuple(np.concatenate([p[i] for p in parts]) for i in range(3)), rng)
    print(f"[pooled ] {len(results)} splits:  {_fmt('all', pooled['full'])}   |  "
          f"{_fmt('robust', pooled['robust'])}")
    worse = [k for k, r in results.items() if r["robust_stats"][1] > 0]
    if worse:
        return f"does NOT win: significantly worse on {', '.join(worse)}"
    if pooled["robust"][2] < 0:
        return f"wins (pooled robust CI below zero over {len(results)} splits)"
    if pooled["robust"][0] < 0:
        return "better on the pooled point estimate, but the CI crosses zero: not established"
    return "does NOT win"


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
