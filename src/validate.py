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
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from data import align_categoricals, flight_ids, load_training
from features import drop_arvt, drop_dest, drop_episode, drop_mixed, drop_proxy
from constraints import check_lobt_support, project_lobt
from high_lobt import APPLY_GAP_SEC, train_high_lobt
from encode import KEY_SETS, add_offset_encodings, add_target_encodings
from resproxy import RES_PROXY_COLS, add_residual_proxy
from model import (MU_FAMILIES, trace_anchored, lirf_schedmixed, apply_model_args, seasonal_weight, TAIL_THR, add_model_args, combine_mu,
                   analytic_tail_value, combine, day_late_step_weight, day_late_weight,
                   predict_tail_proba, tail_options, train_tail_clf)

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


def ranking_unseen_rates():
    """Per airport, the share of July 2026 departures whose flight key never
    appears in 2025. Label-free: flight keys only."""
    raw = pd.read_parquet("dataset/ranking.parquet",
                          columns=["PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[(raw["PHASE_mvt"] == "DEP") & (raw["MVT_TIME_UTC_mvt"].dt.month == 7)]
    seen = set()
    for f in sorted(pathlib.Path("dataset").glob("training_*.parquet")):
        t = pd.read_parquet(f, columns=["PHASE_mvt", "ADEP_mvt", "FLIGHT_mvt"])
        seen |= set(flight_ids(t[t["PHASE_mvt"] == "DEP"]))
    unseen = ~pd.Series(flight_ids(raw)).isin(seen).values
    return pd.Series(unseen).groupby(raw["ADEP_mvt"].values).mean().to_dict()


def flight_mask(meta_va, summer_month, rates):
    """Rows whose flight key a renamed callsign would hide: in the split's summer
    month only (like July 2026; January 2026 is 2.3% unseen), a stable hash of
    the key selects each airport's July 2026 share of flights."""
    # str() per key: astype(str) keeps a missing key as a float NaN under pandas 3
    h = np.array([zlib.crc32(str(k).encode()) % 10000 for k in meta_va["flight"]])
    rate = meta_va["ADEP_mvt"].map(rates).fillna(0.0).values
    return (meta_va["month"].values == summer_month) & (h < rate * 10000)


def run_split(name, X, y, meta, args, rates=None):
    """The split's predictions; with rates (--mask-flight), a pair: the same
    models' predictions with renamed-callsign flights masked, then unmasked."""
    spec = SPLITS[name]
    va = meta["month"].isin(spec["valid"]).values
    tr = ~va & ~meta["month"].isin(spec["drop"]).values
    X_tr, X_va = X.loc[tr].copy(), X.loc[va].copy()
    y_tr, y_va = y[tr], y[va]
    if args.adsb and args.adsb_day_dropout:
        import adsb
        drop = adsb.day_dropout_rows(meta.loc[tr, "ADEP_mvt"].values, meta.loc[tr, "date"].values,
                                     args.adsb_day_dropout)
        adsb.blank_untraced(X_tr, drop)
        print(f"  adsb day dropout: {drop.mean():.1%} of training rows blanked", flush=True)
    if args.lobt_bounds:
        check_lobt_support(y_tr, X_tr["gap_LOBT_flt"].values)
    align_categoricals(X_tr, X_va)
    X_va_m = unseen = None
    if rates is not None:
        unseen = flight_mask(meta.loc[va], spec["valid"][1], rates)
        X_va_m = X_va.copy()
        print(f"  masking {int(unseen.sum())} rows ({unseen.mean():.1%} of the split) as renamed callsigns",
              flush=True)
    if args.te:
        add_target_encodings(X_tr, y_tr, meta.loc[tr, "month"].values, meta.loc[tr, "flight"].values,
                             X_va, meta.loc[va, "flight"].values,
                             keys=KEY_SETS[args.te_keys],
                             counts=args.te_counts, fold_size=args.te_fold_size,
                             count_oof_basis=args.te_count_oof_basis,
                             mean_oof_basis=args.te_mean_oof_basis,
                             flight_backoff=args.te_flight_backoff, flight_dropout=args.te_flight_dropout,
                             airport_prior=args.te_airport_prior,
                             X_apply_masked=X_va_m, flight_unseen=unseen)
    if args.adsb and args.adsb_offset_te:
        add_offset_encodings(X_tr, y_tr, meta.loc[tr, "month"].values, X_va, X_va_m)
    if args.res_proxy:
        if not args.te:
            raise SystemExit("--res-proxy needs --te (it subtracts te_stand_rwy)")
        add_residual_proxy(X_tr, meta.loc[tr, "t"].values)
        add_residual_proxy(X_va, meta.loc[va, "t"].values)
        if X_va_m is not None:  # te_stand_rwy is not flight-keyed, so the mask leaves these alone
            for c in RES_PROXY_COLS:
                X_va_m[c] = X_va[c].values
    if getattr(args, "mask_adsb_lirf", False):
        # July 2026 LIRF: no traces at all. Same rows, same models, adsb_* blanked
        import adsb
        unseen = ((meta.loc[va, "ADEP_mvt"].values == "LIRF")
                  & (meta.loc[va, "month"].values == spec["valid"][1]))
        X_va_m = X_va.copy()
        adsb.blank_untraced(X_va_m, unseen)
        print(f"  masking ADS-B on {int(unseen.sum())} summer LIRF rows", flush=True)
    print(f"\n[{name}] train={len(X_tr)} valid={len(X_va)} "
          f"tail={int((y_tr > TAIL_THR).sum())}/{int((y_va > TAIL_THR).sum())}", flush=True)

    # several families are averaged with equal weight; each is saved on its own
    # so blend.py can re-weight them without retraining
    mus, mus_m = {}, {}
    if args.reuse_mu:
        saved = pd.read_parquet(OUT_DIR / args.reuse_mu / f"{name}.parquet")
        assert np.array_equal(saved["y"].values, y_va), "reuse-mu run has different rows"
        mus = {c: saved[c].values for c in saved.columns if c.startswith("mu_")}
        mu = saved["mu"].values
    else:
        for fam in args.mu:
            kw = {"monotone": True} if (getattr(args, "monotone", False) and fam == "lgb") else {}
            if getattr(args, "season_tau", 0) and fam == "lgb":
                kw["weight"] = seasonal_weight(meta.loc[tr, "month"].values,
                                               spec["valid"], args.season_tau)
            fit = trace_anchored(MU_FAMILIES[fam]) if args.adsb_anchor else MU_FAMILIES[fam]
            if args.lirf_schedmix:
                fit = lirf_schedmixed(fit)
            model = fit(X_tr, y_tr, **kw)
            mus[f"mu_{fam}"] = np.clip(model.predict(X_va), 0, None)
            if X_va_m is not None:
                mus_m[f"mu_{fam}"] = np.clip(model.predict(X_va_m), 0, None)
            print(f"  {fam}: done", flush=True)
        mu = combine_mu(mus, args.mu, args)
    clf_tr, clf_va = X_tr, X_va
    if "inbound_air_after_sched" in meta.columns:
        # the airframe's trace arrival reaches the tail classifier only, never mu
        clf_tr = X_tr.assign(inbound_air_after_sched=meta.loc[tr, "inbound_air_after_sched"].values)
        clf_va = X_va.assign(inbound_air_after_sched=meta.loc[va, "inbound_air_after_sched"].values)
    if args.clf_copy_p:
        from copyp import add_copy_p
        clf_tr, clf_va = clf_tr.copy(), clf_va.copy()
        add_copy_p(clf_tr, y_tr, meta.loc[tr, "month"].values, clf_va)
    if args.clf_service:
        import service
        clf_tr = clf_tr.assign(**{c: meta.loc[tr, c].values for c in service.COLS})
        clf_va = clf_va.assign(**{c: meta.loc[va, c].values for c in service.COLS})
    if args.clf_arr_habit:
        import service
        clf_tr = clf_tr.assign(**{c: meta.loc[tr, c].values for c in service.ARR_COLS})
        clf_va = clf_va.assign(**{c: meta.loc[va, c].values for c in service.ARR_COLS})
    clf = train_tail_clf(clf_tr, y_tr, pooled=args.clf_pooled, no_lobt=args.clf_nolobt,
                         stakes=args.stakes, fit_pooled=not args.high_lobt, inbound=args.clf_inbound,
                         inbound_mono=args.clf_inbound_mono)
    p = predict_tail_proba(clf, clf_va)
    v = analytic_tail_value(X_va)
    # always saved, so blend.py can switch the day-late component on or off
    q_day = day_late_weight(clf, X_va)
    q_day_step = day_late_step_weight(clf, X_va)
    tail_kw = tail_options(args, clf, X_va)
    high_pred = np.full(len(X_va), np.nan)
    high = X_va["gap_LOBT_flt"].values > APPLY_GAP_SEC
    if args.high_lobt:
        # the specialist sees no te_* column, so masking cannot change it
        specialist = train_high_lobt(X_tr, y_tr)
        if high.any():
            high_pred[high] = specialist.predict(X_va.loc[high])

    def assemble(mu, mus):
        pred = combine(mu, p, v, **tail_kw)
        pred_unconstrained = pred.copy()
        if args.lobt_bounds:
            pred = project_lobt(pred, X_va["gap_LOBT_flt"].values)
        if args.high_lobt and high.any():
            pred[high] = high_pred[high]
        return pd.DataFrame({
            "MVT_ID_mvt": meta.loc[va, "MVT_ID_mvt"].values,
            "date": meta.loc[va, "date"].values, "ADEP_mvt": meta.loc[va, "ADEP_mvt"].values,
            "has_lobt": ~np.isnan(X_va["gap_LOBT_flt"].values.astype(float)),
            "y": y_va, "mu": mu, "p": p, "v": v, "pred": pred, "q_day": q_day, "q_day_step": q_day_step,
            "p_calibrated": True, **mus,
            "pred_unconstrained": pred_unconstrained,
            "lobt_bounds": args.lobt_bounds,
            "high_lobt": args.high_lobt, "high_lobt_pred": high_pred,
            **{c: X_va[c].values.astype(float) for c in GAP_COLS},
        })

    out = assemble(mu, mus)
    if X_va_m is None:
        return out
    masked = assemble(combine_mu(mus_m, args.mu, args), mus_m)
    masked["flight_masked"] = unseen
    return masked, out


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
    ap.add_argument("--mask-flight", action="store_true",
                    help="evaluation only: in each split's summer month, encode each airport's "
                         "July 2026 share of flights as unseen (renamed callsigns). The masked "
                         "run is saved as TAG, the same models unmasked as TAG_unmasked")
    ap.add_argument("--mask-adsb-lirf", action="store_true",
                    help="evaluation only: in each split's summer month, blank LIRF's adsb_* "
                         "columns as July 2026 LIRF carries them (no traces). Saved as TAG "
                         "(masked) and TAG_unmasked, like --mask-flight (NEXT_STEPS.md, B32)")
    args = ap.parse_args()
    apply_model_args(args)
    if (args.mask_flight or args.mask_adsb_lirf) and args.reuse_mu:
        ap.error("--mask-flight / --mask-adsb-lirf need the mu models, not a saved mu")
    if args.mask_flight and args.mask_adsb_lirf:
        ap.error("use one mask at a time")

    print("Loading training data and building features...", flush=True)
    X, y, meta = load_training(adsb=args.adsb, climb=args.adsb_climb_id,
                               climb_nbr_clean=args.adsb_climb_nbr_clean,
                               inbound_air=args.clf_inbound and args.clf_inbound_air,
                               service=args.clf_service, arr_habit=args.clf_arr_habit)
    if not args.proxy:
        X = drop_proxy(X)
    if not args.arvt:
        X = drop_arvt(X)
    if not args.episode:
        X = drop_episode(X)
    if not args.mixed_mode:
        X = drop_mixed(X)
    if not args.dest_proxy:
        X = drop_dest(X)
    if args.adsb and not args.adsb_nbr:
        import adsb
        X = adsb.drop_adsb_nbr(X)
    if args.adsb and not args.adsb_pos:
        import adsb
        X = adsb.drop_adsb_pos(X)
    if args.adsb and args.trace_te:
        from tracete import add_trace_encodings
        add_trace_encodings(X, meta["date"].values)

    out = OUT_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    rates = None
    if args.mask_flight:
        rates = ranking_unseen_rates()
        print("July 2026 unseen flight share by airport: "
              + "  ".join(f"{a}:{r:.1%}" for a, r in sorted(rates.items())), flush=True)
    if args.mask_flight or args.mask_adsb_lirf:
        (OUT_DIR / f"{args.tag}_unmasked").mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.splits:
        r = run_split(name, X, y, meta, args, rates)
        if args.mask_flight or args.mask_adsb_lirf:
            r, r_unmasked = r
            r_unmasked.to_parquet(OUT_DIR / f"{args.tag}_unmasked" / f"{name}.parquet", index=False)
            yv, m = r["y"].values, r["flight_masked"].values
            nm = yv <= TAIL_THR
            print(f"[{name}] masking cost: all "
                  f"{rmse(yv, r['pred'].values) - rmse(yv, r_unmasked['pred'].values):+.3f}  normal "
                  f"{rmse(yv[nm], r['pred'].values[nm]) - rmse(yv[nm], r_unmasked['pred'].values[nm]):+.3f}  "
                  f"masked normal rows {rmse(yv[m & nm], r['pred'].values[m & nm]):.1f} vs unmasked "
                  f"{rmse(yv[m & nm], r_unmasked['pred'].values[m & nm]):.1f} (n={int((m & nm).sum())})",
                  flush=True)
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
