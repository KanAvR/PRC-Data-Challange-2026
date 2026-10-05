"""Retrain on all 12 months and write output/submitting.parquet.

Takes the same model options as validate.py (see --help), so a configuration
that passed validation is reproduced exactly.
"""
import argparse
import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data import align_categoricals, flight_ids, load_training, mvt_seconds
from resproxy import add_residual_proxy
from constraints import check_lobt_support, project_lobt
from high_lobt import APPLY_GAP_SEC, train_high_lobt
from encode import KEY_SETS, add_offset_encodings, add_target_encodings
from featcache import build_features_cached
from features import drop_arvt, drop_dest, drop_episode, drop_mixed, drop_proxy
from model import (MU_FAMILIES, trace_anchored, apply_model_args, add_model_args, analytic_tail_value, combine,
                   combine_mu,
                   predict_tail_proba, tail_options, train_tail_clf)
from runtime_profile import RuntimeProfile

RANKING_FILE = "dataset/ranking.parquet"
SUBMITTING_TEMPLATE = "dataset/submitting.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_model_args(ap)
    ap.add_argument("--out", default="output/submitting.parquet")
    ap.add_argument("--timings", metavar="JSON",
                    help="write low-overhead wall/CPU/RSS timings after every pipeline stage")
    args = ap.parse_args()
    apply_model_args(args)
    timings = RuntimeProfile(args.timings)

    print("Loading training data and building features...", flush=True)
    train_data, y_train, meta = load_training(adsb=args.adsb, timings=timings, climb=args.adsb_climb_id)
    with timings.stage("training.select_features"):
        if not args.proxy:
            train_data = drop_proxy(train_data)
        if not args.arvt:
            train_data = drop_arvt(train_data)
        if not args.episode:
            train_data = drop_episode(train_data)
        if not args.mixed_mode:
            train_data = drop_mixed(train_data)
        if not args.dest_proxy:
            train_data = drop_dest(train_data)
        if args.adsb and not args.adsb_nbr:
            import adsb
            train_data = adsb.drop_adsb_nbr(train_data)
        if args.lobt_bounds:
            check_lobt_support(y_train, train_data["gap_LOBT_flt"].values)
    print("Training rows:", len(train_data))

    print("Loading ranking data and building features...", flush=True)
    with timings.stage("ranking.parquet_read", path=RANKING_FILE):
        ranking_raw = pd.read_parquet(RANKING_FILE)
    ranking_feats = build_features_cached(RANKING_FILE, ranking_raw, timings=timings,
                                          stage="ranking.features")
    with timings.stage("ranking.select_departures"):
        dep_mask = (ranking_raw["PHASE_mvt"] == "DEP").values
    if args.adsb:
        with timings.stage("ranking.adsb_features"):
            import adsb
            ranking_feats = ranking_feats.join(adsb.dep_features(ranking_raw, "ranking",
                                                                 climb=args.adsb_climb_id))
    with timings.stage("ranking.select_features"):
        ranking_dep = ranking_feats.loc[dep_mask].copy()
        if not args.proxy:
            ranking_dep = drop_proxy(ranking_dep)
        if not args.arvt:
            ranking_dep = drop_arvt(ranking_dep)
        if not args.episode:
            ranking_dep = drop_episode(ranking_dep)
        if not args.mixed_mode:
            ranking_dep = drop_mixed(ranking_dep)
        if not args.dest_proxy:
            ranking_dep = drop_dest(ranking_dep)
        if args.adsb and not args.adsb_nbr:
            ranking_dep = adsb.drop_adsb_nbr(ranking_dep)
        ranking_ids = ranking_raw.loc[dep_mask, "MVT_ID_mvt"].values

    with timings.stage("features.align_categoricals"):
        align_categoricals(train_data, ranking_dep)
    if args.te:
        with timings.stage("features.target_encodings"):
            add_target_encodings(train_data, y_train, meta["month"].values, meta["flight"].values,
                                 ranking_dep, flight_ids(ranking_raw.loc[dep_mask]),
                                 keys=KEY_SETS[args.te_keys],
                                 counts=args.te_counts, fold_size=args.te_fold_size,
                                 count_oof_basis=args.te_count_oof_basis,
                                 mean_oof_basis=args.te_mean_oof_basis,
                                 flight_backoff=args.te_flight_backoff,
                                 flight_dropout=args.te_flight_dropout,
                                 airport_prior=args.te_airport_prior)
    if args.adsb and args.adsb_offset_te:
        with timings.stage("features.adsb_offset_encodings"):
            add_offset_encodings(train_data, y_train, meta["month"].values, ranking_dep)
    if args.res_proxy:
        if not args.te:
            raise SystemExit("--res-proxy needs --te (it subtracts te_stand_rwy)")
        with timings.stage("features.residual_proxy"):
            add_residual_proxy(train_data, meta["t"].values)
            add_residual_proxy(ranking_dep, mvt_seconds(ranking_raw.loc[dep_mask]))

    mus = []
    for fam in args.mu:
        print(f"Training normal-regime regressor ({fam}) on all data...", flush=True)
        fit = trace_anchored(MU_FAMILIES[fam]) if args.adsb_anchor else MU_FAMILIES[fam]
        with timings.stage(f"mu.{fam}.fit"):
            model = fit(train_data, y_train)
        with timings.stage(f"mu.{fam}.predict"):
            mus.append(np.clip(model.predict(ranking_dep), 0, None))
    with timings.stage("mu.combine"):
        mu_pred = combine_mu({f"mu_{f}": m for f, m in zip(args.mu, mus)}, args.mu, args)

    print("Training tail classifier on all data...", flush=True)
    with timings.stage("tail_classifier.fit"):
        clf = train_tail_clf(train_data, y_train, pooled=args.clf_pooled, no_lobt=args.clf_nolobt,
                             stakes=args.stakes, fit_pooled=not args.high_lobt)

    print("Predicting on ranking DEP rows...")
    with timings.stage("tail_classifier.predict"):
        p = predict_tail_proba(clf, ranking_dep)
    with timings.stage("tail.combine_and_constrain"):
        tail_kw = tail_options(args, clf, ranking_dep)
        q_day = tail_kw.get("q_day")
        v = analytic_tail_value(ranking_dep)
        gap_lobt = ranking_dep["gap_LOBT_flt"].values.astype(float)
        preds = combine(mu_pred, p, v, **tail_kw)
        if args.lobt_bounds:
            preds = project_lobt(preds, gap_lobt)
    high_pred = np.full(len(preds), np.nan)
    if args.high_lobt:
        print("Training matched long-taxi specialist...", flush=True)
        with timings.stage("long_taxi_specialist.fit"):
            specialist = train_high_lobt(train_data, y_train)
        high = gap_lobt > APPLY_GAP_SEC
        if high.any():
            with timings.stage("long_taxi_specialist.predict", rows=int(high.sum())):
                high_pred[high] = specialist.predict(ranking_dep.loc[high])
                preds[high] = high_pred[high]

    with timings.stage("submission.assemble_and_validate"):
        pred_df = pd.DataFrame({"MVT_ID_mvt": ranking_ids, "TAXITIME_SEC_mvt": preds})

        template = pd.read_parquet(SUBMITTING_TEMPLATE)
        submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

        assert len(submission) == len(template), "row count mismatch"
        assert submission["TAXITIME_SEC_mvt"].isna().sum() == 0, "missing predictions"
        assert (submission["MVT_ID_mvt"].values == template["MVT_ID_mvt"].values).all(), "id order mismatch"

        submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")
        submission["MVT_ID_mvt"] = submission["MVT_ID_mvt"].astype("float64")

    with timings.stage("submission.parquet_write", path=args.out, rows=len(submission)):
        submission.to_parquet(args.out, index=False)
    # The pieces a post-hoc correction needs, saved beside the submission the way
    # refine_submission.py writes its audit files. predict.py emits only ids and
    # predictions, so without this a mu-keyed correction cannot be applied to a
    # finished submission without retraining the whole ensemble.
    with timings.stage("parts.assemble"):
        parts = pd.DataFrame({
            "MVT_ID_mvt": ranking_ids, "mu": mu_pred, "p": p, "v": v,
            "q_day": np.zeros(len(preds)) if q_day is None else q_day,
            "day_late": bool(args.day_late), "day_late_step": bool(args.day_late_step),
            "tail_cap": bool(args.tail_cap), "gap_LOBT_flt": gap_lobt,
            "lobt_bounds": bool(args.lobt_bounds), "high_lobt": bool(args.high_lobt),
            "high_lobt_pred": high_pred, "pred": preds,
        })
        parts_path = str(Path(args.out).with_suffix("")) + ".parts.parquet"
    with timings.stage("parts.parquet_write", path=parts_path, rows=len(parts)):
        parts.to_parquet(parts_path, index=False)
    print(f"Wrote {parts_path} with {len(parts)} rows")
    print(f"Wrote {args.out} with {len(submission)} rows")
    print(f"  predictions above 6000s: {(preds > 6000).sum()}")
    print(submission["TAXITIME_SEC_mvt"].describe())
    timings.finish()


if __name__ == "__main__":
    main()
