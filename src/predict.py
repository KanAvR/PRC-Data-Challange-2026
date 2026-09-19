"""Retrain on all 12 months and write output/submitting.parquet.

Takes the same model options as validate.py (see --help), so a configuration
that passed validation is reproduced exactly.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data import align_categoricals, flight_ids, load_training
from constraints import check_lobt_support, project_lobt
from high_lobt import train_high_lobt
from encode import add_target_encodings
from featcache import build_features_cached
from model import (MU_FAMILIES, add_model_args, analytic_tail_value, combine,
                   day_late_weight, predict_tail_proba, train_tail_clf)

RANKING_FILE = "dataset/ranking.parquet"
SUBMITTING_TEMPLATE = "dataset/submitting.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_model_args(ap)
    ap.add_argument("--out", default="output/submitting.parquet")
    args = ap.parse_args()

    print("Loading training data and building features...")
    train_data, y_train, meta = load_training()
    if args.lobt_bounds:
        check_lobt_support(y_train, train_data["gap_LOBT_flt"].values)
    print("Training rows:", len(train_data))

    print("Loading ranking data and building features...")
    ranking_raw = pd.read_parquet(RANKING_FILE)
    ranking_feats = build_features_cached(RANKING_FILE, ranking_raw)
    dep_mask = (ranking_raw["PHASE_mvt"] == "DEP").values
    ranking_dep = ranking_feats.loc[dep_mask].copy()
    ranking_ids = ranking_raw.loc[dep_mask, "MVT_ID_mvt"].values

    align_categoricals(train_data, ranking_dep)
    if args.te:
        add_target_encodings(train_data, y_train, meta["month"].values, meta["flight"].values,
                             ranking_dep, flight_ids(ranking_raw.loc[dep_mask]))

    mus = []
    for fam in args.mu:
        print(f"Training normal-regime regressor ({fam}) on all data...", flush=True)
        mus.append(np.clip(MU_FAMILIES[fam](train_data, y_train).predict(ranking_dep), 0, None))
    mu_pred = np.mean(mus, axis=0)

    print("Training tail classifier on all data...", flush=True)
    clf = train_tail_clf(train_data, y_train, pooled=args.clf_pooled, no_lobt=args.clf_nolobt,
                         stakes=args.stakes, fit_pooled=not args.high_lobt)

    print("Predicting on ranking DEP rows...")
    p = predict_tail_proba(clf, ranking_dep)
    q_day = day_late_weight(clf, ranking_dep) if args.day_late else None
    preds = combine(mu_pred, p, analytic_tail_value(ranking_dep), q_day=q_day)
    if args.lobt_bounds:
        preds = project_lobt(preds, ranking_dep["gap_LOBT_flt"].values)
    if args.high_lobt:
        print("Training matched long-taxi specialist...", flush=True)
        specialist = train_high_lobt(train_data, y_train)
        preds = specialist.refine(preds, ranking_dep)

    pred_df = pd.DataFrame({"MVT_ID_mvt": ranking_ids, "TAXITIME_SEC_mvt": preds})

    template = pd.read_parquet(SUBMITTING_TEMPLATE)
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    assert len(submission) == len(template), "row count mismatch"
    assert submission["TAXITIME_SEC_mvt"].isna().sum() == 0, "missing predictions"
    assert (submission["MVT_ID_mvt"].values == template["MVT_ID_mvt"].values).all(), "id order mismatch"

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")
    submission["MVT_ID_mvt"] = submission["MVT_ID_mvt"].astype("float64")

    submission.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} with {len(submission)} rows")
    print(f"  predictions above 6000s: {(preds > 6000).sum()}")
    print(submission["TAXITIME_SEC_mvt"].describe())


if __name__ == "__main__":
    main()
