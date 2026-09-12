import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features import CAT_COLS, build_features
from model import (analytic_tail_value, combine, predict_tail_proba, train_mu,
                   train_tail_clf)

TRAIN_FILES = sorted(glob.glob("dataset/training_*.parquet"))
RANKING_FILE = "dataset/ranking.parquet"
SUBMITTING_TEMPLATE = "dataset/submitting.parquet"

MIN_TAXI_SEC = 0


def load_training():
    frames = []
    for f in TRAIN_FILES:
        df = pd.read_parquet(f)
        feats = build_features(df)
        feats["target"] = df["TAXITIME_SEC_mvt"].astype(float)
        frames.append(feats.loc[df["PHASE_mvt"].values == "DEP"])
    full = pd.concat(frames, ignore_index=True)
    return full[full["target"] >= MIN_TAXI_SEC]


def main():
    print("Loading training data and building features...")
    train_data = load_training()
    y_train = train_data.pop("target").values
    print("Training rows:", len(train_data))

    print("Loading ranking data and building features...")
    ranking_raw = pd.read_parquet(RANKING_FILE)
    ranking_feats = build_features(ranking_raw)
    dep_mask = (ranking_raw["PHASE_mvt"] == "DEP").values
    ranking_dep = ranking_feats.loc[dep_mask].copy()
    ranking_ids = ranking_raw.loc[dep_mask, "MVT_ID_mvt"].values

    for c in CAT_COLS:
        cats = sorted(set(train_data[c].astype(str).unique()) | set(ranking_dep[c].astype(str).unique()))
        dtype = pd.CategoricalDtype(categories=cats)
        train_data[c] = train_data[c].astype(str).astype(dtype)
        ranking_dep[c] = ranking_dep[c].astype(str).astype(dtype)

    print("Training normal-regime regressor on all data...")
    mu = train_mu(train_data, y_train)
    mu.save_model("output/model_final.txt")

    print("Training tail classifier on all data...")
    clf = train_tail_clf(train_data, y_train)

    print("Predicting on ranking DEP rows...")
    mu_pred = np.clip(mu.predict(ranking_dep), 0, None)
    p = predict_tail_proba(clf, ranking_dep)
    preds = combine(mu_pred, p, analytic_tail_value(ranking_dep))

    pred_df = pd.DataFrame({"MVT_ID_mvt": ranking_ids, "TAXITIME_SEC_mvt": preds})

    template = pd.read_parquet(SUBMITTING_TEMPLATE)
    submission = template[["MVT_ID_mvt"]].merge(pred_df, on="MVT_ID_mvt", how="left")

    assert len(submission) == len(template), "row count mismatch"
    assert submission["TAXITIME_SEC_mvt"].isna().sum() == 0, "missing predictions"
    assert (submission["MVT_ID_mvt"].values == template["MVT_ID_mvt"].values).all(), "id order mismatch"

    submission["TAXITIME_SEC_mvt"] = submission["TAXITIME_SEC_mvt"].astype("float64")
    submission["MVT_ID_mvt"] = submission["MVT_ID_mvt"].astype("float64")

    out_path = "output/submitting.parquet"
    submission.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(submission)} rows")
    print(f"  predictions above 6000s: {(preds > 6000).sum()} (was 1 in the capped version)")
    print(submission["TAXITIME_SEC_mvt"].describe())


if __name__ == "__main__":
    main()
