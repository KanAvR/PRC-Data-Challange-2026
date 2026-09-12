import glob
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features import CAT_COLS, build_features

TRAIN_FILES = sorted(glob.glob("dataset/training_*.parquet"))
RANKING_FILE = "dataset/ranking.parquet"
SUBMITTING_TEMPLATE = "dataset/submitting.parquet"

MAX_TAXI_SEC = 6000
MIN_TAXI_SEC = 0
FINAL_NUM_ROUNDS = 900

PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.02,
    num_leaves=511,
    min_data_in_leaf=30,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
)


def load_training():
    frames = []
    for f in TRAIN_FILES:
        df = pd.read_parquet(f)
        feats = build_features(df)
        feats["target"] = df["TAXITIME_SEC_mvt"].astype(float)
        frames.append(feats.loc[df["PHASE_mvt"].values == "DEP"])
    full = pd.concat(frames, ignore_index=True)
    full = full[(full["target"] >= MIN_TAXI_SEC) & (full["target"] <= MAX_TAXI_SEC)]
    return full


def main():
    print("Loading training data and building features...")
    train_data = load_training()
    y_train = train_data.pop("target")
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

    train_set = lgb.Dataset(train_data, label=y_train, categorical_feature=CAT_COLS)

    print("Training final model on all data...")
    model = lgb.train(PARAMS, train_set, num_boost_round=FINAL_NUM_ROUNDS)
    model.save_model("output/model_final.txt")

    print("Predicting on ranking DEP rows...")
    preds = model.predict(ranking_dep)
    preds = np.clip(preds, 30, None)

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
    print(submission.describe())


if __name__ == "__main__":
    main()
