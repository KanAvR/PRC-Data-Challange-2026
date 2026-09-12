import glob
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, "src")
from features import CAT_COLS, build_features

TRAIN_FILES = sorted(glob.glob("dataset/training_*.parquet"))

MAX_TAXI_SEC = 6000
MIN_TAXI_SEC = 0


def load_training():
    frames = []
    for f in TRAIN_FILES:
        df = pd.read_parquet(f)
        feats = build_features(df)
        feats["target"] = df["TAXITIME_SEC_mvt"].astype(float)
        feats["phase"] = df["PHASE_mvt"].values
        feats["month_file"] = f
        frames.append(feats.loc[df["PHASE_mvt"].values == "DEP"])
    full = pd.concat(frames, ignore_index=True)
    full = full[(full["target"] >= MIN_TAXI_SEC) & (full["target"] <= MAX_TAXI_SEC)]
    return full


def main():
    print("Loading training data and building features...")
    data = load_training()
    print("Rows:", len(data))

    holdout_months = data["month_file"].str.contains("2025-01-01|2025-07-01")
    train_df = data.loc[~holdout_months].drop(columns=["phase", "month_file"])
    valid_df = data.loc[holdout_months].drop(columns=["phase", "month_file"])

    y_train = train_df.pop("target")
    y_valid = valid_df.pop("target")

    for c in CAT_COLS:
        cats = sorted(set(train_df[c].astype(str).unique()) | set(valid_df[c].astype(str).unique()))
        dtype = pd.CategoricalDtype(categories=cats)
        train_df[c] = train_df[c].astype(str).astype(dtype)
        valid_df[c] = valid_df[c].astype(str).astype(dtype)

    train_set = lgb.Dataset(train_df, label=y_train, categorical_feature=CAT_COLS)
    valid_set = lgb.Dataset(valid_df, label=y_valid, categorical_feature=CAT_COLS, reference=train_set)

    params = dict(
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

    model = lgb.train(
        params,
        train_set,
        num_boost_round=6000,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)],
    )

    pred = model.predict(valid_df, num_iteration=model.best_iteration)
    pred = np.clip(pred, 0, None)
    rmse = root_mean_squared_error(y_valid, pred)
    print(f"Holdout (Jan+Jul) RMSE: {rmse:.2f}")
    print(f"Best iteration: {model.best_iteration}")

    importances = pd.Series(
        model.feature_importance(importance_type="gain"), index=train_df.columns
    ).sort_values(ascending=False)
    print("\nTop feature importances (gain):")
    print(importances.head(20))

    model.save_model("output/model.txt")
    print("\nModel saved to output/model.txt")


if __name__ == "__main__":
    main()
