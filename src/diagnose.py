import glob
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, "src")
from features import CAT_COLS, build_features

TRAIN_FILES = sorted(glob.glob("dataset/training_*.parquet"))

MAX_TAXI_SEC = 100000
MIN_TAXI_SEC = 0


def main():
    frames = []
    for f in TRAIN_FILES:
        df = pd.read_parquet(f)
        feats = build_features(df)
        feats["target"] = df["TAXITIME_SEC_mvt"].astype(float)
        feats["month_file"] = f
        frames.append(feats.loc[df["PHASE_mvt"].values == "DEP"])
    full = pd.concat(frames, ignore_index=True)

    holdout_months = full["month_file"].str.contains("2025-01-01|2025-07-01")

    train_filtered = full.loc[
        (~holdout_months) & (full["target"] >= MIN_TAXI_SEC) & (full["target"] <= MAX_TAXI_SEC)
    ].drop(columns=["month_file"]).copy()

    valid_all = full.loc[holdout_months].drop(columns=["month_file"]).copy()
    valid_filtered = valid_all[(valid_all["target"] >= MIN_TAXI_SEC) & (valid_all["target"] <= MAX_TAXI_SEC)].copy()

    y_train = train_filtered.pop("target")
    y_valid_all = valid_all.pop("target")
    y_valid_filtered = valid_filtered.pop("target")

    for c in CAT_COLS:
        cats = sorted(
            set(train_filtered[c].astype(str).unique())
            | set(valid_all[c].astype(str).unique())
        )
        dtype = pd.CategoricalDtype(categories=cats)
        train_filtered[c] = train_filtered[c].astype(str).astype(dtype)
        valid_all[c] = valid_all[c].astype(str).astype(dtype)
        valid_filtered[c] = valid_filtered[c].astype(str).astype(dtype)

    train_set = lgb.Dataset(train_filtered, label=y_train, categorical_feature=CAT_COLS)

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=255,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        verbose=-1,
    )
    model = lgb.train(params, train_set, num_boost_round=800)

    pred_filtered = np.clip(model.predict(valid_filtered), 0, None)
    pred_all = np.clip(model.predict(valid_all), 0, None)

    print("N valid_all:", len(valid_all), " N valid_filtered:", len(valid_filtered))
    print("RMSE on filtered holdout (matches train.py):", root_mean_squared_error(y_valid_filtered, pred_filtered))
    print("RMSE on UNFILTERED holdout (all targets incl. outliers):", root_mean_squared_error(y_valid_all, pred_all))

    err = (pred_all - y_valid_all.values)
    sq = err ** 2
    order = np.argsort(-sq)
    top = order[:20]
    print("\nWorst 20 errors in unfiltered holdout:")
    print(pd.DataFrame({
        "true": y_valid_all.values[top],
        "pred": pred_all[top],
        "abs_err": np.abs(err[top]),
    }))

    print("\nContribution to MSE from target outside [0,6000]:")
    mask_out = (y_valid_all.values < 0) | (y_valid_all.values > 6000)
    print("count outside range:", mask_out.sum(), "/", len(y_valid_all))
    mse_total = np.mean(sq)
    mse_out = np.sum(sq[mask_out]) / len(sq)
    print("total MSE:", mse_total, " MSE contribution from out-of-range targets:", mse_out,
          f"({100*mse_out/mse_total:.1f}% of total)")


if __name__ == "__main__":
    main()
