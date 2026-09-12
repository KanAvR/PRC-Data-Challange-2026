import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error

sys.path.insert(0, "src")
from features import CAT_COLS, build_features
from model import (TAIL_THR, analytic_tail_value, combine, predict_tail_proba,
                   train_mu, train_tail_clf)

TRAIN_FILES = sorted(glob.glob("dataset/training_*.parquet"))
HOLDOUT_MONTHS = "2025-01-01|2025-07-01"  # mirrors the Jan/Jul 2026 ranking set

MIN_TAXI_SEC = 0  # negative taxi times are data errors, not outliers to tune away


def load_training():
    frames = []
    for f in TRAIN_FILES:
        df = pd.read_parquet(f)
        feats = build_features(df)
        feats["target"] = df["TAXITIME_SEC_mvt"].astype(float)
        feats["month_file"] = f
        frames.append(feats.loc[df["PHASE_mvt"].values == "DEP"])
    full = pd.concat(frames, ignore_index=True)
    return full[full["target"] >= MIN_TAXI_SEC]


def main():
    print("Loading training data and building features...")
    data = load_training()

    holdout = data["month_file"].str.contains(HOLDOUT_MONTHS)
    train_df = data.loc[~holdout].drop(columns=["month_file"]).copy()
    valid_df = data.loc[holdout].drop(columns=["month_file"]).copy()
    y_train = train_df.pop("target").values
    y_valid = valid_df.pop("target").values
    print(f"train={len(train_df)} valid={len(valid_df)} "
          f"tail={int((y_train > TAIL_THR).sum())}/{int((y_valid > TAIL_THR).sum())}")

    for c in CAT_COLS:
        cats = sorted(set(train_df[c].astype(str).unique()) | set(valid_df[c].astype(str).unique()))
        dtype = pd.CategoricalDtype(categories=cats)
        train_df[c] = train_df[c].astype(str).astype(dtype)
        valid_df[c] = valid_df[c].astype(str).astype(dtype)

    print("Training normal-regime regressor...")
    mu = train_mu(train_df, y_train)
    mu_valid = np.clip(mu.predict(valid_df), 0, None)

    print("Training tail classifier...")
    clf = train_tail_clf(train_df, y_train)
    p_valid = predict_tail_proba(clf, valid_df)
    print(f"  tail AUC = {roc_auc_score((y_valid > TAIL_THR).astype(int), p_valid):.4f}")

    pred = combine(mu_valid, p_valid, analytic_tail_value(valid_df))

    normal = y_valid <= TAIL_THR
    print("\nHoldout (Jan+Jul), scored over ALL rows as the leaderboard does:")
    print(f"  mu alone : {root_mean_squared_error(y_valid, mu_valid):8.2f}")
    print(f"  mixture  : {root_mean_squared_error(y_valid, pred):8.2f}")
    print(f"  mixture, normal rows only : {root_mean_squared_error(y_valid[normal], pred[normal]):8.2f}")

    mu.save_model("output/model.txt")
    print("\nModel saved to output/model.txt")


if __name__ == "__main__":
    main()
