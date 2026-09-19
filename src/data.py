"""Loading shared by validate.py and predict.py, so the two cannot drift apart."""
import glob
import re

import pandas as pd

from airline import flight_key
from featcache import build_features_cached
from features import CAT_COLS

TRAIN_FILES = sorted(glob.glob("dataset/training_*.parquet"))

MIN_TAXI_SEC = 0  # negative taxi times are data errors, not outliers to tune away


def flight_ids(df):
    """Per-airport flight identity, the key for the te_flight encoding."""
    return (df["ADEP_mvt"].astype(str) + "_" + flight_key(df["FLIGHT_mvt"])).values


def month_of(path):
    return re.search(r"training_(\d{4}-\d{2})", path).group(1)


def load_training():
    """Features, target and per-row metadata for every training DEP row.

    Metadata (month, date) is returned separately so it can never end up as a
    model input: train_mu() trains on every column of X.
    """
    xs, ys, metas = [], [], []
    for f in TRAIN_FILES:
        df = pd.read_parquet(f)
        dep = df["PHASE_mvt"].values == "DEP"
        xs.append(build_features_cached(f, df).loc[dep])
        ys.append(df.loc[dep, "TAXITIME_SEC_mvt"].astype(float))
        metas.append(pd.DataFrame({
            "MVT_ID_mvt": df.loc[dep, "MVT_ID_mvt"],
            "month": month_of(f),
            "date": df.loc[dep, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d"),
            "ADEP_mvt": df.loc[dep, "ADEP_mvt"],
            "flight": flight_ids(df.loc[dep]),
        }))
    X = pd.concat(xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True).values
    meta = pd.concat(metas, ignore_index=True)
    keep = y >= MIN_TAXI_SEC
    return (X.loc[keep].reset_index(drop=True), y[keep],
            meta.loc[keep].reset_index(drop=True))


def align_categoricals(*frames):
    """Give every frame the same category list, so codes mean the same thing."""
    for c in CAT_COLS:
        cats = sorted(set().union(*(f[c].astype(str).unique() for f in frames)))
        dtype = pd.CategoricalDtype(categories=cats)
        for f in frames:
            f[c] = f[c].astype(str).astype(dtype)
