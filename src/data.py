"""Loading shared by validate.py and predict.py, so the two cannot drift apart."""
import glob
import pathlib
import re
from contextlib import nullcontext

import pandas as pd

from airline import flight_key
from featcache import build_features_cached
from features import CAT_COLS

TRAIN_FILES = sorted(glob.glob("dataset/training_*.parquet"))

MIN_TAXI_SEC = 0  # negative taxi times are data errors, not outliers to tune away


def flight_ids(df):
    """Per-airport flight identity, the key for the te_flight encoding."""
    return (df["ADEP_mvt"].astype(str) + "_" + flight_key(df["FLIGHT_mvt"])).values


def mvt_seconds(df):
    """Take-off time as seconds since the epoch."""
    t = pd.to_datetime(df["MVT_TIME_UTC_mvt"], utc=True)
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().values


def month_of(path):
    return re.search(r"training_(\d{4}-\d{2})", path).group(1)


def load_training(adsb=False, timings=None, climb=False, climb_nbr_clean=False, inbound_air=False,
                  service=False, arr_habit=False, mu_arr_habit=False):
    """Features, target and per-row metadata for every training DEP row.

    adsb=True appends each departure's own-trace columns (adsb.ADSB_COLS);
    climb=True also matches traces by the climb-out airframe (--adsb-climb-id).

    Metadata (month, date) is returned separately so it can never end up as a
    model input: train_mu() trains on every column of X.
    """
    measure = timings.stage if timings is not None else lambda *args, **kwargs: nullcontext()
    xs, ys, metas = [], [], []
    for f in TRAIN_FILES:
        month = month_of(f)
        prefix = f"training.{month}"
        with measure(f"{prefix}.parquet_read", path=f):
            df = pd.read_parquet(f)
        with measure(f"{prefix}.select_departures"):
            dep = df["PHASE_mvt"].values == "DEP"
        feats = build_features_cached(f, df, timings=timings, stage=f"{prefix}.features")
        if adsb:
            with measure(f"{prefix}.adsb_features"):
                import adsb as _adsb
                af = _adsb.dep_features(df, pathlib.Path(f).stem, climb=climb)
                if climb and climb_nbr_clean:
                    from climbnbr import clean_neighbours
                    af = clean_neighbours(df, af)
                feats = feats.join(af)
        if mu_arr_habit:
            # --mu-arr-habit: a model input for every family (service.ARR_COLS)
            import service as _service
            feats = feats.join(_service.arrival_habit(df))
        with measure(f"{prefix}.extract"):
            xs.append(feats.loc[dep])
            ys.append(df.loc[dep, "TAXITIME_SEC_mvt"].astype(float))
            metas.append(pd.DataFrame({
                "MVT_ID_mvt": df.loc[dep, "MVT_ID_mvt"],
                "month": month,
                "date": df.loc[dep, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d"),
                "ADEP_mvt": df.loc[dep, "ADEP_mvt"],
                "flight": flight_ids(df.loc[dep]),
                # absolute take-off time, for features computed after the target
                # encodings (resproxy.py); metadata never reaches a model
                "t": mvt_seconds(df.loc[dep]),
            }))
            if inbound_air:
                # tail-classifier input only (--clf-inbound-air), kept out of X
                import inbound as _inbound
                metas[-1][_inbound.COL] = _inbound.air_arrival(df, pathlib.Path(f).stem).loc[dep].values
            if service:
                # tail-classifier input only (--clf-service), kept out of X
                import service as _service
                sf = _service.service_features(df).loc[dep]
                for c in _service.COLS:
                    metas[-1][c] = sf[c].values
            if arr_habit:
                import service as _service
                ah = _service.arrival_habit(df).loc[dep]
                for c in _service.ARR_COLS:
                    metas[-1][c] = ah[c].values
            if service or arr_habit:
                import service as _service
                metas[-1][_service.GATE_COL] = _service.cross_midnight(df).loc[dep].values
    with measure("training.concat_and_filter"):
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
