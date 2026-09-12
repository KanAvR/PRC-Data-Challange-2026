import numpy as np
import pandas as pd

WINDOWS_MIN = [15, 30, 60]

TIME_COLS = [
    "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt",
    "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "ARVT_1_flt", "ARVT_3_flt",
    "BLOCK_TIME_UTC_mvt",
]

CAT_COLS = [
    "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt",
    "AIRCRAFT_TYPE_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt", "AIRCRAFT_OPERATOR_flt",
    "ADEP_STAND", "ADEP_RUNWAY", "ADEP_AIRCRAFT_TYPE", "ADEP_STAND_RUNWAY",
]


def _rolling_counts(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=g.index)
    t = g["MVT_TIME_UTC_mvt"].values.astype("datetime64[s]")
    is_dep = (g["PHASE_mvt"] == "DEP").values
    is_arr = ~is_dep
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]
    dep_cum = np.cumsum(is_dep[order])
    arr_cum = np.cumsum(is_arr[order])

    for w in WINDOWS_MIN:
        delta = np.timedelta64(w, "m")
        lo_idx = np.searchsorted(t_sorted, t_sorted - delta, side="left")
        hi_idx = np.searchsorted(t_sorted, t_sorted, side="left")

        dep_before = np.where(lo_idx > 0, dep_cum[np.clip(lo_idx - 1, 0, None)], 0)
        dep_upto = np.where(hi_idx > 0, dep_cum[np.clip(hi_idx - 1, 0, None)], 0)
        arr_before = np.where(lo_idx > 0, arr_cum[np.clip(lo_idx - 1, 0, None)], 0)
        arr_upto = np.where(hi_idx > 0, arr_cum[np.clip(hi_idx - 1, 0, None)], 0)

        dep_count = np.empty(len(g), dtype=np.int64)
        arr_count = np.empty(len(g), dtype=np.int64)
        dep_count[order] = dep_upto - dep_before
        arr_count[order] = arr_upto - arr_before

        out[f"dep_count_{w}m"] = dep_count
        out[f"arr_count_{w}m"] = arr_count

    return out


def _rolling_queue(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=g.index)
    t = g["AOBT_3_flt"].values.astype("datetime64[s]")
    valid = ~pd.isna(g["AOBT_3_flt"]).values
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]
    valid_sorted = valid[order]
    valid_cum = np.cumsum(valid_sorted)

    for w in [15, 30]:
        delta = np.timedelta64(w, "m")
        lo_idx = np.searchsorted(t_sorted, t_sorted - delta, side="left")
        hi_idx = np.searchsorted(t_sorted, t_sorted, side="left")
        before = np.where(lo_idx > 0, valid_cum[np.clip(lo_idx - 1, 0, None)], 0)
        upto = np.where(hi_idx > 0, valid_cum[np.clip(hi_idx - 1, 0, None)], 0)
        count_sorted = upto - before
        count = np.empty(len(g), dtype=np.int64)
        count[order] = count_sorted
        count = np.where(valid, count, -1)
        out[f"pushback_queue_{w}m"] = count

    return out


def _arr_context(g: pd.DataFrame) -> pd.DataFrame:
    """Ground conditions from ARR rows only. Their TAXITIME/BLOCK_TIME survive in
    ranking.parquet (only DEP rows are stripped), so this is safe at predict time."""
    out = pd.DataFrame(index=g.index)
    t = g["MVT_TIME_UTC_mvt"].values.astype("datetime64[s]")
    is_arr = (g["PHASE_mvt"] != "DEP").values

    arr_t = t[is_arr]
    arr_taxi = pd.to_numeric(g["TAXITIME_SEC_mvt"], errors="coerce").values[is_arr]
    order = np.argsort(arr_t, kind="mergesort")
    arr_t, arr_taxi = arr_t[order], arr_taxi[order]
    ok = ~np.isnan(arr_taxi)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, arr_taxi, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok)])

    for w in [30, 60]:
        hi = np.searchsorted(arr_t, t, side="left")
        lo = np.searchsorted(arr_t, t - np.timedelta64(w, "m"), side="left")
        n = ccnt[hi] - ccnt[lo]
        out[f"arr_taxi_mean_{w}m"] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
        out[f"arr_landed_{w}m"] = n
    return out


def _secs_since_prev_arr_block(df: pd.DataFrame) -> pd.Series:
    """Seconds from the last arrival that parked on this stand to this takeoff.
    Bounds how long the aircraft could have been off-block."""
    arr = df.loc[df["PHASE_mvt"] != "DEP", ["apt", "STAND_mvt", "BLOCK_TIME_UTC_mvt"]].dropna()
    if arr.empty:
        return pd.Series(np.nan, index=df.index)
    arr = pd.DataFrame({
        "key": arr["apt"].astype(str) + "|" + arr["STAND_mvt"].astype(str),
        "blk": arr["BLOCK_TIME_UTC_mvt"].values,
    }).sort_values("blk")
    left = pd.DataFrame({
        "idx": np.arange(len(df)),
        "key": df["apt"].astype(str).values + "|" + df["STAND_mvt"].astype(str).values,
        "t": df["MVT_TIME_UTC_mvt"].values,
    }).sort_values("t")
    merged = pd.merge_asof(left, arr, left_on="t", right_on="blk", by="key", direction="backward")
    secs = (merged["t"] - merged["blk"]).dt.total_seconds()
    # ranking.parquet is one file spanning a 5-month gap while training is per-month,
    # so an unbounded lookback would have a different upper tail at serve time
    secs = secs.where(secs <= 86400, np.nan)
    return pd.Series(secs.values, index=merged["idx"].values).sort_index().set_axis(df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in TIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)

    df["AIRCRAFT_TYPE_mvt"] = df["AIRCRAFT_TYPE_mvt"].fillna(df["AIRCRAFT_TYPE_flt"])

    feats = pd.DataFrame(index=df.index)

    mvt_t = df["MVT_TIME_UTC_mvt"]
    for c in ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]:
        feats[f"gap_{c}"] = (mvt_t - df[c]).dt.total_seconds()
        # missingness is signal, not absence of it: AOBT_3 is absent on 64.8% of
        # long-taxi rows vs 2.0% of normal ones
        feats[f"missing_{c}"] = df[c].isna().astype(int)

    for a, b in [("EOBT_1_flt", "AOBT_3_flt"), ("IOBT_flt", "AOBT_3_flt"),
                 ("LOBT_flt", "EOBT_1_flt"), ("SCHED_TIME_UTC_mvt", "EOBT_1_flt")]:
        feats[f"delta_{a}_{b}"] = (df[a] - df[b]).dt.total_seconds()

    feats["n_missing_times"] = sum(
        df[c].isna().astype(int)
        for c in ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]
    )

    feats["hour"] = mvt_t.dt.hour + mvt_t.dt.minute / 60.0
    feats["dow"] = mvt_t.dt.dayofweek
    feats["month"] = mvt_t.dt.month
    feats["is_weekend"] = (feats["dow"] >= 5).astype(int)

    df["ADEP_STAND"] = df["ADEP_mvt"].astype(str) + "_" + df["STAND_mvt"].astype(str)
    df["ADEP_RUNWAY"] = df["ADEP_mvt"].astype(str) + "_" + df["RUNWAY_mvt"].astype(str)
    df["ADEP_AIRCRAFT_TYPE"] = df["ADEP_mvt"].astype(str) + "_" + df["AIRCRAFT_TYPE_mvt"].astype(str)
    # taxi time is mostly stand-to-runway distance, which neither part captures alone
    df["ADEP_STAND_RUNWAY"] = df["ADEP_STAND"] + "_" + df["RUNWAY_mvt"].astype(str)

    for c in CAT_COLS:
        feats[c] = df[c].fillna("missing").astype(str).astype("category")

    # for an ARR row the movement airport is the destination, not ADEP
    df["apt"] = np.where(df["PHASE_mvt"].values == "DEP", df["ADEP_mvt"], df["ADES_mvt"])

    count_parts = [_rolling_counts(g) for _, g in df.groupby("apt", observed=True)]
    counts = pd.concat(count_parts).reindex(df.index)

    arr_parts = [_arr_context(g) for _, g in df.groupby("apt", observed=True)]
    arr_ctx = pd.concat(arr_parts).reindex(df.index)

    feats["secs_since_prev_arr_block"] = _secs_since_prev_arr_block(df)

    dep_only = df[df["PHASE_mvt"] == "DEP"]
    queue_parts = [_rolling_queue(g) for _, g in dep_only.groupby("apt", observed=True)]
    queue = pd.concat(queue_parts).reindex(df.index)
    queue = queue.fillna(-1)

    feats = pd.concat([feats, counts, arr_ctx, queue], axis=1)
    return feats
