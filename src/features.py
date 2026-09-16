import numpy as np
import pandas as pd

from airline import airline_code

WINDOWS_MIN = [15, 30, 60]

AIRPORT_TZ = {
    "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}

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
    "AIRLINE_mvt", "ADEP_AIRLINE",
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


def _bounds(ev, query_t, w, after):
    """Index range of sorted events in [t-w, t), or in (t, t+w] when after=True."""
    if after:
        return (np.searchsorted(ev, query_t, side="right"),
                np.searchsorted(ev, query_t + np.timedelta64(w, "m"), side="right"))
    return (np.searchsorted(ev, query_t - np.timedelta64(w, "m"), side="left"),
            np.searchsorted(ev, query_t, side="left"))


def _window_counts(event_t, query_t, windows, after=False):
    """For each query time, how many events fall in the window (self excluded)."""
    ev = np.sort(event_t)
    out = {}
    for w in windows:
        lo, hi = _bounds(ev, query_t, w, after)
        out[w] = hi - lo
    return out


def _window_mean(event_t, event_val, query_t, windows, after=False):
    """Mean of event_val over events in the window. NaN when the window is empty."""
    order = np.argsort(event_t, kind="mergesort")
    ev, val = event_t[order], event_val[order]
    ok = ~np.isnan(val)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, val, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok)])
    out = {}
    for w in windows:
        lo, hi = _bounds(ev, query_t, w, after)
        n = ccnt[hi] - ccnt[lo]
        out[w] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
    return out


def _pending_queue(sched_t, mvt_t, query_t, self_sched):
    """Flights whose scheduled push-back has passed but which have not taken off.

    Both timestamps exist for every ranking DEP row, so this is a queue estimate
    that survives to predict time -- actual push-back (BLOCK_TIME) does not.
    """
    s = np.sort(sched_t)
    m = np.sort(mvt_t)
    started = np.searchsorted(s, query_t, side="right")
    gone = np.searchsorted(m, query_t, side="left")
    q = started - gone
    return q - (self_sched <= query_t).astype(np.int64)  # drop self


def _dep_dynamics(g: pd.DataFrame) -> pd.DataFrame:
    """Departure-side congestion at one airport, split airport-wide vs runway.

    Everything keys off takeoff time and scheduled time only. Neighbouring
    flights' gap_SCHED is the disruption signal: when an airport is in trouble,
    every departure around it takes off long after schedule, and that is visible
    without knowing anybody's taxi time.
    """
    out = pd.DataFrame(index=g.index)
    t = g["MVT_TIME_UTC_mvt"].values.astype("datetime64[s]")
    sched = g["SCHED_TIME_UTC_mvt"].values.astype("datetime64[s]")
    gap_sched = (t - sched) / np.timedelta64(1, "s")

    for w, c in _window_counts(t, t, [5, 15, 30, 60, 120]).items():
        out[f"dep_off_apt_{w}m"] = c
    for w, v in _window_mean(t, gap_sched, t, [30, 60, 120]).items():
        out[f"nbr_gapsched_apt_{w}m"] = v
    out["queue_ahead_apt"] = _pending_queue(sched, t, t, sched)
    # Scoring is retrospective and ranking.parquet holds every takeoff, so the
    # traffic that followed this one is known too: a queue behind the aircraft
    # and delays that persist after it are part of the same episode.
    for w, c in _window_counts(t, t, [15, 30], after=True).items():
        out[f"dep_off_apt_next_{w}m"] = c
    for w, v in _window_mean(t, gap_sched, t, [60], after=True).items():
        out[f"nbr_gapsched_apt_next_{w}m"] = v

    ev = np.sort(t)
    prev_idx = np.searchsorted(ev, t, side="left") - 1
    out["secs_since_prev_dep_apt"] = np.where(
        prev_idx >= 0, (t - ev[np.clip(prev_idx, 0, None)]) / np.timedelta64(1, "s"), np.nan)

    for col in ["dep_off_rwy_5m", "dep_off_rwy_15m", "dep_off_rwy_30m", "dep_off_rwy_60m",
                "queue_ahead_rwy", "secs_since_prev_dep_rwy", "nbr_gapsched_rwy_60m",
                "queue_rate_rwy", "dep_off_rwy_next_15m", "secs_to_next_dep_rwy"]:
        out[col] = np.nan

    rwy = g["RUNWAY_mvt"].fillna("missing").astype(str).values
    for r in pd.unique(rwy):
        sel = rwy == r
        tr, sr, gr = t[sel], sched[sel], gap_sched[sel]
        for w, c in _window_counts(tr, tr, [5, 15, 30, 60]).items():
            out.loc[sel, f"dep_off_rwy_{w}m"] = c
        q = _pending_queue(sr, tr, tr, sr)
        out.loc[sel, "queue_ahead_rwy"] = q
        # queue 30 min earlier -> is congestion building or clearing
        q30 = _pending_queue(sr, tr, tr - np.timedelta64(30, "m"), sr)
        out.loc[sel, "queue_rate_rwy"] = q - q30
        for w, v in _window_mean(tr, gr, tr, [60]).items():
            out.loc[sel, "nbr_gapsched_rwy_60m"] = v
        evr = np.sort(tr)
        pi = np.searchsorted(evr, tr, side="left") - 1
        out.loc[sel, "secs_since_prev_dep_rwy"] = np.where(
            pi >= 0, (tr - evr[np.clip(pi, 0, None)]) / np.timedelta64(1, "s"), np.nan)
        out.loc[sel, "dep_off_rwy_next_15m"] = _window_counts(tr, tr, [15], after=True)[15]
        ni = np.searchsorted(evr, tr, side="right")
        out.loc[sel, "secs_to_next_dep_rwy"] = np.where(
            ni < len(evr), (evr[np.clip(ni, None, len(evr) - 1)] - tr) / np.timedelta64(1, "s"), np.nan)

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
    for w, v in _window_mean(arr_t, arr_taxi, t, [30], after=True).items():
        out[f"arr_taxi_mean_next_{w}m"] = v
    for w, c in _window_counts(arr_t, t, [30], after=True).items():
        out[f"arr_landed_next_{w}m"] = c
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
    # schedules and staffing follow local clocks, and the UTC offset shifts by an
    # hour between the Jan and Jul ranking months everywhere except LTFM
    apt = np.where(df["PHASE_mvt"].values == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    feats["hour_local"] = np.nan
    for code, tz in AIRPORT_TZ.items():
        sel = apt == code
        if sel.any():
            lt = mvt_t[sel].dt.tz_convert(tz)
            feats.loc[sel, "hour_local"] = (lt.dt.hour + lt.dt.minute / 60.0).values
    feats["dow"] = mvt_t.dt.dayofweek
    feats["month"] = mvt_t.dt.month
    feats["is_weekend"] = (feats["dow"] >= 5).astype(int)

    # FLIGHT_mvt survives on the rows where every _flt column is absent, so its
    # operator prefix is the only operator signal available on exactly the rows
    # the tail model has to judge (AIRCRAFT_OPERATOR_flt is "missing" there).
    # IATA-form codes are mapped to ICAO so LH400 and DLH400 are one operator.
    df["AIRLINE_mvt"], _ = airline_code(df["FLIGHT_mvt"])
    df["ADEP_AIRLINE"] = df["ADEP_mvt"].astype(str) + "_" + df["AIRLINE_mvt"]

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

    dyn_parts = [_dep_dynamics(g) for _, g in dep_only.groupby("apt", observed=True)]
    dyn = pd.concat(dyn_parts).reindex(df.index)

    feats = pd.concat([feats, counts, arr_ctx, queue, dyn], axis=1)
    return feats
