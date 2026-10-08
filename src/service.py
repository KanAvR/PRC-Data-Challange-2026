"""--clf-service: what kind of operation a no-flight-plan departure is.

At LIRF the no-LOBT tail candidates (delayed by hours, no aircraft type, no
FLIGHT_ID) carry either the scheduled time as block time (a copy: the airport
record kept the schedule) or a real off-block. Which one follows how the flight
is run (NEXT_STEPS.md, B36): copies are rare on the home carrier's and other
daily scheduled services (ITY 5%, a flight number that departs again within 24h
41%) and near-certain on ad-hoc and charter operations (77% without a repeat;
Israir, LAV, TKJ, KMM 100%), more so on domestic legs (90% vs 55%) and on
corrupted callsigns (86%). The classifier had the operator but not the
destination, the regularity or the callsign shape.

All columns come from provided movement columns of the same file and are built
the same way on ranking.parquet; they reach the no-LOBT tail classifier only.
"""
import numpy as np
import pandas as pd

# --clf-arr-habit: the same recording habit seen on arrivals. Arrival block times
# are provided on ranking rows too; the share of an operator's arrivals at the
# airport logged with block == scheduled time (within 60s) tracks the copy rate
# of its no-LOBT departure candidates (2025 LIRF: share > 0.4 -> 100% copies,
# 0.05-0.1 -> 34%), and so does the day's share. Shrunk toward the airport's share.
# (no arrival count: a training file holds one month, ranking.parquet two, so a
# count would not mean the same thing on both sides)
ARR_COLS = ["svc_op_arr_copy", "svc_day_arr_copy"]
ARR_PRIOR_N = 20
COPY_S = 60

COLS = ["svc_domestic", "svc_ades_country", "svc_n_same_flight_24h", "svc_cs_long_prefix",
        "svc_cs_alpha_suffix"]
CAT = ["svc_ades_country"]
WINDOW_S = 86400


def service_features(raw):
    """COLS for every row of a raw frame (training month or ranking); NaN on ARR rows."""
    out = pd.DataFrame(index=raw.index, columns=COLS, dtype=float)
    dep = (raw["PHASE_mvt"] == "DEP").values
    d = raw.loc[dep]
    adep = d["ADEP_mvt"].astype(str).str[:2]
    ades = pd.Series([s[:2] if isinstance(s, str) else "missing" for s in d["ADES_mvt"]], index=d.index)
    out.loc[dep, "svc_domestic"] = (ades.values == adep.values).astype(float)
    fl = pd.Series([s if isinstance(s, str) else "" for s in d["FLIGHT_mvt"]], index=d.index)
    out.loc[dep, "svc_cs_long_prefix"] = fl.str.match(r"^[A-Z]{4,}").astype(float).values
    out.loc[dep, "svc_cs_alpha_suffix"] = fl.str.contains(r"\d[A-Z]+$").astype(float).values
    # other departures of the same flight number from the same airport within 24h
    t = (d["MVT_TIME_UTC_mvt"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().values
    key = (d["ADEP_mvt"].astype(str) + "|" + fl).values
    n = np.zeros(len(d))
    order = np.lexsort((t, key))
    ks, ts = key[order], t[order]
    start = 0
    for i in range(1, len(ks) + 1):
        if i == len(ks) or ks[i] != ks[start]:
            g = ts[start:i]
            lo = np.searchsorted(g, g - WINDOW_S, side="left")
            hi = np.searchsorted(g, g + WINDOW_S, side="right")
            n[order[start:i]] = hi - lo - 1
            start = i
    n[fl.values == ""] = np.nan
    out.loc[dep, "svc_n_same_flight_24h"] = n
    out["svc_ades_country"] = pd.Series(np.where(dep, None, None), index=raw.index, dtype=object)
    out.loc[dep, "svc_ades_country"] = ades.values
    return out


# --svc-midnight-gate: day-late labels (86,400 s + a normal taxi: the record kept the
# scheduled local date with the actual time of day) need the flight to leave on a
# later local date than scheduled. 2025 LIRF candidates: 12 of 12 day-late rows
# cross local midnight, 0 of 760 non-crossing candidates are day-late. The
# operation-type inputs separate copies from normal off-blocks but read day-late
# rows (regular carriers) as normal, so crossing rows keep the shipped classifier.
GATE_COL = "svc_cross_midnight"


def cross_midnight(raw):
    """1.0 where a departure's local take-off date is after its local scheduled date."""
    from features import AIRPORT_TZ
    out = pd.Series(np.nan, index=raw.index, name=GATE_COL)
    dep = (raw["PHASE_mvt"] == "DEP").values
    d = raw.loc[dep]
    res = np.zeros(len(d))
    apt = d["ADEP_mvt"].astype(str).values
    for code, tz in AIRPORT_TZ.items():
        sel = apt == code
        if sel.any():
            s = d["SCHED_TIME_UTC_mvt"].iloc[sel].dt.tz_convert(tz).dt.normalize()
            m = d["MVT_TIME_UTC_mvt"].iloc[sel].dt.tz_convert(tz).dt.normalize()
            res[sel] = (m.dt.tz_localize(None) > s.dt.tz_localize(None)).astype(float).values
    out.loc[dep] = res
    return out


def _operator(flight):
    return pd.Series([s[:3] if isinstance(s, str) and len(s) >= 3 else "" for s in flight])


def arrival_habit(raw):
    """ARR_COLS for every row of a raw frame, from the frame's own arrivals; NaN on ARR rows."""
    out = pd.DataFrame(np.nan, index=raw.index, columns=ARR_COLS)
    arr = raw[raw["PHASE_mvt"] == "ARR"]
    a = pd.DataFrame({
        "apt": arr["ADES_mvt"].astype(str).values,
        "op": _operator(arr["FLIGHT_mvt"]).values,
        "day": arr["BLOCK_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").values,
        "eq": ((arr["BLOCK_TIME_UTC_mvt"] - arr["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs()
               <= COPY_S).astype(float).values,
    })
    apt_rate = a.groupby("apt")["eq"].mean()
    op = a.groupby(["apt", "op"])["eq"].agg(["sum", "size"])
    day = a.groupby(["apt", "day"])["eq"].mean()
    dep = (raw["PHASE_mvt"] == "DEP").values
    d = raw.loc[dep]
    apt = d["ADEP_mvt"].astype(str).values
    k = pd.MultiIndex.from_arrays([apt, _operator(d["FLIGHT_mvt"]).values])
    s = op["sum"].reindex(k).fillna(0).values
    n = op["size"].reindex(k).fillna(0).values
    prior = apt_rate.reindex(apt).fillna(a["eq"].mean() if len(a) else 0.0).values
    out.loc[dep, "svc_op_arr_copy"] = (s + ARR_PRIOR_N * prior) / (n + ARR_PRIOR_N)
    kd = pd.MultiIndex.from_arrays([apt, d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").values])
    out.loc[dep, "svc_day_arr_copy"] = day.reindex(kd).values
    return out
