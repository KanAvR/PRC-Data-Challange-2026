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
