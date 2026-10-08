"""--clf-inbound-air: when the departing airframe itself arrived, from its ADS-B trace.

Second measurement of the B17 mechanism (late inbound -> LIRF records the
scheduled time as block time). --clf-inbound reads the stand's previous arrival
from the movement rows; this reads the airframe: its climb-out (or its ground
points) carrying the row's callsign names the icao, and that icao's last
airborne point near the airport more than 20 min before take-off is its
arrival. The value is that arrival minus the scheduled departure. Computed for
no-LOBT departures only (the rows the no-LOBT tail classifier sees), the same
way on training months and ranking.parquet. It travels in `meta`, so it reaches
the tail classifier and never a mu model.
"""
import hashlib
import pathlib

import numpy as np
import pandas as pd

import adsb

COL = "inbound_air_after_sched"
CLIMB_WINDOW = (-60, 300)          # airborne points with the callsign, s around MVT_TIME
GROUND_WINDOW = (-4 * 3600, 120)   # fallback: ground points with the callsign
ARR_WINDOW = (-14 * 3600, -1200)   # the airframe's last airborne point here, before take-off


def _cache_key(days, raw):
    src = pathlib.Path(__file__).read_bytes()
    for d in days:
        p = adsb.ADSB_DIR / f"{d}.parquet"
        src += f"{d}|{p.stat().st_size if p.exists() else -1}".encode()
    src += adsb.raw_fingerprint(raw)  # same rows in the same order, or rebuild
    return hashlib.sha256(src.replace(b"\r\n", b"\n")).hexdigest()[:16]


def _day(rows, pts):
    """COL for one day's no-LOBT departures (rows: index, ADEP_mvt, mvt_ts, sched_ts, keys)."""
    pts = pts.assign(cs=pts["callsign"].str.upper().str.strip())
    ident = []
    for key in ["CALLSIGN_flt", "FLIGHT_mvt"]:
        k = rows[["index", "ADEP_mvt", "mvt_ts", key]].dropna(subset=[key])
        k = k.assign(cs=k[key].astype(str).str.upper().str.strip())
        ident.append(k.merge(pts[["icao", "cs", "airport", "ts", "alt"]], left_on=["ADEP_mvt", "cs"],
                             right_on=["airport", "cs"]))
    m = pd.concat(ident, ignore_index=True)
    if m.empty:
        return pd.Series(dtype=float)
    dt = m["ts"] - m["mvt_ts"]
    air = (m["alt"] != -1) & (dt >= CLIMB_WINDOW[0]) & (dt <= CLIMB_WINDOW[1])
    gnd = (m["alt"] == -1) & (dt >= GROUND_WINDOW[0]) & (dt <= GROUND_WINDOW[1])
    m = m[air | gnd].assign(pref=np.where(air[air | gnd], 0, 1))
    if m.empty:
        return pd.Series(dtype=float)
    m = m.sort_values(["pref", "ts"]).drop_duplicates("index")[["index", "icao", "ADEP_mvt", "mvt_ts"]]
    a = pts.loc[pts["alt"] != -1, ["icao", "airport", "ts"]]
    j = m.merge(a, left_on=["icao", "ADEP_mvt"], right_on=["icao", "airport"])
    dt = j["ts"] - j["mvt_ts"]
    j = j[(dt >= ARR_WINDOW[0]) & (dt <= ARR_WINDOW[1])]
    arr = j.groupby("index")["ts"].max()
    sched = rows.set_index("index")["sched_ts"]
    return (arr - sched.reindex(arr.index)).rename(COL)


def air_arrival(raw, tag):
    """COL for every row of raw (NaN where not a no-LOBT departure or not seen)."""
    dep = raw[(raw["PHASE_mvt"] == "DEP") & raw["LOBT_flt"].isna()][
        ["ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]].copy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    t = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True)
    dep["mvt_ts"] = (t - epoch).dt.total_seconds()
    dep["sched_ts"] = (pd.to_datetime(dep["SCHED_TIME_UTC_mvt"], utc=True) - epoch).dt.total_seconds()
    dep["day"] = t.dt.strftime("%Y-%m-%d")
    days = sorted(dep["day"].unique())
    prev = {d: (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in days}
    cache = adsb.CACHE_DIR / f"{tag}.inbound.{_cache_key(sorted(set(days) | set(prev.values())), raw)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)[COL].reindex(raw.index)
    parts = []
    for d in days:
        pts = [p for p in (adsb._load_points(prev[d]), adsb._load_points(d)) if p is not None]
        if not pts:
            continue
        parts.append(_day(dep[dep["day"] == d].reset_index(), pd.concat(pts, ignore_index=True)))
    out = (pd.concat(parts) if parts else pd.Series(dtype=float, name=COL)).reindex(raw.index).astype("float32")
    adsb.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.rename(COL).to_frame().to_parquet(cache)
    return out
