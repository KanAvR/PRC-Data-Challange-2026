"""Each departure's own ADS-B trace, from the adsb.lol globe_history archive (ODbL).

adsb.lol publishes one readsb `globe_history` tar per UTC day as GitHub release
assets (~3.5 GB). We stream each day once and keep only low points near the ten
airports (~13 MB/day), then match every DEP row to the trace take-off of the same
aircraft (same airport, take-off within MATCH_S of MVT_TIME, callsign preferred).
Where the receivers see the ground, the trace shows the aircraft parked on its
stand (apt.dat coordinates, see aptdat.py) and the moment it leaves: a separate
observation of the off-block the label is measured from. Coverage follows the
feeder network and changes between years (NEXT_STEPS.md, lead A1), so the
features describe the trace and its quality and the model decides how far to
trust them.

    python src/adsb.py download --start 2025-01-01 --end 2025-01-31 --streams 4
"""
import argparse
import os
import gzip
import hashlib
import math
import pathlib
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

ADSB_DIR = pathlib.Path("dataset/adsb")
CACHE_DIR = ADSB_DIR / "feat"
RELEASE = ("https://github.com/adsblol/globe_history_{y}/releases/download/"
           "v{d}-planes-readsb-{tag}/v{d}-planes-readsb-{tag}.tar")
# most days are "prod-0" split into .tar.aa/.tar.ab; some are one unsplit .tar,
# and a few (2025-05-28..06-10) exist only as "prod-0tmp"
RELEASE_TAGS = ["prod-0", "prod-0tmp"]

APT = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
    "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519),
}
RADIUS_KM = 8.0
MAX_ALT_FT = 1500      # airborne points kept up to this far above the field
# field elevation (ft): barometric altitude is above sea level, and a fixed
# 1,500 ft MSL cut dropped every airborne point at LEMD (~2,000 ft) and most
# at EDDM/LSZH, so their take-offs were never seen (NEXT_STEPS.md, A2c)
ELEV_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12,
           "LEMD": 1998, "LFPG": 392, "LIRF": 15, "LSZH": 1416, "LTFM": 325}
_ALT_CEIL = max(ELEV_FT.values()) + MAX_ALT_FT
_BOX = {a: (la - 0.08, la + 0.08, lo - 0.12, lo + 0.12) for a, (la, lo) in APT.items()}
POINT_COLS = ["icao", "reg", "type", "callsign", "airport", "ts", "lat", "lon",
              "alt", "gs", "track", "flags", "src"]

MATCH_S = 180          # trace take-off vs MVT_TIME
LOOKBACK_S = 4 * 3600  # ground history searched before take-off
SEG_GAP_S = 1800       # a ground gap this long starts a new segment
STAND_R_M = 60         # "on the stand" radius around the apt.dat position
PARK_R_M = 80          # first point this close counts as parked on the stand
MOVE_M = 15            # displacement from the parked position that counts as movement

ADSB_COLS = ["adsb_match", "adsb_cs_ok", "adsb_to_dt", "adsb_first_seen", "adsb_leave0",
             "adsb_leave_stand", "adsb_push", "adsb_last_stop", "adsb_d_first_stand",
             "adsb_gs_first", "adsb_n_ground", "adsb_cadence", "adsb_max_gap", "adsb_since_land",
             "adsb_cov_apt_day", "adsb_via_cs"]
CS_AFTER_S = 120       # callsign fallback: ground points may end this long after MVT_TIME
# --adsb-climb-id: the airborne climb-out carries the departure callsign even where
# the ground points do not (none set yet, or the inbound flight's), so it names
# the airframe; that airframe's ground points then need no callsign
CLIMB_BEFORE_S, CLIMB_AFTER_S = 60, 300
CLIMB_COLS = ["adsb_via_climb"]
# --adsb-pos: where on the airport the trace's last segment starts (metres east /
# north of the reference point) and how much ground path it covers. A trace that
# starts mid-taxi (80% of matched rows without a stand exit, B9) says how far
# along the route the aircraft already was; the trees learn per location how long
# that took. Another team found removing first-point lat/lon cost 1.54 live
POS_COLS = ["adsb_x0", "adsb_y0", "adsb_path_m"]

# --adsb-nbr: the traces' view of everybody else. Neighbours' trace-measured
# taxi (stand exit, pushback, first seen) is the neighbour proxy measured
# directly; surface counts are the aircraft actually taxiing or holding
NBR_WINDOWS_MIN = [15, 30]
NBR_CLIP_S = 3600
NBR_COLS = ([f"adsb_nbr_exit_apt_prev{w}" for w in NBR_WINDOWS_MIN]
            + ["adsb_nbr_exit_apt_next30", "adsb_nbr_exit_n_apt_prev30",
               "adsb_nbr_push_apt_prev30", "adsb_nbr_first_apt_prev30",
               "adsb_nbr_exit_rwy_prev30", "adsb_nbr_exit_rwy_next30"])
SURF_WINDOWS_MIN = [10, 20]
TAXI_GS = (3.0, 50.0)  # kt: moving on the ground but not on a take-off roll
HOLD_GS = 1.0          # kt: stopped ...
HOLD_OFF_STAND_M = 80  # ... this far from any apt.dat stand: queueing, not parked
SURF_COLS = ([f"adsb_surf_taxi_{w}" for w in SURF_WINDOWS_MIN]
             + [f"adsb_surf_hold_{w}" for w in SURF_WINDOWS_MIN] + ["adsb_surf_taxi_rel_20"])
ADSB_NBR_COLS = NBR_COLS + SURF_COLS


# ---------------------------------------------------------------- extraction

def _near(lat, lon):
    for a, (la0, la1, lo0, lo1) in _BOX.items():
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            la, lo = APT[a]
            dy = (lat - la) * 111.2
            dx = (lon - lo) * 111.2 * math.cos(math.radians(la))
            if dx * dx + dy * dy <= RADIUS_KM ** 2:
                return a
    return None


def _parse_trace(b, legacy_cut=False):
    """Points near the airports. legacy_cut keeps airborne points up to MAX_ALT_FT
    above sea level, the cut every cached 2025 and Jan/Jul 2026 day was extracted
    with (before d0414cb); new days for the same models must match it."""
    import orjson
    try:
        b = gzip.decompress(b)
    except OSError:
        pass
    d = orjson.loads(b)
    t0 = d.get("timestamp", 0)
    icao, reg, typ = d.get("icao"), d.get("r"), d.get("t")
    out, cs = [], None
    for p in d.get("trace", ()):
        if len(p) > 8 and isinstance(p[8], dict) and p[8].get("flight"):
            cs = p[8]["flight"].strip()
        lat, lon, alt = p[1], p[2], p[3]
        if lat is None or lon is None:
            continue
        ground = alt == "ground"
        if not ground and (alt is None or alt > (MAX_ALT_FT if legacy_cut else _ALT_CEIL)):
            continue
        a = _near(lat, lon)
        if a is None or (not ground and not legacy_cut and alt > ELEV_FT[a] + MAX_ALT_FT):
            continue
        out.append((icao, reg, typ, cs, a, t0 + p[0], lat, lon,
                    -1 if ground else alt, p[4], p[5], p[6],
                    p[9] if len(p) > 9 else None))
    return out


def _parse_many(bs, legacy_cut=False):
    out = []
    for b in bs:
        try:
            out.extend(_parse_trace(b, legacy_cut))
        except Exception:  # a malformed trace file is skipped, not fatal
            pass
    return out


def _exists(url):
    req = urllib.request.Request(url, method="HEAD")
    for attempt in range(4):
        try:
            urllib.request.urlopen(req, timeout=60).close()
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(10 * (attempt + 1))
        except OSError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"cannot reach {url}")


def release_urls(day):
    """The asset URLs holding one day's tar, in order."""
    y, d = day[:4], day.replace("-", ".")
    for tag in RELEASE_TAGS:
        base = RELEASE.format(y=y, d=d, tag=tag)
        if _exists(base + ".aa"):
            parts = [base + ".aa"]
            for s in "bcdefghij":
                if not _exists(f"{base}.a{s}"):
                    break
                parts.append(f"{base}.a{s}")
            return parts
        if _exists(base):
            return [base]
    raise RuntimeError(f"{day}: no release")


class _Parts:
    """A day's release assets read as one stream."""

    def __init__(self, urls):
        self.urls, self.cur = iter(urls), None
        self._next()

    def _next(self):
        url = next(self.urls, None)
        if url is None:
            self.cur = None
            return False
        for attempt in range(4):
            try:
                self.cur = urllib.request.urlopen(url, timeout=300)
                return True
            except OSError:
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"cannot fetch {url}")

    def read(self, n=-1):
        buf = b""
        while self.cur is not None and (n < 0 or len(buf) < n):
            chunk = self.cur.read(n - len(buf) if n >= 0 else -1)
            if chunk:
                buf += chunk
            elif not self._next():
                break
        return buf


def extract_day(day, workers=6, out_dir=None, legacy_cut=False):
    """Stream one UTC day and write <out_dir or dataset/adsb>/<day>.parquet."""
    out_dir = pathlib.Path(out_dir) if out_dir else ADSB_DIR
    out = out_dir / f"{day}.parquet"
    if out.exists():
        return f"{day}: exists"
    t = time.time()
    tf = tarfile.open(fileobj=_Parts(release_urls(day)), mode="r|")
    rows, n, batch, futs = [], 0, [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m in tf:
            if not m.isfile() or "trace_full" not in m.name:
                continue
            batch.append(tf.extractfile(m).read())
            n += 1
            if len(batch) >= 400:
                futs.append(ex.submit(_parse_many, batch, legacy_cut))
                batch = []
                while len(futs) > 24:
                    rows.extend(futs.pop(0).result())
        if batch:
            futs.append(ex.submit(_parse_many, batch, legacy_cut))
        for f in futs:
            rows.extend(f.result())
    if n == 0:
        raise RuntimeError(f"{day}: no traces in the release")
    df = pd.DataFrame(rows, columns=POINT_COLS)
    for c in ["gs", "track"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["alt"] = pd.to_numeric(df["alt"], errors="coerce").astype("float32")
    df["src"] = df["src"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")  # two streams may meet on one day
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"{day}: traces {n} points {len(df)} {time.time() - t:.0f}s"


def download(start, end, streams=4, reverse=False, out_dir=None, legacy_cut=False):
    out_dir = pathlib.Path(out_dir) if out_dir else ADSB_DIR
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]
    days = days[::-1] if reverse else days
    days = [d for d in days if not (out_dir / f"{d}.parquet").exists()]
    print(f"{len(days)} days to fetch with {streams} streams into {out_dir}"
          f"{' (legacy altitude cut)' if legacy_cut else ''}", flush=True)
    with ThreadPoolExecutor(max_workers=streams) as ex:
        for msg in ex.map(lambda d: _safe_extract(d, out_dir, legacy_cut), days):
            print(msg, flush=True)


def _safe_extract(day, out_dir=None, legacy_cut=False):
    try:
        return extract_day(day, workers=max(2, 12 // 2), out_dir=out_dir, legacy_cut=legacy_cut)
    except Exception as e:
        return f"{day}: FAILED {e}"


# ------------------------------------------------------------------ features

def _dist_m(lat, lon, la0, lo0):
    return np.hypot((lat - la0) * 111200.0, (lon - lo0) * 111200.0 * np.cos(np.radians(la0)))


def _load_points(day):
    p = ADSB_DIR / f"{day}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, columns=["icao", "callsign", "airport", "ts", "lat", "lon", "alt", "gs"])


def _takeoffs(pts):
    """Last ground point before an airborne one, at speed: the trace's take-off."""
    ground = (pts["alt"].values == -1)
    same = pts["icao"].values[1:] == pts["icao"].values[:-1]
    nxt_air = np.zeros(len(pts), bool)
    nxt_air[:-1] = same & ground[:-1] & ~ground[1:]
    dt = np.full(len(pts), np.inf)
    dt[:-1] = pts["ts"].values[1:] - pts["ts"].values[:-1]
    gs = np.nan_to_num(pts["gs"].values.astype(float))
    return pts[nxt_air & (dt < 60) & (gs > 60)]


def _landings(pts):
    """First ground point after an airborne one: the trace's touchdown."""
    ground = (pts["alt"].values == -1)
    same = pts["icao"].values[1:] == pts["icao"].values[:-1]
    land = np.zeros(len(pts), bool)
    land[1:] = same & ground[1:] & ~ground[:-1] & ((pts["ts"].values[1:] - pts["ts"].values[:-1]) < 60)
    return pts.loc[land, ["icao", "airport", "ts"]]


def _ground_series(pts):
    """icao -> (ts, lat, lon, gs, airport) of ground points, time-sorted."""
    g = pts[pts["alt"].values == -1]
    out = {}
    for icao, q in g.groupby("icao", sort=False):
        out[icao] = (q["ts"].to_numpy(float), q["lat"].to_numpy(float), q["lon"].to_numpy(float),
                     np.nan_to_num(q["gs"].to_numpy(float)), q["airport"].to_numpy())
    return out


def _trace_features(series, to_ts, ref, apt, stand_lat, stand_lon):
    """Trace timings as seconds before `ref` (MVT_TIME), the label's scale."""
    ts, lat, lon, gs, ap = series
    sel = (ts <= to_ts) & (ts >= to_ts - LOOKBACK_S) & (ap == apt)
    if not sel.any():
        return None
    ts, lat, lon, gs = ts[sel], lat[sel], lon[sel], gs[sel]
    brk = np.where(np.diff(ts) > SEG_GAP_S)[0]
    s0 = brk[-1] + 1 if len(brk) else 0
    ts, lat, lon, gs = ts[s0:], lat[s0:], lon[s0:], gs[s0:]
    f = dict(adsb_first_seen=ref - ts[0], adsb_gs_first=gs[0], adsb_n_ground=len(ts),
             adsb_cadence=np.median(np.diff(ts)) if len(ts) > 1 else np.nan,
             adsb_max_gap=np.max(np.diff(ts)) if len(ts) > 1 else np.nan)
    la0, lo0 = APT[apt]
    f["adsb_x0"] = (lon[0] - lo0) * 111200.0 * np.cos(np.radians(la0))
    f["adsb_y0"] = (lat[0] - la0) * 111200.0
    f["adsb_path_m"] = float(np.sum(_dist_m(lat[1:], lon[1:], lat[:-1], lon[:-1]))) if len(ts) > 1 else 0.0
    d0 = _dist_m(lat, lon, lat[0], lon[0])
    mv = np.where(d0 > 30)[0]
    f["adsb_leave0"] = ref - ts[mv[0]] if (len(mv) and gs[0] < 3) else np.nan
    # the last stop before take-off (holding point / line-up)
    stops = np.where(gs < 1)[0]
    f["adsb_last_stop"] = ref - ts[stops[-1]] if len(stops) else np.nan
    f["adsb_leave_stand"] = f["adsb_push"] = f["adsb_d_first_stand"] = np.nan
    if not np.isnan(stand_lat):
        ds = _dist_m(lat, lon, stand_lat, stand_lon)
        f["adsb_d_first_stand"] = ds[0]
        near = np.where(ds < STAND_R_M)[0]
        if len(near) and near[-1] + 1 < len(ts):
            f["adsb_leave_stand"] = ref - ts[near[-1] + 1]
        if ds[0] < PARK_R_M:
            lead = np.where(ds >= PARK_R_M)[0]
            k = lead[0] if len(lead) else len(ts)
            still = np.arange(k)[gs[:k] < 1]
            if len(still):
                da = _dist_m(lat, lon, np.median(lat[still]), np.median(lon[still]))
                j = np.where((da > MOVE_M) & (np.arange(len(ts)) > still[0]))[0]
                if len(j):
                    f["adsb_push"] = ref - ts[j[0]]
    return f


def _callsign_matches(dep, both, to):
    """Fallback for rows whose take-off the trace does not show.

    Receivers often see an airport's aprons and taxiways but not the runway end
    (LEMD, EDDF, LFPG: hundreds of aircraft with ground points a day, few
    visible take-off rolls). Match on the callsign instead: the same airport's
    ground points carrying this row's CALLSIGN_flt or FLIGHT_mvt, ending between
    LOOKBACK_S before and CS_AFTER_S after MVT_TIME; the latest such aircraft
    wins. Its last ground point stands in for the take-off (`to_ts`). An
    aircraft whose take-off the trace does show, at another time, is not this
    flight and is left out.
    """
    cols = ["index", "icao", "callsign", "airport", "to_ts", "dt", "cs_ok", "adt", "via_cs"]
    g = both.loc[(both["alt"].values == -1) & both["callsign"].notna(), ["icao", "callsign", "airport", "ts"]]
    if dep.empty or g.empty:
        return pd.DataFrame(columns=cols)
    g = g.assign(cs=g["callsign"].str.upper().str.strip())
    d = dep.reset_index()
    parts = []
    for key in ["CALLSIGN_flt", "FLIGHT_mvt"]:
        k = d[["index", "ADEP_mvt", "mvt_ts", key]].dropna(subset=[key])
        k = k.assign(cs=k[key].astype(str).str.upper().str.strip())
        parts.append(k.merge(g, left_on=["ADEP_mvt", "cs"], right_on=["airport", "cs"]))
    m = pd.concat(parts, ignore_index=True)
    m = m[(m["ts"] >= m["mvt_ts"] - LOOKBACK_S) & (m["ts"] <= m["mvt_ts"] + CS_AFTER_S)]
    seen = m[["index", "icao", "mvt_ts", "airport"]].drop_duplicates().merge(
        to[["icao", "airport", "ts"]], on=["icao", "airport"])
    seen = seen[(seen["ts"] >= seen["mvt_ts"] - LOOKBACK_S) & (seen["ts"] <= seen["mvt_ts"] + LOOKBACK_S)]
    m = m[~m.set_index(["index", "icao"]).index.isin(seen.set_index(["index", "icao"]).index)]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m = m.groupby(["index", "icao"], as_index=False).agg(
        to_ts=("ts", "max"), mvt_ts=("mvt_ts", "first"), callsign=("callsign", "first"),
        airport=("airport", "first"))
    m = m.sort_values("to_ts", ascending=False).drop_duplicates("index")
    m = m.drop_duplicates(["icao", "to_ts"])
    m["dt"] = m["to_ts"] - m["mvt_ts"]
    m["adt"], m["cs_ok"], m["via_cs"] = m["dt"].abs(), 1, 1
    out = dep.loc[m["index"]].reset_index().drop(columns=["mvt_ts"], errors="ignore")
    return out.merge(m[["index", "icao", "callsign", "to_ts", "dt", "cs_ok", "adt", "via_cs", "mvt_ts"]],
                     on="index")


def _climb_matches(dep, both, taken):
    """Second fallback (--adsb-climb-id), for rows neither earlier match found.

    An airborne point at the airport carrying this row's CALLSIGN_flt or
    FLIGHT_mvt within CLIMB_BEFORE_S before to CLIMB_AFTER_S after MVT_TIME is
    the climb-out; its icao is the airframe. The stand-in take-off (`to_ts`) is
    that airframe's last ground point here between LOOKBACK_S before and
    CS_AFTER_S after MVT_TIME, whatever callsign it carried. Rows with no such
    ground point stay unmatched. On labelled Jul 2025 days 77% (EDDF) to 20%
    (LFPG) of these last segments are the departure's own taxi; the rest are
    the inbound taxi-in, which adsb_to_dt (far below zero) tells apart.
    """
    cols = ["index", "icao", "callsign", "to_ts", "dt", "cs_ok", "adt", "via_cs", "via_climb", "mvt_ts"]
    air = both.loc[(both["alt"].values != -1) & both["callsign"].notna(), ["icao", "callsign", "airport", "ts"]]
    if dep.empty or air.empty:
        return pd.DataFrame(columns=cols)
    air = air.assign(cs=air["callsign"].str.upper().str.strip())
    d = dep.reset_index()
    parts = []
    for key in ["CALLSIGN_flt", "FLIGHT_mvt"]:
        k = d[["index", "ADEP_mvt", "mvt_ts", key]].dropna(subset=[key])
        k = k.assign(cs=k[key].astype(str).str.upper().str.strip())
        parts.append(k.merge(air, left_on=["ADEP_mvt", "cs"], right_on=["airport", "cs"]))
    m = pd.concat(parts, ignore_index=True)
    m = m[(m["ts"] >= m["mvt_ts"] - CLIMB_BEFORE_S) & (m["ts"] <= m["mvt_ts"] + CLIMB_AFTER_S)]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m = m.sort_values("ts").drop_duplicates("index")[["index", "icao", "callsign", "airport", "mvt_ts"]]
    g = both.loc[both["alt"].values == -1, ["icao", "airport", "ts"]]
    m = m.merge(g, on=["icao", "airport"])
    m = m[(m["ts"] >= m["mvt_ts"] - LOOKBACK_S) & (m["ts"] <= m["mvt_ts"] + CS_AFTER_S)]
    if m.empty:
        return pd.DataFrame(columns=cols)
    m = m.groupby(["index", "icao"], as_index=False).agg(
        to_ts=("ts", "max"), mvt_ts=("mvt_ts", "first"), callsign=("callsign", "first"))
    m = m.drop_duplicates("index")
    # an airframe segment already matched to another row is that row's
    m = m[~m.set_index(["icao", "to_ts"]).index.isin(taken)]
    m = m.drop_duplicates(["icao", "to_ts"])
    m["dt"] = m["to_ts"] - m["mvt_ts"]
    m["adt"], m["cs_ok"], m["via_cs"], m["via_climb"] = m["dt"].abs(), 1, 0, 1
    out = dep.loc[m["index"]].reset_index().drop(columns=["mvt_ts"], errors="ignore")
    return out.merge(m[cols], on="index")


def _day_features(dep, day, prev_pts, pts, climb=False):
    """ADS-B columns for the DEP rows taking off on `day`."""
    both = pd.concat([p for p in (prev_pts, pts) if p is not None], ignore_index=True)
    both = both.sort_values(["icao", "ts"], kind="stable").reset_index(drop=True)
    to = _takeoffs(pts.sort_values(["icao", "ts"], kind="stable").reset_index(drop=True))
    cand = dep.reset_index().merge(
        to[["icao", "callsign", "airport", "ts"]].rename(columns={"ts": "to_ts"}),
        left_on="ADEP_mvt", right_on="airport")
    cand["dt"] = cand["to_ts"] - cand["mvt_ts"]
    cand = cand[cand["dt"].abs() < MATCH_S].copy()
    cs = cand["callsign"].fillna("").str.upper().str.strip()
    cand["cs_ok"] = ((cs == cand["CALLSIGN_flt"].fillna("#").astype(str).str.upper())
                     | (cs == cand["FLIGHT_mvt"].fillna("#").astype(str).str.upper())).astype(int)
    cand["adt"] = cand["dt"].abs()
    cand = cand.sort_values(["cs_ok", "adt"], ascending=[False, True])
    # one trace per row and one row per trace take-off
    cand = cand.drop_duplicates("index").drop_duplicates(["icao", "to_ts"])
    cand["via_cs"] = 0
    cand = pd.concat([cand, _callsign_matches(dep[~dep.index.isin(cand["index"])], both, to)],
                     ignore_index=True)
    if climb:
        cand["via_climb"] = 0
        taken = pd.MultiIndex.from_arrays([cand["icao"], cand["to_ts"].astype(float)])
        cand = pd.concat([cand, _climb_matches(dep[~dep.index.isin(cand["index"])], both, taken)],
                         ignore_index=True)
    series = _ground_series(both)
    lands = {k: (g["ts"].to_numpy(float), g["airport"].to_numpy())
             for k, g in _landings(both).groupby("icao", sort=False)}
    rows = []
    for r in cand.itertuples():
        f = {"index": r.index, "adsb_match": 1, "adsb_cs_ok": r.cs_ok, "adsb_to_dt": r.dt,
             "adsb_via_cs": r.via_cs}
        if climb:
            f["adsb_via_climb"] = r.via_climb
        ld = lands.get(r.icao)
        if ld is not None:
            # the same airframe's last touchdown here, over this day and the one before
            prior = ld[0][(ld[0] < r.to_ts - 60) & (ld[1] == r.ADEP_mvt)]
            if len(prior):
                f["adsb_since_land"] = r.mvt_ts - prior.max()
        s = series.get(r.icao)
        if s is not None:
            tf = _trace_features(s, r.to_ts, r.mvt_ts, r.ADEP_mvt, r.stand_lat, r.stand_lon)
            if tf:
                f.update(tf)
        f["_icao"] = r.icao
        rows.append(f)
    out = pd.DataFrame(rows, columns=["index"] + ADSB_COLS + POS_COLS + (CLIMB_COLS if climb else [])
                       + ["_icao"])
    return out.set_index("index")


def _stand_xy(apt):
    import aptdat
    st = aptdat.stands(apt)
    if st.empty:
        return None
    lat0 = APT[apt][0]
    return np.c_[st["lat"].to_numpy() * 111200.0,
                 st["lon"].to_numpy() * 111200.0 * np.cos(np.radians(lat0))]


def _surface(both, today, icao):
    """SURF_COLS for one day's DEP rows: distinct other aircraft taxiing or
    holding at the airport per minute, averaged over the minutes before take-off."""
    from scipy.spatial import cKDTree
    out = pd.DataFrame(np.nan, index=today.index, columns=SURF_COLS)
    g = both.loc[both["alt"].values == -1, ["icao", "airport", "ts", "lat", "lon", "gs"]]
    if g.empty or today.empty:
        return out
    gs = np.nan_to_num(g["gs"].to_numpy(float))
    taxi = (gs >= TAXI_GS[0]) & (gs <= TAXI_GS[1])
    hold = np.zeros(len(g), bool)
    for apt in g["airport"].unique():
        sel = (g["airport"] == apt).to_numpy() & (gs < HOLD_GS)
        xy = _stand_xy(apt)
        if xy is None or not sel.any():
            continue
        lat0 = APT[apt][0]
        q = np.c_[g["lat"].to_numpy()[sel] * 111200.0,
                  g["lon"].to_numpy()[sel] * 111200.0 * np.cos(np.radians(lat0))]
        d, _ = cKDTree(xy).query(q)
        hold[np.where(sel)[0]] = d > HOLD_OFF_STAND_M
    minute = (g["ts"].to_numpy() // 60).astype(np.int64)
    pres = pd.DataFrame({"icao": g["icao"].to_numpy(), "airport": g["airport"].to_numpy(),
                         "minute": minute, "taxi": taxi, "hold": hold})
    day0 = int(today["mvt_ts"].min() // 60) - max(SURF_WINDOWS_MIN) - 1
    for state in ["taxi", "hold"]:
        p = pres[pres[state]].drop_duplicates(["airport", "minute", "icao"])
        own = {k: np.sort(v["minute"].to_numpy()) for k, v in p.groupby(["airport", "icao"])}
        for apt, rows in today.groupby("ADEP_mvt"):
            pa = p[p["airport"] == apt]
            m1 = int(rows["mvt_ts"].max() // 60) + 1
            dense = np.zeros(max(m1 - day0, 1) + 1)
            mm = pa["minute"].to_numpy() - day0
            ok = (mm >= 0) & (mm < len(dense))
            np.add.at(dense, mm[ok], 1)
            cs = np.concatenate([[0.0], np.cumsum(dense)])
            m0 = (rows["mvt_ts"].to_numpy() // 60).astype(np.int64) - day0
            me = icao.reindex(rows.index).to_numpy()
            for w in SURF_WINDOWS_MIN:
                hi = np.clip(m0, 0, len(dense))
                lo = np.clip(m0 - w, 0, len(dense))
                tot = cs[hi] - cs[lo]
                selfn = np.array([
                    0 if not isinstance(ic, str) or (apt, ic) not in own else
                    np.searchsorted(own[(apt, ic)], m + day0, "left")
                    - np.searchsorted(own[(apt, ic)], m + day0 - w, "left")
                    for ic, m in zip(me, m0)], dtype=float)
                out.loc[rows.index, f"adsb_surf_{state}_{w}"] = (tot - selfn) / w
            if state == "taxi":
                level = dense[dense > 0].mean() if (dense > 0).any() else np.nan
                out.loc[rows.index, "adsb_surf_taxi_rel_20"] = (
                    out.loc[rows.index, "adsb_surf_taxi_20"] / level)
    return out


def _neighbours(dep, feats):
    """NBR_COLS: other departures' trace-measured taxi around this take-off,
    airport- and runway-wide, self excluded (ties at the same second excluded)."""
    out = pd.DataFrame(np.nan, index=dep.index, columns=NBR_COLS)
    val = {k: feats[c].clip(0, NBR_CLIP_S).reindex(dep.index).to_numpy(float)
           for k, c in [("exit", "adsb_leave_stand"), ("push", "adsb_push"),
                        ("first", "adsb_first_seen")]}
    want = [("apt", "exit", "prev", w) for w in NBR_WINDOWS_MIN] + [
        ("apt", "exit", "next", 30), ("apt", "push", "prev", 30), ("apt", "first", "prev", 30),
        ("rwy", "exit", "prev", 30), ("rwy", "exit", "next", 30)]
    for scope, keys in [("apt", ["ADEP_mvt"]), ("rwy", ["ADEP_mvt", "RUNWAY_mvt"])]:
        for _, g in dep.groupby(keys, dropna=False):
            pos = dep.index.get_indexer(g.index)
            t = g["mvt_ts"].to_numpy()
            order = np.argsort(t, kind="stable")
            ts = t[order]
            for sc, k, side, w in want:
                if sc != scope:
                    continue
                v = val[k][pos][order]
                ok = ~np.isnan(v)
                cv = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
                cn = np.concatenate([[0], np.cumsum(ok)])
                if side == "prev":
                    lo, hi = np.searchsorted(ts, ts - w * 60, "left"), np.searchsorted(ts, ts, "left")
                else:
                    lo, hi = np.searchsorted(ts, ts, "right"), np.searchsorted(ts, ts + w * 60, "right")
                n = cn[hi] - cn[lo]
                mean = np.where(n > 0, (cv[hi] - cv[lo]) / np.maximum(n, 1), np.nan)
                res = np.empty(len(ts))
                res[order] = mean
                out.iloc[pos, out.columns.get_loc(f"adsb_nbr_{k}_{scope}_{side}{w}")] = res
                if (scope, k, side, w) == ("apt", "exit", "prev", 30):
                    cnt = np.empty(len(ts))
                    cnt[order] = n
                    out.iloc[pos, out.columns.get_loc("adsb_nbr_exit_n_apt_prev30")] = cnt
    return out


def drop_adsb_nbr(frame):
    """The --no-adsb-nbr arm: same cache, without the neighbour and surface columns."""
    return frame.drop(columns=[c for c in ADSB_NBR_COLS if c in frame.columns])


def drop_adsb_pos(frame):
    """The --no-adsb-pos arm: same cache, without POS_COLS."""
    return frame.drop(columns=[c for c in POS_COLS if c in frame.columns])


# What a departure looks like when the archive has the day but the receivers saw
# nothing at its airport (July 2026 LIRF: 0.3% matched): these columns read 0,
# every other adsb_* column NaN (NEXT_STEPS.md, B32)
ZERO_WHEN_UNTRACED = ["adsb_match", "adsb_cov_apt_day", "adsb_nbr_exit_n_apt_prev30",
                      "adsb_surf_taxi_10", "adsb_surf_taxi_20", "adsb_surf_hold_10",
                      "adsb_surf_hold_20", "adsb_via_climb"]
DAY_DROPOUT_SALT = "adsb-day"


def blank_untraced(frame, rows):
    """Set the rows' adsb_* columns, in place, to an airport-day with no trace."""
    rows = np.asarray(rows, bool)
    if not rows.any():
        return
    for c in frame.columns:
        if c.startswith("adsb_"):
            fill = 0.0 if c in ZERO_WHEN_UNTRACED else np.nan
            frame[c] = np.where(rows, fill, frame[c].to_numpy(dtype=float))


def day_dropout_rows(airport, date, frac):
    """A stable hashed `frac` of airport-days (all rows of a chosen day go)."""
    import zlib
    keys = pd.Series(np.asarray(airport).astype(str)) + "|" + pd.Series(np.asarray(date).astype(str))
    codes, uniq = pd.factorize(keys)
    h = np.array([zlib.crc32(f"{k}|{DAY_DROPOUT_SALT}".encode()) % 10000 for k in uniq])
    return (h < frac * 10000)[codes]


def raw_fingerprint(raw):
    """Which rows, in which order: cached features are reindexed onto raw by row
    position, so a dataset update that keeps the file name must change the key."""
    if "MVT_ID_mvt" in raw.columns:
        ids = pd.to_numeric(raw["MVT_ID_mvt"], errors="coerce").to_numpy(dtype="float64")
    else:
        t = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True)
        ids = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype="float64")
    return f"{len(raw)}|{hashlib.sha256(np.nan_to_num(ids, nan=-1.0).tobytes()).hexdigest()[:16]}".encode()


def _cache_key(days_needed, raw=None):
    import aptdat
    src = pathlib.Path(__file__).read_bytes() + pathlib.Path(aptdat.__file__).read_bytes()
    for d in days_needed:
        p = ADSB_DIR / f"{d}.parquet"
        src += f"{d}|{p.stat().st_size if p.exists() else -1}".encode()
    if raw is not None:
        src += raw_fingerprint(raw)
    return hashlib.sha256(src.replace(b"\r\n", b"\n")).hexdigest()[:16]


def dep_features(raw, tag, climb=False):
    """ADSB_COLS + ADSB_NBR_COLS for every row of `raw` (a training month or ranking.parquet);
    NaN on ARR rows, on DEP rows without a matched trace, and on days whose
    archive has not been fetched (adsb_match = 0 only where the day exists).
    climb=True adds the climb-out airframe match and CLIMB_COLS (--adsb-climb-id)."""
    import aptdat
    dep = raw[raw["PHASE_mvt"] == "DEP"][["ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
                                          "CALLSIGN_flt", "FLIGHT_mvt", "RUNWAY_mvt"]].copy()
    t = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True)
    dep["mvt_ts"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    dep["day"] = t.dt.strftime("%Y-%m-%d")
    days = sorted(dep["day"].unique())
    prev = {d: (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in days}
    key = _cache_key(sorted(set(days) | set(prev.values())), raw)
    cache = CACHE_DIR / f"{tag}{'.climb' if climb else ''}.{key}.parquet"
    if cache.exists():
        return pd.read_parquet(cache).reindex(raw.index)
    dep[["stand_lat", "stand_lon"]] = aptdat.lookup(dep["ADEP_mvt"], dep["STAND_mvt"]).to_numpy()
    parts = []
    for d in days:
        pts = _load_points(d)
        if pts is None:
            continue
        today = dep[dep["day"] == d]
        pp = _load_points(prev[d])
        f = _day_features(today, d, pp, pts, climb=climb).reindex(today.index)
        both = pd.concat([x for x in (pp, pts) if x is not None], ignore_index=True)
        surf = _surface(both, today, f["_icao"])
        f = pd.concat([f.drop(columns="_icao"), surf], axis=1)
        # a take-off hour the archive does not cover at all is unknown, not unseen
        # (2025-06-11 holds 5 hours); elsewhere no match means not seen
        covered = set(((pts["ts"] // 3600) % 24).astype(int).unique())
        hour = ((today["mvt_ts"] // 3600) % 24).astype(int)
        f["adsb_match"] = f["adsb_match"].fillna(0).where(hour.isin(covered))
        # how often a departure is seen here today: what a missing trace means
        f["adsb_cov_apt_day"] = f["adsb_match"].groupby(today["ADEP_mvt"]).transform("mean")
        # an hour the archive does not cover has no surface to count
        f.loc[f["adsb_match"].isna(), SURF_COLS] = np.nan
        if climb:
            f["adsb_via_climb"] = f["adsb_via_climb"].fillna(0).where(f["adsb_match"].notna())
        parts.append(f)
    cols = ADSB_COLS + ADSB_NBR_COLS + POS_COLS + (CLIMB_COLS if climb else [])
    feats =(pd.concat(parts) if parts else pd.DataFrame(columns=cols)).reindex(dep.index)
    feats[NBR_COLS] = _neighbours(dep, feats)
    feats = feats[cols].reindex(raw.index).astype("float32")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache)
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--streams", type=int, default=4)
    d.add_argument("--reverse", action="store_true", help="newest day first, to meet a forward stream")
    d.add_argument("--out-dir", help="write here instead of dataset/adsb (e.g. to test an extractor change)")
    d.add_argument("--alt-cut", choices=["field", "legacy"], default="field",
                   help="airborne points up to field elevation + MAX_ALT_FT (field), or up to "
                        "MAX_ALT_FT above sea level (legacy): every cached day so far was extracted "
                        "with legacy, so days added for the same models must use it too")
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.start, a.end, a.streams, a.reverse, a.out_dir, a.alt_cut == "legacy")


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    main()
