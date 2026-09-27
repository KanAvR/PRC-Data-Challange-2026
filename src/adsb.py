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
MAX_ALT_FT = 1500
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
             "adsb_cov_apt_day"]


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


def _parse_trace(b):
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
        if not ground and (alt is None or alt > MAX_ALT_FT):
            continue
        a = _near(lat, lon)
        if a is None:
            continue
        out.append((icao, reg, typ, cs, a, t0 + p[0], lat, lon,
                    -1 if ground else alt, p[4], p[5], p[6],
                    p[9] if len(p) > 9 else None))
    return out


def _parse_many(bs):
    out = []
    for b in bs:
        try:
            out.extend(_parse_trace(b))
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


def extract_day(day, workers=6):
    """Stream one UTC day and write dataset/adsb/<day>.parquet."""
    out = ADSB_DIR / f"{day}.parquet"
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
                futs.append(ex.submit(_parse_many, batch))
                batch = []
                while len(futs) > 24:
                    rows.extend(futs.pop(0).result())
        if batch:
            futs.append(ex.submit(_parse_many, batch))
        for f in futs:
            rows.extend(f.result())
    if n == 0:
        raise RuntimeError(f"{day}: no traces in the release")
    df = pd.DataFrame(rows, columns=POINT_COLS)
    for c in ["gs", "track"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["alt"] = pd.to_numeric(df["alt"], errors="coerce").astype("float32")
    df["src"] = df["src"].astype(str)
    ADSB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")  # two streams may meet on one day
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return f"{day}: traces {n} points {len(df)} {time.time() - t:.0f}s"


def download(start, end, streams=4, reverse=False):
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]
    days = days[::-1] if reverse else days
    days = [d for d in days if not (ADSB_DIR / f"{d}.parquet").exists()]
    print(f"{len(days)} days to fetch with {streams} streams", flush=True)
    with ThreadPoolExecutor(max_workers=streams) as ex:
        for msg in ex.map(_safe_extract, days):
            print(msg, flush=True)


def _safe_extract(day):
    try:
        return extract_day(day, workers=max(2, 12 // 2))
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


def _day_features(dep, day, prev_pts, pts):
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
    series = _ground_series(both)
    lands = {k: (g["ts"].to_numpy(float), g["airport"].to_numpy())
             for k, g in _landings(both).groupby("icao", sort=False)}
    rows = []
    for r in cand.itertuples():
        f = {"index": r.index, "adsb_match": 1, "adsb_cs_ok": r.cs_ok, "adsb_to_dt": r.dt}
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
        rows.append(f)
    out = pd.DataFrame(rows, columns=["index"] + ADSB_COLS).set_index("index")
    return out


def _cache_key(days_needed):
    import aptdat
    src = pathlib.Path(__file__).read_bytes() + pathlib.Path(aptdat.__file__).read_bytes()
    for d in days_needed:
        p = ADSB_DIR / f"{d}.parquet"
        src += f"{d}|{p.stat().st_size if p.exists() else -1}".encode()
    return hashlib.sha256(src.replace(b"\r\n", b"\n")).hexdigest()[:16]


def dep_features(raw, tag):
    """ADSB_COLS for every row of `raw` (a training month or ranking.parquet);
    NaN on ARR rows, on DEP rows without a matched trace, and on days whose
    archive has not been fetched (adsb_match = 0 only where the day exists)."""
    import aptdat
    dep = raw[raw["PHASE_mvt"] == "DEP"][["ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt",
                                          "CALLSIGN_flt", "FLIGHT_mvt"]].copy()
    t = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True)
    dep["mvt_ts"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    dep["day"] = t.dt.strftime("%Y-%m-%d")
    days = sorted(dep["day"].unique())
    prev = {d: (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for d in days}
    key = _cache_key(sorted(set(days) | set(prev.values())))
    cache = CACHE_DIR / f"{tag}.{key}.parquet"
    if cache.exists():
        return pd.read_parquet(cache).reindex(raw.index)
    dep[["stand_lat", "stand_lon"]] = aptdat.lookup(dep["ADEP_mvt"], dep["STAND_mvt"]).to_numpy()
    parts = []
    for d in days:
        pts = _load_points(d)
        if pts is None:
            continue
        today = dep[dep["day"] == d]
        f = _day_features(today, d, _load_points(prev[d]), pts).reindex(today.index)
        # a take-off hour the archive does not cover at all is unknown, not unseen
        # (2025-06-11 holds 5 hours); elsewhere no match means not seen
        covered = set(((pts["ts"] // 3600) % 24).astype(int).unique())
        hour = ((today["mvt_ts"] // 3600) % 24).astype(int)
        f["adsb_match"] = f["adsb_match"].fillna(0).where(hour.isin(covered))
        # how often a departure is seen here today: what a missing trace means
        f["adsb_cov_apt_day"] = f["adsb_match"].groupby(today["ADEP_mvt"]).transform("mean")
        parts.append(f)
    feats = (pd.concat(parts) if parts else pd.DataFrame(columns=ADSB_COLS)).reindex(raw.index)
    feats = feats[ADSB_COLS].astype("float32")
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
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.start, a.end, a.streams, a.reverse)


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    main()
