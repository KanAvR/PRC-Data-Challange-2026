"""METAR weather for the ten airports, from the Iowa Environmental Mesonet.

Taxi times react to local, short-lived conditions -- low-visibility procedures,
de-icing, thunderstorms, strong wind -- which the airport's own METAR observes
directly. IEM's ASOS archive serves routine and special reports (METAR/SPECI)
for every ICAO station as CSV, with the main fields already decoded.

    python src/weather.py download    # raw CSVs into dataset/weather/
"""
import argparse
import io
import pathlib
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# features.py imports this module, so the station list lives here
STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
WEATHER_DIR = pathlib.Path("dataset/weather")
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# one day of margin before training starts, so lookback windows are full on Jan 1
START, END = "2024-12-31", "2026-08-02"
CHUNK_MONTHS = 3
PAUSE_S = 2.0  # IEM is a free service; keep requests sequential and spaced


def _fetch(station, start, end, retries=3):
    params = [("station", station), ("data", "all"),
              ("year1", start.year), ("month1", start.month), ("day1", start.day),
              ("year2", end.year), ("month2", end.month), ("day2", end.day),
              ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
              ("elev", "no"), ("missing", "M"), ("trace", "T"), ("direct", "no"),
              ("report_type", "3"), ("report_type", "4")]  # 3 = METAR, 4 = SPECI
    url = IEM_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                text = r.read().decode()
            return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except Exception as e:  # network hiccups: back off and retry
            if attempt == retries - 1:
                raise
            print(f"  {station} {start:%Y-%m-%d}: {e}; retrying", flush=True)
            time.sleep(PAUSE_S * 5 * (attempt + 1))


def download(start=START, end=END, force=False):
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    edges = list(pd.date_range(start, end, freq=f"{CHUNK_MONTHS}MS"))
    edges = [pd.Timestamp(start)] + [e for e in edges if e > pd.Timestamp(start)]
    if edges[-1] < pd.Timestamp(end):
        edges.append(pd.Timestamp(end))
    for station in STATIONS:
        path = WEATHER_DIR / f"{station}.csv"
        if path.exists() and not force:
            print(f"{station}: exists, skipping")
            continue
        parts = []
        for a, b in zip(edges[:-1], edges[1:]):
            parts.append(_fetch(station, a, b))
            time.sleep(PAUSE_S)
        df = pd.concat(parts, ignore_index=True).drop_duplicates()
        df = df.sort_values("valid", kind="mergesort")
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"{station}: {len(df)} reports, "
              f"{df['valid'].min()} .. {df['valid'].max()}", flush=True)


NO_CEILING_FT = 30000.0
WX_FLAGS = {  # METAR present-weather groups
    "wx_snow": ("SN", "SG", "PL", "GS"),
    "wx_freezing": ("FZ",),
    "wx_rain": ("RA", "DZ"),
    "wx_ts": ("TS",),
    "wx_fog": ("FG",),
    "wx_mist": ("BR", "HZ"),
}


def _num(s):
    return pd.to_numeric(s.replace({"M": None, "T": "0"}), errors="coerce")


def load_station(station):
    """One station's reports, decoded to metric units and weather flags."""
    path = WEATHER_DIR / f"{station}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python src/weather.py download`")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    # naive UTC at ns resolution throughout, so merge keys always match
    w = pd.DataFrame({"t": pd.to_datetime(raw["valid"]).astype("datetime64[ns]")})
    w["temp_c"] = (_num(raw["tmpf"]) - 32) * 5 / 9
    w["dewpt_spread_c"] = w["temp_c"] - (_num(raw["dwpf"]) - 32) * 5 / 9
    w["wind_kt"] = _num(raw["sknt"])
    w["wind_dir"] = _num(raw["drct"])
    w["gust_kt"] = _num(raw["gust"]).fillna(w["wind_kt"])
    w["vis_km"] = (_num(raw["vsby"]) * 1.609).clip(upper=10)
    cavok = raw["metar"].str.contains(" CAVOK", regex=False)
    w.loc[cavok, "vis_km"] = 10.0
    ceiling = pd.Series(np.nan, index=raw.index)
    for i in range(1, 5):
        cover, height = raw[f"skyc{i}"], _num(raw[f"skyl{i}"])
        layer = cover.isin(["BKN", "OVC", "VV"]) & height.notna()
        ceiling = ceiling.where(ceiling.notna() | ~layer, height)
    # a report with no broken/overcast layer has no ceiling, not a missing one
    has_report = raw["metar"].str.len() > 0
    w["ceiling_ft"] = ceiling.where(ceiling.notna() | ~has_report, NO_CEILING_FT)
    codes = raw["wxcodes"].replace("M", "")
    for name, groups in WX_FLAGS.items():
        w[name] = codes.str.contains("|".join(groups), regex=True).astype(float)
    w["wx_heavy"] = codes.str.contains("+", regex=False).astype(float)

    return w.dropna(subset=["t"]).sort_values("t", kind="mergesort").reset_index(drop=True)


_CACHE = {}


def _station(station):
    if station not in _CACHE:
        _CACHE[station] = load_station(station)
    return _CACHE[station]


def _runway_heading(runway):
    deg = pd.to_numeric(runway.astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
    return deg * 10.0


def _window(w, t, col, minutes, how):
    """Aggregate col over reports in [t - minutes, t]; NaN when there are none."""
    ts = w["t"].values.astype("datetime64[s]").astype(np.int64)
    q = t.astype(np.int64)
    lo = np.searchsorted(ts, q - minutes * 60, side="left")
    hi = np.searchsorted(ts, q, side="right")
    v = w[col].values.astype(float)
    out = np.full(len(q), np.nan)
    ok = hi > lo
    if how == "any":  # 0/1 flags: a cumulative count is enough
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])
        out[ok] = (cs[hi[ok]] - cs[lo[ok]] > 0).astype(float)
    elif how == "max":
        # windows hold only a few reports, so a direct loop over them is cheap
        out[ok] = [np.nanmax(v[a:b]) if np.isfinite(v[a:b]).any() else np.nan
                   for a, b in zip(lo[ok], hi[ok])]
    else:
        out[ok] = [np.nanmin(v[a:b]) if np.isfinite(v[a:b]).any() else np.nan
                   for a, b in zip(lo[ok], hi[ok])]
    return out


LATEST_COLS = ["temp_c", "dewpt_spread_c", "wind_kt", "gust_kt", "vis_km", "ceiling_ft",
               *WX_FLAGS, "wx_heavy"]
WINDOWS = [  # (report column, feature name, minutes before t, aggregate)
    ("wx_snow", "met_snow_60m", 60, "any"), ("wx_freezing", "met_freezing_60m", 60, "any"),
    ("wx_ts", "met_ts_60m", 60, "any"), ("wx_fog", "met_fog_60m", 60, "any"),
    ("vis_km", "met_min_vis_60m", 60, "min"), ("gust_kt", "met_max_gust_60m", 60, "max"),
    ("wx_snow", "met_snow_6h", 360, "any"),
]
# Three further rounds were measured with validate.py (LightGBM mu, four splits)
# and all came out worse than the list above, so this set is deliberate:
#   - derived operational flags: de-icing conditions and precipitation counts over
#     3h, low-visibility-procedure thresholds, 12h snow, 12h minimum temperature
#     (+0.36 robust pooled, significantly worse on the holdout)
#   - everything parseable from the report text: RVR and its value, vertical
#     visibility, recent-weather (RE) snow and freezing, pressure and its 3h
#     change, humidity, variable-wind groups, 1-3h changes in visibility,
#     temperature and wind direction, reports per hour (+0.36 robust pooled)
#   - only the four sharpest of those (RVR value and flag, RE snow, RE freezing),
#     whose raw signal looks strong -- RVR under 600 m averages 1518s of taxi
#     against 980s, recent freezing 1763s -- but which add nothing once
#     visibility, the weather flags and temperature are present (+0.17 robust
#     pooled, significantly worse on holdout and inner)


def weather_features(apt, t, runway):
    """Weather at each movement: the latest report at or before t (within 2h),
    runway-relative wind, and what happened in the hour(s) before."""
    t = pd.Series(pd.to_datetime(t, utc=True)).dt.tz_localize(None).astype("datetime64[ns]")
    apt = pd.Series(apt).reset_index(drop=True)
    runway = pd.Series(runway).reset_index(drop=True)
    cols = ([f"met_{c}" for c in LATEST_COLS]
            + ["met_crosswind_kt", "met_tailwind_kt", "met_report_age_min"]
            + [name for _, name, _, _ in WINDOWS])
    out = pd.DataFrame(np.nan, index=t.index, columns=cols)
    for station in STATIONS:
        sel = (apt == station).values
        if not sel.any():
            continue
        w = _station(station)
        q = pd.DataFrame({"t": t[sel].values, "i": np.flatnonzero(sel)}).sort_values("t")
        m = pd.merge_asof(q, w, on="t", direction="backward",
                          tolerance=pd.Timedelta("2h"), allow_exact_matches=True)
        idx = m["i"].values
        for c in LATEST_COLS:
            out.loc[idx, f"met_{c}"] = m[c].values
        rep_t = pd.merge_asof(q, w[["t"]].assign(rt=w["t"]), on="t", direction="backward",
                              tolerance=pd.Timedelta("2h"))["rt"]
        out.loc[idx, "met_report_age_min"] = ((m["t"] - rep_t).dt.total_seconds() / 60).values
        rel = np.deg2rad(m["wind_dir"].values - _runway_heading(runway[idx]).values)
        spd = m["wind_kt"].values
        out.loc[idx, "met_crosswind_kt"] = np.abs(spd * np.sin(rel))
        out.loc[idx, "met_tailwind_kt"] = -spd * np.cos(rel)
        qt = q["t"].values.astype("datetime64[s]")
        for col, name, minutes, how in WINDOWS:
            out.loc[idx, name] = _window(w, qt, col, minutes, how)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch raw METAR/SPECI CSVs into dataset/weather/")
    d.add_argument("--start", default=START)
    d.add_argument("--end", default=END, help="exclusive")
    d.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.start, args.end, args.force)


if __name__ == "__main__":
    main()
