"""Airport geometry for the ten airports, from the X-Plane Scenery Gateway.

Each airport's recommended scenery pack carries an `apt.dat` with the stands
(row 1300: position, heading, type, allowed operations, name; row 1301: ICAO
size class and airlines) and the taxi network (1201 nodes, 1202 edges, 1204
active zones). We use it for stand coordinates, matched to `STAND_mvt` by
name.

    python src/aptdat.py download    # one <ICAO>.dat per airport into dataset/aptdat/
    python src/aptdat.py match       # stand match rate per airport
"""
import argparse
import base64
import io
import json
import pathlib
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from weather import STATIONS

APTDAT_DIR = pathlib.Path("dataset/aptdat")
API = "https://gateway.x-plane.com/apiv1"


def _get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def download(force=False):
    APTDAT_DIR.mkdir(parents=True, exist_ok=True)
    for apt in STATIONS:
        path = APTDAT_DIR / f"{apt}.dat"
        if path.exists() and not force:
            print(f"{apt}: exists, skipping")
            continue
        sid = _get(f"{API}/airport/{apt}")["airport"]["recommendedSceneryId"]
        blob = _get(f"{API}/scenery/{sid}")["scenery"]["masterZipBlob"]
        z = zipfile.ZipFile(io.BytesIO(base64.b64decode(blob)))
        dat = [n for n in z.namelist() if n.endswith(".dat")][0]
        path.write_bytes(z.read(dat))
        print(f"{apt}: scenery {sid}")


_PREFIX = re.compile(r"^(STAND|GATE|RAMP|PARKING|POS)\s*", re.I)


def norm_stand(name):
    """Canonical stand key: upper case, no prefix word or '-(C)' size suffix,
    no separators, no zero padding inside the number (`H01` -> `H1`)."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).upper().strip()
    s = _PREFIX.sub("", s)
    s = re.sub(r"-?\([A-Z]\)$", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", s)


def _base(key):
    """Drop a trailing variant letter (`210A`, `133L`) when it follows a digit."""
    return re.sub(r"(?<=[0-9])[A-Z]$", "", key)


def stands(apt):
    """One row per apt.dat stand: key, name, lat, lon, heading, type, ops, size."""
    rows, last = [], None
    for line in (APTDAT_DIR / f"{apt}.dat").read_text(encoding="latin-1").splitlines():
        p = line.split(None, 6)
        if not p:
            continue
        if p[0] == "1300" and len(p) >= 6:
            name = p[6].strip() if len(p) > 6 else ""
            last = dict(airport=apt, name=name, key=norm_stand(name),
                        lat=float(p[1]), lon=float(p[2]), heading=float(p[3]),
                        kind=p[4], ops=p[5], size=None)
            rows.append(last)
        elif p[0] == "1301" and last is not None and len(p) >= 2:
            last["size"] = p[1]
    return pd.DataFrame(rows)


def stand_table(apt):
    """Lookup key -> (lat, lon, size) with a fallback to the centroid of a
    stand's variants (`210` -> mean of `210A`, `210B`)."""
    s = stands(apt)
    if s.empty:
        return pd.DataFrame(columns=["lat", "lon", "size"])
    exact = s.groupby("key").agg(lat=("lat", "mean"), lon=("lon", "mean"), size=("size", "max"))
    s["base"] = s.key.map(_base)
    base = s.groupby("base").agg(lat=("lat", "mean"), lon=("lon", "mean"), size=("size", "max"))
    return pd.concat([exact, base[~base.index.isin(exact.index)]])


def lookup(apt_col, stand_col):
    """Stand coordinates for a pair of columns; NaN where unmatched."""
    out = pd.DataFrame(index=apt_col.index, columns=["stand_lat", "stand_lon"], dtype=float)
    for apt in STATIONS:
        m = apt_col == apt
        if not m.any() or not (APTDAT_DIR / f"{apt}.dat").exists():
            continue
        t = stand_table(apt)
        k = stand_col[m].map(norm_stand)
        k = k.where(k.isin(t.index), k.map(_base))
        out.loc[m, "stand_lat"] = k.map(t["lat"]).to_numpy()
        out.loc[m, "stand_lon"] = k.map(t["lon"]).to_numpy()
    return out


def _match_report():
    import glob
    cols = ["ADEP_mvt", "PHASE_mvt", "STAND_mvt"]
    for label, files in (("train", sorted(glob.glob("dataset/training_*.parquet"))),
                         ("ranking", ["dataset/ranking.parquet"])):
        d = pd.concat([pd.read_parquet(f, columns=cols) for f in files])
        d = d[(d.PHASE_mvt == "DEP") & d.ADEP_mvt.isin(STATIONS) & d.STAND_mvt.notna()]
        hit = lookup(d.ADEP_mvt, d.STAND_mvt).stand_lat.notna()
        print(label, hit.groupby(d.ADEP_mvt).mean().round(3).to_dict(), f"all {hit.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "match"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    download(a.force) if a.cmd == "download" else _match_report()


if __name__ == "__main__":
    main()
