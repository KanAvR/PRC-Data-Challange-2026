"""Disk cache for build_features().

Feature engineering is 19s per run and both train.py and predict.py redo it from
scratch every time. The cache key includes a hash of features.py, airline.py and
airline_map.csv, weather.py and the weather CSVs, so editing the feature code or
refreshing the weather data invalidates every entry automatically -- a stale cache
can never silently survive a feature change.
"""
import hashlib
import os
import pathlib
from contextlib import nullcontext

import pandas as pd

import airline as _airline
import features as _features
import weather as _weather

CACHE_DIR = pathlib.Path(os.environ.get("PRC_CACHE_DIR", ".featcache"))


# everything build_features() output depends on besides the input file
_DEPENDENCIES = [pathlib.Path(_features.__file__), pathlib.Path(_airline.__file__),
                 _airline.MAP_PATH, pathlib.Path(_weather.__file__)]


def _cache_path(path):
    # line endings are normalised: a git checkout that only converts them must
    # not invalidate the cache
    src = b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in _DEPENDENCIES)
    for w in sorted(_weather.WEATHER_DIR.glob("*.csv")):
        ws = w.stat()
        src += f"{w.name}|{ws.st_mtime_ns}|{ws.st_size}".encode()
    st = os.stat(path)
    h = hashlib.sha256(src + f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{pathlib.Path(path).stem}.{h}.parquet"


def build_features_cached(path, df, timings=None, stage="features"):
    """build_features(df) for a parquet file, memoised on disk."""
    measure = timings.stage if timings is not None else lambda *args, **kwargs: nullcontext()
    with measure(f"{stage}.cache_key"):
        cp = _cache_path(path)
    if cp.exists():
        with measure(f"{stage}.cache_read", path=str(cp)):
            return pd.read_parquet(cp)
    with measure(f"{stage}.build"):
        feats = _features.build_features(df)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp.parquet")
    with measure(f"{stage}.cache_write", path=str(cp)):
        feats.reset_index(drop=True).to_parquet(tmp, index=False)
        tmp.replace(cp)  # atomic: a crash mid-write cannot leave a partial cache
    with measure(f"{stage}.cache_reopen", path=str(cp)):
        return pd.read_parquet(cp)
