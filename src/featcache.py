"""Disk cache for build_features().

Feature engineering is 19s per run and both train.py and predict.py redo it from
scratch every time. The cache key includes a hash of features.py, airline.py and
airline_map.csv, so editing the feature code invalidates every entry automatically -- a stale cache can never
silently survive a feature change.
"""
import hashlib
import os
import pathlib

import pandas as pd

import airline as _airline
import features as _features

CACHE_DIR = pathlib.Path(os.environ.get("PRC_CACHE_DIR", ".featcache"))


# everything build_features() output depends on besides the input file
_DEPENDENCIES = [pathlib.Path(_features.__file__), pathlib.Path(_airline.__file__),
                 _airline.MAP_PATH]


def _cache_path(path):
    src = b"".join(p.read_bytes() for p in _DEPENDENCIES)
    st = os.stat(path)
    h = hashlib.sha256(src + f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{pathlib.Path(path).stem}.{h}.parquet"


def build_features_cached(path, df):
    """build_features(df) for a parquet file, memoised on disk."""
    cp = _cache_path(path)
    if cp.exists():
        return pd.read_parquet(cp)
    feats = _features.build_features(df)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp.parquet")
    feats.reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(cp)  # atomic: a crash mid-write cannot leave a partial cache
    return pd.read_parquet(cp)
