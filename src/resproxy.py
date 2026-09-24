"""Residualised neighbour proxy: how much slower than usual are the neighbours.

The neighbour proxy (features._dep_proxy) averages nearby departures'
MVT-to-AOBT_3 and MVT-to-LOBT gaps. That mean mixes "taxi is slow right now"
with "the neighbours happened to use far stands", and a tree cannot undo the
mix once twenty neighbours are averaged. Here each neighbour's gap is taken
relative to its own expected taxi, te_stand_rwy, before averaging.

te_stand_rwy only exists after the target encodings, so this runs after them,
on each frame separately (the fit frame with its out-of-fold encodings, the
apply frame with full ones). A fit row's neighbours share its month, and their
out-of-fold encodings exclude that whole month, so no row sees its own label.
"""
import numpy as np
import pandas as pd

from features import PROXY_MAX_SEC, _window_mean

RES_SOURCES = ["gap_AOBT_3_flt", "gap_LOBT_flt"]
RES_WINDOWS = [30, 60]
EXPECTED = "te_stand_rwy"

RES_PROXY_COLS = [
    f"resproxy_{scope}_{src}_{d}_{w}"
    for scope in ("apt", "rwy") for src in RES_SOURCES for d in ("before", "after") for w in RES_WINDOWS
]


def add_residual_proxy(X, t):
    """Add RES_PROXY_COLS to X in place. t: take-off time, seconds (meta['t'])."""
    for c in RES_PROXY_COLS:
        X[c] = np.nan
    t = np.asarray(t, dtype=float).astype("int64").astype("datetime64[s]")
    apt = X["ADEP_mvt"].astype(str).values
    rwy = X["RUNWAY_mvt"].astype(str).values
    expected = X[EXPECTED].values.astype(float)
    res = {}
    for src in RES_SOURCES:
        gap = X[src].values.astype(float)
        ok = (gap >= 0) & (gap <= PROXY_MAX_SEC)
        res[src] = np.where(ok, gap - expected, np.nan)
    out = {c: np.full(len(X), np.nan) for c in RES_PROXY_COLS}
    for a in pd.unique(apt):
        in_apt = apt == a
        groups = [("apt", in_apt)] + [("rwy", in_apt & (rwy == r)) for r in pd.unique(rwy[in_apt])]
        for scope, sel in groups:
            idx = np.flatnonzero(sel)
            ts = t[idx]
            for src in RES_SOURCES:
                for after in (False, True):
                    d = "after" if after else "before"
                    for w, v in _window_mean(ts, res[src][idx], ts, RES_WINDOWS, after=after).items():
                        out[f"resproxy_{scope}_{src}_{d}_{w}"][idx] = v
    for c, v in out.items():
        X[c] = v
