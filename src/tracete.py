"""--trace-te: label-free "encodings" from the own-trace stand exits of other flights.

adsb_leave_stand is a trace-measured taxi time, available without the label on
training and ranking rows alike. Averaging it over the other departures from the
same runway in the same UTC hour of the same day gives each row that hour's
measured taxi level, including rows whose own trace shows no stand exit. Unlike
a target encoding it reads no labels, so fit and apply rows are built the same
way, and on ranking rows it measures 2026 itself.

Residual check against climb_lgb (normal rows without an own exit, centred
within airport): +0.033 / +0.015 / +0.017 / +0.020 on the four splits
(NEXT_STEPS.md, B13). adsb-nbr's +0.02..+0.03 became -0.24 offline.
"""
import numpy as np
import pandas as pd

TRACE_TE_COLS = ["trace_rwy_hour", "trace_rwy_hour_n"]
MIN_OTHERS = 3
CLIP_S = 3600.0


def add_trace_encodings(X, dates):
    """Add TRACE_TE_COLS to X in place. dates: UTC date string per row."""
    ex = X["adsb_leave_stand"].to_numpy(dtype=float)
    ex = np.clip(ex, 0, CLIP_S)
    seen = np.isfinite(ex)
    key = (pd.Series(np.asarray(dates), index=X.index).astype(str) + "|"
           + X["ADEP_RUNWAY"].astype(str) + "|" + X["hour"].astype(int).astype(str))
    codes, _ = pd.factorize(key)
    s = np.bincount(codes, weights=np.where(seen, ex, 0.0))
    n = np.bincount(codes, weights=seen.astype(float))
    others_s = s[codes] - np.where(seen, ex, 0.0)
    others_n = n[codes] - seen
    X["trace_rwy_hour"] = np.where(others_n >= MIN_OTHERS, others_s / np.maximum(others_n, 1), np.nan)
    X["trace_rwy_hour_n"] = others_n.astype(float)


def drop_trace_te(frame):
    return frame.drop(columns=[c for c in TRACE_TE_COLS if c in frame.columns])
