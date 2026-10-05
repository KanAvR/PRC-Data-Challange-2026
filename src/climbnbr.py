"""--adsb-climb-nbr-clean: keep inbound-only climb-out matches out of the neighbour means.

A climb-out match (--adsb-climb-id) whose airframe's last ground segment ends
long before take-off is the inbound taxi-in: its own columns are useful (the
row knows when the airframe arrived; on climb_lgb these rows gain as much as
departure segments), but its "first seen" is hours before take-off, and it
enters every other departure's adsb_nbr_* means as if it were a taxi. On
climb_lgb the rows the match does not touch got worse on all four splits
(+38..+143 normal-row MSE). This recomputes adsb.NBR_COLS with those rows'
stand exit, pushback and first-seen timings left out.
"""
import numpy as np
import pandas as pd

import adsb

INBOUND_TO_DT = -300.0   # last ground point this long before MVT_TIME: not this departure's taxi


def clean_neighbours(raw, feats):
    """Recompute adsb.NBR_COLS in feats (indexed like raw) and return it."""
    dep = raw[raw["PHASE_mvt"] == "DEP"][["ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]].copy()
    t = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True)
    dep["mvt_ts"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    f = feats.loc[dep.index, ["adsb_leave_stand", "adsb_push", "adsb_first_seen",
                              "adsb_via_climb", "adsb_to_dt"]].astype(float)
    inbound = (f["adsb_via_climb"] == 1) & (f["adsb_to_dt"] < INBOUND_TO_DT)
    f.loc[inbound, ["adsb_leave_stand", "adsb_push", "adsb_first_seen"]] = np.nan
    nb = adsb._neighbours(dep, f)
    feats = feats.copy()
    feats.loc[dep.index, adsb.NBR_COLS] = nb[adsb.NBR_COLS].to_numpy(dtype="float32")
    return feats
