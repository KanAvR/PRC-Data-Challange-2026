import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adsb import ADSB_COLS, _day_features, _trace_features
from aptdat import _base, norm_stand

STAND = (52.3100, 4.7600)
T0 = 1_750_000_000.0      # first ground point
PUSH = T0 + 600           # leaves the stand
MVT = T0 + 1500           # take-off in the movement data
TO = MVT - 20             # take-off in the trace (last ground point)


def trace(icao="abc123", callsign="KLM1"):
    """Parked 10 min, pushes back and taxis ~2 km north, then climbs."""
    rows = []
    for t in np.arange(T0, PUSH, 10.0):                      # parked, gs 0
        rows.append((t, STAND[0], STAND[1], -1, 0.0))
    for i, t in enumerate(np.arange(PUSH, TO + 1, 10.0)):    # moving, ~10 m/s
        rows.append((t, STAND[0] + (i + 1) * 10 / 111200, STAND[1], -1, 20.0))
    rows[-1] = (TO, rows[-1][1], rows[-1][2], -1, 140.0)     # take-off roll
    rows.append((TO + 5, rows[-1][1] + 0.001, STAND[1], 300, 150.0))
    rows.append((TO + 15, rows[-1][1] + 0.002, STAND[1], 800, 160.0))
    df = pd.DataFrame(rows, columns=["ts", "lat", "lon", "alt", "gs"])
    df["icao"], df["callsign"], df["airport"] = icao, callsign, "EHAM"
    return df


class TraceFeatureTest(unittest.TestCase):
    def series(self):
        q = trace()
        q = q[q.alt == -1]
        return (q.ts.to_numpy(float), q.lat.to_numpy(float), q.lon.to_numpy(float),
                q.gs.to_numpy(float), q.airport.to_numpy())

    def test_timings_on_label_scale(self):
        f = _trace_features(self.series(), TO, MVT, "EHAM", *STAND)
        self.assertAlmostEqual(f["adsb_first_seen"], MVT - T0)
        # movement is detected at the first point > 15 m from the parked position
        self.assertAlmostEqual(f["adsb_push"], MVT - (PUSH + 10))
        # point i after PUSH is (i+1)*10 m out: last within 60 m is PUSH+40, next PUSH+50
        self.assertAlmostEqual(f["adsb_leave_stand"], MVT - (PUSH + 50))
        self.assertLess(f["adsb_d_first_stand"], 1.0)

    def test_no_stand_no_stand_features(self):
        f = _trace_features(self.series(), TO, MVT, "EHAM", np.nan, np.nan)
        self.assertTrue(np.isnan(f["adsb_push"]) and np.isnan(f["adsb_leave_stand"]))
        # first point beyond 30 m of the first position: PUSH+30 (40 m)
        self.assertAlmostEqual(f["adsb_leave0"], MVT - (PUSH + 30))

    def test_other_airport_ignored(self):
        self.assertIsNone(_trace_features(self.series(), TO, MVT, "EDDF", *STAND))


class MatchTest(unittest.TestCase):
    def dep(self, mvt_ts, callsign):
        return pd.DataFrame({"ADEP_mvt": ["EHAM"], "STAND_mvt": ["X"], "CALLSIGN_flt": [callsign],
                             "FLIGHT_mvt": [None], "mvt_ts": [mvt_ts], "day": ["2025-06-15"],
                             "stand_lat": [STAND[0]], "stand_lon": [STAND[1]]}, index=[7])

    def test_match_and_columns(self):
        out = _day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, trace())
        self.assertEqual(list(out.columns), ADSB_COLS)
        self.assertEqual(out.loc[7, "adsb_match"], 1)
        self.assertEqual(out.loc[7, "adsb_cs_ok"], 1)
        self.assertAlmostEqual(out.loc[7, "adsb_to_dt"], TO - MVT)

    def test_since_land_uses_same_airframe(self):
        arr = trace()                      # an earlier arrival of the same airframe
        arr = arr.iloc[::-1].copy()        # airborne first, then ground
        arr["ts"] = T0 - 7200 + np.arange(len(arr)) * 10.0
        arr.loc[arr.alt != -1, "alt"] = 500
        out = _day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, pd.concat([arr, trace()]))
        touchdown = T0 - 7200 + 2 * 10.0   # first ground point after the two airborne ones
        self.assertAlmostEqual(out.loc[7, "adsb_since_land"], MVT - touchdown)

    def test_too_far_in_time_is_unmatched(self):
        out = _day_features(self.dep(MVT + 600, "KLM1"), "2025-06-15", None, trace())
        self.assertEqual(len(out), 0)

    def test_callsign_preferred(self):
        pts = pd.concat([trace("aaa111", "OTHER"), trace("bbb222", "KLM1")])
        pts.loc[pts.icao == "aaa111", "ts"] += 5    # closer in time, wrong callsign
        out = _day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, pts)
        self.assertEqual(out.loc[7, "adsb_cs_ok"], 1)


class CoverageTest(unittest.TestCase):
    def test_uncovered_hour_is_unknown_and_rate(self):
        import tempfile
        import adsb
        with tempfile.TemporaryDirectory() as tmp:
            old = adsb.ADSB_DIR, adsb.CACHE_DIR
            adsb.ADSB_DIR, adsb.CACHE_DIR = Path(tmp), Path(tmp) / "feat"
            try:
                day = pd.Timestamp(T0, unit="s").strftime("%Y-%m-%d")
                pts = trace()
                for c in ["reg", "type", "track", "flags", "src"]:
                    pts[c] = None
                pts.to_parquet(Path(tmp) / f"{day}.parquet")
                t_seen = pd.Timestamp(MVT, unit="s", tz="UTC")
                raw = pd.DataFrame({
                    "PHASE_mvt": ["DEP", "DEP", "DEP"], "ADEP_mvt": ["EHAM"] * 3,
                    "STAND_mvt": ["X", "Y", "Z"], "CALLSIGN_flt": ["KLM1", "KLM2", "KLM3"],
                    "FLIGHT_mvt": [None] * 3,
                    # matched; same hour but unseen; an hour with no archive data
                    "MVT_TIME_UTC_mvt": [t_seen, t_seen + pd.Timedelta(minutes=5),
                                         t_seen + pd.Timedelta(hours=6)]})
                f = adsb.dep_features(raw, "t")
            finally:
                adsb.ADSB_DIR, adsb.CACHE_DIR = old
        self.assertEqual(list(f.adsb_match[:2]), [1, 0])
        self.assertTrue(np.isnan(f.adsb_match[2]))
        self.assertAlmostEqual(f.adsb_cov_apt_day[0], 0.5)


class StandNameTest(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(norm_stand("Gate A02"), "A2")
        self.assertEqual(norm_stand("H01"), "H1")
        self.assertEqual(norm_stand("100-(C)"), "100")
        self.assertEqual(norm_stand("Stand W 42"), "W42")
        self.assertEqual(norm_stand(None), "")
        self.assertEqual(_base("210A"), "210")
        self.assertEqual(_base("A10"), "A10")


if __name__ == "__main__":
    unittest.main()
