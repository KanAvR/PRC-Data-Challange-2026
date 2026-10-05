import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adsb import ADSB_COLS, NBR_COLS, SURF_COLS, _day_features, _neighbours, _surface, _trace_features
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
        self.assertEqual([c for c in out.columns if c != "_icao"], ADSB_COLS)
        self.assertEqual(out.loc[7, "_icao"], "abc123")
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

    def test_callsign_fallback_without_takeoff_roll(self):
        pts = trace()
        pts = pts[(pts.alt == -1) & (pts.gs < 100)]     # taxiways seen, runway end not
        out = _day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, pts)
        self.assertEqual(out.loc[7, "adsb_via_cs"], 1)
        self.assertAlmostEqual(out.loc[7, "adsb_first_seen"], MVT - T0)
        self.assertLess(out.loc[7, "adsb_to_dt"], 0)
        # a different callsign finds nothing
        self.assertEqual(len(_day_features(self.dep(MVT, "OTHER9"), "2025-06-15", None, pts)), 0)

    def test_climb_out_names_the_airframe(self):
        pts = trace()
        # ground points carry no callsign and the runway end is not seen; the
        # climb-out (airborne, with the callsign) is
        pts = pts[(pts.alt != -1) | (pts.gs < 100)].copy()
        pts.loc[pts.alt == -1, "callsign"] = None
        self.assertEqual(len(_day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, pts)), 0)
        out = _day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, pts, climb=True)
        self.assertEqual(out.loc[7, "adsb_via_climb"], 1)
        self.assertEqual(out.loc[7, "adsb_via_cs"], 0)
        self.assertAlmostEqual(out.loc[7, "adsb_first_seen"], MVT - T0)
        self.assertAlmostEqual(out.loc[7, "adsb_leave_stand"], MVT - (PUSH + 50))
        # a climb-out with another callsign names nothing
        self.assertEqual(len(_day_features(self.dep(MVT, "OTHER9"), "2025-06-15", None, pts, climb=True)), 0)

    def test_climb_flag_leaves_earlier_matches_alone(self):
        a = _day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, trace())
        b = _day_features(self.dep(MVT, "KLM1"), "2025-06-15", None, trace(), climb=True)
        self.assertEqual(b.loc[7, "adsb_via_climb"], 0)
        pd.testing.assert_frame_equal(a, b.drop(columns="adsb_via_climb"))

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
                    "FLIGHT_mvt": [None] * 3, "RUNWAY_mvt": ["24"] * 3,
                    # matched; same hour but unseen; an hour with no archive data
                    "MVT_TIME_UTC_mvt": [t_seen, t_seen + pd.Timedelta(minutes=5),
                                         t_seen + pd.Timedelta(hours=6)]})
                f = adsb.dep_features(raw, "t")
            finally:
                adsb.ADSB_DIR, adsb.CACHE_DIR = old
        self.assertEqual(list(f.adsb_match[:2]), [1, 0])
        self.assertTrue(np.isnan(f.adsb_match[2]))
        self.assertAlmostEqual(f.adsb_cov_apt_day[0], 0.5)


class AltitudeCutTest(unittest.TestCase):
    def test_cut_is_relative_to_field_elevation(self):
        import orjson
        from adsb import APT, _parse_trace

        def one(apt, alt):
            la, lo = APT[apt]
            b = orjson.dumps({"icao": "abc", "timestamp": 0, "trace": [[0, la, lo, alt, 150, 0, 0]]})
            return len(_parse_trace(b))
        self.assertEqual(one("LEMD", 3000), 1)     # 1,000 ft above a ~2,000 ft field
        self.assertEqual(one("LEMD", 3600), 0)
        self.assertEqual(one("EHAM", 1400), 1)
        self.assertEqual(one("EHAM", 2500), 0)     # above 1,500 ft at a sea-level field
        self.assertEqual(one("LEMD", "ground"), 1)


class NeighbourSurfaceTest(unittest.TestCase):
    def test_neighbours_exclude_self_and_split_sides(self):
        t0 = 1_750_000_000.0
        dep = pd.DataFrame({"ADEP_mvt": ["EHAM"] * 4, "RUNWAY_mvt": ["24", "24", "18", "24"],
                            "mvt_ts": [t0, t0 + 300, t0 + 600, t0 + 3600]}, index=[10, 11, 12, 13])
        feats = pd.DataFrame({"adsb_leave_stand": [600.0, 900.0, np.nan, 700.0],
                              "adsb_push": np.nan, "adsb_first_seen": np.nan}, index=dep.index)
        out = _neighbours(dep, feats)
        self.assertEqual(list(out.columns), NBR_COLS)
        # row 12 (t0+600): previous 30 min on the airport = rows 10, 11 (not itself)
        self.assertAlmostEqual(out.loc[12, "adsb_nbr_exit_apt_prev30"], 750.0)
        self.assertEqual(out.loc[12, "adsb_nbr_exit_n_apt_prev30"], 2)
        # row 10: nothing before it; next 30 min = row 11 only (row 12 has no estimate)
        self.assertTrue(np.isnan(out.loc[10, "adsb_nbr_exit_apt_prev30"]))
        self.assertAlmostEqual(out.loc[10, "adsb_nbr_exit_apt_next30"], 900.0)
        # runway scope: row 11 on 24 sees row 10, not row 12 on 18
        self.assertAlmostEqual(out.loc[11, "adsb_nbr_exit_rwy_prev30"], 600.0)
        self.assertTrue(np.isnan(out.loc[12, "adsb_nbr_exit_rwy_prev30"]))

    def test_surface_counts_others_taxiing(self):
        from adsb import APT
        la, lo = APT["EHAM"]
        la += 0.02                                   # well away from any stand
        mvt = 1_750_000_000.0 - (1_750_000_000.0 % 60) + 1200
        rows = []
        for ic, gs in [("me", 15.0), ("other1", 15.0), ("other2", 0.0)]:
            for k in range(20):                      # one point a minute for 20 minutes
                rows.append((ic, "EHAM", mvt - 1200 + 60 * k, la, lo, -1, gs))
        both = pd.DataFrame(rows, columns=["icao", "airport", "ts", "lat", "lon", "alt", "gs"])
        today = pd.DataFrame({"ADEP_mvt": ["EHAM"], "mvt_ts": [mvt]}, index=[5])
        out = _surface(both, today, pd.Series({5: "me"}))
        self.assertEqual(list(out.columns), SURF_COLS)
        self.assertAlmostEqual(out.loc[5, "adsb_surf_taxi_10"], 1.0)   # other1, self excluded
        self.assertAlmostEqual(out.loc[5, "adsb_surf_hold_20"], 1.0)   # other2 stopped off-stand


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
