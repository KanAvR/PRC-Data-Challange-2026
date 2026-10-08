import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from service import COLS, service_features


def frame():
    t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
    return pd.DataFrame({
        "PHASE_mvt": ["DEP", "DEP", "DEP", "ARR", "DEP"],
        "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "EGLL", "LIRF"],
        "ADES_mvt": ["LIMC", "EGLL", "LIMC", "LIRF", None],
        "FLIGHT_mvt": ["ITY100", "ITY100", "ITYTY680", "BAW1", "AEZ12AB"],
        "MVT_TIME_UTC_mvt": [t, t + pd.Timedelta(hours=23), t, t, t + pd.Timedelta(hours=30)],
    })


class ServiceFeaturesTest(unittest.TestCase):
    def test_columns(self):
        f = service_features(frame())
        self.assertEqual(list(f.columns), COLS)
        self.assertEqual(f.loc[0, "svc_domestic"], 1.0)              # LIRF -> LIMC
        self.assertEqual(f.loc[1, "svc_domestic"], 0.0)              # LIRF -> EGLL
        self.assertEqual(f.loc[1, "svc_ades_country"], "EG")
        self.assertEqual(f.loc[4, "svc_ades_country"], "missing")
        self.assertEqual(f.loc[0, "svc_n_same_flight_24h"], 1.0)     # ITY100 again 23h later
        self.assertEqual(f.loc[2, "svc_n_same_flight_24h"], 0.0)
        self.assertEqual(f.loc[2, "svc_cs_long_prefix"], 1.0)        # ITYTY680
        self.assertEqual(f.loc[0, "svc_cs_long_prefix"], 0.0)
        self.assertEqual(f.loc[4, "svc_cs_alpha_suffix"], 1.0)       # AEZ12AB
        self.assertTrue(np.isnan(f.loc[3, "svc_domestic"]))          # ARR rows stay empty


class ArrivalHabitTest(unittest.TestCase):
    def test_operator_and_day_shares(self):
        t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
        rows = []
        for i in range(10):   # ISR arrivals: all block == sched; ITY: none
            rows.append(("ARR", "LLBG", "LIRF", f"ISR{i}", t, t, t))
            rows.append(("ARR", "LIMC", "LIRF", f"ITY{i}", t, t + pd.Timedelta(minutes=5), t))
        rows.append(("DEP", "LIRF", "LLBG", "ISR99", t, None, t + pd.Timedelta(hours=3)))
        rows.append(("DEP", "LIRF", "LIMC", "ITY99", t, None, t + pd.Timedelta(hours=3)))
        df = pd.DataFrame(rows, columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt",
                                         "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"])
        from service import ARR_COLS, ARR_PRIOR_N, arrival_habit
        f = arrival_habit(df)
        self.assertEqual(list(f.columns), ARR_COLS)
        isr, ity = f.iloc[-2], f.iloc[-1]
        self.assertAlmostEqual(isr["svc_op_arr_copy"], (10 + ARR_PRIOR_N * 0.5) / (10 + ARR_PRIOR_N))
        self.assertAlmostEqual(ity["svc_op_arr_copy"], (0 + ARR_PRIOR_N * 0.5) / (10 + ARR_PRIOR_N))
        self.assertAlmostEqual(isr["svc_day_arr_copy"], 0.5)
        self.assertTrue(np.isnan(f.iloc[0]["svc_op_arr_copy"]))   # ARR rows stay empty


class CrossMidnightTest(unittest.TestCase):
    def test_local_dates(self):
        from service import cross_midnight
        s = pd.Timestamp("2025-07-01 21:30", tz="UTC")             # 23:30 in Rome
        df = pd.DataFrame({
            "PHASE_mvt": ["DEP", "DEP", "ARR"],
            "ADEP_mvt": ["LIRF", "LIRF", "EGLL"],
            "SCHED_TIME_UTC_mvt": [s, s, s],
            "MVT_TIME_UTC_mvt": [s + pd.Timedelta(hours=1), s + pd.Timedelta(minutes=20), s],
        })
        c = cross_midnight(df)
        self.assertEqual(c.iloc[0], 1.0)   # 00:30 local next day
        self.assertEqual(c.iloc[1], 0.0)   # 23:50 local same day
        self.assertTrue(np.isnan(c.iloc[2]))


class DayLateCrossTest(unittest.TestCase):
    def test_rate_and_weight_only_on_crossing_rows(self):
        import model
        X = pd.DataFrame({"gap_SCHED_TIME_UTC_mvt": [60000.0] * 4, "gap_LOBT_flt": [np.nan] * 4,
                          "svc_cross_midnight": [1.0, 1.0, 0.0, 0.0]})
        y = np.array([87000.0, 60010.0, 60005.0, 87100.0])
        # crossing tail candidates: rows 0 (day-late) and 1 -> (1 + 1) / (2 + 2)
        self.assertAlmostEqual(model.fit_day_late_rate(X, y), 0.5)
        w = model.day_late_weight({"q_day": 0.5}, X)
        np.testing.assert_array_equal(w, [0.5, 0.5, 0.0, 0.0])
        # without the gate column every in-window candidate counts, as before
        Xn = X.drop(columns="svc_cross_midnight")
        self.assertAlmostEqual(model.fit_day_late_rate(Xn, y), (2 + 1) / (4 + 2))


if __name__ == "__main__":
    unittest.main()
