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


if __name__ == "__main__":
    unittest.main()
