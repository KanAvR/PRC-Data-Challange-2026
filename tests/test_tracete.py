import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracete import add_trace_encodings


class TraceEncodingTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "ADEP_RUNWAY": ["A_1"] * 5 + ["A_2"],
            "hour": [8.2, 8.7, 8.9, 8.1, 9.0, 8.5],
            "adsb_leave_stand": [600.0, 700.0, 800.0, np.nan, 900.0, 500.0],
        })

    def test_mean_of_the_others_same_runway_hour_day(self):
        X = self.frame()
        add_trace_encodings(X, ["2025-07-01"] * 6)
        # rows 0-3 share runway A_1, hour 8; three seen exits among them
        self.assertTrue(np.isnan(X.loc[0, "trace_rwy_hour"]))      # only 2 others seen
        self.assertEqual(X.loc[0, "trace_rwy_hour_n"], 2)
        self.assertAlmostEqual(X.loc[3, "trace_rwy_hour"], 700.0)   # unseen row: all 3 others
        self.assertEqual(X.loc[3, "trace_rwy_hour_n"], 3)
        self.assertTrue(np.isnan(X.loc[4, "trace_rwy_hour"]))      # hour 9: alone
        self.assertTrue(np.isnan(X.loc[5, "trace_rwy_hour"]))      # other runway

    def test_days_do_not_mix(self):
        X = self.frame()
        add_trace_encodings(X, ["2025-07-01", "2025-07-02", "2025-07-01", "2025-07-01", "2025-07-01", "2025-07-01"])
        self.assertEqual(X.loc[3, "trace_rwy_hour_n"], 2)


if __name__ == "__main__":
    unittest.main()
