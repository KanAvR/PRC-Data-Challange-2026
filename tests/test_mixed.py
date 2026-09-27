import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import MIXED_COLS, _arr_same_runway, drop_mixed


def ts(minutes):
    return pd.Timestamp("2026-01-05T08:00:00Z") + pd.to_timedelta(minutes, unit="m")


class MixedModeTest(unittest.TestCase):
    def frame(self):
        # DEP at 0 on 27; landings on 27 at -8, -2, +4, +30 and one on 09 at -1
        return pd.DataFrame({
            "PHASE_mvt": ["DEP", "ARR", "ARR", "ARR", "ARR", "ARR", "DEP"],
            "RUNWAY_mvt": ["27", "27", "27", "27", "27", "09", "18"],
            "MVT_TIME_UTC_mvt": [ts(0), ts(-8), ts(-2), ts(4), ts(30), ts(-1), ts(0)],
        })

    def test_counts_and_gaps_same_runway_only(self):
        out = _arr_same_runway(self.frame())
        self.assertEqual(list(out.columns), MIXED_COLS)
        self.assertEqual(out.loc[0, "arr_rwy_10m"], 2)        # -8, -2 (not the 09 at -1)
        self.assertEqual(out.loc[0, "arr_rwy_next_10m"], 1)   # +4
        self.assertAlmostEqual(out.loc[0, "secs_since_arr_rwy"], 120)
        self.assertAlmostEqual(out.loc[0, "secs_to_next_arr_rwy"], 240)

    def test_runway_without_landings(self):
        out = _arr_same_runway(self.frame())
        self.assertEqual(out.loc[6, "arr_rwy_10m"], 0)
        self.assertTrue(np.isnan(out.loc[6, "secs_since_arr_rwy"]))

    def test_arr_rows_left_nan_and_drop(self):
        out = _arr_same_runway(self.frame())
        self.assertTrue(out.loc[1:5].isna().all().all())
        self.assertFalse(set(MIXED_COLS) & set(drop_mixed(out).columns))


if __name__ == "__main__":
    unittest.main()
