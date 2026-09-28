import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import DEST_COLS, _dest_proxy, drop_dest


def ts(minutes):
    return pd.Timestamp("2026-07-05T08:00:00Z") + pd.to_timedelta(minutes, unit="m")


class DestProxyTest(unittest.TestCase):
    def frame(self):
        # three departures to LEMD from different airports, one to EGLL
        mvt = [ts(0), ts(30), ts(90), ts(40)]
        gap_aobt = [1000.0, 3000.0, 700.0, 500.0]
        return pd.DataFrame({
            "ADEP_mvt": ["EDDF", "EHAM", "LSZH", "EDDF"],
            "ADES_mvt": ["LEMD", "LEMD", "LEMD", "EGLL"],
            "MVT_TIME_UTC_mvt": mvt,
            "AOBT_3_flt": [m - pd.to_timedelta(g, unit="s") for m, g in zip(mvt, gap_aobt)],
            "EOBT_1_flt": [m - pd.to_timedelta(900, unit="s") for m in mvt],
        })

    def test_same_destination_across_airports_self_excluded(self):
        out = _dest_proxy(self.frame())
        self.assertEqual(list(out.columns), DEST_COLS)
        # row 1 (t=30, to LEMD): before 60 min = row 0 only (other airport, same destination)
        self.assertAlmostEqual(out.loc[1, "dest_AOBT_3_flt_before_60"], 1000.0)
        # row 1 after 60 min = row 2 (t=90)
        self.assertAlmostEqual(out.loc[1, "dest_AOBT_3_flt_after_60"], 700.0)
        # row 2: before 120 = rows 0 and 1; before 60 = row 1 only
        self.assertAlmostEqual(out.loc[2, "dest_AOBT_3_flt_before_120"], 2000.0)
        self.assertAlmostEqual(out.loc[2, "dest_AOBT_3_flt_before_60"], 3000.0)
        self.assertEqual(out.loc[2, "dest_count_before_120"], 2)
        # the EGLL row has no same-destination neighbours
        self.assertTrue(out.loc[3].isna().all())

    def test_drop(self):
        out = _dest_proxy(self.frame())
        self.assertFalse(set(DEST_COLS) & set(drop_dest(out).columns))


if __name__ == "__main__":
    unittest.main()
