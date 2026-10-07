import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adsb import ZERO_WHEN_UNTRACED, blank_untraced, day_dropout_rows


class BlankUntracedTest(unittest.TestCase):
    def test_rows_look_like_july_2026_lirf(self):
        f = pd.DataFrame({"adsb_match": [1.0, 1.0], "adsb_leave_stand": [300.0, 400.0],
                          "adsb_cov_apt_day": [0.9, 0.9], "gap_LOBT_flt": [100.0, 200.0]})
        blank_untraced(f, [True, False])
        self.assertEqual(f.loc[0, "adsb_match"], 0.0)
        self.assertEqual(f.loc[0, "adsb_cov_apt_day"], 0.0)
        self.assertTrue(np.isnan(f.loc[0, "adsb_leave_stand"]))
        self.assertEqual(f.loc[0, "gap_LOBT_flt"], 100.0)          # non-adsb untouched
        self.assertEqual(f.loc[1].tolist(), [1.0, 400.0, 0.9, 200.0])  # unselected row untouched
        self.assertIn("adsb_match", ZERO_WHEN_UNTRACED)

    def test_nothing_selected_is_a_no_op(self):
        f = pd.DataFrame({"adsb_match": [1.0]})
        blank_untraced(f, [False])
        self.assertEqual(f.loc[0, "adsb_match"], 1.0)


class DayDropoutTest(unittest.TestCase):
    def test_whole_airport_days_at_about_the_rate(self):
        days = pd.date_range("2025-01-01", periods=365).strftime("%Y-%m-%d")
        apt = np.repeat(["EHAM", "LIRF"], len(days) * 3)
        date = np.tile(np.repeat(days, 3), 2)
        d = day_dropout_rows(apt, date, 0.2)
        g = pd.Series(d).groupby([apt, date]).agg(["min", "max"])
        self.assertTrue((g["min"] == g["max"]).all())               # a day goes whole
        self.assertAlmostEqual(g["min"].mean(), 0.2, delta=0.05)
        np.testing.assert_array_equal(d, day_dropout_rows(apt, date, 0.2))  # stable
        self.assertFalse(day_dropout_rows(apt, date, 0.0).any())


if __name__ == "__main__":
    unittest.main()
