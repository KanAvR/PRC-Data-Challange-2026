import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resproxy import RES_PROXY_COLS, add_residual_proxy


class ResProxyTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "ADEP_mvt": ["A", "A", "A", "B"],
            "RUNWAY_mvt": ["R1", "R1", "R2", "R1"],
            "gap_AOBT_3_flt": [900., 1200., 700., 800.],
            "gap_LOBT_flt": [1000., np.nan, 9000., 800.],
            "te_stand_rwy": [800., 1000., 600., 500.],
        })

    def test_residuals_self_excluded_and_scoped(self):
        X = self.frame()
        t = np.array([0., 600., 900., 600.])  # seconds
        add_residual_proxy(X, t)
        self.assertEqual(set(RES_PROXY_COLS) - set(X.columns), set())
        # row 1 (t=600): airport neighbours before it within 30 min = row 0 only
        # (residual 900-800 = 100); row 3 is another airport, and row 1 is not its own neighbour
        self.assertAlmostEqual(X.loc[1, "resproxy_apt_gap_AOBT_3_flt_before_30"], 100.0)
        # row 0: after it, rows 1 (+200) and 2 (+100) at the airport; on runway R1 only row 1
        self.assertAlmostEqual(X.loc[0, "resproxy_apt_gap_AOBT_3_flt_after_30"], 150.0)
        self.assertAlmostEqual(X.loc[0, "resproxy_rwy_gap_AOBT_3_flt_after_30"], 200.0)
        # LOBT gap of 9000 exceeds PROXY_MAX_SEC and row 1 has none: row 0 sees no LOBT neighbour after it
        self.assertTrue(np.isnan(X.loc[0, "resproxy_apt_gap_LOBT_flt_after_30"]))
        # airport B has a single row: nothing to average
        self.assertTrue(np.isnan(X.loc[3, "resproxy_apt_gap_AOBT_3_flt_before_60"]))


if __name__ == "__main__":
    unittest.main()
