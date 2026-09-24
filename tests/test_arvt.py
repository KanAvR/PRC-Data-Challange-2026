import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import ARVT_COLS, _arvt_features, drop_arvt


def ts(minutes):
    return pd.Timestamp("2025-03-03T08:00:00Z") + pd.to_timedelta(minutes, unit="m")


class ArvtTest(unittest.TestCase):
    def frame(self):
        # two DEP rows on one route, one on another, one ARR
        return pd.DataFrame({
            "PHASE_mvt": ["DEP", "DEP", "DEP", "ARR"],
            "ADEP_mvt": ["EDDF", "EDDF", "EDDF", "EGLL"],
            "ADES_mvt": ["EGLL", "EGLL", "LIRF", "EDDF"],
            "MVT_TIME_UTC_mvt": [ts(20), ts(30), ts(25), ts(0)],
            "EOBT_1_flt": [ts(0), ts(5), ts(0), ts(-90)],
            "AOBT_3_flt": [ts(2), ts(8), ts(1), ts(-88)],
            "ARVT_1_flt": [ts(100), ts(110), ts(140), ts(-5)],
            "ARVT_3_flt": [ts(105), ts(125), ts(150), ts(0)],
        })

    def test_values_and_route_medians(self):
        out = _arvt_features(self.frame())
        self.assertEqual(list(out.columns), ARVT_COLS)
        # airborne: 85 and 95 min on EDDF-EGLL (median 90), 125 on EDDF-LIRF
        np.testing.assert_allclose(out["arvt_air"].values[:3], [85 * 60, 95 * 60, 125 * 60])
        np.testing.assert_allclose(out["arvt_air_dev"].values[:3], [-300, 300, 0])
        # planned block minus the route's median airborne time
        np.testing.assert_allclose(out["arvt_plan_block"].values[:3], [(100 - 90) * 60, (105 - 90) * 60, (140 - 125) * 60])
        # arrival delay minus off-block delay: (5 - 2), (15 - 3), (10 - 1) min
        np.testing.assert_allclose(out["arvt_dly_diff"].values[:3], [3 * 60, 12 * 60, 9 * 60])

    def test_arrivals_left_empty(self):
        out = _arvt_features(self.frame())
        self.assertTrue(out.iloc[3].isna().all())

    def test_drop(self):
        out = drop_arvt(pd.concat([self.frame(), _arvt_features(self.frame())], axis=1))
        self.assertFalse(set(ARVT_COLS) & set(out.columns))


if __name__ == "__main__":
    unittest.main()
