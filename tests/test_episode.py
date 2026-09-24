import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import EPISODE_CLIP, EPISODE_COLS, _episode, drop_episode


def ts(minutes):
    return pd.Timestamp("2026-01-05T08:00:00Z") + pd.to_timedelta(minutes, unit="m")


class EpisodeTest(unittest.TestCase):
    def frame(self):
        mvt = [ts(0), ts(10), ts(20), ts(25), ts(200)]
        gaps = [8000., 1000., np.nan, 90000., 7000.]  # seconds before take-off
        lobt = [m - pd.to_timedelta(g, unit="s") if np.isfinite(g) else pd.NaT for m, g in zip(mvt, gaps)]
        return pd.DataFrame({"MVT_TIME_UTC_mvt": mvt, "LOBT_flt": pd.to_datetime(lobt, utc=True)})

    def test_share_and_mean_self_excluded(self):
        out = _episode(self.frame())
        self.assertEqual(list(out.columns), EPISODE_COLS)
        # row 1 (t=10): matched neighbours within 30 min are rows 0 and 3 (row 2 unmatched);
        # both over 6000s; gaps 8000 and 90000 clipped to EPISODE_CLIP
        self.assertAlmostEqual(out.loc[1, "ep_long_share_30"], 1.0)
        self.assertAlmostEqual(out.loc[1, "ep_gap_mean_30"], (8000 + EPISODE_CLIP) / 2)
        # row 2 is unmatched but still gets its neighbours' view: rows 0, 1, 3
        self.assertAlmostEqual(out.loc[2, "ep_long_share_30"], 2 / 3)
        # row 4 is alone within 60 min: no neighbours
        self.assertTrue(np.isnan(out.loc[4, "ep_long_share_60"]))

    def test_drop(self):
        f = pd.concat([self.frame(), _episode(self.frame())], axis=1)
        self.assertFalse(set(EPISODE_COLS) & set(drop_episode(f).columns))


if __name__ == "__main__":
    unittest.main()
