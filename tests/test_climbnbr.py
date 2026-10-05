import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adsb import NBR_COLS
from climbnbr import clean_neighbours


class CleanNeighboursTest(unittest.TestCase):
    def data(self):
        t0 = pd.Timestamp("2025-07-01 10:00", tz="UTC")
        raw = pd.DataFrame({
            "PHASE_mvt": ["DEP"] * 3 + ["ARR"],
            "ADEP_mvt": ["EDDF"] * 4,
            "RUNWAY_mvt": ["18"] * 4,
            "MVT_TIME_UTC_mvt": [t0, t0 + pd.Timedelta(minutes=5), t0 + pd.Timedelta(minutes=10), t0],
        })
        feats = pd.DataFrame(np.nan, index=raw.index,
                             columns=["adsb_leave_stand", "adsb_push", "adsb_first_seen",
                                      "adsb_via_climb", "adsb_to_dt"] + NBR_COLS)
        # row 0: a departure segment; row 1: inbound taxi-in found by the climb-out
        feats.loc[0, ["adsb_leave_stand", "adsb_first_seen", "adsb_via_climb", "adsb_to_dt"]] = [600, 700, 0, -10]
        feats.loc[1, ["adsb_leave_stand", "adsb_first_seen", "adsb_via_climb", "adsb_to_dt"]] = [900, 5000, 1, -4000]
        return raw, feats

    def test_inbound_rows_leave_the_neighbour_means_but_keep_their_own(self):
        raw, feats = self.data()
        out = clean_neighbours(raw, feats)
        # row 2 sees only row 0 among its predecessors now
        self.assertAlmostEqual(out.loc[2, "adsb_nbr_first_apt_prev30"], 700.0)
        self.assertAlmostEqual(out.loc[2, "adsb_nbr_exit_apt_prev30"], 600.0)
        self.assertEqual(out.loc[1, "adsb_first_seen"], 5000)   # own columns untouched
        self.assertTrue(np.isnan(out.loc[3, "adsb_nbr_first_apt_prev30"]))  # ARR row

    def test_departure_climb_matches_still_count(self):
        raw, feats = self.data()
        feats.loc[1, "adsb_to_dt"] = -60          # its last segment ends at take-off
        out = clean_neighbours(raw, feats)
        self.assertAlmostEqual(out.loc[2, "adsb_nbr_exit_apt_prev30"], 750.0)


if __name__ == "__main__":
    unittest.main()
