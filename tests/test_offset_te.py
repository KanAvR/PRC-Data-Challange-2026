import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from encode import OFFSET_COLS, PRIOR_WEIGHT, add_offset_encodings


class OffsetEncodingTest(unittest.TestCase):
    def test_out_of_fold_offset_and_corrected_estimate(self):
        # two months; stand A offsets -100 in both; rows without an exit are ignored
        X_fit = pd.DataFrame({"ADEP_mvt": ["LEMD"] * 5, "ADEP_STAND": ["A", "A", "B", "A", "B"],
                              "ADEP_AIRLINE": ["IBE"] * 5,
                              "adsb_leave_stand": [700.0, 900.0, np.nan, 800.0, 5000.0]})
        y = np.array([600.0, 800.0, 1200.0, 700.0, 9000.0])     # last row is a tail row
        folds = np.array(["m1", "m1", "m1", "m2", "m2"])
        X_app = pd.DataFrame({"ADEP_mvt": ["LEMD"], "ADEP_STAND": ["A"], "ADEP_AIRLINE": ["IBE"],
                              "adsb_leave_stand": [1000.0]})
        add_offset_encodings(X_fit, y, folds, X_app)
        for c in OFFSET_COLS:
            self.assertIn(c, X_fit.columns)
            self.assertIn(c, X_app.columns)
        # every usable offset is -100, so every level and the prior agree
        self.assertAlmostEqual(X_app.loc[0, "te_off_apt"], -100.0)
        self.assertAlmostEqual(X_app.loc[0, "te_off_stand"], -100.0)
        self.assertAlmostEqual(X_app.loc[0, "adsb_exit_off_stand"], 900.0)
        # month m2's row sees only m1's offsets (out of fold): still -100
        self.assertAlmostEqual(X_fit.loc[3, "te_off_stand"], -100.0)


if __name__ == "__main__":
    unittest.main()
