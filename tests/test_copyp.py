import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import copyp
import model


class CopyPTest(unittest.TestCase):
    def frame(self, n, rng):
        stand = rng.choice(["A", "B"], n)
        g = rng.uniform(2000, 5000, n)
        X = pd.DataFrame({"gap_SCHED_TIME_UTC_mvt": g, "STAND_mvt": pd.Categorical(stand),
                          "secs_since_prev_arr_block": rng.uniform(0, 9000, n),
                          "hour": rng.integers(0, 24, n).astype(float)})
        copy = (stand == "A") & (rng.random(n) < 0.6)
        y = np.where(copy, g, 900.0)
        return X, y, copy

    def test_learns_the_habit_out_of_month(self):
        rng = np.random.default_rng(0)
        X, y, copy = self.frame(6000, rng)
        months = np.repeat(["2025-01", "2025-02", "2025-03"], 2000)
        Xa, _, _ = self.frame(500, rng)
        saved = model.CAT_COLS, copyp.CAT_COLS
        model.CAT_COLS = copyp.CAT_COLS = ["STAND_mvt"]
        try:
            copyp.add_copy_p(X, y, months, Xa)
        finally:
            model.CAT_COLS, copyp.CAT_COLS = saved
        stand_a = (X["STAND_mvt"] == "A").values
        self.assertGreater(X[copyp.COL][stand_a].mean(), 0.45)
        self.assertLess(X[copyp.COL][~stand_a].mean(), 0.1)
        self.assertTrue(np.isfinite(Xa[copyp.COL]).all())
        # the probability ignores gap_SCHED itself: the habit, not the delay
        self.assertNotIn("gap_SCHED_TIME_UTC_mvt", copyp.FEATURES)

    def test_tail_classifier_sees_it_only_when_present(self):
        X = pd.DataFrame({c: [0.0] for c in model.NO_LOBT_FEATURES})
        self.assertNotIn(copyp.COL, model._usable(model.NO_LOBT_FEATURES + [copyp.COL], X))
        X[copyp.COL] = 0.5
        self.assertIn(copyp.COL, model._usable(model.NO_LOBT_FEATURES + [copyp.COL], X))


if __name__ == "__main__":
    unittest.main()
