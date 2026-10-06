import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model


class SchedMixTest(unittest.TestCase):
    def test_copies_follow_gap_sched(self):
        rng = np.random.default_rng(0)
        n = 4000
        flag = rng.random(n) < 0.3                       # observable copy indicator
        g = rng.uniform(1500, 5000, n)
        X = pd.DataFrame({"gap_SCHED_TIME_UTC_mvt": g, "flag": flag.astype(float)})
        y = np.where(flag, g, 800 + rng.normal(0, 30, n))
        saved = model.CAT_COLS
        model.CAT_COLS = []
        try:
            m = model.train_mu_schedmix(X, y, num_boost_round=200, clf_rounds=100)
        finally:
            model.CAT_COLS = saved
        pred = m.predict(X)
        # copies are reproduced exactly through gap_SCHED, the rest by the regressor
        self.assertLess(np.abs(pred[flag] - g[flag]).mean(), 50)
        self.assertLess(np.abs(pred[~flag] - 800).mean(), 60)

    def test_lirf_only_wrapper(self):
        rng = np.random.default_rng(2)
        n = 6000
        apt = np.where(rng.random(n) < 0.5, "LIRF", "EDDF")
        flag = (apt == "LIRF") & (rng.random(n) < 0.3)
        g = rng.uniform(1500, 5000, n)
        X = pd.DataFrame({"gap_SCHED_TIME_UTC_mvt": g, "flag": flag.astype(float),
                          "ADEP_mvt": pd.Categorical(apt)})
        y = np.where(flag, g, 800 + rng.normal(0, 30, n))
        seen = {}

        def fake_family(X, y):
            seen["n_normal"] = int((y <= model.TAIL_THR).sum())
            return type("M", (), {"predict": lambda s, X: np.full(len(X), 800.0)})()

        saved = model.CAT_COLS
        model.CAT_COLS = ["ADEP_mvt"]
        try:
            m = model.lirf_schedmixed(fake_family, clf_rounds=100)(X, y)
        finally:
            model.CAT_COLS = saved
        self.assertEqual(seen["n_normal"], n - flag.sum())      # copies left out of the fit
        pred = m.predict(X)
        np.testing.assert_allclose(pred[apt == "EDDF"], 800.0)  # other airports untouched
        self.assertLess(np.abs(pred[flag] - g[flag]).mean(), 100)

    def test_missing_sched_uses_regressor(self):
        mix = model._SchedMixMu(type("C", (), {"predict": lambda s, X: np.full(len(X), 0.9)})(),
                                type("R", (), {"predict": lambda s, X: np.full(len(X), 700.0)})())
        X = pd.DataFrame({"gap_SCHED_TIME_UTC_mvt": [np.nan, 2000.0]})
        np.testing.assert_allclose(mix.predict(X), [700.0, 0.9 * 2000 + 0.1 * 700])


if __name__ == "__main__":
    unittest.main()
