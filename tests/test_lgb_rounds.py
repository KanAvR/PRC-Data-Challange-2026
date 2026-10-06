import argparse
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model


class LgbRoundsTest(unittest.TestCase):
    def tearDown(self):
        model.LGB_MU_ROUNDS = 1200

    def parse(self, argv):
        ap = argparse.ArgumentParser()
        model.add_model_args(ap)
        args = ap.parse_args(argv)
        model.apply_model_args(args)
        return args

    def test_defaults_follow_v33(self):
        args = self.parse([])
        self.assertEqual(model.LGB_MU_ROUNDS, 1200)
        self.assertEqual(model.CAT_MU_PARAMS["iterations"], 10000)
        self.assertTrue(args.adsb_climb_id and args.clf_inbound)

    def test_flag_sets_rounds_used_by_train_mu(self):
        self.parse(["--lgb-rounds", "7"])
        self.assertEqual(model.LGB_MU_ROUNDS, 7)
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=200)})
        y = 600 + 100 * X["a"].values
        saved = model.CAT_COLS
        model.CAT_COLS = []
        try:
            booster = model.train_mu(X, y)
        finally:
            model.CAT_COLS = saved
        self.assertEqual(booster.current_iteration(), 7)

    def test_linear_tree_extrapolates_a_gap(self):
        self.parse(["--lgb-linear-tree", "--lgb-rounds", "400"])
        try:
            rng = np.random.default_rng(1)
            X = pd.DataFrame({"gap": rng.uniform(300, 2000, 3000)})
            y = X["gap"].values + 60.0
            saved = model.CAT_COLS
            model.CAT_COLS = []
            try:
                m = model.train_mu(X, y)
            finally:
                model.CAT_COLS = saved
            pred = m.predict(pd.DataFrame({"gap": [1000.0, 2500.0, 1e6]}))
            self.assertAlmostEqual(pred[0], 1060.0, delta=20)
            self.assertGreater(pred[1], 2300.0)          # beyond the training range
            self.assertEqual(pred[2], model.LINEAR_CLIP)  # clipped
        finally:
            model.LGB_LINEAR_TREE = False

    def test_pa_cat_iterations(self):
        import per_airport
        try:
            self.parse([])
            self.assertEqual(per_airport.CAT_ITERATIONS, 2000)
            self.parse(["--pa-cat-iterations", "5000"])
            self.assertEqual(per_airport.CAT_ITERATIONS, 5000)
        finally:
            per_airport.CAT_ITERATIONS = 2000


if __name__ == "__main__":
    unittest.main()
