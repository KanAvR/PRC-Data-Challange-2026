import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from high_lobt import HighLobtModel
from constraints import project_lobt
from model import combine


class ConstantResidual:
    def predict(self, X):
        return np.full(len(X), -1000.0)


class HighLobtTest(unittest.TestCase):
    def test_skipping_pooled_classifier_preserves_final_predictions(self):
        X = pd.DataFrame({"gap_LOBT_flt": [5000., 7000., 87000., np.nan]})
        mu = np.array([2000., 4000., 3000., 1000.])
        v = np.array([5000., 7000., 87000., 12000.])
        specialist = HighLobtModel(ConstantResidual(), ["gap_LOBT_flt"])
        results = []
        for p in ([0.9, 0.1, 0.7, 0.4], [0., 0., 0., 0.4]):
            pred = project_lobt(combine(mu, np.array(p), v), X["gap_LOBT_flt"])
            results.append(specialist.refine(pred, X))
        np.testing.assert_array_equal(*results)

    def test_long_times_extrapolate_without_changing_other_rows(self):
        X = pd.DataFrame({"gap_LOBT_flt": [np.nan, 1000., 6000., 7000., 87000.]},
                         index=[50, 40, 30, 20, 10])
        before = np.array([1200., 800., 4500., 4000., 61000.])
        model = HighLobtModel(ConstantResidual(), ["gap_LOBT_flt"])
        after = model.refine(before, X)
        np.testing.assert_array_equal(after[:3], before[:3])
        np.testing.assert_array_equal(after[3:], [6000., 86000.])
        np.testing.assert_array_equal(before, [1200., 800., 4500., 4000., 61000.])


if __name__ == "__main__":
    unittest.main()
