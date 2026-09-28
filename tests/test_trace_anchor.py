import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import TAIL_THR, trace_anchor, trace_anchored


class _Mean:
    """A stand-in family: predicts the mean fit target over normal rows."""
    seen = None

    def __init__(self, X, y):
        normal = y <= TAIL_THR
        _Mean.seen = (normal.copy(), y.copy())
        self.m = y[normal].mean()

    def predict(self, X):
        return np.full(len(X), self.m)


class TraceAnchorTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"adsb_leave_stand": [600.0, np.nan, 5000.0, -20.0]})

    def test_anchor_clipped_and_missing_is_zero(self):
        a = trace_anchor(self.frame())
        np.testing.assert_allclose(a, [600.0, 0.0, 3600.0, 0.0])
        self.assertTrue((trace_anchor(pd.DataFrame({"x": [1, 2]})) == 0).all())

    def test_residual_target_keeps_the_tail_mask(self):
        X = self.frame()
        y = np.array([650.0, 900.0, 9000.0, 700.0])   # row 2 is a tail row
        model = trace_anchored(lambda X, y: _Mean(X, y))(X, y)
        normal, y_fit = _Mean.seen
        np.testing.assert_array_equal(normal, [True, True, False, True])
        np.testing.assert_allclose(y_fit, [50.0, 900.0, 9000.0, 700.0])
        # prediction = mean residual + anchor
        m = (50.0 + 900.0 + 700.0) / 3
        np.testing.assert_allclose(model.predict(X), [m + 600, m, m + 3600, m])


if __name__ == "__main__":
    unittest.main()
