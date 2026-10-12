import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constraints import LOBT_MARGIN_SEC
from tracespec import apply_trace_spec, fit_trace_trust


def frame(apt, exit_s, gap):
    return pd.DataFrame({"ADEP_mvt": pd.Categorical(apt), "adsb_leave_stand": exit_s,
                         "gap_LOBT_flt": np.asarray(gap, dtype=float)})


class TraceTrustTest(unittest.TestCase):
    def setUp(self):
        # GOOD: offset 10s, long exits land on the label; BAD: long exits miss by 2000s
        n = 20
        ex_norm = np.linspace(500, 1500, n)
        ex_long = np.linspace(3700, 5500, 6)
        apt = ["GOOD"] * n + ["GOOD"] * 6 + ["BAD"] * n + ["BAD"] * 6
        ex = np.r_[ex_norm, ex_long, ex_norm, ex_long]
        y = np.r_[ex_norm + 10, ex_long + 10, ex_norm + 200, ex_long - 2000]
        self.X = frame(apt, ex, np.full(len(apt), np.nan))
        self.y = y

    def test_gate_and_offset(self):
        t = fit_trace_trust(self.X, self.y)
        self.assertEqual(t.reliable, ["GOOD"])
        self.assertAlmostEqual(t.offset["GOOD"], 10.0)
        self.assertAlmostEqual(t.offset["BAD"], 200.0)

    def test_too_few_long_exits_not_trusted(self):
        X, y = self.X.iloc[:23], self.y[:23]  # 3 long exits at GOOD
        self.assertEqual(fit_trace_trust(X, y).reliable, [])

    def test_apply_only_specialist_rows_trusted_airport_with_exit(self):
        t = fit_trace_trust(self.X, self.y)
        Xa = frame(["GOOD", "GOOD", "GOOD", "BAD"], [4000.0, np.nan, 4000.0, 4000.0],
                   [6500.0, 6500.0, 6500.0, 6500.0])
        pred = np.array([5000.0, 5000.0, 5000.0, 5000.0])
        rows = np.array([True, True, False, True])
        out, used = apply_trace_spec(pred, Xa, t, rows)
        np.testing.assert_array_equal(used, [True, False, False, False])
        np.testing.assert_allclose(out, [4010.0, 5000.0, 5000.0, 5000.0])

    def test_projected_into_lobt_window(self):
        t = fit_trace_trust(self.X, self.y)
        Xa = frame(["GOOD"], [1000.0], [9000.0])  # window starts at 9000 - margin
        out, used = apply_trace_spec(np.array([7000.0]), Xa, t, np.array([True]))
        self.assertTrue(used[0])
        self.assertGreaterEqual(out[0], 9000.0 - LOBT_MARGIN_SEC - 1e-6)

    def test_no_trace_columns_is_a_no_op(self):
        X = pd.DataFrame({"ADEP_mvt": ["GOOD"], "gap_LOBT_flt": [6500.0]})
        t = fit_trace_trust(X, np.array([4000.0]))
        out, used = apply_trace_spec(np.array([5000.0]), X, t, np.array([True]))
        self.assertFalse(used.any())
        self.assertEqual(out[0], 5000.0)


if __name__ == "__main__":
    unittest.main()
