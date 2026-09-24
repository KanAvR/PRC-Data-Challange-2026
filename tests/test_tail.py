import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import (DAY, DAY_LATE_BASE, DAY_STEP_EDGES, V_HI_RATIO, combine,
                   day_late_step_weight, fit_day_late_steps)


class TailCapTest(unittest.TestCase):
    def setUp(self):
        self.mu = np.full(5, 1500.)
        self.p = np.array([0.9, 0.9, 0.9, 0.9, 0.0])
        self.v = np.array([3000., 50000., 80000., 112000., 90000.])

    def test_defaults_unchanged(self):
        """No new argument: the pre-existing formula, bit for bit."""
        v_h = np.where(self.v >= 40000, self.v * V_HI_RATIO, self.v)
        v_g = np.where(v_h > 6000, v_h, self.mu)
        expected = np.clip(0.1 * self.mu + 0.9 * v_g, 0, None)
        expected[4] = self.mu[4]
        np.testing.assert_array_equal(combine(self.mu, self.p, self.v), expected)

    def test_cap_only_binds_above_both_labels(self):
        plain = combine(self.mu, self.p, self.v)
        capped = combine(self.mu, self.p, self.v, cap=True)
        # below ~81k the hedge stays under DAY + base: untouched
        np.testing.assert_array_equal(capped[:3], plain[:3])
        # 112k: the label is v or DAY + base, both below 1.08 v; the tail becomes v
        self.assertAlmostEqual(capped[3], 0.1 * 1500 + 0.9 * 112000.)
        self.assertTrue((capped <= plain).all())

    def test_day_value_overrides_day_plus_mu(self):
        q = np.array([0, 0, 0.5, 0, 0])
        out = combine(self.mu, self.p, self.v, q_day=q, day_value=DAY + DAY_LATE_BASE)
        self.assertAlmostEqual(out[2], 0.1 * 1500 + 0.9 * (0.5 * 80000 + 0.5 * (DAY + DAY_LATE_BASE)))


class DayLateStepTest(unittest.TestCase):
    def frame(self):
        lo, mid, hi = DAY_STEP_EDGES
        gs = np.array([lo + 1000, lo + 2000, lo + 3000, mid + 1000, mid + 2000, 20000., hi + 5000, lo + 4000])
        lobt = np.array([np.nan] * 7 + [5000.])
        y = np.array([DAY + 900, gs[1], gs[2], DAY + 800, DAY + 1100, 20000., gs[6], DAY + 900])
        return pd.DataFrame({"gap_SCHED_TIME_UTC_mvt": gs, "gap_LOBT_flt": lobt}), y

    def test_rates_are_smoothed_per_step(self):
        X, y = self.frame()
        low, high = fit_day_late_steps(X, y)
        self.assertAlmostEqual(low, (1 + 1) / (3 + 2))   # 1 of 3 candidates day-late
        self.assertAlmostEqual(high, (2 + 1) / (2 + 2))  # 2 of 2

    def test_weight_only_on_unmatched_candidates(self):
        X, y = self.frame()
        q = day_late_step_weight({"q_day_steps": (0.3, 0.8)}, X)
        np.testing.assert_array_equal(q, [0.3, 0.3, 0.3, 0.8, 0.8, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
