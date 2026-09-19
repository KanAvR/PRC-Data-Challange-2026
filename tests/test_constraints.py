import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constraints import check_lobt_support, project_lobt
from constrain_submission import constrain_submission


class ConstraintsTest(unittest.TestCase):
    def test_new_training_data_cannot_silently_break_the_assumption(self):
        check_lobt_support([1000, 87000, 2000], [1500, 87001, np.nan])
        with self.assertRaises(ValueError):
            check_lobt_support([1000], [10000])

    def test_projection_never_worsens_error_for_supported_targets(self):
        rng = np.random.default_rng(17)
        gap = rng.uniform(-3500, 140000, 10000)
        low, high = np.maximum(0, gap - 3660), gap + 3660
        target = rng.uniform(low, high)
        pred = rng.uniform(0, 140000, len(gap))
        adjusted = project_lobt(pred, gap)
        self.assertTrue(np.all((adjusted - target) ** 2 <= (pred - target) ** 2 + 1e-6))
        np.testing.assert_array_equal(project_lobt(adjusted, gap), adjusted)

    def test_missing_or_impossible_match_is_unchanged(self):
        pred = np.array([1000., 2000., 3000., 4000.])
        np.testing.assert_array_equal(project_lobt(pred, [np.nan, np.inf, -np.inf, -4000]), pred)

    def test_submission_joins_by_id_and_preserves_template_order(self):
        template = pd.DataFrame({"MVT_ID_mvt": [2., 1.]})
        submission = pd.DataFrame({"MVT_ID_mvt": [1., 2.], "TAXITIME_SEC_mvt": [1000., 2000.]})
        ranking = pd.DataFrame({"MVT_ID_mvt": [1., 2.], "PHASE_mvt": ["DEP", "DEP"],
                                "ADEP_mvt": ["EHAM", "LIRF"],
                                "MVT_TIME_UTC_mvt": pd.to_datetime(["2026-01-01T03:00Z"] * 2),
                                "LOBT_flt": pd.to_datetime(["2026-01-01T00:00Z", None])})
        out, changes = constrain_submission(submission, ranking, template)
        np.testing.assert_array_equal(out["MVT_ID_mvt"], [2., 1.])
        np.testing.assert_array_equal(out["TAXITIME_SEC_mvt"], [2000., 7140.])
        self.assertEqual(changes["MVT_ID_mvt"].tolist(), [1.])
        with self.assertRaises(ValueError):
            constrain_submission(pd.concat([submission, submission.iloc[:1]]), ranking, template)
        with self.assertRaises(ValueError):
            constrain_submission(submission.iloc[:1], ranking, template)


if __name__ == "__main__":
    unittest.main()
