import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subset_submission import subset


class SubsetTest(unittest.TestCase):
    def setUp(self):
        self.pred = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0],
                                  "TAXITIME_SEC_mvt": [100.0, 200.0, 300.0, 400.0]})

    def test_template_rows_in_template_order(self):
        out = subset(self.pred, pd.DataFrame({"MVT_ID_mvt": [3.0, 1.0], "TAXITIME_SEC_mvt": np.nan}))
        self.assertEqual(out["MVT_ID_mvt"].tolist(), [3.0, 1.0])
        self.assertEqual(out["TAXITIME_SEC_mvt"].tolist(), [300.0, 100.0])
        self.assertEqual(list(out.dtypes), [np.float64, np.float64])

    def test_missing_row_refused(self):
        with self.assertRaises(ValueError):
            subset(self.pred, pd.DataFrame({"MVT_ID_mvt": [1.0, 9.0]}))


if __name__ == "__main__":
    unittest.main()
