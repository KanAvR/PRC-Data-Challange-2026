import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import INBOUND_COLS, with_inbound


class InboundTest(unittest.TestCase):
    def test_inbound_after_sched(self):
        # take-off 10,000s after schedule; the stand's last arrival 3,000s before take-off,
        # i.e. 7,000s after the scheduled departure: a late inbound
        X = pd.DataFrame({"gap_SCHED_TIME_UTC_mvt": [10000.0, 2000.0],
                          "secs_since_prev_arr_block": [3000.0, np.nan]})
        out = with_inbound(X)
        self.assertAlmostEqual(out.loc[0, "inbound_after_sched"], 7000.0)
        self.assertTrue(np.isnan(out.loc[1, "inbound_after_sched"]))
        self.assertNotIn("inbound_after_sched", X.columns)   # input untouched
        self.assertIn("inbound_after_sched", INBOUND_COLS)


if __name__ == "__main__":
    unittest.main()
