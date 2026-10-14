import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import predict


class TraceDaysTest(unittest.TestCase):
    def test_take_off_days_and_the_day_before(self):
        raw = pd.DataFrame({
            "PHASE_mvt": ["DEP", "DEP", "ARR"],
            "MVT_TIME_UTC_mvt": ["2026-02-01 00:10:00", "2026-06-15 12:00:00", "2026-03-01 08:00:00"],
        })
        # arrivals read no trace day; each departure needs its day and the one before
        self.assertEqual(predict.trace_days(raw),
                         ["2026-01-31", "2026-02-01", "2026-06-14", "2026-06-15"])


if __name__ == "__main__":
    unittest.main()
