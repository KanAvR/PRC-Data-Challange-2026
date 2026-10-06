import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inbound import COL, _day

MVT = 1_750_000_000.0
SCHED = MVT - 9000.0


def points():
    """Airframe abc arrives (airborne, landing ~MVT-7200) then climbs out as ITY1 at MVT+60."""
    rows = [("abc", "AZA9", "LIRF", MVT - 7400, 900), ("abc", "AZA9", "LIRF", MVT - 7200, 200),
            ("abc", None, "LIRF", MVT - 7100, -1), ("abc", "ITY1", "LIRF", MVT + 60, 500),
            ("zzz", "ITY1", "LIRF", MVT + 40000, 500)]      # same callsign, another day: ignored
    return pd.DataFrame(rows, columns=["icao", "callsign", "airport", "ts", "alt"])


def dep(callsign="ITY1"):
    return pd.DataFrame({"index": [7], "ADEP_mvt": ["LIRF"], "mvt_ts": [MVT], "sched_ts": [SCHED],
                         "CALLSIGN_flt": [callsign], "FLIGHT_mvt": [None]})


class InboundAirTest(unittest.TestCase):
    def test_arrival_after_schedule(self):
        out = _day(dep(), points())
        # last airborne point more than 20 min before take-off: MVT-7200, i.e. 1800s after SCHED
        self.assertAlmostEqual(out.loc[7], (MVT - 7200) - SCHED)
        self.assertEqual(out.name, COL)

    def test_unknown_callsign_is_nan(self):
        self.assertTrue(_day(dep("XXX9"), points()).empty)


if __name__ == "__main__":
    unittest.main()
