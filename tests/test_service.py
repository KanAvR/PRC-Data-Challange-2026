import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from service import COLS, service_features


def frame():
    t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
    return pd.DataFrame({
        "PHASE_mvt": ["DEP", "DEP", "DEP", "ARR", "DEP"],
        "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "EGLL", "LIRF"],
        "ADES_mvt": ["LIMC", "EGLL", "LIMC", "LIRF", None],
        "FLIGHT_mvt": ["ITY100", "ITY100", "ITYTY680", "BAW1", "AEZ12AB"],
        "MVT_TIME_UTC_mvt": [t, t + pd.Timedelta(hours=23), t, t, t + pd.Timedelta(hours=30)],
    })


class ServiceFeaturesTest(unittest.TestCase):
    def test_columns(self):
        f = service_features(frame())
        self.assertEqual(list(f.columns), COLS)
        self.assertEqual(f.loc[0, "svc_domestic"], 1.0)              # LIRF -> LIMC
        self.assertEqual(f.loc[1, "svc_domestic"], 0.0)              # LIRF -> EGLL
        self.assertEqual(f.loc[1, "svc_ades_country"], "EG")
        self.assertEqual(f.loc[4, "svc_ades_country"], "missing")
        self.assertEqual(f.loc[0, "svc_n_same_flight_24h"], 1.0)     # ITY100 again 23h later
        self.assertEqual(f.loc[2, "svc_n_same_flight_24h"], 0.0)
        self.assertEqual(f.loc[2, "svc_cs_long_prefix"], 1.0)        # ITYTY680
        self.assertEqual(f.loc[0, "svc_cs_long_prefix"], 0.0)
        self.assertEqual(f.loc[4, "svc_cs_alpha_suffix"], 1.0)       # AEZ12AB
        self.assertTrue(np.isnan(f.loc[3, "svc_domestic"]))          # ARR rows stay empty


if __name__ == "__main__":
    unittest.main()
