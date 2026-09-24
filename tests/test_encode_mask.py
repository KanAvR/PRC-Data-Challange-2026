import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from encode import PRIOR_WEIGHT, add_target_encodings


def frame(n):
    return pd.DataFrame({
        "ADEP_STAND_RUNWAY": ["A_1_R"] * n, "ADEP_STAND": ["A_1"] * n, "ADEP_AIRLINE": ["A_X"] * n,
        "ADEP_AIRCRAFT_TYPE": ["A_T"] * n, "ADEP_RUNWAY": ["A_R"] * n, "hour": [8.5] * n,
    })


class MaskTest(unittest.TestCase):
    def test_masked_rows_get_the_unseen_fallback_on_flight_keys_only(self):
        X_fit, y = frame(6), np.array([600., 800., 1000., 1200., 1400., 1600.])
        folds = np.array(["m1", "m1", "m2", "m2", "m3", "m3"])
        flight_fit = np.array(["A_F1", "A_F1", "A_F1", "A_F2", "A_F2", "A_F2"])
        X_app, flight_app = frame(2), np.array(["A_F1", "A_F2"])
        X_plain, X_masked = X_app.copy(), X_app.copy()
        add_target_encodings(X_fit.copy(), y, folds, flight_fit, X_plain, flight_app)
        add_target_encodings(X_fit, y, folds, flight_fit, X_app, flight_app,
                             X_apply_masked=X_masked, flight_unseen=np.array([True, False]))
        prior = y.mean()
        # unmasked output is unchanged by asking for a masked copy
        pd.testing.assert_frame_equal(X_app, X_plain)
        self.assertAlmostEqual(X_masked["te_flight"].iloc[0], prior)
        self.assertEqual(X_masked["te_flight_n"].iloc[0], 0.0)
        self.assertAlmostEqual(X_masked["te_flight"].iloc[1], X_app["te_flight"].iloc[1])
        self.assertAlmostEqual(X_app["te_flight"].iloc[0], (600 + 800 + 1000 + PRIOR_WEIGHT * prior) / (3 + PRIOR_WEIGHT))
        for c in ["te_stand_rwy", "te_airline", "te_rwy_hour"]:
            np.testing.assert_array_equal(X_masked[c].values, X_app[c].values)


class BackoffTest(unittest.TestCase):
    def data(self):
        X_fit = frame(6).assign(ADES_mvt=["B"] * 6, hour_local=[8.5] * 6, gap_SCHED_TIME_UTC_mvt=[900.] * 6)
        y = np.array([600., 800., 1000., 1200., 1400., 1600.])
        folds = np.array(["m1", "m1", "m2", "m2", "m3", "m3"])
        flights = np.array(["A_F1", "A_F1", "A_F1", "A_F2", "A_F2", "A_F2"])
        X_app = frame(2).assign(ADES_mvt=["B", "B"], hour_local=[8.5, 8.5], gap_SCHED_TIME_UTC_mvt=[900., 900.])
        return X_fit, y, folds, flights, X_app, np.array(["A_F1", "A_NEW"])

    def test_off_is_bit_identical(self):
        X_fit, y, folds, fl, X_app, fa = self.data()
        a, b = X_app.copy(), X_app.copy()
        add_target_encodings(X_fit.copy(), y, folds, fl, a, fa)
        add_target_encodings(X_fit.copy(), y, folds, fl, b, fa, flight_backoff=False, flight_dropout=0.0)
        pd.testing.assert_frame_equal(a, b)

    def test_unseen_flight_backs_off_to_the_slot_not_the_global_mean(self):
        X_fit, y, folds, fl, X_app, fa = self.data()
        # a second route elsewhere drags the global mean away from this route
        X_fit = pd.concat([X_fit, X_fit.assign(ADEP_AIRLINE="A_Z", ADES_mvt="C")], ignore_index=True)
        y = np.concatenate([y, y + 3000])
        folds, fl = np.concatenate([folds, folds]), np.concatenate([fl, ["A_G1"] * 6])
        add_target_encodings(X_fit, y, folds, fl, X_app, fa, flight_backoff=True)
        self.assertEqual(X_app["te_flight_n"].iloc[1], 0.0)
        # global mean 2600; route (6 rows summing 6600) shrinks toward it, the slot
        # toward the route, and the unseen flight takes the slot value
        route = (6600 + PRIOR_WEIGHT * 2600) / (6 + PRIOR_WEIGHT)
        slot = (6600 + PRIOR_WEIGHT * route) / (6 + PRIOR_WEIGHT)
        self.assertAlmostEqual(X_app["te_flight"].iloc[1], slot)

    def test_dropout_zeroes_whole_keys_on_fit_rows(self):
        X_fit, y, folds, fl, X_app, fa = self.data()
        add_target_encodings(X_fit, y, folds, fl, X_app, fa, flight_backoff=True, flight_dropout=1.0)
        self.assertTrue((X_fit["te_flight_n"] == 0).all())
        # apply rows are untouched by dropout
        self.assertGreater(X_app["te_flight_n"].iloc[0], 0)


class AirportPriorTest(unittest.TestCase):
    def test_unseen_stand_takes_its_airport_mean(self):
        fit = pd.concat([frame(4).assign(ADEP_mvt="A"),
                         frame(4).assign(ADEP_mvt="Z", ADEP_STAND="Z_1", ADEP_STAND_RUNWAY="Z_1_R")],
                        ignore_index=True)
        y = np.array([1000.] * 4 + [3000.] * 4)
        folds = np.array(["m1", "m2"] * 4)
        flights = np.array(["A_F"] * 4 + ["Z_F"] * 4)
        app = frame(1).assign(ADEP_mvt="A", ADEP_STAND="A_NEW", ADEP_STAND_RUNWAY="A_NEW_R")
        off, on = app.copy(), app.copy()
        add_target_encodings(fit.copy(), y, folds, flights, off, np.array(["A_F"]))
        add_target_encodings(fit.copy(), y, folds, flights, on, np.array(["A_F"]), airport_prior=True)
        self.assertAlmostEqual(off["te_stand"].iloc[0], 2000.0)  # global mean
        apt_a = (4000 + PRIOR_WEIGHT * 2000) / (4 + PRIOR_WEIGHT)  # airport A, shrunk to global
        self.assertAlmostEqual(on["te_stand"].iloc[0], apt_a)


if __name__ == "__main__":
    unittest.main()
