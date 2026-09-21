import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import PROXY_COLS, PROXY_MAX_SEC, _dep_proxy, drop_proxy


def frame(n=400, seed=3):
    """Departures at one airport across two runways, with realistic gaps."""
    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2025-03-01T06:00:00")
    t = t0 + np.sort(rng.integers(0, 6 * 3600, n)).astype("timedelta64[s]")
    taxi = rng.gamma(4, 260, n)
    # a quarter of rows carry no flight plan, and a few sit outside the filter
    aobt = t - taxi.astype("timedelta64[s]")
    aobt = np.where(rng.random(n) < 0.25, np.datetime64("NaT", "s"), aobt)
    lobt = t - (taxi + rng.normal(300, 600, n)).astype("timedelta64[s]")
    # a day-late LOBT: gap far above the filter, which must be excluded not clipped
    lobt = np.where(rng.random(n) < 0.05, t - np.timedelta64(99999, "s"), lobt)
    return pd.DataFrame({
        "MVT_TIME_UTC_mvt": pd.to_datetime(t, utc=True),
        "AOBT_3_flt": pd.to_datetime(aobt, utc=True),
        "LOBT_flt": pd.to_datetime(lobt, utc=True),
        "RUNWAY_mvt": rng.choice(["09L", "27R"], n),
    })


def brute(df, col_t, col_src, sel, w, after):
    """The definition the vectorised builder is supposed to implement."""
    t = df["MVT_TIME_UTC_mvt"].values.astype("datetime64[s]")
    gap = (df[col_t] - df[col_src]).dt.total_seconds().values
    gap = np.where((gap >= 0) & (gap <= PROXY_MAX_SEC), gap, np.nan)
    width = np.timedelta64(w, "m")
    out = np.full(len(df), np.nan)
    for i in np.flatnonzero(sel):
        if after:
            inw = sel & (t > t[i]) & (t <= t[i] + width)
        else:
            inw = sel & (t >= t[i] - width) & (t < t[i])
        vals = gap[inw]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[i] = vals.mean()
    return out


class ProxyFeatureTest(unittest.TestCase):
    def setUp(self):
        self.df = frame()
        self.out = _dep_proxy(self.df)

    def test_matches_a_brute_force_window_mean(self):
        allrows = np.ones(len(self.df), bool)
        for src in ["AOBT_3_flt", "LOBT_flt"]:
            for after in [False, True]:
                d = "after" if after else "before"
                for w in [15, 30, 60]:
                    got = self.out[f"proxy_apt_{src}_{d}_{w}"].values
                    want = brute(self.df, "MVT_TIME_UTC_mvt", src, allrows, w, after)
                    np.testing.assert_allclose(got, want, rtol=1e-9)

    def test_runway_scope_only_sees_its_own_runway(self):
        rwy = self.df["RUNWAY_mvt"].values
        for r in ["09L", "27R"]:
            sel = rwy == r
            got = self.out.loc[sel, "proxy_rwy_AOBT_3_flt_before_30"].values
            want = brute(self.df, "MVT_TIME_UTC_mvt", "AOBT_3_flt", sel, 30, False)[sel]
            np.testing.assert_allclose(got, want, rtol=1e-9)
        # and differs from the airport-wide version, or the scope is doing nothing
        both = self.out["proxy_rwy_AOBT_3_flt_before_30"].values
        apt = self.out["proxy_apt_AOBT_3_flt_before_30"].values
        self.assertTrue(np.nanmax(np.abs(both - apt)) > 1.0)

    def test_a_row_never_enters_its_own_window(self):
        """One departure alone in its window must have no proxy at all."""
        lone = pd.DataFrame({
            "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-03-01T06:00:00"], utc=True),
            "AOBT_3_flt": pd.to_datetime(["2025-03-01T05:50:00"], utc=True),
            "LOBT_flt": pd.to_datetime(["2025-03-01T05:45:00"], utc=True),
            "RUNWAY_mvt": ["09L"],
        })
        out = _dep_proxy(lone)
        means = [c for c in PROXY_COLS if not c.endswith("_count")]
        self.assertTrue(out[means].isna().all().all())
        self.assertTrue((out[[c for c in PROXY_COLS if c.endswith("_count")]] == 0).all().all())

    def test_departures_sharing_a_second_are_excluded_from_each_other(self):
        """The documented tie behaviour: _bounds is strict on both sides.

        If tie handling ever changes, this feature shifts silently at exactly
        the busy moments it is meant to describe."""
        same = pd.to_datetime(["2025-03-01T06:00:00"] * 2, utc=True)
        tie = pd.DataFrame({
            "MVT_TIME_UTC_mvt": same,
            "AOBT_3_flt": same - pd.Timedelta(seconds=900),
            "LOBT_flt": same - pd.Timedelta(seconds=1200),
            "RUNWAY_mvt": ["09L", "09L"],
        })
        out = _dep_proxy(tie)
        means = [c for c in PROXY_COLS if not c.endswith("_count")]
        self.assertTrue(out[means].isna().all().all())
        self.assertTrue((out[[c for c in PROXY_COLS if c.endswith("_count")]] == 0).all().all())

    def test_gaps_outside_the_filter_are_excluded_not_clipped(self):
        """A neighbour with an absurd LOBT gap must not enter the average."""
        df = self.df.copy()
        gap = (df.MVT_TIME_UTC_mvt - df.LOBT_flt).dt.total_seconds().values
        self.assertTrue((gap > PROXY_MAX_SEC).any(), "fixture lost its out-of-range rows")
        means = _dep_proxy(df)["proxy_apt_LOBT_flt_before_60"].values
        self.assertTrue(np.nanmax(means) <= PROXY_MAX_SEC)

    def test_counts_only_count_usable_neighbours(self):
        col = "proxy_apt_AOBT_3_flt_before_30_count"
        n = self.out[col].values
        mean30 = self.out["proxy_apt_AOBT_3_flt_before_30"].values
        # a count of zero and a defined mean cannot coexist
        self.assertFalse(np.any((n == 0) & np.isfinite(mean30)))
        self.assertTrue(np.all(n[np.isfinite(mean30)] > 0))

    def test_drop_proxy_removes_exactly_the_proxy_columns(self):
        x = pd.concat([self.out, pd.DataFrame({"keep_me": np.zeros(len(self.out))})], axis=1)
        d = drop_proxy(x)
        self.assertEqual(list(d.columns), ["keep_me"])
        self.assertEqual(len(drop_proxy(d).columns), 1)  # idempotent


if __name__ == "__main__":
    unittest.main()
