import importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location('neighbor_features', Path(__file__).with_name('features.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def sample():
    t = pd.date_range('2025-02-01 12:00', periods=3, freq='5min', tz='UTC')
    return pd.DataFrame({'MVT_ID_mvt': [1, 2, 3], 'MVT_TIME_UTC_mvt': t,
        'PHASE_mvt': 'DEP', 'ADEP_mvt': 'LIRF', 'RUNWAY_mvt': '16R',
        'AOBT_3_flt': t - pd.to_timedelta([600, 1200, 1800], unit='s'),
        'LOBT_flt': t - pd.to_timedelta([600, 1200, 1800], unit='s')})

def test_neighbor_values_and_self_exclusion():
    d = sample()
    r = module.build(d)
    assert r.loc[1, 'proxy_apt_AOBT_3_flt_before_15'] == 600
    assert r.loc[1, 'proxy_apt_AOBT_3_flt_after_15'] == 1800
    d.loc[1, 'AOBT_3_flt'] = d.loc[1, 'MVT_TIME_UTC_mvt'] - pd.Timedelta(seconds=5000)
    z = module.build(d)
    pd.testing.assert_series_equal(r.loc[1], z.loc[1])

def test_departure_target_independence():
    d = sample()
    r = module.build(d)
    d['TAXITIME_SEC_mvt'] = [90000, 1, 99999]
    d['BLOCK_TIME_UTC_mvt'] = d.MVT_TIME_UTC_mvt - pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(r, module.build(d))

def test_airports_do_not_mix():
    d = sample()
    d.loc[0, 'ADEP_mvt'] = 'LFPG'
    r = module.build(d)
    assert pd.isna(r.loc[1, 'proxy_apt_AOBT_3_flt_before_15'])

def test_wide_range_retains_long_reports_without_self_inclusion():
    d = sample()
    d.loc[0, 'AOBT_3_flt'] = d.loc[0, 'MVT_TIME_UTC_mvt'] - pd.Timedelta(seconds=9000)
    narrow, wide = module.build(d), module.build(d, max_proxy_gap=12000)
    key = 'proxy_apt_AOBT_3_flt_before_15'
    assert pd.isna(narrow.loc[1, key])
    assert wide.loc[1, key] == 9000
    assert pd.isna(wide.loc[0, key])
