import importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location('ground_queue', Path(__file__).with_name('ground_queue.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def sample():
    return pd.DataFrame({'MVT_ID_mvt': [1, 2, 3], 'PHASE_mvt': 'DEP',
        'ADEP_mvt': 'LIRF', 'RUNWAY_mvt': ['16R', '16R', '16L'],
        'MVT_TIME_UTC_mvt': pd.to_datetime(['2025-02-01 12:10', '2025-02-01 12:15', '2025-02-01 12:20'], utc=True),
        'AOBT_3_flt': pd.to_datetime(['2025-02-01 12:00'] * 3, utc=True)})

def test_active_interval_and_self_exclusion():
    r = module.build(sample())
    assert list(r.ground_apt_start) == [2, 2, 2]
    assert list(r.ground_apt_takeoff) == [2, 1, 0]
    assert list(r.ground_rwy_start) == [1, 1, 0]

def test_invalid_intervals_and_label_independence():
    d = sample()
    d.loc[2, 'AOBT_3_flt'] = pd.NaT
    r = module.build(d)
    assert r.loc[0, 'ground_apt_start'] == 1
    d['TAXITIME_SEC_mvt'] = [1, 90000, 12]
    d['BLOCK_TIME_UTC_mvt'] = d.MVT_TIME_UTC_mvt - pd.Timedelta(hours=12)
    pd.testing.assert_frame_equal(r, module.build(d))
