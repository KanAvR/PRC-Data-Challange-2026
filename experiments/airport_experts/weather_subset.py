"""Read weather features for one airport without loading all training rows."""
import numpy as np
import pandas as pd
from data import TRAIN_FILES
from featcache import _cache_path

def load_weather_airport(airport, expected_y):
    frames, labels = [], []
    columns = ['PHASE_mvt', 'ADEP_mvt', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt']
    for path in TRAIN_FILES:
        raw = pd.read_parquet(path, columns=columns, filters=[('ADEP_mvt', '=', airport)]).reset_index(drop=True)
        x = pd.read_parquet(_cache_path(path), filters=[('ADEP_mvt', '=', airport)]).reset_index(drop=True)
        assert len(raw) == len(x)
        gap = (raw.MVT_TIME_UTC_mvt - raw.AOBT_3_flt).dt.total_seconds().to_numpy()
        assert np.allclose(x.gap_AOBT_3_flt, gap, equal_nan=True)
        use = raw.PHASE_mvt.eq('DEP') & raw.TAXITIME_SEC_mvt.ge(0)
        frames.append(x.loc[use])
        labels.append(raw.loc[use, 'TAXITIME_SEC_mvt'].to_numpy())
    assert np.array_equal(np.concatenate(labels), expected_y)
    return pd.concat(frames, ignore_index=True)
