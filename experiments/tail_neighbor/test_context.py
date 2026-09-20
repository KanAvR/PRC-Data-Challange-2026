import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from context_features import add_context

def test_reporting_coverage_and_target_independence():
    x = pd.DataFrame({'dep_off_apt_30m': [4., 0.], 'dep_off_apt_next_30m': [2., 5.]})
    extra = pd.DataFrame(index=x.index)
    for direction in ['before', 'after']:
        for source in ['AOBT_3_flt', 'LOBT_flt']:
            key = 'proxy_apt_' + source + '_' + direction + '_30'
            extra[key], extra[key + '_count'] = [1200., 1800.], [2., 0.]
    a = x.copy()
    cols = add_context(a, extra)
    assert a.loc[0, 'proxy_apt_LOBT_flt_before_30_coverage'] == .5
    assert a.loc[0, 'proxy_apt_LOBT_flt_after_30_coverage'] == 1.
    assert a.loc[1, 'proxy_apt_LOBT_flt_before_30_coverage'] == 0.
    b = x.copy()
    b['TAXITIME_SEC_mvt'] = [1., 90000.]
    add_context(b, extra)
    pd.testing.assert_frame_equal(a[cols], b[cols])

def test_misaligned_context_is_rejected():
    x = pd.DataFrame(index=[0, 1])
    try:
        add_context(x, pd.DataFrame(index=[1, 0]))
    except ValueError:
        return
    raise AssertionError('misaligned context accepted')
