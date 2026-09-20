"""Past carrier reporting completeness; uses inputs, never taxi-time labels."""
import numpy as np
import pandas as pd
COLS=['reporting_count_1d','reporting_missing_rate_1d','reporting_count_7d','reporting_missing_rate_7d']
def build(raw,airline):
    assert len(raw)==len(airline)
    out=pd.DataFrame(np.nan,index=np.arange(len(raw)),columns=COLS)
    # Explicit epoch seconds avoids dependence on pandas datetime storage units.
    t=raw.MVT_TIME_UTC_mvt.map(lambda v:v.timestamp() if pd.notna(v) else np.nan).to_numpy()
    d=pd.DataFrame({'t':t,'missing':raw.LOBT_flt.isna().to_numpy(),'apt':raw.ADEP_mvt.to_numpy(),'airline':np.asarray(airline)})
    for _,g in d.groupby(['apt','airline'],dropna=False,observed=True):
        g=g[np.isfinite(g.t)].sort_values('t')
        ts=g.t.to_numpy();values=g.missing.to_numpy(dtype=int)
        prefix=np.r_[0,np.cumsum(values)];right=np.searchsorted(ts,ts,side='left')
        for days in [1,7]:
            left=np.searchsorted(ts,ts-days*86400,side='left');n=right-left
            rate=np.divide(prefix[right]-prefix[left],n,out=np.full(len(g),np.nan),where=n>0)
            out.loc[g.index,f'reporting_count_{days}d']=n
            out.loc[g.index,f'reporting_missing_rate_{days}d']=rate
    return out

def test_inputs():
    raw=pd.DataFrame({'MVT_TIME_UTC_mvt':pd.to_datetime(['2025-01-01','2025-01-02','2025-01-02','2025-01-03'],utc=True),'LOBT_flt':[pd.NaT,pd.Timestamp('2025-01-02',tz='UTC'),pd.NaT,pd.NaT],'ADEP_mvt':['LIRF']*4})
    f=build(raw,['A']*4)
    assert f.reporting_count_1d.tolist()==[0,1,1,2]
    assert f.reporting_missing_rate_1d.iloc[1:].tolist()==[1,1,.5]
    raw['TAXITIME_SEC_mvt']=[0,1,90000,3]
    pd.testing.assert_frame_equal(f,build(raw,['A']*4))
    raw.loc[3,'LOBT_flt']=pd.Timestamp('2025-01-03',tz='UTC')
    pd.testing.assert_frame_equal(f,build(raw,['A']*4))
if __name__=='__main__':
    test_inputs(); print('Past-only, self/tie exclusion and label-independence checks passed')
