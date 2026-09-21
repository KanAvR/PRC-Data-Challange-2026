"""Independently reconstruct the fixed candidate arithmetic for upload auditing."""
import sys, hashlib, json
from pathlib import Path
sys.path.insert(0,'output/experiments/v9_src')
import numpy as np
import pandas as pd
from featcache import _cache_path
root=Path('output/experiments')
source=Path('output/submitting_v15.parquet')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='551d4f8e5af1bc94331e3c5ace81ba322eeaad27e38bad68794a6b0df971068a'
v15=pd.read_parquet(source)
raw=pd.read_parquet('dataset/ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt'])
gaps=pd.read_parquet(_cache_path('dataset/ranking.parquet'),columns=['gap_LOBT_flt','gap_SCHED_TIME_UTC_mvt'])
raw=pd.concat([raw,gaps],axis=1).set_index('MVT_ID_mvt').loc[v15.MVT_ID_mvt]
ids=raw.index
gap=raw.gap_LOBT_flt.to_numpy(); sched=raw.gap_SCHED_TIME_UTC_mvt.to_numpy()
missing=~np.isfinite(gap); rome=raw.ADEP_mvt.eq('LIRF').to_numpy()
lo=np.maximum(0,gap-3660); hi=gap+3660
pred=v15.TAXITIME_SEC_mvt.to_numpy()
inside=(~missing)&(gap<=6000)&(pred>lo+1e-6)&(pred<hi-1e-6)
def vec(file,col):
    f=pd.read_parquet(root/file).set_index('MVT_ID_mvt')
    assert f.index.is_unique
    return f.loc[ids,col].to_numpy()
old=pd.read_parquet(root/'baseline_identity/tail_probabilities.parquet').set_index('MVT_ID_mvt')
ctx=pd.read_parquet(root/'tail_neighbor/ranking_probabilities.parquet').set_index('MVT_ID_mvt')
p=np.zeros(len(ids)); p[missing]=old.loc[ids[missing],'p5']
s=missing&rome; p[s]=.5*p[s]+.5*ctx.loc[ids[s],'p'].to_numpy()
delta=.25*(vec('neighbor_proxy/ranking_weather_anchor.parquet','mu')-vec('neighbor_proxy/ranking_weather_neighbor.parquet','mu'))
a=pred.copy(); a[inside]=np.clip(pred[inside]+delta[inside],lo[inside],hi[inside])
share=np.where(np.nan_to_num(sched[missing],nan=0)>6000,1-p[missing],1)
a[missing]=np.maximum(0,pred[missing]+share*delta[missing])
expert=pd.read_parquet(root/'airport_experts/ranking_LIRF_weather_neighbor.parquet').set_index('MVT_ID_mvt')
s=(~missing)&rome&(gap<=6000)&(a>lo+1e-6)&(a<hi-1e-6)
b=a.copy(); b[s]=np.clip(a[s]+.2*(expert.loc[ids[s],'mu'].to_numpy()-a[s]),lo[s],hi[s])
results={}
for name,expected in [('anchor',a),('anchor_rome',b)]:
    path=Path(f'output/submitting_{name}_candidate.parquet')
    c=pd.read_parquet(path); values=c.TAXITIME_SEC_mvt.to_numpy()
    assert len(c)==344841 and c.MVT_ID_mvt.is_unique
    assert c.MVT_ID_mvt.equals(v15.MVT_ID_mvt) and c.dtypes.equals(v15.dtypes)
    assert np.isfinite(values).all() and (values>=0).all()
    assert np.allclose(values,expected,rtol=0,atol=1e-10)
    protected=(~missing)&~inside
    assert np.array_equal(values[protected],pred[protected])
    results[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(c),'changed_rows':int(np.count_nonzero(values-pred)),'protected_rows':int(protected.sum()),'all_checks_passed':True}
assert np.array_equal(a[missing|~rome],b[missing|~rome])
(root/'neighbor_proxy/anchor_candidate_audit.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
