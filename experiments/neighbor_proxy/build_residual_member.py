"""Prepare a local-only, fixed 20-percent residual-member candidate from v17."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'output/experiments/v9_src')
import numpy as np,pandas as pd
from featcache import _cache_path
from normal_correction import apply_normal_correction
source=Path('output/submitting_v17.parquet')
expected='016e89500b0a797ec22c1b55f4bf80a75b52d7a3412c449b150270df54817fd3'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
base=pd.read_parquet(source);ids=base.MVT_ID_mvt
raw=pd.read_parquet('dataset/ranking.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt'])
gaps=pd.read_parquet(_cache_path('dataset/ranking.parquet'),columns=['gap_LOBT_flt','gap_SCHED_TIME_UTC_mvt'])
raw=pd.concat([raw,gaps],axis=1).set_index('MVT_ID_mvt').loc[ids]
assert raw.PHASE_mvt.eq('DEP').all()
mu=pd.read_parquet('output/experiments/neighbor_proxy/ranking_weather_anchor.parquet').set_index('MVT_ID_mvt').loc[ids,'mu'].to_numpy()
gap=raw.gap_LOBT_flt.to_numpy();pred=base.TAXITIME_SEC_mvt.to_numpy()
sel=np.isfinite(gap)&(gap<=6000)&raw.ADEP_mvt.ne('LIRF').to_numpy()
member=pred.copy();member[sel]=mu[sel]
result=base.copy()
values=apply_normal_correction(pred,gap,raw.gap_SCHED_TIME_UTC_mvt,np.zeros(len(base)),pred,member,weight=.2)
# Independent convex-combination identity and support preservation.
lo=np.maximum(0,gap-3660);hi=gap+3660
eligible=sel&(pred>lo+1e-6)&(pred<hi-1e-6)
manual=pred.copy();manual[eligible]=np.clip(.8*pred[eligible]+.2*mu[eligible],lo[eligible],hi[eligible])
assert np.allclose(values,manual,rtol=0,atol=1e-10)
assert np.array_equal(values[~eligible],pred[~eligible])
result['TAXITIME_SEC_mvt']=values
assert len(result)==344841 and ids.is_unique
assert result.dtypes.equals(base.dtypes) and result.MVT_ID_mvt.equals(ids)
assert np.isfinite(values).all() and (values>=0).all()
path=Path('output/submitting_residual_member_candidate.parquet');result.to_parquet(path,index=False)
record={'source':'v17','source_sha256':expected,'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(result),'changed_rows':int(np.count_nonzero(values-pred)),'all_checks_passed':True,'weight':.2,'eligible':'Matched normal non-Rome rows strictly within existing support','local_pooled_gain':.7070660937085904,'strong_weather_inner_gain':.15851701645547678,'status':'Local candidate only; stronger-control CI crosses zero, defer promotion pending other experiments.','submitted':False}
Path('output/experiments/neighbor_proxy/residual_member_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
