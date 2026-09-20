"""Evaluate the new CatBoost member against the corrected v17 weather control."""
import sys,json
from pathlib import Path
sys.path.insert(0,'output/experiments/v9_src')
import numpy as np
import pandas as pd
from normal_correction import apply_normal_correction
from validate import compare

out=Path('output/experiments/neighbor_proxy')
new=pd.read_parquet(out/'cat_weather_inner.parquet')
old=pd.read_parquet('output/experiments/aobt_residual/cat_ctr1_weather_inner.parquet')
base=pd.read_parquet(out/'strong_corrected_anchor_rome_member_inner.parquet')
control=pd.read_parquet('output/experiments/tail_aux_labels/control_inner.parquet')
context=pd.read_parquet('output/experiments/tail_neighbor/inner.parquet')
assert all(base.id.equals(f.id) for f in [new,old,control,context])
assert np.array_equal(new.y,old.y) and np.array_equal(new.y,base.y)
rome=base.ADEP_mvt.eq('LIRF');missing=~base.has_lobt
p15=control.p.copy()
p15.loc[rome&missing]=.5*(control.loc[rome&missing,'p']+context.loc[rome&missing,'p'])
# Transfer a quarter-sized normal-member update. Keep the current Rome
# specialist's 20 percent share unchanged on matched normal rows.
scale=np.where(rome&~missing&base.gap_LOBT_flt.le(6000),.8,1.)
new_mu=old.mu+(new.mu-old.mu)*scale
result=base.copy()
result['pred']=apply_normal_correction(base.pred,base.gap_LOBT_flt,base.gap_SCHED_TIME_UTC_mvt,p15,old.mu,new_mu)
result.to_parquet(out/'strong_v17_cat_neighbor_inner.parquet',index=False)
compare('inner',result,base,'new-Cat-member-in-strong-v17')
by_airport=[]
for airport,g in result.groupby('ADEP_mvt',observed=True):
 b=base.loc[g.index]
 by_airport.append({'airport':str(airport),'rows':len(g),'base_rmse':float(np.sqrt(np.mean((b.y-b.pred)**2))),'candidate_rmse':float(np.sqrt(np.mean((g.y-g.pred)**2)))})
record={'comparison':'corrected two-model weather v17 proxy; not exact production ensemble','weight':.25,'Rome_matched_delta_scale':.8,'baseline_rmse':float(np.sqrt(np.mean((base.y-base.pred)**2))),'candidate_rmse':float(np.sqrt(np.mean((result.y-result.pred)**2))),'by_airport':by_airport,'submitted':False,'remaining_gate':'Require additional temporal splits and full artifact audit before promotion.'}
(out/'cat_v17_inner_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
