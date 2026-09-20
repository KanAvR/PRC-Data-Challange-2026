"""Fixed 20-percent residual expert on matched normal rows outside Rome."""
import sys,json
from pathlib import Path
sys.path.insert(0,'output/experiments/v9_src')
import numpy as np,pandas as pd
from normal_correction import apply_normal_correction
from validate import compare,verdict
out=Path('output/experiments/neighbor_proxy')
checks={};stats={};sums=np.zeros(3)
for split in ['inner','dec_jun','mar_sep','holdout','strong_inner']:
 if split=='strong_inner':
  base=pd.read_parquet(out/'strong_corrected_anchor_rome_member_inner.parquet')
  expert=pd.read_parquet(out/'weather_anchor_inner.parquet')
 else:
  base=pd.read_parquet('output/experiments/v15_corrected/anchor_rome_'+split+'.parquet')
  expert=pd.read_parquet(out/('anchor_'+split+'.parquet'))
 assert base.id.equals(expert.id) and np.array_equal(base.y,expert.y)
 sel=base.has_lobt & base.gap_LOBT_flt.le(6000) & base.ADEP_mvt.ne('LIRF')
 member=base.pred.copy();member.loc[sel]=expert.loc[sel,'mu']
 r=base.copy()
 r['pred']=apply_normal_correction(base.pred,base.gap_LOBT_flt,base.gap_SCHED_TIME_UTC_mvt,np.zeros(len(base)),base.pred,member,weight=.2)
 assert np.array_equal(r.loc[~sel,'pred'],base.loc[~sel,'pred'])
 r.to_parquet(out/('residual_member_'+split+'.parquet'),index=False)
 c=compare(split,r,base,'residual-member-vs-v17')
 bse=float(np.square(base.y-base.pred).sum());cse=float(np.square(r.y-r.pred).sum());n=len(r)
 stats[split]={'rows':n,'baseline_rmse':np.sqrt(bse/n),'candidate_rmse':np.sqrt(cse/n),'improvement':np.sqrt(bse/n)-np.sqrt(cse/n)}
 if split!='strong_inner':checks[split]=c;sums+=[bse,cse,n]
b,c,n=sums
result={'recipe':'Fixed20percent residual expert, matchednormal nonRome only','comparisons':stats,'pooled_improvement':float(np.sqrt(b/n)-np.sqrt(c/n)),'weight':.2,'submitted':False}
(out/'residual_member_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(verdict(checks));print(json.dumps(result,indent=2))
