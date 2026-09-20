"""Correct local v15 reconstruction to match the shipped tail-value scaling."""
import sys,json
from pathlib import Path
sys.path.insert(0,'output/experiments/v9_src')
import numpy as np,pandas as pd
from model import combine,V_HI_THR,V_HI_RATIO
from normal_correction import apply_normal_correction
from validate import compare,verdict
out=Path('output/experiments/v15_corrected');out.mkdir(exist_ok=True)
checks={};totals=np.zeros(3);details={}
for split in ['inner','dec_jun','mar_sep','holdout']:
 root=Path('output/experiments')
 old=pd.read_parquet(root/'tail_aux_labels'/('control_'+split+'.parquet'))
 context=pd.read_parquet(root/'tail_neighbor'/(split+'.parquet'))
 neigh=pd.read_parquet(root/'neighbor_proxy'/(split+'.parquet'))
 anchor=pd.read_parquet(root/'neighbor_proxy'/('anchor_'+split+'.parquet'))
 expert=pd.read_parquet(root/'airport_experts'/('LIRF_'+split+'.parquet'))
 previous=pd.read_parquet(root/'tail_neighbor'/('combined_rome_'+split+'.parquet'))
 assert all(old.id.equals(f.id) for f in [context,neigh,anchor,expert,previous])
 rome=old.ADEP_mvt.eq('LIRF');missing=~old.has_lobt;p15=old.p.copy()
 p15.loc[rome&missing]=.5*(old.loc[rome&missing,'p']+context.loc[rome&missing,'p'])
 mu=old.mu+.25*(neigh.mu-old.mu)
 base=previous.copy()
 eligible=rome&missing&(old.v>=V_HI_THR)
 base.loc[eligible,'pred']+=.5*(context.loc[eligible,'p']-old.loc[eligible,'p'])*old.loc[eligible,'v']*(V_HI_RATIO-1)
 # Independent mixture identity for all unbounded missing-LOBT rows.
 exact=combine(mu.to_numpy(),p15.to_numpy(),old.v.to_numpy())
 assert np.allclose(base.loc[missing,'pred'],exact[missing],atol=1e-8,rtol=0)
 base['mu']=mu;base['p']=p15
 base.to_parquet(out/('v15_'+split+'.parquet'),index=False)
 r=base.copy();r['pred']=apply_normal_correction(base.pred,old.gap_LOBT_flt,old.gap_SCHED_TIME_UTC_mvt,p15,neigh.mu,anchor.mu)
 local=2*expert.mu-neigh.mu; member=r.pred.copy();sel=rome&~missing&(old.gap_LOBT_flt<=6000)
 member.loc[sel]=local.loc[sel]
 r['pred']=apply_normal_correction(r.pred,old.gap_LOBT_flt,old.gap_SCHED_TIME_UTC_mvt,p15,r.pred,member,weight=.2)
 r.to_parquet(out/('anchor_rome_'+split+'.parquet'),index=False)
 checks[split]=compare(split,r,base,'corrected-v15-anchor-Rome')
 b=float(np.square(base.y-base.pred).sum());c=float(np.square(r.y-r.pred).sum());n=len(r);totals+=[b,c,n]
 details[split]={'baseline_rmse':np.sqrt(b/n),'candidate_rmse':np.sqrt(c/n),'improvement':np.sqrt(b/n)-np.sqrt(c/n),'corrected_rows':int(eligible.sum())}
b,c,n=totals;gain=float(np.sqrt(b/n)-np.sqrt(c/n))
result={'folds':details,'baseline_rmse':float(np.sqrt(b/n)),'candidate_rmse':float(np.sqrt(c/n)),'improvement':gain,'one_point_gate_passes':gain>=1,'rows':int(n),'uploaded':False}
(out/'upload_gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(verdict(checks));print(json.dumps(result,indent=2))
