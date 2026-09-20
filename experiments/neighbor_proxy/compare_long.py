"""Compare extra rounds alone and as the same residual member in v17."""
import sys
sys.path.insert(0,'output/experiments/v9_src')
import pandas as pd,numpy as np
from normal_correction import apply_normal_correction
from validate import compare
out='output/experiments/neighbor_proxy/'
old=pd.read_parquet(out+'weather_anchor_inner.parquet')
new=pd.read_parquet(out+'weather_anchor_long_inner.parquet')
base=pd.read_parquet(out+'strong_corrected_anchor_rome_member_inner.parquet')
control=pd.read_parquet('output/experiments/tail_aux_labels/control_inner.parquet')
context=pd.read_parquet('output/experiments/tail_neighbor/inner.parquet')
assert all(old.id.equals(f.id) for f in [new,base,control,context])
compare('inner',new,old,'1800-vs-900-residual-rounds')
p15=control.p.copy();rome=base.ADEP_mvt.eq('LIRF');missing=~base.has_lobt
p15.loc[rome&missing]=.5*(control.loc[rome&missing,'p']+context.loc[rome&missing,'p'])
# The Rome expert owns 20 percent of matched normal predictions in v17.
new_mu=old.mu+(new.mu-old.mu)*np.where(rome&~missing&base.gap_LOBT_flt.le(6000),.8,1.)
r=base.copy()
r['pred']=apply_normal_correction(base.pred,base.gap_LOBT_flt,base.gap_SCHED_TIME_UTC_mvt,p15,old.mu,new_mu)
r.to_parquet(out+'strong_v17_long_inner.parquet',index=False)
compare('inner',r,base,'1800-round-member-in-strong-v17')
