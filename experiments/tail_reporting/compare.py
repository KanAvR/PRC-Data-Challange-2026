"""Fixed Rome-only quarter probability change against the v15 local recipe."""
import sys
sys.path.insert(0,'output/experiments/v9_src')
import numpy as np,pandas as pd
from model import V_HI_THR, V_HI_RATIO
from validate import compare
for split in sys.argv[1:] or ['inner']:
    root='output/experiments/'
    new=pd.read_parquet(root+'tail_reporting/'+split+'.parquet')
    context=pd.read_parquet(root+'tail_neighbor/'+split+'.parquet')
    old=pd.read_parquet(root+'tail_aux_labels/control_'+split+'.parquet')
    neigh=pd.read_parquet(root+'neighbor_proxy/'+split+'.parquet')
    base=pd.read_parquet(root+'v15_corrected/v15_'+split+'.parquet')
    assert all(base.id.equals(f.id) for f in [new,context,old,neigh])
    mu=old.mu+.25*(neigh.mu-old.mu)
    sel=(~base.has_lobt)&base.ADEP_mvt.eq('LIRF')&(base.v>6000)
    tail=base.v.where(base.v < V_HI_THR,base.v*V_HI_RATIO)
    result=base.copy()
    result.loc[sel,'pred']=np.maximum(0,base.loc[sel,'pred']+.25*(new.loc[sel,'p']-context.loc[sel,'p'])*(tail.loc[sel]-mu.loc[sel]))
    result.to_parquet(root+'tail_reporting/v15_increment_'+split+'.parquet',index=False)
    compare(split,result,base,'reporting-increment-over-v15')
