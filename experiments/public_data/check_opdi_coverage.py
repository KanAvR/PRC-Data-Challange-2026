"""Assess public flight-list joins only; no model inputs or predictions changed."""
import sys,json
from pathlib import Path
sys.path.insert(0,'output/experiments/v9_src')
import pandas as pd
from airline import flight_key
out=Path('output/experiments/public_data')
o=pd.read_parquet(out/'flight_list_202501.parquet')
r=pd.read_parquet('output/experiments/raw_dep.parquet',columns=['MVT_ID_mvt','FLIGHT_mvt','CALLSIGN_flt','ADEP_mvt','MVT_TIME_UTC_mvt','LOBT_flt','month'])
r=r[r.MVT_TIME_UTC_mvt.dt.month.eq(1)].copy()
r['key']=r.CALLSIGN_flt.fillna(flight_key(r.FLIGHT_mvt)).str.strip()
r['date']=r.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
o['key']=o.flt_id.str.strip();o['date']=o.dof.astype(str)
j=r.merge(o[['key','date','adep','icao24','first_seen','last_seen']],left_on=['key','date','ADEP_mvt'],right_on=['key','date','adep'],how='left')
j['diff_sec']=(pd.to_datetime(j.first_seen,utc=True)-pd.to_datetime(j.MVT_TIME_UTC_mvt,utc=True)).dt.total_seconds()
j['plausible']=j.icao24.notna()&j.diff_sec.abs().le(1800)
counts=j.groupby('MVT_ID_mvt').plausible.sum()
r['unique_match']=r.MVT_ID_mvt.map(counts).eq(1)
r['missing_lobt']=r.LOBT_flt.isna()
summary=r.groupby(['ADEP_mvt','missing_lobt'],observed=True).agg(rows=('unique_match','size'),matched=('unique_match','sum'))
summary['coverage']=summary.matched/summary.rows
summary.to_json(out/'opdi_coverage.json',orient='table',indent=2)
print(summary.to_string())
print('Total',len(r),'unique plausible matches',r.unique_match.sum())
