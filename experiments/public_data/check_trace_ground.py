"""Measure ground coverage near the queried departure; no target columns read."""
import pandas as pd,numpy as np,json,sys
from pathlib import Path
out=Path('output/experiments/public_data')
a=pd.read_csv(out/'airports.csv').set_index('ident')
r=pd.read_parquet('output/experiments/raw_dep.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt']).set_index('MVT_ID_mvt')
stem=sys.argv[1] if len(sys.argv)>1 else 'trace_sample'
results=json.loads((out/(stem+'_results.json')).read_text())
for s in results:
 if s['status']!='downloaded': continue
 url=s['url']; date='/'.join(url.split('/')[4:7]).replace('/','-'); hexcode=url.rsplit('_',1)[1][:-5]
 tr=json.loads((out/'traces'/f'{date}_{hexcode}.json').read_text())
 p=tr['trace']; t=np.array([v[0]+tr['timestamp'] for v in p]);
 lat=np.radians(np.array([v[1] for v in p]));lon=np.radians(np.array([v[2] for v in p]))
 apt=a.loc[s['airport']];la,lo=np.radians([apt.latitude_deg,apt.longitude_deg])
 dist=6371*2*np.arcsin(np.minimum(1,np.sqrt(np.sin((lat-la)/2)**2+np.cos(lat)*np.cos(la)*np.sin((lon-lo)/2)**2)))
 takeoff=r.loc[s['id'],'MVT_TIME_UTC_mvt'].timestamp();dt=t-takeoff
 sel=(dist<10)&(dt>=-7200)&(dt<=120)
 ground=np.array([v[3]=='ground' for v in p]); keep=sel&ground
 s.update({'near_departure_points':int(sel.sum()),'near_departure_ground_points':int(keep.sum()),'ground_first_seconds_before_takeoff':float(-dt[keep].min()) if keep.any() else None,'ground_last_seconds_before_takeoff':float(-dt[keep].max()) if keep.any() else None})
 print(s['airport'],s['id'],{k:v for k,v in s.items() if k.startswith('near_') or k.startswith('ground_')})
(out/(stem+'_ground_coverage.json')).write_text(json.dumps(results,indent=2)+'\n')
