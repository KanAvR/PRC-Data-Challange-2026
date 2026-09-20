"""Fixed random sample of 12 matched missing-LOBT departures per airport."""
import runpy,json,gzip,hashlib
from urllib.request import urlopen,Request
from urllib.error import HTTPError
from pathlib import Path
ns=runpy.run_path('output/experiments/public_data/check_opdi_coverage.py')
j=ns['j'];r=ns['r']
s=j[j.plausible & j.MVT_ID_mvt.isin(r.loc[r.unique_match & r.missing_lobt & r.ADEP_mvt.isin(['LIRF','LFPG']),'MVT_ID_mvt'])]
s=s.groupby('ADEP_mvt',observed=True).sample(n=12,random_state=1729)
out=Path('output/experiments/public_data/traces');out.mkdir(exist_ok=True)
results=[]
for row in s.itertuples():
    day=row.date.replace('-','/');hexcode=row.icao24
    url=f'https://adsb.lol/globe_history/{day}/traces/{hexcode[-2:]}/trace_full_{hexcode}.json'
    try:
        with urlopen(Request(url,headers={'User-Agent':'public-aviation-research'}),timeout=30) as response:
            b=response.read(5_000_001)
        if len(b)>5_000_000: raise ValueError('sample too large')
        content=gzip.decompress(b) if b[:2]==b'\x1f\x8b' else b
        trace=json.loads(content)
        file=out/f'{row.date}_{hexcode}.json';file.write_bytes(content)
        points=trace.get('trace',[])
        ground=sum(len(p)>3 and p[3]=='ground' for p in points)
        result={'airport':row.ADEP_mvt,'id':int(row.MVT_ID_mvt),'url':url,'status':'downloaded','points':len(points),'ground_points':ground,'sha256':hashlib.sha256(content).hexdigest()}
    except Exception as error:
        result={'airport':row.ADEP_mvt,'url':url,'status':'unavailable','error':str(error)}
    print(result,flush=True);results.append(result)
(out.parent/'trace_random_results.json').write_text(json.dumps(results,indent=2)+'\n')
