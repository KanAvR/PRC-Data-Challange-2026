"""Label-free route geography using the public-domain OurAirports snapshot."""
from pathlib import Path
import numpy as np
import pandas as pd
SOURCE=Path('output/experiments/public_data/airports.csv')
def build(frame,airports=None):
    if airports is None:
        airports=pd.read_csv(SOURCE,usecols=['ident','latitude_deg','longitude_deg','elevation_ft'])
    assert airports.ident.is_unique
    a=airports.set_index('ident')
    dep=a.reindex(frame.ADEP_mvt).reset_index(drop=True)
    dest=a.reindex(frame.ADES_mvt).reset_index(drop=True)
    lat1,lat2=np.radians(dep.latitude_deg.to_numpy()),np.radians(dest.latitude_deg.to_numpy())
    lon1,lon2=np.radians(dep.longitude_deg.to_numpy()),np.radians(dest.longitude_deg.to_numpy())
    dl=lon2-lon1;dp=lat2-lat1
    h=np.sin(dp/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2
    distance=12742*np.arcsin(np.sqrt(np.clip(h,0,1)))
    bearing=np.arctan2(np.sin(dl)*np.cos(lat2),np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dl))
    return pd.DataFrame({'route_distance_km':distance,'route_bearing_sin':np.sin(bearing),'route_bearing_cos':np.cos(bearing),
        'route_dest_lat':dest.latitude_deg,'route_dest_lon':dest.longitude_deg,
        'route_elevation_difference_m':(dest.elevation_ft-dep.elevation_ft)*.3048})

def test_geometry():
    airports=pd.DataFrame({'ident':['AAAA','BBBB'],'latitude_deg':[0.,0.],'longitude_deg':[0.,90.],'elevation_ft':[0.,1000.]})
    x=pd.DataFrame({'ADEP_mvt':['AAAA','BBBB','AAAA'],'ADES_mvt':['BBBB','AAAA','XXXX']})
    z=build(x,airports)
    assert np.allclose(z.route_distance_km[:2],np.pi*6371/2)
    assert np.allclose(z.route_bearing_sin[:2],[1,-1])
    assert np.allclose(z.route_elevation_difference_m[:2],[304.8,-304.8])
    assert z.iloc[2].isna().all()
    x['TAXITIME_SEC_mvt']=[1,90000,2]
    pd.testing.assert_frame_equal(z,build(x,airports))
if __name__=='__main__':
    test_geometry(); print('Geometry, unknown-code and label-independence checks passed')
