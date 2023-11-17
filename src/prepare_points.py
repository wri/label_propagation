import geopandas as gpd
import numpy as np
from datetime import datetime
import pandas as pd
from shapely.geometry import Point
import rasterio as rs

def get_unlabeled_pts(country, num_pts): # add high_conf param when import is removed
    
    '''
    Points must fall within the country bounds but outside of
    the high confidence areas.
    Warning: overlay could take some time to process. Use sparingly 
    and save output.
    '''
    
#     over = gpd.overlay(country, high_conf, 'difference', keep_geom_type=True)
#     over.to_file(f'../data/label_prop/{country}/hc_overlay.shp')
    
    # import for now
    over = gpd.read_file(f'../data/interim/{country}/hc_overlay.shp')
    minx = over.bounds.minx[0]
    miny = over.bounds.miny[0]
    maxx = over.bounds.maxx[0]
    maxy = over.bounds.maxy[0]
    
    df = pd.DataFrame(columns=['geometry'])
    points = []
    counter = 0
    
    start = datetime.now()
    while counter < num_pts:
        x = np.random.uniform(minx, maxx, 1)
        y = np.random.uniform(miny, maxy, 1)
        pt = Point([x,y])
        if over.contains(pt).all():
            points.append(pt)
            counter += 1

    df['geometry'] = points
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    gdf['label'] = 'unlabeled'
    end = datetime.now()
    print(f'Completed in: {end - start}')
    
    return gdf


def get_labeled_pts(country):
    '''
    this will extract centroids from high confidence polygons
    '''
    interim = (f'../data/interim/{country}/')
    pts = gpd.read_file(f'{interim}labeled_pts.shp')
    return pts


# this might not live here 
def extract_features(pts):
    
    '''
    Extracts features for each point from the input datasets. 
    Returns dataframe where each row represents an unlabeled observation.
    
    Even though lulc, sdpt and ttc values are not needed for unlabeled data
    they are extracted for the validation phase.

    '''
    # operate on a copy
    df = pts.copy()
    
    # import ttc, sdpt and lulc data and confirm crs
    raw = ('../data/raw/costa_rica/')
    interim = ('../data/interim/costa_rica/')
    
    ttc = rs.open(f'{raw}CostaRica.tif')
    sdpt = gpd.read_file(f'{raw}cri_sdpt_v2.shp')
    lulc = rs.open(f'{interim}lulc_cr_reprj.tif')
    s1 = rs.open(f'{interim}/costa_ricas1_clipped.tif')
    s2 = rs.open(f'{interim}costa_ricas2_clipped.tif')
    assert ttc.crs == sdpt.crs == lulc.crs == s1.crs == s2.crs == pts.crs
    
    coords = [(x,y) for x, y in zip(pts.geometry.x, pts.geometry.y)]
    
    # create new df to hold extracted attributes
    df['ttc'] = [x[0] for x in ttc.sample(coords)]
    df['lulc'] = [x[0] for x in lulc.sample(coords)]
    df['s1'] = [x[0] for x in s1.sample(coords)]
    df['s2'] = [x[0] for x in s2.sample(coords)]
    
    # for sdpt must perform a join before extraction
    intersection = gpd.sjoin(pts, sdpt[['originalCo', 'originalNa', 'geometry']], how='left', predicate='within')
    df['sdpt'] = intersection['originalCo']
    
    return df

def eval_centroids(centroid_pts):
    
    '''
    Extracts features for a centroid to determine if the sample should
    be included in the training dataset. If there is no sentinel 1 or 2 data, 
    drop the sample (there would be no s1 bc of terrain shadow and no s2 bc of clouds).
    Does not return df with extracted features, this is simply a checkpoint.
    '''
    df = extract_features(centroid_pts)
    
    no_sentinel = df[(df.s1 == 0.0)|(df.s2 == 0.0)]
    no_data = df[(df.sdpt == 'nan')&(df.ttc == '255')&(df.lulc == '255')]
    
    df = df[(df.s1 != 0.0)]
    df = df[df.s2 != 0.0]
    if len(no_data) > 0:
        df = df.drop(df[no_data].index)
    
    print(f'{len(no_sentinel)} samples have no sentinel values.')
    print(f'{len(no_data)} samples have no data values.')
    print(f'{len(no_sentinel)+len(no_data)} samples will be dropped.')
    
    return df[['geometry', 'label']]


    