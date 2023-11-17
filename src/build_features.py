
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
from prepare_points import extract_features

# high confidence polygon is needed here

def calc_grid(minx, maxx, miny, maxy, grid_res = 10):
    
    '''
    Buffer is a df column that contains a polygon
    form the grid by setting up the coordinates and then fill grid with numpy.meshgrid
    adding half the grid_res to the maxx (or maxy) value ensures the end point is 
    included in the range
    '''
    x = np.arange(minx, maxx + grid_res / 2, grid_res)
    y = np.arange(miny, maxy + grid_res / 2, grid_res)
    xx, yy = np.meshgrid(x, y)

    # flatten points to 1D vectors
    x_in_buff = xx.ravel()
    y_in_buff = yy.ravel()
    grid_as_list = list(zip(x_in_buff, y_in_buff))
    grid_as_pts = [Point([xy]) for xy in grid_as_list]
    
    return grid_as_pts


def convert_centroid_to_grid(centroid_df, country, labels, radius = 65):
    '''
    Converts a lat/lon centroid into a 14x14 grid of points
    with 10m spacing by buffering the point with a radius and
    using the bounding box to determine the location of 
    points. Returns a geodataframe with the original centroid,
    buffer and 14x14 point grid.

    for labeled grids, labels are updated to null if the grid falls
    outside of the high confidence polygons

    Parameters:
    radius: 65 meters will place each point as the centroid of the 10m pixel
    
    TODO: Confirm if crs will vary by country
    '''
    df = centroid_df.to_crs(epsg='3857')
    df['buffer'] = df['geometry'].buffer(radius, cap_style=3)
    df['minx'] = df['buffer'].bounds.minx
    df['maxx'] = df['buffer'].bounds.maxx
    df['miny'] = df['buffer'].bounds.miny
    df['maxy'] = df['buffer'].bounds.maxy
    
    df['grid'] = df.apply(lambda x: calc_grid(x['minx'], x['maxx'], x['miny'], x['maxy']), axis=1)
    df = df.explode('grid').reset_index(drop=False)
    
    output = pd.DataFrame()
    output['plot_id'] = df['index']
    output['geometry'] = df['grid']
    output['label'] = df['label']
    
    output_gdf = gpd.GeoDataFrame(output, geometry='geometry', crs='EPSG:3857')
    output_gdf = output_gdf.to_crs(epsg='4326')
    output_gdf['point_x'] = output_gdf.geometry.x
    output_gdf['point_y'] = output_gdf.geometry.y

    # update label for grid points falling outside high conf polygons
    if labels:
        high_conf = gpd.read_file(f'../data/interim/{country}/HighConf_poly1.shp')
        start_labels = output_gdf.label.value_counts().reset_index()
        inside = gpd.sjoin(output_gdf, high_conf, how='inner') # identifies grid pts within high conf poly
        keys = ['point_x', 'point_y']
        i1 = output_gdf.set_index(keys).index
        i2 = inside.set_index(keys).index
        outside = output_gdf[~i1.isin(i2)]
        print(f'total points: {len(output_gdf)}, inside: {len(inside)}, outside: {len(outside)}')
        output_gdf.loc[outside.index, 'label'] = np.NaN
        end_labels = output_gdf.label.value_counts().reset_index()
        print('The following label counts were updated to null:')
        print(start_labels['count'] - end_labels['count'])
        output_gdf.to_file(f'../data/interim/{country}/labeled_grid.shp')

    else:
        output_gdf.to_file(f'../data/interim/{country}/unlabeled_grid.shp')    

    return output_gdf[['plot_id', 'geometry', 'label', 'point_x', 'point_y']]

def extract_clean_grid_features(grid, labels):
    '''
    Applies some cleaning steps and drops poor quality training samples.
    Option to sample the dataset to balance classes
    TODO: address hard coding of sample count
    
    '''
    # # operate on a copy
    # df = pts.copy()
    
    df = extract_features(grid)

    # turn lulc values into categories
    df['lulc'] = df['lulc'].map({7: 'forest plantation', 
                                8: 'mature forest', 
                                9: 'bare', 
                                10: 'urban', 
                                14: 'mangrove',
                                20: 'palm',
                                22: 'pineapple', 
                                23: 'coffee', 
                                30: 'water'}).astype(str)
    
    # turn sdpt values into categories
    df['sdpt'] = df['sdpt'].map({2.0: 'forest plantation', 
                                 1.0: 'oil palm', 
                                 7215.0: 'orchard'}).astype(str)


    # at this stage, if there are remaining pts that dnt have s1 or
    # s2 they should be dropped.
    no_sentinel = df[(df.s1 == 0.0)|(df.s2 == 0.0)]
    no_data = df[(df.sdpt == 'nan')&(df.ttc == '255')&(df.lulc == '255')]
    
    df = df[(df.s1 != 0.0)]
    df = df[df.s2 != 0.0]
    if len(no_data) > 0:
        df = df.drop(df[no_data].index)
    
    print(f'{len(no_sentinel)} points have no sentinel values.')
    print(f'{len(no_data)} points have no data values.')
    print(f'{len(no_sentinel)+len(no_data)} points will be dropped.')
    
    if labels:
        df = df.dropna()
        df.loc[df['sdpt'] == 'forest plantation', 'label'] = 'monoculture'
        df['label'] = df['label'].map({'notplantation': 0, 'monoculture': 1, 'agroforestry': 2}).astype(int)
   
    else:
        df['label'] = -1
    
    print(f'{len(df)} total samples with class distribution:') 
    print(f'{round(df.label.value_counts(normalize=True),2)}')

    return df

