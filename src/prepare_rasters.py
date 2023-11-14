# -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

import os
import rasterio as rs
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from glob import glob
import geopandas as gpd

def sentinel_merge_and_clip(country, data):

    '''
    Mosaic sentinel files then clip to Costa Rica boundary
    future iterations will download sentinel-1 and 2 tiles for given input country
    '''
    
    folder = (f'../data/label_prop/{country}/')

    # now open each item in dataset reader mode (required to merge)
    reader_mode = []

    for file in glob(f'{folder}{data}/*.tif'):
        src = rs.open(file)
        reader_mode.append(src) 

    # for s2 and s1 nodata value is 0
    mosaic, out_transform = merge(reader_mode)
    
    # outpath will be the new filename
    merged = f'{folder}{country + data}_merged.tif'
    out_meta = src.meta.copy()  
    out_meta.update({'driver': "GTiff",
                     'dtype': 'uint16',  
                     'height': mosaic.shape[1],
                     'width': mosaic.shape[2],
                     'transform': out_transform,
                     'compress':'lzw'})

    with rs.open(merged, "w", **out_meta, BIGTIFF='YES') as dest:
        dest.write(mosaic)

    # now clip merged tif to country boundaries
    merged = f'{folder}{country + data}_merged.tif'
    shapefile = gpd.read_file(f'{folder}gadm41_CRI_0.shp')
    clipped = f'{folder}{country + data}_clipped.tif'

    with rs.open(merged) as src:
        shapefile = shapefile.to_crs(src.crs)
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        out_meta = src.meta.copy() 

    out_meta.update({
        "driver":"Gtiff",
        "height":out_image.shape[1], # height starts with shape[1]
        "width":out_image.shape[2], # width starts with shape[2]
        "transform":out_transform
    })

    with rs.open(clipped,'w',**out_meta, BIGTIFF='YES') as dst:
        dst.write(out_image)
    
    print(f'{len(reader_mode)} tifs merged and clipped to {country}.')
    os.remove(merged)
    return None


def reprj_lulc_raster(lulc_tif, dst_crs='EPSG:4326'):

    '''
    reproject the lulc dataset to espg 4326 to match other input data
    taken verbatim from rasterio documentation w/ added compression params
    future iterations would check if all inputs meet desired crs
    '''
    input = f'../data/raw/{lulc_tif}'
    output = f'../data/interim/{lulc_tif}_reprj.tif'

    with rs.open(input) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw',
            'dtype': 'uint8'})
    
        with rs.open(output, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rs.band(src, i),
                    destination=rs.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return None









# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
