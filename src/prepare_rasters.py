#!/usr/bin/env python

import os
import rasterio as rs
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import Affine
from glob import glob
import geopandas as gpd
import yaml
import subprocess
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from osgeo import gdalconst


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

# def reprj_raster(src_ds = '../data/costa_rica/raw/lulc_cr_2016.tif', 
#                  ref_ds = '../data/costa_rica/raw/CostaRica.tif',
#                  dst_ds = '../data/interim/lulc_cr_reprj.tif',
#                  crs='EPSG:4326'):

#     '''
#     Reproject the src_ds (LULC) to match the ref ds (TTC) projection and bounds
#     Affine transform is not working
#     '''
#     ref_ds = gdal.Open(ref_ds, gdalconst.GA_ReadOnly)
#     ref_geotrans = ref_ds.GetGeoTransform()
#     width = ref_ds.RasterXSize 
#     height = ref_ds.RasterYSize 

#     with rs.open(src_ds) as src:
#         assert src.crs != crs, print(f"source crs: {src.crs}, target crs: {crs}")
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': crs,
#             'transform': Affine(ref_geotrans),
#             'width': width,
#             'height': height,
#             'compress': 'lzw',
#             'dtype': 'uint8'})

#         with rs.open(dst_ds, 'w', **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 reproject(
#                     source=rs.band(src, i),
#                     destination=rs.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=Affine(ref_geotrans),
#                     dst_crs=crs,
#                     resampling=Resampling.nearest)
#     return None



def rasterize_sdpt(src, 
                    ref, 
                    dst_path,
                    attribute,
                    no_data_value=255,
                    rdtype=gdal.GDT_Float32, 
                    **kwargs):
    """
    Converts any shapefile to a raster
    :param in_shp_file_name: STR of a shapefile name (with directory e.g., "C:/temp/poly.shp")
    :param out_raster_file_name: STR of target file name, including directory; must end on ".tif"
    :param pixel_size: INT of pixel size (default: 10)
    :param no_data_value: Numeric (INT/FLOAT) for no-data pixels (default: 255)
    :param rdtype: gdal.GDALDataType raster data type - default=gdal.GDT_Float32 (32 bit floating point)
    :kwarg field_name: name of the shapefile's field with values to burn to the raster
    :return: produces the shapefile defined with in_shp_file_name

    Applies gdal.RasterizeLayer with the following params:
    options=["ALL_TOUCHED=TRUE"] defines that all pixels touched by a polygon get the polygon's field 
    value - if not set: only pixels that are entirely in the polygon get a value assigned
    burn_values=[0] (a default value that is burned to the raster)
    
    SDPT
    2: 'forest plantation'
    1: 'oil palm'
    7215: 'orchard'
    """
    src_ds = ogr.Open(src)
    src_lyr = src_ds.GetLayer()

    # set up the reference file 
    ref_ds = gdal.Open(ref, gdalconst.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    width = ref_ds.RasterXSize 
    height = ref_ds.RasterYSize 
    
    # create destination data source (GeoTIff raster)
    dst = gdal.GetDriverByName('GTiff').Create(dst_path, 
                                                 width, 
                                                 height, 
                                                 1,
                                                 eType=rdtype, 
                                                 options=['COMPRESS=LZW'])
    
    dst.SetGeoTransform(ref_geotrans)
    dst.SetProjection(ref_proj)
    band = dst.GetRasterBand(1)
    band.Fill(no_data_value)
    band.SetNoDataValue(no_data_value)

    gdal.RasterizeLayer(dst, 
                        [1], 
                        src_lyr, 
                        None, 
                        None, 
                        options=["ALL_TOUCHED=TRUE", f"ATTRIBUTE={attribute}"]
                       )

    # release raster band
    band.FlushCache()

    return None


# this is the call for rasterize
    # rasterize(src='../data/costa_rica/raw/cri_sdpt_v2.shp', 
    #       ref='../data/costa_rica/raw/CostaRica.tif', 
    #       dst_path='../data/costa_rica/interim/cri_sdpt_v2.tif',
            # attribute='originalCo',
    #       no_data_value=255,
    #       rdtype=gdal.GDT_Float32
    #      )
