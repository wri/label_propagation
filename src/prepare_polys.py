#!/usr/bin/env python
import pprint
from datetime import datetime
import pandas as pd
import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import numpy as np
from scipy.ndimage import label, sum as ndi_sum, minimum_filter, generate_binary_structure
import subprocess
import yaml
import argparse
from osgeo import gdalconst
from osgeo import gdal
from osgeo import ogr
from scipy import ndimage

# CREDIT: Erin Glen https://github.com/erin-glen
# Setting up the local workspace... only user input needed is home plantation directory
# workspace = "C:\GIS\Data\Plantation"  # this is the only user input needed


def confirm_inputs_match(param_path):
    """
    this func should simply confirm that 
    the crs and bounds of the SDPT and LULC datasets match
    that of the TTC data -- functions in prepare_rasters

    """
    return None

def reclassify_by_value(array, values_to_reclassify):
    """
    This function utilizes numpy to reclassify an array based on a given value.
    The array is reclassified to binary (0 where != to the provided value; 1 where = to provided value
    In this workflow, the value is provided in the inputParams csv.

    :param array: a numpy array
    :param values_to_reclassify: the value to be reclassified to 1
    :return: a binary array
    """
    out_array = np.isin(array, values_to_reclassify).astype(int)
    return out_array


def reclassify_above_threshold(array, threshold):
    """
    This function utilizes numpy to reclassify an array based on a threshold value
    The array is reclassified to binary (0 where < the provided value; 1 where > provided value
    In this workflow, the value is provided in the inputParams csv.

    :param array: a numpy array
    :param threshold: the minimum value threshold
    :return: a binary array
    """
    # array = np.where(array > threshold, 1, 0)

    array[array < threshold] = 0
    array[array >= threshold] = 1


    return array


def reclassify_below_threshold(array, threshold):
    """
    This function utilizes numpy to reclassify an array based on a threshold value
    The array is reclassified to binary (0 where > the provided value; 1 where <>> provided value
    In this workflow, the value is provided in the inputParams csv.

    :param array: a numpy array
    :param threshold: the maximum value threshold
    :return: a binary array
    """
    # array = np.where(array < threshold, 1, 0)

    array[array > threshold] = 0
    array[array <= threshold] = 1
    
    return array


def multiply_arrays(array_list):
    """
    This function takes a list of arrays and multiplies all arrays by each other.
    It is used to find the "intersection" of several datasets
    :param array_list: list of arrays to be multiplied
    :return: a binary array
    """
    if not array_list:
        raise ValueError("The list is empty")
    if len(array_list) == 1:
        return array_list[0]

    print(f"Multiplying {len(array_list)} arrays...")
    result = array_list[0]
    for array in array_list[1:]:
        result = np.multiply(result, array)
    return result


def classifyRasters(lulc_path, sdpt_path, ttc_path, params_path, category):
    """
    This function takes processed input datasets (land use / land cover, 
    spatial database of planted trees, and tropical tree cover) and a csv 
    of dataset combinations per reclassification category to
    create new rasters that represent the intersection of various datasets. 
    The new rasters are saved to an intermediate output folder. 
    It utilizes the reclasssify_* and multiply_arrays functions defined above.


    :param lulc: land use / land cover processed dataset
    :param sdpt: spatial database of planted trees processed dataset
    :param ttc: tropical tree cover processed dataset
    :param inputParams: csv with input parameters and reclass values
    :param category: the category for which to run the reclassification
    :param dataFolder: home data folder path
    :return: None
    """
    # Define a dictionary to map data names to reclassification functions and comparison types
    reclassification_functions = {
        "lulc": reclassify_by_value,
        "sdpt": reclassify_by_value,
        "ttc_greater": reclassify_above_threshold,
        "ttc_less": reclassify_below_threshold,
    }

    inputParams = pd.read_csv(params_path)
    dataList = [lulc_path, sdpt_path, ttc_path] 
    nameList = ["lulc", "sdpt", "ttc"]
    # this list will hold 3 arrays of 1 and 0 for each dataset
    binary_list = []

    # creates a binary 0/1 version of each dataset based on the
    # given parameters, checks that the three arrays are the same shape
    # and multiplies them to get a single raster for the category
    for index, data in enumerate(dataList):
        data_array = rasterio.open(data).read(1)
        data_name = str(nameList[index])
        value_name = data_name + "_value" 

        data_value = inputParams.loc[inputParams['Category'] == str(category), value_name].item()
        comparison_type = inputParams.loc[inputParams['Category'] == str(category), "Comparison"].item()

        print(f"{data_name} array shape: {data_array.shape}")
        print(f"{category} value in {data_name} is {data_value}")
       
        # this will be null for certain classes w/o sdpt polygons
        if pd.isna(data_value):
            print(f"Skipping reclassification, value is NaN\n")
        else:
            if data_name in ["lulc", "sdpt"]:
                reclassify_func = reclassification_functions.get(data_name)
            else:
                print(f"Comparison type: {comparison_type}")
                reclassify_func = reclassification_functions.get(data_name + "_" + comparison_type)
            if reclassify_func:
                print(f"Calculating binary arr for {data_name} with {reclassify_func}")
                binary_array = reclassify_func(data_array, int(data_value))
                # check if binary array contains at least one non zero element
                assert binary_array.any(), f"Error: Array is empty."
                binary_list.append(binary_array)
    
    # for index, array in enumerate(binary_list):
    #     print(f"Array {index} shape: {array.shape}")

    intersection = multiply_arrays(binary_list)
    allzeros = intersection.any()
    assert allzeros, "Error: Intersection array is empty."

    output_raster_file = os.path.join('../data/Output/Intermediate/Rasters/', str(category) + ".tif")
    ttc = rasterio.open(ttc_path)
    meta = ttc.meta
    meta.update(dtype='uint8', nodata=0, compress='lzw')

    with rasterio.open(output_raster_file, 'w', **meta) as dst:
        dst.write(intersection.astype('uint8'), 1)




def rastertoPoly(min_size, category):
    """
    Convert raster files to polygons, only including areas with a 
    minimum number of connected pixels.
    Diagonal connections are not considered.

    :param min_size: Minimum number of connected pixels to be included
    :return: None
    """
    input_folder =  '../data/Output/Intermediate/Rasters/'
    output_folder = '../data/Output/Intermediate/Poly/'


    with rasterio.open(input_folder + f"{category}.tif") as src:
        image = src.read(1)  # read the first band
        crs = src.crs  # get the CRS from the raster

        # Label connected components
        if category == 'Agro_Coffee':
            structure = generate_binary_structure(2, 2)
        else:
            structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=int)
        labeled, _ = label(image, structure=structure) # label connected components in the raster.
        sizes = ndi_sum(image, labeled, range(labeled.max() + 1)) #calculate the size of the components
        mask = np.isin(labeled, np.where(sizes > min_size)[0]) # create a mask to filter out components less than min size

        filtered_image = image * mask

        # create a sequence of dictionaries, each containing properties (raster value) 
        # and geometry (shape) for the non-zero pixels in the filtered image.
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(filtered_image,
                                            mask=filtered_image, 
                                            transform=src.transform))
                                            )

        geoms = list(results)
        if geoms:
            gdf = gpd.GeoDataFrame.from_features(geoms)
            gdf.crs = crs  # set the CRS for the GeoDataFrame
            gdf.to_file(os.path.join(output_folder + f"{category}.shp"))
    
    return None

def merge_hc_polys(min_size, inputParams):
    """
    This function processes all shapefiles in a specified folder. It creates two shapefiles:
    one with centroids of all polygons from all shapefiles and another with all polygons merged.
    It also counts the number of polygons and centroids for each shapefile, calculates the
    average size and variance of polygon sizes in hectares, and saves the results in a CSV file.

    :param dataFolder: Folder path containing shapefiles
    :param min_size: Minimum size parameter used for filtering polygons
    :return: None
    """
    input_folder =  '../data/Output/Intermediate/Poly/'
    output_folder = '../data/Output/Final/'

    all_polygons = []
    shapefile_data = []

    categories = inputParams['Category'].unique().tolist()
    label_map = inputParams.set_index('Category')['label']

    # gather data on specific category for output csv
    for category in categories:
        gdf = gpd.read_file(f"{input_folder + category}.shp")
        # add column for labels
        gdf['category'] = category
        gdf['label'] = gdf['category'].map(label_map)
        all_polygons.append(gdf)
        num_polygons = len(gdf)

        # Transform to a projected CRS for area calculation
        # Calculate average size and variance in hectares
        gdf_projected = gdf.to_crs('EPSG:3857')
        areas_hectares = gdf_projected['geometry'].area / 10000  # Convert from square meters to hectares
        avg_size_hectares = areas_hectares.mean()
        variance_hectares = areas_hectares.var()
        shapefile_data.append([category, num_polygons, min_size, avg_size_hectares, variance_hectares])

    # Merge polygons into a single GeoDataFrame
    merged_polygons = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=gdf.crs)
    output_file = os.path.join(output_folder, "merged_polygons2.shp")
    merged_polygons.to_file(output_file)
    print(f"All polygons have been merged and saved to {output_file}")

    # Save shapefile data to CSV
    columns = ["Land Cover", "Number of Polygons", "Min Size", "Average Size (Hectares)", "Variance in Size (Hectares)"]
    df = pd.DataFrame(shapefile_data, columns=columns)
    output_csv = os.path.join(output_folder, "stats_jessica5.csv")
    df.to_csv(output_csv, index=False)
    print(f"Statistics saved to {output_csv}")

    return None


def create_highconf_polygons(params_path,
                            lulc_path,
                            sdpt_path, 
                            ttc_path,
                            min_size = 196):
    '''
    Inputs:
        - National Land Use Land Cover map
        - Tropical Tree Cover
        - Spatial Database of Planted Trees
        - GADM National Boundary
    
    Objective: extract "high confidence" areas where the input datasets have overlap.
    Potential to have this live as it's own script
    
    Rules: see ppt
    
    Harmonization Steps:
        1. Assert all inputs use the same crs. Reproject if necessary.
        2. Assert all inputs have the same resolution. Resample if necessary.
        3. Get the extent of all rasters, intersect extent and make windows.
        4. Create high confidence polygons based on overlap of all datasets.
        5. Extract the centroid of the polygon
        
    https://gis.stackexchange.com/questions/447158/getting-the-intersection-of-two-raster-files-using-pandas-python
    
    '''
    input_params = pd.read_csv(params_path)
    categories = input_params['Category'].unique().tolist()
    
    for category in categories:
        print("Creating a binary raster of high confidence areas for", category,)
        classifyRasters(lulc_path, 
                        sdpt_path, 
                        ttc_path, 
                        params_path, 
                        category
                        )
    
        print(f"Converting {category} high confidence raster to polygon...")
        rastertoPoly(min_size, category)

    print("Merging all high conf polygons...")
    merge_hc_polys(min_size, input_params)

    return None


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--params", dest="params", required=True)
    args = args_parser.parse_args()
    create_highconf_polygons(params_path=args.params)
