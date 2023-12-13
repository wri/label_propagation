import pprint
from datetime import datetime
import pandas as pd
import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import numpy as np
from scipy.ndimage import label, sum as ndi_sum
import subprocess

# CREDIT: Erin Glen https://github.com/erin-glen

# Setting up the local workspace... only user input needed is home plantation directory
workspace = "C:\GIS\Data\Plantation"  # this is the only user input needed

# these paths are created if they do not already exist
dataFolder = os.path.join(workspace, 'Data')
outputFolder = os.path.join(dataFolder, 'Output')
intermediateFolder = os.path.join(outputFolder, 'Intermediate')
rasterFolder = os.path.join(intermediateFolder, 'Rasters')
polyFolder = os.path.join(intermediateFolder, 'Poly')
finalFolder = os.path.join(outputFolder, 'Final')

# function to create folder if it does not already exist
folderList = [workspace, dataFolder, outputFolder, intermediateFolder, rasterFolder, polyFolder, finalFolder]
for folder in folderList:
    if not os.path.exists(folder):
        os.makedirs(folder)

if __name__ == "__main__":
    # Create paths to datasets and add to input config
    inputConfig: dict[str, str] = dict(
        inputParams=os.path.join(dataFolder,"Input", 'inputParams', "inputParams.csv"),
        dataFolder=dataFolder,
        min_size = 196
    )

def preprocess_datasets(dataFolder):
    """
    Preprocess all datasets in the folder to match the properties of the reference dataset,
    including coordinate system and size (width and height).
    Skips preprocessing if the dataset has already been processed.

    :param dataFolder: Base folder containing the datasets
    :return: None
    """
    input_folder = os.path.join(dataFolder, "Input", "Raw")
    reference_dataset_name = "TTC.tif"
    reference_path = os.path.join(input_folder, reference_dataset_name)
    output_path = os.path.join(dataFolder, "Input", "Processed")

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get properties of reference dataset
    with rasterio.open(reference_path) as ref_dataset:
        ref_crs = ref_dataset.crs.to_string()
        target_width, target_height = ref_dataset.width, ref_dataset.height
        ref_bounds = ref_dataset.bounds
        ref_res_x = (ref_bounds.right - ref_bounds.left) / target_width
        ref_res_y = (ref_bounds.top - ref_bounds.bottom) / target_height

    # Process all files in the folder
    for filename in os.listdir(input_folder):
        if filename == reference_dataset_name:
            continue  # Skip the reference dataset

        processed_file_path = os.path.join(output_path, filename)
        if os.path.exists(processed_file_path):
            print(f"Processed file {processed_file_path} already exists. Skipping.")
            continue

        file_path = os.path.join(input_folder, filename)

        # Check if the file is a raster or polygon and process accordingly
        if filename.endswith('.tif'):  # Raster dataset
            # Using gdalwarp to reproject and resample the raster
            command = [
                'gdalwarp',
                '-t_srs', ref_crs,  # Target SRS from reference
                '-tr', str(ref_res_x), str(ref_res_y),  # Calculated resolution
                '-r', 'near',  # Resampling method
                '-te', str(ref_bounds.left), str(ref_bounds.bottom), str(ref_bounds.right), str(ref_bounds.top),  # Target extent
                file_path,
                processed_file_path
            ]
            subprocess.run(command, check=True)

        elif filename.endswith('.shp'):  # Polygon dataset
            gdf = gpd.read_file(file_path)
            # Reproject to reference dataset's CRS
            gdf = gdf.to_crs(ref_crs)
            # Save the processed file
            gdf.to_file(processed_file_path)

    lulc = os.path.join(output_path,"LULC.tif")
    sdpt = os.path.join(output_path,"SDPT.tif")
    ttc = reference_path

    return lulc, sdpt, ttc


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
    out_array = np.where(array > threshold, 1, 0)
    return out_array


def reclassify_below_threshold(array, threshold):
    """
    This function utilizes numpy to reclassify an array based on a threshold value
    The array is reclassified to binary (0 where > the provided value; 1 where <>> provided value
    In this workflow, the value is provided in the inputParams csv.

    :param array: a numpy array
    :param threshold: the maximum value threshold
    :return: a binary array
    """
    out_array = np.where(array < threshold, 1, 0)
    return out_array


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

    print("Multiplying arrays...")
    result = array_list[0]
    for array in array_list[1:]:
        result = np.multiply(result, array)
    return result


def classifyRasters(lulc, sdpt, ttc, inputParams, category, dataFolder):
    """
    This function takes processed input datasets (land use / land cover, spatial database of planted
    trees, and tropical tree cover) and a csv of dataset combinations per reclassification category to
    create new rasters that represent the intersection of various datasets. The new rasters are saved to an
    intermediate output folder. It it utilizes the reclasssify_* and multiply_arrays functions defined above.


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

    inputParams = pd.read_csv(inputParams)
    dataList = [lulc, sdpt, ttc]
    nameList = ["lulc", "sdpt", "ttc"]
    binary_list = []


    for index, data in enumerate(dataList):
        data = rasterio.open(data)
        data_array = data.read(1)
        data_name = str(nameList[index])
        value_name = data_name + "_value"
        # comparison_name = data_name + "_comparison"

        data_value = inputParams.loc[inputParams['Category'] == str(category), value_name].item()
        comparison_type = inputParams.loc[inputParams['Category'] == str(category), "Comparison"].item()

        print(f"{data_name} array shape: {data_array.shape}")
        print(f"{data_name}: {data_value}")
        print(f"Comparison type: {comparison_type}")

        if pd.isna(data_value):
            print("Skipping, value is NaN\n")
        else:
            print("Calculating binary dataset...\n")
            if data_name == "lulc" or data_name == "sdpt":
                reclassify_func = reclassification_functions.get(data_name)
            else:
                reclassify_func = reclassification_functions.get(data_name + "_" + comparison_type)

            if reclassify_func:
                binary_array = reclassify_func(data_array, int(data_value))
                allzeros = binary_array.any()
                assert allzeros, "Error: Array is empty.\n"
                binary_list.append(binary_array)

    print("Binary list length: ", len(binary_list))
    for index, array in enumerate(binary_list):
        print(f"Array {index} shape: {array.shape}")

    intersection = multiply_arrays(binary_list)
    allzeros = intersection.any()
    assert allzeros, "Error: Intersection array is empty.\n"

    output_raster_file = os.path.join(dataFolder, "Output", "Intermediate", "Rasters", str(category) + ".tif")

    ttc = rasterio.open(ttc)
    meta = ttc.meta
    meta.update(dtype='uint8', nodata=0)

    with rasterio.open(output_raster_file, 'w', **meta) as dst:
        dst.write(intersection.astype('uint8'), 1)


def rastertoPoly(dataFolder, min_size):
    """
    Convert raster files to polygons, only including areas with a minimum number of connected pixels.
    Diagonal connections are not considered.

    :param dataFolder: Folder path containing raster files
    :param min_size: Minimum number of connected pixels to be included
    :return: None
    """
    input_folder = os.path.join(dataFolder, "Output", "Intermediate", "Rasters")
    output_folder = os.path.join(dataFolder, "Output", "Intermediate", "Poly")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            print(f'Processing {filename}')
            raster_path = os.path.join(input_folder, filename)
            with rasterio.open(raster_path) as src:
                image = src.read(1)  # read the first band
                crs = src.crs  # get the CRS from the raster

                # # Define 4-connectivity structure
                # structure = np.array([[0, 1, 0],
                #                       [1, 1, 1],
                #                       [0, 1, 0]], dtype=int)

                # Define 4-connectivity structure
                structure = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]], dtype=int)


                # Label connected components
                labeled, _ = label(image, structure=structure)
                sizes = ndi_sum(image, labeled, range(labeled.max() + 1))
                mask = np.isin(labeled, np.where(sizes > min_size)[0])

                filtered_image = image * mask

                results = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for i, (s, v) in enumerate(
                    shapes(filtered_image, mask=filtered_image, transform=src.transform)))

                geoms = list(results)
                if geoms:
                    gdf = gpd.GeoDataFrame.from_features(geoms)
                    gdf.crs = crs  # set the CRS for the GeoDataFrame
                    gdf.to_file(os.path.join(output_folder, filename.replace('.tif', '.shp')))


def createCentroidAndMergedPolygons(dataFolder, min_size,inputParams):
    """
    This function processes all shapefiles in a specified folder. It creates two shapefiles:
    one with centroids of all polygons from all shapefiles and another with all polygons merged.
    It also counts the number of polygons and centroids for each shapefile, calculates the
    average size and variance of polygon sizes in hectares, and saves the results in a CSV file.

    :param dataFolder: Folder path containing shapefiles
    :param min_size: Minimum size parameter used for filtering polygons
    :return: None
    """
    input_folder = os.path.join(dataFolder, "Output", "Intermediate", "Poly")
    output_folder = os.path.join(dataFolder, "Output", "Final")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_centroids = []
    all_polygons = []
    shapefile_data = []
    label_map = inputParams.set_index('Category')['label_value']

    polygon_shapefiles = [f for f in os.listdir(input_folder) if f.endswith('.shp')]
    for shp_file in polygon_shapefiles:
        print(f"Processing {shp_file}")
        shp_path = os.path.join(input_folder, shp_file)
        gdf = gpd.read_file(shp_path)

        # Transform to a projected CRS for area calculation
        gdf_projected = gdf.to_crs('EPSG:3857')

        # Process centroids
        centroids = gdf_projected['geometry'].representative_point()

        # Transform centroids back to the original CRS
        centroids = centroids.to_crs(gdf.crs)

        centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=gdf.crs)
        centroids_gdf['Category'] = shp_file.split(".")[0]
        centroids_gdf['label_value'] = centroids_gdf['Category'].map(label_map)

        all_centroids.append(centroids_gdf)

        # Accumulate polygons
        all_polygons.append(gdf)

        # Count polygons and centroids
        num_polygons = len(gdf)
        num_centroids = len(centroids)

        # Calculate average size and variance in hectares
        areas_hectares = gdf_projected['geometry'].area / 10000  # Convert from square meters to hectares
        avg_size_hectares = areas_hectares.mean()
        variance_hectares = areas_hectares.var()

        shapefile_data.append([shp_file, num_polygons, num_centroids, min_size, avg_size_hectares, variance_hectares])

    # Merge centroids into a single GeoDataFrame and save
    final_centroids = gpd.GeoDataFrame(pd.concat(all_centroids, ignore_index=True), crs=gdf.crs)
    output_shapefile_centroids = os.path.join(output_folder, "all_centroids.shp")
    final_centroids.to_file(output_shapefile_centroids)
    print(f"\nAll centroids have been saved to {output_shapefile_centroids}")

    # Merge polygons into a single GeoDataFrame
    merged_polygons = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=gdf.crs)
    output_shapefile_polygons = os.path.join(output_folder, "merged_polygons.shp")
    merged_polygons.to_file(output_shapefile_polygons)
    print(f"All polygons have been merged and saved to {output_shapefile_polygons}")

    # Save shapefile data to CSV
    columns = ["Filename", "Number of Polygons", "Number of Centroids", "Min Size", "Average Size (Hectares)", "Variance in Size (Hectares)"]
    df = pd.DataFrame(shapefile_data, columns=columns)
    output_csv = os.path.join(output_folder, "stats.csv")
    df.to_csv(output_csv, index=False)
    print(f"Statistics saved to {output_csv}\n")

    pprint.pprint(df)


def create_highconf_polygons(inputParams, dataFolder, min_size):
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
    # print out the paths to all the inputs
    print("INPUTS: {} \n".format(inputParams))

    print("\n Step 1: Preprocessing raw data... \n")
    lulc, sdpt, ttc = preprocess_datasets(dataFolder)

    df = pd.read_csv(inputParams)
    categoryList = df['Category'].unique().tolist()
    print("\n Step 2: Finding Unique values in column 'Category':\n", categoryList)

    print("\n Step 3: Executing classifyRasters function for each unique category:\n")
    for category in categoryList:
        category = str(category)
        print("Processing ", category, ":\n")
        try:
            classifyRasters(lulc, sdpt, ttc, inputParams, category, dataFolder)
        except:
            print("Encountered an error with " + str(category))

    print("\n Step 4: Converting high confidence rasters to polygons...\n")
    rastertoPoly(dataFolder, min_size)

    print("\n Step 5: Getting centroids from polygons...\n")
    createCentroidAndMergedPolygons(dataFolder, min_size, df)

    return None


create_highconf_polygons(**inputConfig)
