from osgeo import gdal 
from osgeo import ogr

def create_filtered_shapefile(value, filter_field, in_shapefile):

    input_layer = ogr.Open(in_shapefile).GetLayer()
    out_shapefile = f'../data/raw/{filter_field}_lulc.shp'

    # Filter by our query
    query_str = '"{}" = "{}"'.format(filter_field, value)
    input_layer.SetAttributeFilter(query_str)

    # Copy Filtered Layer and Output File
    driver = ogr.GetDriverByName('ESRI Shapefile')
    out_ds = driver.CreateDataSource(out_shapefile)
    out_layer = out_ds.CopyLayer(input_layer, str(value))
    del input_layer, out_layer, out_ds

    # save output
    
    return out_shapefile