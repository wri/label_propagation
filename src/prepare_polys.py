
## PLACEHOLDER

def create_highconf_polygons():
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

    return None