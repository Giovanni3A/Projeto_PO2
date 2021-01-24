
from geopandas import GeoDataFrame, read_file

from .h3_utils import generate_H3_discretization
from .travel_times.graphhopper import distance_matrix_from_gdf

'''
This file defines functions for ease of use, redirecting to the correct implementation in other files
'''

def generate_discretization(gdf, shape = 'hexagons', h3_discretization_level = 6, calculate_distance_matrix = False, export_friendly = False):
    '''
    Generate an enriched, discretized GeoDataFrame from the original geodataframe. The GeoDataFrame returned should work seamlessly with
        other functions provided within this module.

    Params:
        gdf : (string, GeoDataFrame) - a path to a GeoDataFrame or a GeoDataFrame object
        shape : ('rectangles', 'hexagons', 'none', False) - the shape in which the space should be discretized. If 'none' or False, no discretization is done
        h3_discretization_level - if using 'hexagons', this sets the resolution level passed to the H3 library. A bigger number means smaller hexagons. Valid range [0,15]
        calculate_distance_matrix : (bool) - whether we should calculate and return a distance matrix for the cells. If True, this will spend credits on Graphhopper
        export_friendly : (bool) - if True, the returned geodataframe is transformed to contain only columns that can be easily exported
    '''
    if(isinstance(gdf, str)):
        gdf = geopandas.read_file(gdf).reset_index()
    elif (isinstance(gdf, GeoDataFrame)):
        pass
    else:
        raise TypeError

    discretized_gdf = None
    if(shape == 'hexagons'):
        discretized_gdf = generate_H3_discretization(gdf, h3_discretization_level)
    elif(shape == 'rectangles'):
        raise NotImplementedError
    elif shape == 'none' or shape == False:
        discretized_gdf = gdf
        raise NotImplementedError
        #fill missing columns: center_lon, center_lat, neighbors, area

    #fills center_lat and center_lon
    centers = [shape.centroid for shape in discretized_gdf.geometry]
    discretized_gdf['center_lat'] = [p.y for p in centers]
    discretized_gdf['center_lon'] = [p.x for p in centers]

    if not 'area' in discretized_gdf:
        #fills area
        area = [shape.area for shape in discretized_gdf.geometry]
        discretized_gdf['area'] = area
  
        
    if export_friendly:
        discretized_gdf = to_export_friendly(discretized_gdf)

    #set desired order of columns
    col = ['geometry', 'area', 'center_lat', 'center_lon', 'neighbors']
    col = col + [c for c in discretized_gdf.columns if c not in col] # but don't drop any
    discretized_gdf = discretized_gdf.reindex(columns=col)


    distance_matrix = None
    if calculate_distance_matrix:
        distance_matrix = distance_matrix_from_gdf(discretized_gdf)
    
    return discretized_gdf, distance_matrix


def to_export_friendly(gdf : GeoDataFrame):
    '''
        Takes the familiar format for gdf and transforms it into an io friendly format
    '''

    #transform list into string separated by -
    temp_neighbors = ['-'.join(map(str, lista)) for lista in gdf['neighbors']]
    gdf['neighbors'] = temp_neighbors

    return gdf

    


def from_export_friendly(gdf : GeoDataFrame):
    '''
        Detransforms the transformation done by to_export_friendly
    '''

    temp_neighbors = [map(int, mystring.split(','))]
    gdf['neighbors'] = temp_neighbors

