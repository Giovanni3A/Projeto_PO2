from geopandas import GeoDataFrame
import h3
from shapely.geometry import Polygon


def generateH3Discretization(gdf : GeoDataFrame, resolution : int = 7):
    
    '''
    Generate a hexagonal discretization of the area using Uber's H3 library, and returns a new GeoDataFrame with it. 

    Parameters
        param gdf        : GeoDataFrame - the original geodataframe whose boundaries will be considered to generate the discretization
        param resolution : int          - the desired resolution level passed to the H3 library. A bigger number means smaller hexagons. Valid range [0,15]
        return : GeoDataFrame - a new GeoDataFrame containing the H3 hexagons that approximately cover the original region 
    '''

    if(resolution < 0 or resolution > 15):
        raise ValueError("resolution must be in range [0, 15]. Got {}".format(resolution))

    gdf = gdf.to_crs(epsg=4326) #convert coordinate system to lat long

    #maybe do some optional preprocessing to remove unreachable regions such as islands

    convex_hull = gdf.geometry.convex_hull
    temp_list = list(convex_hull[0].exterior.coords) #in the convex hull, there is only one polygon:

    #we have to manually convert from a list of list of tuples to a list of list of lists (and also invert lat, long ordering)
    coords = []
    for (long,lat) in temp_list:
        coords.append([lat,long])

    #we dont want the loop around here
    coords.pop()

    geoJson = {'type': 'Polygon', 'coordinates': [coords] }

    hex_indexes = list(h3.polyfill(geoJson, resolution)) #h3.polyfill is the important method in the H3 library that does the heavy work of finding a good hex-cover

    polygons = []
    for hex in hex_indexes:
        #send a warning if there is a pentagon in the study region!
        if h3.h3_is_pentagon(hex):
            warn('A H3 cell in the study region is a pentagon. See H3\'s documentation for further details.')
        
        #once again: h3 uses 2-tuples for points, but shapely uses 2-lists
        hex_coords_h3 = h3.h3_set_to_multi_polygon([hex], geo_json = False)
        hex_coords_sh = []
        for coords in hex_coords_h3[0]:
            for coord in coords:
                hex_coords_sh.append([coord[1], coord[0]]) #and in a differente (lat,long) order
        
        #do i need to loop back, to close the polygon?
        pol = Polygon(hex_coords_sh)
        polygons.append(pol)

    #is there an other relevant info that could be calculated here?
    temp_dict = {'geometry': polygons, 'h3_index':hex_indexes}
    return GeoDataFrame(temp_dict, crs="EPSG:4326") #crs="EPSG:4326" -> (lat, long) coordinates
