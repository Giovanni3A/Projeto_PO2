from geopandas import GeoDataFrame
import folium
import h3

def visualize_h3(gdf : GeoDataFrame):
    '''
    Visualize gdf's region using folium. 

    Parameters
        param gdf        : GeoDataFrame - a geodataframe in a very specific format, ie. the one returned by the generateH3Discretization function

        return           : a folium map
    '''
    hexagons = gdf['h3_index']
    #for index in range(len(gdf)):
    #    for i in range(5, 5+6):
    #        if gdf.iloc[index, i] != None:
    #            hexagons.append(gdf.iloc[index, i])
    m = visualize_hexagons(hexagons, color = 'red')
    return m
    

def visualize_h3_index(gdf : GeoDataFrame, index : int):
    '''
    Visualize only the hexagon given by index in a given gdf region, using folium. 

    Parameters
        param gdf        : GeoDataFrame - a geodataframe in a very specific format, ie. the one returned by the generateH3Discretization function
        param index      : int          - the index of the desired hexagon in gdf

        return           : a folium map
    '''

    neighbors = []
    for i in range(5, 5+6):
        if gdf.iloc[index, i] != None:
            neighbors.append(gdf.iloc[index, i])

    m = visualize_hexagons(neighbors, color = 'blue')
    m = visualize_hexagons([gdf['h3_index'][index]], color = 'red', folium_map = m)
    return m




def visualize_hexagons(hexagons, color="red", folium_map=None):
    """
    hexagons is a list of hexcluster. Each hexcluster is a list of hexagons. 
    eg. [[hex1, hex2], [hex3, hex4]]
    """
    polylines = []
    lat = []
    lng = []
    for hex in hexagons:
        polygons = h3.h3_set_to_multi_polygon([hex], geo_json=False)
        # flatten polygons into loops.
        outlines = [loop for polygon in polygons for loop in polygon]
        polyline = [outline + [outline[0]] for outline in outlines][0]
        lat.extend(map(lambda v:v[0],polyline))
        lng.extend(map(lambda v:v[1],polyline))
        polylines.append(polyline)
    
    if folium_map is None:
        m = folium.Map(location=[sum(lat)/len(lat), sum(lng)/len(lng)], zoom_start=13, tiles='cartodbpositron')
    else:
        m = folium_map
    for polyline in polylines:
        my_PolyLine=folium.PolyLine(locations=polyline,weight=8,color=color)
        m.add_child(my_PolyLine)
    return m
    

def visualize_polygon(polyline, color):
    polyline.append(polyline[0])
    lat = [p[0] for p in polyline]
    lng = [p[1] for p in polyline]
    m = folium.Map(location=[sum(lat)/len(lat), sum(lng)/len(lng)], zoom_start=13, tiles='cartodbpositron')
    my_PolyLine=folium.PolyLine(locations=polyline,weight=8,color=color)
    m.add_child(my_PolyLine)
    return m