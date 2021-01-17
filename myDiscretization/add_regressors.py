import os

import pandas as pd
import geopandas as gpd

sys.path.append('..')
from config import *

# - file inputs example
regressor_area = gpd.read_file(os.path.join(ENTR_DATA_PATH,'Bairros_rio_de_janeiro.shp'))
regressor_area = regressor_area.set_crs(epsg=29183)
regressor_area = regressor_area.to_crs(epsg=4326)
regressor = regressor_area[['area','geometry']]
regressor.head()

df = gpd.read_file(os.path.join(ENTR_DATA_PATH,'sh','sh.shp'))
df.rename({'h3_index':'discr_id'},axis=1,inplace=True)
discretization = df[['discr_id','geometry']].copy()

def addRegressorWeithedAverage(df : gpd.GeoDataFrame, regressor_df : gpd.GeoDataFrame) -> gpd.GeoDataFrame:


    '''
    Apply parameters from regressor database to geographic discretization as a wieghted average of the areas of intersection.

    Parameters
        df : gpd.GeoDataFrame - geodataframe containing the final geographic discretization and a columns 'discr_id' containing unique IDs.
        regressor_df : gpd.GeoDataFrame - geodataframe containing the desired regressors and geometries defining geographic boundaries.

    Returns
        overlay_df : gpd.GeoDataFrame - geodataframe with same structure as df with new columns of applied regressors.

    '''

    # merge calculating area overlays
    overlay_df = gpd.overlay(discretization,regressor,how='intersection').rename({'geometry':'geometry_overlay'},axis=1)
    # merge with df to get original discretization
    overlay_df = pd.merge(
        overlay_df,
        df,
        on='discr_id',
        how='outer'
    ).rename({'geometry':'geometry_discr'},axis=1)

    # complete area zero for non overlay dicretizations
    overlay_df['geometry_overlay_area'] = overlay_df['geometry_overlay'].area.copy()
    overlay_df.loc[overlay_df['geometry_overlay'].isna(),'geometry_overlay_area'] = 0

    # calculate percentage of areas on each overlay
    overlay_df['overlay_area_percentage'] = overlay_df['geometry_overlay_area'] / overlay_df['geometry_discr'].area

    # apply regressor columns multiplying merged values with calculated percentages
    regressor_cols = [col for col in regressor_df.columns if not(col in ['geometry','regr_id'])]
    for regressor_col in regressor_cols:
        overlay_df[regressor_col] *= overlay_df['overlay_area_percentage']

    # sum weithed partials and recover original discretization geometries
    discr_geometries = df.set_index('discr_id')['geometry']
    overlay_df = overlay_df.groupby(['discr_id'])[regressor_cols].sum()
    overlay_df['geometry'] = discr_geometries

    return overlay_df.reset_index()

def addRegressorUniformDistribution(df : gpd.GeoDataFrame, regressor_df : gpd.GeoDataFrame) -> gpd.GeoDataFrame:


    '''
    Apply parameters from regressor database to geographic discretization as a uniform distribution of parameters in regressors area.

    Parameters
        df : gpd.GeoDataFrame - geodataframe containing the final geographic discretization and a columns 'discr_id' containing unique IDs.
        regressor_df : gpd.GeoDataFrame - geodataframe containing the desired regressors and geometries defining geographic boundaries.

    Returns
        overlay_df : gpd.GeoDataFrame - geodataframe with same structure as df with new columns of applied regressors.

    '''

    # create regressors ID column
    regressor_df['regr_id'] = list(range(len(regressor_df)))

    # merge calculating area overlays
    overlay_df = gpd.overlay(discretization,regressor,how='intersection').rename({'geometry':'geometry_overlay'},axis=1)
    # merge with df to get original discretization
    overlay_df = pd.merge(
        overlay_df,
        regressor_df[['regr_id','geometry']],
        on='regr_id',
        how='left'
    ).rename({'geometry':'geometry_regr'},axis=1)

    # calculate percentage of regressor areas on each overlay
    overlay_df['overlay_area_percentage'] = overlay_df['geometry_overlay'].area / overlay_df['geometry_regr'].area

    # apply regressor columns multiplying merged values with calculated percentages
    regressor_cols = [col for col in regressor_df.columns if not(col in ['geometry','regr_id'])]
    for regressor_col in regressor_cols:
        overlay_df[regressor_col] *= overlay_df['overlay_area_percentage']

    # sum weithed partials and recover original discretization geometries
    discr_geometries = df.set_index('discr_id')['geometry']
    overlay_df = overlay_df.groupby(['discr_id'])[regressor_cols].sum()
    overlay_df['geometry'] = discr_geometries
    
    return overlay_df.reset_index()

r2 = addRegressorUniformDistribution(discretization,regressor)