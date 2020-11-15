import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm

grupo_dias_dict = {
        0: {
            0:1,1:1,2:1,3:1,4:1,5:2,6:2
        },
        1: {
            0:1,1:1,2:1,3:1,4:1,5:2,6:3
        },
        2: {
            0:1,1:2,2:3,3:4,4:5,5:6,6:7
        }
    }

def definir_bairros(df,df_aux):
    '''
    Discretiza os dados geograficos por bairros
    
    Parâmetros
    ----------
    df : {pandas.DataFrame}
        dataframe com dados caracterizados por latitude e longitude
    df_aux : {pandas.DataFrame}
        dataframe com dados de latitude e longitude em bairros
    '''

    bairros = []
    for _,row in tqdm(df.iterrows(),total=len(df),desc='Definindo bairros das chamadas'):
        df_aux['dist_lat'] = (df_aux['lat'] - row['latitude']).abs()
        df_aux['dist_lng'] = (df_aux['lng'] - row['longitude']).abs()
        
        df_aux['dist'] = df_aux['dist_lat']**2 + df_aux['dist_lng']**2
        
        r = df_aux.sort_values('dist').iloc[0,:]
        
        if r['cidade'] == 'Rio de Janeiro':
            b = r['bairro']
        else:
            b = 'ERRO: Fora do Rio de Janeiro ({})'.format(r['cidade'])
        
        bairros.append(b)

    df['bairro'] = bairros

    return df

def geo_discretizar(df,nx=30,ny=30):
    '''
    Discretiza os dados geograficos de latitude e longitude no dataframe
    
    Parâmetros
    ----------
    df : {pandas.DataFrame}
        dataframe com dados caracterizados por latitude e longitude
    nx : {int}
        número de divisões no espaço de latitudes
    ny : {int}
        número de divisões no espaço de longitudes
    '''
    
    # latitude
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    lat_delta = (lat_max - lat_min) / nx
    
    df['discr_x'] = ((df['latitude'] - lat_min) / lat_delta).astype(int)
    
    # longitude
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    lon_delta = (lon_max - lon_min) / ny
    
    df['discr_y'] = ((df['longitude'] - lon_min) / lon_delta).astype(int)
    
    # numerar discretizações
    enum_df = df.sort_values(['discr_x','discr_y'])[['discr_x','discr_y']]\
                                .drop_duplicates().reset_index(drop=True).reset_index()
    enum_df.columns = ['geo_discr','discr_x','discr_y']
    enum_df['geo_discr'] = 'd-'+(enum_df['geo_discr']+1).astype(str)
    
    # merge para pegar enumeração
    if 'geo_discr' in df.columns:
        df.drop('geo_discr',axis=1,inplace=True)
        
    df = pd.merge(
        df,
        enum_df,
        on=['discr_x','discr_y'],
        how='left'
    )
    
    return df

def temp_discretizar(df,w='meia hora'):
    '''
    Discretiza os dados temporais no dataframe
    
    Parâmetros
    ----------
    df : {pandas.DataFrame}
        dataframe com dados caracterizados por latitude e longitude
    w : {str ou int}
        tipo de janela de tempo (a partir da primeira data às zero horas) usada na discretização:
            meia hora => janelas de 30 minutos
            hora      => janelas de 60 minutos
            dia       => janelas de 24 horas
    '''
    # o que importa é a hora do dia (em segundos)
    dt_seconds = (df['data_idx'].dt.hour*3600) + (df['data_idx'].dt.minute*60) + (df['data_idx'].dt.second*1)
    # variação de tempo em janelas de meia hora
    if w == 'meia hora':
        st = (dt_seconds / (60*30)).astype(int)
    elif w == 'hora':
        st = (dt_seconds / (60*60)).astype(int)
    elif w == 'dia':
        st = (dt_seconds / (60*60*24)).astype(int)
    elif type(w) is int:
        st = (w*dt_seconds / (60*60*24)).astype(int)
    df['t'] = st - st.min() + 1
    
    return df

def agrupar_dias(df,group=0):
    '''
    Função que agrupa dias em categorias de grupo

    Parâmetros
    ----------
    df : {pandas.DataFrame}
        Dados utilizados para agrupar. Deve possuir a coluna 'data' com formato de datetime
    group : {int}
        Tipo de agrupamento a ser feito.
            0 => semana vs fim de semana
            1 => semana vs sabado vs domingo
            2 => cada um dos 7 dias
    '''
    wd = df['data'].dt.weekday

    return wd.map(grupo_dias_dict[group])

def calcular_distribuicao(df,t,g,i,p,
                        tipo_discretizacao_temp=0,col1='t',
                        col2='grupo_dia',col3='bairro',col4='TipoViatura',
                        dt_range=None):
    '''
    Calcula a distribuição de número de chamadas para um conjunto de filtros

    Parâmetros
    ----------
    df : {pandas.DataFrame}
        Dataframe com chamadas.
    t : {int}
        Período de tempo do dia (ex: 30 em 30 min).
    g : {int}
        Grupo de dias (ex: dias de semana).
    i : {int}
        Grupo geográfico discretizado.
    p : {int}
        Indicador de prioridade (1 a 3).

    cols: {str's}
        Nome das colunas em df para match dos valores
    '''
    filtro = df.copy()
    # aplicar filtros
    filtro = filtro[
        (filtro['t']==t) &
        (filtro['grupo_dia']==g) &
        (filtro['bairro']==i) &
        (filtro['TipoViatura']==p)
    ]
    
    # pegar datas com chamadas
    datas = filtro.groupby('data')['hora'].count()
    
    # adicionar datas sem chamadas (com zero)
    if dt_range is None:
        dt_range = pd.date_range(df['data'].min(),df['data'].max(),freq='D')
        
    for dt in dt_range:

        # se a data passa no filtro de grupo
        if grupo_dias_dict[tipo_discretizacao_temp][dt.weekday()] == g:
            if not(dt in datas.index):
                datas[dt] = 0

    datas.sort_index(inplace=True)

    return datas.values