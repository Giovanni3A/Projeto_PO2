import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import *
from utils import *

df = pd.read_pickle(os.path.join(TRTD_DATA_PATH,'eventos.pkl'))
bairro_to_idx = df[['i','nome_bairro']].drop_duplicates().set_index('nome_bairro')['i'].to_dict()
lista_bairros = bairro_to_idx.values()

r = []
for i in tqdm(lista_bairros,desc='Calculando distribuições'):
    d = calcular_distribuicao(df,12,1,i,1,tipo_discretizacao_temp=0)
    for j in d:
        r.append([i,j])

r_df = pd.DataFrame(r,columns=['bairro','y'])
r_df.to_csv(os.path.join(TRTD_DATA_PATH,'distribuicoes.csv'),index=False)