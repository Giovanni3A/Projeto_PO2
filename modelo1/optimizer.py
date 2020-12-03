import os,sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
from config import *
from utils import calcular_distribuicao

class optimizer:

    def __init__(self, df, vizinhancas, dists_df, alpha: float, t: int, g: int, p: int):
        self.alpha = alpha
        self.df = df
        self.vizinhancas = vizinhancas

        # criar estruturas
        # map nome bairro para id bairro
        self.bairro_to_idx = self.df[['i','nome_bairro']].drop_duplicates().set_index('nome_bairro')['i'].to_dict()
        self.vizinhancas['ID_BAIRRO'] = self.vizinhancas['NOME'].map(self.bairro_to_idx)
        # map id bairro para lista de ids de vizinhos
        self.bairro_to_vizinhos = self.vizinhancas.set_index('ID_BAIRRO')['NEIGHBORS'].apply(lambda v:[self.bairro_to_idx[b.strip()] for b in v.split(',')]).to_dict()
        
        self.nobairros = len(self.bairro_to_idx)
        self.lista_bairros = self.vizinhancas['ID_BAIRRO'].unique()

        self.Y = dict()
        self.calcular_todas_distribuicoes(dists_df)
        self.L = dict()
        self.loss = dict()

    def calcular_todas_distribuicoes(self,dists_df):
        for i in self.lista_bairros:
            self.Y[i] = dists_df[i].values#calcular_distribuicao(df,t,g,i,p,tipo_discretizacao_temp=0)

    def calcular_funcao_custo(self):
        for bairro in self.lista_bairros:
            # pegar distribuicao
            y = self.Y[bairro]

            left_loss = sum([self.lamb[bairro]-yi*np.log(self.lamb[bairro]) for yi in y])
            right_loss = self.alpha * sum([(self.lamb[bairro] - self.lamb[id_vizinho])**2 for id_vizinho in self.bairro_to_vizinhos[bairro]])
            self.loss[bairro] = left_loss + right_loss
        return sum(list(self.loss.values()))

    def gera_lamb_inicial(self) -> np.array:
        for k in self.Y.keys():
            self.L[k] = 1.
        return self.L

    def max_likelihood_grad(self, lamb) -> np.array:
        '''
        Parâmetros
    ----------
        lamb : {np.array} de dimensão nº bairros
        '''
        self.lamb = lamb
        grad = dict()
        for bairro in self.lista_bairros:
            # pegar distribuicao
            y = self.Y[bairro]

            left_loss = sum([lamb[bairro]-yi*np.log(lamb[bairro]) for yi in y])
            right_loss = self.alpha * sum([(lamb[bairro] - lamb[id_vizinho])**2 for id_vizinho in self.bairro_to_vizinhos[bairro]])
            self.loss[bairro] = left_loss + right_loss
            
            left_side = sum([1-yi/(lamb[bairro]) for yi in y])
            right_side = 2 * self.alpha * sum([lamb[bairro] - lamb[id_vizinho] for id_vizinho in self.bairro_to_vizinhos[bairro]])
            grad[bairro] = left_side + right_side

        return grad

    
