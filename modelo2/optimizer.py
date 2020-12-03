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
        self.X = dict()
        self.calcular_todas_distribuicoes(df,dists_df)
        self.beta = dict()
        self.mu = 0.
        self.L = dict()
        self.loss = dict()

    def calcular_todas_distribuicoes(self,df,dists_df):
        X_s = df.groupby('i')['area'].max() * 1e-6
        # X_s = (X_s - X_s.mean()) / X_s.std()
        # X_s = X_s + np.abs(X_s.min()) + 1
        for i in self.lista_bairros:
            self.Y[i] = dists_df[i].values#calcular_distribuicao(df,t,g,i,p,tipo_discretizacao_temp=0)
            self.X[i] = X_s[i]

    def calcular_funcao_custo(self):
        for bairro in self.lista_bairros:
            # pegar distribuicao
            y = self.Y[bairro]
            b = self.beta[bairro]
            x = self.X[bairro]
            l = self.lamb[bairro]
        
            left_loss = sum([l-yi*np.log(l) for yi in y])
            right_loss = self.alpha * sum([(b - self.beta[id_vizinho])**2 for id_vizinho in self.bairro_to_vizinhos[bairro]])
            self.loss[bairro] = left_loss + right_loss
        return sum(list(self.loss.values()))

    def gera_lamb_inicial(self):
        for k in self.Y.keys():
            self.beta[k] = 1e-6
        return self.beta,self.mu

    def max_likelihood_grad(self, lamb, beta, mu, X):

        self.lamb = lamb
        self.beta = beta
        self.mu = mu
        grad_beta = dict()
        grad_mu = 0
        for bairro in self.lista_bairros:
            # pegar distribuicao
            y = self.Y[bairro]
            b = beta[bairro]
            x = X[bairro]
            l = self.lamb[bairro]

            left_loss = sum([l-yi*np.log(l) for yi in y])
            right_loss = self.alpha * sum([(b - beta[id_vizinho])**2 for id_vizinho in self.bairro_to_vizinhos[bairro]])
            self.loss[bairro] = left_loss + right_loss
            
            left_side_beta = sum([x-x*yi/l for yi in y])
            right_side_beta = 2 * self.alpha * sum([b - beta[id_vizinho] for id_vizinho in self.bairro_to_vizinhos[bairro]])
            grad_beta[bairro] = left_side_beta + right_side_beta

            left_side_mu = sum([1-yi/l for yi in y])
            grad_mu += left_side_mu

        return grad_beta,grad_mu

    
