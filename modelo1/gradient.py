import os,sys
import numpy as np
import pandas as pd
from optimizer import optimizer
from tqdm import tqdm
from utils import calcular_distribuicao

sys.path.append('../')
from config import *

#função que deve ser minimizada

def next_is_close_to_prev(xt1, xt, delta) -> bool:
    D = np.linalg.norm(np.array(list(xt1.values())) - np.array(list(xt.values())))
    # print('Delta:',D)
    return D < delta

def projected_gradient(x_k: np.array, epsilon: float, delta: float, opt):

    k = 1
    while True:
        # print('\nIteração {}...'.format(k),end='')
        prev_x = x_k.copy()
        # calcular gradiente
        d_k = opt.max_likelihood_grad(x_k)
        # projetar lambdas
        d_norm = np.linalg.norm(np.array(list(d_k.values())))
        for bairro in opt.lista_bairros:
            x = x_k[bairro]
            d = d_k[bairro]
            x_k[bairro] = max(x - (2/k)*(d/d_norm),epsilon)
        # print('ok')
        # print('Norma do gradiente:',d_norm)
        # print('Função de custo:',sum(list(opt.loss.values())))

        if next_is_close_to_prev(prev_x, x_k, delta):
            opt.max_likelihood_grad(x_k)
            break
        else:
            k += 1
    return pd.DataFrame(list(opt.loss.items()),columns=['bairro_id','valor_custo']),pd.DataFrame(list(x_k.items()),columns=['bairro_id','lambda'])

def otimizar_modelo(dists,alpha):
    # instanciar otimizador
    opt = optimizer(df,vizinhancas,dists,alpha,12,1,1)

    # otimizar
    loss_df,lambdas_df = projected_gradient(
        x_k=opt.gera_lamb_inicial(),
        epsilon=1e-6,
        delta=.001,
        opt=opt
        )

    return opt,loss_df,lambdas_df

def cross_validate(k):

    alphas = [0,.01,.05,.1,.2,.25,.5,1.,2.,5.,10.,20.,50.]
    losses = []
    for alpha in alphas:
        alpha_losses = []

        # separar distribuicoes de eventos
        N = int(dists_df.shape[0]/k)
        folds = [dists_df.iloc[i*N:(i+1)*N,:].copy() for i in range(k)]

        for ik in tqdm(range(k),desc='avaliando alpha={}'.format(alpha)):
            # pegar dados de treino
            ifolds = [folds[i] for i in range(k) if i != ik]
            idists = pd.concat(ifolds)

            # calcular funcao de custo para dados de teste
            opt,loss_df,lambdas_df = otimizar_modelo(idists,alpha)
            opt.calcular_todas_distribuicoes(folds[ik])
            loss = opt.calcular_funcao_custo()

            alpha_losses.append(loss)
        
        # calcular erro médio de alpha
        V = np.mean(alpha_losses)
        print('Erro de validação médio:',V,'\n')
        losses.append([alpha,V])
    return losses


if __name__ == '__main__':
    # leitura dos dados
    df = pd.read_pickle(os.path.join(TRTD_DATA_PATH,'eventos.pkl'))
    vizinhancas = pd.read_csv(os.path.join(ENTR_DATA_PATH,'bairros_vizinhos.csv'),sep=';',encoding='latin-1',index_col=0)
    d_df = pd.read_csv(os.path.join(TRTD_DATA_PATH,'distribuicoes.csv'))

    dists_df = pd.DataFrame()
    for b in d_df['bairro'].unique():
        dists_df[b] = d_df[d_df['bairro']==b]['y'].values


    # dropar paquetá
    vizinhancas.drop(0,inplace=True)
    df = df[df['i']!=0]

    # validação cruzada
    losses = cross_validate(5)
    losses.sort(key=lambda x:x[1])
    alpha = losses[0][0]
    print('Melhor alpha:',alpha)

    # resultado final
    opt,loss_df,lambdas_df = otimizar_modelo(dists_df,alpha)
    lambdas_df.to_csv(os.path.join(REST_DATA_PATH,'lambdas_modelo1_alpha_{}.csv'.format(alpha)),index=False)