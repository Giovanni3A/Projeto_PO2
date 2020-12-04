import os,sys
import numpy as np
import pandas as pd
from optimizer import optimizer
from tqdm import tqdm
from utils import calcular_distribuicao

sys.path.append('../')
from config import *

from qpsolvers import solve_qp


# função de minimização quadrática
def quadratic_solver(mu, d_beta, d_ri, epsilon): #xk = (lambda, mu, beta) 
    beta = pd.Series(d_beta)
    ri   = pd.Series(d_ri)

    indx = ri.index
    beta = beta.values
    ri   = ri.values
    mu   = np.array([mu])
    i    = len(ri)

    L = mu[0] + np.multiply(beta, ri)
    xk = np.concatenate((L, mu, beta))

    P = 2*np.eye(2*i + 1)
    q = -2*xk
    G = np.diag(np.concatenate( (-np.ones(i),np.zeros(1+i)) ))
    h = -np.concatenate( (np.ones(i)*epsilon,np.zeros(i+1)) )
    A = np.concatenate((
        np.eye(i),
        -1*np.ones((i,1)),
        -ri*np.eye(i)
        ),axis=1)
    b = np.zeros(i)

    x_r = solve_qp(P, q, G, h, A, b)

    lambda_r = pd.Series(x_r[:i],index=indx).to_dict()
    mu_r = x_r[i]
    beta_r = pd.Series(x_r[-i:],index=indx).to_dict()
    return lambda_r,mu_r,beta_r

#função que deve ser minimizada
def calcular_lambdas(X,beta,mu):
    lambdas = dict()
    for k in X.keys():
        lambdas[k] = X[k]*beta[k] + mu
    return lambdas

def next_is_close_to_prev(xt1, xt, delta) -> bool:
    D = np.linalg.norm(np.array(list(xt1.values())) - np.array(list(xt.values())))
    print('Delta:',D)
    return D < delta

def projected_gradient(X ,params, epsilon: float, delta: float, opt):

    k = 1
    beta,mu = params
    x_k = calcular_lambdas(X,beta,mu)

    while True:
        print('\nIteração {}...'.format(k),end='')
        prev_x = x_k.copy()
        # calcular gradiente
        d_b,d_m = opt.max_likelihood_grad(x_k,beta,mu,X)

        d_norm = np.linalg.norm(np.array(list(d_b.values())))
        for bairro in opt.lista_bairros:
            b = beta[bairro]
            d = d_b[bairro]
            beta[bairro] = b - (.01/k)*(d/d_norm)
        mu = mu - (.01/k)*d_m/np.abs(d_m)

        x_k,mu,beta = quadratic_solver(mu,beta,X,epsilon)

        print('ok')
        print('Norma do gradiente:',d_norm)
        print('Mu:',mu)
        print('Função de custo:',sum(list(opt.loss.values())))

        if next_is_close_to_prev(prev_x, x_k, delta):
            opt.max_likelihood_grad(x_k,beta,mu,X)
            break
        else:
            k += 1
    return pd.DataFrame(list(opt.loss.items()),columns=['bairro_id','valor_custo']),pd.DataFrame(list(x_k.items()),columns=['bairro_id','lambda'])

def otimizar_modelo(dists,alpha):
    # instanciar otimizador
    opt = optimizer(df,vizinhancas,dists,alpha,12,1,1)

    # otimizar
    loss_df,lambdas_df = projected_gradient(
        X=opt.X,
        params=opt.gera_lamb_inicial(),
        epsilon=1e-10,
        delta=1e-3,
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
            opt.calcular_todas_distribuicoes(df,folds[ik])
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
    # losses = cross_validate(5)
    # losses.sort(key=lambda x:x[1])
    alpha = 0#losses[0][0]
    print('Melhor alpha:',alpha)

    # resultado final
    opt,loss_df,lambdas_df = otimizar_modelo(dists_df,alpha)
    lambdas_df.to_csv(os.path.join(REST_DATA_PATH,'lambdas_modelo2_alpha_{}.csv'.format(alpha)),index=False)