#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Aprendizado não supervisionado de métricas utilizando geometria diferencial e o algoritmo ISOMAP no agrupamento de dados

Created on Wed Jul 24 16:59:33 2019

Modified on Sun Jul 31 19:21:56 2023

@authors: Alexandre L. M. Levada, Gustavo H. Chavari

"""

# Imports
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import calinski_harabasz_score
from scipy.sparse.csgraph import dijkstra as graph_shortest_path

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def PCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados

# ISOMAP implementation
def myIsomap(dados, k, d):
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()
    D  = graph_shortest_path(A, directed=False, return_predecessors=False)
    n = D.shape[0]
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output

# K-ISOMAP implementation 
# Pré-aloca dados para acelerar processamento, porém memória pode não ser suficiente em alguns casos!
def GeodesicIsomap(dados, k, d):
    
    n = dados.shape[0]
    m = dados.shape[1]

    # Componentes principais extraídos de cada patch (espaço tangente)
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    A = knnGraph.toarray()
    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            # Decomposição espectral da matriz de covariância^T
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]     # Esse é o oficial
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)

                ##### Medidas baseadas nas curvaturas principais
                # Deve-se escolher apenas uma das 10 opções a seguir a cada execução

                # Esta é a medida utilizada para resultados preliminares
                B[i, j] = norm(delta)                   # métrica A1 - Normas das curvaturas principais

                # Outras medidas podem ser consideradas, tais como
                #B[i, j] = delta[0]                     # métrica A2 - Curvatura da primeira componente principal
                #B[i, j] = delta[-1]                    # métrica A3 - Curvatura da última componente principal
                #B[i, j] = (delta[0] + delta[-1])/2     # métrica A4 - Média das curvaturas da primeira e última componentes principais
                #B[i, j] = np.sum(delta)/len(delta)     # métrica A5 - Curvatura média
                #B[i, j] = max(delta)                   # métrica A6 - Curvatura máxima
                #B[i, j] = min(delta)                   # métrica A7 - Curvatura mínima
                #B[i, j] = min(delta)*max(delta)        # métrica A8 - Produto entre a mínima e máxima curvaturas
                #B[i, j] = max(delta) - min(delta)      # métrica A9 - Curvatura máxima menos curvatura mínima
                
                ##### métrica 10 - Diferença das projeções nos espaços tangentes
                # Wi = matriz_pcs[i, :, :]
                # Wj = matriz_pcs[j, :, :]
                # ti = np.dot(Wi.T, dados[i, :])
                # tj = np.dot(Wj.T, dados[j, :])
                # B[i, j] = norm(ti - tj)
    
    # Computes geodesic distances in B
    D = graph_shortest_path(B, directed=False, return_predecessors=False)
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    #print(np.isnan(B).any())
    #print(np.isinf(B).any())
    # Pode gerar nan ou inf na matriz B
    # Remove infs e nans
    maximo = np.nanmax(B[B != np.inf])   # encontra o maior elemento que não seja inf
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    # Demora em casos de alta dimensionalidade, dividir em matrizes menores
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output



def TotalCurvatureIsomap(dados, k, d):
    
    n = dados.shape[0]
    m = dados.shape[1]

    # Componentes principais extraídos de cada patch (espaço tangente)
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    A = knnGraph.toarray()
    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            # Decomposição espectral da matriz de covariância^T
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]     # Esse é o oficial
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    B = A.copy()

    # Computes geodesic distances in B
    D, predecessors = graph_shortest_path(B, directed=False, return_predecessors=True)

    for i in range(n):

        # Reconstruir o caminho a partir dos predecessores
        path = [i]
     
        for j in range(n):
            if B[i, j] > 0:
                while path[-1] != j:
                    path.append(predecessors[j, path[-1]])

                # Inverter o caminho para obter a ordem correta
                path.reverse()
                delta = 0
                # Calcula a soma das variações das componentes espaços tangentes de X[i] até X[j] ao longo da geodésica!
                for p in range(len(path)-1):
                    delta += norm(matriz_pcs[path[p], :, :] - matriz_pcs[path[p+1], :, :], axis=0)
            
                ##### Medidas baseadas nas curvaturas principais
                # Deve-se escolher apenas uma das 10 opções a seguir a cada execução

                # Esta é a medida utilizada para resultados preliminares
                B[i, j] = norm(delta)                   # métrica A1 - Normas das curvaturas principais

                # Outras medidas podem ser consideradas, tais como
                #B[i, j] = delta[0]                     # métrica A2 - Curvatura da primeira componente principal
                #B[i, j] = delta[-1]                    # métrica A3 - Curvatura da última componente principal
                #B[i, j] = (delta[0] + delta[-1])/2     # métrica A4 - Média das curvaturas da primeira e última componentes principais
                #B[i, j] = np.sum(delta)/len(delta)     # métrica A5 - Curvatura média
                #B[i, j] = max(delta)                   # métrica A6 - Curvatura máxima
                #B[i, j] = min(delta)                   # métrica A7 - Curvatura mínima
                #B[i, j] = min(delta)*max(delta)        # métrica A8 - Produto entre a mínima e máxima curvaturas
                #B[i, j] = max(delta) - min(delta)      # métrica A9 - Curvatura máxima menos curvatura mínima
                
                ##### métrica 10 - Diferença das projeções nos espaços tangentes
                # Wi = matriz_pcs[i, :, :]
                # Wj = matriz_pcs[j, :, :]
                # ti = np.dot(Wi.T, dados[i, :])
                # tj = np.dot(Wj.T, dados[j, :])
                # B[i, j] = norm(ti - tj)
    
    # Computes geodesic distances in B
    D = graph_shortest_path(B, directed=False, return_predecessors=False)
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    #print(np.isnan(B).any())
    #print(np.isinf(B).any())
    # Pode gerar nan ou inf na matriz B
    # Remove infs e nans
    maximo = np.nanmax(B[B != np.inf])   # encontra o maior elemento que não seja inf
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output



'''
 Computes the Silhouette coefficient and the supervised classification
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
'''
def Clustering(dados, target):
    # Número de classes
    c = len(np.unique(target))

    # Kmédias 
    kmeans = KMeans(n_clusters=c, random_state=42).fit(dados.T)
    # Usar também o GMM e o DBSCAN
    
    rand = rand_score(target, kmeans.labels_)
    ca = calinski_harabasz_score(dados.T, kmeans.labels_)

    # Outras medidas de qualidades que podem ser utilizadas

    #fm = fowlkes_mallows_score(target, kmeans.labels_)
    #sc = silhouette_score(dados.T, kmeans.labels_, metric='euclidean')
    #db = davies_bouldin_score(dados.T, kmeans.labels_) 

    return [rand, ca, kmeans.labels_]
    # return [rand, fm, mi, ho, co, vm, sc, db, ca, kmeans.labels_]
        

# Plota gráficos de dispersão para o caso 2D
def PlotaDados(dados, labels, metodo):
    
    nclass = len(np.unique(labels))

    # Converte labels para inteiros
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     # contém as classes (sem repetição)

    # Mapeia rotulos para números
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)

    # Converte para vetor
    rotulos = np.array(rotulos)

    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']

    plt.figure(10)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        #cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon', 'silver', 'gold', 'darkcyan', 'royalblue', 'darkorchid', 'plum', 'crimson', 'lightcoral', 'orchid', 'powderblue', 'pink', 'darkmagenta', 'turquoise', 'wheat', 'tomato', 'chocolate', 'teal', 'lightcyan', 'lightgreen', ]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    
    nome_arquivo = metodo + '.png'
    plt.title(metodo +' clusters')

    plt.savefig(nome_arquivo)
    plt.close()

