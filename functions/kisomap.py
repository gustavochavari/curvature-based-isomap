#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering

Created on Wed Jul 24 16:59:33 2023

"""

# Imports
import sys
import time
import warnings
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
from scipy.sparse import issparse
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import calinski_harabasz_score
from scipy import sparse
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.cluster import (rand_score, calinski_harabasz_score, 
fowlkes_mallows_score, v_measure_score, silhouette_score, davies_bouldin_score,
adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, 
homogeneity_score, completeness_score)

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

#################################################################################
# Fast K-ISOMAP implementation - consumes more memory, but it is a lot faster
# Pre-allocate matrices to speed up performance
#################################################################################
def KIsomap(dados, k, d, option, alpha=0.5):
    # Number of samples and features  
    n = dados.shape[0]
    m = dados.shape[1]
    # Matrix to store the principal components for each neighborhood
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    A = knnGraph.toarray()
    # Computes the means and covariance matrices for each patch

    # Verificar o número de componentes conectados
    n_connected_components, labels = connected_components(knnGraph)

    # Caso o número de componentes conectados seja maior que 1
    if n_connected_components > 1:
        # Verificação de métrica 'precomputed' com matriz esparsa
        if issparse(A):
            raise RuntimeError(
                "The number of connected components of the neighbors graph"
                f" is {n_connected_components} > 1. The graph cannot be "
                "completed with metric='precomputed', and Isomap cannot be"
                " fitted. Increase the number of neighbors to avoid this "
                "issue, or precompute the full distance matrix instead "
                "of passing a sparse neighbors graph."
            )

        # Emitir aviso sobre desempenho
        warnings.warn(
            (
                "The number of connected components of the neighbors graph "
                f"is {n_connected_components} > 1. Completing the graph to fit"
                " Isomap might be slow. Increase the number of neighbors to "
                "avoid this issue."
            ),
            stacklevel=2,
        )

        # Corrigir componentes conectados
        nbg = _fix_connected_components(
            X=dados,
            graph=knnGraph,
            n_connected_components=n_connected_components,
            component_labels=labels,
            mode="distance",
            metric='euclidean',  # Substitua por sua métrica de distância
            # Adicione parâmetros adicionais de métrica se necessário
        )

        A = nbg.toarray()

        

    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:                   # Treat isolated points
            matriz_pcs[i, :, :] = np.eye(m)     # Eigenvectors in columns
        else:
            # Get the neighboring samples
            amostras = dados[indices]

            maximo = np.nanmax(amostras[amostras != np.inf])   
            amostras[np.isnan(amostras)] = 0
            amostras[np.isinf(amostras)] = maximo
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]                 
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = norm(matriz_pcs[i, :, :d+1] - matriz_pcs[j, :, :d+1], axis=0)
                ##### Functions of the principal curvatures (definition of the metric)
                # We must choose one single option for each execution
                if option == 0:
                    B[i, j] = norm(delta)                  # metric A0 - Norms of the principal curvatures
                elif option == 1:
                    B[i, j] = delta[0]                      # metric A1 - Curvature of the first principal component
                elif option == 2:
                    B[i, j] = delta[-1]                    # metric A2 - Curvature of the last principal component
                elif option == 3:
                    B[i, j] = (delta[0] + delta[-1])/2     # metric A3 - Average between the curvatures of first and last principal components
                elif option == 4:
                    B[i, j] = np.sum(delta)/len(delta)     # metric A4 - Mean curvature
                elif option == 5:
                    B[i, j] = max(delta)                   # metric A5 - Maximum curvature
                elif option == 6:
                    B[i, j] = min(delta)                   # metric A6 - Minimum curvature
                elif option == 7:
                    B[i, j] = min(delta)*max(delta)        # metric A7 - Product between minimum and maximum curvatures
                elif option == 8:
                    B[i, j] = max(delta) - min(delta)      # metric A8 - Difference between maximum and minimum curvatures
                elif option == 9:
                    B[i, j] = 1 - np.exp(-delta.mean())     # metric A9 - Negative exponential kernel
                else:
                    B[i, j] = ((1-alpha)*A[i, j]/sum(A[i, :]) + alpha*norm(delta))      # alpha = 0 => regular ISOMAP, alpha = 1 => K-ISOMAP 
                
    # Computes geodesic distances using the previous selected metric
    G = nx.from_numpy_array(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)    
    # Remove infs and nans from B (if the graph is not connected)
    maximo = np.nanmax(B[B != np.inf])   
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = np.linalg.eig(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)    
    # Return the low dimensional coordinates
    return output.real


def _fix_connected_components(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    """Add connections to sparse graph to connect unconnected components.

    For each pair of unconnected components, compute all pairwise distances
    from one component to the other, and add a connection on the closest pair
    of samples. This is a hacky way to get a graph with a single connected
    component, which is necessary for example to compute a shortest path
    between all pairs of samples in the graph.

    Parameters
    ----------
    X : array of shape (n_samples, n_features) or (n_samples, n_samples)
        Features to compute the pairwise distances. If `metric =
        "precomputed"`, X is the matrix of pairwise distances.

    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples.

    n_connected_components : int
        Number of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.

    component_labels : array of shape (n_samples)
        Labels of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.

    mode : {'connectivity', 'distance'}, default='distance'
        Type of graph matrix: 'connectivity' corresponds to the connectivity
        matrix with ones and zeros, and 'distance' corresponds to the distances
        between neighbors according to the given metric.

    metric : str
        Metric used in `sklearn.metrics.pairwise.pairwise_distances`.

    kwargs : kwargs
        Keyword arguments passed to
        `sklearn.metrics.pairwise.pairwise_distances`.

    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples, with a single connected component.
    """
    if metric == "precomputed" and sparse.issparse(X):
        raise RuntimeError(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(component_labels == j)
            Xj = X[idx_j]

            if metric == "precomputed":
                D = X[np.ix_(idx_i, idx_j)]
            else:
                D = pairwise_distances(Xi, Xj, metric=metric, **kwargs)

            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)
            if mode == "connectivity":
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj]
            else:
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    return graph


'''
 Performs clustering in data, returns the obtained labels and evaluates the clusters
'''


def Clustering(dados, target, DR_method):
    # Dicionário para armazenar as métricas de cada método de agrupamento
    resultados = {}
    
    # Número de classes
    c = len(np.unique(target))

    # Função auxiliar para calcular as métricas
    def calcular_metricas(target, labels, dados):
        try:
            ri = rand_score(target, labels)
            ch = calinski_harabasz_score(dados.T, labels)
            fm = fowlkes_mallows_score(target, labels)
            v = v_measure_score(target, labels)
            s = silhouette_score(dados.T, labels)
            db = davies_bouldin_score(dados.T, labels)
        except Exception as e:
            print(f"Erro ao calcular as métricas: {e}")
            ri = ch = fm = v = s = db = -1
        return {
            'ri': ri,
            'ch': ch,
            'fm': fm,
            'v': v,
            's': s,
            'db': db
        }

    # KMeans
    try:
        cluster = 'KMeans'
        kmeans = KMeans(n_clusters=c, random_state=42).fit(dados.T)
        labels_kmeans = kmeans.labels_
        resultados[cluster] = calcular_metricas(target, labels_kmeans, dados)
    except Exception as e:
        print(f"{DR_method} {cluster} -------- erro no agrupamento:", e)
        resultados[cluster] = calcular_metricas(target, labels_kmeans, dados)
    
    # GMM
    try:
        cluster = 'GMM'
        labels_gmm = GaussianMixture(n_components=c, random_state=42).fit_predict(dados.T)
        resultados[cluster] = calcular_metricas(target, labels_gmm, dados)
    except Exception as e:
        print(f"{DR_method} {cluster} -------- erro no agrupamento:", e)
        resultados[cluster] = calcular_metricas(target, labels_gmm, dados)
    
    # Ward (Agglomerative Clustering)
    try:
        cluster = 'Ward'
        ward = AgglomerativeClustering(n_clusters=c, linkage='ward').fit(dados.T)
        labels_ward = ward.labels_
        resultados[cluster] = calcular_metricas(target, labels_ward, dados)
    except Exception as e:
        print(f"{DR_method} {cluster} -------- erro no agrupamento:", e)
        resultados[cluster] = calcular_metricas(target, labels_ward, dados)
    
    # HDBSCAN
    #try:
    #    cluster = 'HDBSCAN'
    #    hdbscan = HDBSCAN(min_cluster_size=5).fit(dados.T)
    #    labels_hdbscan = hdbscan.labels_
    #    resultados[cluster] = calcular_metricas(target, labels_hdbscan, dados)
    #except Exception as e:
    #    print(f"{DR_method} {cluster} -------- erro no agrupamento:", e)

    return resultados


    #print(silhouette)
    #print(davies_bouldin)
    #ari = adjusted_rand_score(target, labels)
    #ami = adjusted_mutual_info_score(target, labels)
    #nmi = normalized_mutual_info_score(target, labels)
    #homogeneity = homogeneity_score(target, labels)
    #completeness = completeness_score(target, labels)
    #purity = cluster_purity(target, labels)

     #purity #ari, ami, nmi, homogeneity, completeness,

# Exemplo de uso
# result = Clustering(dados, target, DR_method='PCA', cluster='kmeans')



'''
Produces scatter plots of the 2D mappings
'''
def PlotaDados(dados, labels, metodo, CLUSTER):
    # Number of classes
    nclass = len(np.unique(labels))
    # Converts list to an array
    rotulos = np.array(labels)
    # Define colors according to the number of classes
    if nclass > 11:
        #cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', ]
        cores = list(mcolors.CSS4_COLORS.keys())
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']
    # Create figure
    plt.figure(10)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]        
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    # Save figure in image fila
    nome_arquivo = 'images/' + metodo + '_' + CLUSTER + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()


##################### Beginning of thescript

#####################  Data loading

