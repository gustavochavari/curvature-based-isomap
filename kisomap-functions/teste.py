
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
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score
from kisomap_latest import PlotaDados, KIsomap, Clustering
from kisomap import TotalCurvatureIsomap, GeodesicIsomap, myIsomap
import pandas as pd
import json

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')
import repliclust 

archetype = repliclust.Archetype(
                    dim=3,
                    n_samples=500,
                    max_overlap=0.00001, min_overlap=0.000004,name="oblong"
                    )
X, y, _ = (repliclust.DataGenerator(archetype).synthesize(quiet=True))

# Parameters
noise_scale = 0  # adjust the noise gaussian parameter here

# Add noise
# Definir a magnitude do ruído gaussiano
magnitude = np.linspace(0,4,26)
# Gerar ruído gaussiano com média zero e desvio padrão baseado na magnitude
ruido = np.random.normal(0, scale=magnitude[0], size=X.shape)
# Adicionar o ruído aos dados
# Surface equation
x1 = X.T[0] + ruido.T[0]
y1 = X.T[1] + ruido.T[1]
z1 = X.T[2] + ruido.T[2]

result_var_ri = []
result_var_ch = []
result_var_fm = []

for q in range(1):

    #if type(X['data']) == sp.sparse._csr.csr_matrix:
    #    X['data'] = X['data'].todense()
    #    X['data'] = np.asarray(X['data'])

    #if not isinstance(dados, np.ndarray):
    #    cat_cols = X['data'].select_dtypes(['category']).columns
    #    X['data'][cat_cols] = X['data'][cat_cols].apply(lambda x: x.cat.codes)
    #    # Convert to numpy
    #    X['data'] = X['data'].to_numpy()

    #X['data'] = PCA(n_components=100).fit_transform(X['data'])


    ch_kiso = []
    fm_kiso = []
    ri_kiso = []
    v_kiso = []

    ch_iso = []
    fm_iso = []
    ri_iso = []
    v_iso = []

    ch_iso_r = []
    fm_iso_r = []
    ri_iso_r = []
    v_iso_r = []
    
    for counter in range(26):

        dados = X.copy() 
        target = y

        #Convert labels to integers
        #lista = []
        #for x in target:
        #    if x not in lista:  
        #        lista.append(x)     
        ##Map labels to respective numbers
        #rotulos = []
        #for x in target:  
        #    for i in range(len(lista)):
        #        if x == lista[i]:  
        #            rotulos.append(i)
        #target = np.array(rotulos)

        # Some adjustments are require in opnML datasets
        # Categorical features must be encoded manually
        if type(dados) == sp.sparse._csr.csr_matrix:
            dados = dados.todense()
            dados = np.asarray(dados)

        if not isinstance(dados, np.ndarray):
            cat_cols = dados.select_dtypes(['category']).columns
            dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
            # Convert to numpy
            dados = dados.to_numpy()

        # Number of samples, features and classes
        n = dados.shape[0]
        m = dados.shape[1]
        c = len(np.unique(target))

        # To remove NaNs
        dados = np.nan_to_num(dados)

            # Definir a magnitude do ruído gaussiano
        magnitude = np.linspace(0,np.max(np.std(dados,axis=0)),26)

        # Gerar ruído gaussiano com média zero e desvio padrão baseado na magnitude
        ruido = np.random.normal(0, scale=magnitude[counter], size=dados.shape)
        
        # Adicionar o ruído aos dados
        dados_ruido = dados.copy() + ruido.copy()

        # Data standardization (to deal with variables having different units/scales)
        dados_ruido = preprocessing.scale(dados_ruido.copy())

        # Adicionar outliers em alguma linha

        # OPTIONAL: set this flag to True to reduce the number of samples
        reduce_samples = False
        reduce_dim = False

        if not reduce_samples and not reduce_dim:
            raw_data = dados_ruido

        if reduce_samples:
            percentage = 0.10
            dados_ruido, garbage, target, garbage_t = train_test_split(dados_ruido, target, train_size=percentage, random_state=42)
            raw_data = dados_ruido

        # OPTIONAL: set this flag to True to reduce the dimensionality with PCA prior to metric learning
        if reduce_dim:
            num_features = 100
            raw_data = dados_ruido
            dados_ruido = PCA(n_components=num_features).fit_transform(dados_ruido)

        # Number of samples, features and classes
        n = dados_ruido.shape[0]
        m = dados_ruido.shape[1]

        # Print data info
        #print('N = ', n)        
        #print('M = ', m)        
        #print('C = %d' %c)      
        # Number of neighbors in KNN graph (patch size)
        nn = round(sqrt(n))                 

        # GMM algorithm
        CLUSTER = 'gmm'


        ############## K-ISOMAP 
        # Number of neighbors
        #print('K = ', nn)      
        #print()


        # Computes the results for all 10 curvature based metrics
        lista_ri = []
        lista_ch = []
        lista_fm = []
        lista_v = []
        inicio = time.time()
        for i in range(11):
            dados_kiso = KIsomap(dados_ruido, nn, 2, i)       
            L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
            lista_ri.append(L_kiso[0])
            lista_ch.append(L_kiso[1])
            lista_fm.append(L_kiso[2])
            lista_v.append(L_kiso[3])
            labels_kiso = L_kiso[4]
        fim = time.time()
        #print('K-ISOMAP time: %f s' %(fim - inicio))
        #print()

        # Find best result in terms of Rand index
        #print('*********************************************')
        #print('******* SUMMARY OF THE RESULTS **************')
        #print('*********************************************')
        #print()
        #print('Best K-ISOMAP result in terms of Rand index')
        #print('----------------------------------------------')
        ri_star = max(enumerate(lista_ri), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, ri_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP RI')
        ri_kiso.append(max(lista_ri))

        #print('Best K-ISOMAP result in terms of Fowlkes Mallows')
        #print('----------------------------------------------')
        fm_star = max(enumerate(lista_fm), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, fm_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP FM')
        fm_kiso.append(max(lista_fm))

        # Find best result in terms of Rand index
        #print('Best K-ISOMAP result in terms of Calinski-Harabasz')
        #print('-----------------------------------------------------')
        ch_star = max(enumerate(lista_ch), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, ch_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP CH')
        ch_kiso.append(max(lista_ch))

        # Find best result in terms of V measure
        #print('Best K-ISOMAP result in terms of V measure')
        #print('-----------------------------------------------------')
        v_star = max(enumerate(lista_v), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, v_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP VS')
        v_kiso.append(max(lista_v))

        ############## Regular ISOMAP 
        #print('ISOMAP result')
        #print('---------------')
        model = Isomap(n_neighbors=nn, n_components=2)
        dados_isomap = model.fit_transform(dados_ruido)
        L_iso = Clustering(dados_isomap.T, target, 'ISOMAP', CLUSTER)
        labels_iso = L_iso[4]
        #PlotaDados(dados_isomap, labels_iso, 'ISOMAP')
        ri_iso.append(L_iso[0])
        ch_iso.append(L_iso[1])
        fm_iso.append(L_iso[2])
        v_iso.append(L_iso[3])


        ############## RAW DATA
        #print('RAW DATA result')
        #print('-----------------')
        L_ = Clustering(raw_data.T, target, 'RAW DATA', CLUSTER)
        labels_ = L_[4]
        ri_iso_r.append(L_[0])
        ch_iso_r.append(L_[1])
        fm_iso_r.append(L_[2])
        v_iso_r.append(L_[3])

        result_var_ri.append([np.var(ri_kiso),np.var(ri_iso),np.var(ri_iso_r)])
        result_var_ch.append([np.var(ch_kiso),np.var(ch_iso),np.var(ch_iso_r)])
        result_var_fm.append([np.var(fm_kiso),np.var(fm_iso),np.var(fm_iso_r)])
        print(counter)

        
    #print(X['url'], 'done!')