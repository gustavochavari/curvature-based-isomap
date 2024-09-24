#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering - Noise experiments

Created on Wed May 09 14:55:26 2024

"""
# Imports
import time
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from numpy import sqrt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from kisomap import KIsomap, Clustering
import json

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# First set of experiments
datasets = [skdata.fetch_openml(name='servo', version=1)     
#skdata.fetch_openml(name='car-evaluation', version=1),    
#skdata.fetch_openml(name='breast-tissue', version=2),
#skdata.fetch_openml(name='Engine1', version=1),                   
#skdata.fetch_openml(name='xd6', version=1),                 
#skdata.fetch_openml(name='heart-h', version=3),
#skdata.fetch_openml(name='steel-plates-fault', version=3)]
#skdata.fetch_openml(name='PhishingWebsites', version=1),                # 10% of the samples
#skdata.fetch_openml(name='satimage', version=1),                        # 25% of the samples
#skdata.fetch_openml(name='led24', version=1),                           # 20% of the samples
#skdata.fetch_openml(name='hayes-roth', version=2),
#skdata.fetch_openml(name='rabe_131', version=2),
#skdata.fetch_openml(name='prnn_synth', version=1),           
#skdata.fetch_openml(name='visualizing_environmental', version=2),
#skdata.fetch_openml(name='diggle_table_a2', version=2),  
#skdata.fetch_openml(name='newton_hema', version=2),   
#skdata.fetch_openml(name='wisconsin', version=2),                  
#skdata.fetch_openml(name='fri_c4_250_100', version=2),          
#skdata.fetch_openml(name='conference_attendance', version=1),       
#skdata.fetch_openml(name='tic-tac-toe', version=1),
#skdata.fetch_openml(name='qsar-biodeg', version=1),
#skdata.fetch_openml(name='spambase', version=1),                        # 25% of the samples
#skdata.fetch_openml(name='cmc', version=1),
#skdata.fetch_openml(name='heart-statlog', version=1),
]

#skdata.fetch_openml(name='cnae-9', version=1)]                          # 50-D 
#skdata.fetch_openml(name='AP_Breast_Kidney', version=1)]                # 500-D
#skdata.fetch_openml(name='AP_Endometrium_Breast', version=1)]           # 400-D
#skdata.fetch_openml(name='AP_Ovary_Lung', version=1),                   # 100-D
#skdata.fetch_openml(name='OVA_Uterus', version=1)]                      # 100-D
#skdata.fetch_openml(name='micro-mass', version=1),                      # 100-D
#skdata.fetch_openml(name='har', version=1)]                             # 10%  of the samples and 100-D       
#skdata.fetch_openml(name='eating', version=1)]                          # 100-D       
#skdata.fetch_openml(name='oh5.wc', version=1)]                          # 40-D
#skdata.fetch_openml(name='leukemia', version=1)]                        # 40-D

results = {}

for X in datasets:

    ch_kiso = []
    ch_iso = []
    ri_kiso = []
    ri_iso = []
    fm_kiso = []
    fm_iso = []
    v_kiso = []
    v_iso = []
    
    dados = X['data'].copy()
    target = X['target']
    name = X['details']['name']
    
    print('############################')
    print('Initiating ', name,' dataset')

    # Convert labels to integers
    lista = []
    for x in target:
        if x not in lista:  
            lista.append(x)     

    # Map labels to respective numbers
    rotulos = []
    for x in target:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)
    target_original = np.array(rotulos)

    # Number of samples, features and classes
    n = dados.shape[0]
    m = dados.shape[1]
    c = len(np.unique(target))

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

    # To remove NaNs
    dados = np.nan_to_num(dados)

    # OPTIONAL: set this flag to True to reduce the number of samples

    reduce_samples = False
    reduce_dim = True

    if not reduce_samples and not reduce_dim:
        raw_data = dados
    if reduce_samples:
        percentage = 0.25
        dados, garbage, target_original, garbage_t = train_test_split(dados, target, train_size=percentage, random_state=42)
        raw_data = dados
        
    # OPTIONAL: set this flag to True to reduce the dimensionality with PCA prior to metric learning
    if reduce_dim:
        num_features = 400
        raw_data = dados
        dados = PCA(n_components=num_features).fit_transform(dados)

    # Number of samples, features and classes
    n = dados.shape[0]
    m = dados.shape[1]
    # Print data info
    print('N = ', n)        
    print('M = ', m)        
    print('C = %d' %c)      
    # Number of neighbors in KNN graph (patch size)
    nn = round(np.floor(sqrt(n)))                 
    # GMM algorithm
    CLUSTER = 'gmm'
    ############## K-ISOMAP 
    # Number of neighbors
    print('K = ', nn)      

    # Data standardization (to deal with variables having different units/scales)
    dados = preprocessing.scale(dados).astype(np.float64)
      
    for r in range(11):      
        # Computes the results for all 10 curvature based metrics
        lista_ri = []
        lista_ch = []
        lista_fm = []
        lista_v = []
        inicio = time.time()

        # Noise parameters
        noise_scale = 1  # adjust the noise gaussian parameter here

        # Define magnitude
        magnitude = np.linspace(0,noise_scale,11)

        # Generate noise
        ruido = np.random.normal(0, scale=magnitude[r], size=dados.shape)
        
        # Apply noise in feature level
        dados_ruido = dados.copy() + ruido

        target = target_original.copy()
        
        for i in range(11):
            dados_kiso = KIsomap(dados_ruido, nn, 2, i)       
            L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
            lista_ri.append(L_kiso[0])
            lista_ch.append(L_kiso[1])
            lista_fm.append(L_kiso[2])
            lista_v.append(L_kiso[3])
            labels_kiso = L_kiso[4]
        fim = time.time()
        print('K-ISOMAP time: %f s' %(fim - inicio))

        # Find best result in terms of Rand index
        ri_star = max(enumerate(lista_ri), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, ri_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        #labels_kiso = L_kiso[4]
        ri_kiso.append(L_kiso[0])
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP RI')

        fm_star = max(enumerate(lista_fm), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, fm_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        #labels_kiso = L_kiso[4]
        fm_kiso.append(L_kiso[2])
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP FM')

        # Find best result in terms of Rand index
        ch_star = max(enumerate(lista_ch), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, ch_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        #labels_kiso = L_kiso[4]
        ch_kiso.append(L_kiso[1])
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP CH')

        # Find best result in terms of V measure
        v_star = max(enumerate(lista_v), key=lambda x: x[1])[0]
        dados_kiso = KIsomap(dados_ruido, nn, 2, v_star) 
        L_kiso = Clustering(dados_kiso.T, target, 'K-ISOMAP', CLUSTER)
        labels_kiso = L_kiso[4]
        v_kiso.append(L_kiso[3])
        #PlotaDados(dados_kiso, labels_kiso, 'K-ISOMAP VS')

        ############## Regular ISOMAP 
        model = Isomap(n_neighbors=nn, n_components=2)
        dados_isomap = model.fit_transform(dados_ruido)
        dados_isomap = dados_isomap.T
        L_iso = Clustering(dados_isomap, target, 'ISOMAP', CLUSTER)
        #labels_iso = L_iso[4]
        #PlotaDados(dados_isomap.T, labels_iso, 'ISOMAP')
        ri_iso.append(L_iso[0])
        ch_iso.append(L_iso[1])
        fm_iso.append(L_iso[2])
        v_iso.append(L_iso[3])

        ############## RAW DATA
        #L_ = Clustering(dados_ruido.T, target, 'RAW DATA', CLUSTER)
        #labels_ = L_[4]
        #ri_r.append(L_[0])
        #ch_r.append(L_[1])
        #fm_r.append(L_[2])
        #v_r.append(L_[3])
        
        #print('Run #',r,' complete')

    results[name] = {"KISOMAP":[ri_kiso,ch_kiso,fm_kiso,v_kiso],
                     "ISOMAP":[ri_iso,ch_iso,fm_iso,v_iso]}
    
    print('Dataset ', name,' complete')
    print()

# Save results
with open('results_'+name+'_noise.json', 'w') as f:
    json.dump(results, f)

#################################################
# Plot results

fig, axs = plt.subplots(6, 4, figsize=(15, 10))

datasets = [#'servo', 
#'car-evaluation', 
'breast-tissue', 
'Engine1', 
'xd6', 
#'heart-h', 
#'steel-plates-fault',
#'hayes-roth',
#'rabe_131',        
#'visualizing_environmental',
#'diggle_table_a2',
#'newton_hema',  
#'wisconsin',                 
#'fri_c4_250_100',         
'conference_attendance',     
'tic-tac-toe',
#'qsar-biodeg',
'cmc',
#'heart-statlog'
]

datasets_2 = [
'cnae-9',                    
#'AP_Breast_Kidney',    
#'AP_Endometrium_Breast',        
#'AP_Ovary_Lung',               
'OVA_Uterus',              
#'micro-mass',                  
'har',                        
'eating',                      
'oh5.wc',                        
'leukemia']                        

metrics = ['Rand Index', 'Calinski-Harabasz Score', 'Fowlkes-Mallow Index', 'V Score']
methods = ['KISOMAP', 'ISOMAP']

for i, dataset in enumerate(datasets):
    for j, metric in enumerate(metrics):
        ax = axs[i, j]  
        for method in methods:
            ax.plot(magnitude, results[dataset][method][j], label=method)
            
        if j == 0:
            ax.set_ylabel(dataset)  # Set the y label here
            plt.setp(ax.get_yticklabels(), visible=True)
        else:
            plt.setp(ax.get_yticklabels(), visible=True) 
        if i == 0:
             ax.set_title(metric)  
        if i == 0 and j==3:
            ax.legend() 
             
plt.savefig('results_1_battery.jpg',dpi=300,format='jpeg')
plt.show()