"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering - Noise experiments

Created on Wed May 09 14:55:26 2024
Modified on Mon Nov 25 13:23:36 2024

"""
# Imports
import time
import warnings
import numpy as np
import scipy as sp
import sklearn.datasets as skdata
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import umap
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import skdim
from kisomap import PlotaDados, KIsomap, Clustering
import umap
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from umap import UMAP                 
from kisomap import KIsomap, Clustering, PlotaDados, ConstrainedKIsomap
from make_datasets import HopfLink, RepliclustArchetype, COIL20_data, PenDigits_data, MNIST_data, FMNIST_data, Olivetti_data, APOmentum_data
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.manifold import LocallyLinearEmbedding

# To avoid unnecessary warning messages
# warnings.simplefilter(action='ignore')

CLUSTER = 'GMM'

#hopf_link_d, hopf_link_t = HopfLink(noise=True)
repliclust_d, repliclust_t = RepliclustArchetype()
#coil20_d, coil20_t = COIL20_data()
#mnist_d, mnist_t = MNIST_data()
#fmnist_d, fmnist_t = FMNIST_data()
#olivetti_d, olivetti_t = Olivetti_data()
#ap_omentum_d, ap_omentum_t = APOmentum_data()
#pen_digits_d, pen_digits_t = PenDigits_data()

#hopf_link = {'data': hopf_link_d, 'target': hopf_link_t, 'details': {'name':'Hopf-Link'}}
repliclust_archetype = {'data': repliclust_d, 'target': repliclust_t, 'details': {'name':'Repliclust-Archetype'}}
#coil_20 = {'data': coil20_d, 'target': coil20_t, 'details': {'name':'COIL-20'}}
#mnist = {'data': mnist_d, 'target': mnist_t, 'details': {'name':'MNIST'}}
#fmnist = {'data': fmnist_d, 'target': fmnist_t, 'details': {'name':'F-MNIST'}}
#olivetti = {'data': olivetti_d, 'target': olivetti_t, 'details': {'name':'Olivetti-Faces'}}
#ap_omentum = {'data': ap_omentum_d, 'target': ap_omentum_t, 'details': {'name':'AP-Omentum-Kidney'}}
#pen_digits = {'data': pen_digits_d, 'target': pen_digits_t, 'details': {'name':'Pen-Digits'}}
#####################  Data loading
    
#To perform the experiments according to the article, uncomment the desired sets of datasets
datasets = [
    ####### First set of experiments
          #{"db": skdata.fetch_openml(name='servo', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='servo', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='car-evaluation', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='breast-tissue', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='Engine1', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='xd6', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0}]
          #{"db": skdata.fetch_openml(name='heart-h', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='steel-plates-fault', version=3), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='PhishingWebsites', version=1), "reduce_samples": True, "percentage":.1, "reduce_dim":False, "num_features": 0},              # 10% of the samples
          #{"db": skdata.fetch_openml(name='satimage', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                     # 25% of the samples
          #{"db": skdata.fetch_openml(name='led24', version=1), "reduce_samples": True, "percentage":.20, "reduce_dim":False, "num_features": 0},                       # 20% of the samples
          #{"db": skdata.fetch_openml(name='hayes-roth', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='rabe_131', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='prnn_synth', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='visualizing_environmental', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='diggle_table_a2', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='newton_hema', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='wisconsin', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='fri_c4_250_100', version=2), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          {"db": skdata.fetch_openml(name='conference_attendance', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='tic-tac-toe', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='qsar-biodeg', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='spambase', version=1), "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},                     # 25% of the samples
          #{"db": skdata.fetch_openml(name='cmc', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": skdata.fetch_openml(name='heart-statlog', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": pen_digits, "reduce_samples": True, "percentage":.25, "reduce_dim":False, "num_features": 0},
          #{"db": hopf_link, "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},
          #{"db": repliclust_archetype, "reduce_samples": False, "percentage":0, "reduce_dim":False, "num_features": 0},

    ###########################
         # Second set of experiments
         #{"db": skdata.fetch_openml(name='cnae-9', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 50},                     # 50-D
         #{"db": skdata.fetch_openml(name='AP_Breast_Kidney', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 500},          # 500-D
         #{"db": skdata.fetch_openml(name='AP_Endometrium_Breast', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 400},     # 400-D
         #{"db": skdata.fetch_openml(name='AP_Ovary_Lung', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},             # 100-D
         #{"db": skdata.fetch_openml(name='OVA_Uterus', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                # 100-D
         #{"db": skdata.fetch_openml(name='micro-mass', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                # 100-D
         #{"db": skdata.fetch_openml(name='har', version=1), "reduce_samples": True, "percentage":0.1, "reduce_dim":True, "num_features": 100},                      # 10%  of the samples and 100-D
         #{"db": skdata.fetch_openml(name='eating', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 100},                    # 100-D
         #{"db": skdata.fetch_openml(name='oh5.wc', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 40},                     # 40-D
         #{"db": skdata.fetch_openml(name='leukemia', version=1), "reduce_samples": False, "percentage":0, "reduce_dim":True, "num_features": 40},                   # 40-D
         #{"db": olivetti, "reduce_samples": False, "percentage":1., "reduce_dim":True, "num_features": 100},    
         #{"db": coil_20, "reduce_samples": False, "percentage":1, "reduce_dim":True, "num_features": 80},
         #{"db": mnist, "reduce_samples": True, "percentage":0.05, "reduce_dim":True, "num_features": 100},
         #{"db": fmnist, "reduce_samples": True, "percentage":0.05, "reduce_dim":True, "num_features": 100},
         #{"db": ap_omentum, "reduce_samples": False, "percentage":1., "reduce_dim":True, "num_features":90}                                               
    ]                                                       
    
# If True creates 2D plots
plot_results = False

magnitude = np.linspace(0, 0, 1)

# File result
results = {}

# Run the experiment
for dataset in datasets:
    X = dataset["db"]
    raw_data = X['data']
    dataset_data = X['data']
    dataset_target = X['target']
    dataset_name = X['details']['name']
    reduce_samples = dataset["reduce_samples"]
    reduce_dim = dataset["reduce_dim"]


    # Convert labels to integers
    label_list = []
    for x in dataset_target:
        if x not in label_list:  
            label_list.append(x)     
            
    # Map labels to respective numbers
    labels = []
    for x in dataset_target:  
        for i in range(len(label_list)):
            if x == label_list[i]:  
                labels.append(i)
    dataset_target = np.array(labels)

    # Number of samples, features and classes
    n = dataset_data.shape[0]
    m = dataset_data.shape[1]
    c = len(np.unique(dataset_target))

    # Some adjustments are require in opnML datasets
    # Categorical features must be encoded manually
    if type(dataset_data) == sp.sparse._csr.csr_matrix:
        dataset_data = dataset_data.todense()
        dataset_data = np.asarray(dataset_data)
    if not isinstance(dataset_data, np.ndarray):
        cat_cols = dataset_data.select_dtypes(['category']).columns
        dataset_data[cat_cols] = dataset_data[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dataset_data = dataset_data.to_numpy()

    # To remove NaNs
    dataset_data = np.nan_to_num(dataset_data)
    # Data standardization (to deal with variables having different units/scales)
    dataset_data = preprocessing.scale(dataset_data).astype(np.float64)


    if not reduce_samples and not reduce_dim:
        raw_data = dataset_data
    if reduce_samples:
        percentage = dataset["percentage"]
        dataset_data, garbage, dataset_target, garbage_t = train_test_split(dataset_data, dataset_target, train_size=percentage, random_state=42)
        raw_data = dataset_data
    if reduce_dim:
        num_features = dataset["num_features"]
        raw_data = dataset_data
        dataset_data = PCA(n_components=num_features).fit_transform(dataset_data)

    # Number of samples, features and classes
    n = dataset_data.shape[0]
    m = dataset_data.shape[1]
    
    ######## INTRINSIC DIMENSION ESTIMATION
    # Estimate global intrinsic dimension with MLE - Levina & Bickel
    MLE = skdim.id.MLE().fit(dataset_data)
    # Get estimated intrinsic dimension
    if int(np.floor(MLE.dimension_)) == 0 or int(np.floor(MLE.dimension_)) == 1:
        d_star = 2
    else: 
        d_star = int(np.floor(MLE.dimension_)) 

    # Number of neighbors in KNN graph (patch size)
    nn = int(np.floor(np.sqrt(n)))  

    # Print data info
    print(dataset_name)
    print('n = ', n)
    print('m = ', m)
    print('c = ', c)
    print('d = ', d_star)
    print('k = ', nn)
   
   
    # K-ISOMAP results
    ri_kiso, ch_kiso, fm_kiso, v_kiso, s_kiso, db_kiso, ac_kiso = [], [], [], [], [], [], []
    ri_best_metric, ch_best_metric, fm_best_metric, v_best_metric, s_best_metric, db_best_metric, ac_best_metric = [], [], [], [], [], [], []


    # ISOMAP results
    ri_iso, ch_iso, fm_iso, v_iso, s_iso, db_iso, ac_iso= [], [], [], [], [], [], []


    # UMAP results
    ri_umap, ch_umap, fm_umap, v_umap, s_umap, db_umap, ac_umap = [], [], [], [], [], [], []
    

    # RAW results
    ri_raw, ch_raw, fm_raw, v_raw, s_raw, db_raw, ac_raw = [], [], [], [], [], [], []

    # Kernel PCA results
    ri_kpca, ch_kpca, fm_kpca, v_kpca, s_kpca, db_kpca, ac_kpca = [], [], [], [], [], [], []
    
    # t-SNE results
    ri_tsne, ch_tsne, fm_tsne, v_tsne, s_tsne, db_tsne, ac_tsne = [], [], [], [], [], [], []
    
    # LLE Eigenmaps results
    ri_laplacian, ch_laplacian, fm_laplacian, v_laplacian, s_laplacian, db_laplacian, ac_laplacian = [], [], [], [], [], [], []
    
    # LLE results
    ri_LLE, ch_LLE, fm_LLE, v_LLE, s_LLE, db_LLE, ac_LLE = [], [], [], [], [], [], []


    dataset_data_copy = dataset_data.copy()

    # Inicialização das variáveis de tempo no início do código
    t_kisomap, t_isomap, t_umap, t_kpca, t_tsne, t_laplacian, t_lle = [], 0, 0, 0, 0, 0, 0

    ########## KISOMAP
    
    ri_kmeans, ch_kmeans, fm_kmeans, v_kmeans, s_kmeans, db_kmeans = [], [], [], [], [], []
    ri_gmm, ch_gmm, fm_gmm, v_gmm, s_gmm, db_gmm = [], [], [], [], [], []
    ri_ward, ch_ward, fm_ward, v_ward, s_ward, db_ward = [], [], [], [], [], []

    
    for i in range(11):
        start = time.time()
        DR_method = 'K-ISOMAP' 

        try:
            dados_kiso, _ = KIsomap(dataset_data, nn, d_star, i)  
            print("KISOMAP run: ", i)     
        except Exception as e:
            print(DR_method + " -------- def KIsomap error:", e)
            dados_kiso = []
    
        finish = time.time()
        t_kisomap.append(finish - start)
            
        if dados_kiso.any():
            L_kiso = Clustering(dados_kiso.T, dataset_target, DR_method)
            ri_kmeans.append(L_kiso['KMeans']['ri'])
            ch_kmeans.append(L_kiso['KMeans']['ch'])
            fm_kmeans.append(L_kiso['KMeans']['fm'])
            v_kmeans.append(L_kiso['KMeans']['vs'])
            s_kmeans.append(L_kiso['KMeans']['ss'])
            db_kmeans.append(L_kiso['KMeans']['db'])
            ri_gmm.append(L_kiso['GMM']['ri'])
            ch_gmm.append(L_kiso['GMM']['ch'])
            fm_gmm.append(L_kiso['GMM']['fm'])
            v_gmm.append(L_kiso['GMM']['vs'])
            s_gmm.append(L_kiso['GMM']['ss'])
            db_gmm.append(L_kiso['GMM']['db'])
            ri_ward.append(L_kiso['Ward']['ri'])
            ch_ward.append(L_kiso['Ward']['ch'])
            fm_ward.append(L_kiso['Ward']['fm'])
            v_ward.append(L_kiso['Ward']['vs'])
            s_ward.append(L_kiso['Ward']['ss'])
            db_ward.append(L_kiso['Ward']['db'])

    t_kisomap = np.mean(t_kisomap)
    print(f'K-ISOMAP execution time: {t_kisomap:.2f} seconds')


    # Find best result in terms of Rand index of metric function
    ri_star_kmeans = max(ri_kmeans,default=0)
    ri_star_gmm = max(ri_gmm,default=0)
    if ri_star_gmm > 0:
        print('Best metric for RI: ',ri_gmm.index(ri_star_gmm))
    ri_star_ward = max(ri_ward,default=0)
    ri_kiso.append([ri_star_kmeans,ri_star_gmm,ri_star_ward])
    ri_best_metric.append([ri_kmeans.index(ri_star_kmeans),ri_gmm.index(ri_star_gmm),ri_ward.index(ri_star_ward)])
                
    # Find best result in terms of Calinski-Harabasz Score of metric function
    ch_star_kmeans = max(ch_kmeans, default=0)
    ch_star_gmm = max(ch_gmm, default=0)
    if ch_star_gmm > 0:
        print('Best metric for CH: ',ch_gmm.index(np.nan_to_num(ch_star_gmm)))
    ch_star_ward = max(ch_ward, default=0)
    ch_kiso.append([ch_star_kmeans, ch_star_gmm, ch_star_ward])
    ch_best_metric.append([ch_kmeans.index(ch_star_kmeans), ch_gmm.index(ch_star_gmm), ch_ward.index(ch_star_ward)])
    # Find best result in terms of Fowlkes-Mallows Score of metric function
    fm_star_kmeans = max(fm_kmeans, default=0)
    fm_star_gmm = max(fm_gmm, default=0)
    if fm_star_gmm > 0:
        print('Best metric for FM: ',fm_gmm.index(fm_star_gmm))
    fm_star_ward = max(fm_ward, default=0)
    fm_kiso.append([fm_star_kmeans, fm_star_gmm, fm_star_ward])
    fm_best_metric.append([fm_kmeans.index(fm_star_kmeans), fm_gmm.index(fm_star_gmm), fm_ward.index(fm_star_ward)])
    # Find best result in terms of V-measure of metric function
    v_star_kmeans = max(v_kmeans, default=0)
    v_star_gmm = max(v_gmm, default=0)
    if v_star_gmm > 0:
        print('Best metric for VS: ',v_gmm.index(v_star_gmm))
    v_star_ward = max(v_ward, default=0)
    v_kiso.append([v_star_kmeans, v_star_gmm, v_star_ward])
    v_best_metric.append([v_kmeans.index(v_star_kmeans), v_gmm.index(v_star_gmm), v_ward.index(v_star_ward)])
    # Find best result in terms of Silhouette of metric function
    s_star_kmeans = max(s_kmeans, default=0)
    s_star_gmm = max(s_gmm, default=0)
    if s_star_gmm > 0:
        print('Best metric for SS: ',s_gmm.index(s_star_gmm))
    s_star_ward = max(s_ward, default=0)

    s_kiso.append([s_star_kmeans, s_star_gmm, s_star_ward])
    s_best_metric.append([s_kmeans.index(s_star_kmeans), s_gmm.index(s_star_gmm), s_ward.index(s_star_ward)])
    # Find best result in terms of Davies-Bouldin of metric function
    db_star_kmeans = max(db_kmeans, default=0)
    db_star_gmm = max(db_gmm, default=0)
    if db_star_gmm > 0:
        print('Best metric for DB: ',db_gmm.index(db_star_gmm))
    db_star_ward = max(db_ward, default=0)
    db_kiso.append([db_star_kmeans, db_star_gmm, db_star_ward])
    db_best_metric.append([db_kmeans.index(db_star_kmeans), db_gmm.index(db_star_gmm), db_ward.index(db_star_ward)])
    

    ############## Regular ISOMAP 
    start = time.time()
    model = Isomap(n_neighbors=nn, n_components=d_star)
    isomap_data = model.fit_transform(dataset_data)
    isomap_data = isomap_data.T
    DR_method = 'ISOMAP' 
    L_iso = Clustering(isomap_data, dataset_target, DR_method)
    ri_iso.append([L_iso['KMeans']['ri'],L_iso['GMM']['ri'],L_iso['Ward']['ri']])
    ch_iso.append([L_iso['KMeans']['ch'],L_iso['GMM']['ch'],L_iso['Ward']['ch']])
    fm_iso.append([L_iso['KMeans']['fm'],L_iso['GMM']['fm'],L_iso['Ward']['fm']])
    v_iso.append([L_iso['KMeans']['vs'],L_iso['GMM']['vs'],L_iso['Ward']['vs']])
    s_iso.append([L_iso['KMeans']['ss'],L_iso['GMM']['ss'],L_iso['Ward']['ss']])
    db_iso.append([L_iso['KMeans']['db'],L_iso['GMM']['db'],L_iso['Ward']['db']])
    finish = time.time()
    t_isomap = finish - start
    print(f'ISOMAP execution time: {t_isomap:.2f} seconds')

    ############## UMAP
    start = time.time()
    model = UMAP(n_components=d_star,n_neighbors=nn)
    umap_data = model.fit_transform(dataset_data)
    umap_data = umap_data.T
    DR_method = 'UMAP'
    L_umap = Clustering(umap_data, dataset_target, DR_method)
    ri_umap.append([L_umap['KMeans']['ri'],L_umap['GMM']['ri'],L_umap['Ward']['ri']])
    ch_umap.append([L_umap['KMeans']['ch'],L_umap['GMM']['ch'],L_umap['Ward']['ch']])
    fm_umap.append([L_umap['KMeans']['fm'],L_umap['GMM']['fm'],L_umap['Ward']['fm']])
    v_umap.append([L_umap['KMeans']['vs'],L_umap['GMM']['vs'],L_umap['Ward']['vs']])
    s_umap.append([L_umap['KMeans']['ss'],L_umap['GMM']['ss'],L_umap['Ward']['ss']])
    db_umap.append([L_umap['KMeans']['db'],L_umap['GMM']['db'],L_umap['Ward']['db']])
    finish = time.time()
    t_umap = finish - start
    print(f'UMAP execution time: {t_umap:.2f} seconds')

    # Kernel PCA
    start = time.time()
    model = KernelPCA(n_components=d_star, kernel='rbf')
    kpca_data = model.fit_transform(dataset_data)
    kpca_data = kpca_data.T
    DR_method = 'Kernel PCA'
    L_kpca = Clustering(kpca_data, dataset_target, DR_method)
    ri_kpca.append([L_kpca['KMeans']['ri'],L_kpca['GMM']['ri'],L_kpca['Ward']['ri']])
    ch_kpca.append([L_kpca['KMeans']['ch'],L_kpca['GMM']['ch'],L_kpca['Ward']['ch']])
    fm_kpca.append([L_kpca['KMeans']['fm'],L_kpca['GMM']['fm'],L_kpca['Ward']['fm']])
    v_kpca.append([L_kpca['KMeans']['vs'],L_kpca['GMM']['vs'],L_kpca['Ward']['vs']])
    s_kpca.append([L_kpca['KMeans']['ss'],L_kpca['GMM']['ss'],L_kpca['Ward']['ss']])
    db_kpca.append([L_kpca['KMeans']['db'],L_kpca['GMM']['db'],L_kpca['Ward']['db']])
    finish = time.time()
    t_kpca = finish - start
    print(f'Kernel PCA execution time: {t_kpca:.2f} seconds')

    # t-SNE
    start = time.time()
    model = TSNE(n_components=d_star, random_state=42, method='exact')
    tsne_data = model.fit_transform(dataset_data)
    tsne_data = tsne_data.T
    DR_method = 't-SNE'
    L_tsne = Clustering(tsne_data, dataset_target, DR_method)
    ri_tsne.append([L_tsne['KMeans']['ri'],L_tsne['GMM']['ri'],L_tsne['Ward']['ri']])
    ch_tsne.append([L_tsne['KMeans']['ch'],L_tsne['GMM']['ch'],L_tsne['Ward']['ch']])
    fm_tsne.append([L_tsne['KMeans']['fm'],L_tsne['GMM']['fm'],L_tsne['Ward']['fm']])
    v_tsne.append([L_tsne['KMeans']['vs'],L_tsne['GMM']['vs'],L_tsne['Ward']['vs']])
    s_tsne.append([L_tsne['KMeans']['ss'],L_tsne['GMM']['ss'],L_tsne['Ward']['ss']])
    db_tsne.append([L_tsne['KMeans']['db'],L_tsne['GMM']['db'],L_tsne['Ward']['db']])
    finish = time.time()
    t_tsne = finish - start
    print(f't-SNE execution time: {t_tsne:.2f} seconds')

    # Laplacian Eigenmaps
    start = time.time()
    model = SpectralEmbedding(n_components=d_star, n_neighbors=nn)
    laplacian_data = model.fit_transform(dataset_data)
    laplacian_data = laplacian_data.T
    DR_method = 'Laplacian Eigenmaps'
    L_laplacian = Clustering(laplacian_data, dataset_target, DR_method)
    ri_laplacian.append([L_laplacian['KMeans']['ri'],L_laplacian['GMM']['ri'],L_laplacian['Ward']['ri']])
    ch_laplacian.append([L_laplacian['KMeans']['ch'],L_laplacian['GMM']['ch'],L_laplacian['Ward']['ch']])
    fm_laplacian.append([L_laplacian['KMeans']['fm'],L_laplacian['GMM']['fm'],L_laplacian['Ward']['fm']])
    v_laplacian.append([L_laplacian['KMeans']['vs'],L_laplacian['GMM']['vs'],L_laplacian['Ward']['vs']])
    s_laplacian.append([L_laplacian['KMeans']['ss'],L_laplacian['GMM']['ss'],L_laplacian['Ward']['ss']])
    db_laplacian.append([L_laplacian['KMeans']['db'],L_laplacian['GMM']['db'],L_laplacian['Ward']['db']])
    finish = time.time()
    t_laplacian = finish - start
    print(f'Laplacian Eigenmaps execution time: {t_laplacian:.2f} seconds')

    # LLE
    start = time.time()
    model = LocallyLinearEmbedding(n_neighbors=nn, n_components=d_star)
    LLE_data = model.fit_transform(dataset_data)
    LLE_data = LLE_data.T
    DR_method = 'LLE'
    L_LLE = Clustering(LLE_data, dataset_target, DR_method)
    ri_LLE.append([L_LLE['KMeans']['ri'],L_LLE['GMM']['ri'],L_LLE['Ward']['ri']])
    ch_LLE.append([L_LLE['KMeans']['ch'],L_LLE['GMM']['ch'],L_LLE['Ward']['ch']])
    fm_LLE.append([L_LLE['KMeans']['fm'],L_LLE['GMM']['fm'],L_LLE['Ward']['fm']])
    v_LLE.append([L_LLE['KMeans']['vs'],L_LLE['GMM']['vs'],L_LLE['Ward']['vs']])
    s_LLE.append([L_LLE['KMeans']['ss'],L_LLE['GMM']['ss'],L_LLE['Ward']['ss']])
    db_LLE.append([L_LLE['KMeans']['db'],L_LLE['GMM']['db'],L_LLE['Ward']['db']])
    finish = time.time()
    t_lle = finish - start
    print(f'LLE execution time: {t_lle:.2f} seconds')

    # RAW Clustering
    DR_method = 'RAW'
    L_raw = Clustering(dataset_data.T, dataset_target, DR_method)
    ri_raw.append([L_raw['KMeans']['ri'],L_raw['GMM']['ri'],L_raw['Ward']['ri']])
    ch_raw.append([L_raw['KMeans']['ch'],L_raw['GMM']['ch'],L_raw['Ward']['ch']])
    fm_raw.append([L_raw['KMeans']['fm'],L_raw['GMM']['fm'],L_raw['Ward']['fm']])
    v_raw.append([L_raw['KMeans']['vs'],L_raw['GMM']['vs'],L_raw['Ward']['vs']])
    s_raw.append([L_raw['KMeans']['ss'],L_raw['GMM']['ss'],L_raw['Ward']['ss']])
    db_raw.append([L_raw['KMeans']['db'],L_raw['GMM']['db'],L_raw['Ward']['db']])

    # Adicionar os tempos de execução ao dicionário de resultados
    results[dataset_name + '_time']= {
            "KISOMAP": t_kisomap,
            "ISOMAP": t_isomap,
            "UMAP": t_umap,
            "Kernel PCA": t_kpca,
            "t-SNE": t_tsne,
            "Laplacian Eigenmaps": t_laplacian,
            "LLE": t_lle,
            "Samples": n,
            "Features": m,
            "Dimension": d_star,
            "Classes": c,
            "Neighbors": nn
    }
    
    # Adicionar os resultados do GMM
    results[dataset_name] = {
        "KISOMAP": np.array([ri_kiso, ch_kiso, fm_kiso, v_kiso, s_kiso, db_kiso]).tolist(),
        "ISOMAP": np.array([ri_iso, ch_iso, fm_iso, v_iso, s_iso, db_iso]).tolist(),
        "Kernel PCA": np.array([ri_kpca, ch_kpca, fm_kpca, v_kpca, s_kpca, db_kpca]).tolist(),
        "t-SNE": np.array([ri_tsne, ch_tsne, fm_tsne, v_tsne, s_tsne, db_tsne]).tolist(),
        "Laplacian Eigenmaps": np.array([ri_laplacian, ch_laplacian, fm_laplacian, v_laplacian, s_laplacian, db_laplacian]).tolist(),
        "UMAP": np.array([ri_umap, ch_umap, fm_umap, v_umap, s_umap, db_umap]).tolist(),
        "LLE": np.array([ri_LLE, ch_LLE, fm_LLE, v_LLE, s_LLE, db_LLE]).tolist(),
        "RAW": np.array([ri_raw, ch_raw, fm_raw, v_raw, s_raw, db_raw]).tolist()
    }

    # Salvar os resultados em um arquivo JSON (exemplo)
    file_results = '1st_battery_revision_final.json'

    # Verifica se já existe o arquivo para não sobreescrever
    try:
        with open(file_results, 'r') as f:
            previous_results = json.load(f)
    except FileNotFoundError:
        previous_results = {}
        
    results = {key: {**results.get(key, {}), **previous_results.get(key, {})} for key in results.keys() | previous_results.keys()}
    # Save results
    try:
        with open(file_results, 'w') as f:
            json.dump(results, f)
    except IOError as e:
        print(f"An error occurred while writing to the file: {file_results} - {e}")

    if plot_results:
    #*********************************************
    #******* SUMMARY OF THE RESULTS **************
    #*********************************************

        # Find best result in terms of Rand index
        ri_star = ri_gmm.index(max(ri_gmm,default=0))
        DR_method = 'K-ISOMAP' 
        try:
            dados_kiso, _ = KIsomap(dataset_data, nn, 2, ri_star)     
        except Exception as e:
            print(DR_method + " -------- Plot KIsomap error:", e)
            dados_kiso = []
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'KISOMAP')
        labels_kiso = L_kiso['GMM']['labels']
        PlotaDados(dados_kiso, labels_kiso, dataset_name + '_KISOMAP_RI')
        print('Best K-ISOMAP metric in terms of Rand Index: ', ri_star)

        # Find best result in terms of Calisnki-Harabasz
        ch_star = ch_gmm.index(max(ch_gmm))
        try:
            dados_kiso, _ = KIsomap(dataset_data, nn, 2, ch_star)     
        except Exception as e:
            print(DR_method + " -------- Plot KIsomap error:", e)
            dados_kiso = []
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'KISOMAP')
        labels_kiso = L_kiso['GMM']['labels']
        PlotaDados(dados_kiso, labels_kiso, dataset_name + '_KISOMAP_CH')
        print('Best K-ISOMAP metric in terms of Calisnki-Harabasz: ', ch_star)

        # Find best result in terms of Fowlkes-Mallows
        fm_star = fm_gmm.index(max(fm_gmm,default=0))
        try:
            dados_kiso, _ = KIsomap(dataset_data, nn, 2, fm_star)     
        except Exception as e:
            print(DR_method + " -------- Plot KIsomap error:", e)
            dados_kiso = []
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'KISOMAP')
        labels_kiso = L_kiso['GMM']['labels']
        PlotaDados(dados_kiso, labels_kiso, dataset_name + '_KISOMAP_FM')
        print('Best K-ISOMAP metric in terms of Fowlkes-Mallows: ', fm_star)

        # Find best result in terms of V-Measure
        v_star = v_gmm.index(max(v_gmm,default=0))
        try:
            dados_kiso, _ = KIsomap(dataset_data, nn, 2, v_star)     
        except Exception as e:
            print(DR_method + " -------- Plot KIsomap error:", e)
            dados_kiso = []
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'KISOMAP')
        labels_kiso = L_kiso['GMM']['labels']
        PlotaDados(dados_kiso, labels_kiso, dataset_name + '_KISOMAP_V')
        print('Best K-ISOMAP metric in terms of V measure: ', v_star)

        # Find best result in terms of Silhuette Score
        s_star = s_gmm.index(max(s_gmm,default=0))
        try:
            dados_kiso, _ = KIsomap(dataset_data, nn, 2, s_star)     
        except Exception as e:
            print(DR_method + " -------- Plot KIsomap error:", e)
            dados_kiso = []
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'KISOMAP')
        labels_kiso = L_kiso['GMM']['labels']
        PlotaDados(dados_kiso, labels_kiso, dataset_name + '_KISOMAP_S')
        print('Best K-ISOMAP metric in terms of Silhuette Score: ', s_star)

        # Find best result in terms of Davies-Bouldin
        db_star = db_gmm.index(max(db_gmm,default=0))
        try:
            dados_kiso, _ = KIsomap(dataset_data, nn, 2, db_star)     
        except Exception as e:
            print(DR_method + " -------- Plot KIsomap error:", e)
            dados_kiso = []
        L_kiso = Clustering(dados_kiso.T, dataset_target, 'KISOMAP')
        labels_kiso = L_kiso['GMM']['labels']
        PlotaDados(dados_kiso, labels_kiso, dataset_name + '_KISOMAP_DB')
        print('Best K-ISOMAP metric in terms of Davies-Bouldin: ', db_star)

        ############## Regular ISOMAP 
        print('ISOMAP results')
        model = Isomap(n_neighbors=nn, n_components=2)
        isomap_data = model.fit_transform(dataset_data)
        isomap_data = isomap_data.T
        L_iso = Clustering(isomap_data, dataset_target, 'ISOMAP')
        labels_iso = L_iso['GMM']['labels']
        PlotaDados(isomap_data.T, labels_iso, dataset_name + '_ISOMAP')
        ############## UMAP
        print('UMAP results')
        model = UMAP(n_neighbors=nn, n_components=2)
        umap_data = model.fit_transform(dataset_data)
        umap_data = umap_data.T
        L_umap = Clustering(umap_data, dataset_target, 'UMAP')
        labels_umap = L_umap['GMM']['labels']
        PlotaDados(umap_data.T, labels_umap, dataset_name + '_UMAP')


        # Kernel PCA
        print('Kernel PCA results')
        model = KernelPCA(n_components=2, kernel='rbf')
        kpca_data = model.fit_transform(dataset_data)
        kpca_data = kpca_data.T
        L_kpca = Clustering(kpca_data, dataset_target, 'KPCA')
        labels_kpca = L_kpca['GMM']['labels']
        PlotaDados(kpca_data.T, labels_kpca, dataset_name + '_KPCA')

        # t-SNE
        print('t-SNE results')
        model = TSNE(n_components=2, random_state=42, method='exact')
        tsne_data = model.fit_transform(dataset_data)
        tsne_data = tsne_data.T
        L_tsne = Clustering(tsne_data, dataset_target, 't-SNE')
        labels_tsne = L_tsne['GMM']['labels']
        PlotaDados(tsne_data.T, labels_tsne, dataset_name + '_t-SNE')

        # Laplacian Eigenmaps (Spectral Embedding)
        print('Laplacian Eigenmaps results')
        model = SpectralEmbedding(n_components=2,n_neighbors=nn)
        laplacian_data = model.fit_transform(dataset_data)
        laplacian_data = laplacian_data.T
        L_laplacian = Clustering(laplacian_data, dataset_target, 'Laplacian Eigenmaps')
        labels_laplacian = L_laplacian['GMM']['labels']
        PlotaDados(laplacian_data.T, labels_laplacian, dataset_name + '_LaplacianEigenmaps')

        # LLE
        print('LLE results')
        model = LocallyLinearEmbedding(n_neighbors=nn, n_components=2)
        LLE_data = model.fit_transform(dataset_data)
        LLE_data = LLE_data.T
        L_LLE = Clustering(LLE_data, dataset_target,'LLE')
        labels_LLE = L_LLE['GMM']['labels']
        PlotaDados(LLE_data.T, labels_LLE, dataset_name + '_LLE')
        print('Plots for ' + dataset_name + ' created.')

    print('Dataset '+ dataset_name + ' completed.')

