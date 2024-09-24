#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering - Noise experiments

Created on Sat May 25 13:41:56 2024

"""
# Imports
import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.manifold import Isomap
from kisomap_latest import KIsomap
import umap
import pandas as pd
import json
from matplotlib import cm

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')


#######################################
######## TORUS LINK ################

# Parameters
u = np.linspace(0, 2*np.pi, 26)
v = np.linspace(0, 2*np.pi, 26)
U, V = np.meshgrid(u, v)

R = 5
r = 1

# S-Surface equation
x = (R+r*np.cos(V))*np.cos(U)
y = (R+r*np.cos(V))*np.sin(U)-5
z = r*np.sin(V)


x_1 = r*np.sin(V) 
y_1 = (R+r*np.cos(V))*np.sin(U) 
z_1 = (R+r*np.cos(V))*np.cos(U)

# Add Gaussian noise with 0.3 standard deviation
np.random.seed(127)
noise_matrix_1 = np.random.normal(0, 0.3, (len(z.flatten()), 3))
data_matrix_1 = np.column_stack((x.flatten(), y.flatten(), z.flatten())) + noise_matrix_1

noise_matrix_2 = np.random.normal(0, 0.3, (len(z_1.flatten()), 3))
data_matrix_2 = np.column_stack((x_1.flatten(), y_1.flatten(), z_1.flatten())) + noise_matrix_2

result_matrix = np.vstack([np.column_stack((data_matrix_1, np.full(len(z.flatten()), 0.2))), 
                          np.column_stack((data_matrix_2, np.full(len(z_1.flatten()), 0.8)))]).astype('float64')



#### Embedding 

nn = int(np.floor(round(sqrt(result_matrix[:,:3].shape[0]))))

dados_kisomap =  KIsomap(result_matrix[:,:3],k=nn, d=2,option=0)

model = Isomap(n_neighbors=nn, n_components=2)
dados_isomap = model.fit_transform(result_matrix[:,:3])
dados_isomap = dados_isomap

reducer = umap.UMAP(n_neighbors=nn)
dados_umap = reducer.fit_transform(result_matrix[:,:3])


#### Plot
fig = plt.figure(figsize=(12,3))


# Torus link
ax1 = fig.add_subplot(141, projection='3d')  
ax1.scatter(result_matrix[:,:3].T[0], result_matrix[:,:3].T[1], result_matrix[:,:3].T[2], c=[cm.rainbow(valor) for valor in result_matrix.T[3]], alpha=0.5)
ax1.view_init(elev=30, azim=30)
ax1.set_title('Torus Link',pad=16)  
ax1.axis('off')
ax1.set_xlim3d(-10, 5)
ax1.set_ylim3d(-10, 5)
ax1.set_zlim3d(-7, 5)
ax1.set_xlim(-10, 5)
ax1.set_ylim(-10, 5)
# Projecting the points onto the xy-plane by plotting them with a fixed z-coordinate that matches the lower z-limit.
ax1.scatter(result_matrix.T[0], result_matrix.T[1], -7*np.ones_like(result_matrix.T[2]), c='gray',alpha=0.02)


# ISOMAP
ax2 = fig.add_subplot(142)  
ax2.scatter(dados_isomap.T[0], dados_isomap.T[1],c=[cm.rainbow(valor) for valor in result_matrix.T[3]], alpha=0.5)
ax2.set_title('ISOMAP')  
ax2.axis('off')
ax2.set_xlim(-11, 14)
ax2.set_ylim(-9, 9)

# UMAP
ax3 = fig.add_subplot(143)  
ax3.scatter(dados_umap.T[0], dados_umap.T[1],  c=[cm.rainbow(valor) for valor in result_matrix.T[3]], alpha=0.5)
ax3.set_title('UMAP')  
ax3.axis('off')
ax3.set_xlim(-10, 20)
ax3.set_ylim(-1, 10)


# K-ISOMAP
ax4 = fig.add_subplot(144)  
ax4.scatter(dados_kisomap.T[0],dados_kisomap.T[1], c=[cm.rainbow(valor) for valor in result_matrix.T[3]], alpha=0.5)
ax4.set_title('K-ISOMAP')  
ax4.axis('off')
ax4.set_xlim(-10, 10)
ax4.set_ylim(-8, 7)

plt.savefig('linked_torus.tiff',format='tiff',dpi=300)
plt.savefig('linked_torus.jpeg',format='jpeg',dpi=300)
plt.close()
