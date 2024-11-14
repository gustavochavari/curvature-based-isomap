#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unsupervised metric learning via K-ISOMAP for high-dimensional data clustering - Noise experiments

Created on Sat May 25 13:41:56 2024

"""


# Imports
import sys
#sys.path.append('C:/Users/Gustavo/Documents/Mestrado/curvature-based-isomap/functions')

import repliclust
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.manifold import Isomap
from sklearn.datasets import make_swiss_roll
from kisomap import KIsomap
import umap
import pandas as pd
from matplotlib import cm



# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')



####################################
######## TORUS LINK ################

# Parameters
u = np.linspace(0, 2*np.pi, 26)
v = np.linspace(0, 2*np.pi, 26)
U, V = np.meshgrid(u, v)

R = 5
r = 1

# Torus Surface equation
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

result_matrix = np.vstack([np.column_stack((data_matrix_1, np.full(len(z.flatten()), 0))), 
                          np.column_stack((data_matrix_2, np.full(len(z_1.flatten()), 1)))]).astype('float64')


df = pd.DataFrame(result_matrix, columns=['x', 'y', 'z', 'label'])
df.to_csv('torus_link_data.csv', index=False)



#### Embedding 

nn = int(np.floor(round(sqrt(result_matrix[:,:3].shape[0]))))

dados_kisomap =  KIsomap(result_matrix[:,:3],k=nn, d=2,option=0)

model = Isomap(n_neighbors=nn, n_components=2)
dados_isomap = model.fit_transform(result_matrix[:,:3])
dados_isomap = dados_isomap

reducer = umap.UMAP(n_neighbors=nn)
dados_umap = reducer.fit_transform(result_matrix[:,:3])


# Plot points
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
# Shadow = Projecting the points onto the xy-plane by plotting them with a fixed z-coordinate that matches the lower z-limit.
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

#plt.savefig('torus_link.tiff',format='tiff',dpi=300)
plt.savefig('torus_link.jpeg',format='jpeg',dpi=300)
plt.close()

#################################
######## REPLICLUST #############

# Gaussian Dataset
archetype = repliclust.Archetype(
                    dim=3,
                    n_samples=500,
                    max_overlap=0.00001, min_overlap=0.000004 ,name="oblong"
                    )
X1, y1, _ = (repliclust.DataGenerator(archetype).synthesize(quiet=True))

X1 = X1.astype(np.float64)

# Parameters
noise_scale = 1

# Add noise
magnitude = np.linspace(0,1,26)

ruido = np.random.normal(0, scale=magnitude[0], size=X1.shape)
X1.T[0] = X1.T[0] + ruido.T[0]
X1.T[1] = X1.T[1] + ruido.T[1]
X1.T[2] = X1.T[2] + ruido.T[2]


df = pd.DataFrame({
    'x': X1[:, 0],  # First column of X1
    'y': X1[:, 1],  # Second column of X1
    'z': X1[:, 2],  # Third column of X1
    'label': y1     # Label data
})

df.to_csv('repliclust_data.csv', index=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(X1.T[0], X1.T[1], X1.T[2], c=y1, marker='.')
ax.set_title('Repliclust Data')  

#plt.savefig('repliclust.tiff',format='tiff',dpi=300)
plt.savefig('repliclust.jpeg',format='jpeg',dpi=300)
plt.close()

####################################
######## SWISS ROLL ################

# Swiss Roll Dataset
n_samples = 3000
X, color = make_swiss_roll(n_samples, noise=0.0)

# Add noise
noise_level = 0.05 * (np.max(X[:, 0]) - np.min(X[:, 0]))
X_noisy = X + noise_level * np.random.randn(*X.shape)

### Plot points
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title('Swiss Roll Dataset')

#plt.savefig('swiss_roll.tiff',format='tiff',dpi=300)
plt.savefig('swiss_roll.jpeg',format='jpeg',dpi=300)
plt.close()

df.to_csv('swiss_roll_data.csv', index=False)


####################################
#################### 3D CUBE #######


# Criando uma grade tridimensional
x = np.linspace(-5, 5, 11)  # 11 pontos de -5 a 5
y = np.linspace(-5, 5, 11)
z = np.linspace(-5, 5, 11)
X, Y, Z = np.meshgrid(x, y, z)

# Remodelando para obter um array 3D
X3 = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

X3 = X3.astype(np.float64)

# Parameters
noise_scale = 1  # adjust the noise gaussian parameter here

# Add noise
# Definir a magnitude do ruído gaussiano
magnitude = np.linspace(0,1,11)
# Gerar ruído gaussiano com média zero e desvio padrão baseado na magnitude
ruido = np.random.normal(0, scale=magnitude[0], size=X3.shape)
# Adicionar o ruído aos dados
# Surface equation
X3.T[0] = X3.T[0] + ruido.T[0]
X3.T[1] = X3.T[1] + ruido.T[1]
X3.T[2] = X3.T[2] + ruido.T[2]

fig = plt.figure(figsize=(12,5))

# 3D
ax1 = fig.add_subplot(131,projection='3d')  
ax1.scatter(X3.T[0], X3.T[1], X3.T[2], c=[cm.rainbow(valor) for valor in X3.T[2]/10+0.5], marker='.')
#ax1.view_init(elev=30, azim=30)
ax1.set_title('3D Cube')  
ax1.axis('off')
#ax1.set_xlim3d(-10, 5)
#ax1.set_ylim3d(-10, 5)
#ax1.set_zlim3d(-7, 5)
#ax1.set_xlim(-10, 5)
#ax1.set_ylim(-10, 5)
# Projecting the points onto the xy-plane by plotting them with a fixed z-coordinate that matches the lower z-limit.
ax1.scatter(X3.T[0]+3, X3.T[1]-10, -7*np.ones_like(X3.T[2]),
            c='gray',alpha=0.005)


model = Isomap(n_neighbors=7, n_components=2)
dados_isomap = model.fit_transform(X3)
dados_isomap = dados_isomap.T

kiso = KIsomap(X3, 7, 2, 0)

# 2D
ax2 = fig.add_subplot(132)  
ax2.scatter(dados_isomap[0], dados_isomap[1],c=[cm.rainbow(valor) for valor in X3.T[2]/10+0.5], alpha=0.5)
ax2.set_title('ISOMAP Embedding')  
ax2.axis('off')
#ax2.set_xlim(-60, 60)
#ax2.set_ylim(-30, 30)

# 2D
x_2d = kiso.T[0]
y_2d = kiso.T[1]
ax3 = fig.add_subplot(133)  
ax3.scatter(x_2d/2, y_2d/2, c=[cm.rainbow(valor) for valor in X3.T[2]/10+0.5], alpha=0.3)
ax3.set_title('K-ISOMAP Embedding')  
ax3.axis('off')
#ax3.set_xlim(-30, 30)
#ax3.set_ylim(-13, 13)

plt.savefig('3d_cube.png',format='png',dpi=300)
plt.show()