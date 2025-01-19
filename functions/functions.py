import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.decomposition import KernelPCA
import umap
from kisomap import ConstrainedKIsomap
from mpl_toolkits.mplot3d import Axes3D

# Definição da superfície S com fórmulas baseadas em seno e cosseno
def surface_S(u, v):
    x = np.sin(2 * u)
    z = v
    y = np.cos(u)
    return x, y, z

# Função para comparar os métodos de redução e plotar os resultados
def compare_and_plot_reduction(X, color, dataset_name):
    k = max(2, int(np.floor(np.sqrt(X.shape[0]))))  # Garantir pelo menos 2 vizinhos

    methods = {
        'ISOMAP': Isomap(n_components=2, n_neighbors=k),
        'KPCA': KernelPCA(n_components=2, kernel='rbf'),
        'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=k),
        'Laplacian Eigenmaps': SpectralEmbedding(n_components=2, n_neighbors=k, affinity='rbf'),
        't-SNE': TSNE(n_components=2),
        'UMAP': umap.UMAP(n_components=2, n_neighbors=k)
    }

    fig = plt.figure(figsize=(20, 3))
    ax = fig.add_subplot(1, len(methods) + 1, 1, projection='3d')
    ax.view_init(azim=30, elev=30)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, marker='.')
    ax.set_title(dataset_name)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.axis(False)

    for idx, (name, model) in enumerate(methods.items(), start=2):
        try:
            X_transformed = model.fit_transform(X)
            ax = fig.add_subplot(1, len(methods) + 1, idx)
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color, marker='.')
            ax.set_title(name)
            ax.grid(False)
            ax.axis(False)
        except Exception as e:
            print(f"Erro ao aplicar {name}: {e}")

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_comparison.jpg', dpi=300, bbox_inches='tight')
    plt.show()

# Função para criar múltiplas projeções do KISOMAP
def plot_kiso_iterations(X, color, dataset_name):
    k = max(2, int(np.floor(np.sqrt(X.shape[0]))) - 20)

    fig = plt.figure(figsize=(20, 8))
    for idx in range(10):
        try:
            X_transformed, _ = ConstrainedKIsomap(X, k, 2, 'mixed', idx / 10)
            ax = fig.add_subplot(2, 5, idx + 1)
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color, marker='.', alpha=0.6)
            ax.set_title(f'KISOMAP (alpha={idx/10:.1f})')
            ax.grid(False)
            ax.axis(False)
        except Exception as e:
            print(f"Erro na iteração {idx}: {e}")

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_kiso_iterations.jpg', dpi=300, bbox_inches='tight')
    plt.show()

# Geração de pontos para a superfície S
u = np.linspace(-0.3 * np.pi, 1.3 * np.pi, 30)
u += np.random.normal(loc=0, scale=0.01, size=u.shape)  # Parâmetro u
v = np.linspace(0, np.pi, 30)
v += np.random.normal(loc=0, scale=0.01, size=v.shape)  # Parâmetro v
u, v = np.meshgrid(u, v)

x, y, z = surface_S(u, v)

# Adicionar o ruído às coordenadas
x_noisy = x + np.random.normal(loc=0, scale=0.1, size=x.shape)
y_noisy = y + np.random.normal(loc=0, scale=0.1, size=y.shape)
z_noisy = z + np.random.normal(loc=0, scale=0.1, size=z.shape)

# Transformar os dados em DataFrame
df = pd.DataFrame(np.array([x_noisy, y_noisy, z_noisy]).reshape(3, -1).T, columns=['x', 'y', 'z'])
X = df.values
color = [cm.rainbow(1 - valor) for valor in np.linspace(0, 1, X.shape[0])]

# Chamar as funções de plotagem
compare_and_plot_reduction(X, color, "Superfície S")
plot_kiso_iterations(X, color, "Superfície S")
