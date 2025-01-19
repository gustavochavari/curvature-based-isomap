# Função para comparar os métodos e plotar numa única figura
def compare_and_plot_surface_reduction(X,color,dataset_name):

    k = int(np.floor(np.sqrt(X.shape[0]))) 

    methods = {
        'ISOMAP': Isomap(n_components=2, n_neighbors=k),
        'KPCA': KernelPCA(n_components=2, kernel='rbf'),
        'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=k),
        'Laplacian Eigenmaps': SpectralEmbedding(n_components=2, n_neighbors=k,affinity='rbf'),
        't-SNE': TSNE(n_components=2),
        'UMAP': umap.UMAP(n_components=2, n_neighbors=k),
        'K-ISOMAP': KIsomap
    }

    fig = plt.figure(figsize=(18, 3))  # Figura maior para múltiplas subplots
    #color = [cm.rainbow(valor) for valor in np.linspace(0,1,X.shape[0])]
    # Plotando a superfície original em R^3
    ax = fig.add_subplot(1, 8, 1, projection='3d')
    #ax.view_init(azim=90,elev=-70)
    ax.view_init(azim=30,elev=30)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, marker='.')
    ax.set_title(dataset_name)
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.grid(False)
    ax.axis(False)
    
    print('n: ', X.shape[0])

    # Aplicar cada método e plotar a projeção em 2D
    for idx, (name, model) in enumerate(methods.items(), start=2):
        print(name)
        if name == 'K-ISOMAP':
            X_transformed, _ = model(X,k,2,'norm')
            ax = fig.add_subplot(1, 8, idx)
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color, marker='.')
            ax.set_title(f'{name}')
            ax.grid(False)
            ax.axis(False)
        else:
            X_transformed = model.fit_transform(X)
            ax = fig.add_subplot(1, 8, idx)
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color, marker='.')
            ax.set_title(f'{name}')
            ax.grid(False)
            ax.axis(False)
        
    plt.show()
    plt.savefig('S_Surface.jpg', dpi=300)




# Função para comparar os métodos e plotar numa única figura
def compare_and_plot_option_kisomap(X,color,dataset_name):

    k = int(np.floor(np.sqrt(X.shape[0]))) - 20

    methods = {
        'norm': ConstrainedKIsomap,
        'first': ConstrainedKIsomap,
        'last': ConstrainedKIsomap,
        'avg(first,last)': ConstrainedKIsomap,
        'mean': ConstrainedKIsomap,
        'max': ConstrainedKIsomap,
        'min': ConstrainedKIsomap,
        'min*max': ConstrainedKIsomap,
        'max-min': ConstrainedKIsomap,
        'exp': ConstrainedKIsomap,
        'mixed': ConstrainedKIsomap,
    }

    options = ['norm','first','last','avg_first_last','mean','max','min','product_min_max','difference_max_min','exponential','mixed']


    fig = plt.figure(figsize=(18, 3))  # Figura maior para múltiplas subplots
    #color = [cm.rainbow(valor) for valor in np.linspace(0,1,X.shape[0])]
    # Plotando a superfície original em R^3
    ax = fig.add_subplot(1, 12, 1, projection='3d')
    #ax.view_init(azim=90,elev=-70)
    ax.view_init(azim=30,elev=30)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, marker='.')
    ax.set_title(dataset_name + '\n', pad=21)
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.grid(False)
    ax.axis(False)
    
    print('n: ', X.shape[0])

    # Aplicar cada método e plotar a projeção em 2D
    for idx, (name, model) in enumerate(methods.items()):
        print(idx)
        X_transformed, _ = model(X,k,2,options[idx])
        ax = fig.add_subplot(1, 12, idx+2)
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color, marker='.')
        ax.set_title(f'{name}')
        ax.grid(False)
        ax.axis(False)
        
    plt.show()
    plt.savefig('S_Surface.jpg', dpi=300)


def plot_kiso_iterations(X, color, dataset_name):
    # Calculate k as sqrt of number of points
    k = int(np.floor(np.sqrt(X.shape[0]))) - 20
    
    # Create figure with 2x5 subplot grid
    fig = plt.figure(figsize=(20, 8))
    
    print('Number of samples:', X.shape[0])
    
    # Create 10 projections with different i values
    for idx in range(10):
        print('value: ', idx/10)
        # Calculate subplot position (2 rows, 5 columns)
        ax = fig.add_subplot(2, 5, idx + 1)
        
        # Apply KISOMAP with current iteration value
        X_transformed, _ = ConstrainedKIsomap(X, k, 2, 'mixed',idx/10)
        
        # Create scatter plot
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                  c=color, marker='.', alpha=0.6)
        
        # Set subplot title and styling
        ax.set_title(f'KISOMAP (alpha={idx/10})')
        ax.grid(False)
        ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_kiso_iterations.jpg', dpi=300, bbox_inches='tight')
    plt.show()


# Definição da superfície S com fórmulas baseadas em seno e cosseno
def surface_S(u, v):
    x = np.sin(2*u)
    z = v
    y = np.cos(u)
    return x, y, z

# Geração de pontos para a superfície
u = np.linspace(-0.3*np.pi, 1.3*np.pi, 30) 
u += np.random.normal(loc=0, scale=0.01, size=u.shape) # Parâmetro u
v = np.linspace(0, np.pi, 30) 
v += np.random.normal(loc=0, scale=0.01, size=v.shape)     # Parâmetro v
u, v = np.meshgrid(u, v)

x, y, z = surface_S(u, v)

# Adicionar o ruído às coordenadas
x_noisy = x + np.random.normal(loc=0, scale=0.1, size=x.shape)
y_noisy = y + np.random.normal(loc=0, scale=0.1, size=y.shape)
z_noisy = z + np.random.normal(loc=0, scale=0.1, size=z.shape)

df = pd.DataFrame(np.array([x_noisy, y_noisy, z_noisy]).reshape(3, -1).T, columns=['x','y','z'])
color = [cm.rainbow(1-valor) for valor in np.linspace(0,1,df.values.shape[0])]