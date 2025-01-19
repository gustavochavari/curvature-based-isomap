import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.simplefilter(action='ignore')

def main():

    # Definindo o diretório principal
    research_dir = os.environ.get('MS_DIR')
    file_path = os.path.join(research_dir, 'curvature-based-isomap', 'experiments', '1st_battery_revision_final.json')

    with open(file_path, 'r') as file:
        results = json.load(file)

    rows = []
    rows_t = []
    battery_name = '1st_battery_results'

    methods_name = ['RAW','ISOMAP','Kernel PCA','LLE','Laplacian Eigenmaps','t-SNE','UMAP','KISOMAP']

    metrics = ["RI", "CH", "FM", "VS", "SS", "DB"]

    for dataset_name, methods in results.items():
        for method, values in methods.items():
            if "_time" in dataset_name:
                rows_t.append([dataset_name, method, values])
            else:
                if values == [[], [], [], [], [], []]:
                    rows.append([dataset_name, method, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
                else:
                    rows.append([dataset_name, method, values[0][0][1],values[1][0][1],values[2][0][1],values[3][0][1],values[4][0][1],values[5][0][1]])


    df = pd.DataFrame(rows, columns=['Dataset', 'Method', 'RI', 'CH','FM', 'VS', 'SS','DB'])

    ## Tempo de execução (em segundos)
    df_t = pd.DataFrame(rows_t, columns=['Dataset', 'Method', 'Time'])

    ## Manipulação do dicionário
    # Remover '_time' da coluna 'Dataset'
    df_t['Dataset'] = df_t['Dataset'].str.replace('_time', '', regex=False)

    # Pegando apenas o tempo de execução do dataset em cada método
    df_info = df_t[df_t['Method'].isin(['Samples', 'Features', 'Dimension', 'Classes', 'Neighbors'])]

    # Agrupando por dataset e pivotando para as colunas as dimensões 'Samples', 'Features', 'Dimension', 'Classes', 'Neighbors'
    df_info_pivot = df_info.pivot_table(index='Dataset', values='Time', columns='Method').reset_index()

    # Criar o df com o tempo de execução
    df_time = df_t[~df_t['Method'].isin(['Samples', 'Features', 'Dimension', 'Classes', 'Neighbors'])]

    # Realizando o join das duas tabelas 
    df_final = pd.merge(df_time, df_info_pivot, on="Dataset", how="left")

    with open(battery_name+'_info.txt', "w", encoding="utf-8") as file:
        file.write(df_final.to_latex(index=False))

    # Definindo paleta de cores
    custom_palette = {
    'ISOMAP':   '#F67A0C',  # orange
    'Kernel PCA':     '#2ca02c',  # green
    'LLE':      '#6B421D',  # red
    'Laplacian Eigenmaps':       '#9331BD',  # purple
    't-SNE':    '#d62728',  # brown
    'UMAP':     '#e377c2',  # pink
    'KISOMAP': '#1F77FF',   # cyan
    'RAW': 'gray'
    }

    # Plot de tempo de execução

    sns.set_theme()
    sns.set_palette("rainbow")

    for value in methods_name:
        if value != 'RAW':
            ax = sns.regplot(data= df_final[df_final['Method'] == value], x='Samples', y='Time', order=3, label=value, color=custom_palette[value],ci=None)
            ax.set_yscale('log')  # set_yscale is a function, not a string
            ax.set_xlabel('Amostras (n)')
            ax.set_ylabel('Tempo (s)')
            plt.legend(loc='upper right', bbox_to_anchor=(0,1,1.45,0))
    
    plt.savefig(battery_name+'_time.jpeg',format='jpeg',dpi=300, bbox_inches='tight')
    plt.close()


    #####################################
    ######### Testes de hipótese
    #####################################

    statistics = []

    # Iterando sobre todas as combinações de métricas e métodos
    for metric in metrics:
        if '_norm' not in df['Dataset'].unique():
            kiso_data = df[df['Method'] == 'KISOMAP'][metric]
            raw_data = df[df['Method'] == 'RAW'][metric]
        for method in methods_name:
            if method != 'KISOMAP' and method != "RAW":  # Comparar com KISOMAP
                other_data = df[df['Method'] == method][metric]
                if len(kiso_data) == len(other_data):  # Certificar-se de que temos o mesmo número de amostras
                    # Realizando os testes
                    stat, p_value = friedmanchisquare(raw_data, kiso_data, other_data)
                    p_values_nemenyi = sp.posthoc_nemenyi_friedman(np.array([raw_data, kiso_data, other_data]).T)

                    # Armazenando os resultados
                    statistics.append({
                        'Metric': metric,
                        'Method': method,
                        'Stat': stat,
                        'P-Value Friedman': p_value,
                        # Esse aqui é o valor do teste do KISOMAP vs. (method)
                        'P-Value Nemenyi': p_values_nemenyi[0][2]
                    })

                else:
                    print(f'Número de amostras não corresponde entre KISOMAP e {method} para a métrica {metric}')

    results_df = pd.DataFrame(statistics)

    results_df['Test Friedman'] = results_df['P-Value Friedman'].apply(lambda x: 1 if x < 0.05 else 0)
    results_df['Stat'] = results_df['Stat'].apply(lambda x: f"{x:.3f}")
    results_df['Test Nemenyi'] = results_df['P-Value Nemenyi'].apply(lambda x: 1 if x < 0.05 else 0)
    results_df['P-Value Friedman'] = results_df['P-Value Friedman'].apply(lambda x: f"{x:.3f}")
    results_df['P-Value Nemenyi'] = results_df['P-Value Nemenyi'].apply(lambda x: f"{x:.3f}")

    results_df = results_df.drop('Stat',axis=1)

    ###### Exportar tabela em latex
    latex_code = results_df[['Method','Metric','P-Value Friedman','P-Value Nemenyi','Test Friedman','Test Nemenyi']].sort_values(['Method','Metric']).to_latex(index=False)

    with open(battery_name, "w", encoding="utf-8") as file:
        file.write(latex_code)

    df_melted = pd.melt(df, id_vars=['Dataset', 'Method'], 
                    value_vars=['RI',
                                'CH',
                                'FM',
                                'VS', 
                                'SS',
                                'DB'],
                    var_name='metric', value_name='value')

    # Pivotando para criar uma tabela com subcolunas para cada métrica e método
    df_pivot = df_melted.pivot_table(index='Dataset', columns=['metric', 'Method'], values='value')

    df_pivot = df_pivot.swaplevel(i=0, j=1, axis=1)

    df_pivot = df_pivot.sort_index(axis=1)

    results_data = {}

    for metric in metrics:
        # Filtrar os dados para a métrica específica
        df_metric = df_pivot.filter(like=metric)  # Filtra colunas relacionadas à métrica

        # Criar dicionário de resultados para a métrica atual
        results_data[metric] = {}  # Inicializa o dicionário para a métrica

        for method in methods_name:
            if method in df_metric.columns:
                # Adiciona os dados para o método atual no dicionário da métrica
                results_data[metric][method] = df_metric.loc[:, method].values  # Usa loc para acessar os valores



        # Gerar o LaTeX da tabela de dados
        latex_code = df_metric[methods_name].round(3).to_latex(index=False)

        # Calcular o resumo estatístico
        summary = {"Dataset": ["Média", "Mediana", "Mínimo", "Máximo"]}
        for method in methods_name:
            summary[method] = [
                np.mean(df_pivot[method][metric]),
                np.median(df_pivot[method][metric]),
                np.min(df_pivot[method][metric]),
                np.max(df_pivot[method][metric]),
            ]

        # Converter o resumo em DataFrame e gerar o LaTeX
        df_summary = pd.DataFrame(summary)
        summary_latex = df_summary[methods_name].round(3).to_latex(index=False)

        # Combinar as tabelas e salvar no arquivo
        final_latex = latex_code + "\n\n" + summary_latex
        with open(f"{battery_name}_{metric}.txt", "w", encoding="utf-8") as file:
            file.write(final_latex)

    ####################################################
    ####### BOX PLOT
    ####################################################

    # Criando subplots para as métricas
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)

    # Iterando sobre as métricas e eixos para criar os gráficos
    for ax, metric in zip(axes.flat, metrics):

        data = df_pivot[methods_name].copy().filter(like=metric)


        # Criando o boxplot com patch_artist=True para permitir coloração
        box = ax.boxplot(data[methods_name], patch_artist=True, labels=['RAW','ISOMAP','KPCA','LLE','LE','t-SNE','UMAP','KISOMAP']
)

        # Colorindo apenas os métodos KISOMAP e ISOMAP
        for patch, label in zip(box['boxes'], methods_name):
            if label in methods_name:
                patch.set_facecolor(custom_palette[label])  # Define a cor do método
            else:
                patch.set_facecolor("white")  # Outros métodos sem cor

        # Configurações do gráfico
        ax.set_ylabel(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if metric == 'CH':
            ax.set_ylim(-100, 5000)  # Ajuste para o intervalo desejado

    # Salvando os gráficos
    plt.savefig(battery_name+'_boxplots.jpeg', format='jpeg', dpi=300)
    plt.close()


    ####################################################
    ####### LINE PLOTS
    ####################################################
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plt.style.use('seaborn')

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    # Create subplots for each metric
    for idx, metric in enumerate(metrics):
        # Sort dataframe to put KISOMAP last for this metric
        df_sorted = df.copy()
        df_sorted['plot_order'] = df_sorted['Method'].map(lambda x: 2 if x == 'KISOMAP' else 1 if x == 'ISOMAP' else 0)
        df_sorted = df_sorted.sort_values('plot_order')

        # Define custom color palette
        custom_palette = {method: '#023EFF' if method == 'KISOMAP' else '#A8A8A8'
                         for method in df_sorted['Method'].unique()}

        # Create the line plot for this metric
        sns.lineplot(data=df_sorted, x='Dataset', y=metric, hue='Method',
                    palette=custom_palette, linewidth=2.5, ax=axes_flat[idx])

        # Customize each subplot
        #axes_flat[idx].set_xlabel('Datasets')
        axes_flat[idx].set_ylabel(f'{metric} Values')
        if idx == 1:
            axes_flat[idx].set_ylim(-100,5000)
        axes_flat[idx].set_title(f'{metric} Metric')
        axes_flat[idx].tick_params(axis='x', rotation=45, labelbottom=False)
        # Move legend outside of plot
        
        if idx == 2:
            axes_flat[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes_flat[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes_flat[idx].get_legend().remove()


    plt.savefig(battery_name+'_lineplots.jpeg',format='jpeg',dpi=300)
    plt.close()

if __name__ == '__main__':
    sys.exit(main())