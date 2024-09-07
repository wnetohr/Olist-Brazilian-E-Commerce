import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Clusterização")
st.subheader("Visualização dos dados clusterizados: Escolha do número de clusters e tratamento dos dados")

# Vizualição da escolha da formatação dos dados e do número de clusters
formato_dados = ["original", "normalizado", "padronizado"]
formato_escolhido = st.selectbox("Selecione o tratamento dos dados", formato_dados)
formato_arquivo = {
    "original": "og",
    "normalizado": "nml",
    "padronizado": "std"
}.get(formato_escolhido, "og")
        
cotovelo_path = "./data_visualization/elbow_20_perc_#.png"
scores_path = "./data_visualization/score_20_perc_#.png"
multiplas_silhuetas_path = "./data_visualization/multi_silhouette_score_#.png"
st.image(cotovelo_path.replace("#", formato_arquivo), caption='Cotovelo', use_column_width=False)
st.image(scores_path.replace("#", formato_arquivo), caption='Scores', use_column_width=False)
st.image(multiplas_silhuetas_path.replace("#", formato_arquivo), caption='Silhouetas de entre 2 e 12 clusters', use_column_width=False)

# Visualização dos dados individualmente por cluster, podendo escolher multiplos pra visualizar um por imagem
df = pd.read_parquet(path="./data/clean_data/df_cluster_kmeans_3.parquet")
st.subheader("Visualização dos dados individualmente por cluster")
dados = df.columns
dados_escolhidos = st.multiselect('Selecione a feature', dados, default=["comdate_diff"], placeholder="Selecione as features")

if dados_escolhidos:
    for feature in dados_escolhidos:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(x='cluster', y=feature, data=df, palette=['blue', 'green', 'red'], hue='cluster', legend=False, ax=ax)
        ax.set_title(f'Violin Plot de {feature} por Cluster')
        st.pyplot(fig, use_container_width=False) 
else:
    st.write("Nenhuma feature selecionada.")

# Visualização de todas as features por cluster
st.subheader("Visualização geral dos dados por cluster")
clusters = ["Cluster 0", "Cluster 1", "Cluster 2"]
cluster_escolhido = st.selectbox("Selecione o cluster", clusters, placeholder="Escolha um cluster")
clusters_opt = {
    "Cluster 0": "0",
    "Cluster 1": "1",
    "Cluster 2": "2"
}.get(cluster_escolhido, "0")
        
cluster_info_path = "./data_visualization/cluster_#_info.png"
st.image(cluster_info_path.replace("#", clusters_opt), caption=f'Dados gerais do cluster {clusters_opt}', use_column_width=False)

# Visualização da quantidade de compras por categorias dos clusters
dados_agrupados = pd.read_parquet(path="./data/clean_data/df_grouped_category_per_cluster.parquet")
st.subheader("Filtragem da quantidade de compras por categorias por cluster")
categorias = dados_agrupados['filtered_category'].unique()
selected_categories = st.multiselect('Selecione as categorias', categorias, default=categorias)
filtered_df = dados_agrupados[dados_agrupados['filtered_category'].isin(selected_categories)]

if not filtered_df.empty:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=filtered_df, x='cluster', y='count', hue='filtered_category')
    plt.title('Contagem de categorias por Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Compras', fontsize=14)
    plt.legend(title='Categorias', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt.gcf())
else:
    st.write("Nenhuma categoria selecionada.")