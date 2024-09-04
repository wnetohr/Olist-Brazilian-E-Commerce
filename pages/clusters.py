import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dados_agrupados = pd.read_parquet(path="./data/clean_data/df_grouped_category_per_cluster.parquet")

st.title("Visualização dos dados clusterizados")

formato_dados = ["original", "normalizado", "padronizado"]
formato_escolhido = st.selectbox("Selecione o tratamento dos dados", formato_dados)

cotovelo_path = "./data_visualization/elbow_20_perc_#.png"
scores_path = "./data_visualization/score_20_perc_#.png"
multiplas_silhuetas_path = "./data_visualization/multi_silhouette_score_#.png"

formato_arquivo = {
    "original": "og",
    "normalizado": "nml",
    "padronizado": "std"
}.get(formato_escolhido, "og")
        
st.image(cotovelo_path.replace("#", formato_arquivo), caption='Cotovelo', use_column_width=False)
st.image(scores_path.replace("#", formato_arquivo), caption='Scores', use_column_width=False)
st.image(multiplas_silhuetas_path.replace("#", formato_arquivo), caption='Silhouetas de entre 2 e 12 clusters', use_column_width=False)

categories = dados_agrupados['filtered_category'].unique()
selected_categories = st.multiselect('Select Categories', categories, default=categories)

filtered_df = dados_agrupados[dados_agrupados['filtered_category'].isin(selected_categories)]

if not filtered_df.empty:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=filtered_df, x='cluster', y='count', hue='filtered_category')
    plt.title('Contagem de categorias por Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    plt.legend(title='Filtered Category', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt.gcf())
else:
    st.write("No categories selected.")