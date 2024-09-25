import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from math import ceil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Função para carregar os dados
@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

# Função para aplicar a amostragem e escalonamento
@st.cache_data
def process_data(df, formato_dados, sample_percentual, random_state=42):
    df_sample = df.sample(ceil(df.shape[0] * (sample_percentual / 100)), random_state=random_state)

    if formato_dados == "standard":
        scaler = StandardScaler()
        df_sample = scaler.fit_transform(df_sample)
    elif formato_dados == "minmax":
        scaler = MinMaxScaler()
        df_sample = scaler.fit_transform(df_sample)

    return df_sample

# Função para calcular scores de clusters
@st.cache_data
def calculate_cluster_scores(df_sample, k_range, score_types):
    fitted_kmeans = {}
    labels_kmeans = {}
    df_scores = []

    for n_clusters in range(k_range[0], k_range[1] + 1):
        tmp_scores = {"n_clusters": n_clusters}
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_clusters = kmeans.fit_predict(df_sample)

        fitted_kmeans[n_clusters] = kmeans
        labels_kmeans[n_clusters] = labels_clusters

        if "silhouette" in score_types:
            tmp_scores["silhouette"] = silhouette_score(df_sample, labels_clusters)
        if "calinski_harabasz" in score_types:
            tmp_scores["calinski_harabasz"] = calinski_harabasz_score(df_sample, labels_clusters)
        if "davies_bouldin" in score_types:
            tmp_scores["davies_bouldin"] = davies_bouldin_score(df_sample, labels_clusters)

        df_scores.append(tmp_scores)

    return pd.DataFrame(df_scores).set_index("n_clusters")

# Carregar e processar os dados
df = load_data("./data/cluster_data/category_seasonal_data.parquet")
formato_dados = st.selectbox("Selecione o tratamento dos dados", ["original", "normalizado", "padronizado"])
formato_dados = {"original": "original", "normalizado": "minmax", "padronizado": "standard"}[formato_dados]

# Aplicar processamento de dados (com cache para eficiência)
sample_percentual = 20
df_sample = process_data(df, formato_dados, sample_percentual)

# Divisão de interface em duas colunas
col1, col2 = st.columns(2)

# Gráfico de Elbow
with col1:
    k_range = st.slider("Selecione o intervalo de clusters", min_value=2, max_value=10, value=(2, 7))
    plt.figure()
    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(k_range[0], k_range[1]))
    visualizer.fit(df_sample)
    st.pyplot(plt.gcf())

# Gráfico de Silhouette
with col2:
    k_silhouette = st.slider("Selecione o número de clusters", min_value=2, max_value=10, value=3)
    plt.figure()
    km = KMeans(n_clusters=k_silhouette, random_state=42)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
    visualizer.fit(df_sample)
    st.pyplot(plt.gcf())

# Expandir para exibir as métricas de avaliação
with st.expander("Silhueta, Calinski Harabasz e Davies Bouldin", expanded=False):
    score_types = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    df_scores = calculate_cluster_scores(df_sample, k_range, score_types)
    df_scores.plot(subplots=True, layout=(1, len(score_types)), figsize=(len(score_types) * 6, 5))
    st.pyplot(plt.gcf())
st.subheader("EDA dos clusters obtidos")
st.markdown("""
Aqui, é possivel visualizar as caracteristicas de cada clustar referente a dimensão escolhida.
""")

tab_categorias, tab_valor_venda, tab_datas_comerciais = st.tabs(['Categorias', 'Valor de venda', 'Datas Comerciais'])

with tab_categorias:
    dados_agrupados = load_data("./data/cluster_data/df_grouped_category_per_cluster.parquet")

    with st.expander("Clique aqui os dados em um gráfico unico", expanded=False):
        plt.figure(figsize=(10, 4))
        sns.barplot(data=dados_agrupados, x='hue', y='qtd_percategory', hue='filtered_category')
        plt.title('Contagem de categorias por Cluster', fontsize=16)
        plt.xlabel('Cluster', fontsize=14)
        plt.ylabel('Quantidade de vendas', fontsize=14)
        plt.legend(title='Categorias', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt.gcf())

    st.subheader("Filtragem da quantidade de compras por categorias por cluster")
    clusters = dados_agrupados['hue'].unique()
    selected_clusters = st.multiselect('Selecione os clusters', clusters, default=clusters, key='f_categ')

    categorias = dados_agrupados['filtered_category'].unique()
    selected_categories = st.multiselect('Selecione as categorias', categorias, default=categorias)
    filtered_df = dados_agrupados[dados_agrupados['filtered_category'].isin(selected_categories)]

    if not filtered_df.empty:

        for i in selected_clusters:
            plt.figure(figsize=(8, 4))
                
            data_per_cluster = filtered_df[filtered_df['hue'] == i].sort_values(['qtd_percategory'], ascending=False).reset_index(drop=True) 
            top3 = list(data_per_cluster['filtered_category'])[:3]
            lower3 = list(data_per_cluster['filtered_category'])[-3:]
            sns.set(style="whitegrid")
            sns.barplot(data=data_per_cluster, x='qtd_percategory', y='hue', hue='filtered_category')
            plt.legend(title='Categorias', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Quantidade de vendas por categorias no {i}', fontsize=16)
            plt.ylabel('')
            plt.xlabel('Quantidade de vendas', fontsize=14)
            st.pyplot(plt.gcf())
    else:
        st.write("Nenhuma categoria selecionada.")

with tab_datas_comerciais:
    dados_feriados = load_data("./data/cluster_data/df_grouped_commercialdate_per_cluster.parquet")

    with st.expander("Clique aqui os dados em um gráfico unico", expanded=False):
        plt.figure(figsize=(10, 4))
        sns.barplot(data=dados_feriados, x='hue', y='qtd_percommercialdate', hue='commercial_date')
        plt.title('Contagem de feriados por Cluster', fontsize=16)
        plt.xlabel('Cluster', fontsize=14)
        plt.ylabel('Quantidade de vendas', fontsize=14)
        plt.legend(title='Feriados', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt.gcf())

    st.subheader("Filtragem da quantidade de compras por datas comerciais por cluster")
    clusters_2 = dados_feriados['hue'].unique()
    selected_clusters_2 = st.multiselect('Selecione os clusters', clusters_2, default=clusters_2, key='f_datacomerc')

    feriados = dados_feriados['commercial_date'].unique()
    selected_feriados = st.multiselect('Selecione as datas comerciais', feriados, default=feriados)
    filtered_df_feriados = dados_feriados[dados_feriados['commercial_date'].isin(selected_feriados)]

    if not filtered_df.empty:

        for i in selected_clusters_2:
            plt.figure(figsize=(8, 3))
                
            data_per_cluster = filtered_df_feriados[filtered_df_feriados['hue'] == i].sort_values(['qtd_percommercialdate'], ascending=False).reset_index(drop=True) 
            top3 = list(data_per_cluster['commercial_date'])[:3]
            lower3 = list(data_per_cluster['commercial_date'])[-3:]
            sns.set(style="whitegrid")
            sns.barplot(data=data_per_cluster, x='qtd_percommercialdate', y='hue', hue='commercial_date')
            plt.legend(title='Datas comerciais', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Quantidade de vendas por datas comerciais no {i}', fontsize=16)
            plt.ylabel('')
            plt.xlabel('Quantidade de vendas', fontsize=14)
            st.pyplot(plt.gcf())
    else:
        st.write("Nenhuma categoria selecionada.")

with tab_valor_venda:
    gasto_medio = load_data("./data/cluster_data/avg_spending.parquet")

    st.subheader("Filtragem da quantidade de compras por datas comerciais por cluster")
    clusters_3 = dados_feriados['hue'].unique()
    selected_clusters_3 = st.multiselect('Selecione os clusters', clusters_3, default=clusters_3, key='f_gastomed')
    filtered_data = gasto_medio[gasto_medio['hue'].isin(selected_clusters_3)]

    selected_dates = st.multiselect(
        'Selecione as datas comerciais:',
        options=gasto_medio['commercial_date'].unique(),
        default=gasto_medio['commercial_date'].unique()
    )
    filtered_data = filtered_data[filtered_data['commercial_date'].isin(selected_dates)]

    heatmap = alt.Chart(filtered_data).mark_rect().encode(
        x=alt.X('hue:N', title='Clusters'),
        y=alt.Y('commercial_date:N', title='Datas Comerciais'),
        color=alt.Color('total_spending:O', title='Gasto Médio', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['commercial_date:N', 'hue:N', 'total_spending:N'],
        
    ).properties(
        width=600,
        height=400
    )
    text = heatmap.mark_text(baseline='middle', color='black').encode(
        text='total_spending:N',
        color=alt.value('black')
    )

    final_heatmap = heatmap + text

    st.altair_chart(final_heatmap, use_container_width=True)
