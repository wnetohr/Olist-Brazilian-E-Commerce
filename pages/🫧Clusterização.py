import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

st.title("Clusterização")
st.markdown("""
Esta pagina conterá visualizações do processo de clusterização e do EDA desses clusters. 
            
O intuito dessa pagina é fornecer uma visualização dos algortimos de cotovelo, silhoueta, Calinski Harabasz e Davies Bouldin, 
            algoritmos esses utilizados para decidir qual o melhor número de clusters para o nosso algoritmo de clusterização, K-Means.
Essa pagina conterá também o EDA realizado sobre esses clusters. As visoões sobre os clusters aplicadas aqui serão referentes as categorias
            dos produtos vendidos, a relação dessas vendas por feriados comercias e por valor das vendas. 
""")

# Vizualição da escolha da formatação dos dados e do número de clusters
st.subheader("Escolha do número de clusters e tratamento dos dados")
st.markdown("""
Como possuimos um quantitativo de mais de 80 mil vendas, utilizamos uma amostragem de 20% dos dados para aplicar os algoritmos de 
            cotovelo, silhoueta, Calinski Harabasz e Davies Bouldin a procura do melhor número de clusters.]
Com essa amostragem de 20%, aplicamos os algortimos nos dados em sua forma original, depois os mesmos algortimos com os dados normalizados 
            e depois os mesmos algoritmos com os dados padronizados. Ao comparar os resultados obtidos, a clusterização dos dados normalizados
            em 3 clusters mostraram resultados mais satisfatorios.  
""")

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

# Visualização do EDA
st.subheader("EDA dos clusters obtidos")
st.markdown("""
Aqui, é possivel visualizar as caracteristicas de cada clustar referente a dimensão escolhida.
""")

tab_categorias, tab_valor_venda, tab_datas_comerciais = st.tabs(['Categorias', 'Valor de venda', 'Datas Comerciais'])

with tab_categorias:
    dados_agrupados = pd.read_parquet(path="./data/cluster_data/df_grouped_category_per_cluster.parquet")

    with st.expander("Clique aqui os dados em um gráfico unico", expanded=False):
        plt.figure(figsize=(10, 4))
        sns.barplot(data=dados_agrupados, x='hue', y='qtd_percategory', hue='filtered_category')
        plt.title('Contagem de categorias por Cluster', fontsize=16)
        plt.xlabel('Cluster', fontsize=14)
        plt.ylabel('Quantidade de vendas', fontsize=14)
        plt.legend(title='Categorias', fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt.gcf())

    # Flitragem de clusters
    st.subheader("Filtragem da quantidade de compras por categorias por cluster")
    clusters = dados_agrupados['hue'].unique()
    selected_clusters = st.multiselect('Selecione os clusters', clusters, default=clusters, key='f_categ')

    # Flitragem de categorias
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
    dados_feriados = pd.read_parquet(path="./data/cluster_data/df_grouped_commercialdate_per_cluster.parquet")

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

    # Flitragem de datas comerciais
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
    gasto_medio = pd.read_parquet(path="./data/cluster_data/avg_spending.parquet")

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

    # Compor o heatmap e o texto
    final_heatmap = heatmap + text

    st.altair_chart(final_heatmap, use_container_width=True)
