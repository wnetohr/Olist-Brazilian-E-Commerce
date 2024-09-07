import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import altair as alt
from scipy import stats
import notebooks.tools as tools
plt.style.use('ggplot')


def load_data(file_path):
    return pd.read_parquet(file_path)

seed = 42

df = pd.read_parquet(path='./data/outputs/eda_dataset.parquet')

df_summary = df.copy()
df_summary.rename(columns=tools.translations, inplace=True)

st.title("🔬Análise Exploratória")

st.subheader("Olist E-Commerce Dataset")
st.write("""Este conjunto de dados foi gentilmente fornecido pela Olist, uma grande loja de departamento nos marketplaces brasileiros.

Após a compra do produto na Olist Store, um vendedor é notificado para atender a esse pedido. Assim que o cliente recebe o produto, ou quando a data de entrega estimada chega, o cliente recebe uma pesquisa de satisfação por e-mail, onde pode dar uma nota para a experiência de compra e escrever alguns comentários.""")

st.subheader("Esquema do Dataset:")
st.write("""Os dados estão divididos em vários conjuntos de dados para melhor compreensão e organização. Aqui está sua arquitetura:""")
st.image(".\project_assets\olist_dataset_schema.png")

def resumirtabela(df):
    print(f"Shape: {df.shape}")
    resumo = pd.DataFrame(df.dtypes, columns=['dtypes'])
    resumo = resumo.reset_index()
    resumo['Nome'] = resumo['index']
    resumo = resumo[['Nome', 'dtypes']]
    resumo['Ausentes'] = df.isnull().sum().values
    resumo['Únicos'] = df.nunique().values
    resumo['Primeiro Valor'] = df.loc[0].values
    resumo['Segundo Valor'] = df.loc[1].values
    resumo['Terceiro Valor'] = df.loc[2].values

    for name in resumo['Nome'].value_counts().index:
        resumo.loc[resumo['Nome'] == name, 'Entropia'] = round(
            stats.entropy(df[name].value_counts(normalize=True), base=2), 2)
    return resumo

st.subheader("Resumo do Dataset")
with st.expander("Clique aqui para ver o Resumo do Dataset", expanded=False):
    st.dataframe(resumirtabela(df_summary))

id_cols = [
    'order_id', 'seller_id', 'customer_id', 'order_item_id',
      'product_id', 'review_id', 'customer_unique_id', 'seller_zip_code_prefix']

cat_cols = df.nunique()[df.nunique() <= 27].keys().tolist()

num_cols = num_cols = ['review_score', 'payment_sequential', 'payment_installments',
                   'payment_value', 'price', 'freight_value', 'product_name_lenght',
                   'product_description_lenght', 'product_photos_qty', 'product_weight_g',
                   'product_length_cm', 'product_height_cm', 'product_width_cm']

bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()

timestamp_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                  'order_estimated_delivery_date']

corr = df[num_cols].corr()

corr.rename(columns=tools.translations, inplace=True)
corr.index = corr.index.map(tools.translations)

tab_correlacoes, tab_categorias, tab_valores_e_pagamentos, tab_clientes_e_vendedores, tab_datas_comerciais = st.tabs(['Correlações', 'Categorias', 'Valores e Pagamentos', 'Clientes e Vendedores', 'Datas Comerciais'])

with tab_correlacoes:

    st.title("Heatmap de Correlação com Filtros")

    selected_y_axis = st.multiselect("Selecione as colunas para incluir no Eixo Y:", 
                                    options=corr.columns.tolist(), 
                                    default=['Valor do Pagamento', 'Preço', 'Valor do Frete', 'Peso do Produto (g)'])
    selected_x_axis = st.multiselect("Selecione as colunas para incluir no Eixo X:", 
                                    options=corr.columns.tolist(), 
                                    default=corr.columns.tolist())

    filtered_corr = corr.loc[selected_y_axis, selected_x_axis]

    filtered_corr_melted = filtered_corr.reset_index().melt(id_vars='index', var_name='variable', value_name='correlation')
    filtered_corr_melted.rename(columns={'index': 'feature'}, inplace=True)

    heatmap = (
        alt.Chart(filtered_corr_melted)
        .mark_rect()
        .encode(
            x=alt.X('variable:O', title='Eixo X', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('feature:O', title='Eixo Y'),
            color=alt.Color('correlation:Q', scale=alt.Scale(domain=[-1, 1], range=['blue', 'white', 'red']), title='Correlação'),
        )
        .properties(
            title='Heatmap de Correlação',
            width=600,
            height=600
        )
    )

    text = (
        alt.Chart(filtered_corr_melted)
        .mark_text(color='black')
        .encode(
            x='variable:O',
            y='feature:O',
            text=alt.Text('correlation:Q', format='.2f'),
        )
    )

    final_chart = heatmap + text

    st.altair_chart(final_chart, use_container_width=True)

with tab_categorias:

    df_mean_score = df.copy()
    df_mean_score.dropna(subset=['product_category_name'], inplace=True)

    df_mean_score['product_category_name'] = df_mean_score['product_category_name'].apply(lambda x: x.replace('_', ' ').title())
    avg_rating_best = df_mean_score.groupby('product_category_name')['review_score'].mean().reset_index()

    top_categories_best = avg_rating_best.nlargest(14, 'review_score')

    categories = st.multiselect(
        'Selecione as Categorias', 
        options=avg_rating_best['product_category_name'].unique(), 
        default=top_categories_best['product_category_name'].tolist()
    )

    filtered_df = avg_rating_best[avg_rating_best['product_category_name'].isin(categories)]

    barplot_best = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('review_score:Q', title='Nota da Avaliação'),
        y=alt.Y('product_category_name:N', title='Categoria do Produto', sort='-x'),
        color='product_category_name:N',
        tooltip=['product_category_name:N', 'review_score:Q']
    ).properties(
        title='Avaliação Média',
        width='container'
    )

    text = barplot_best.mark_text(
        align='left',
        baseline='middle',
        dx=3 
    ).encode(
        text=alt.Text('review_score:Q', format='.2f')
    )

    final_plot = barplot_best + text

    st.altair_chart(final_plot, use_container_width=True)

with tab_valores_e_pagamentos:
    df_sample = df.sample(500, random_state=seed)

    df_sample['product_weight_g'] = df_sample['product_weight_g'].apply(lambda x: x/1000)
    df_sample = df_sample.rename(columns={'product_weight_g': 'Peso do Produto (Kg)'})

    payment_value_q99 = df_sample['payment_value'].quantile(0.99)
    df_sample = df_sample[df_sample['payment_value'] < payment_value_q99]

    freight_value_q99 = df_sample['freight_value'].quantile(0.99)
    df_sample = df_sample[df_sample['freight_value'] < freight_value_q99]

    product_weight_q95 = df_sample['Peso do Produto (Kg)'].quantile(0.95)
    df_sample = df_sample[df_sample['Peso do Produto (Kg)'] < product_weight_q95]

    df_sample.rename(columns=tools.translations, inplace=True)

    payment_types = df_sample['Tipo de Pagamento'].unique()
    selected_payment_types = st.multiselect('Selecione os Tipos de Pagamento', options=payment_types, default=payment_types.tolist())

    filtered_df = df_sample[df_sample['Tipo de Pagamento'].isin(selected_payment_types)]

    variables = ['Valor do Pagamento', 'Preço', 'Valor do Frete', 'Peso do Produto (Kg)']

    scatterplot = alt.Chart(filtered_df).mark_circle().encode(
        alt.X(alt.repeat('column'), type='quantitative'),
        alt.Y(alt.repeat('row'), type='quantitative'),
        color='Tipo de Pagamento:N',
        tooltip=['Tipo de Pagamento:N', 'Valor do Pagamento:Q', 'Preço:Q', 'Valor do Frete:Q', 'Peso do Produto (Kg):Q']
    ).properties(
        width=150,
        height=200
    ).repeat(
        row=variables,
        column=variables
    ).interactive()

    histograms = alt.vconcat(
        *[alt.Chart(filtered_df).mark_bar().encode(
            alt.X(f'{var}:Q', bin=alt.Bin(maxbins=30)),
            alt.Y('count()', title='Contagem'),
            color='Tipo de Pagamento:N'
        ).properties(
            width=300,
            height=200
        ) for var in variables]
    )

    final_chart = alt.hconcat(
        histograms,
        scatterplot
    ).resolve_scale(color='shared')

    st.altair_chart(final_chart, use_container_width=True)

    df['price_log'] = np.log(df['price'] + 1.5)

    filtered_df = df[df['payment_type'].isin(selected_payment_types) & (df['payment_type'] != 'not_defined')]

    count_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('payment_type:N', title='Tipos de Pagamento', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('count():Q', title='Contagem'),
        color='payment_type:N',
        tooltip=['payment_type:N', alt.Tooltip('count():Q', title='Contagem')]
    ).properties(
        title='Distribuição dos Tipos de Pagamento',
        width='container',
        height=400
    )

    final_chart = alt.hconcat(
        count_chart 
    ).resolve_scale(color='shared')

    col1, col2 = st.columns(2)
    
    with col1:
        st.altair_chart(count_chart, use_container_width=True)

    chart = alt.Chart(filtered_df).transform_density(
        'price_log', 
        as_=['price_log', 'densidade'],
        groupby=['payment_type']
    ).mark_area(
        interpolate='basis',
        opacity=0.5
    ).encode(
        x='price_log:Q',
        y='densidade:Q',
        color='payment_type:N'
    ).properties(
        title='Distribuição de Preço por Tipo de Pagamento',
        width='container',
        height=400
    )

    with col2:
        st.altair_chart(chart, use_container_width=True)

with tab_clientes_e_vendedores:
    #TODO: FAZER FUNCIONAR
    count_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('count():Q', title='Count'),
        y=alt.Y('customer_state:N', title='Estados', sort='-x'),
        color='customer_state:N',
        tooltip=['customer_state:N', 'count():Q']
    ).properties(
        title='Distribuição de Clientes por Estados',
        width='container',
        height=400
    )

    box_chart = alt.Chart(df[df['price'] != -1]).mark_boxplot().encode(
        x=alt.X('customer_state:N', title='Estados'),
        y=alt.Y('price_log:Q', title='Preço (Log)'),
        color='customer_state:N'
    ).properties(
        title='Preço por Estados',
        width='container',
        height=400
    )

    box_chart2 = alt.Chart(df[df['price'] != -1]).mark_boxplot().encode(
        x=alt.X('freight_value:Q', title='Valor do Frete'),
        y=alt.Y('density:Q', title='Densidade'),
        color='customer_state:N'
    ).properties(
        title='Fretes por Estados',
        width='container',
        height=400
    )

    st.title('Distribuição dos Estados dos Clientes')

    st.altair_chart(count_chart)

    st.altair_chart(box_chart)

    st.altair_chart(box_chart2)