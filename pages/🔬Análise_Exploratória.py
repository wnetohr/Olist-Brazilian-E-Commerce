import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import altair as alt
from scipy import stats
import notebooks.tools as tools
from scipy.stats import gaussian_kde
plt.style.use('ggplot')

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

seed = 42

df = load_data('./data/outputs/eda_dataset.parquet')

@st.cache_data
def df_sampler(seed, df):
    df_sample = df.sample(500, random_state=seed)

    df_sample['product_weight_g'] = df_sample['product_weight_g'].apply(lambda x: x/1000)
    df_sample = df_sample.rename(columns={'product_weight_g': 'Peso do Produto (Kg)'})

    payment_value_q99 = df_sample['payment_value'].quantile(0.99)
    df_sample = df_sample[df_sample['payment_value'] < payment_value_q99]

    freight_value_q99 = df_sample['freight_value'].quantile(0.99)
    df_sample = df_sample[df_sample['freight_value'] < freight_value_q99]

    product_weight_q95 = df_sample['Peso do Produto (Kg)'].quantile(0.95)
    df_sample = df_sample[df_sample['Peso do Produto (Kg)'] < product_weight_q95]

    return df_sample

df_summary = df.copy()
df_summary.rename(columns=tools.translations, inplace=True)

st.title("🔬Análise Exploratória")

st.subheader("Olist E-Commerce Dataset")
st.write("""Este conjunto de dados foi gentilmente fornecido pela Olist, uma grande loja de departamento nos marketplaces brasileiros.

Após a compra do produto na Olist Store, um vendedor é notificado para atender a esse pedido. Assim que o cliente recebe o produto, ou quando a data de entrega estimada chega, o cliente recebe uma pesquisa de satisfação por e-mail, onde pode dar uma nota para a experiência de compra e escrever alguns comentários.""")

st.subheader("Esquema do Dataset:")
st.write("""Os dados estão divididos em vários conjuntos de dados para melhor compreensão e organização. Aqui está sua arquitetura:""")
st.image("./project_assets/olist_dataset_schema.png")

def resumirtabela(df):
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

@st.cache_data
def calcular_correlacao(df):
    corr = df[num_cols].corr()
    corr.rename(columns=tools.translations, inplace=True)
    corr.index = corr.index.map(tools.translations)
    return corr

corr = calcular_correlacao(df)

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
    df_sample = df_sampler(seed, df)


    payment_types = df_sample['payment_type'].unique()
    selected_payment_types = st.multiselect('Selecione os Tipos de Pagamento', options=payment_types, default=payment_types.tolist())

    if 'price_log' not in df.columns:
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


    @st.cache_data
    def calculate_density(data, column, group_column):
        density_data = []
        for group in data[group_column].unique():
            subset = data[data[group_column] == group][column]
            kde = gaussian_kde(subset)
            x = np.linspace(subset.min(), subset.max(), 100)
            y = kde(x)
            density_data.append(pd.DataFrame({column: x, 'densidade': y, group_column: group}))
        return pd.concat(density_data)

    density_df = calculate_density(filtered_df, 'price_log', 'payment_type')

    chart = alt.Chart(density_df).mark_area(
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

    with st.expander('Contagem e Distribuição:', expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.altair_chart(count_chart, use_container_width=True)

        with col2:
            st.altair_chart(chart, use_container_width=True)

    df_pairplot = df_sample.copy()

    df_pairplot.rename(columns=tools.translations, inplace=True)

    filtered_df = df_pairplot[df_pairplot['Tipo de Pagamento'].isin(selected_payment_types)]

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
    )

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

with tab_clientes_e_vendedores:

    st.title('Distribuição dos Estados dos Clientes')

    df_filtered = df[df['freight_value'] != -1]

    count_data = df['customer_state'].value_counts().reset_index()
    count_data.columns = ['customer_state', 'count']

    count_chart = alt.Chart(count_data).mark_bar(size=20).encode(
        x=alt.X('customer_state:N', title='Estados', sort='-x'),
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('customer_state:N', legend=None),
        tooltip=[alt.Tooltip('customer_state:N', title='Estado'), alt.Tooltip('count:Q', title='Total')]
    ).properties(
        title='Distribuição de Clientes por Estado',
        width='container',
        height=400
    ).configure_axisX(
        labelAngle=-45
    )

    box_chart = alt.Chart(df_filtered).mark_boxplot(size=25).encode(
        x=alt.X('customer_state:N', title='Estados', sort='-x'),
        y=alt.Y('freight_value:Q', title='Valor do Frete', scale=alt.Scale(zero=False)),
        color='customer_state:N',
        tooltip=['customer_state:N', 'freight_value:Q']
    ).properties(
        title='Preço por Estado',
        width='container',
        height=400
    ).configure_axisX(
        labelAngle=-45
    )

    st.altair_chart(count_chart, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.altair_chart(box_chart, use_container_width=True)

    with col2:
        st.subheader("São Paulo forma a maior parte dos clientes.")
        st.write("São Paulo lidera na quantidade de compras com aproximadamente 50.000, seguido pelo Rio de Janeiro e Minas Gerais com 15 mil e 13 mil compras respectivamente.")
        st.write("O Sudeste do país é responsável por, aproximadamente, 65% das compras registradas no dataset.")

with tab_datas_comerciais:

    df_w_dates = load_data('./data/outputs/df_w_commercial_dates.parquet')

    st.title('Análise de Pedidos')

    # Gráfico de linha: Evolução dos Pedidos Totais
    line_data =  df_w_dates['order_purchase_year_month'].value_counts().reset_index()
    line_data.columns = ['year_month', 'count']

    line_chart = alt.Chart(line_data).mark_line(interpolate='linear', strokeWidth=2).encode(
        x=alt.X('year_month:O', title='Mês e Ano'),
        y=alt.Y('count:Q', title='Contagem'),
    ).properties(
        title='Evolução dos Pedidos Totais',
        width=800,
        height=400
    ).configure_axisX(
        labelAngle=-45
    )

    st.altair_chart(line_chart, use_container_width=True)

    # Gráfico de histogramas: Pedidos Totais por Dia da Semana
    weekday_data =  df_w_dates['order_purchase_dayofweek'].value_counts().reset_index()
    weekday_data.columns = ['day_of_week', 'count']

    # Renomear dias da semana
    weekday_labels = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']
    weekday_data['day_of_week'] = weekday_data['day_of_week'].apply(lambda x: weekday_labels[x])

    histogram_weekday = alt.Chart(weekday_data).mark_bar().encode(
        x=alt.X('day_of_week:O', title='Dias', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('count:Q', title='Contagem'),
        color=alt.Color('day_of_week:N', legend=None),
        tooltip=['day_of_week:N', 'count:Q']
    ).properties(
        title='Pedidos Totais por Dia da Semana',
        width=800,
        height=400
    )

    # Adicionando rótulos de contagem e porcentagem
    total_weekday = weekday_data['count'].sum()
    text_weekday = histogram_weekday.mark_text(dy=8).encode(
        text=alt.Text('count:Q'),
        x='day_of_week:O',
        y='count:Q'
    )

    text_percentage_weekday = histogram_weekday.mark_text(dy=20).encode(
        text=alt.Text('count:Q', format='.1%'),
        x='day_of_week:O',
        y='count:Q'
    )


    # Gráfico de histogramas: Total de pedidos por Período do Dia
    time_period_data =  df_w_dates['order_purchase_time_day'].value_counts().reset_index()
    time_period_data.columns = ['time_period', 'count']

    histogram_time_period = alt.Chart(time_period_data).mark_bar().encode(
        x=alt.X('time_period:O', title='Período do Dia', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('count:Q', title='Contagem'),
        color=alt.Color('time_period:N', legend=None),
        tooltip=['time_period:N', 'count:Q']
    ).properties(
        title='Total de Pedidos por Período do Dia',
        width=800,
        height=400
    )

    # Adicionando rótulos de contagem e porcentagem
    total_time_period = time_period_data['count'].sum()
    text_time_period = histogram_time_period.mark_text(dy=8).encode(
        text=alt.Text('count:Q'),
        x='time_period:O',
        y='count:Q'
    )

    text_percentage_time_period = histogram_time_period.mark_text(dy=20).encode(
        text=alt.Text('count:Q', format='.1%'),
        x='time_period:O',
        y='count:Q'
    )
    col1, col2 = st.columns(2)

    with col1:
        st.altair_chart(histogram_weekday + text_weekday + text_percentage_weekday, use_container_width=True)

    with col2:
        st.altair_chart(histogram_time_period + text_time_period + text_percentage_time_period, use_container_width=True)
