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

products_file = './data/olist_products_dataset.parquet'
orders_file = './data/olist_orders_dataset.parquet'
order_items_file = './data/olist_order_items_dataset.parquet'
customers_file = './data/olist_customers_dataset.parquet'
geolocation_file = './data/olist_geolocation_dataset.parquet'
payments_file = './data/olist_order_payments_dataset.parquet'
reviews_file = './data/olist_order_reviews_dataset.parquet'
sellers_file = './data/olist_sellers_dataset.parquet'
category_file = './data/product_category_name_translation.parquet'

products_df = load_data(products_file)
orders_df = load_data(orders_file)
order_items_df = load_data(order_items_file)
customers_df = load_data(customers_file)
geolocation_df = load_data(geolocation_file)
payments_df = load_data(payments_file)
reviews_df = load_data(reviews_file)
sellers_df = load_data(sellers_file)

orders_products = orders_df.merge(order_items_df, on='order_id', how='inner')
orders_products_customers = orders_products.merge(customers_df, on='customer_id', how='inner')
orders_products_customers_reviews = orders_products_customers.merge(reviews_df, on='order_id', how='inner')
orders_products_customers_reviews_payments = orders_products_customers_reviews.merge(payments_df, on='order_id', how='inner')
orders_products_customers_reviews_payments_sellers = orders_products_customers_reviews_payments.merge(sellers_df, on='seller_id', how='inner')
df = orders_products_customers_reviews_payments_sellers.merge(products_df, on='product_id', how='inner')

df_summary = df.copy()
df_summary.rename(columns=tools.translations, inplace=True)

st.title("🔬Análise Exploratória")

st.subheader("Olist E-Commerce Dataset")
st.write("""Este conjunto de dados foi gentilmente fornecido pela Olist, uma grande loja de departamento nos marketplaces brasileiros.

Após a compra do produto na Olist Store, um vendedor é notificado para atender a esse pedido. Assim que o cliente recebe o produto, ou quando a data de entrega estimada chega, o cliente recebe uma pesquisa de satisfação por e-mail, onde pode dar uma nota para a experiência de compra e escrever alguns comentários.""")

st.subheader("Esquema do Dataset:")
st.write("""Os dados estão divididos em vários conjuntos de dados para melhor compreensão e organização. Aqui está sua arquitetura:""")
st.image("G:\op0Games\Projects\Olist-Brazilian-E-Commerce\project_assets\olist_dataset_schema.png")

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
        x=alt.X('variable:O', title='Variável', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('feature:O', title='Características'),
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