import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import altair as alt

produtos = pd.read_parquet('./data/olist_products_dataset.parquet')
pedidos = pd.read_parquet('./data/olist_orders_dataset.parquet')
items_de_pedido = pd.read_parquet('./data/olist_order_items_dataset.parquet')

# Produtos

st.title("Produtos")
st.write(produtos.head())

st.subheader("Estatísticas")
st.write(produtos.describe())
st.write(f'Shape: {produtos.shape}')

plt.figure(figsize=(10, 6))
produtos['product_category_name'].value_counts().plot(kind='bar')
plt.title('Contagem de Produtos por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Contagem')
st.pyplot(plt)

st.subheader("Valores únicos")
st.write(f'Categoria: {produtos.product_category_name.nunique()}')

st.subheader("Valores Nulos")
st.write(produtos.isnull().sum())

produtos = produtos[['product_id','product_category_name']].copy()
st.subheader('Produtos - ID e Categoria')
st.write(produtos.head())

# Pedidos

st.title("Pedidos")
st.write(pedidos.head())

st.subheader("Estatísticas")
st.write(pedidos.describe())
st.write(f'Shape: {pedidos.shape[0]}, {pedidos.shape[1]}')

plt.figure(figsize=(10, 6))
pedidos['order_status'].value_counts().plot(kind='bar')
plt.title('Distribuição de Status dos Pedidos')
plt.xlabel('Status')
plt.ylabel('Contagem')
st.pyplot(plt)

pedidos['order_purchase_timestamp'] = pd.to_datetime(pedidos['order_purchase_timestamp'])

plt.figure(figsize=(12, 6))
pedidos['order_purchase_timestamp'].groupby(pedidos['order_purchase_timestamp'].dt.to_period('M')).count().plot(kind='line')
plt.title('Distribuição de Pedidos ao Longo do Tempo')
plt.xlabel('Mês')
plt.ylabel('Número de Pedidos')
st.pyplot(plt)

pedidos = pedidos[['order_id', 'customer_id', 'order_status']].copy()
st.subheader('Pedidos - ID e Status')
st.write(pedidos.head())

st.subheader("Valores únicos")
st.write(f'order_id: {pedidos.order_id.nunique()}')
st.write(f'customer_id: {pedidos.customer_id.nunique()}')
st.write(f'order_status: {pedidos.order_status.nunique()}')

# Items de pedidos

st.title("Items de pedidos")
st.write(items_de_pedido.head())

st.subheader("Estatísticas")
st.write(items_de_pedido.describe())
st.write(f'Shape: {items_de_pedido.shape[0]}, {items_de_pedido.shape[1]}')

st.subheader('Estatísticas - Preços')

media_precos = items_de_pedido.price.mean()
desviopadrao_precos = items_de_pedido.price.std()
mediana_precos = items_de_pedido.price.median(numeric_only=True)

st.write(f'Média dos Preços: {media_precos:.2f}')
st.write(f'Desvio Padrão dos Preços: {desviopadrao_precos:.2f}')
st.write(f'Mediana dos Preços: {mediana_precos:.2f}')

c = (alt.Chart(items_de_pedido).mark_boxplot().encode(x='price'))

st.altair_chart(c, use_container_width=True)

st.subheader('Estatísticas - Fretes')

media_fretes = items_de_pedido.freight_value.mean()
desviopadrao_fretes = items_de_pedido.freight_value.std()
mediana_fretes = items_de_pedido.freight_value.median(numeric_only=True)

st.write(f'Média dos Fretes: {media_fretes:.2f}')
st.write(f'Desvio Padrão dos Fretes: {desviopadrao_fretes:.2f}')
st.write(f'Mediana dos Fretes: {mediana_fretes:.2f}')

c = (alt.Chart(items_de_pedido).mark_boxplot().encode(x='freight_value'))

st.altair_chart(c, use_container_width=True)

plt.figure(figsize=(10, 6))
items_de_pedido['order_id'].value_counts().plot(kind='hist', bins=20)
plt.title('Distribuição de Itens por Pedido')
plt.xlabel('Número de Itens')
plt.ylabel('Contagem de Pedidos')
st.pyplot(plt)

st.subheader("Valores únicos")
st.write(f'order_id: {items_de_pedido.order_id.nunique()}')
st.write(f'product_id: {items_de_pedido.product_id.nunique()}')
st.write(f'seller_id: {items_de_pedido.seller_id.nunique()}')

st.subheader("Valores Nulos")
st.write(items_de_pedido.isnull().sum())

