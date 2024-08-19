import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt

def load_data(dataset):
    path = f'./data/olist_{dataset}_dataset.parquet'
    return pd.read_parquet(path=path)

customers = load_data('customers')
geolocation = load_data('geolocation')
order_items = load_data('order_items')
order_payments = load_data('order_payments')
order_reviews = load_data('order_reviews')
orders = load_data('orders')
sellers = load_data('sellers')

st.title('Clientes')
st.table(customers.head())
st.table(customers.describe())
st.subheader('Valores nulos')
st.table(customers.isnull().sum())

st.title('Geolocação')
st.table(geolocation.head())
st.table(geolocation.describe())
st.subheader('Valores nulos')
st.table(geolocation.isnull().sum())

st.title('Itens de Pedidos')
st.table(order_items.head())
st.table(order_items.describe())
st.subheader('Valores nulos')
st.table(order_items.isnull().sum())

st.title('Pagamentos')
st.table(order_payments.head())
st.table(order_payments.describe())
st.subheader('Valores nulos')
st.table(order_payments.isnull().sum())

st.title('Resenhas')
st.table(order_reviews.head())
st.table(order_reviews.describe())
st.subheader('Valores nulos')
st.table(order_reviews.isnull().sum())

st.title('Pedidos')
st.table(orders.head())
st.table(orders.describe())
st.subheader('Valores nulos')
st.table(orders.isnull().sum())

st.title('Vendedores')
st.table(sellers.head())
st.table(sellers.describe())
st.subheader('Valores nulos')
st.table(sellers.isnull().sum())