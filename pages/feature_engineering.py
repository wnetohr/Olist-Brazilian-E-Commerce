import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import datetime as dt
import os
import warnings
warnings.filterwarnings('ignore')

st.title("Engenharia de Features")

def load_data(dataset):
    path = f'./data/olist_{dataset}_dataset.parquet'
    return pd.read_parquet(path=path)

customers = load_data('customers')
geolocation = load_data('geolocation')
order_items = load_data('order_items')
order_payments = load_data('order_payments')
order_reviews = load_data('order_reviews')
orders = load_data('orders')
products = load_data('products')
sellers = load_data('sellers')

#união de clientes e pedidos
customers_orders = pd.merge(customers, orders, on='customer_id', how='inner')
st.subheader("Clientes e pedidos")
st.write(customers_orders.count())
st.table(customers_orders.head())
st.write('Duplicatas')
st.write(customers_orders.duplicated().sum())
st.write('Nulos')
st.write(customers_orders.isnull().sum())

# Vamos remover os valores que estão ausentes
customers_orders.dropna(inplace=True)

st.subheader("Clientes que compraram mais de uma vez:")
st.write(customers_orders['customer_unique_id'].duplicated().sum())
st.table(customers_orders[customers_orders['customer_unique_id'].duplicated()].head())
st.table(customers_orders[customers_orders['customer_unique_id'].duplicated()].describe())
st.table(customers_orders.nunique())

# Verificando distribuição de order status
st.subheader("Distribuição - Order Status")
st.write(customers_orders['order_status'].value_counts())

st.write(customers_orders.groupby('order_status')['customer_unique_id'].nunique())

plt.figure(figsize=(10,6))
sns.countplot(x='order_status', data=customers_orders)
plt.title('Order Status Count', fontsize=16)
plt.xlabel('Order Status', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
st.pyplot(plt)

customers_orders['order_purchase_timestamp'] = pd.to_datetime(customers_orders['order_purchase_timestamp'])
customers_orders['order_approved_at'] = pd.to_datetime(customers_orders['order_approved_at'])
customers_orders['order_delivered_carrier_date'] = pd.to_datetime(customers_orders['order_delivered_carrier_date'])
customers_orders['order_delivered_customer_date'] = pd.to_datetime(customers_orders['order_delivered_customer_date'])
customers_orders['order_estimated_delivery_date'] = pd.to_datetime(customers_orders['order_estimated_delivery_date'])

# Mudando formato para YYYY-MM-DD

customers_orders['order_purchase_timestamp'] = customers_orders['order_purchase_timestamp'].dt.strftime('%Y-%m-%d')
customers_orders['order_approved_at'] = customers_orders['order_approved_at'].dt.strftime('%Y-%m-%d')
customers_orders['order_delivered_carrier_date'] = customers_orders['order_delivered_carrier_date'].dt.strftime('%Y-%m-%d')
customers_orders['order_delivered_customer_date'] = customers_orders['order_delivered_customer_date'].dt.strftime('%Y-%m-%d')
customers_orders['order_estimated_delivery_date'] = customers_orders['order_estimated_delivery_date'].dt.strftime('%Y-%m-%d')

customers_orders['on_time_delivery'] = np.where(customers_orders['order_estimated_delivery_date'] == customers_orders['order_delivered_customer_date'], 'Yes', 'No')

# Entregas que foram entregues no tempo estimado
st.subheader("Entregas feitas na data estimada:")
st.write(customers_orders['on_time_delivery'].value_counts())

# Checando se a aprovação dos pedidos aconteceu antes da data de compra
st.subheader("Checando se há aprovação de pedidos antes da compra:")
st.write(customers_orders.where(customers_orders['order_approved_at'] < customers_orders['order_purchase_timestamp']).count())

# Checando diferenca em dias entre a compra e a aprovação
st.subheader("Checando diferença em dias entre compra e aprovação:")
customers_orders['order_purchase_timestamp'] = pd.to_datetime(customers_orders['order_purchase_timestamp'])
customers_orders['order_approved_at'] = pd.to_datetime(customers_orders['order_approved_at'])
customers_orders['approved_date'] = customers_orders['order_purchase_timestamp'] - customers_orders['order_approved_at']
customers_orders['approved_date'] = customers_orders['approved_date'].dt.days
st.write(customers_orders['approved_date'].value_counts())

customers_orders['order_delivered_carrier_date'] = pd.to_datetime(customers_orders['order_delivered_carrier_date'])
customers_orders['order_delivered_customer_date'] = pd.to_datetime(customers_orders['order_delivered_customer_date'])

# Clientes, pedidos e pagamentos
customers_orders_payments = pd.merge(customers_orders, order_payments, on='order_id', how='inner')
st.subheader('Tipos de pagamentos')
st.write(customers_orders_payments['payment_type'].value_counts())

customers_orders_payments = customers_orders_payments[customers_orders_payments['payment_type'] != 'not_defined']

# distribuição de parcelas
st.subheader("Distribuição de parcelas")
st.write(customers_orders_payments['payment_sequential'].value_counts())

# Numero de pedidos por cliente
customers_orders_payments['number_of_orders'] = customers_orders_payments.groupby('customer_unique_id')['order_id'].transform('count')
st.subheader('Número de pedidos por cliente')
st.write(customers_orders_payments['number_of_orders'].value_counts())

percentage_of_multiple_sales = customers_orders_payments[customers_orders_payments['number_of_orders'] > 1]['customer_unique_id'].nunique()/customers_orders_payments['customer_unique_id'].nunique()*100

st.write(f"Apenas {percentage_of_multiple_sales:.2f}% dos clientes compraram mais de uma vez!")

c = (alt.Chart(customers_orders_payments).mark_bar(opacity=0.7).encode(
    alt.X('payment_installments:Q', bin=True), 
    alt.Y('count()', title='Payment Installments Count')
    ).properties(
        title='Payment Installments Count'
    ))
st.altair_chart(c, use_container_width=True)