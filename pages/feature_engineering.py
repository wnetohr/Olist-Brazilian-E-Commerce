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
st.subheader("Pagamentos com múltiplos metodos")
st.write(customers_orders_payments['payment_sequential'].value_counts())

# Numero de pedidos por cliente
customers_orders_payments['number_of_orders'] = customers_orders_payments.groupby('customer_unique_id')['order_id'].transform('count')
st.subheader('Número de pedidos por cliente')
st.write(customers_orders_payments['number_of_orders'].value_counts())

percentage_of_multiple_sales = customers_orders_payments[customers_orders_payments['number_of_orders'] > 1]['customer_unique_id'].nunique()/customers_orders_payments['customer_unique_id'].nunique()*100

st.write(f"Apenas {percentage_of_multiple_sales:.2f}% dos clientes compraram mais de uma vez!")

c = (alt.Chart(customers_orders_payments).mark_bar(opacity=0.7).encode(
    alt.X('payment_installments', bin=True), 
    alt.Y('count()', title='Payment Installments Count')
    ).properties(
        title='Payment Installments Count'
    ))
st.altair_chart(c, use_container_width=True)

anos = customers_orders_payments['order_purchase_timestamp'].dt.year
st.write(anos.value_counts())

# Datas comerciais
commerce_dates = pd.read_parquet("./data/feriados_comerciais.parquet")
st.subheader("Datas comerciais:")
st.write(commerce_dates)

customers_orders_payments['order_purchase_timestamp'] = pd.to_datetime(customers_orders_payments['order_purchase_timestamp'])
commerce_dates['Datas'] = pd.to_datetime(commerce_dates['Datas'])

def ocorreu_antes_data_comercial(data_venda, datas_comerciais):
    for data_comercial in datas_comerciais:
        if data_comercial - pd.Timedelta(days=7) <= data_venda <= data_comercial:
            return "Yes"
    return "No"

customers_orders_payments['order_week_before_comdate'] = customers_orders_payments['order_purchase_timestamp'].apply(lambda x: ocorreu_antes_data_comercial(x, commerce_dates['Datas']))
st.subheader("Quantas compras foram feitas em um período de uma semana antes de uma data comercial:")
st.write(customers_orders_payments['order_week_before_comdate'].value_counts())
st.write(customers_orders_payments.head())

c = (alt.Chart(customers_orders_payments).mark_bar().encode(
    alt.X('order_week_before_comdate'),
    alt.Y('count()', title='Week Before Commercial Date Count')
    ).properties(
        title='Week Before Commercial Date Count'
    ))
st.altair_chart(c, use_container_width=True)

customers_orders_payments_items = pd.merge(customers_orders_payments, order_items, on='order_id', how='inner')
customer_products = customers_orders_payments_items.merge(
    products, on='product_id', how='inner'
)

# Distribuição de categorias
st.subheader('Distribuição de categorias:')
count_category = customer_products['product_category_name'].value_counts()
c = (alt.Chart(customer_products).mark_bar().encode(
    alt.Y('product_category_name', title='Categories').sort('-x'),
    alt.X('count()', title='Category Sales')
    ).properties(
        title='Category Sales'
    ))
st.altair_chart(c, use_container_width=True)
st.write(count_category)
category_cut = int(customer_products['product_category_name'].nunique() * 0.2)
st.write(int(category_cut))
top_20percent_category = count_category.head(category_cut).index
st.write(top_20percent_category)

customer_products['filtered_category'] = customer_products['product_category_name'].apply(
    lambda x: x if x in top_20percent_category else 'outros'
)

st.write(customer_products['filtered_category'].value_counts())

c = (alt.Chart(customer_products).mark_bar().encode(
    alt.Y('filtered_category', title='Filtered Category').sort('-x'),
    alt.X('count()', title='Category Sales')
    ).properties(
        title='Category Sales'
    ))
st.altair_chart(c, use_container_width=True)
