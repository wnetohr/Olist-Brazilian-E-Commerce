import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

reviews = pd.read_parquet('./data/olist_order_reviews_dataset.parquet')
orders = pd.read_parquet('./data/olist_orders_dataset.parquet')

st.title("Análise de Satisfação")
st.subheader("Média de Avalíações por período (Ano-Mês)")

scores_by_order_id = reviews[['order_id','review_score']]
orders_by_date_of_purchase = orders[['order_id', 'order_purchase_timestamp']].copy()

orders_by_date_of_purchase['order_purchase_timestamp'] = pd.to_datetime(orders_by_date_of_purchase['order_purchase_timestamp'])
orders_by_date_of_purchase.loc[:, 'month_year_of_purchase'] = orders_by_date_of_purchase['order_purchase_timestamp'].dt.to_period('M')

reviews_by_period = pd.merge(scores_by_order_id, orders_by_date_of_purchase[['order_id', 'month_year_of_purchase']], how='left', on='order_id')

mean_scores_by_period = reviews_by_period.groupby('month_year_of_purchase')['review_score'].mean()

mean_scores_by_period.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Média das avaliações por Período')
plt.xlabel('Período (Ano-Mês)')
plt.ylabel('Média dos Scores')
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)