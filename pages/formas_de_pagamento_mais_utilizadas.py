import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_data(file_path):
    return pd.read_parquet(file_path)

def analyze_data(df):
    st.title("Análise de Formas de Pagamento e Parcelamento")

    payment_counts = df['payment_type'].value_counts()
    st.subheader("Formas de Pagamento Mais Utilizadas:")
    st.write(payment_counts)

    avg_installments = df[df['payment_type'] == 'credit_card']['payment_installments'].mean()
    st.subheader("Média de Parcelas para Compras com Cartão de Crédito:")
    st.write(round(avg_installments, 2))

file_path = './data/olist_order_payments_dataset.parquet'
data = load_data(file_path)

analyze_data(data)