import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def load_data(file_path):
    return pd.read_parquet(file_path)


if __name__ == "__main__":
    dt_customerrs = load_data('./data/olist_sellers_dataset.parquet')
    cities = dt_customerrs['seller_city'].value_counts()
    top_10 = cities.head(10)

    plt.figure(figsize=(10, 6))
    top_10.plot(kind='bar')
    st.title("As dez cidades com mais vendedores")
    plt.xlabel('Cidades')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)

    st.pyplot(plt)