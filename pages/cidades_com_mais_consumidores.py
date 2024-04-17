import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def load_data(file_path):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    dt_customers = load_data('./data/olist_customers_dataset.csv')
    cities = dt_customers['customer_city'].value_counts()
    top_10 = cities.head(10)
    
    plt.figure(figsize=(10, 6))
    top_10.plot(kind='bar')
    st.title("As dez cidades com mais consumidores")
    plt.xlabel('Cidades')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)

    st.pyplot(plt)