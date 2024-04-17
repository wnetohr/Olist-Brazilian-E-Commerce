import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def load_data(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    dt_customers = load_data('./data/olist_customers_dataset.csv')
    dt_order = load_data('./data/olist_orders_dataset.csv')
    dt_order_per_customer = dt_order[['order_id', 'customer_id']]
    dt_customer_per_city = dt_customers[['customer_id', 'customer_city']]
    
    dt_customers_and_orders = pd.merge(dt_order_per_customer, dt_customer_per_city, on="customer_id")
    cities_per_order = dt_customer_per_city \
        .groupby(['customer_city']) \
        .size() \
        .reset_index(name='quantidade') \
        .sort_values(['quantidade'], ascending=False)
    top_10 = cities_per_order.head(10)

    plt.figure(figsize=(10, 6))
    top_10.plot(kind='bar')
    plt.xlabel('Cidades')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=45)

    # st.title("As dez cidades com mais entregas")
    # st.pyplot(plt)