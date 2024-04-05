import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

dados = pd.read_csv('./data/olist_products_dataset.csv')

categoria = dados['product_category_name'].value_counts()

st.title("As dez categorias de produtos mais vendidas")

plt.figure(figsize=(10, 6))
top_10_df = categoria.head(10)
top_10_df.plot(kind='bar')

plt.title('Quantidade X Categoria')
plt.xlabel('Categoria')
plt.ylabel('Quantidade')
plt.grid(True)
plt.xticks(rotation=45)

st.pyplot(plt)