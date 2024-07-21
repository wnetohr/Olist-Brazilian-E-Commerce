import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análise de Sentimento e Satisfação do Consumidor", layout="wide")

def load_data(file_path):
    return pd.read_parquet(file_path)

products_file = 'data/olist_products_dataset.parquet'
orders_file = 'data/olist_orders_dataset.parquet'
reviews_file = 'data/olist_order_reviews_dataset.parquet'

products = load_data(products_file)
orders = load_data(orders_file)
reviews = load_data(reviews_file)

orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

orders_reviews = orders.merge(reviews, on='order_id', how='inner')

st.title('Visualização de Dados')

# Criar figura com gráficos empilhados
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Gráfico de distribuição das notas
sns.countplot(data=orders_reviews, x='review_score', ax=ax[0])
ax[0].set_title('Distribuição das Notas das Avaliações')

# Calcular a diferença entre a data de entrega estimada e a data de entrega real
orders_reviews['delivery_diff_days'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_estimated_delivery_date']).dt.days

# Classificar a entrega como 'No Tempo' ou 'Atrasada'
orders_reviews['delivery_status'] = orders_reviews['delivery_diff_days'].apply(lambda x: 'No Tempo' if x <= 0 else 'Atrasada')

# Calcular a diferença entre as datas para os eixos
orders_reviews['estimated_to_delivered_diff'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_estimated_delivery_date']).dt.days

# Gráfico de dispersão entre a diferença de entrega estimada e real
sns.scatterplot(data=orders_reviews, x='estimated_to_delivered_diff', y='review_score', hue='review_score', palette='viridis', ax=ax[1])
ax[1].set_title('Dispersão entre Diferença de Datas de Entrega e Nota')
ax[1].set_xlabel('Diferença em Dias (Entregas Estimada - Real)')
ax[1].set_ylabel('Nota da Avaliação')

# Ajustar o layout para adicionar espaçamento
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Ajuste o valor para aumentar ou diminuir o espaçamento

st.pyplot(fig)

# Análise de Comentários com Base na Pontualidade da Entrega
st.title('Análise de Sentimento')

# Contagem de comentários para cada status de entrega
comment_count_by_status = orders_reviews.groupby('delivery_status')['review_comment_message'].count().reset_index()
comment_count_by_status.columns = ['Status de Entrega', 'Quantidade de Comentários']

st.write(comment_count_by_status)

# Exemplo de comentários
st.write("Exemplos de Comentários para Entregas no Tempo:")
st.write(orders_reviews[orders_reviews['delivery_status'] == 'No Tempo']['review_comment_message'].dropna().sample(5).tolist())

st.write("Exemplos de Comentários para Entregas Atrasadas:")
st.write(orders_reviews[orders_reviews['delivery_status'] == 'Atrasada']['review_comment_message'].dropna().sample(5).tolist())

# Tabela de resumo
st.title('Número de Comentários x Avaliação')
st.write(orders_reviews.groupby('review_score').agg({
    'review_comment_message': ['count']
}).reset_index())