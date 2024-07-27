import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
orders_reviews.rename(columns={'review_score': 'Nota'}, inplace=True)

# Notas
st.title('Notas')

# gráfico de distribuição das notas por avaliação
st.subheader('Distribuição das notas por avaliação')
plt.figure(figsize=(12, 6))
sns.countplot(data=orders_reviews, x='Nota')
plt.title('Notas x Número de Avaliações')
st.pyplot(plt)

orders_reviews['delivery_diff_days'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_estimated_delivery_date']).dt.days
orders_reviews['delivery_status'] = orders_reviews['delivery_diff_days'].apply(lambda x: 'No Tempo' if x <= 0 else 'Atrasada')
orders_reviews['estimated_to_delivered_diff'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_estimated_delivery_date']).dt.days

# gráfico de dispersão entre a diferença de entrega estimada e real por nota
st.subheader('Dispersão entre notas atribuídas e em quantos dias a entrega se adiantou ou atrasou, em relação a data prevista')
plt.figure(figsize=(12, 6))
sns.scatterplot(data=orders_reviews, x='estimated_to_delivered_diff', y='Nota', hue='Nota', palette='viridis')
plt.title('Estimativa de entregas x Nota atribuída')
plt.xlabel('Diferença em Dias (Entregas Estimada - Real)')
plt.ylabel('Nota da Avaliação')
st.pyplot(plt)

# Comentários
st.title('Comentários')
st.subheader('Número de comentários a partir da Nota atribuída')
st.write(orders_reviews.groupby('Nota').agg({
    'review_comment_message': ['count']
}).reset_index())

st.subheader('Número de comentários para cada status de entrega')
comment_count_by_status = orders_reviews.groupby('delivery_status')['review_comment_message'].count().reset_index()
comment_count_by_status.columns = ['Status de Entrega', 'Quantidade de Comentários']
st.write(comment_count_by_status)

# exemplos de comentários
st.title("Comentários exemplificados:")

st.subheader('Exemplos de comentários para entregas feitas em tempo hábil')
st.write(orders_reviews[orders_reviews['delivery_status'] == 'No Tempo']['review_comment_message'].dropna().sample(10).tolist())

st.subheader("Exemplos de comentários para entregas feitas com atraso/não entregues:")
st.write(orders_reviews[orders_reviews['delivery_status'] == 'Atrasada']['review_comment_message'].dropna().sample(10).tolist())

#Análise de Sentimento
st.title('Análise de Sentimento')
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(comment):
    if pd.isna(comment):
        return None
    score = analyzer.polarity_scores(comment)
    if score['compound'] >= 0.1:
        return 'Positivo'
    elif score['compound'] <= -0.1:
        return 'Negativo'
    else:
        return 'Neutro'

orders_reviews['Sentimento'] = orders_reviews['review_comment_message'].apply(analyze_sentiment)

# porcentagem de sentimentos
sentiment_distribution = orders_reviews['Sentimento'].value_counts(normalize=True) * 100
st.subheader('Distribuição dos sentimentos dos comentários pré-filtro de palavras')
st.write(sentiment_distribution)

# plot da distribuição dos escores 'compound'
st.text('A nota é uma atribuída a uma medida compostade sentimento calculada pelo VADER (biblioteca), chamada compound.')
st.text('Ele é uma média ponderada das valências dos termos encontrados em um texto, normalizada para um valor que varia de -1 (extremamente negativo) a +1 (extremamente positivo)')
plt.figure(figsize=(12, 6))
sns.histplot(orders_reviews['review_comment_message'].dropna().apply(lambda x: analyzer.polarity_scores(x)['compound']), bins=30, kde=True)
plt.title('Distribuição das notas/compound dos comentários')
plt.xlabel('Score Compound')
plt.ylabel('Frequência')
st.pyplot(plt)