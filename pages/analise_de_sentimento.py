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
st.title('Notas e Tempo de Envio')

# gráfico de distribuição das notas por avaliação
st.subheader('Distribuição das notas por avaliação')
plt.figure(figsize=(20,5))
sns.countplot(data=orders_reviews, x='Nota')
plt.title('Notas x Número de Avaliações')
st.pyplot(plt)

orders_reviews['delivery_diff_days'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_estimated_delivery_date']).dt.days
orders_reviews['delivery_status'] = orders_reviews['delivery_diff_days'].apply(lambda x: 'No Tempo' if x <= 0 else 'Atrasada')
orders_reviews['estimated_to_delivered_diff'] = (orders_reviews['order_delivered_customer_date'] - orders_reviews['order_estimated_delivery_date']).dt.days

# gráfico de distribuição do tempo de envio sem outliers
st.subheader('Distribuição do tempo de envio em dias (sem outliers)')
plt.figure(figsize=(20, 5), dpi=500)
test2 = orders_reviews['estimated_to_delivered_diff'][orders_reviews['estimated_to_delivered_diff'] > 60]
test = orders_reviews['estimated_to_delivered_diff'][orders_reviews['estimated_to_delivered_diff'] < 60]
sns.histplot(test, bins=15)
plt.title('Distribuição do Tempo de Envio em Dias (Sem Outliers)', fontsize=16, loc='left')
plt.xticks(fontsize=14)
plt.ylabel('')
plt.xlabel('days', color='lightgrey')
st.pyplot(plt)

st.text('Porém, têm algumas entregas que podem demorar de 60 a 100 dias, e isso demanda verificar os outliers')

# gráfico de distribuição do tempo de envio dos outliers
st.subheader('Distribuição do tempo de envio dos outliers')
plt.figure(figsize=(20, 5), dpi=500)
sns.histplot(test2, bins=15, color='red')
plt.title('Distribuição do Tempo de Envio dos Outliers', fontsize=16, loc='left')
plt.xticks(fontsize=14)
plt.ylabel('')
plt.xlabel('days', color='lightgrey')
st.pyplot(plt)

# gráfico de violinplot para a diferença de entrega estimada e real por nota
st.subheader('Distribuição da diferença de entrega estimada e real por nota')
plt.figure(figsize=(20, 8))
sns.violinplot(data=orders_reviews, x='Nota', y='estimated_to_delivered_diff', palette='viridis')
plt.title('Diferença de Entregas Estimada - Real x Nota atribuída', fontsize=16, loc='left')
plt.xlabel('Nota', fontsize=16)
plt.ylabel('Diferença em Dias (Entregas Estimada - Data Real)', fontsize=16)
st.pyplot(plt)


# Comentários
st.title('Comentários')
st.subheader('Número de comentários a partir da nota atribuída')
st.write(orders_reviews.groupby('Nota').agg({
    'review_comment_message': ['count']
}).reset_index())

st.subheader('Número de comentários para cada status de entrega')
delivery_comment_counts = orders_reviews.groupby('order_status')['review_comment_message'].agg(['count']).reset_index() 
delivery_comment_counts.columns = ['Status de Entrega', 'Número de Comentários']
st.write(delivery_comment_counts)

st.subheader('Comprimento dos Comentários')
orders_reviews['comment_length'] = orders_reviews['review_comment_message'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
plt.figure(figsize=(20, 5), dpi=500)
sns.histplot(orders_reviews['comment_length'], bins=30, kde=True)
plt.title('Distribuição do Comprimento dos Comentários', fontsize = 16)
plt.xlabel('Número de Palavras', fontsize = 14)
plt.ylabel('Frequência de Aparição', fontsize = 14)
st.pyplot(plt)


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
plt.figure(figsize=(20, 5), dpi=500)
sns.histplot(orders_reviews['review_comment_message'].dropna().apply(lambda x: analyzer.polarity_scores(x)['compound']), bins=30, kde=True)
plt.title('Distribuição das notas/compound dos comentários', fontsize = 16)
plt.xlabel('Score Compound', fontsize= 14)
plt.ylabel('Frequência', fontsize = 14)
st.pyplot(plt)

# Top 3 comentários mais positivos e negativos baseados nos scores 'compound'
#st.subheader('Top 3 Comentários mais Positivos e Negativos baseados no Score Compound')

# Ordenar os comentários pelos escores 'compound'
#orders_reviews['compound'] = orders_reviews['review_comment_message'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if pd.notna(x) else None)

#st.subheader('Comentários Mais Positivos:')
#top_positive_comments = orders_reviews.sort_values(by='compound', ascending=False).head(3)
#for i, row in top_positive_comments.iterrows():
#    st.write(f"{i+1}. Nota: {row['Nota']}, Score Compound: {row['compound']:.2f}")
#    st.write(row['review_comment_message'])
#    st.write('---')

#st.subheader('Comentários Mais Negativos:')
#top_negative_comments = orders_reviews.sort_values(by='compound', ascending=True).head(3)
#for i, row in top_negative_comments.iterrows():
#    st.write(f"{i+1}. Nota: {row['Nota']}, Score Compound: {row['compound']:.2f}")
#    st.write(row['review_comment_message'])
#    st.write('---')