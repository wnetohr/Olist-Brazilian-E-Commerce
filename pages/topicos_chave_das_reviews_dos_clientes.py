import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

def load_data(file_path):
    return pd.read_parquet(file_path)

def preprocess_text(text):
    if text is None:
        return ''
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('portuguese'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text, language='portuguese')
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def get_top_keywords(texts, num_keywords=10):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    sum_words = X.sum(axis=0)
    word_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    return word_freq[:num_keywords]

def categorize_review(text):
        quality_keywords = ['qualidade', 'produto', 'excelente', 'bom', 'ruim', 'defeito']
        delivery_keywords = ['entrega', 'prazo', 'atraso', 'tempo', 'rápido', 'demorado']
        price_keywords = ['preço', 'custo', 'caro', 'barato', 'valor']
        service_keywords = ['atendimento', 'serviço', 'suporte', 'ajuda', 'resolvido']

        if any(keyword in text for keyword in quality_keywords):
            return 'Qualidade do Produto'
        elif any(keyword in text for keyword in delivery_keywords):
            return 'Entrega'
        elif any(keyword in text for keyword in price_keywords):
            return 'Preço'
        elif any(keyword in text for keyword in service_keywords):
            return 'Atendimento'
        else:
            return 'Outros'

st.title('Análise de Palavras-chave das Reviews')

file_path = './data/olist_order_reviews_dataset.parquet'
df = load_data(file_path)
download_nltk_resources()

if 'review_comment_message' in df.columns:
    df['review_comment_message'] = df['review_comment_message'].fillna('')
    df['processed_review'] = df['review_comment_message'].apply(preprocess_text)
    
    
    df['category'] = df['processed_review'].apply(categorize_review)
    
    selected_category = st.selectbox(
        'Escolha uma categoria para visualizar',
        df['category'].unique()
    )
    
    filtered_df = df[df['category'] == selected_category]

    st.write(f"Número de reviews na categoria '{selected_category}': {len(filtered_df)}")

    top_keywords = get_top_keywords(filtered_df['processed_review'], num_keywords=10)

    st.write(f"Principais palavras-chave na categoria '{selected_category}':")
    for word, freq in top_keywords:
        st.write(f"- {word}: {freq}")

else:
    st.error("O dataset deve conter a coluna 'review_comment_message'.")
