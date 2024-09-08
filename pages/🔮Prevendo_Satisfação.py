import streamlit as st
import gzip
import pickle
import io
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from LeIA import SentimentIntensityAnalyzer

# Definição da função identity
def identity(X):
    return X

# Definição da classe SentimentAnalyzer
class SentimentAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        compound_scores = []
        neg_scores = []
        pos_scores = []
        neu_scores = []

        for review in X:
            s = self.analyzer.polarity_scores(review)
            compound_scores.append(s['compound'])
            neg_scores.append(s['neg'])
            pos_scores.append(s['pos'])
            neu_scores.append(s['neu'])
        
        return pd.DataFrame({
            'comp_score': compound_scores,
            'neg': neg_scores,
            'pos': pos_scores,
            'neu': neu_scores
        })

def create_pipeline():
    features = FeatureUnion([
        ('sentiment', Pipeline([
            ('extract', FunctionTransformer(identity, validate=False)),
            ('analyzer', SentimentAnalyzer())
        ])),
        ('tfidf', TfidfVectorizer(max_features=1000))
    ])
    
    pipeline = Pipeline([
        ('features', features),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

# Função para descompactar o modelo
def decompress_model(compressed_path):
    with gzip.open(compressed_path, 'rb') as f:
        model_bytes = f.read()
    return pickle.load(io.BytesIO(model_bytes))

# Caminho para o arquivo compactado do modelo
model_path = './notebooks/model_rf.pkl.gz'

# Carregar o modelo descompactado
try:
    model_rf = decompress_model(model_path)
except FileNotFoundError:
    st.error("Arquivo do modelo não encontrado. Verifique o caminho.")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

st.title("Prevendo a satisfação do cliente baseado nos comentários")

review_text = st.text_input("Escreva o comentário da avaliação")

if st.button("Fazer Previsão"):
    try:
        prediction = model_rf.predict([review_text])[0]
        
        # Mapeie a nota para a categoria
        if 1 <= prediction <= 2:
            category = "Cliente Insatisfeito"
        elif 3 <= prediction <= 4:
            category = "Avaliação Neutra"
        elif prediction == 5:
            category = "Cliente Satisfeito"
        else:
            category = "Nota fora do intervalo esperado"
        
        st.markdown(f"## {category}")
    except Exception as e:
        st.error(f"Erro durante a previsão: {e}")
