import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def load_data(file_path):
    return pd.read_parquet(file_path)

def show_topics(dataframe, num_topics=5, num_top_words=10):
    # Pré-processamento com TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=None,max_features=1000,tokenizer=lambda text: text.split())
    tfidf = tfidf_vectorizer.fit_transform(dataframe['review_comment_message'].values.astype('U'))
    
    # Aplicação do modelo LDA
    lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online', random_state=0)
    lda_output = lda_model.fit_transform(tfidf)
    
    # Obtendo as palavras-chave para cada tópico
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topics.append({
            "Tópico": topic_idx + 1,
            "Palavras-chave": [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        })
    
    return topics

if __name__ == "__main__":
    df = pd.DataFrame(load_data("./data/olist_order_reviews_dataset.parquet"))

    num_topics = 3  

    st.title('Análise de Tópicos em Comentários de Avaliação')
    st.write("Exemplo de análise de tópicos usando Latent Dirichlet Allocation (LDA)")

    topics = show_topics(df, num_topics=num_topics)
    for topic in topics:
        st.header(f"Tópico {topic['Tópico']}")
        st.write(", ".join(topic['Palavras-chave']))

