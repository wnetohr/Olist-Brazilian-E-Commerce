import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    return pd.read_parquet('./data/outputs/results.parquet')

df = load_data()

st.title('Comparação de Modelos de Classificação')

models = df['Model'].unique()
selected_models = st.multiselect('Selecione os Modelos para Comparar', options=models, default=models)

# Adicionar seleção para treino ou teste
set_options = df['Set'].unique()
selected_set = st.selectbox('Selecione o Conjunto de Dados', options=set_options, index=0)

filtered_df = df[(df['Model'].isin(selected_models)) & (df['Set'] == selected_set)]

metrics = [
    'Accuracy', 
    'Precision', 
    'Recall', 
    'F1'
]

metrics_df = filtered_df.melt(id_vars=['Model'], value_vars=metrics, 
                              var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df, palette="viridis")
plt.title(f'Métricas dos Modelos Selecionados ({selected_set})')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(plt)

st.subheader('Detalhes dos Modelos Selecionados')
st.table(filtered_df)
