import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv('./data/outputs/classification_reports_clean.csv')

df = load_data()

st.title('Comparação de Modelos de Classificação')

models = df['Model'].unique()
selected_models = st.multiselect('Selecione os Modelos para Comparar', options=models, default=models)

filtered_df = df[df['Model'].isin(selected_models)]

metrics = [
    'Accuracy', 
    'Macro Avg Precision', 
    'Macro Avg Recall', 
    'Macro Avg F1-Score', 
    'Weighted Avg Precision', 
    'Weighted Avg Recall', 
    'Weighted Avg F1-Score'
]

metrics_df = filtered_df.melt(id_vars=['Model'], value_vars=metrics, 
                              var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df, palette="viridis")
plt.title('Métricas Agregadas dos Modelos Selecionados')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(plt)

st.subheader('Detalhes dos Modelos Selecionados')

for model_name in selected_models:
    model_info = filtered_df[filtered_df['Model'] == model_name]
    
    with st.container():
        st.write(f"### **{model_name}**")
        with st.expander(f"Detalhes de {model_name}", expanded=True):
            st.write(f"**Modelo:** {model_name}")
            for col in filtered_df.columns:
                if col != 'Model':
                    st.write(f"**{col}:** {model_info[col].values[0]}")
        
        
