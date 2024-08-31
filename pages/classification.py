import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração do estilo dos gráficos
sns.set_theme(style="whitegrid")

# Carregar os dados do arquivo CSV
@st.cache_data
def load_data():
    return pd.read_csv('./data/outputs/classification_reports.csv')

df = load_data()

# Página do Streamlit
st.title('Comparação de Modelos de Classificação')

# Filtro para seleção dos modelos
models = df['Model'].unique()
selected_models = st.multiselect('Selecione os Modelos para Comparar', options=models, default=models)

# Filtrar os dados com base na seleção
filtered_df = df[df['Model'].isin(selected_models)]

# Prepare o DataFrame para o gráfico
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

# Criar o gráfico
plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df, palette="viridis")
plt.title('Métricas Agregadas dos Modelos Selecionados')
plt.xticks(rotation=45)
plt.tight_layout()

# Exibir o gráfico no Streamlit
st.pyplot(plt)

# Exibir informações detalhadas de cada modelo
st.subheader('Detalhes dos Modelos Selecionados')

# Criar uma coluna para exibir detalhes dos modelos
col1, col2 = st.columns(2)

with col1:
    st.write("### Informações Detalhadas dos Modelos")
    for model_name in selected_models:
        st.write(f"**{model_name}:**")
        model_info = filtered_df[filtered_df['Model'] == model_name]
        st.write(model_info.set_index('Model').T)  # Transpose for better readability

with col2:
    st.write("### Tabelas de Métricas")
    for model_name in selected_models:
        st.write(f"**Métricas do {model_name}:**")
        model_metrics = filtered_df[filtered_df['Model'] == model_name][metrics]
        st.write(model_metrics.set_index(filtered_df.columns.difference(metrics).tolist()).T)  # Transpose for better readability