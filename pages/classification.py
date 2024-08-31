import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração do estilo dos gráficos
sns.set_theme(style="whitegrid")

# Carregar os dados do arquivo CSV
@st.cache_data
def load_data():
    return pd.read_csv('./data/outputs/classification_reports_clean.csv')

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

# Exibir informações detalhadas de cada modelo em cards
st.subheader('Detalhes dos Modelos Selecionados')

for model_name in selected_models:
    model_info = filtered_df[filtered_df['Model'] == model_name]
    
    # Exibir cada modelo em um "card"
    with st.container():
        st.write(f"### **{model_name}**")
        
        # Adicionar um contêiner para informações do modelo
        with st.expander(f"Detalhes de {model_name}", expanded=True):
            st.write(f"**Modelo:** {model_name}")
            for col in filtered_df.columns:
                if col != 'Model':
                    st.write(f"**{col}:** {model_info[col].values[0]}")
        
       
        # # Adicionar gráfico de barras horizontais para as métricas do modelo
        # st.write(f"### Métricas de {model_name}")
        # model_metrics = filtered_df[filtered_df['Model'] == model_name][metrics]
        # fig, ax = plt.subplots(figsize=(10, 6))
        # model_metrics.plot(kind='barh', ax=ax, colormap="viridis")
        # ax.set_title(f'Métricas de {model_name}')
        # ax.set_xlabel('Score')
        # st.pyplot(fig)
        
