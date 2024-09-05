import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

@st.cache_data
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Formato de arquivo não suportado. Use '.csv' ou '.parquet'.")

# Carregar o arquivo de resultados
df = load_data('./data/outputs/results.parquet')

st.title('Comparação de Modelos de Classificação')

# Selecionar modelos e conjunto de dados
models = df['Model'].unique()
selected_models = st.multiselect('Selecione os Modelos para Comparar', options=models, default=models)

set_options = df['Set'].unique()
selected_set = st.selectbox('Selecione o Conjunto de Dados', options=set_options, index=0)

filtered_df = df[(df['Model'].isin(selected_models)) & (df['Set'] == selected_set)]

# Mostrar métricas dos modelos
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
metrics_df = filtered_df.melt(id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df, palette="viridis")
plt.title(f'Métricas dos Modelos Selecionados ({selected_set})')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

# Mostrar detalhes dos modelos
st.subheader('Detalhes dos Modelos Selecionados')
st.table(filtered_df)

# Carregar importâncias das características e matrizes de confusão
feature_importances_path = './data/outputs/all_feature_importances.csv'
confusion_matrices_path = './data/outputs/new_all_confusion_matrices.csv'

feature_importances_df = load_data(feature_importances_path)
confusion_matrices_df = load_data(confusion_matrices_path)

# Selecione o modelo para visualização
st.title('Comparação de Importâncias e Matrizes de Confusão dos Modelos')

# Atualize a lista de modelos a partir do DataFrame de importâncias
models = feature_importances_df['Model'].unique()
selected_models = st.multiselect('Selecione os Modelos para Comparar', options=models, default=models)

# Filtro para Importâncias das Características
st.subheader('Importâncias das Características')

filtered_importances_df = feature_importances_df[feature_importances_df['Model'].isin(selected_models)]

if not filtered_importances_df.empty:
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Importance', y='Feature', hue='Model', data=filtered_importances_df, palette="viridis")
    plt.title('Importâncias das Características dos Modelos Selecionados')
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.warning('Nenhuma importância de característica encontrada para os modelos selecionados.')

# Filtro para Matrizes de Confusão
st.subheader('Matrizes de Confusão')

filtered_confusion_matrices_df = confusion_matrices_df[confusion_matrices_df['Model'].isin(selected_models)]

if not filtered_confusion_matrices_df.empty:
    num_models = len(selected_models)
    num_rows = (num_models + 1) // 2  # Calcula o número de linhas necessárias
    
    # Dividir a exibição em linhas e colunas
    for row in range(num_rows):
        cols = st.columns(2)  # Criar duas colunas para cada linha
        for col in range(2):
            index = row * 2 + col
            if index < num_models:
                model = selected_models[index]
                matrix = filtered_confusion_matrices_df[filtered_confusion_matrices_df['Model'] == model]
                
                if not matrix.empty:
                    # Configurar a matriz de confusão
                    matrix_data = matrix.pivot(index='Actual', columns='Class', values='Count').fillna(0)
                    class_labels = matrix_data.columns  # Labels das classes
                    
                    with cols[col]:
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=class_labels,
                                    yticklabels=class_labels)
                        plt.title(f'Matriz de Confusão para {model}')
                        plt.xlabel('Classe Prevista')
                        plt.ylabel('Classe Real')
                        plt.tight_layout()
                        st.pyplot(plt)
else:
    st.warning('Nenhuma matriz de confusão encontrada para os modelos selecionados.')
