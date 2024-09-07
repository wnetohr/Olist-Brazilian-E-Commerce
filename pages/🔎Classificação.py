import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

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

    # Mapeamento das classes
    class_mapping = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}
    
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
                    
                    # Mapeamento para o eixo Y (classes reais)
                    y_labels = [class_mapping.get(int(idx.split(' ')[-1]), idx) for idx in matrix_data.index]
                    
                    with cols[col]:
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=matrix_data.columns,  # Manter rótulos originais no eixo X
                                    yticklabels=y_labels)  # Aplicar mapeamento ao eixo Y
                        plt.title(f'Matriz de Confusão para {model}')
                        plt.xlabel('Classe Prevista')
                        plt.ylabel('Classe Real')
                        plt.tight_layout()
                        st.pyplot(plt)
else:
    st.warning('Nenhuma matriz de confusão encontrada para os modelos selecionados.')

st.subheader('SHAP para o Random Forest')
# Cria duas colunas
col1, col2 = st.columns([2, 1])  # Ajuste os números para definir a largura das colunas

# Exibe a imagem na primeira coluna
with col1:
    image = Image.open('data/outputs/SHAP_RF_small.png')
    st.image(image, caption='Gráfico SHAP - Importância das Features')

# Adiciona o texto na segunda coluna
with col2:
    st.write("""
    ### Interpretação do Gráfico SHAP para o Random Forest
    Este gráfico mostra a importância das features para o modelo Random Forest. Cada barra representa a contribuição média absoluta de uma feature em relação à decisão final do modelo.

    - **payment_value**: Indica o valor pago pelo cliente.
    - **payment_installments**: Refere-se ao número de parcelas utilizadas para o pagamento.
    - **product_name_length**: Refere-se ao comprimento do nome do produto.

    Essas features são as mais influentes para o modelo na predição das avaliações/satisfação dadas pelos clientes, de acordo com o SHAP. Ou seja, podemos inferir que:
    """)

    st.write(""" O comprimento do nome do produto pode estar influenciando como o modelo faz as previsões, potencialmente indicando que produtos com nomes mais longos ou curtos estão associados a certos comportamentos ou padrões de compra. """)
    st.write(""" A quantidade de parcelas e o valor do pagamento são relevantes, sugerindo que o modelo está capturando como diferentes padrões de pagamento influenciam as avaliações ou decisões dos clientes. """)