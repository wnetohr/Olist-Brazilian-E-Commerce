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

st.title('🔎 Classificação')
st.subheader('Comparação de Modelos de Classificação')
st.markdown("""Esta página foi criada para fornecer uma visão detalhada do desempenho de vários modelos de classificação treinados em diferentes conjuntos de dados. 
            Aqui, você pode comparar as principais métricas de cada modelo, explorar as características que mais influenciam as previsões, e analisar as matrizes
             de confusão para entender melhor os erros de classificação. Vale ressaltar que agrupamos as notas entre alguns ''review_score_factor'', e o que se comportou de forma mais realista foi separá-las em 3 grupos.
            Eles são o grupo 0 (notas 1), grupo 1 (notas 2, 3 e 4) e o grupo 2 (notas 5).""")

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
st.markdown("""Aqui você pode visualizar quais características foram mais importantes para os modelos selecionados ao tomar decisões de classificação. Compreender essas importâncias pode ajudar a identificar quais fatores influenciam mais as previsões.""")
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
st.markdown("""Aqui exibimos as 4 matrizes de confusão que pareceram exibir os erros de classificação da melhor forma. 
            Analisando as matrizes, podemos identificar alguns padrões de erro e áreas onde os modelos podem ser aprimorados. 
            Majoritariamente essas áreas seriam as ''Neutras'', visto que os extremos ''Negativos'' e ''Positivos'', os modelos conseguem determinar bem.
            """)

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
                    
                    # Extraindo e mapeando os rótulos das classes
                    class_labels = [class_mapping.get(int(col.split(' ')[-1]), col) for col in matrix_data.columns]
                    y_labels = [class_mapping.get(int(idx.split(' ')[-1]), idx) for idx in matrix_data.index]
                    
                    with cols[col]:
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(matrix_data, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=class_labels,
                                    yticklabels=y_labels)
                        plt.title(f'Matriz de Confusão para {model}')
                        plt.xlabel('Classe Prevista')
                        plt.ylabel('Classe Real')
                        plt.tight_layout()
                        st.pyplot(plt)
                        plt.clf()  # Limpa a figura após exibição
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
    Este gráfico mostra a importância das features mais relevantes para o modelo Random Forest. Cada barra representa a contribuição média absoluta de uma feature em relação à decisão final do modelo.

    - **payment_value**: Indica o valor pago pelo cliente.
    - **payment_installments**: Refere-se ao número de parcelas utilizadas para o pagamento.
    - **product_name_length**: Refere-se ao comprimento do nome do produto.

    Essas features são as mais influentes para o modelo na predição das avaliações/satisfação dadas pelos clientes, de acordo com o SHAP. Ou seja, podemos inferir que:
    """)

    st.write(""" O comprimento do nome do produto pode estar influenciando como o modelo faz as previsões, potencialmente indicando que produtos com nomes mais longos ou curtos estão associados a certos comportamentos ou padrões de compra. """)
    st.write(""" A quantidade de parcelas e o valor do pagamento são relevantes, sugerindo que o modelo está capturando como diferentes padrões de pagamento influenciam as avaliações ou decisões dos clientes. """)

st.subheader('Conclusão e Próximos Passos')
st.markdown("""Com as informações e visualizações fornecidas, agora temos uma base sólida para comparar o desempenho dos modelos e identificar oportunidades de melhoria, tendo em vista possíveis ajustes nos dados de entrada ou parâmetros do modelo, para impactar nos resultados.
""")
