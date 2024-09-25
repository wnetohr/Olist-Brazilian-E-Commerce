import streamlit as st

st.set_page_config(
    page_title="Brazilian E-Commerce",
    layout="wide",
    menu_items={
        'About': '''Projeto para a cadeira de projeto 3 dos professores 
        Ceça Maria e Gabriel Alves feito pelos alunos: Gabriel Reis, Matheus Costa, Roberto Leite e Washington Rocha.
        '''
    }
)

# Cria duas colunas para posicionar a imagem e o título lado a lado
col1, col2 = st.columns([1, 5])

# Adiciona a imagem na primeira coluna
with col1:
    st.image("./project_assets/olist_logo.png", width=100)  # Substitua "caminho_para_a_imagem.png" pelo caminho da sua imagem

# Adiciona o título na segunda coluna
with col2:
    st.markdown(f'''
        <h1>Olist Brazilian E-Commerce</h1>
    ''', unsafe_allow_html=True)

# Conteúdo da página
st.markdown(f'''
    <br>
    Dado o e-commerce brasileiro, a nossa pesquisa tem como objetivo entender os padrões de comportamento do 
    consumidor por meio da análise abrangente do Dataset público do Olist. Com um dataset que possui
    mais de 100 mil pedidos realizados entre 2016 e 2018 em diversos marketplaces do Brasil,
    o nosso estudo se propõe a analisar uma série de desafios enfrentados pelo setor. Ao identificar padrões de consumo, a satisfação dos clientes, fatores
    sazonais, e entre outros mecanismos de avaliação e consumo, o estudo busca fornecer uma compreensão 
    mais profunda do impacto socioeconômico do e-commerce no Brasil. Para atingirmos estes objetivos,
    levamos em consideração os seguintes questionamentos que foram levantados, mtendo em vista como o aprendizado de máquina e a análise de dados poderiam nos auxiliar,
    que seriam:
    <br>
    <h2>Perguntas</h2>
    <ul>
        <li>Como as variações sazonais afetam o desempenho das vendas em certos grupos de produtos no comércio eletrônico brasileiro?</li>
        <li>Observando as avaliações feitas pelos consumidores em suas compras, quais são os fatores que mais influenciam na satisfação dos consumidores?</li>
    </ul>
    <h2>Alunos</h2>
    <ul>
        <li>Gabriel Reis</li>
        <li>Matheus Costa</li>
        <li>Roberto Leite</li>
        <li>Washington Rocha</li>
    </ul>
''', unsafe_allow_html=True)