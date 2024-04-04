import streamlit as st

st.set_page_config(
    page_title = "Olist Brazilian E-Commerce",
    layout = "wide",
    menu_items = {
        'About': '''Projeto para a cadeira de projeto 3 dos professores 
        Ceça Maria e Gabriel Alves feito pelos alunos: Gabriel Reis, Matheus Costa, Roberto Leite e Washington Rocha.
        '''
    }
)
st.markdown(f'''
    <h1>Olist Brazilian E-Commerce</h1>
    <br>
    Dado o E-commerce brasileiro, a pesquisa tem como objetivo desvendar os padrões de comportamento do 
            consumidor e otimizar as operações logísticas por meio da análise abrangente do Dataset 
            público do Olist. Com um dataset que possui mais de 100 mil pedidos realizados entre 2016 
            e 2018 em diversos marketplaces do Brasil, o estudo se propõe a resolver uma série de desafios 
            enfrentados pelo setor.
    <br>
    <h2>Alunos</h2>
    <ul>
            <li>Gabriel Reis</li>
            <li>Matheus Costa</li>
            <li>Roberto Leite</li>
            <li>Washington Rocha</li>
    </ul>

''', unsafe_allow_html=True)