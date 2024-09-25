# Documentação do processo de clusterização

## Pipeline para obtenção de clusters e EDA

### Fluxo principal
1. feature_engineering.ipynb
2. clustering-category_seasonal_data.ipynb
3. eda_cleaning_cluster.ipynb
4. eda_cluster_dataframe.ipynb
### Funções suporte
1. cluster_tools.py
2. eda_tools.py

## Relação de feriados
* Feriados comerciais: 
    * Black Friday;
    * Reveillon.
* Feriados religiosos:
    * Natal;
    * Páscoa.
* Feriados familiares:
    * Dia das crianças;
    * Dia da mulher;
    * Dia das mães
    * Dia dos pais;
    * Dia dos namorados.

## Dicionário dos dados

### eda_clusters.parquet
* comdate_diff: Diferença entre o dia da venda e a data comercial mais próxima
    * Tipo: int64 - número inteiro  
    * Exemplo: 11
* price: Valor da venda
    * Tipo: float64 - decimal padrão americano
    * Exemplo: 289.00
* freight_value: Frete do envio do produto vendido
    * Tipo: float64 - decimal padrão americano
    * Exemplo: 46.48
* commercial_dates_day_since_year_start: Dia do ano do feriado mais próximo da venda, começando em 0
    * Tipo: int64 - número natural
    * Exemplo: 0
* commercial_dates_year: Ano do feriado mais próximo da venda
    * Tipo: int64 - número natural
    * Exemplo: 2018
* commercial_dates_month: Mês do ano do feriado mais próximo da venda, começando em 1
    * Tipo: int64 - número natural
    * Exemplo: 1
* commercial_dates_day: Dia do mês do feriado mais próximo da venda, começando em 1
    * Tipo: int64 - número natural
    * Exemplo: 1
* order_purchase_day_since_year_start: Dia do ano da venda, começando em 0
    * Tipo: int64 - número natural
    * Exemplo: 11
* order_purchase_year: Ano da venda
    * Tipo: int64 - número natural
    * Exemplo: 2018
* order_purchase_month: Mês da venda, começando em 1
    * Tipo: int64 - número natural
    * Exemplo: 1
* order_purchase_day: Dia do mês da venda, começando em 1
    * Tipo: int64 - número natural
    * Exemplo: 12
* hue: nome para o cluster
    * Tipo: object - string
    * Exemplo: cluster_0
* filtered_category: Categoria do produto vendido
    * Tipo: object - string
    * Exemplo: Utilidades domésticas
* commercial_date: Feriado comercial
    * Tipo: object - string
    * Exemplo: Não se aplica
* time_window_order: Janela de tempo entre a venda e o feriado comercial
    * Tipo: object - string
    * Exemplo: Mais de duas semanas
<!-- 
* commercial_date_coded: Feriado comercial, codificado
    * Tipo: int32 - número natural
    * Exemplo: 7
* time_window_order_coded: Janela de tempo entre a venda e o feriado comercial, codificado
    * Tipo: int32 - número natural
    * Exemplo: 1
* filtered_category_coded: Categoria do produto vendido, codificado 
    * Tipo: int32 - número natural 
    * Exemplo: 13
* cluster:  identificador do cluster
    * Tipo: int32 - número natural 
    * Exemplo: 0 
* commercial_date_seazonal_weight: Percentual das vendas deste feriado em relação a todos os feriados
    * Tipo: float64 - decimal padrão americano
    * Exemplo: 0.000000
* sensitivity: Percentual das vendas por feriado para cada categoria de produto
    * Tipo: float64 - decimal padrão americano
    * Exemplo: 0.000000
* mean_price_by_commercial_date: Média do valor das vendas por feriado para cada categoria de produto
    * Tipo: float64 - decimal padrão americano 
    * Exemplo: 92.179011
* std_price_by_commercial_date: Desvio padrão do valor das vendas por feriado para cada categoria de produto
    * Tipo: float64 - decimal padrão americano
    * Exemplo: 169.492277
-->

## Objetivo
Como as variações sazonais afetam o desempenho das vendas em certos grupos de produtos no comércio eletrônico brasileiro?
* A pergunta busca entender como a demanda sazonal afeta as vendas de grupos de produtos no e-commerce brasileiro. Compreender as flutuações do mercado em diferentes períodos permite prever aumentos e quedas no fluxo de compras, além de mostrar a sensibilidade de certos produtos a essas variações sazonais.


### Objetivo específico
Analisar os padrões sazonais de compra dos consumidores brasileiros, identificando picos de demanda em diferentes datas comerciais do ano e agrupamentos de produtos mais afetados;

## Perguntas

### Qual a relação numérica das vendas por categorias entre os clusters?
* Obter insights

```Conclusão:```

### Qual a relação numérica das vendas por feriados entre os clusters?
* Obter insights

```Conclusão:```

### Dentre os feriados listados, qual a relação das vendas por categorias para cada feriado nos clusters
* Obter insights

```Conclusão:```

### Dentre os meses do ano, qual a relação das vendas por categorias para cada mês nos clusters
* Obter insights

```Conclusão:```

### Qual a relação do valor das vendas nos cluster?
* Preço médio* por venda nos clusters
* Preço por venda por categoria nos clusters
    * Relação de quais categorias possuem valor por venda acima da média* no cluster
* Preço por venda por feriado nos clusters
    * Relação de quais feriados possuem valor por venda acima da média* no cluster
* *Abordagem por média? mediana? desvio padrão?

```Conclusão:```

### Qual a relação do valor do frente das vendas nos cluster?
* Preço médio* por frete da venda nos clusters
* Preço por frete da venda por categoria nos clusters
    * Relação de quais categorias possuem valor por frete da venda acima da média* no cluster
* Preço por frete da venda por feriado nos clusters
    * Relação de quais feriados possuem valor por frete da venda acima da média* no cluster
* *Abordagem por média? mediana? desvio padrão?

```Conclusão:```

## Clusters
Descrição das caracteristicas de cada cluster. As informações obtidas pelo EDA estão postas nessa secção.

### Cluster 0: 
    
* Feriados: As vendas desse cluster são pouco vinculadas a feriados, possuindo um quantitativo de 27.808 vendas que não são vinculadas a nenhum feriado, 1.735 vendas vinculadas ao reveillon e 1.565 vendas vinculadas a páscoa. É possivel afirmar que esse cluster reflete primariamente um padrão de consumo não vinculado a feriados.

### Cluster 1:

* Feriados: As vendas desse cluster estão distruibuidas em 3 feriados e também há algumas vendas não vinculadas a nenhum feriados. Os feriados são black friday, com 7.055 vendas, dia das criançãs, com 4.320 vendas e natal, com 2.699. Além dos feriados há um quantitativo de 4.740 vendas não vinculadas a nenhum feriado. Ao considerar que esse cluster possui um quantitativo de vendas não vinculados a feriados e outros 3 quantitavos de feriados onde um é um feriado comercial, um religioso e um familiar, podendos afirmar que esse cluster possui um padrão de consumo distribuido entre os diversos tipos de feriados.
  
### Cluster 2:
    
* Feriados: As vendas desse cluster estão distribuidas nos seguintes feriados, dia dos pais, com 10.261, dia das mães, com 8.973, dia da mulher, com 8.118, dia dos namorados, com 6.504 e páscoa, 4.165. Ao observar a distribuição do quantitativo de vendas desse cluster por feriados, é possivel afirmar que o padrão de consumo desse cluster está voltado a feriados familiares.

