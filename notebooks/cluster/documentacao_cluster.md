# Documentação do processo de clusterização

## Pipeline

### Fluxo principal
1. feature_engineering.ipynb
2. clustering-category_seasonal_data.ipynb
3. eda_cleaning_cluster.ipynb
4. eda_cluster_dataframe.ipynb
### Funções suporte
1. cluster_tools.py
2. eda_tools.py

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