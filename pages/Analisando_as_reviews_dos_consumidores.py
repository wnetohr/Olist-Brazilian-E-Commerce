import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings 

#Funções

def load_data(file_path):
    return pd.read_parquet(file_path)

file_path = './data/olist_order_reviews_dataset.parquet'
df = load_data(file_path)

def skimming_data(data):
    skimmed_data = pd.DataFrame({
        'feature': data.columns.values,  
        'data_type': data.dtypes.values,  
        'null_value(%)': data.isna().mean().values * 100,  
        'neg_value(%)': [len(data[col][data[col] < 0]) / len(data) * 100 if col in data.select_dtypes(include=[np.number]).columns else 0 for col in data.columns], 
        '0_value(%)': [len(data[col][data[col] == 0]) / len(data) * 100 if col in data.select_dtypes(include=[np.number]).columns else 0 for col in data.columns], 
        'duplicate': data.duplicated().sum(),  
        'n_unique': data.nunique().values, 
        #'sample_unique': [', '.join(map(str, data[col].unique())) for col in data.columns] 
    })
    
    return skimmed_data.round(3)


#Main

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
df_resumo = skimming_data(df)

st.title('Overview da tabela de reviews dos consumidores')
st.dataframe(df_resumo)

st.title('Reviews dos consumidores')
st.dataframe(df)

st.dataframe(df.count())