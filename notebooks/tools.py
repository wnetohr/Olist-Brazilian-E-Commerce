import pandas as pd
import numpy as np

def visualize_data(dataFrame):
    data = pd.DataFrame({
        'Fetaure': dataFrame.columns.values,
        'Tipo': dataFrame.dtypes.values,
        'Nulos (%)': dataFrame.isna().mean().values * 100,
        'Negativos (%)': [len(dataFrame[col][dataFrame[col] < 0]) / len(dataFrame) * 100 if col in dataFrame.select_dtypes(include=[np.number]).columns else 0 for col in dataFrame.columns],
        'Zeros (%)': [len(dataFrame[col][dataFrame[col] == 0]) / len(dataFrame) * 100 if col in dataFrame.select_dtypes(include=[np.number]).columns else 0 for col in dataFrame.columns], 
        'Duplicados': dataFrame.duplicated().sum(),
        'Unicos': dataFrame.nunique().values,
        'Valores unicos': [dataFrame[col].unique() for col in dataFrame.columns]
    })
    
    return data.round(2)