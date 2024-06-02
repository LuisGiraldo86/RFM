import os
import sys
import json

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from plotly.colors import n_colors



from rfm_clustering.helpers import fetch_web_data, load_dataset
from rfm_clustering.rfm_sklearn import Recency, Spending, Identity, raw_rfm, QuartileEncoder, RFMcalculator

# get current working directory
path_to_wd = os.getcwd()

# append path to local modules for streamlit use
sys.path.append(
    os.path.join(path_to_wd, 'rfm_clustering')
)

def compute_rfm(df:pd.DataFrame)->pd.DataFrame:

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    recency = Pipeline([
    ('recency', Recency(date_to_compare=pd.Timestamp(2013,1,10)).set_output(transform='pandas'))
    ])
    spending = Pipeline([
        ('spending', Spending().set_output(transform='pandas'))
    ])
    identity = Pipeline([
        ('identity', Identity().set_output(transform='pandas'))
    ])

    rfm_trans = ColumnTransformer([
        ('recency', recency, ['Customer ID', 'InvoiceDate']),
        ('spending', spending, ['Quantity', 'Price']),
        ('identity', identity, ['Customer ID', 'Invoice'])
    ],
    remainder='drop').set_output(transform='pandas')

    df_1 = rfm_trans.fit_transform(df)

    del df

    df_2 = raw_rfm(df_1)

    del df_1

    encoder = Pipeline([
        ('encoder', QuartileEncoder().set_output(transform='pandas'))
    ])

    encoder_trans = ColumnTransformer([
        ('quantiles', encoder, ['frequency', 'recency', 'monetary']),
        ('identity1', identity, ['Customer ID'])
    ],
    remainder='passthrough').set_output(transform='pandas')

    df_3 = encoder_trans.fit_transform(df_2)

    del df_2

    rfm_pipe = Pipeline([
        ('rfm_pipe', RFMcalculator().set_output(transform='pandas'))
    ])
    rfm_value = ColumnTransformer([
        ('rfm_calc', rfm_pipe, ['quantiles__frequency_enc', 'quantiles__recency_enc', 'quantiles__monetary_enc']),
        ('identity1', identity, ['identity1__Customer ID'])
    ],
    remainder='drop').set_output(transform='pandas')

    df_4 = rfm_value.fit_transform(df_3)

    del df_3

    df_4.columns = [col.split('__')[-1] for col in df_4.columns]

    return df_4

def execute_main():

    WEB_PATH = 'https://raw.githubusercontent.com/LuisGiraldo86/auxiliarData/main/RFM/retails.tgz'
    FOLDER_PATH = os.path.join(path_to_wd, 'data')

    fetch_web_data(WEB_PATH, FOLDER_PATH)

    df = load_dataset(FOLDER_PATH)

    df_rfm = compute_rfm(df=df)


    pass

if __name__=="__main__":

    execute_main()