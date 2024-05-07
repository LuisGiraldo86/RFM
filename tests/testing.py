
from rfm_clustering.helpers import fetch_web_data, load_dataset
from rfm_clustering.rfm_sklearn import Recency, Spending, Identity

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#WEB_PATH = 'https://raw.githubusercontent.com/LuisGiraldo86/auxiliarData/main/RFM/retails.tgz'
FOLDER_PATH = '/mnt/287A29DF7A29AA90/PythonProjects/auxiliarData/RFM'

#fetch_web_data(WEB_PATH, FOLDER_PATH)

df = load_dataset(FOLDER_PATH)

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

df_2 = df_1.groupby(by=['identity__Customer ID', 'identity__Invoice'], as_index=False).agg({
    'recency__recency'  : 'min',
    'spending__spending': 'sum',
})
df_3 = df_2.groupby(by='identity__Customer ID', as_index=False).agg({
    'identity__Invoice': 'count',
    'recency__recency': 'min',
    'spending__spending': 'sum'
})
df_3.columns = ['customerID', 'frequency', 'recency', 'money']
df_3