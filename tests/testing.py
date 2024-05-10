
from rfm_clustering.helpers import fetch_web_data, load_dataset
from rfm_clustering.rfm_sklearn import Recency, Spending, Identity, raw_rfm, QuartileEncoder

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