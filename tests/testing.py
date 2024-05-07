
from rfm_clustering.helpers import fetch_web_data, load_dataset
from rfm_clustering.rfm_sklearn import Recency, Spending, Identity

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#WEB_PATH = 'https://raw.githubusercontent.com/LuisGiraldo86/auxiliarData/main/RFM/retails.tgz'
FOLDER_PATH = '/mnt/287A29DF7A29AA90/PythonProjects/auxiliarData/RFM'

#fetch_web_data(WEB_PATH, FOLDER_PATH)

df = load_dataset(FOLDER_PATH)
df

rfm_pipe = Pipeline([
    ('recency', Recency(date_to_compare=pd.Timestamp(2013,1,10)).set_output(transform='pandas'))
])

rfm_trans = ColumnTransformer([
    ('rfm', rfm_pipe, ['Customer ID', 'InvoiceDate'])
],
remainder='drop').set_output(transform='pandas')

rfm_trans.fit_transform(df)

