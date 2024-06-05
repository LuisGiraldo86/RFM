
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from rfm_clustering.rfm_sklearn import Identity

def rfm_by_operation(raw_sales:pd.DataFrame, reference_date:pd.Timestamp)->pd.DataFrame:

    from rfm_clustering.rfm_sklearn import Recency, Spending

    recency = Pipeline([
    ('recency', Recency(date_to_compare=reference_date).set_output(transform='pandas'))
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

    return rfm_trans.fit_transform(raw_sales)

def raw_rfm(data:pd.DataFrame)->pd.DataFrame:

    df = data.groupby(by='identity__Customer ID', as_index=False).agg(
        frequency= pd.NamedAgg(column='identity__Invoice', aggfunc='nunique'),
        recency  = pd.NamedAgg(column='recency__recency', aggfunc='min'),
        monetary = pd.NamedAgg(column='spending__spending', aggfunc='sum')
    )

    df.columns = [col.split('__')[-1] if '__' in col else col for col in df.columns]

    return df

def encoder_by_quartiles(data:pd.DataFrame)->pd.DataFrame:

    from rfm_clustering.rfm_sklearn import QuartileEncoder

    encoder = Pipeline([
        ('encoder', QuartileEncoder().set_output(transform='pandas'))
    ])
    identity = Pipeline([
        ('identity', Identity().set_output(transform='pandas'))
    ])

    encoder_trans = ColumnTransformer([
        ('quantiles', encoder, ['frequency', 'recency', 'monetary']),
        ('identity1', identity, ['Customer ID'])
    ],
    remainder='passthrough').set_output(transform='pandas')

    return encoder_trans.fit_transform(data)

def category_encoder(data:pd.DataFrame)->pd.DataFrame:

    from rfm_clustering.rfm_sklearn import RFMcalculator

    rfm_pipe = Pipeline([
        ('rfm_pipe', RFMcalculator().set_output(transform='pandas'))
    ])
    identity = Pipeline([
        ('identity', Identity().set_output(transform='pandas'))
    ])
    rfm_value = ColumnTransformer([
        ('rfm_calc', rfm_pipe, ['quantiles__frequency_enc', 'quantiles__recency_enc', 'quantiles__monetary_enc']),
        ('identity1', identity, ['identity1__Customer ID'])
    ],
    remainder='drop').set_output(transform='pandas')

    df = rfm_value.fit_transform(data)
    df.columns = [col.split('__')[-1] for col in df.columns]

    return df
