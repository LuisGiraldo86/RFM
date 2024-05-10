
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin

def raw_rfm(data:pd.DataFrame)->pd.DataFrame:

    df = data.groupby(by='identity__Customer ID', as_index=False).agg(
        frequency= pd.NamedAgg(column='identity__Invoice', aggfunc='nunique'),
        recency  = pd.NamedAgg(column='recency__recency', aggfunc='min'),
        monetary = pd.NamedAgg(column='spending__spending', aggfunc='sum')
    )

    df.columns = [col.split('__')[-1] if '__' in col else col for col in df.columns]

    return df

class Identity(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self)->None:
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:
        return X

class Recency(TransformerMixin, BaseEstimator):

    def __init__(self, date_to_compare:datetime) -> None:
        super().__init__()
        self.date_to_compare = date_to_compare

    def get_feature_names_out(self)->None:
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        col = X_copy.columns

        date_to_compare = self.date_to_compare

        X_copy[col[1]] = pd.to_datetime(X_copy[col[1]])

        X_copy['recency'] = date_to_compare - X_copy[col[1]]

        X_copy['recency'] = X_copy['recency'].apply(lambda x: x.days)

        return X_copy[['recency']]*(-1)
    
class Spending(TransformerMixin, BaseEstimator):

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out()->None:
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        col = X_copy.columns

        X_copy['spending'] = X_copy[col[0]]*X_copy[col[1]]

        return X_copy['spending']

class QuartileEncoder(TransformerMixin, BaseEstimator):

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self)->None:
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        cols = X_copy.columns

        new_cols = []

        for col in cols:

            df = pd.DataFrame()

            minimum = np.min(X_copy[col])
            fst_quart = np.percentile(X_copy[col], 25)
            scnd_quart = np.percentile(X_copy[col], 50)
            thrd_quart = np.percentile(X_copy[col], 75)

            df['third']= (X_copy[col]>thrd_quart).astype('int')
            df['scnd'] = (X_copy[col]>scnd_quart).astype('int')
            df['first']= (X_copy[col]>fst_quart).astype('int')
            df['zero'] = (X_copy[col]>=minimum).astype('int')

            X_copy[f"{col}_enc"] = df['third'] + df['scnd'] + df['first'] + df['zero']

            new_cols.append(f"{col}_enc")

        return X_copy[new_cols]