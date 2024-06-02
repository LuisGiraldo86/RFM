
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin

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

            X_copy[f"{col}_enc"] = (df['third'] + df['scnd'] + df['first'] + df['zero'])-1

            new_cols.append(f"{col}_enc")

        return X_copy[new_cols]

class RFMcalculator(TransformerMixin, BaseEstimator):

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self)->None:
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
        
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        cols = X_copy.columns

        X_res = pd.DataFrame(np.zeros(X_copy.shape[0]), columns=["RFM"])

        for col in cols:
            X_res["RFM"] += X_copy[col]

        mask_gold = (X_res['RFM']>6)
        mask_bronze = (X_res['RFM']<3)

        X_res["scale"]= None
        X_res['subscale']=None

        X_res.loc[mask_gold, 'scale'] = "Gold"
        X_res.loc[mask_bronze, 'scale'] = "Bronze"
        X_res.loc[~mask_gold & ~mask_bronze, 'scale'] = "Silver"

        X_res.loc[mask_gold, 'subscale'] = X_res.loc[mask_gold, 'RFM']-7
        X_res.loc[mask_bronze, 'subscale'] = X_res.loc[mask_bronze, 'RFM']
        X_res.loc[~mask_gold & ~mask_bronze, 'subscale'] = X_res.loc[~mask_gold & ~mask_bronze, 'RFM']-3

        return X_res
