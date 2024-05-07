
import pandas as pd
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

        return X_copy[['recency']]
    
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