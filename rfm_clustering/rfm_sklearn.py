
import pandas as pd
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin


class Recency(TransformerMixin, BaseEstimator):

    def __init__(self, date_to_compare:datetime) -> None:
        super().__init__()
        self.date_to_compare = date_to_compare

    def get_feature_names_out(self)->None:
        pass

    def fit(self, X:pd.DataFrame, y=None):
        pass

    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        X_copy = X.copy()
        col = X_copy.columns[0]

        date_to_compare = self.date_to_compare

        X_copy[col] = pd.to_datetime(X_copy[col])

        X_copy['recency'] = (date_to_compare-X_copy[col]).days

        return X_copy['recency']