import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from src.entities.feature_params import FeatureParams


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """ Custom Transformer """

    def __init__(self, feature_params: FeatureParams):
        self.feature_params = feature_params

    def fit(self, df: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame, columns=None, y=None) -> pd.DataFrame:
        """ Drop rows with outliers values. """
        if columns is None:
            columns = self.feature_params.numerical_features

        for column in columns:
            perc25 = df[column].quantile(0.25)
            perc75 = df[column].quantile(0.75)
            IQR = perc75 - perc25
            left = perc25 - 1.5 * IQR
            right = perc75 + 1.5 * IQR
            df = df[(df[column] >= left) & (df[column] <= right)]

        return df
