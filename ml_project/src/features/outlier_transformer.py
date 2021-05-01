import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """ Custom Transformer """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """ Drop rows with outliers values. """
        columns = list(df.columns.values)

        for column in columns:
            left = df[column].quantile(self.threshold)
            right = df[column].quantile(1 - self.threshold)
            df = df[(df[column] >= left) & (df[column] <= right)]

        return df
