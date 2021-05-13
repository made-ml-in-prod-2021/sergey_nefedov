import logging
import sys

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.entities.feature_params import FeatureParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """ Custom Transformer """

    def __init__(self, feature_params: FeatureParams):
        self.feature_params = feature_params

    def fit(self, df: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame, columns=None, y=None) -> pd.DataFrame:
        """ Drop rows with outliers values. """
        logger.info("Starting OutlierTransformer...")
        if columns is None:
            columns = self.feature_params.numerical_features

        logger.info(f"Removing outlier for columns: {columns}")

        for column in columns:
            logger.info(f"Current column is '{column}'")
            perc25 = df[column].quantile(0.25)
            perc75 = df[column].quantile(0.75)
            IQR = perc75 - perc25
            left = perc25 - 1.5 * IQR
            right = perc75 + 1.5 * IQR
            logger.info(f"Outliers boundary: [{round(left, 3)}, {round(right, 3)}]")
            number_outliers = df[(df[column] > right) | (df[column] < left)].shape[0]
            if number_outliers == 0:
                logger.info(f"Column '{column}' has no outliers values")
            else:
                logger.info(f"Outliers count: {number_outliers}")
                logger.info(f"Outliers percent: {round(number_outliers / df.shape[0] * 100, 2)}%")
            df = df[(df[column] >= left) & (df[column] <= right)]

        return df
