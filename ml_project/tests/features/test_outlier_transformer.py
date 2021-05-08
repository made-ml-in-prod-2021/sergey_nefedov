import pytest
import pandas as pd

from src.features.outlier_transformer import OutlierTransformer
from src.entities import FeatureParams


@pytest.fixture(scope='session')
def test_df_with_outliers():
    return pd.DataFrame({"value": [-10, 0, 1, 2, 3, 4, 100]})


def test_outlier_transformer(test_df_with_outliers: pd.DataFrame, feature_params: FeatureParams):
    transformer = OutlierTransformer(feature_params)
    transformed_df = transformer.transform(test_df_with_outliers, columns=["value"])
    assert test_df_with_outliers.shape[0] - transformed_df.shape[0] == 2
