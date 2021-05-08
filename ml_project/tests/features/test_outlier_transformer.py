import pandas as pd

from src.features.outlier_transformer import OutlierTransformer
from src.entities import FeatureParams


def test_outlier_transformer(test_df: pd.DataFrame, feature_params: FeatureParams):
    transformer = OutlierTransformer(feature_params)
    transformed_df = transformer.transform(test_df)
    assert transformed_df.shape == test_df.shape