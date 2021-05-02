import pytest
import pandas as pd

from src.entities.feature_params import FeatureParams
from typing import List

from src.features.outlier_transformer import OutlierTransformer


@pytest.fixture()
def test_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    return df


@pytest.fixture()
def categorical_features() -> List[str]:
    return ['']


@pytest.fixture()
def numerical_features() -> List[str]:
    return ['feature']


@pytest.fixture()
def features_to_drop() -> List[str]:
    return ['']


@pytest.fixture()
def target_col() -> str:
    return ''


@pytest.fixture()
def feature_params(categorical_features: List[str],
                   numerical_features: List[str],
                   features_to_drop: List[str],
                   target_col: str,
                   ) -> FeatureParams:
    return FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )


def test_outlier_transformer(test_df: pd.DataFrame):
    transformer = OutlierTransformer(feature_params, threshold=0.05)
    transformed_df = transformer.transform(test_df, columns=['feature'])
    assert transformed_df.shape == (test_df.shape[0] - 2, test_df.shape[1])