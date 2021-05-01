import pytest
import pandas as pd

from src.features.outlier_transformer import OutlierTransformer


@pytest.fixture()
def test_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    return df


def test_outlier_transformer(test_df: pd.DataFrame):
    transformer = OutlierTransformer(threshold=0.1)
    transformed_df = transformer.transform(test_df)
    assert transformed_df.shape == (test_df.shape[0] - 2, test_df.shape[1])