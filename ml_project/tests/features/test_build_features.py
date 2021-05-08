import pytest
import pandas as pd
import numpy as np

from src.entities.feature_params import FeatureParams
from src.features.build_features import (
    make_features,
    build_transformer,
    extract_target,
    build_numerical_pipeline,
    build_categorical_pipeline,
)
from src.features.outlier_transformer import OutlierTransformer
from src.data.make_dataset import read_data


def test_make_features(feature_params: FeatureParams, dataset_path: str):
    df = read_data(dataset_path)

    outlier_transformer = OutlierTransformer(feature_params)
    outlier_transformer.fit(df)
    df = outlier_transformer.transform(df)

    column_transformer = build_transformer(feature_params)
    column_transformer.fit(df)
    features = make_features(column_transformer, df)
    assert features.shape[1] == 30, (
        f"Its expected to have 30 features after transformer, got {features.shape[1]}"
    )


def test_extract_target(feature_params: FeatureParams, dataset_path: str):
    df = read_data(dataset_path)
    extracted_target = extract_target(df, feature_params)
    assert np.allclose(extracted_target, df[feature_params.target_col])


@pytest.fixture(scope='session')
def test_cat_df():
    return pd.DataFrame({"range": ["1", "2", "1"]})


def test_build_categorical_pipeline(test_cat_df: pd.DataFrame):
    output = build_categorical_pipeline().fit_transform(test_cat_df)
    assert output.shape == (3, 2)
    assert output[:, 0].sum() == 2


@pytest.fixture(scope='session')
def test_num_df():
    return pd.DataFrame({"value": [i for i in range(10)]})


def test_build_numerical_pipeline(test_num_df: pd.DataFrame):
    output = build_numerical_pipeline().fit_transform(test_num_df)
    assert output.shape == (10, 1)
    assert output[:, 0].mean() == 4.5
    assert abs(output[:, 0].std() - 2.872) < 1e-3
