import pytest
import numpy as np

from src.entities.feature_params import FeatureParams
from src.features.build_features import (
    make_features,
    build_transformer,
    extract_target,
)
from src.data.make_dataset import read_data


@pytest.fixture()
def categorical_features() -> list:
    return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


@pytest.fixture()
def numerical_features() -> list:
    return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


@pytest.fixture()
def features_to_drop() -> list:
    return ['']


@pytest.fixture()
def target_col() -> str:
    return 'target'


@pytest.fixture()
def dataset_path() -> str:
    return 'data/raw/heart.csv'


@pytest.fixture()
def feature_params(categorical_features,
                   numerical_features,
                   features_to_drop,
                   target_col,
                   ) -> FeatureParams:
    return FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )


def test_make_features(feature_params: FeatureParams, dataset_path: str):
    df = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(df)
    features = make_features(transformer, df)
    assert features.shape[1] == 30, (
        f"Its expected to have 30 features after transformer, got {features.shape[1]}"
    )


def test_extract_target(feature_params: FeatureParams, dataset_path: str):
    df = read_data(dataset_path)
    extracted_target = extract_target(df, feature_params)
    assert np.allclose(extracted_target, df[feature_params.target_col])



