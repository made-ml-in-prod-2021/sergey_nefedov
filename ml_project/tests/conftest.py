from typing import List

import pytest
import numpy as np
import pandas as pd

from src.entities import (
    FeatureParams,
    TrainingParams,
    SplittingParams,
)


@pytest.fixture(scope='session')
def categorical_features() -> List[str]:
    return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


@pytest.fixture(scope='session')
def numerical_features() -> List[str]:
    return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


@pytest.fixture(scope='session')
def features_to_drop() -> List[str]:
    return ['']


@pytest.fixture(scope='session')
def target_col() -> str:
    return 'target'


@pytest.fixture(scope='session')
def dataset_path() -> str:
    return 'data/raw/heart.csv'


@pytest.fixture(scope='session')
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


@pytest.fixture(scope='session')
def training_params():
    params = TrainingParams(
        model_type="RandomForestClassifier",
        n_estimators=100,
        random_state=42,
    )
    return params


@pytest.fixture(scope='session')
def splitting_params():
    params = SplittingParams(
        val_size=0.2,
        random_state=42,
    )
    return params


@pytest.fixture(scope='session')
def test_df() -> pd.DataFrame:
    """ Generates random tests data. """
    size = 10
    random_state = 42

    np.random.seed(random_state)

    test_df = pd.DataFrame({
        'age': np.random.randint(29, 77, size=size),
        'cp': np.random.randint(4, size=size),
        'sex': np.random.randint(2, size=size),
        'trestbps': np.random.randint(94, 200, size=size),
        'chol': np.random.randint(126, 564, size=size),
        'fbs': np.random.randint(2, size=size),
        'restecg': np.random.randint(2, size=size),
        'thalach': np.random.randint(71, 202, size=size),
        'exang': np.random.randint(2, size=size),
        'oldpeak': 4 * np.random.rand(),
        'slope': np.random.randint(3, size=size),
        'ca': np.random.randint(5, size=size),
        'thal': np.random.randint(4, size=size),
        'target': np.random.randint(2, size=size),
    })

    return test_df
