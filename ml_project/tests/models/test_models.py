from typing import List

import pytest
from sklearn.ensemble import RandomForestClassifier


from src.entities import (
    FeatureParams, TrainingParams
)
from src.features.build_features import (
    make_features,
    extract_target,
    build_transformer,
)
from src.models import (
    train_model,
    evaluate_model,
    predict_model,
    serialize_model,
    load_model,
)
from src.data.make_dataset import read_data


@pytest.fixture()
def dataset_path() -> str:
    return 'data/raw/heart.csv'


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


@pytest.fixture()
def training_params():
    params = TrainingParams(
        model_type="RandomForestClassifier",
        n_estimators=100,
        random_state=42,
    )
    return params


def test_model_fit_predict(
        feature_params: FeatureParams,
        dataset_path: str,
        training_params: TrainingParams,
):
    df = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(df)
    features = make_features(transformer, df)
    target = extract_target(df, feature_params)

    if training_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=training_params.n_estimators,
            random_state=training_params.random_state,
        )

    model.fit(features, target)
    predicts = predict_model(model, features)
    metrics = evaluate_model(predicts, target)

    assert metrics["accuracy"] > 0.5
    assert metrics["recall"] > 0.5
    assert metrics["f1_score"] > 0.5
    assert metrics["roc_auc"] > 0.5
