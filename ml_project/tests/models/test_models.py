import pytest
import pandas as pd

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
    evaluate_model,
    predict_model,
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
        test_df: pd.DataFrame,
        training_params: TrainingParams,
):
    transformer = build_transformer(feature_params)
    transformer.fit(test_df)
    features = make_features(transformer, test_df)
    target = extract_target(test_df, feature_params)

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
