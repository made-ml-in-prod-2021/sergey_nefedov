import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.entities import (
    FeatureParams, TrainingParams
)
from src.features.build_features import (
    extract_target,
    build_transformer,
)
from src.models import (
    evaluate_model,
    predict_model,
)


def test_model_fit_predict(
        feature_params: FeatureParams,
        test_df: pd.DataFrame,
        training_params: TrainingParams,
):
    transformer = build_transformer(feature_params)
    X = test_df.drop(columns=[feature_params.target_col])
    transformer.fit(X)
    target = extract_target(test_df, feature_params)

    model = RandomForestClassifier(
        n_estimators=training_params.n_estimators,
        random_state=training_params.random_state,
    )

    clf_model = Pipeline(
        steps=[('preprocessor', transformer),
               ('classifier', model),
               ]
    )

    clf_model.fit(X, target)
    predicts = predict_model(clf_model, X)
    metrics = evaluate_model(predicts, target)

    expected_metrics = ['accuracy', 'recall', 'f1_score', 'roc_auc']

    assert set(metrics) == set(expected_metrics)
    assert metrics["accuracy"] > 0.5
    assert metrics["recall"] > 0.5
    assert metrics["f1_score"] > 0.5
    assert metrics["roc_auc"] > 0.5
