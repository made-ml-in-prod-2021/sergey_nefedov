import pytest

from src.train_pipeline import train_pipeline
from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    TrainingParams,
)


@pytest.fixture()
def training_params():
    params = TrainingParams(
        model_type="RandomForestClassifier",
        n_estimators=100,
        random_state=42,
    )
    return params


@pytest.fixture()
def splitting_params():
    params = SplittingParams(
        val_size=0.2,
        random_state=42,
    )
    return params


def test_train_pipeline(
        tmpdir,
        test_df,
        feature_params,
        training_params,
        splitting_params):
    input_dataset_path = tmpdir.join("data.csv")
    test_df.to_csv(input_dataset_path)
    expected_output_model_path = tmpdir.join("model.pkl")

    train_params = TrainingPipelineParams(
        input_data_path=input_dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=tmpdir.join("metrics.json"),
        splitting_params=splitting_params,
        feature_params=feature_params,
        train_params=training_params,
    )
    path_to_model, metrics = train_pipeline(train_params)
    assert path_to_model == expected_output_model_path
    assert "accuracy" in metrics.keys()
    assert "recall" in metrics.keys()
    assert "f1_score" in metrics.keys()
    assert "roc_auc" in metrics.keys()
