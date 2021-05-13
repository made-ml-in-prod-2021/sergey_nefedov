import json
import logging
import sys

from typing import Tuple
import click
import pandas as pd
from sklearn.base import TransformerMixin

from src.data import (
    read_data,
    split_train_val_data,
    save_data,
)
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.features.build_features import (
    extract_target,
    build_transformer,
)
from src.features.outlier_transformer import OutlierTransformer
from src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)

    data = remove_outliers(data, training_pipeline_params)

    train_df, val_df = split_train_val_data(data, training_pipeline_params.splitting_params)
    X_train, y_train, X_test, y_test, transformer = transform_data(train_df, val_df, training_pipeline_params)

    logger.info("Save test_data without labels")
    save_data(X_test, training_pipeline_params.output_test_data_path)
    path_to_test_data = training_pipeline_params.output_test_data_path

    logger.info(f"Training model: model_type = {training_pipeline_params.train_params.model_type}")
    if training_pipeline_params.train_params.model_type == 'RandomForestClassifier':
        logger.info(f"model's params: "
                    f"n_estimators = {training_pipeline_params.train_params.n_estimators}, "
                    f"random_state = {training_pipeline_params.train_params.random_state}")
    elif training_pipeline_params.train_params.model_type == 'LogisticRegression':
        logger.info(f"Model's params:"
                    f"max_iter = {training_pipeline_params.train_params.max_iter},"
                    f"random_state = {training_pipeline_params.train_params.random_state}")
    model = train_model(transformer, X_train, y_train, training_pipeline_params.train_params)
    logger.info("Model is ready for predict")

    logger.info(f"Making predicts...")
    predicts = predict_model(model, X_test)
    logger.info(f"predicts.shape is {predicts.shape}")
    metrics = evaluate_model(predicts, y_test)
    logger.info(f"metrics is {metrics}")

    logger.info(f"Dumping metrics to {training_pipeline_params.metric_path}")
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    return path_to_model, path_to_test_data, metrics


def remove_outliers(
        data: pd.DataFrame,
        training_pipeline_params: TrainingPipelineParams,
) -> pd.DataFrame:
    logger.info("build OutlierTransformer...")
    outlier_transformer = OutlierTransformer(training_pipeline_params.feature_params)
    outlier_transformer.fit(data)

    logger.info(f"Removing outliers values: before shape is {data.shape}")
    data = outlier_transformer.transform(data)
    logger.info(f"Removing outliers values: after data.shape is {data.shape}")

    return data


def transform_data(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        training_pipeline_params: TrainingPipelineParams,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, TransformerMixin]:

    logger.info("making features...")
    X_train = train_df.drop(columns=[training_pipeline_params.feature_params.target_col])
    y_train = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info(f"X_train.shape is {X_train.shape}")
    logger.info(f"y_train.shape is {y_train.shape}")

    X_test = val_df.drop(columns=[training_pipeline_params.feature_params.target_col])
    y_test = extract_target(val_df, training_pipeline_params.feature_params)
    logger.info(f"X_test.shape is {X_test.shape}")
    logger.info(f"y_test.shape is {y_test.shape}")

    logger.info("build Transformer...")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)

    return X_train, y_train, X_test, y_test, transformer


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
