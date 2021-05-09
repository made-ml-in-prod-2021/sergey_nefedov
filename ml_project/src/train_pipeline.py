import json
import logging
import sys

from typing import Tuple
import click
import pandas as pd

from src.data import read_data, split_train_val_data
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.features import make_features
from src.features.build_features import extract_target, build_transformer
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
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    train_features, train_target, val_features, val_target = transform_data(train_df, val_df, training_pipeline_params)

    logger.info(f"Training model: model_type = {training_pipeline_params.train_params.model_type}")
    model = train_model(train_features, train_target, training_pipeline_params.train_params)
    logger.info("Model is ready for predict")

    logger.info(f"Making predicts...")
    predicts = predict_model(model, val_features)
    logger.info(f"predicts.shape is {predicts.shape}")
    metrics = evaluate_model(predicts, val_target)
    logger.info(f"metrics is {metrics}")

    logger.info(f"Dumping metrics to {training_pipeline_params.metric_path}")
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)

    return path_to_model, metrics


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
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info("build Transformer...")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)

    logger.info("making features...")
    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    logger.info(f"train_features.shape is {train_features.shape}")
    logger.info(f"train_target.shape is {train_target.shape}")

    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    logger.info(f"val_features.shape is {val_features.shape}")
    logger.info(f"val_target.shape is {val_target.shape}")

    return train_features, train_target, val_features, val_target


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
