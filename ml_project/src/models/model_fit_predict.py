import logging
import sys

import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

from src.entities.train_params import TrainingParams

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_model(
        transformer: TransformerMixin,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        train_params: TrainingParams,
) -> Pipeline:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            random_state=train_params.random_state,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(max_iter=train_params.max_iter)
    else:
        raise NotImplementedError()

    clf_model = Pipeline(
        steps=[('preprocessor', transformer),
               ('classifier', model),
               ]
    )

    logger.info("Starting fit model...")
    clf_model.fit(X_train, y_train)

    return clf_model


def predict_model(
    model: Pipeline, X_test: pd.DataFrame, use_log_trick: bool = False
) -> np.ndarray:
    logger.info("Starting model predict...")
    predicts = model.predict(X_test)
    if use_log_trick:
        predicts = np.exp(predicts)
    logger.info("Predicts are ready.")
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    logger.info(f"Starting model evaluate...")
    if use_log_trick:
        target = np.exp(target)
    return {
        "accuracy": accuracy_score(target, predicts),
        "recall": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }


def serialize_model(model: Pipeline, output: str) -> str:
    logger.info(f"Serialize model to {output}")
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(model_path: str) -> Pipeline:
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as mp:
        model = pickle.load(mp)
    return model
