import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

from src.enities.train_params import TrainingParams

SklearnRegressionModel = Union[RandomForestClassifier]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame, use_log_trick: bool = True
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "accuracy": accuracy_score(target, predicts),
        "recall": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "roc_auc:": roc_auc_score(target, predicts),
    }


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
