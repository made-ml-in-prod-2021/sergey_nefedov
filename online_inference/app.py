import logging
import sys
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse
from pydantic import BaseModel, conlist, validator
from sklearn.pipeline import Pipeline

from src.entities.app_paprams import read_app_params
from src.entities.feature_params import read_features_params


CONFIG_PATH = "configs/app_config.yaml"
FEATURES = read_features_params("configs/features_config.yaml").features


model: Optional[Pipeline] = None

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class HeartDiseaseClassifierModel(BaseModel):
    data: List[conlist(Union[float, int])]
    features: List[str]

    @validator('features')
    def validate_model_features(cls, features):
        if set(features) != set(FEATURES):
            raise ValueError(f"Invalid features! Valid features are: {FEATURES}")
        elif features != FEATURES:
            raise ValueError(f"Invalid features order! Valid features order is: {FEATURES}")
        return features

    @validator('data')
    def validate_number_data_columns_and_features(cls, data):
        if pd.DataFrame(data).shape[1] != len(FEATURES):
            raise ValueError(f"Invalid number of features! Valid number is: {len(FEATURES)}")
        return data


class HeartDiseaseResponseModel(BaseModel):
    class_id: int


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as fin:
        return pickle.load(fin)


def make_predict(data: List, features: List[str], model: Pipeline):
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)
    return [
        HeartDiseaseResponseModel(class_id=class_id) for class_id in predicts
    ]


app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_app_model():
    app_params = read_app_params(CONFIG_PATH)
    logger.info(f"Features are {FEATURES}")
    logger.info(f"Start loading model from {app_params.model_path}")
    global model
    model = load_model(app_params.model_path)
    logger.info("Model was loaded successfully!")
    logger.info(f"Model is: {model.__repr__()}")


@app.get("/predict/", response_model=List[HeartDiseaseResponseModel])
def predict(request: HeartDiseaseClassifierModel):
    logger.info("Starting make predicts...")
    return make_predict(request.data, request.features, model)


def setup_app():
    app_params = read_app_params(CONFIG_PATH)
    logger.info(f"Running app on {app_params.host} with port {app_params.port}")
    uvicorn.run(app, host=app_params.host, port=app_params.port)


if __name__ == "__main__":
    setup_app()
