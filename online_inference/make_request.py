import click
import requests
import logging
import sys

import numpy as np
import pandas as pd

from src.entities.app_paprams import read_app_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command(name='main')
@click.option("--dataset_path", default='data/test_data.csv')
@click.option("--app_config", default='configs/app_config.yaml')
def make_predict(dataset_path, app_config):
    logger.info(f"Read data from {dataset_path}")
    data = pd.read_csv(dataset_path)

    app_params = read_app_params(app_config)

    request_url = f"http://{app_params.host}:{app_params.port}/predict/"
    logger.info(f"Request URL is {request_url}")

    request_features = list(data.columns)
    logger.info(f"Features: {request_features}")

    for i in range(data.shape[0]):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        for feature in request_features:
            logger.info(f"feature: {feature}, value: {data.iloc[i][feature]}")
        response = requests.get(
            request_url,
            json={"data": [request_data], "features": request_features},
        )
        logger.info(f'Response code: {response.status_code}, body: {response.json()}')


if __name__ == "__main__":
    make_predict()
