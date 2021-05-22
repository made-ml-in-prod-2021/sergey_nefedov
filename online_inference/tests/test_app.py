from fastapi.testclient import TestClient

import pytest
import pandas as pd

from app import app


@pytest.fixture(scope='session')
def model_path() -> str:
    return 'models/model.pkl'


@pytest.fixture()
def dataset_path() -> str:
    return 'data/test_data.csv'


def test_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_predict_invalid_order_of_features(dataset_path):
    df = pd.read_csv(dataset_path)
    columns = df.columns.tolist()
    invalid_data = df[columns[1:] + columns[:1]]
    with TestClient(app) as client:
        data = invalid_data.values.tolist()
        features = invalid_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == 400
        assert "Invalid features order! Valid features order is" in response.text


def test_predict_invalid_data(dataset_path):
    df = pd.read_csv(dataset_path)
    columns = df.columns.tolist()
    invalid_data = df[columns[1:]]
    with TestClient(app) as client:
        data = invalid_data.values.tolist()
        features = invalid_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == 400
        assert "Invalid number of features! Valid number is:" in response.text


def test_predict_invalid_feature(dataset_path):
    df = pd.read_csv(dataset_path)
    invalid_data = df.rename(columns={'age':'kek'})
    with TestClient(app) as client:
        data = invalid_data.values.tolist()
        features = invalid_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == 400
        assert "Invalid features! Valid features are" in response.text


def test_predict_invalid_data_columns(dataset_path):
    df = pd.read_csv(dataset_path)
    invalid_data = df.copy()
    invalid_data['target'] = 0 * df.shape[0]
    with TestClient(app) as client:
        data = invalid_data.values.tolist()
        features = invalid_data.columns.tolist()
        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == 400


def test_prediction(model_path, dataset_path):
    with TestClient(app) as client:
        df = pd.read_csv(dataset_path)

        data = df.iloc[:2].values.tolist()
        features = df.iloc[:2].columns.tolist()

        response = client.get("/predict/",
                              json={"data": data, "features": features},
                              )
        assert response.status_code == 200
        assert response.json()[0]['class_id'] in [0, 1]
        assert response.json()[1]['class_id'] in [0, 1]