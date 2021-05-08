import numpy as np

from src.entities.feature_params import FeatureParams
from src.features.build_features import (
    make_features,
    build_transformer,
    extract_target,
)
from src.features.outlier_transformer import OutlierTransformer
from src.data.make_dataset import read_data


def test_make_features(feature_params: FeatureParams, dataset_path: str):
    df = read_data(dataset_path)

    outlier_transformer = OutlierTransformer(feature_params)
    outlier_transformer.fit(df)
    df = outlier_transformer.transform(df)

    column_transformer = build_transformer(feature_params)
    column_transformer.fit(df)
    features = make_features(column_transformer, df)
    assert features.shape[1] == 30, (
        f"Its expected to have 30 features after transformer, got {features.shape[1]}"
    )


def test_extract_target(feature_params: FeatureParams, dataset_path: str):
    df = read_data(dataset_path)
    extracted_target = extract_target(df, feature_params)
    assert np.allclose(extracted_target, df[feature_params.target_col])



