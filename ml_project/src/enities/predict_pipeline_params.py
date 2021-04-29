from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from .feature_params import FeatureParams
import yaml


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    model_path: str
    output_path: str
    feature_params: FeatureParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))