from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class AppParams:
    host: str
    port: int
    model_path: str


AppParamsSchema = class_schema(AppParams)


def read_app_params(path: str):
    with open(path, "r") as input_stream:
        schema = AppParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
