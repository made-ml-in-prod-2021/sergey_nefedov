from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from typing import List


@dataclass()
class FeaturesParams:
    features: List[str]


FeaturesSchema = class_schema(FeaturesParams)


def read_features_params(path: str):
    with open(path, "r") as input_stream:
        schema = FeaturesSchema()
        return schema.load(yaml.safe_load(input_stream))
