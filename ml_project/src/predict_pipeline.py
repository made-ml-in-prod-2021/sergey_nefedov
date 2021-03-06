import logging
import sys
import click


from src.entities.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_pipeline_params,
)
from src.data import (
    read_data,
    save_data,
    target_to_dataframe,
)
from src.models import (
    load_model,
    predict_model,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    X_test = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"loaded dataset from {predict_pipeline_params.input_data_path}")

    model = load_model(predict_pipeline_params.model_path)
    logger.info("Model loading has finished. ")

    logger.info("Start making predictions...")
    logger.info(f"X_test.shape is {X_test.shape}")

    predicts = predict_model(
        model,
        X_test,
    )
    logger.info("Predictions computed. Save to file. ")
    save_data(target_to_dataframe(predicts), predict_pipeline_params.output_path)
    logger.info(f"Predictions successfully saved in {predict_pipeline_params.output_path}")

    return


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
