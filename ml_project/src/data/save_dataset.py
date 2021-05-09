import sys
import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def save_data(df: pd.DataFrame, output_path: str):
    logger.info(f"Saving data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Data was saved successfully!")


def target_to_dataframe(predicts: np.ndarray) -> pd.DataFrame:
    """


    :rtype: pd.DataFrame
    """
    logger.info("Convert predicts to dataframe")
    return pd.DataFrame({"target": predicts})
