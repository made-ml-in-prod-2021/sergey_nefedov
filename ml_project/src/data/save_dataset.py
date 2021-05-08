import pandas as pd
import numpy as np


def save_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)


def target_to_dataframe(predicts: np.ndarray) -> pd.DataFrame:
    """


    :rtype: pd.DataFrame
    """
    return pd.DataFrame({"target": predicts})
