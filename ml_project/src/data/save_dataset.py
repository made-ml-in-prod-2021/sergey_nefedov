import pandas as pd


def save_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)


def target_to_dataframe(predicts):
    """


    :rtype: pd.DataFrame
    """
    return pd.DataFrame({"target": predicts})