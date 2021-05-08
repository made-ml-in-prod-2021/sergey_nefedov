import numpy as np
import pandas as pd

from src.data.make_dataset import read_data, split_train_val_data
from src.entities import SplittingParams


def test_read_data(tmpdir, test_df):
    filepath = tmpdir.join('test.csv')
    test_df.to_csv(filepath, index=False)
    read_df = read_data(filepath)

    assert np.allclose(test_df.values, read_df.values)


def test_split_train_val_data(test_df: pd.DataFrame):
    splitting_params = SplittingParams(random_state=42, val_size=0.2)
    train_data, val_data = split_train_val_data(test_df, splitting_params)

    assert train_data.shape[0] == test_df.shape[0] * (1 - splitting_params.val_size)
    assert val_data.shape[0] == test_df.shape[0] * splitting_params.val_size
