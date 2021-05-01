import numpy as np
import pandas as pd


def generate_test_dataframe(size: int, random_state: int) -> pd.DataFrame:
    """ Generates random tests data. """
    np.random.seed(random_state)

    df = pd.DataFrame({
        'age': np.random.randint(29, 77, size=size),
        'cp': np.random.randint(4, size=size),
        'sex': np.random.randint(2, size=size),
        'trestbps': np.random.randint(94, 200, size=size),
        'chol': np.random.randint(126, 564, size=size),
        'fbs': np.random.randint(2, size=size),
        'restecg': np.random.randint(2, size=size),
        'thalach': np.random.randint(71, 202, size=size),
        'exang': np.random.randint(2, size=size),
        'oldpeak': 4 * np.random.rand(),
        'slope': np.random.randint(3, size=size),
        'ca': np.random.randint(5, size=size),
        'thal': np.random.randint(4, size=size),
        'target': np.random.randint(2, size=size),
    })

    return df
