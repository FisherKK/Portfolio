import os
import pandas as pd


def load_csv(path):
    """Loads .csv file via pandas library. Throws error if passed path does not exist.

    Parameters:
    -----------
    path: str
        Path to .csv file.
    """
    if not os.path.exists(path):
        raise Exception("File '{}' does not exists.".format(path))

    return pd.read_csv(path)
