import os
import pandas as pd


def load_csv(path):
    if not os.path.exists(path):
        raise Exception("File '{}' does not exists.".format(path))

    return pd.read_csv(path)
