import os

from project.config.constants import MODEL_DIR
from tensorflow.keras.models import load_model
from sklearn.externals import joblib


def load_mlp_model(name):
    """Loads tf.keras model from .h5 file, from MODEL_DIR dir.

    Parameters:
    -----------
    name: str
        Model name.

    Returns:
    -----------
    model: tensorflow.python.keras.engine.sequential.Sequential
        Loaded tf.keras model.
    """
    filepath = os.path.join(MODEL_DIR, "{}.h5".format(name))
    model = load_model(filepath)
    print("Successfully loaded model from '{}' filepath.".format(filepath))
    return model


def load_xgboost_model(name):
    """Loads XGBClassifier model from .pkl file, from MODEL_DIR dir.

    Parameters:
    -----------
    name: str
        Model name.

    Returns:
    -----------
    model: XGBClassifier
        Loaded XGBClassifier model.
    """
    filepath = os.path.join(MODEL_DIR, "{}.pkl".format(name))
    model = joblib.load(filepath)
    print("Successfully loaded model model '{}' filepath.".format(filepath))
    return model
