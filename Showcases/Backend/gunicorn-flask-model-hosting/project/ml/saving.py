import os

from project.config.constants import MODEL_DIR
from sklearn.externals import joblib


def save_mlp_model(model, name):
    """Saves tf.keras model into .h5 file under MODEL_DIR dir.

    Parameters:
    -----------
    model: tensorflow.python.keras.engine.sequential.Sequential
        Trained tf.keras model.
    name: str
        Model name.
    """
    filepath = os.path.join(MODEL_DIR, "{}.h5".format(name))
    model.save(filepath)
    print("Successfully saved model to '{}' filepath.".format(filepath))


def save_xgboost_model(model, name):
    """Saves XGBClassifier model into .pkl file under MODEL_DIR dir.

    Parameters:
    -----------
    model: XGBClassifier
        Trained XGBClassifier model.
    name: str
        Model name.
    """
    filepath = os.path.join(MODEL_DIR, "{}.pkl".format(name))
    joblib.dump(model, filepath)
    print("Successfully saved model to '{}' filepath.".format(filepath))
