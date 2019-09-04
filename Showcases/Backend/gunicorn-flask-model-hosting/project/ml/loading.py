import os
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from project.config.constants import MODEL_DIR
from sklearn.externals import joblib


class GraphAwareTfKerasWrapper:
    def __init__(self):
        self.graph = None
        self.model = None

    def predict(self, sample):
        """Uses stored model for inference with the scope of it's graph.

        Parameters:
        -----------
        sample: ndarray
            Array with single image or many images. Expected shape is (? - image num, 784).
        """
        with self.graph.as_default():
            return self.model.predict(sample)

    def load_h5(self, filepath):
        """Creates fresh model graph to which it loads model from sent filepath.

        Parameters:
        -----------
        filepath: string
            Path to .h5 file.
        """
        self.graph = tf.get_default_graph()
        self.model = load_model(filepath)


def load_mlp_model(name):
    """Loads tf.keras model from .h5 file, from MODEL_DIR dir.

    Parameters:
    -----------
    name: str
        Model name.

    Returns:
    -----------
    model: GraphAwareTfKerasWrapper
        Loaded tf.keras model wrapped in graph storing class.
    """
    filepath = os.path.join(MODEL_DIR, "{}.h5".format(name))
    model = GraphAwareTfKerasWrapper()
    model.load_h5(filepath)
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
    print("Successfully loaded model from '{}' filepath.".format(filepath))
    return model
