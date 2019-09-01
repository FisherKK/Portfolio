import numpy as np

from project.data.preprocessing import min_max_scale_image_data


def predict_with_xgboost_model(model, sample):
    """Uses XGBClassifier model to obtain prediction on given sample. Returns a dictionary with prediction under the
    key.

    Parameters:
    -----------
    model: XGBClassifier
        Trained model.
    sample: ndarray
        Numpy array containing flattened image.

    Returns:
    -----------
    result: dict
        Dictionary with prediction saved under the "predicted_number" key.
    """
    prediction = int(model.predict(min_max_scale_image_data(sample))[0])
    result = {"predicted_number": prediction}
    return result


def predict_with_mlp_model(model, sample):
    """Uses tf.keras model to obtain prediction on given sample. Returns a dictionary with prediction under the key.

    Parameters:
    -----------
    model: tensorflow.python.keras.engine.sequential.Sequential
        Trained model.
    sample: ndarray
        Numpy array containing flattened image.

    Returns:
    -----------
    result: dict
        Dictionary with prediction saved under the "predicted_number" key.
    """
    prediction = int(np.argmax(model.predict(min_max_scale_image_data(sample))))
    result = {"predicted_number": prediction}
    return result
