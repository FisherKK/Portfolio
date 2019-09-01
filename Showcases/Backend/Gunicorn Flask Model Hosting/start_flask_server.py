from flask import Flask, request, jsonify
import numpy as np

from project.config.model import MODEL_PROPERTIES

from project.config.constants import (
    SERVER_PORT,
    SERVER_HOST
)

app = Flask(__name__)


@app.route("/classify_image_vector", methods=["POST"])
def classify_image_vector():
    """It reads JSON file sent via application/json
    content type, post request and looks for data vector under "image_vector" key. The vector is used to obtain
    predictions from each model. Returns a JSON string if data vector is correct or error message if data format is
    incorrect.

    Returns:
    -----------
    result: string
        Python dict with model outputs parsed to JSON string.
    """
    try:
        vector = np.array([request.json["image_vector"]])
        result = jsonify({key: p["prediction_function"](model[key], vector) for key, p in MODEL_PROPERTIES.items()})
        return result
    except ValueError:
        return jsonify({"error": "Invalid data shape."})


def init_models():
    """Loads all models set in MODEL_PROPERTIES and loads them into global dictionary named model."""
    global model
    model = {key: prop["load_function"](name=key) for key, prop in MODEL_PROPERTIES.items()}


def _app():
    """Setups models and starts Flask server under SERVER_HOST:SERVER_PORT url."""
    init_models()
    app.run(host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":
    _app()
