import os
import json
import time
import numpy as np
import tensorflow as tf

from project.config.constants import TESTING_DIR
from project.config.model import MODEL_PROPERTIES

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
tf.Session(config=config)

TRIALS = 1000


def _load_image():
    test_image = os.path.join(TESTING_DIR, "mnist_image_example.json")
    with open(test_image, "r") as file:
        image_json = json.load(file)
        print("Loaded test image from '{}'.".format(test_image))
        return np.array([image_json["image_vector"]])


def _test_prediction_function_speed(model, vector, prediction_function, trials):
    t = []
    for _ in range(trials):
        start_time = time.time()
        _ = prediction_function(model, vector)
        t.append(time.time() - start_time)
    return np.round(np.mean(t), 4), np.round(np.min(t), 4), np.round(np.max(t), 4), np.round(np.std(t), 4)


if __name__ == "__main__":
    test_image = _load_image()

    for model_key, properties in MODEL_PROPERTIES.items():
        print("----------------------------")
        model = properties["load_function"](name=model_key)
        prediction_func = properties["prediction_function"]

        averaged_time, min_time, max_time, std_time = _test_prediction_function_speed(
            model, test_image, prediction_func, TRIALS
        )

        print("\t- {}: {} inference trials took on average '{}' (min: {}, max: {}, std: {}) seconds.".format(
            model_key, TRIALS, averaged_time, min_time, max_time, std_time))
