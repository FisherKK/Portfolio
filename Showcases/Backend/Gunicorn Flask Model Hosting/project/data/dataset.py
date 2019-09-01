import tensorflow as tf

from project.data.preprocessing import (
    one_hot_encode_categorical_data,
    min_max_scale_image_data,
    flatten,
    split
)


def load_mnist_data(ohe=False):
    """Loadads and preprocess MNIST dataset. Returns data divided into train, val and test datasets.

    Parameters:
    -----------
    ohe: boolean
       Flag used to decide whether output targets should be one-hot-encoded.

    Returns:
    -----------
    data: tuple
        Three element tuple containing preprocessed MNIST data divided into with train, val test datasets.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = flatten(x_train)
    x_test = flatten(x_test)

    x_train = min_max_scale_image_data(x_train)
    x_test = min_max_scale_image_data(x_test)

    if ohe:
        y_train = one_hot_encode_categorical_data(y_train)
        y_test = one_hot_encode_categorical_data(y_test)

    x_train, x_val, y_train, y_val = split(x_train, y_train)

    data = (x_train, y_train), (x_val, y_val), (x_test, y_test)
    return data
