from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from project.config.constants import GLOBAL_SEED


def flatten(samples):
    """Turns 3-dimensional array into 2-dimensional by merging 2nd and 3rd dimensions responsible for image width and
    height.

    Parameters:
    -----------
    samples: ndarray
        3-dimensional matrix holding images where:
            - 1st dimension is a number of images
            - 2nd dimension is image width
            - 3rd dimension is image height
    sample: ndarray
        Numpy array containing flattened image.

    Returns:
    -----------
    flattened_samples: ndarray
        2-dimensional matrix holding images where:
            - 1st dimension is a number of images
            - 2nd dimension is a vector of pixel values
    """
    samples_num = samples.shape[0]
    flattened_samples = samples.reshape(samples_num, -1)
    return flattened_samples


def min_max_scale_image_data(samples, min_val=0.0, max_val=255.0):
    """Scales image values to <0.0, 1.0> range.

    Parameters:
    -----------
    samples: ndarray
        Container with image or images. Can be either 1-dimensional vector holding pixel values or 2-dimensional matrix
        where:
            - 1st dimension is a number of images
            - 2nd dimension is a vector of pixel values
    min_val: float
        Minimum value of the pixel. In RGB min value is 0.0.
    max_val: float
        Maximum value of the pixel. In RGB max value is 255.0.

    Returns:
    -----------
    scaled_samples: ndarray
        Array with scaled data and the same shape as an input.
    """
    scaled_samples = (samples - min_val) / (max_val - min_val)
    return scaled_samples


def one_hot_encode_categorical_data(targets):
    """Performs one-hot-encoding (replacing class index with array of size equal to number of indices, and placing 1.0
    value in place corresponding to class index and 0.0 in others) on target values.

    Parameters:
    -----------
    targets: ndarray
       Container with target value indices.

    Returns:
    -----------
    one_hot_encoded_data: ndarray
       Container with one-hot-encoded data.
    """
    one_hot_encoder = OneHotEncoder(sparse=False, categories="auto")
    targets = targets.reshape(len(targets), 1)
    one_hot_encoded_data = one_hot_encoder.fit_transform(targets)
    return one_hot_encoded_data


def split(samples, targets, ratio=0.2, shuffle=True):
    """Splits samples in to two datasets. First dataset is of (num_samples * 1 - ratio) size and the other dataset
    contains the rest of the samples.

    Parameters:
    -----------
    samples: ndarray
        Array containing data. Can be either 2-dimensional array with flattened images or 3-dimensional data with
        image matrices.
    targets: targets
        Array containing expected results of prediction (class indices) for each corresponding row in samples array.

    Returns:
    -----------
    samples_split: tuple
        Data split containing split data.
    """
    samples_split = train_test_split(samples, targets, test_size=ratio, random_state=GLOBAL_SEED, shuffle=shuffle)
    return samples_split
