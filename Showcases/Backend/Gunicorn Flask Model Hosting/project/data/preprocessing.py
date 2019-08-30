from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from project.config import GLOBAL_SEED


def flatten(samples):
    samples_num = samples.shape[0]
    return samples.reshape(samples_num, -1)


def min_max_scale_image_data(samples, min_val=0.0, max_val=255.0):
    return (samples - min_val) / (max_val - min_val)


def one_hot_encode_categorical_data(targets):
    one_hot_encoder = OneHotEncoder(sparse=False, categories="auto")
    targets = targets.reshape(len(targets), 1)
    return one_hot_encoder.fit_transform(targets)


def split(samples, targets, ratio=0.2, shuffle=True):
    return train_test_split(samples, targets, test_size=ratio, random_state=GLOBAL_SEED, shuffle=shuffle)
