import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def _metric_per_class(Y_expected, Y_pred, metric):
    """Function returns a list of scores for each class for given metric.

    Parameters:
    -----------
    Y_expected: numpy.ndarray
        Numpy array containing expected predictions for each inserted input samples with the same order.
    Y_pred: numpy.ndarray
        Numpy array containing actual predictions for each inserted input samples with the same order.
    metric: function
        Function which is able to calculate metric for two vectors.
    """
    return [metric(Y_expected.iloc[:, i], Y_pred[:, i], average="micro") for i in range(Y_pred.shape[1])]


def get_classification_result(Y_expected, Y_pred, class_names):
    """Calculates three metrics on targets and predictions: f1_score, precision and recall. Constructs dataframe
    containing metric value for each class. Class names are stored as index. Returns pd.DataFrame object.

    Parameters:
    -----------
    Y_expected: numpy.ndarray
        Numpy array containing expected predictions for each inserted input samples with the same order.
    Y_pred: numpy.ndarray
        Numpy array containing actual predictions for each inserted input samples with the same order.
    class_names: list
        List of classes which will be used as index for output dataframe.
    """
    metrics = {}
    for metric_name, metric in zip(["f1_score", "precision", "recall"], [f1_score, precision_score, recall_score]):
        metrics[metric_name] = _metric_per_class(Y_expected, Y_pred, metric)

    df_metrics = pd.DataFrame(data=metrics)
    df_metrics.index = class_names
    return df_metrics
