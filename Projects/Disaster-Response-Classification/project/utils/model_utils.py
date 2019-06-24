from sklearn.externals import joblib


def save_model(path, model):
    """Saves model to specified directory.

    Parameters:
    -----------
    path: string
        Model output filepath including model name.

    model: sklearn.pipeline.Pipeline
        Class which is a predictor.
    """
    joblib.dump(model, path)
